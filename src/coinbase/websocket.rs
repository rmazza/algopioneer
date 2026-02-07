use crate::strategy::dual_leg::MarketData;
use chrono::Utc;
use futures_util::{SinkExt, StreamExt};
use rust_decimal::Decimal;
use serde::Deserialize;
use serde_json::json;
use std::env;
use std::error::Error;
use tokio::sync::mpsc::Sender;
use tokio::time::{interval, Duration};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use tracing::{debug, error, info};

const WS_URL: &str = "wss://advanced-trade-ws.coinbase.com";

#[derive(Debug, Deserialize)]
struct WsMessage {
    channel: String,
    events: Vec<WsEvent>,
}

#[derive(Debug, Deserialize)]
struct WsEvent {
    r#type: String,
    tickers: Vec<WsTicker>,
}

#[derive(Debug, Deserialize)]
struct WsTicker {
    product_id: String,
    price: String,
}

pub struct CoinbaseWebsocket {
    /// Reserved for authenticated WebSocket connections (signing requests)
    #[allow(dead_code)]
    api_key: String,
    /// Reserved for authenticated WebSocket connections (signing requests)
    #[allow(dead_code)]
    api_secret: String,
}

impl CoinbaseWebsocket {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let api_key = env::var("COINBASE_API_KEY")?;
        let api_secret = env::var("COINBASE_API_SECRET")?;
        Ok(Self::with_credentials(api_key, api_secret))
    }

    /// Creates a new CoinbaseWebsocket with explicit credentials (thread-safe).
    ///
    /// This constructor does NOT use or mutate environment variables, making it
    /// safe to use in multi-threaded contexts.
    pub fn with_credentials(api_key: String, api_secret: String) -> Self {
        Self {
            api_key,
            api_secret,
        }
    }

    /// CF2 FIX: Connect with retry logic and exponential backoff
    async fn connect_with_retry(
        &self,
        max_retries: u32,
    ) -> Result<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
        Box<dyn Error + Send + Sync>,
    > {
        let mut backoff = Duration::from_secs(1);
        let max_backoff = Duration::from_secs(60);

        for attempt in 1..=max_retries {
            match connect_async(WS_URL).await {
                Ok((stream, _)) => {
                    info!("WebSocket connected successfully (attempt {})", attempt);
                    return Ok(stream);
                }
                Err(e) if attempt < max_retries => {
                    error!(
                        "WebSocket connection failed (attempt {}/{}): {}",
                        attempt, max_retries, e
                    );
                    info!("Retrying in {:?}...", backoff);
                    tokio::time::sleep(backoff).await;
                    backoff = std::cmp::min(backoff * 2, max_backoff);
                }
                Err(e) => {
                    error!("WebSocket connection failed after {} attempts", max_retries);
                    return Err(Box::new(std::io::Error::other(format!(
                        "Failed to connect after {} retries: {}",
                        max_retries, e
                    ))));
                }
            }
        }

        // This should never be reached since the for loop handles all attempts,
        // but provide a fallback error for safety
        Err(Box::new(std::io::Error::other(
            "Connection retry loop exited unexpectedly",
        )))
    }

    pub async fn connect_and_subscribe(
        &self,
        product_ids: Vec<String>,
        sender: Sender<MarketData>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // CF2 FIX: Wrap connection in reconnection loop
        const MAX_RECONNECT_ATTEMPTS: u32 = u32::MAX; // Infinite reconnection attempts
        const CONNECTION_MAX_RETRIES: u32 = 5; // Max retries per connection attempt

        let mut reconnect_count = 0;

        loop {
            reconnect_count += 1;
            info!(
                "WebSocket connection attempt {} (products: {:?})",
                reconnect_count, product_ids
            );

            // Attempt to connect with retries
            let ws_stream = match self.connect_with_retry(CONNECTION_MAX_RETRIES).await {
                Ok(stream) => stream,
                Err(e) => {
                    error!("Failed to establish WebSocket connection: {}", e);
                    if reconnect_count == MAX_RECONNECT_ATTEMPTS {
                        return Err(e);
                    }
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    continue;
                }
            };

            let (mut write, mut read) = ws_stream.split();

            // Subscribe
            let subscribe_msg = json!({
                "type": "subscribe",
                "product_ids": product_ids,
                "channel": "ticker",
            });

            if let Err(e) = write
                .send(Message::Text(subscribe_msg.to_string().into()))
                .await
            {
                error!("Failed to send subscribe message: {}", e);
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }

            info!("Subscribed to products: {:?}", product_ids);

            // Process messages
            let mut heartbeat_interval = interval(Duration::from_secs(10));
            let should_reconnect;

            loop {
                tokio::select! {
                    _ = heartbeat_interval.tick() => {
                        debug!("Heartbeat: WebSocket connection active...");
                    }
                    msg = read.next() => {
                        match msg {
                            Some(Ok(Message::Text(text))) => {
                                if let Ok(parsed) = serde_json::from_str::<WsMessage>(&text) {
                                    if parsed.channel == "ticker" {
                                        for event in parsed.events {
                                            if event.r#type == "snapshot" || event.r#type == "update" {
                                                for ticker in event.tickers {
                                                    if let Ok(price) = Decimal::from_str_exact(&ticker.price) {
                                                        let data = MarketData {
                                                            symbol: ticker.product_id,
                                                            price,
                                                            timestamp: Utc::now().timestamp_millis(),
                                                            instrument_id: None,
                                                        };
                                                        if let Err(e) = sender.send(data).await {
                                                            error!("Error sending market data: {}", e);
                                                            return Ok(()); // Receiver dropped - permanent exit
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            Some(Ok(Message::Close(frame))) => {
                                info!("WebSocket closed by server: {:?}", frame);
                                should_reconnect = true;
                                break;
                            }
                            Some(Err(e)) => {
                                error!("WebSocket error: {}", e);
                                should_reconnect = true;
                                break;
                            }
                            None => {
                                info!("WebSocket stream ended");
                                should_reconnect = true;
                                break;
                            }
                            _ => {}
                        }
                    }
                }
            }

            // If we should reconnect, continue the outer loop
            if should_reconnect {
                info!("Initiating reconnection...");
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            } else {
                // Permanent exit (e.g., receiver dropped)
                break;
            }
        }

        Ok(())
    }
}
