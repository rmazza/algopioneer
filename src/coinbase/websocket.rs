use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use tokio::time::{interval, Duration};
use futures_util::{StreamExt, SinkExt};
use url::Url;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::error::Error;
use crate::strategy::basis_trading::MarketData;
use rust_decimal::Decimal;
use chrono::{DateTime, Utc};
use tokio::sync::mpsc::Sender;

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
    api_key: String,
    api_secret: String,
}

impl CoinbaseWebsocket {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let api_key = env::var("COINBASE_API_KEY")?;
        let api_secret = env::var("COINBASE_API_SECRET")?;
        Ok(Self { api_key, api_secret })
    }

    pub async fn connect_and_subscribe(
        &self, 
        product_ids: Vec<String>, 
        sender: Sender<MarketData>
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let (ws_stream, _) = connect_async(WS_URL).await.map_err(|e| e.to_string())?;
        println!("Connected to Coinbase WebSocket");

        let (mut write, mut read) = ws_stream.split();

        // Subscribe
        let subscribe_msg = json!({
            "type": "subscribe",
            "product_ids": product_ids,
            "channel": "ticker",
        });

        write.send(Message::Text(subscribe_msg.to_string().into())).await.map_err(|e| e.to_string())?;
        println!("Subscribed to products: {:?}", product_ids);

        // Process messages
        let mut heartbeat_interval = interval(Duration::from_secs(10));

        loop {
            tokio::select! {
                _ = heartbeat_interval.tick() => {
                    println!("Heartbeat: WebSocket connection active...");
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
                                                        timestamp: Utc::now().timestamp(),
                                                    };
                                                    if let Err(e) = sender.send(data).await {
                                                        eprintln!("Error sending market data: {}", e);
                                                        return Ok(()); // Receiver dropped
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Some(Ok(Message::Close(_))) => {
                            println!("WebSocket closed");
                            break;
                        }
                        Some(Err(e)) => eprintln!("WebSocket error: {}", e),
                        None => break, // Stream ended
                        _ => {}
                    }
                }
            }
        }

        Ok(())
    }

    fn sign_message(&self, timestamp: &str, channel: &str, product_ids: &[String]) -> String {
        // NOTE: Coinbase Advanced Trade WS auth requires specific signature generation.
        // For public channels like 'ticker', auth might not be strictly required if not accessing user data,
        // but the API docs suggest it for higher limits or specific endpoints.
        // However, looking at docs, public ticker might be accessible without auth or with simplified auth.
        // Let's implement a placeholder or basic HMAC if needed.
        // Actually, for public ticker data, we might not need full auth.
        // But the plan implies we use API keys.
        // Let's assume we need to sign.
        
        // Signature = HMAC-SHA256(timestamp + channel + product_ids_comma_sep, secret)
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        use hex;

        let products_str = product_ids.join(",");
        let message = format!("{}{}{}", timestamp, channel, products_str);
        
        let mut mac = Hmac::<Sha256>::new_from_slice(self.api_secret.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(message.as_bytes());
        let result = mac.finalize();
        hex::encode(result.into_bytes())
    }
}
