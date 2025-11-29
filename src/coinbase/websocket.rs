use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use tokio::time::{interval, Duration};
use futures_util::{StreamExt, SinkExt};
use serde::Deserialize;
use serde_json::json;
use std::env;
use std::error::Error;
use crate::strategy::basis_trading::MarketData;
use rust_decimal::Decimal;
use chrono::Utc;
use tokio::sync::mpsc::Sender;
use log::{info, debug, error};

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
    _api_key: String,
    _api_secret: String,
}

impl CoinbaseWebsocket {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let api_key = env::var("COINBASE_API_KEY")?;
        let api_secret = env::var("COINBASE_API_SECRET")?;
        Ok(Self { _api_key: api_key, _api_secret: api_secret })
    }

    pub async fn connect_and_subscribe(
        &self, 
        product_ids: Vec<String>, 
        sender: Sender<MarketData>
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let (ws_stream, _) = connect_async(WS_URL).await.map_err(|e| e.to_string())?;
        info!("Connected to Coinbase WebSocket");

        let (mut write, mut read) = ws_stream.split();

        // Subscribe
        let subscribe_msg = json!({
            "type": "subscribe",
            "product_ids": product_ids,
            "channel": "ticker",
        });

        write.send(Message::Text(subscribe_msg.to_string().into())).await.map_err(|e| e.to_string())?;
        info!("Subscribed to products: {:?}", product_ids);

        // Process messages
        let mut heartbeat_interval = interval(Duration::from_secs(10));

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
                                                        timestamp: Utc::now().timestamp(),
                                                    };
                                                    if let Err(e) = sender.send(data).await {
                                                        error!("Error sending market data: {}", e);
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
                            info!("WebSocket closed");
                            break;
                        }
                        Some(Err(e)) => error!("WebSocket error: {}", e),
                        None => break, // Stream ended
                        _ => {}
                    }
                }
            }
        }

        Ok(())
    }

}
