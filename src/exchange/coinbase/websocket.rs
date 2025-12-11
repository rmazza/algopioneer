//! Coinbase WebSocket Provider Adapter
//!
//! Wraps the existing CoinbaseWebsocket to implement the WebSocketProvider trait.

use async_trait::async_trait;
use std::error::Error;
use tokio::sync::mpsc;

use crate::coinbase::websocket::CoinbaseWebsocket;
use crate::exchange::{ExchangeConfig, WebSocketProvider};
use crate::strategy::dual_leg_trading::MarketData;

/// Coinbase WebSocket provider implementing the WebSocketProvider trait
pub struct CoinbaseWebSocketProvider {
    api_key: String,
    api_secret: String,
}

impl CoinbaseWebSocketProvider {
    /// Create a new CoinbaseWebSocketProvider from configuration
    pub fn new(config: &ExchangeConfig) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            api_key: config.api_key.clone(),
            api_secret: config.api_secret.clone(),
        })
    }

    /// Create from environment variables (legacy compatibility)
    pub fn from_env() -> Result<Self, Box<dyn Error>> {
        let api_key =
            std::env::var("COINBASE_API_KEY").map_err(|_| "COINBASE_API_KEY must be set")?;
        let api_secret =
            std::env::var("COINBASE_API_SECRET").map_err(|_| "COINBASE_API_SECRET must be set")?;

        Ok(Self {
            api_key,
            api_secret,
        })
    }
}

#[async_trait]
impl WebSocketProvider for CoinbaseWebSocketProvider {
    async fn connect_and_subscribe(
        &self,
        symbols: Vec<String>,
        sender: mpsc::Sender<MarketData>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Set env vars for legacy CoinbaseWebsocket
        std::env::set_var("COINBASE_API_KEY", &self.api_key);
        std::env::set_var("COINBASE_API_SECRET", &self.api_secret);

        let ws = CoinbaseWebsocket::new().map_err(|e| -> Box<dyn Error + Send + Sync> {
            Box::new(std::io::Error::other(e.to_string()))
        })?;

        ws.connect_and_subscribe(symbols, sender).await
    }
}
