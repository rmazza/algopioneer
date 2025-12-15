//! Coinbase WebSocket Provider Adapter
//!
//! Wraps the existing CoinbaseWebsocket to implement the WebSocketProvider trait.

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::coinbase::websocket::CoinbaseWebsocket;
use crate::exchange::{ExchangeConfig, ExchangeError, WebSocketProvider};
use crate::strategy::dual_leg_trading::MarketData;

/// Coinbase WebSocket provider implementing the WebSocketProvider trait
pub struct CoinbaseWebSocketProvider {
    api_key: String,
    api_secret: String,
}

impl CoinbaseWebSocketProvider {
    /// Create a new CoinbaseWebSocketProvider from configuration
    pub fn new(config: &ExchangeConfig) -> Result<Self, ExchangeError> {
        Ok(Self {
            api_key: config.api_key.clone(),
            api_secret: config.api_secret.clone(),
        })
    }

    /// Create from environment variables (legacy compatibility)
    pub fn from_env() -> Result<Self, ExchangeError> {
        let api_key = std::env::var("COINBASE_API_KEY").map_err(|_| {
            ExchangeError::Configuration("COINBASE_API_KEY must be set".to_string())
        })?;
        let api_secret = std::env::var("COINBASE_API_SECRET").map_err(|_| {
            ExchangeError::Configuration("COINBASE_API_SECRET must be set".to_string())
        })?;

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
    ) -> Result<(), ExchangeError> {
        // Use direct credential injection (thread-safe, no env var mutation)
        let ws = CoinbaseWebsocket::with_credentials(self.api_key.clone(), self.api_secret.clone());

        ws.connect_and_subscribe(symbols, sender)
            .await
            .map_err(ExchangeError::from_boxed)
    }
}
