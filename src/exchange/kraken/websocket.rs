//! Kraken WebSocket Provider (Stub)
//!
//! Placeholder implementation for Kraken WebSocket provider.
//! All methods return `unimplemented!()` - ready for future implementation.

use async_trait::async_trait;
use std::error::Error;
use tokio::sync::mpsc;

use crate::exchange::{ExchangeConfig, WebSocketProvider};
use crate::strategy::dual_leg_trading::MarketData;

/// Kraken WebSocket provider (stub implementation)
pub struct KrakenWebSocketProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    api_secret: String,
}

impl KrakenWebSocketProvider {
    /// Create a new KrakenWebSocketProvider from configuration
    pub fn new(config: &ExchangeConfig) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            api_key: config.api_key.clone(),
            api_secret: config.api_secret.clone(),
        })
    }
}

#[async_trait]
impl WebSocketProvider for KrakenWebSocketProvider {
    async fn connect_and_subscribe(
        &self,
        _symbols: Vec<String>,
        _sender: mpsc::Sender<MarketData>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        unimplemented!("Kraken WebSocket not yet implemented. Kraken uses wss://ws.kraken.com for public data.")
    }
}
