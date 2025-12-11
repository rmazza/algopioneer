//! Exchange Abstraction Layer
//!
//! This module provides exchange-agnostic traits and factories for trading.
//! New exchanges can be added by implementing the core traits without
//! modifying business logic in strategies.

pub mod coinbase;
pub mod kraken;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use std::error::Error;
use std::sync::Arc;
use tokio::sync::mpsc;

// Re-export shared types for convenience
pub use crate::types::{MarketData, OrderSide};

// Re-export commonly used types
pub use coinbase::CoinbaseExchangeClient;
pub use kraken::KrakenExchangeClient;

/// Exchange-agnostic candle data
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Convert from cbadv Candle to our Candle
impl From<cbadv::models::product::Candle> for Candle {
    fn from(c: cbadv::models::product::Candle) -> Self {
        Self {
            timestamp: Utc::now(), // cbadv uses unix timestamp, we'll convert in implementation
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close,
            volume: c.volume,
        }
    }
}

/// Unified granularity enum (maps to exchange-specific values)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Granularity {
    OneMinute,
    FiveMinute,
    FifteenMinute,
    ThirtyMinute,
    OneHour,
    TwoHour,
    SixHour,
    OneDay,
}

impl From<Granularity> for cbadv::time::Granularity {
    fn from(g: Granularity) -> Self {
        match g {
            Granularity::OneMinute => cbadv::time::Granularity::OneMinute,
            Granularity::FiveMinute => cbadv::time::Granularity::FiveMinute,
            Granularity::FifteenMinute => cbadv::time::Granularity::FifteenMinute,
            Granularity::ThirtyMinute => cbadv::time::Granularity::ThirtyMinute,
            Granularity::OneHour => cbadv::time::Granularity::OneHour,
            Granularity::TwoHour => cbadv::time::Granularity::TwoHour,
            Granularity::SixHour => cbadv::time::Granularity::SixHour,
            Granularity::OneDay => cbadv::time::Granularity::OneDay,
        }
    }
}

/// Configuration for exchange connections
#[derive(Debug, Clone)]
pub struct ExchangeConfig {
    pub api_key: String,
    pub api_secret: String,
    pub sandbox: bool,
}

impl ExchangeConfig {
    /// Create config from environment variables for the specified exchange
    pub fn from_env(exchange: ExchangeId) -> Result<Self, Box<dyn Error>> {
        let (key_var, secret_var) = match exchange {
            ExchangeId::Coinbase => ("COINBASE_API_KEY", "COINBASE_API_SECRET"),
            ExchangeId::Kraken => ("KRAKEN_API_KEY", "KRAKEN_API_SECRET"),
        };

        let api_key = std::env::var(key_var)
            .map_err(|_| format!("{} must be set in environment", key_var))?;
        let api_secret = std::env::var(secret_var)
            .map_err(|_| format!("{} must be set in environment", secret_var))?;

        Ok(Self {
            api_key,
            api_secret,
            sandbox: false,
        })
    }
}

/// Core trait for order execution - exchange implementations must provide this
#[async_trait]
pub trait Executor: Send + Sync {
    /// Execute an order on the exchange
    async fn execute_order(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: Decimal,
        price: Option<Decimal>,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;

    /// Get current position for a symbol
    async fn get_position(&self, symbol: &str) -> Result<Decimal, Box<dyn Error + Send + Sync>>;
}

/// Extended exchange client trait with full capabilities
#[async_trait]
pub trait ExchangeClient: Executor + Send + Sync {
    /// Test API connectivity
    async fn test_connection(&mut self) -> Result<(), Box<dyn Error>>;

    /// Get historical candles
    async fn get_candles(
        &mut self,
        product_id: &str,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
        granularity: Granularity,
    ) -> Result<Vec<Candle>, Box<dyn Error>>;

    /// Get candles with pagination (for large date ranges)
    async fn get_candles_paginated(
        &mut self,
        product_id: &str,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
        granularity: Granularity,
    ) -> Result<Vec<Candle>, Box<dyn Error>>;

    /// Normalize a symbol to exchange-specific format
    /// e.g., "BTC-USD" -> "XXBTZUSD" for Kraken
    fn normalize_symbol(&self, symbol: &str) -> String;

    /// Get the exchange identifier
    fn exchange_id(&self) -> ExchangeId;
}

/// Trait for WebSocket market data providers
#[async_trait]
pub trait WebSocketProvider: Send + Sync {
    /// Connect and subscribe to market data for given symbols
    async fn connect_and_subscribe(
        &self,
        symbols: Vec<String>,
        sender: mpsc::Sender<MarketData>,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;
}

/// Exchange identifier for factory/registry
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExchangeId {
    Coinbase,
    Kraken,
}

impl std::fmt::Display for ExchangeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExchangeId::Coinbase => write!(f, "coinbase"),
            ExchangeId::Kraken => write!(f, "kraken"),
        }
    }
}

impl std::str::FromStr for ExchangeId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "coinbase" => Ok(ExchangeId::Coinbase),
            "kraken" => Ok(ExchangeId::Kraken),
            _ => Err(format!(
                "Unknown exchange: {}. Valid options: coinbase, kraken",
                s
            )),
        }
    }
}

/// Factory function to create exchange clients
pub fn create_exchange_client(
    exchange: ExchangeId,
    config: ExchangeConfig,
) -> Result<Arc<dyn ExchangeClient>, Box<dyn Error>> {
    match exchange {
        ExchangeId::Coinbase => {
            let client = coinbase::CoinbaseExchangeClient::new(config)?;
            Ok(Arc::new(client))
        }
        ExchangeId::Kraken => {
            let client = kraken::KrakenExchangeClient::new(config)?;
            Ok(Arc::new(client))
        }
    }
}

/// Factory function to create WebSocket providers
pub fn create_websocket_provider(
    exchange: ExchangeId,
    config: &ExchangeConfig,
) -> Result<Box<dyn WebSocketProvider>, Box<dyn Error>> {
    match exchange {
        ExchangeId::Coinbase => {
            let provider = coinbase::CoinbaseWebSocketProvider::new(config)?;
            Ok(Box::new(provider))
        }
        ExchangeId::Kraken => {
            let provider = kraken::KrakenWebSocketProvider::new(config)?;
            Ok(Box::new(provider))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exchange_id_from_str() {
        assert_eq!(
            "coinbase".parse::<ExchangeId>().unwrap(),
            ExchangeId::Coinbase
        );
        assert_eq!(
            "Coinbase".parse::<ExchangeId>().unwrap(),
            ExchangeId::Coinbase
        );
        assert_eq!("kraken".parse::<ExchangeId>().unwrap(), ExchangeId::Kraken);
        assert!("binance".parse::<ExchangeId>().is_err());
    }

    #[test]
    fn test_exchange_id_display() {
        assert_eq!(ExchangeId::Coinbase.to_string(), "coinbase");
        assert_eq!(ExchangeId::Kraken.to_string(), "kraken");
    }
}
