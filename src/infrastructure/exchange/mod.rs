//! Exchange Infrastructure Layer
//!
//! This module provides concrete exchange implementations (Coinbase, Kraken, Alpaca)
//! and factories to instantiate them.

pub mod alpaca;
pub mod coinbase;
pub mod kraken;

use chrono::{Utc, TimeZone};
use rust_decimal::prelude::FromPrimitive;
use rust_decimal::Decimal;
use std::sync::Arc;

pub use crate::domain::exchange::{ExchangeError, ExchangeId, Candle, Granularity};
pub use crate::application::ports::exchange::{Executor, ExchangeClient, WebSocketProvider, WebSocketHandle};

// Re-export concrete clients for convenience
pub use alpaca::AlpacaClient;
pub use coinbase::CoinbaseExchangeClient;
pub use kraken::KrakenExchangeClient;

/// Configuration for exchange connections
#[derive(Debug, Clone)]
pub struct ExchangeConfig {
    pub api_key: String,
    pub api_secret: String,
    pub sandbox: bool,
}

impl ExchangeConfig {
    pub fn from_env(exchange: ExchangeId) -> Result<Self, ExchangeError> {
        let (key_var, secret_var) = match exchange {
            ExchangeId::Coinbase => ("COINBASE_API_KEY", "COINBASE_API_SECRET"),
            ExchangeId::Kraken => ("KRAKEN_API_KEY", "KRAKEN_API_SECRET"),
            ExchangeId::Alpaca => ("ALPACA_API_KEY", "ALPACA_API_SECRET"),
        };

        let api_key = std::env::var(key_var).map_err(|_| {
            ExchangeError::Configuration(format!("{} must be set in environment", key_var))
        })?;
        let api_secret = std::env::var(secret_var).map_err(|_| {
            ExchangeError::Configuration(format!("{} must be set in environment", secret_var))
        })?;

        Ok(Self {
            api_key,
            api_secret,
            sandbox: false,
        })
    }
}

/// Factory function to create exchange clients
pub fn create_exchange_client(
    exchange: ExchangeId,
    config: ExchangeConfig,
) -> Result<Arc<dyn ExchangeClient>, ExchangeError> {
    match exchange {
        ExchangeId::Coinbase => {
            let client = coinbase::CoinbaseExchangeClient::new(config)?;
            Ok(Arc::new(client))
        }
        ExchangeId::Kraken => {
            let client = kraken::KrakenExchangeClient::new(config)?;
            Ok(Arc::new(client))
        }
        ExchangeId::Alpaca => {
            let client = alpaca::AlpacaClient::from_config(config)?;
            Ok(Arc::new(client))
        }
    }
}

/// Factory function to create WebSocket providers
pub fn create_websocket_provider(
    exchange: ExchangeId,
    config: &ExchangeConfig,
) -> Result<Box<dyn WebSocketProvider>, ExchangeError> {
    match exchange {
        ExchangeId::Coinbase => {
            let provider = coinbase::CoinbaseWebSocketProvider::new(config)?;
            Ok(Box::new(provider))
        }
        ExchangeId::Kraken => {
            let provider = kraken::KrakenWebSocketProvider::new(config)?;
            Ok(Box::new(provider))
        }
        ExchangeId::Alpaca => {
            let provider = alpaca::AlpacaWebSocketProvider::new(config)?;
            Ok(Box::new(provider))
        }
    }
}

// Convert from cbadv Candle to our Domain Candle
impl From<cbadv::models::product::Candle> for Candle {
    fn from(c: cbadv::models::product::Candle) -> Self {
        let timestamp = Utc
            .timestamp_opt(c.start as i64, 0)
            .single()
            .unwrap_or_else(Utc::now);

        let convert_with_warning = |value: f64, field: &str| -> Decimal {
            match Decimal::from_f64(value) {
                Some(d) => d,
                None => {
                    tracing::warn!(
                        field = field,
                        value = value,
                        "Failed to convert candle {} to Decimal, using ZERO",
                        field
                    );
                    Decimal::ZERO
                }
            }
        };

        Self {
            timestamp,
            open: convert_with_warning(c.open, "open"),
            high: convert_with_warning(c.high, "high"),
            low: convert_with_warning(c.low, "low"),
            close: convert_with_warning(c.close, "close"),
            volume: convert_with_warning(c.volume, "volume"),
        }
    }
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
