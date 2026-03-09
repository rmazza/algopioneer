use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use std::error::Error;
use thiserror::Error as ThisError;

/// Exchange-agnostic candle data
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: DateTime<Utc>,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
}

impl Default for Candle {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            open: Decimal::ZERO,
            high: Decimal::ZERO,
            low: Decimal::ZERO,
            close: Decimal::ZERO,
            volume: Decimal::ZERO,
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

/// Exchange-layer error type for actionable error handling.
#[derive(ThisError, Debug, Clone)]
pub enum ExchangeError {
    #[error("Network error: {0}")]
    Network(String),

    #[error("Rate limited, retry after {0}ms")]
    RateLimited(u64),

    #[error("Order rejected: {0}")]
    OrderRejected(String),

    #[error("Exchange internal error: {0}")]
    ExchangeInternal(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("{0}")]
    Other(String),
}

impl ExchangeError {
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ExchangeError::Network(_)
                | ExchangeError::RateLimited(_)
                | ExchangeError::ExchangeInternal(_)
        )
    }

    pub fn retry_delay_ms(&self) -> Option<u64> {
        match self {
            ExchangeError::RateLimited(ms) => Some(*ms),
            ExchangeError::Network(_) => Some(1000),
            ExchangeError::ExchangeInternal(_) => Some(5000),
            _ => None,
        }
    }

    pub fn from_boxed(e: Box<dyn Error + Send + Sync>) -> Self {
        let s = e.to_string();
        let lower = s.to_lowercase();

        if lower.contains("network") || lower.contains("timeout") || lower.contains("connection") {
            ExchangeError::Network(s)
        } else if lower.contains("rate limit") || lower.contains("too many requests") {
            ExchangeError::RateLimited(1000)
        } else if lower.contains("insufficient funds")
            || lower.contains("rejected")
            || lower.contains("invalid")
        {
            ExchangeError::OrderRejected(s)
        } else if lower.contains("internal") || lower.contains("server error") {
            ExchangeError::ExchangeInternal(s)
        } else {
            ExchangeError::Other(s)
        }
    }
}

/// Exchange identifier for factory/registry
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExchangeId {
    Coinbase,
    Kraken,
    Alpaca,
}

impl std::fmt::Display for ExchangeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExchangeId::Coinbase => write!(f, "coinbase"),
            ExchangeId::Kraken => write!(f, "kraken"),
            ExchangeId::Alpaca => write!(f, "alpaca"),
        }
    }
}

impl std::str::FromStr for ExchangeId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "coinbase" => Ok(ExchangeId::Coinbase),
            "kraken" => Ok(ExchangeId::Kraken),
            "alpaca" => Ok(ExchangeId::Alpaca),
            _ => Err(format!(
                "Unknown exchange: {}. Valid options: coinbase, kraken, alpaca",
                s
            )),
        }
    }
}
