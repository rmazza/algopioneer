//! Trade Recording System
//!
//! Provides a pluggable `TradeRecorder` trait for recording trades to various backends:
//! - CSV (development/testing)
//! - CloudWatch Logs via tracing (observability)
//! - DynamoDB (persistent storage) - requires `dynamodb` feature

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use std::fmt;
use thiserror::Error;

/// Error type for trade recording operations
#[derive(Debug, Error)]
pub enum RecordError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[cfg(feature = "dynamodb")]
    #[error("DynamoDB error: {0}")]
    DynamoDb(String),
}

/// A single trade record with all relevant fields
#[derive(Debug, Clone)]
pub struct TradeRecord {
    /// Unique trade identifier
    pub trade_id: String,
    /// Timestamp of trade execution
    pub timestamp: DateTime<Utc>,
    /// Trading pair (e.g., "BTC-USD")
    pub symbol: String,
    /// Trade side
    pub side: TradeSide,
    /// Trade size in base currency
    pub size: Decimal,
    /// Execution price (None for market orders)
    pub price: Option<Decimal>,
    /// Strategy that generated the trade
    pub strategy: Option<String>,
    /// Whether this is a paper trade
    pub is_paper: bool,
}

/// Trade side enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeSide {
    Buy,
    Sell,
}

impl fmt::Display for TradeSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TradeSide::Buy => write!(f, "BUY"),
            TradeSide::Sell => write!(f, "SELL"),
        }
    }
}

impl TradeRecord {
    /// Create a new trade record with explicit timestamp (deterministic).
    ///
    /// Use this constructor when you need reproducible timestamps for testing
    /// or when replaying historical trades.
    pub fn with_timestamp(
        symbol: String,
        side: TradeSide,
        size: Decimal,
        price: Option<Decimal>,
        is_paper: bool,
        timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            trade_id: uuid::Uuid::new_v4().to_string(),
            timestamp,
            symbol,
            side,
            size,
            price,
            strategy: None,
            is_paper,
        }
    }

    /// Create a new trade record with the current time (convenience).
    ///
    /// This is the typical constructor for live trading. For deterministic
    /// testing, use `with_timestamp()` instead.
    pub fn now(
        symbol: String,
        side: TradeSide,
        size: Decimal,
        price: Option<Decimal>,
        is_paper: bool,
    ) -> Self {
        Self::with_timestamp(symbol, side, size, price, is_paper, Utc::now())
    }

    /// Set the strategy name
    #[must_use]
    pub fn with_strategy(mut self, strategy: impl Into<String>) -> Self {
        self.strategy = Some(strategy.into());
        self
    }

    /// Format as CSV line
    pub fn to_csv_line(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{}",
            self.trade_id,
            self.timestamp.to_rfc3339(),
            self.symbol,
            self.side,
            self.size,
            self.price.map(|p| p.to_string()).unwrap_or_default(),
            self.strategy.as_deref().unwrap_or(""),
            self.is_paper,
        )
    }

    /// CSV header
    pub fn csv_header() -> &'static str {
        "trade_id,timestamp,symbol,side,size,price,strategy,is_paper"
    }
}

/// Trait for recording trades to various backends
#[async_trait]
pub trait TradeRecorder: Send + Sync {
    /// Record a trade. Implementations should be non-blocking.
    async fn record(&self, trade: &TradeRecord) -> Result<(), RecordError>;

    /// Flush any buffered records (optional, default no-op)
    async fn flush(&self) -> Result<(), RecordError> {
        Ok(())
    }
}

/// A recorder that fans out to multiple backends
pub struct MultiRecorder {
    recorders: Vec<Box<dyn TradeRecorder>>,
}

impl MultiRecorder {
    /// Create a new multi-recorder with the given backends
    pub fn new(recorders: Vec<Box<dyn TradeRecorder>>) -> Self {
        Self { recorders }
    }

    /// Add a recorder
    pub fn add(&mut self, recorder: Box<dyn TradeRecorder>) {
        self.recorders.push(recorder);
    }
}

#[async_trait]
impl TradeRecorder for MultiRecorder {
    async fn record(&self, trade: &TradeRecord) -> Result<(), RecordError> {
        for recorder in &self.recorders {
            // Best-effort: log errors but don't fail the whole chain
            if let Err(e) = recorder.record(trade).await {
                tracing::error!(error = %e, "Failed to record trade to backend");
            }
        }
        Ok(())
    }

    async fn flush(&self) -> Result<(), RecordError> {
        for recorder in &self.recorders {
            recorder.flush().await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_trade_record_csv() {
        let record = TradeRecord::now(
            "BTC-USD".to_string(),
            TradeSide::Buy,
            dec!(0.5),
            Some(dec!(50000)),
            true,
        );
        let csv = record.to_csv_line();
        assert!(csv.contains("BTC-USD"));
        assert!(csv.contains("BUY"));
        assert!(csv.contains("0.5"));
        assert!(csv.contains("50000"));
    }
}
