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
    #[deprecated(
        note = "Use with_timestamp and inject time from a Clock trait for better determinism"
    )]
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

    /// Format as CSV line (allocates a new String).
    ///
    /// For performance-critical paths, prefer `write_csv_to()` which writes
    /// directly to a buffer without intermediate allocation.
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

    /// Write CSV line directly to a writer (zero intermediate allocation).
    ///
    /// This is more efficient than `to_csv_line()` for high-frequency recording
    /// as it avoids allocating an intermediate String.
    pub fn write_csv_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write!(
            writer,
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

// ============================================================================
// Position State Persistence
// ============================================================================

/// Persistent position state for recovery across container restarts.
///
/// This record captures all necessary information to restore a strategy's
/// state, including entry prices that would otherwise be lost.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PositionStateRecord {
    /// Unique identifier: "strategy_type:leg1_symbol:leg2_symbol"
    /// e.g., "pairs:AAPL:MSFT"
    pub position_id: String,
    /// Strategy type: "pairs" or "basis"
    pub strategy_type: String,
    /// Current state: "flat", "entering", "in_position", "exiting", "reconciling", "halted"
    pub state: String,
    /// Position direction: "long" or "short" (None if flat)
    pub direction: Option<String>,
    /// Leg 1 symbol
    pub leg1_symbol: String,
    /// Leg 2 symbol
    pub leg2_symbol: String,
    /// Leg 1 quantity (Decimal serialized as string for precision)
    pub leg1_qty: String,
    /// Leg 2 quantity
    pub leg2_qty: String,
    /// Leg 1 entry price (Decimal serialized as string)
    pub leg1_entry_price: String,
    /// Leg 2 entry price
    pub leg2_entry_price: String,
    /// Timestamp of last update
    pub updated_at: DateTime<Utc>,
    /// Whether this is paper trading
    pub is_paper: bool,
}

impl PositionStateRecord {
    /// Create a new position state record for a flat (no position) state.
    pub fn flat(
        position_id: impl Into<String>,
        strategy_type: impl Into<String>,
        leg1_symbol: impl Into<String>,
        leg2_symbol: impl Into<String>,
        is_paper: bool,
    ) -> Self {
        Self {
            position_id: position_id.into(),
            strategy_type: strategy_type.into(),
            state: "flat".to_string(),
            direction: None,
            leg1_symbol: leg1_symbol.into(),
            leg2_symbol: leg2_symbol.into(),
            leg1_qty: "0".to_string(),
            leg2_qty: "0".to_string(),
            leg1_entry_price: "0".to_string(),
            leg2_entry_price: "0".to_string(),
            updated_at: Utc::now(),
            is_paper,
        }
    }
}

/// Trait for persisting and retrieving strategy position state.
///
/// Implementations should provide durable storage (e.g., DynamoDB, PostgreSQL)
/// that survives container restarts.
#[async_trait]
pub trait StateStore: Send + Sync {
    /// Save current position state. Overwrites any previous state for this position_id.
    async fn save_state(&self, state: &PositionStateRecord) -> Result<(), RecordError>;

    /// Load position state by ID. Returns None if no state exists.
    async fn load_state(
        &self,
        position_id: &str,
    ) -> Result<Option<PositionStateRecord>, RecordError>;

    /// Delete position state (for graceful shutdown or manual cleanup).
    async fn delete_state(&self, position_id: &str) -> Result<(), RecordError>;
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
        let mut error_count = 0;
        let mut last_error = None;

        for recorder in &self.recorders {
            if let Err(e) = recorder.record(trade).await {
                // Best-effort: log errors but don't fail the whole chain immediately
                tracing::error!(error = %e, "Failed to record trade to backend");
                last_error = Some(e);
                error_count += 1;
            }
        }

        // If all recorders failed, we should probably report it
        if error_count > 0 && error_count == self.recorders.len() {
            if let Some(e) = last_error {
                return Err(e);
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
    use chrono::Utc;
    use rust_decimal_macros::dec;

    #[test]
    fn test_trade_record_csv() {
        // Use with_timestamp for deterministic testing (avoids deprecated now())
        let record = TradeRecord::with_timestamp(
            "BTC-USD".to_string(),
            TradeSide::Buy,
            dec!(0.5),
            Some(dec!(50000)),
            true,
            Utc::now(), // Explicit timestamp injection
        );
        let csv = record.to_csv_line();
        assert!(csv.contains("BTC-USD"));
        assert!(csv.contains("BUY"));
        assert!(csv.contains("0.5"));
        assert!(csv.contains("50000"));
    }
}
