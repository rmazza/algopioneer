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
    pub trade_id: String,
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub side: TradeSide,
    pub size: Decimal,
    pub price: Option<Decimal>,
    pub strategy: Option<String>,
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

/// Persistent position state for recovery across container restarts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PositionStateRecord {
    pub position_id: String,
    pub strategy_type: String,
    pub state: String,
    pub direction: Option<String>,
    pub leg1_symbol: String,
    pub leg2_symbol: String,
    pub leg1_qty: String,
    pub leg2_qty: String,
    pub leg1_entry_price: String,
    pub leg2_entry_price: String,
    pub updated_at: DateTime<Utc>,
    pub is_paper: bool,
}
