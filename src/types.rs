//! Common Types Module
//!
//! Shared types used across the codebase to avoid circular dependencies.
//!
//! # Examples
//!
//! ## Creating Market Data
//!
//! ```rust
//! use algopioneer::types::{MarketData, OrderSide};
//! use rust_decimal_macros::dec;
//!
//! // Create a market data tick
//! let tick = MarketData {
//!     symbol: "BTC-USD".to_string(),
//!     instrument_id: Some("coinbase".to_string()),
//!     price: dec!(50000.00),
//!     timestamp: 1700000000000, // milliseconds since epoch
//! };
//!
//! // Order sides
//! let buy = OrderSide::Buy;
//! let sell = OrderSide::Sell;
//! assert_eq!(buy.to_string(), "buy");
//! ```

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Order side (buy or sell).
///
/// Used throughout the trading system to indicate trade direction.
/// Implements Display for easy logging and API serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl std::fmt::Display for OrderSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderSide::Buy => write!(f, "buy"),
            OrderSide::Sell => write!(f, "sell"),
        }
    }
}

/// Represents a market data update (price tick).
///
/// This is the fundamental data structure for real-time price updates
/// received from exchanges via WebSocket connections or generated
/// synthetically for testing/backtesting.
///
/// # Fields
///
/// * `symbol` - The trading pair identifier (e.g., "BTC-USD", "ETH-USDT")
/// * `instrument_id` - Optional exchange identifier for multi-exchange support
/// * `price` - Current price using `Decimal` for financial precision
/// * `timestamp` - Unix timestamp in milliseconds for tick freshness validation
///
/// # Example
///
/// ```rust
/// use algopioneer::types::MarketData;
/// use rust_decimal_macros::dec;
///
/// let tick = MarketData {
///     symbol: "ETH-USD".to_string(),
///     instrument_id: None, // Single-exchange mode
///     price: dec!(2500.50),
///     timestamp: chrono::Utc::now().timestamp_millis(),
/// };
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MarketData {
    /// The trading symbol (e.g., "BTC-USD").
    pub symbol: String,
    /// Optional instrument identifier for multi-exchange support (e.g., "coinbase", "kraken").
    pub instrument_id: Option<String>,
    /// The current price (uses `Decimal` for financial precision).
    pub price: Decimal,
    /// The timestamp of the update in milliseconds since Unix epoch.
    pub timestamp: i64,
}
