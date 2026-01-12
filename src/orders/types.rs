//! Core types for order management.
//!
//! Provides type-safe order identifiers and state tracking.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::types::OrderSide;

/// Type-safe order identifier (exchange-assigned).
///
/// Uses a newtype wrapper to prevent accidentally mixing order IDs
/// with other string types at compile time.
///
/// # Thread Safety
///
/// `OrderId` is `Clone`, `Send`, and `Sync`, making it safe for use
/// across async boundaries and thread pools.
///
/// # Example
///
/// ```
/// use algopioneer::orders::OrderId;
///
/// let id = OrderId::new("abc-123-def");
/// assert_eq!(id.as_str(), "abc-123-def");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrderId(String);

impl OrderId {
    /// Create a new OrderId from any string-like type.
    ///
    /// # Panics
    ///
    /// Debug builds will panic if the ID is empty. Release builds log a warning.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        let s: String = id.into();
        debug_assert!(!s.is_empty(), "OrderId cannot be empty");
        if s.is_empty() {
            tracing::warn!("Creating OrderId with empty string - this may cause tracking issues");
        }
        Self(s)
    }

    /// Get the underlying string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume and return the inner String.
    #[must_use]
    pub fn into_inner(self) -> String {
        self.0
    }

    /// Check if the order ID is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Display for OrderId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for OrderId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for OrderId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl AsRef<str> for OrderId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Order lifecycle states.
///
/// Follows the standard exchange order lifecycle with terminal states
/// clearly distinguished for cleanup and tracking purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderState {
    /// Order submitted, awaiting acknowledgment from exchange
    Pending,
    /// Order accepted by exchange, awaiting fills
    Open,
    /// Order partially filled (some quantity executed)
    PartiallyFilled,
    /// Order fully filled (all quantity executed)
    Filled,
    /// Order cancelled (by user or exchange)
    Cancelled,
    /// Order rejected by exchange (insufficient funds, invalid params, etc.)
    Rejected,
    /// Order expired (time-in-force exceeded)
    Expired,
}

impl OrderState {
    /// Returns true if order is in a terminal state (no further updates expected).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Filled | Self::Cancelled | Self::Rejected | Self::Expired
        )
    }

    /// Returns true if order may still receive fills.
    pub fn may_fill(&self) -> bool {
        matches!(self, Self::Pending | Self::Open | Self::PartiallyFilled)
    }

    /// Returns true if order was successful (fully or partially filled).
    pub fn has_fills(&self) -> bool {
        matches!(self, Self::Filled | Self::PartiallyFilled)
    }
}

impl std::fmt::Display for OrderState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "Pending"),
            Self::Open => write!(f, "Open"),
            Self::PartiallyFilled => write!(f, "PartiallyFilled"),
            Self::Filled => write!(f, "Filled"),
            Self::Cancelled => write!(f, "Cancelled"),
            Self::Rejected => write!(f, "Rejected"),
            Self::Expired => write!(f, "Expired"),
        }
    }
}

/// Fill status with quantities and pricing.
///
/// Tracks the execution progress of an order including partial fills.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FillStatus {
    /// Originally requested quantity
    pub requested_qty: Decimal,
    /// Quantity filled so far
    pub filled_qty: Decimal,
    /// Volume-weighted average fill price (None if no fills yet)
    pub avg_fill_price: Option<Decimal>,
}

impl FillStatus {
    /// Create a new FillStatus for a fresh order.
    pub fn new(requested_qty: Decimal) -> Self {
        Self {
            requested_qty,
            filled_qty: Decimal::ZERO,
            avg_fill_price: None,
        }
    }

    /// Calculate remaining unfilled quantity.
    pub fn unfilled_qty(&self) -> Decimal {
        self.requested_qty - self.filled_qty
    }

    /// Calculate fill ratio (0.0 to 1.0).
    #[must_use]
    pub fn fill_ratio(&self) -> Decimal {
        if self.requested_qty.is_zero() {
            Decimal::ZERO
        } else {
            self.filled_qty / self.requested_qty
        }
    }

    /// Returns true if order is completely filled.
    pub fn is_complete(&self) -> bool {
        self.filled_qty >= self.requested_qty
    }

    /// Returns true if order has any fills.
    pub fn has_fills(&self) -> bool {
        self.filled_qty > Decimal::ZERO
    }
}

impl Default for FillStatus {
    fn default() -> Self {
        Self::new(Decimal::ZERO)
    }
}

/// Complete tracked order with all metadata.
///
/// Represents the full state of an order throughout its lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackedOrder {
    /// Exchange-assigned order ID
    pub id: OrderId,
    /// Trading symbol (e.g., "AAPL", "BTC-USD")
    pub symbol: String,
    /// Order side (Buy or Sell)
    pub side: OrderSide,
    /// Current order state
    pub state: OrderState,
    /// Fill progress
    pub fill: FillStatus,
    /// When order was submitted
    pub created_at: DateTime<Utc>,
    /// When order was last updated
    pub updated_at: DateTime<Utc>,
}

impl TrackedOrder {
    /// Create a new tracked order in Pending state.
    ///
    /// # Note on Timestamps
    ///
    /// Uses `Utc::now()` at creation time (not injected clock) because:
    /// 1. Order creation is a real-world event with a definite wall-clock time
    /// 2. This timestamp is for logging/debugging, not strategy logic
    /// 3. Similar to how `entry_price` is captured at trade time
    #[must_use]
    pub fn new(id: OrderId, symbol: String, side: OrderSide, quantity: Decimal) -> Self {
        let now = Utc::now();
        Self {
            id,
            symbol,
            side,
            state: OrderState::Pending,
            fill: FillStatus::new(quantity),
            created_at: now,
            updated_at: now,
        }
    }

    /// Check if order is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }

    /// Check if order was successful (any fills received).
    pub fn is_successful(&self) -> bool {
        self.state == OrderState::Filled || (self.state.is_terminal() && self.fill.has_fills())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_order_id_newtype() {
        let id = OrderId::new("abc-123");
        assert_eq!(id.as_str(), "abc-123");
        assert_eq!(id.to_string(), "abc-123");

        // Test From impls
        let id2: OrderId = "xyz-789".into();
        assert_eq!(id2.as_str(), "xyz-789");

        let id3: OrderId = String::from("foo-bar").into();
        assert_eq!(id3.as_str(), "foo-bar");
    }

    #[test]
    fn test_order_state_terminal() {
        assert!(!OrderState::Pending.is_terminal());
        assert!(!OrderState::Open.is_terminal());
        assert!(!OrderState::PartiallyFilled.is_terminal());
        assert!(OrderState::Filled.is_terminal());
        assert!(OrderState::Cancelled.is_terminal());
        assert!(OrderState::Rejected.is_terminal());
        assert!(OrderState::Expired.is_terminal());
    }

    #[test]
    fn test_order_state_may_fill() {
        assert!(OrderState::Pending.may_fill());
        assert!(OrderState::Open.may_fill());
        assert!(OrderState::PartiallyFilled.may_fill());
        assert!(!OrderState::Filled.may_fill());
        assert!(!OrderState::Cancelled.may_fill());
    }

    #[test]
    fn test_fill_status_calculations() {
        let mut fill = FillStatus::new(dec!(100));
        assert_eq!(fill.unfilled_qty(), dec!(100));
        assert_eq!(fill.fill_ratio(), dec!(0));
        assert!(!fill.is_complete());
        assert!(!fill.has_fills());

        fill.filled_qty = dec!(50);
        assert_eq!(fill.unfilled_qty(), dec!(50));
        assert_eq!(fill.fill_ratio(), dec!(0.5));
        assert!(!fill.is_complete());
        assert!(fill.has_fills());

        fill.filled_qty = dec!(100);
        assert_eq!(fill.unfilled_qty(), dec!(0));
        assert_eq!(fill.fill_ratio(), dec!(1));
        assert!(fill.is_complete());
    }

    #[test]
    fn test_tracked_order_creation() {
        let order = TrackedOrder::new(
            OrderId::new("test-123"),
            "AAPL".to_string(),
            OrderSide::Buy,
            dec!(10),
        );

        assert_eq!(order.id.as_str(), "test-123");
        assert_eq!(order.symbol, "AAPL");
        assert_eq!(order.side, OrderSide::Buy);
        assert_eq!(order.state, OrderState::Pending);
        assert_eq!(order.fill.requested_qty, dec!(10));
        assert!(!order.is_terminal());
    }
}
