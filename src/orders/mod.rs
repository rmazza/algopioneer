//! Order Management Module
//!
//! Provides order lifecycle tracking, fill monitoring, and position reconciliation.
//!
//! # Architecture
//!
//! - `OrderTracker` - Thread-safe order state machine
//! - `PositionReconciler` - Syncs local state with exchange
//! - Core types - `OrderId`, `OrderState`, `FillStatus`, `TrackedOrder`
//!
//! # Example
//!
//! ```ignore
//! use algopioneer::orders::{OrderTracker, OrderId};
//!
//! let tracker = OrderTracker::new(3600); // 1 hour retention
//! tracker.register_order(
//!     OrderId::new("abc-123"),
//!     "AAPL".to_string(),
//!     OrderSide::Buy,
//!     dec!(10),
//! ).await;
//! ```

mod reconciler;
mod tracker;
mod types;

pub use reconciler::{
    PositionReconciler, ReconciliationAction, ReconciliationConfig, ReconciliationResult,
};
pub use tracker::{OrderTracker, OrderTrackingError};
pub use types::{FillStatus, OrderId, OrderState, TrackedOrder};
