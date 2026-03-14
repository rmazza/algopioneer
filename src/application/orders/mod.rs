//! Order Management Module
//!
//! Provides order lifecycle tracking, fill monitoring, and position reconciliation.

mod reconciler;
mod tracker;

pub use crate::domain::orders::{FillStatus, OrderId, OrderState, TrackedOrder};
pub use reconciler::{
    PositionReconciler, ReconciliationAction, ReconciliationConfig, ReconciliationResult,
};
pub use tracker::{OrderTracker, OrderTrackingError};
