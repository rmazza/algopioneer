//! Order tracking with thread-safe state machine.
//!
//! Provides in-memory order state tracking for concurrent access
//! from trading loops and WebSocket update tasks.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use rust_decimal::Decimal;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use super::types::{OrderId, OrderState, TrackedOrder};
use crate::types::OrderSide;

/// Errors that can occur during order tracking operations.
#[derive(Error, Debug, Clone)]
pub enum OrderTrackingError {
    /// Order not found in tracker
    #[error("Order not found: {0}")]
    OrderNotFound(OrderId),

    /// Timeout waiting for order to reach terminal state
    #[error("Timeout waiting for fill: {0}")]
    Timeout(OrderId),

    /// Invalid state transition attempted
    #[error("Invalid state transition for order {0}: {1} -> {2}")]
    InvalidTransition(OrderId, OrderState, OrderState),
}

/// Thread-safe order tracker.
///
/// Maintains in-memory state of all active orders. Designed for concurrent
/// access from trading loop and WebSocket update tasks.
///
/// # Thread Safety
///
/// Uses `RwLock` for efficient concurrent reads with exclusive writes.
/// All methods are async to support the lock operations.
///
/// # Memory Management
///
/// Completed orders are retained for `retention_secs` to allow for
/// late-arriving updates and debugging. Use `cleanup_old_orders()` to
/// remove stale entries.
///
/// # Timestamps
///
/// Order update timestamps use `Utc::now()` for logging/debugging purposes.
/// This is intentional as order updates are real-world events.
#[derive(Clone)]
pub struct OrderTracker {
    orders: Arc<RwLock<HashMap<OrderId, TrackedOrder>>>,
    /// Maximum age of completed orders before cleanup (seconds)
    retention_secs: u64,
}

impl OrderTracker {
    /// Create a new OrderTracker with specified retention period.
    ///
    /// # Arguments
    ///
    /// * `retention_secs` - How long to keep completed orders (for debugging)
    pub fn new(retention_secs: u64) -> Self {
        Self {
            orders: Arc::new(RwLock::new(HashMap::new())),
            retention_secs,
        }
    }

    /// Register a new order submission.
    ///
    /// Call this immediately after submitting an order to the exchange.
    /// The order starts in `Pending` state.
    pub async fn register_order(
        &self,
        id: OrderId,
        symbol: String,
        side: OrderSide,
        quantity: Decimal,
    ) {
        let order = TrackedOrder::new(id.clone(), symbol, side, quantity);

        let mut orders = self.orders.write().await;
        orders.insert(id.clone(), order);
        debug!(order_id = %id, "Order registered in tracker");
    }

    /// Update order state from exchange feedback.
    ///
    /// Call this when receiving order updates from WebSocket or polling.
    /// Returns the updated order if found, None otherwise.
    pub async fn update_order(
        &self,
        id: &OrderId,
        new_state: OrderState,
        filled_qty: Decimal,
        avg_price: Option<Decimal>,
    ) -> Option<TrackedOrder> {
        let mut orders = self.orders.write().await;

        if let Some(order) = orders.get_mut(id) {
            let old_state = order.state;
            order.state = new_state;
            order.fill.filled_qty = filled_qty;
            order.fill.avg_fill_price = avg_price;
            order.updated_at = Utc::now();

            info!(
                order_id = %id,
                symbol = %order.symbol,
                old_state = %old_state,
                new_state = %new_state,
                filled = %filled_qty,
                requested = %order.fill.requested_qty,
                "Order state updated"
            );

            Some(order.clone())
        } else {
            warn!(order_id = %id, "Attempted to update unknown order");
            None
        }
    }

    /// Mark order as open (acknowledged by exchange).
    pub async fn mark_open(&self, id: &OrderId) -> Option<TrackedOrder> {
        self.update_order(id, OrderState::Open, Decimal::ZERO, None)
            .await
    }

    /// Record a fill event.
    ///
    /// Automatically transitions to `PartiallyFilled` or `Filled` based
    /// on fill completion.
    pub async fn record_fill(
        &self,
        id: &OrderId,
        filled_qty: Decimal,
        avg_price: Decimal,
    ) -> Option<TrackedOrder> {
        let mut orders = self.orders.write().await;

        if let Some(order) = orders.get_mut(id) {
            order.fill.filled_qty = filled_qty;
            order.fill.avg_fill_price = Some(avg_price);
            order.updated_at = Utc::now();

            // Auto-transition based on fill status
            let old_state = order.state;
            order.state = if order.fill.is_complete() {
                OrderState::Filled
            } else {
                OrderState::PartiallyFilled
            };

            info!(
                order_id = %id,
                symbol = %order.symbol,
                old_state = %old_state,
                new_state = %order.state,
                filled = %filled_qty,
                requested = %order.fill.requested_qty,
                avg_price = %avg_price,
                "Fill recorded"
            );

            Some(order.clone())
        } else {
            warn!(order_id = %id, "Attempted to record fill for unknown order");
            None
        }
    }

    /// Mark order as cancelled.
    pub async fn mark_cancelled(&self, id: &OrderId) -> Option<TrackedOrder> {
        let orders = self.orders.read().await;
        let current_fill = orders
            .get(id)
            .map(|o| o.fill.filled_qty)
            .unwrap_or_default();
        drop(orders);

        self.update_order(id, OrderState::Cancelled, current_fill, None)
            .await
    }

    /// Mark order as rejected.
    ///
    /// Logs the rejection reason for debugging purposes.
    pub async fn mark_rejected(&self, id: &OrderId, reason: &str) -> Option<TrackedOrder> {
        warn!(order_id = %id, reason = reason, "Order rejected");
        self.update_order(id, OrderState::Rejected, Decimal::ZERO, None)
            .await
    }

    /// Get current order state.
    pub async fn get_order(&self, id: &OrderId) -> Option<TrackedOrder> {
        let orders = self.orders.read().await;
        orders.get(id).cloned()
    }

    /// Check if order exists in tracker.
    pub async fn has_order(&self, id: &OrderId) -> bool {
        let orders = self.orders.read().await;
        orders.contains_key(id)
    }

    /// Get all orders for a symbol (active and recent).
    pub async fn orders_for_symbol(&self, symbol: &str) -> Vec<TrackedOrder> {
        let orders = self.orders.read().await;
        orders
            .values()
            .filter(|o| o.symbol == symbol)
            .cloned()
            .collect()
    }

    /// Get all active (non-terminal) orders.
    pub async fn active_orders(&self) -> Vec<TrackedOrder> {
        let orders = self.orders.read().await;
        orders
            .values()
            .filter(|o| !o.state.is_terminal())
            .cloned()
            .collect()
    }

    /// Wait for order to reach terminal state with timeout.
    ///
    /// Polls the order state at regular intervals until it reaches
    /// a terminal state or the timeout expires.
    ///
    /// # Returns
    ///
    /// The final order state, or an error if timeout/not found.
    pub async fn wait_for_fill(
        &self,
        id: &OrderId,
        timeout: Duration,
    ) -> Result<TrackedOrder, OrderTrackingError> {
        let deadline = tokio::time::Instant::now() + timeout;
        let poll_interval = Duration::from_millis(100);

        loop {
            if let Some(order) = self.get_order(id).await {
                if order.state.is_terminal() {
                    return Ok(order);
                }
            } else {
                return Err(OrderTrackingError::OrderNotFound(id.clone()));
            }

            if tokio::time::Instant::now() >= deadline {
                return Err(OrderTrackingError::Timeout(id.clone()));
            }

            tokio::time::sleep(poll_interval).await;
        }
    }

    /// Get count of tracked orders.
    #[must_use]
    pub async fn order_count(&self) -> usize {
        let orders = self.orders.read().await;
        orders.len()
    }

    /// Get count of active (non-terminal) orders.
    #[must_use]
    pub async fn active_order_count(&self) -> usize {
        let orders = self.orders.read().await;
        orders.values().filter(|o| !o.state.is_terminal()).count()
    }

    /// Cleanup old completed orders.
    ///
    /// Removes orders that are in terminal state and older than
    /// the retention period. Call this periodically to prevent
    /// memory growth.
    pub async fn cleanup_old_orders(&self) -> usize {
        let cutoff = Utc::now() - chrono::Duration::seconds(self.retention_secs as i64);
        let mut orders = self.orders.write().await;

        let to_remove: Vec<OrderId> = orders
            .iter()
            .filter(|(_, o)| o.state.is_terminal() && o.updated_at < cutoff)
            .map(|(id, _)| id.clone())
            .collect();

        let removed_count = to_remove.len();
        for id in &to_remove {
            orders.remove(id);
        }

        if removed_count > 0 {
            debug!(count = removed_count, "Cleaned up old orders");
        }

        removed_count
    }

    /// Clear all orders (for testing or reset).
    pub async fn clear(&self) {
        let mut orders = self.orders.write().await;
        orders.clear();
    }
}

impl Default for OrderTracker {
    fn default() -> Self {
        Self::new(3600) // 1 hour default retention
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_order_lifecycle() {
        let tracker = OrderTracker::new(3600);

        let id = OrderId::new("test-order-1");
        tracker
            .register_order(id.clone(), "AAPL".to_string(), OrderSide::Buy, dec!(100))
            .await;

        // Check initial state
        let order = tracker.get_order(&id).await.unwrap();
        assert_eq!(order.state, OrderState::Pending);
        assert_eq!(order.fill.filled_qty, dec!(0));

        // Mark as open
        tracker.mark_open(&id).await;
        let order = tracker.get_order(&id).await.unwrap();
        assert_eq!(order.state, OrderState::Open);

        // Record partial fill
        tracker.record_fill(&id, dec!(50), dec!(150.25)).await;
        let order = tracker.get_order(&id).await.unwrap();
        assert_eq!(order.state, OrderState::PartiallyFilled);
        assert_eq!(order.fill.filled_qty, dec!(50));

        // Record complete fill
        tracker.record_fill(&id, dec!(100), dec!(150.30)).await;
        let order = tracker.get_order(&id).await.unwrap();
        assert_eq!(order.state, OrderState::Filled);
        assert!(order.is_terminal());
    }

    #[tokio::test]
    async fn test_orders_for_symbol() {
        let tracker = OrderTracker::new(3600);

        tracker
            .register_order(
                OrderId::new("order-1"),
                "AAPL".to_string(),
                OrderSide::Buy,
                dec!(10),
            )
            .await;
        tracker
            .register_order(
                OrderId::new("order-2"),
                "AAPL".to_string(),
                OrderSide::Sell,
                dec!(5),
            )
            .await;
        tracker
            .register_order(
                OrderId::new("order-3"),
                "GOOGL".to_string(),
                OrderSide::Buy,
                dec!(20),
            )
            .await;

        let aapl_orders = tracker.orders_for_symbol("AAPL").await;
        assert_eq!(aapl_orders.len(), 2);

        let googl_orders = tracker.orders_for_symbol("GOOGL").await;
        assert_eq!(googl_orders.len(), 1);
    }

    #[tokio::test]
    async fn test_cleanup_old_orders() {
        let tracker = OrderTracker::new(0); // 0 second retention for testing

        let id = OrderId::new("old-order");
        tracker
            .register_order(id.clone(), "AAPL".to_string(), OrderSide::Buy, dec!(10))
            .await;

        // Mark as filled (terminal)
        tracker.record_fill(&id, dec!(10), dec!(150.00)).await;

        // Cleanup should remove it (retention is 0)
        tokio::time::sleep(Duration::from_millis(10)).await;
        let removed = tracker.cleanup_old_orders().await;
        assert_eq!(removed, 1);

        assert!(tracker.get_order(&id).await.is_none());
    }

    #[tokio::test]
    async fn test_active_orders() {
        let tracker = OrderTracker::new(3600);

        tracker
            .register_order(
                OrderId::new("active-1"),
                "AAPL".to_string(),
                OrderSide::Buy,
                dec!(10),
            )
            .await;
        tracker
            .register_order(
                OrderId::new("active-2"),
                "AAPL".to_string(),
                OrderSide::Sell,
                dec!(5),
            )
            .await;

        // Complete one order
        tracker
            .record_fill(&OrderId::new("active-1"), dec!(10), dec!(150.00))
            .await;

        assert_eq!(tracker.active_order_count().await, 1);
        assert_eq!(tracker.order_count().await, 2);
    }
}
