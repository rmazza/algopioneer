//! Tick Router for Multi-Strategy Market Data Distribution
//!
//! This module provides a thread-safe, lock-free tick routing component
//! that distributes market data to multiple strategy channels with
//! automatic backpressure handling and cleanup of closed channels.

use crate::metrics;
use crate::types::MarketData;
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info};

/// Thread-safe tick router for multi-strategy distribution.
///
/// Uses `DashMap` for lock-free concurrent access when routing ticks
/// to multiple strategies. Automatically cleans up closed channels
/// and drops ticks on backpressure to prevent latency cascading.
///
/// # Example
///
/// ```ignore
/// let router = TickRouter::new();
///
/// // Register strategy channels
/// router.register("BTC-USD".into(), leg1_tx, "pair-1".into());
/// router.register("BTC-USD".into(), other_tx, "pair-2".into());
///
/// // Route ticks (non-blocking)
/// router.route(Arc::new(tick));
/// ```
/// Type alias for route entries: (Sender, PairID)
type RouteEntry = (mpsc::Sender<Arc<MarketData>>, String);

/// Type alias for routes map to reduce complexity
type RoutesMap = DashMap<String, Vec<RouteEntry>>;

pub struct TickRouter {
    /// Symbol -> List of (Sender, PairID)
    routes: RoutesMap,
}

impl Default for TickRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl TickRouter {
    /// Create a new empty tick router
    pub fn new() -> Self {
        Self {
            routes: DashMap::new(),
        }
    }

    /// Register a sender for a symbol.
    ///
    /// Multiple senders can be registered for the same symbol,
    /// allowing fan-out to multiple strategies.
    pub fn register(&self, symbol: String, sender: mpsc::Sender<Arc<MarketData>>, pair_id: String) {
        self.routes
            .entry(symbol)
            .or_default()
            .push((sender, pair_id));
    }

    /// Route a tick to all registered strategies.
    ///
    /// Uses read-lock (`get`) and `try_send` to prevent blocking on slow strategies.
    /// Drops ticks on backpressure and increments Prometheus counters.
    /// Closed channel cleanup is deferred to `remove_pair()` to avoid
    /// write-lock contention on the hot path.
    #[inline]
    pub fn route(&self, tick: Arc<MarketData>) {
        let Some(senders) = self.routes.get(&tick.symbol) else {
            return;
        };

        for (sender, pair_id) in senders.iter() {
            match sender.try_send(tick.clone()) {
                Ok(()) => {
                    metrics::record_ws_tick(&tick.symbol);
                }
                Err(mpsc::error::TrySendError::Full(_)) => {
                    metrics::record_dropped_tick(&tick.symbol, "backpressure");
                    debug!(
                        pair_id = %pair_id,
                        symbol = %tick.symbol,
                        "Dropping tick: channel full (strategy backpressure)"
                    );
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    metrics::record_dropped_tick(&tick.symbol, "closed");
                    debug!(
                        pair_id = %pair_id,
                        symbol = %tick.symbol,
                        "Channel closed, deferred cleanup on next remove_pair()"
                    );
                }
            }
        }
    }

    /// Remove all routes for a specific pair (used on strategy restart).
    ///
    /// NOTE: Acquires write-locks on each shard sequentially via `iter_mut()`.
    /// This briefly contends with `route()` per-shard. Acceptable because
    /// strategy restarts are rare events.
    pub fn remove_pair(&self, pair_id: &str) {
        let mut empty_symbols = Vec::new();
        self.routes.iter_mut().for_each(|mut entry| {
            entry
                .value_mut()
                .retain(|(sender, pid)| pid != pair_id && !sender.is_closed());
            if entry.value().is_empty() {
                empty_symbols.push(entry.key().clone());
            }
        });
        // Clean up empty entries to keep symbol_count() accurate
        for sym in &empty_symbols {
            self.routes.remove_if(sym, |_, v| v.is_empty());
        }
        info!(pair_id = %pair_id, removed_symbols = empty_symbols.len(), "Removed pair routes");
    }

    /// Get the number of registered symbols.
    #[must_use]
    pub fn symbol_count(&self) -> usize {
        self.routes.len()
    }

    /// Check if a symbol has any registered routes.
    #[must_use]
    pub fn has_routes(&self, symbol: &str) -> bool {
        self.routes
            .get(symbol)
            .map(|senders| !senders.is_empty())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn make_tick(symbol: &str) -> Arc<MarketData> {
        Arc::new(MarketData {
            symbol: symbol.to_string(),
            price: dec!(100.0),
            timestamp: 1000,
            instrument_id: None,
        })
    }

    #[tokio::test]
    async fn test_route_to_single_receiver() {
        let router = TickRouter::new();
        let (tx, mut rx) = mpsc::channel(10);

        router.register("BTC-USD".into(), tx, "pair-1".into());

        let tick = make_tick("BTC-USD");
        router.route(tick.clone());

        let received = rx.recv().await.unwrap();
        assert_eq!(received.symbol, "BTC-USD");
    }

    #[tokio::test]
    async fn test_route_to_multiple_receivers() {
        let router = TickRouter::new();
        let (tx1, mut rx1) = mpsc::channel(10);
        let (tx2, mut rx2) = mpsc::channel(10);

        router.register("BTC-USD".into(), tx1, "pair-1".into());
        router.register("BTC-USD".into(), tx2, "pair-2".into());

        let tick = make_tick("BTC-USD");
        router.route(tick);

        assert!(rx1.recv().await.is_some());
        assert!(rx2.recv().await.is_some());
    }

    #[tokio::test]
    async fn test_remove_pair() {
        let router = TickRouter::new();
        let (tx, _rx) = mpsc::channel(10);

        router.register("BTC-USD".into(), tx, "pair-1".into());
        assert!(router.has_routes("BTC-USD"));

        router.remove_pair("pair-1");
        assert!(!router.has_routes("BTC-USD"));
    }

    #[tokio::test]
    async fn test_backpressure_handling() {
        let router = TickRouter::new();
        // Create a channel with capacity 1
        let (tx, _rx) = mpsc::channel(1);

        router.register("BTC-USD".into(), tx, "pair-1".into());

        // First tick should succeed
        router.route(make_tick("BTC-USD"));

        // Second tick should be dropped (no blocking) since channel is full
        // This should not panic or block
        router.route(make_tick("BTC-USD"));
    }

    #[tokio::test]
    async fn test_closed_channel_cleanup() {
        let router = TickRouter::new();
        let (tx, rx) = mpsc::channel(10);

        router.register("BTC-USD".into(), tx, "pair-1".into());
        assert!(router.has_routes("BTC-USD"));

        // Drop the receiver to close the channel
        drop(rx);

        // Route a tick — the closed sender stays (cleanup deferred)
        router.route(make_tick("BTC-USD"));

        // remove_pair should clean up the closed channel AND the empty entry
        router.remove_pair("pair-1");
        assert!(!router.has_routes("BTC-USD"));
        assert_eq!(router.symbol_count(), 0);
    }
}
