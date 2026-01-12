//! Position reconciliation with exchange.
//!
//! Provides detection and correction of position drift between
//! local tracking and exchange state.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use rust_decimal::Decimal;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::exchange::{ExchangeError, Executor};

/// Result of a position reconciliation check.
#[derive(Debug, Clone)]
pub struct ReconciliationResult {
    /// Symbol that was reconciled
    pub symbol: String,
    /// Position according to local tracking
    pub local_position: Decimal,
    /// Position according to exchange
    pub exchange_position: Decimal,
    /// Absolute difference between local and exchange
    pub drift: Decimal,
    /// Action taken to resolve drift
    pub action_taken: ReconciliationAction,
}

/// Action taken during reconciliation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconciliationAction {
    /// Positions matched, no action needed
    NoAction,
    /// Local state updated to match exchange (drift was small)
    LocalCorrected,
    /// Drift within tolerance, logged warning only
    ToleratedDrift,
    /// Critical drift detected, trading halted for symbol
    TradingHalted,
}

impl std::fmt::Display for ReconciliationAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAction => write!(f, "NoAction"),
            Self::LocalCorrected => write!(f, "LocalCorrected"),
            Self::ToleratedDrift => write!(f, "ToleratedDrift"),
            Self::TradingHalted => write!(f, "TradingHalted"),
        }
    }
}

/// Configuration for position reconciliation.
#[derive(Debug, Clone)]
pub struct ReconciliationConfig {
    /// Maximum allowed position drift before warning (in base units)
    pub warning_threshold: Decimal,
    /// Maximum allowed position drift before halting trading (in base units)
    pub halt_threshold: Decimal,
    /// Interval between periodic reconciliation checks (seconds)
    pub check_interval_secs: u64,
    /// Whether to auto-correct small drifts
    pub auto_correct: bool,
}

impl Default for ReconciliationConfig {
    fn default() -> Self {
        Self {
            warning_threshold: Decimal::new(1, 4), // 0.0001
            halt_threshold: Decimal::new(1, 2),    // 0.01
            check_interval_secs: 60,
            auto_correct: true,
        }
    }
}

impl ReconciliationConfig {
    /// Config for high-frequency trading (tighter thresholds)
    pub fn strict() -> Self {
        Self {
            warning_threshold: Decimal::new(1, 6), // 0.000001
            halt_threshold: Decimal::new(1, 4),    // 0.0001
            check_interval_secs: 30,
            auto_correct: true,
        }
    }

    /// Config for paper trading (relaxed thresholds)
    pub fn relaxed() -> Self {
        Self {
            warning_threshold: Decimal::new(1, 2), // 0.01
            halt_threshold: Decimal::new(1, 0),    // 1.0
            check_interval_secs: 300,              // 5 minutes
            auto_correct: true,
        }
    }
}

/// Position reconciler that syncs local state with exchange.
///
/// Tracks local position changes and periodically verifies against
/// the exchange to detect and correct drift.
///
/// # Architecture
///
/// - Local positions are updated via `update_local_position()` when orders fill
/// - Periodic reconciliation compares with `executor.get_position()`
/// - Drift above thresholds triggers warnings or trading halt
pub struct PositionReconciler<E: Executor> {
    executor: Arc<E>,
    config: ReconciliationConfig,
    /// Local position state (symbol -> quantity)
    local_positions: Arc<RwLock<HashMap<String, Decimal>>>,
    /// Symbols with halted trading due to critical drift
    halted_symbols: Arc<RwLock<HashMap<String, String>>>,
}

impl<E: Executor + 'static> PositionReconciler<E> {
    /// Create a new PositionReconciler.
    pub fn new(executor: Arc<E>, config: ReconciliationConfig) -> Self {
        Self {
            executor,
            config,
            local_positions: Arc::new(RwLock::new(HashMap::new())),
            halted_symbols: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults(executor: Arc<E>) -> Self {
        Self::new(executor, ReconciliationConfig::default())
    }

    /// Update local position tracking (call after fill).
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading symbol
    /// * `delta` - Position change (positive for buy, negative for sell)
    pub async fn update_local_position(&self, symbol: &str, delta: Decimal) {
        let mut positions = self.local_positions.write().await;
        let current = positions.entry(symbol.to_string()).or_insert(Decimal::ZERO);
        *current += delta;
        debug!(
            symbol = symbol,
            delta = %delta,
            new_position = %*current,
            "Local position updated"
        );
    }

    /// Set local position to absolute value (use for initialization or correction).
    pub async fn set_local_position(&self, symbol: &str, quantity: Decimal) {
        let mut positions = self.local_positions.write().await;
        positions.insert(symbol.to_string(), quantity);
        debug!(
            symbol = symbol,
            position = %quantity,
            "Local position set"
        );
    }

    /// Get local position for a symbol.
    pub async fn get_local_position(&self, symbol: &str) -> Decimal {
        let positions = self.local_positions.read().await;
        positions.get(symbol).copied().unwrap_or(Decimal::ZERO)
    }

    /// Check if trading is halted for a symbol.
    pub async fn is_halted(&self, symbol: &str) -> bool {
        let halted = self.halted_symbols.read().await;
        halted.contains_key(symbol)
    }

    /// Clear halt status for a symbol (after manual intervention).
    pub async fn clear_halt(&self, symbol: &str) {
        let mut halted = self.halted_symbols.write().await;
        if halted.remove(symbol).is_some() {
            info!(symbol = symbol, "Trading halt cleared");
        }
    }

    /// Reconcile a single symbol with exchange.
    pub async fn reconcile_symbol(
        &self,
        symbol: &str,
    ) -> Result<ReconciliationResult, ExchangeError> {
        let local = self.get_local_position(symbol).await;
        let exchange = self.executor.get_position(symbol).await?;
        let drift = (exchange - local).abs();

        let action = if drift.is_zero() {
            ReconciliationAction::NoAction
        } else if drift >= self.config.halt_threshold {
            // Critical drift - halt trading
            error!(
                symbol = symbol,
                local = %local,
                exchange = %exchange,
                drift = %drift,
                threshold = %self.config.halt_threshold,
                "CRITICAL: Position drift exceeds halt threshold - trading halted"
            );

            let mut halted = self.halted_symbols.write().await;
            halted.insert(
                symbol.to_string(),
                format!(
                    "Drift {} exceeds threshold {}",
                    drift, self.config.halt_threshold
                ),
            );

            ReconciliationAction::TradingHalted
        } else if drift >= self.config.warning_threshold {
            // Significant drift - warn and optionally correct
            warn!(
                symbol = symbol,
                local = %local,
                exchange = %exchange,
                drift = %drift,
                "Position drift detected"
            );

            if self.config.auto_correct {
                self.set_local_position(symbol, exchange).await;
                info!(
                    symbol = symbol,
                    "Local position auto-corrected to match exchange"
                );
                ReconciliationAction::LocalCorrected
            } else {
                ReconciliationAction::ToleratedDrift
            }
        } else {
            // Minor drift within tolerance
            if drift > Decimal::ZERO {
                debug!(
                    symbol = symbol,
                    drift = %drift,
                    "Minor position drift within tolerance"
                );
            }
            ReconciliationAction::ToleratedDrift
        };

        Ok(ReconciliationResult {
            symbol: symbol.to_string(),
            local_position: local,
            exchange_position: exchange,
            drift,
            action_taken: action,
        })
    }

    /// Reconcile all tracked symbols.
    pub async fn reconcile_all(&self) -> Vec<ReconciliationResult> {
        let symbols: Vec<String> = {
            let positions = self.local_positions.read().await;
            positions.keys().cloned().collect()
        };

        let mut results = Vec::with_capacity(symbols.len());
        for symbol in symbols {
            match self.reconcile_symbol(&symbol).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    error!(symbol = symbol, error = %e, "Failed to reconcile symbol");
                }
            }
        }

        results
    }

    /// Initialize positions from exchange (call on startup).
    ///
    /// Queries the exchange for current positions and initializes
    /// local tracking to match.
    pub async fn initialize_from_exchange(&self, symbols: &[String]) -> Result<(), ExchangeError> {
        info!(symbols = ?symbols, "Initializing positions from exchange");

        for symbol in symbols {
            let position = self.executor.get_position(symbol).await?;
            self.set_local_position(symbol, position).await;
            info!(
                symbol = symbol,
                position = %position,
                "Position initialized from exchange"
            );
        }

        Ok(())
    }

    /// Start periodic reconciliation background task.
    ///
    /// Returns a `JoinHandle` that can be used to abort the task.
    pub fn start_periodic_reconciliation(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        let interval = Duration::from_secs(self.config.check_interval_secs);

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;

                let results = self.reconcile_all().await;

                let corrections: Vec<_> = results
                    .iter()
                    .filter(|r| r.action_taken != ReconciliationAction::NoAction)
                    .collect();

                if !corrections.is_empty() {
                    info!(
                        total = results.len(),
                        corrections = corrections.len(),
                        "Periodic reconciliation completed"
                    );

                    for c in corrections {
                        info!(
                            symbol = %c.symbol,
                            action = %c.action_taken,
                            drift = %c.drift,
                            "Reconciliation action"
                        );
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use rust_decimal_macros::dec;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Mock executor for testing
    struct MockExecutor {
        positions: RwLock<HashMap<String, Decimal>>,
        should_fail: AtomicBool,
    }

    impl MockExecutor {
        fn new() -> Self {
            Self {
                positions: RwLock::new(HashMap::new()),
                should_fail: AtomicBool::new(false),
            }
        }

        async fn set_position(&self, symbol: &str, qty: Decimal) {
            let mut positions = self.positions.write().await;
            positions.insert(symbol.to_string(), qty);
        }
    }

    #[async_trait]
    impl Executor for MockExecutor {
        async fn execute_order(
            &self,
            _symbol: &str,
            _side: crate::types::OrderSide,
            _quantity: Decimal,
            _price: Option<Decimal>,
        ) -> Result<crate::orders::OrderId, ExchangeError> {
            Ok(crate::orders::OrderId::new("mock-reconciler-order"))
        }

        async fn get_position(&self, symbol: &str) -> Result<Decimal, ExchangeError> {
            if self.should_fail.load(Ordering::SeqCst) {
                return Err(ExchangeError::Network("Mock failure".to_string()));
            }
            let positions = self.positions.read().await;
            Ok(positions.get(symbol).copied().unwrap_or(Decimal::ZERO))
        }
    }

    #[tokio::test]
    async fn test_reconciliation_no_drift() {
        let executor = Arc::new(MockExecutor::new());
        executor.set_position("AAPL", dec!(100)).await;

        let reconciler = PositionReconciler::new(executor, ReconciliationConfig::default());
        reconciler.set_local_position("AAPL", dec!(100)).await;

        let result = reconciler.reconcile_symbol("AAPL").await.unwrap();
        assert_eq!(result.action_taken, ReconciliationAction::NoAction);
        assert_eq!(result.drift, dec!(0));
    }

    #[tokio::test]
    async fn test_reconciliation_with_correction() {
        let executor = Arc::new(MockExecutor::new());
        executor.set_position("AAPL", dec!(100)).await;

        let config = ReconciliationConfig {
            warning_threshold: dec!(0.01),
            halt_threshold: dec!(10),
            auto_correct: true,
            ..Default::default()
        };

        let reconciler = PositionReconciler::new(executor, config);
        reconciler.set_local_position("AAPL", dec!(99)).await; // 1 unit drift

        let result = reconciler.reconcile_symbol("AAPL").await.unwrap();
        assert_eq!(result.action_taken, ReconciliationAction::LocalCorrected);

        // Verify local was corrected
        let local = reconciler.get_local_position("AAPL").await;
        assert_eq!(local, dec!(100));
    }

    #[tokio::test]
    async fn test_reconciliation_halt() {
        let executor = Arc::new(MockExecutor::new());
        executor.set_position("AAPL", dec!(100)).await;

        let config = ReconciliationConfig {
            warning_threshold: dec!(0.01),
            halt_threshold: dec!(5), // Low threshold for testing
            auto_correct: true,
            ..Default::default()
        };

        let reconciler = PositionReconciler::new(executor, config);
        reconciler.set_local_position("AAPL", dec!(90)).await; // 10 unit drift

        let result = reconciler.reconcile_symbol("AAPL").await.unwrap();
        assert_eq!(result.action_taken, ReconciliationAction::TradingHalted);

        // Verify symbol is halted
        assert!(reconciler.is_halted("AAPL").await);

        // Clear halt
        reconciler.clear_halt("AAPL").await;
        assert!(!reconciler.is_halted("AAPL").await);
    }

    #[tokio::test]
    async fn test_update_local_position() {
        let executor = Arc::new(MockExecutor::new());
        let reconciler = PositionReconciler::with_defaults(executor);

        assert_eq!(reconciler.get_local_position("AAPL").await, dec!(0));

        reconciler.update_local_position("AAPL", dec!(50)).await;
        assert_eq!(reconciler.get_local_position("AAPL").await, dec!(50));

        reconciler.update_local_position("AAPL", dec!(-20)).await;
        assert_eq!(reconciler.get_local_position("AAPL").await, dec!(30));
    }
}
