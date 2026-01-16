//! Strategy Supervisor
//!
//! Generic supervisor that manages multiple `LiveStrategy` instances with:
//! - Unified WebSocket market data routing
//! - Panic recovery and automatic restart
//! - Aggregated PnL tracking
//! - Health monitoring

use crate::exchange::WebSocketProvider;
use crate::strategy::{LiveStrategy, MarketData, StrategyInput};
use dashmap::DashMap;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tracing::{debug, error, info};

/// Errors that can occur in the supervisor
#[derive(Error, Debug)]
pub enum SupervisorError {
    #[error("WebSocket connection failed: {0}")]
    WebSocketError(String),
    #[error("No strategies configured")]
    NoStrategies,
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Policy for restarting crashed strategies
///
/// Implements exponential backoff with jitter for resilient strategy recovery.
/// Strategies that panic will be automatically restarted up to `max_restarts`
/// times with increasing delays between attempts.
///
/// # MC-5 Known Limitation: Auto-Restart Not Implemented
///
/// **IMPORTANT**: While this policy calculates backoff delays and tracks restart
/// budgets, the actual restart logic is NOT YET IMPLEMENTED. Currently, when a
/// strategy panics:
///
/// 1. The supervisor logs the failure with backoff timing info
/// 2. The restart budget is decremented
/// 3. But the strategy is NOT automatically restarted
///
/// Implementing auto-restart requires a "Strategy Factory" pattern where the
/// supervisor can reconstruct a strategy from its configuration. This is tracked
/// as a future enhancement.
///
/// For now, strategy panics require manual intervention or process restart.
#[derive(Debug, Clone)]
pub struct RestartPolicy {
    /// Maximum restart attempts before giving up (0 = no restarts)
    pub max_restarts: u32,
    /// Initial delay before first restart attempt (milliseconds)
    pub initial_delay_ms: u64,
    /// Maximum delay cap (milliseconds)
    pub max_delay_ms: u64,
    /// Multiplier for exponential backoff (e.g., 2.0 = doubling)
    pub backoff_multiplier: f64,
    /// Random jitter as fraction of delay (e.g., 0.1 = ±10%)
    pub jitter_fraction: f64,
    /// Reset restart count after this many seconds of stable operation
    pub cooldown_period_secs: u64,
    /// Whether to halt all strategies if one fails permanently
    pub halt_on_permanent_failure: bool,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self {
            max_restarts: 3,
            initial_delay_ms: 1000,    // 1 second
            max_delay_ms: 60_000,      // 1 minute cap
            backoff_multiplier: 2.0,   // Double each time
            jitter_fraction: 0.1,      // ±10%
            cooldown_period_secs: 300, // 5 minutes of stability resets count
            halt_on_permanent_failure: false,
        }
    }
}

impl RestartPolicy {
    /// Calculate backoff delay with jitter for the given attempt number
    pub fn calculate_delay(&self, attempt: u32) -> std::time::Duration {
        use rand::Rng;

        // Exponential backoff: initial * multiplier^attempt
        let base_delay =
            self.initial_delay_ms as f64 * self.backoff_multiplier.powi(attempt as i32);

        // Cap at max delay
        let capped_delay = base_delay.min(self.max_delay_ms as f64);

        // Add jitter: ±jitter_fraction
        let jitter_range = capped_delay * self.jitter_fraction;
        let jitter = rand::rng().random_range(-jitter_range..=jitter_range);
        let final_delay = (capped_delay + jitter).max(0.0) as u64;

        std::time::Duration::from_millis(final_delay)
    }
}

/// Tracks restart state for a single strategy
#[derive(Debug, Clone, Default)]
struct StrategyRestartState {
    /// Number of restart attempts since last cooldown reset
    restart_count: u32,
    /// When the strategy last started successfully
    last_start: Option<std::time::Instant>,
    /// When the strategy last failed
    last_failure: Option<std::time::Instant>,
}

impl StrategyRestartState {
    /// Check if cooldown period has passed and reset count if so
    fn maybe_reset_cooldown(&mut self, cooldown_secs: u64) {
        if let Some(last_start) = self.last_start {
            if last_start.elapsed().as_secs() >= cooldown_secs {
                debug!(
                    "Cooldown period passed ({} secs), resetting restart count from {}",
                    cooldown_secs, self.restart_count
                );
                self.restart_count = 0;
            }
        }
    }

    /// Record a successful start
    fn record_start(&mut self) {
        self.last_start = Some(std::time::Instant::now());
    }

    /// Record a failure and increment restart count
    fn record_failure(&mut self) {
        self.restart_count += 1;
        self.last_failure = Some(std::time::Instant::now());
    }

    /// Check if more restarts are allowed
    fn can_restart(&self, max_restarts: u32) -> bool {
        self.restart_count < max_restarts
    }

    /// Get time since last failure (for crash loop detection logging)
    fn time_since_last_failure(&self) -> Option<std::time::Duration> {
        self.last_failure.map(|t| t.elapsed())
    }
}

/// Aggregated PnL tracker across all strategies
pub struct PortfolioPnL {
    /// Strategy ID -> Current PnL
    strategy_pnl: DashMap<String, Decimal>,
}

impl Default for PortfolioPnL {
    fn default() -> Self {
        Self::new()
    }
}

impl PortfolioPnL {
    pub fn new() -> Self {
        Self {
            strategy_pnl: DashMap::new(),
        }
    }

    /// Update PnL for a strategy.
    ///
    /// # Performance (P-CB-2 FIX)
    /// Uses `get_mut` to avoid allocation when key exists (common case).
    /// Only allocates on first insertion per strategy.
    #[inline]
    pub fn update(&self, strategy_id: &str, pnl: Decimal) {
        // Fast path: key already exists (no allocation)
        if let Some(mut entry) = self.strategy_pnl.get_mut(strategy_id) {
            *entry = pnl;
            return;
        }
        // Slow path: first time seeing this strategy
        self.strategy_pnl.insert(strategy_id.to_string(), pnl);
    }

    pub fn total(&self) -> Decimal {
        self.strategy_pnl.iter().map(|entry| *entry.value()).sum()
    }

    pub fn get(&self, strategy_id: &str) -> Decimal {
        self.strategy_pnl
            .get(strategy_id)
            .map(|v| *v)
            .unwrap_or(Decimal::ZERO)
    }
}

/// Result of a strategy run (for supervisor tracking)
pub struct StrategyRunResult {
    pub id: String,
    pub panicked: bool,
    pub error: Option<String>,
    pub final_pnl: Decimal,
}

/// Generic supervisor for managing multiple `LiveStrategy` instances
pub struct StrategySupervisor {
    strategies: Vec<Box<dyn LiveStrategy>>,
    pnl_tracker: Arc<PortfolioPnL>,
    restart_policy: RestartPolicy,
    risk_engine: Arc<crate::risk::DailyRiskEngine>,
    /// Map of symbol -> list of (strategy_id, sender) for routing
    symbol_routes: HashMap<String, Vec<(String, mpsc::Sender<StrategyInput>)>>,
}

impl StrategySupervisor {
    /// Create a new supervisor
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
            pnl_tracker: Arc::new(PortfolioPnL::new()),
            restart_policy: RestartPolicy::default(),
            risk_engine: Arc::new(crate::risk::DailyRiskEngine::with_defaults()),
            symbol_routes: HashMap::new(),
        }
    }

    /// Set risk configuration
    pub fn with_risk_config(mut self, config: crate::risk::DailyRiskConfig) -> Self {
        self.risk_engine = Arc::new(crate::risk::DailyRiskEngine::new(config));
        self
    }

    /// Set the restart policy
    pub fn with_restart_policy(mut self, policy: RestartPolicy) -> Self {
        self.restart_policy = policy;
        self
    }

    /// Add a strategy to be supervised
    pub fn add_strategy(&mut self, strategy: Box<dyn LiveStrategy>) {
        info!(
            strategy_id = %strategy.id(),
            strategy_type = %strategy.strategy_type(),
            symbols = ?strategy.subscribed_symbols(),
            "Adding strategy to supervisor"
        );
        self.strategies.push(strategy);
    }

    /// Get all unique symbols that need market data subscription
    pub fn all_subscribed_symbols(&self) -> Vec<String> {
        let mut symbols: Vec<String> = self
            .strategies
            .iter()
            .flat_map(|s| s.subscribed_symbols())
            .collect();
        symbols.sort();
        symbols.dedup();
        symbols
    }

    /// Get the PnL tracker for external monitoring
    pub fn pnl_tracker(&self) -> Arc<PortfolioPnL> {
        self.pnl_tracker.clone()
    }

    /// Run all strategies with supervision
    pub async fn run(
        mut self,
        ws_provider: Box<dyn WebSocketProvider>,
    ) -> Result<(), SupervisorError> {
        if self.strategies.is_empty() {
            return Err(SupervisorError::NoStrategies);
        }

        info!(
            strategy_count = self.strategies.len(),
            "Starting strategy supervisor"
        );

        // Collect all symbols needed
        let all_symbols = self.all_subscribed_symbols();
        info!(symbols = ?all_symbols, "Subscribing to market data");

        // Create WebSocket data channel
        let (ws_tx, mut ws_rx) = mpsc::channel::<MarketData>(1000);

        // MC-3 FIX: Use spawn_and_subscribe which returns handle for structured concurrency
        let ws_symbols = all_symbols.clone();
        let ws_handle = match ws_provider.spawn_and_subscribe(ws_symbols, ws_tx).await {
            Ok(handle) => {
                info!("WebSocket task spawned with handle for lifecycle management");
                Some(handle)
            }
            Err(e) => {
                error!("WebSocket connection failed: {}", e);
                return Err(SupervisorError::WebSocketError(e.to_string()));
            }
        };

        // Create channels for each strategy and build routing table
        let mut strategy_senders: HashMap<String, mpsc::Sender<StrategyInput>> = HashMap::new();
        let mut strategy_receivers: HashMap<String, mpsc::Receiver<StrategyInput>> = HashMap::new();

        for strategy in &self.strategies {
            let (tx, rx) = mpsc::channel(100);
            let id = strategy.id();
            strategy_senders.insert(id.clone(), tx);
            strategy_receivers.insert(id, rx);
        }

        // Build symbol -> strategy routing
        for strategy in &self.strategies {
            let id = strategy.id();
            for symbol in strategy.subscribed_symbols() {
                if let Some(sender) = strategy_senders.get(&id) {
                    self.symbol_routes
                        .entry(symbol)
                        .or_default()
                        .push((id.clone(), sender.clone()));
                }
            }
        }

        // Spawn strategy tasks with proper panic recovery
        let mut join_set: JoinSet<StrategyRunResult> = JoinSet::new();

        for mut strategy in self.strategies.drain(..) {
            let id = strategy.id();
            // MC-5 FIX: Graceful handling instead of expect() panic
            let rx = match strategy_receivers.remove(&id) {
                Some(r) => r,
                None => {
                    error!(
                        strategy_id = %id,
                        "BUG: Receiver missing for strategy - skipping. This indicates a logic error in supervisor setup."
                    );
                    continue;
                }
            };
            let pnl_tracker = self.pnl_tracker.clone();

            join_set.spawn(async move {
                let strategy_id = strategy.id();
                let strategy_type = strategy.strategy_type();

                info!(
                    strategy_id = %strategy_id,
                    strategy_type = %strategy_type,
                    "Strategy starting"
                );

                // BI-2 FIX: Spawn the strategy in a separate task to catch panics
                // JoinError::is_panic() detects if the inner task panicked
                let strategy_id_clone = strategy_id.clone();
                let inner_handle = tokio::spawn(async move {
                    strategy.run(rx).await;
                    strategy.current_pnl()
                });

                let (panicked, final_pnl) = match inner_handle.await {
                    Ok(pnl) => {
                        info!(strategy_id = %strategy_id_clone, "Strategy completed normally");
                        (false, pnl)
                    }
                    Err(e) if e.is_panic() => {
                        let panic_info = e.into_panic();
                        let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                            s.to_string()
                        } else if let Some(s) = panic_info.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            "Unknown panic".to_string()
                        };
                        error!(
                            strategy_id = %strategy_id_clone,
                            panic_msg = %panic_msg,
                            "CRITICAL: Strategy panicked! Manual intervention may be required."
                        );
                        (true, rust_decimal::Decimal::ZERO)
                    }
                    Err(e) => {
                        error!(strategy_id = %strategy_id_clone, error = %e, "Strategy task cancelled");
                        (false, rust_decimal::Decimal::ZERO)
                    }
                };

                // Update PnL on completion
                pnl_tracker.update(&strategy_id, final_pnl);

                StrategyRunResult {
                    id: strategy_id,
                    panicked,
                    error: if panicked { Some("Strategy panicked".to_string()) } else { None },
                    final_pnl,
                }
            });
        }

        // Route market data to strategies
        let symbol_routes = self.symbol_routes.clone();
        tokio::spawn(async move {
            // Keep track of latest tick per symbol for paired routing
            let mut latest_ticks: HashMap<String, Arc<MarketData>> = HashMap::new();

            while let Some(data) = ws_rx.recv().await {
                let symbol = data.symbol.clone();
                let arc_data = Arc::new(data);
                latest_ticks.insert(symbol.clone(), arc_data.clone());

                if let Some(routes) = symbol_routes.get(&symbol) {
                    for (strategy_id, sender) in routes {
                        // For now, send as single tick
                        // Paired strategies will need to buffer internally
                        let input = StrategyInput::Tick(arc_data.clone());
                        if sender.try_send(input).is_err() {
                            debug!(
                                strategy_id = %strategy_id,
                                symbol = %symbol,
                                "Strategy channel full, dropping tick"
                            );
                        }
                    }
                }
            }
        });

        // Wait for strategies to complete with restart tracking
        // NP-5 FIX: Track restart state per strategy
        let mut restart_states: HashMap<String, StrategyRestartState> = HashMap::new();

        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(run_result) => {
                    // MC-4: Record PnL in daily risk engine
                    self.risk_engine.record_pnl(run_result.final_pnl);

                    // Get or create restart state for this strategy
                    let restart_state = restart_states.entry(run_result.id.clone()).or_default();

                    if run_result.panicked {
                        // NP-5 FIX: Check cooldown BEFORE recording failure
                        // If enough time has passed since last successful start, reset the budget
                        restart_state
                            .maybe_reset_cooldown(self.restart_policy.cooldown_period_secs);

                        // Capture time since PREVIOUS failure (before we update the timestamp)
                        let time_since_last_failure = restart_state
                            .time_since_last_failure()
                            .map(|d| d.as_secs())
                            .unwrap_or(0);

                        // Record the failure (updates last_failure timestamp)
                        restart_state.record_failure();

                        // Check if we can restart
                        if restart_state.can_restart(self.restart_policy.max_restarts) {
                            let delay = self
                                .restart_policy
                                .calculate_delay(restart_state.restart_count.saturating_sub(1));

                            // NP-5 FIX: Log restart attempt with backoff info
                            // Note: Actual restart not implemented - requires strategy factory pattern
                            error!(
                                strategy_id = %run_result.id,
                                error = ?run_result.error,
                                restart_attempt = restart_state.restart_count,
                                max_restarts = self.restart_policy.max_restarts,
                                would_backoff_delay_ms = delay.as_millis(),
                                secs_since_last_failure = time_since_last_failure,
                                "Strategy panicked! Restart budget remaining but auto-restart NOT implemented. Manual intervention required."
                            );
                        } else {
                            // Restart budget exhausted
                            error!(
                                strategy_id = %run_result.id,
                                restart_attempts = restart_state.restart_count,
                                max_restarts = self.restart_policy.max_restarts,
                                secs_since_last_failure = time_since_last_failure,
                                "CRITICAL: Strategy restart budget exhausted! Manual intervention required."
                            );

                            if self.restart_policy.halt_on_permanent_failure {
                                error!("halt_on_permanent_failure is set - supervisor should halt all strategies");
                                // Future: signal other strategies to gracefully shutdown
                            }
                        }
                    } else {
                        // Strategy exited normally - record stable start time for cooldown tracking
                        restart_state.record_start();

                        info!(
                            strategy_id = %run_result.id,
                            final_pnl = %run_result.final_pnl,
                            "Strategy exited normally"
                        );
                    }
                }
                Err(e) => {
                    error!("Strategy task failed: {}", e);
                }
            }
        }

        // MC-3 FIX: Cleanup WebSocket handle on supervisor exit
        if let Some(handle) = ws_handle {
            if handle.is_finished() {
                // Task already finished - check if it panicked
                match handle.join().await {
                    Ok(()) => info!("WebSocket task completed normally"),
                    Err(e) => error!("WebSocket task panicked: {:?}", e),
                }
            } else {
                // Task still running - abort it
                info!("Aborting WebSocket task on supervisor shutdown");
                handle.abort();
            }
        }

        Ok(())
    }
}

impl Default for StrategySupervisor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_pnl() {
        let pnl = PortfolioPnL::new();
        pnl.update("strategy1", Decimal::new(100, 0));
        pnl.update("strategy2", Decimal::new(-50, 0));

        assert_eq!(pnl.get("strategy1"), Decimal::new(100, 0));
        assert_eq!(pnl.get("strategy2"), Decimal::new(-50, 0));
        assert_eq!(pnl.total(), Decimal::new(50, 0));
    }

    #[test]
    fn test_restart_policy_default() {
        let policy = RestartPolicy::default();
        assert_eq!(policy.max_restarts, 3);
        assert_eq!(policy.initial_delay_ms, 1000);
        assert_eq!(policy.max_delay_ms, 60_000);
        assert!((policy.backoff_multiplier - 2.0).abs() < f64::EPSILON);
        assert!((policy.jitter_fraction - 0.1).abs() < f64::EPSILON);
        assert_eq!(policy.cooldown_period_secs, 300);
        assert!(!policy.halt_on_permanent_failure);
    }

    #[test]
    fn test_restart_policy_backoff_calculation() {
        let policy = RestartPolicy {
            max_restarts: 5,
            initial_delay_ms: 1000,
            max_delay_ms: 10_000,
            backoff_multiplier: 2.0,
            jitter_fraction: 0.0, // No jitter for deterministic test
            cooldown_period_secs: 300,
            halt_on_permanent_failure: false,
        };

        // Attempt 0: 1000 * 2^0 = 1000ms
        let delay0 = policy.calculate_delay(0);
        assert_eq!(delay0.as_millis(), 1000);

        // Attempt 1: 1000 * 2^1 = 2000ms
        let delay1 = policy.calculate_delay(1);
        assert_eq!(delay1.as_millis(), 2000);

        // Attempt 2: 1000 * 2^2 = 4000ms
        let delay2 = policy.calculate_delay(2);
        assert_eq!(delay2.as_millis(), 4000);

        // Attempt 3: 1000 * 2^3 = 8000ms
        let delay3 = policy.calculate_delay(3);
        assert_eq!(delay3.as_millis(), 8000);

        // Attempt 4: 1000 * 2^4 = 16000ms, but capped at 10000ms
        let delay4 = policy.calculate_delay(4);
        assert_eq!(delay4.as_millis(), 10_000);
    }

    #[test]
    fn test_strategy_restart_state() {
        let mut state = StrategyRestartState::default();

        assert_eq!(state.restart_count, 0);
        assert!(state.can_restart(3));

        state.record_failure();
        assert_eq!(state.restart_count, 1);
        assert!(state.can_restart(3));

        state.record_failure();
        state.record_failure();
        assert_eq!(state.restart_count, 3);
        assert!(!state.can_restart(3)); // Budget exhausted
    }

    #[test]
    fn test_time_since_last_failure() {
        let mut state = StrategyRestartState::default();

        // No failure yet
        assert!(state.time_since_last_failure().is_none());

        // Record a failure
        state.record_failure();
        let elapsed = state.time_since_last_failure().unwrap();
        // Should be very recent (within 100ms of recording)
        assert!(elapsed.as_millis() < 100);
    }
}
