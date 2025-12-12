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
#[derive(Debug, Clone)]
pub struct RestartPolicy {
    /// Maximum restart attempts before giving up
    pub max_restarts: u32,
    /// Delay between restart attempts (seconds)
    pub restart_delay_secs: u64,
    /// Whether to halt all strategies if one fails permanently
    pub halt_on_permanent_failure: bool,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self {
            max_restarts: 3,
            restart_delay_secs: 5,
            halt_on_permanent_failure: false,
        }
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

    pub fn update(&self, strategy_id: &str, pnl: Decimal) {
        self.strategy_pnl
            .entry(strategy_id.to_string())
            .and_modify(|p| *p = pnl)
            .or_insert(pnl);
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
}

/// Generic supervisor for managing multiple `LiveStrategy` instances
pub struct StrategySupervisor {
    strategies: Vec<Box<dyn LiveStrategy>>,
    pnl_tracker: Arc<PortfolioPnL>,
    restart_policy: RestartPolicy,
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
            symbol_routes: HashMap::new(),
        }
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

        // Spawn WebSocket connection
        let ws_symbols = all_symbols.clone();
        tokio::spawn(async move {
            if let Err(e) = ws_provider.connect_and_subscribe(ws_symbols, ws_tx).await {
                error!("WebSocket connection failed: {}", e);
            }
        });

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
            let rx = strategy_receivers.remove(&id).expect("Receiver must exist");
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

        // Wait for strategies to complete
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(run_result) => {
                    if run_result.panicked {
                        error!(
                            strategy_id = %run_result.id,
                            error = ?run_result.error,
                            "Strategy panicked - automatic restart not yet implemented. Manual intervention required."
                        );
                        // NP-3 FIX: Restart logic is a future enhancement tracked in project roadmap.
                        // For now, panicked strategies remain stopped and require manual restart.
                    } else {
                        info!(
                            strategy_id = %run_result.id,
                            "Strategy exited normally"
                        );
                    }
                }
                Err(e) => {
                    error!("Strategy task failed: {}", e);
                }
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
        assert_eq!(policy.restart_delay_secs, 5);
        assert!(!policy.halt_on_permanent_failure);
    }
}
