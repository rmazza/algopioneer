use crate::coinbase::websocket::CoinbaseWebsocket;
use crate::coinbase::{AppEnv, CoinbaseClient};
use crate::strategy::dual_leg_trading::{
    DualLegConfig, DualLegStrategy, ExecutionEngine, HedgeMode, InstrumentType, MarketData,
    PairsManager, RecoveryWorker, RiskMonitor, SystemClock,
};

use rust_decimal::Decimal;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio::task::JoinSet;
use tokio::time::Duration;
use tokio_util::sync::CancellationToken;
// OW5: Use DashMap for lock-free concurrent PnL updates
use dashmap::DashMap;
use thiserror::Error;
use tracing::{debug, error, info, instrument, warn};

use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;

// Channel buffer size constants
const WS_CHANNEL_BUFFER: usize = 1000; // ~1 sec of ticks at 1000 ticks/sec
const STRATEGY_CHANNEL_BUFFER: usize = 100; // ~100ms buffer per leg
const RECOVERY_CHANNEL_BUFFER: usize = 100;
const FEEDBACK_CHANNEL_BUFFER: usize = 100;

/// Type alias for the symbol routing map to reduce type complexity
type SymbolRouterMap = HashMap<String, Vec<(mpsc::Sender<Arc<MarketData>>, String)>>;

// Reconnection constants
const MAX_RECONNECT_BACKOFF_SECS: u64 = 30;
const MAX_RECONNECT_ATTEMPTS: u32 = 5;

// Execution engine configuration (NP4)
const EXECUTION_MAX_RETRIES: u32 = 5;
const EXECUTION_TIMEOUT_SECS: u64 = 60;

// Risk monitor defaults (NP3)
const DEFAULT_LEVERAGE: Decimal = dec!(3.0);

// --- Error Types ---

/// Typed errors for PortfolioManager operations
#[derive(Error, Debug)]
pub enum PortfolioError {
    #[error("Configuration error: {0}")]
    Config(String),
    #[error("WebSocket permanent failure after {0} attempts")]
    WebSocketFailure(u32),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Infrastructure error: {0}")]
    Infrastructure(String),
}

// --- Strategy Actor Wrapper ---

/// Wrapper for DualLegStrategy that ensures pair_id is always returned,
/// even if the strategy panics. This enables the supervisor to restart crashed strategies.
struct StrategyActor {
    pair_id: String,
    strategy: DualLegStrategy,
}

impl StrategyActor {
    fn new(pair_id: String, strategy: DualLegStrategy) -> Self {
        Self { pair_id, strategy }
    }

    /// Runs the strategy and always returns the pair_id, even on panic.
    /// This method uses AssertUnwindSafe to catch panics and ensure clean recovery.
    async fn run_with_id(
        mut self,
        leg1_rx: mpsc::Receiver<Arc<MarketData>>,
        leg2_rx: mpsc::Receiver<Arc<MarketData>>,
    ) -> String {
        // Catch panics to ensure pair_id is always returned
        let result = std::panic::AssertUnwindSafe(async {
            self.strategy.run(leg1_rx, leg2_rx).await;
        });

        match futures_util::future::FutureExt::catch_unwind(result).await {
            Ok(_) => {
                debug!(pair_id = %self.pair_id, "Strategy exited cleanly");
            }
            Err(panic_info) => {
                error!(
                    pair_id = %self.pair_id,
                    "Strategy panicked: {:?}",
                    panic_info
                );
            }
        }

        self.pair_id // Always return pair_id for supervisor restart logic
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioPairConfig {
    #[serde(flatten)]
    pub dual_leg_config: DualLegConfig,
    pub window_size: usize,
    pub entry_z_score: f64,
    pub exit_z_score: f64,
}

/// Portfolio-level PnL tracking for risk monitoring
/// OW5: Uses DashMap for lock-free concurrent updates
pub struct PortfolioPnL {
    total_pnl: Arc<Mutex<Decimal>>,
    // OW5: DashMap for lock-free concurrent access
    strategy_pnls: Arc<DashMap<String, Decimal>>,
}

impl Default for PortfolioPnL {
    fn default() -> Self {
        Self::new()
    }
}

impl PortfolioPnL {
    pub fn new() -> Self {
        Self {
            total_pnl: Arc::new(Mutex::new(Decimal::ZERO)),
            // OW5: Use DashMap instead of Mutex<HashMap>
            strategy_pnls: Arc::new(DashMap::new()),
        }
    }

    pub async fn update_pnl(&self, strategy_id: &str, pnl_delta: Decimal) {
        let mut total = self.total_pnl.lock().await;
        *total += pnl_delta;

        // OW5: Lock-free concurrent update with DashMap
        self.strategy_pnls
            .entry(strategy_id.to_string())
            .and_modify(|pnl| *pnl += pnl_delta)
            .or_insert(pnl_delta);
    }

    pub async fn get_total_pnl(&self) -> Decimal {
        *self.total_pnl.lock().await
    }

    pub async fn get_strategy_pnl(&self, strategy_id: &str) -> Decimal {
        // OW5: Lock-free concurrent read with DashMap
        self.strategy_pnls
            .get(strategy_id)
            .map(|entry| *entry.value())
            .unwrap_or(Decimal::ZERO)
    }
}

impl PortfolioPairConfig {
    pub fn pair_id(&self) -> String {
        format!(
            "{}-{}",
            self.dual_leg_config.spot_symbol, self.dual_leg_config.future_symbol
        )
    }

    /// AS2: Validates the configuration at startup to catch errors early.
    /// Returns Ok(()) if valid, or an error message describing the issue.
    pub fn validate(&self) -> Result<(), String> {
        // Validate Z-score thresholds
        if self.entry_z_score <= 0.0 {
            return Err(format!(
                "entry_z_score must be positive, got: {}",
                self.entry_z_score
            ));
        }
        if self.exit_z_score < 0.0 {
            return Err(format!(
                "exit_z_score cannot be negative, got: {}",
                self.exit_z_score
            ));
        }
        if self.exit_z_score >= self.entry_z_score {
            return Err(format!(
                "exit_z_score ({}) must be less than entry_z_score ({})",
                self.exit_z_score, self.entry_z_score
            ));
        }

        // Validate window size
        if self.window_size == 0 {
            return Err("window_size must be greater than 0".to_string());
        }
        if self.window_size < 10 {
            warn!("window_size {} is very small, consider using at least 10 for statistical significance", self.window_size);
        }

        // Validate order size
        if self.dual_leg_config.order_size <= Decimal::ZERO {
            return Err(format!(
                "order_size must be positive, got: {}",
                self.dual_leg_config.order_size
            ));
        }

        // Validate symbols are non-empty
        if self.dual_leg_config.spot_symbol.is_empty() {
            return Err("spot_symbol cannot be empty".to_string());
        }
        if self.dual_leg_config.future_symbol.is_empty() {
            return Err("future_symbol cannot be empty".to_string());
        }
        if self.dual_leg_config.spot_symbol == self.dual_leg_config.future_symbol {
            return Err(format!(
                "spot_symbol and future_symbol cannot be the same: {}",
                self.dual_leg_config.spot_symbol
            ));
        }

        // Validate timeout values
        if self.dual_leg_config.max_tick_age_ms <= 0 {
            return Err(format!(
                "max_tick_age_ms must be positive, got: {}",
                self.dual_leg_config.max_tick_age_ms
            ));
        }
        if self.dual_leg_config.execution_timeout_ms <= 0 {
            return Err(format!(
                "execution_timeout_ms must be positive, got: {}",
                self.dual_leg_config.execution_timeout_ms
            ));
        }

        Ok(())
    }
}

pub struct PortfolioManager {
    config_path: String,
    env: AppEnv,
    ws_task_handle: Option<tokio::task::JoinHandle<()>>,
    ws_cancel_token: CancellationToken,
}

impl PortfolioManager {
    pub fn new(config_path: &str, env: AppEnv) -> Self {
        Self {
            config_path: config_path.to_string(),
            env,
            ws_task_handle: None,
            ws_cancel_token: CancellationToken::new(),
        }
    }

    #[instrument(skip(self), fields(config_path = %self.config_path))]
    pub async fn run(&mut self) -> Result<(), PortfolioError> {
        info!(
            "Starting Portfolio Manager with config: {}",
            self.config_path
        );

        // 1. Load Configuration
        let file = File::open(&self.config_path)?;
        let reader = BufReader::new(file);
        let config_list: Vec<PortfolioPairConfig> = serde_json::from_reader(reader)?;

        if config_list.is_empty() {
            return Err(PortfolioError::Config(
                "No pairs found in configuration.".to_string(),
            ));
        }

        // AS2: Validate all configurations at startup
        for (idx, config) in config_list.iter().enumerate() {
            if let Err(e) = config.validate() {
                return Err(PortfolioError::Config(format!(
                    "Invalid configuration for pair #{} ({}): {}",
                    idx + 1,
                    config.pair_id(),
                    e
                )));
            }
        }

        info!(
            "Loaded and validated {} pairs from config.",
            config_list.len()
        );

        // Store configs for restart capability
        let config_map: HashMap<String, Arc<PortfolioPairConfig>> = config_list
            .into_iter()
            .map(|cfg| {
                let pair_id = cfg.pair_id();
                (pair_id, Arc::new(cfg))
            })
            .collect();

        // 2. Initialize Shared Resources
        let client = Arc::new(
            CoinbaseClient::new(self.env)
                .map_err(|e| PortfolioError::Infrastructure(e.to_string()))?,
        );

        // 3. Spawn Strategy Actors
        // We use JoinSet to monitor actors. They return their pair_id on exit.
        let mut join_set: JoinSet<String> = JoinSet::new();

        // Symbol Map: Symbol -> List of (Sender, PairID)
        // We need PairID to remove stale senders on restart.
        let mut symbol_map: SymbolRouterMap = HashMap::new();

        // OW1: Use HashSet for O(n) instead of O(n²) duplicate checking
        let mut symbols_set: HashSet<String> = HashSet::new();
        for config in config_map.values() {
            symbols_set.insert(config.dual_leg_config.spot_symbol.clone());
            symbols_set.insert(config.dual_leg_config.future_symbol.clone());
        }
        let symbols_to_subscribe: Vec<String> = symbols_set.into_iter().collect();

        // Spawn all strategies
        for config in config_map.values() {
            Self::spawn_strategy(
                &mut join_set,
                &mut symbol_map,
                client.clone(),
                config.clone(),
            )
            .await;
        }

        // 4. WebSocket Connection & Demultiplexer
        let ws_client =
            CoinbaseWebsocket::new().map_err(|e| PortfolioError::Infrastructure(e.to_string()))?;
        let (ws_tx, mut ws_rx) = mpsc::channel(WS_CHANNEL_BUFFER);

        let symbols_clone = symbols_to_subscribe.clone();
        let cancel_token = self.ws_cancel_token.clone();
        let ws_handle = tokio::spawn(async move {
            tokio::select! {
                _ = cancel_token.cancelled() => {
                    debug!("WebSocket task received cancellation signal, shutting down gracefully");
                }
                result = ws_client.connect_and_subscribe(symbols_clone, ws_tx) => {
                    if let Err(e) = result {
                        error!("WebSocket Error: {}", e);
                    }
                }
            }
        });
        self.ws_task_handle = Some(ws_handle);

        info!(
            "Portfolio Manager running. Monitoring {} strategies.",
            config_map.len()
        );

        // Supervisor Loop
        loop {
            tokio::select! {
                // Handle WebSocket Ticks
                res = ws_rx.recv() => {
                    match res {
                        Some(data) => {
                            let arc_data = Arc::new(data);
                            // CF5: Clean up closed senders before routing
                            if let Some(senders) = symbol_map.get_mut(&arc_data.symbol) {
                                senders.retain(|(sender, _)| !sender.is_closed());

                                for (sender, pair_id) in senders {
                                    // FIX: Use try_send to prevent latency cascading
                                    // If one strategy is slow, don't block ticks for all others
                                    match sender.try_send(arc_data.clone()) {
                                        Ok(_) => {},
                                        Err(mpsc::error::TrySendError::Full(_)) => {
                                            // Channel full - strategy is slow, drop tick for this strategy only
                                            debug!(pair_id = %pair_id, symbol = %arc_data.symbol, "Dropping tick: channel full (strategy backpressure)");
                                        },
                                        Err(mpsc::error::TrySendError::Closed(_)) => {
                                            // Channel closed - will be cleaned up next iteration
                                            debug!(pair_id = %pair_id, "Channel closed, will cleanup");
                                        }
                                    }
                                }
                            }
                        }
                        None => {
                            // CF4: Implement WebSocket reconnection instead of fatal error
                            error!("WebSocket channel closed. Attempting reconnection...");

                            let mut reconnect_attempts = 0;
                            let mut backoff = Duration::from_secs(1);

                            loop {
                                reconnect_attempts += 1;
                                if reconnect_attempts > MAX_RECONNECT_ATTEMPTS {
                                    error!("FATAL: WebSocket reconnection failed after {} attempts", MAX_RECONNECT_ATTEMPTS);
                                    return Err(PortfolioError::WebSocketFailure(MAX_RECONNECT_ATTEMPTS));
                                }

                                warn!("Reconnection attempt {} of {}", reconnect_attempts, MAX_RECONNECT_ATTEMPTS);
                                tokio::time::sleep(backoff).await;

                                match CoinbaseWebsocket::new() {
                                    Ok(new_ws_client) => {
                                        let (new_ws_tx, new_ws_rx) = mpsc::channel(WS_CHANNEL_BUFFER);

                                        // CF1 FIX: Use CancellationToken for graceful shutdown
                                        if let Some(old_handle) = self.ws_task_handle.take() {
                                            // Signal old task to stop
                                            self.ws_cancel_token.cancel();
                                            debug!("Sent cancellation signal to old WebSocket task");

                                            // Wait for clean shutdown (with timeout)
                                            let _ = tokio::time::timeout(Duration::from_secs(2), old_handle).await;

                                            // Create new token for new task
                                            self.ws_cancel_token = CancellationToken::new();
                                        }

                                        let cancel_token = self.ws_cancel_token.clone();
                                        let symbols_clone = symbols_to_subscribe.clone();
                                        let new_handle = tokio::spawn(async move {
                                            tokio::select! {
                                                _ = cancel_token.cancelled() => {
                                                    debug!("Reconnected WebSocket task received cancellation during startup");
                                                }
                                                result = new_ws_client.connect_and_subscribe(symbols_clone, new_ws_tx) => {
                                                    if let Err(e) = result {
                                                        error!("WebSocket reconnection error: {}", e);
                                                    }
                                                }
                                            }
                                        });
                                        self.ws_task_handle = Some(new_handle);

                                        ws_rx = new_ws_rx;
                                        info!("WebSocket reconnected successfully!");
                                        break;
                                    }
                                    Err(e) => {
                                        error!("Reconnection failed: {}", e);
                                        backoff = std::cmp::min(backoff * 2, Duration::from_secs(MAX_RECONNECT_BACKOFF_SECS));
                                    }
                                }
                            }
                        }
                    }
                }

                // Handle Actor Termination (Supervisor Pattern)
                Some(res) = join_set.join_next() => {
                    match res {
                        Ok(pair_id) => {
                            error!(pair_id = %pair_id, "Strategy exited unexpectedly, initiating restart");

                            // 1. Clean up old routing entries for this pair
                            for (_, list) in symbol_map.iter_mut() {
                                list.retain(|(_, pid)| pid != &pair_id);
                            }

                            // 2. Retrieve Config and Respawn
                            if let Some(config) = config_map.get(&pair_id) {
                                // CF2: Add backoff to prevent tight crash loops (removed duplicate)
                                tokio::time::sleep(Duration::from_secs(1)).await;

                                Self::spawn_strategy(
                                    &mut join_set,
                                    &mut symbol_map,
                                    client.clone(),
                                    config.clone()
                                ).await;

                                info!(pair_id = %pair_id, "Strategy restarted successfully");
                            } else {
                                error!(pair_id = %pair_id, "CRITICAL: Could not find config for exited pair");
                            }
                        }
                        Err(e) => {
                            // CF2 FIX: With StrategyActor wrapper, panics should never reach here
                            // because the wrapper catches them and returns pair_id.
                            // This branch now only handles task cancellation or abort.
                            let error_type = if e.is_panic() {
                                // Should not happen with StrategyActor, but log if it does
                                error!("UNEXPECTED: Panic escaped StrategyActor wrapper! This is a bug.");
                                "panicked"
                            } else if e.is_cancelled() {
                                "cancelled"
                            } else {
                                "failed"
                            };

                            error!(
                                error_type = error_type,
                                error = %e,
                                "Strategy task {} (likely due to explicit cancellation).",
                                error_type
                            );

                            // Task cancellation is expected during shutdown, so this is less critical
                            warn!("Strategy terminated without returning pair_id. If this happens during normal operation, investigate.");
                        }
                    }
                }
            }
        }
    }

    #[instrument(skip_all, fields(pair_id = %config.pair_id()))]
    async fn spawn_strategy(
        join_set: &mut JoinSet<String>,
        symbol_map: &mut SymbolRouterMap,
        client: Arc<CoinbaseClient>,
        config: Arc<PortfolioPairConfig>,
    ) {
        let pair_id = config.pair_id();
        info!("Initializing strategy for {}", pair_id);

        // Create Channels
        let (leg1_tx, leg1_rx) = mpsc::channel(STRATEGY_CHANNEL_BUFFER);
        let (leg2_tx, leg2_rx) = mpsc::channel(STRATEGY_CHANNEL_BUFFER);

        // DRY: Use helper function for symbol registration
        Self::register_symbol(
            symbol_map,
            config.dual_leg_config.spot_symbol.clone(),
            leg1_tx,
            pair_id.clone(),
        );
        Self::register_symbol(
            symbol_map,
            config.dual_leg_config.future_symbol.clone(),
            leg2_tx,
            pair_id.clone(),
        );

        // Setup Strategy Components
        let (recovery_tx, recovery_rx) = mpsc::channel(RECOVERY_CHANNEL_BUFFER);
        let (feedback_tx, feedback_rx) = mpsc::channel(FEEDBACK_CHANNEL_BUFFER);

        let recovery_worker = RecoveryWorker::new(client.clone(), recovery_rx, feedback_tx);
        tokio::spawn(async move {
            recovery_worker.run().await;
        });

        let execution_engine = ExecutionEngine::new(
            client.clone(),
            recovery_tx,
            EXECUTION_MAX_RETRIES,
            EXECUTION_TIMEOUT_SECS,
        );

        // Strategy Logic (Pairs Trading for Portfolio)
        // Use dynamic parameters from config
        let manager = Box::new(PairsManager::new(
            config.window_size,
            config.entry_z_score,
            config.exit_z_score,
        ));
        let monitor = RiskMonitor::new(
            DEFAULT_LEVERAGE,
            InstrumentType::Linear,
            HedgeMode::DollarNeutral,
        );

        let strategy = DualLegStrategy::new(
            manager,
            monitor,
            execution_engine,
            config.dual_leg_config.clone(),
            feedback_rx,
            Box::new(SystemClock),
        );

        // CF2 FIX: Wrap strategy in StrategyActor to ensure pair_id is always returned
        let actor = StrategyActor::new(pair_id.clone(), strategy);
        join_set.spawn(async move { actor.run_with_id(leg1_rx, leg2_rx).await });
    }

    /// DRY helper: Register a symbol->sender mapping for tick distribution
    fn register_symbol(
        map: &mut SymbolRouterMap,
        symbol: String,
        sender: mpsc::Sender<Arc<MarketData>>,
        pair_id: String,
    ) {
        map.entry(symbol).or_default().push((sender, pair_id));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::dual_leg_trading::{DualLegConfig, TransactionCostModel};

    /// Creates a valid test configuration
    fn valid_config() -> PortfolioPairConfig {
        PortfolioPairConfig {
            dual_leg_config: DualLegConfig {
                spot_symbol: "BTC-USD".to_string(),
                future_symbol: "BTC-USDT".to_string(),
                order_size: dec!(0.001),
                max_tick_age_ms: 1000,
                execution_timeout_ms: 5000,
                min_profit_threshold: dec!(0.001),
                stop_loss_threshold: dec!(-0.02),
                fee_tier: TransactionCostModel::default(),
                throttle_interval_secs: 5,
            },
            window_size: 30,
            entry_z_score: 2.0,
            exit_z_score: 0.5,
        }
    }

    #[test]
    fn test_validate_valid_config() {
        let config = valid_config();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_entry_z_score_zero() {
        let mut config = valid_config();
        config.entry_z_score = 0.0;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("entry_z_score must be positive"));
    }

    #[test]
    fn test_validate_entry_z_score_negative() {
        let mut config = valid_config();
        config.entry_z_score = -1.0;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("entry_z_score must be positive"));
    }

    #[test]
    fn test_validate_exit_z_score_negative() {
        let mut config = valid_config();
        config.exit_z_score = -0.5;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("exit_z_score cannot be negative"));
    }

    #[test]
    fn test_validate_exit_greater_than_entry() {
        let mut config = valid_config();
        config.entry_z_score = 1.0;
        config.exit_z_score = 2.0;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("must be less than entry_z_score"));
    }

    #[test]
    fn test_validate_exit_equal_to_entry() {
        let mut config = valid_config();
        config.entry_z_score = 2.0;
        config.exit_z_score = 2.0;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("must be less than entry_z_score"));
    }

    #[test]
    fn test_validate_window_size_zero() {
        let mut config = valid_config();
        config.window_size = 0;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("window_size must be greater than 0"));
    }

    #[test]
    fn test_validate_order_size_zero() {
        let mut config = valid_config();
        config.dual_leg_config.order_size = Decimal::ZERO;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("order_size must be positive"));
    }

    #[test]
    fn test_validate_order_size_negative() {
        let mut config = valid_config();
        config.dual_leg_config.order_size = dec!(-0.001);
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("order_size must be positive"));
    }

    #[test]
    fn test_validate_empty_spot_symbol() {
        let mut config = valid_config();
        config.dual_leg_config.spot_symbol = "".to_string();
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("spot_symbol cannot be empty"));
    }

    #[test]
    fn test_validate_empty_future_symbol() {
        let mut config = valid_config();
        config.dual_leg_config.future_symbol = "".to_string();
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("future_symbol cannot be empty"));
    }

    #[test]
    fn test_validate_same_symbols() {
        let mut config = valid_config();
        config.dual_leg_config.spot_symbol = "BTC-USD".to_string();
        config.dual_leg_config.future_symbol = "BTC-USD".to_string();
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("cannot be the same"));
    }

    #[test]
    fn test_validate_max_tick_age_zero() {
        let mut config = valid_config();
        config.dual_leg_config.max_tick_age_ms = 0;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("max_tick_age_ms must be positive"));
    }

    #[test]
    fn test_validate_max_tick_age_negative() {
        let mut config = valid_config();
        config.dual_leg_config.max_tick_age_ms = -100;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("max_tick_age_ms must be positive"));
    }

    #[test]
    fn test_validate_execution_timeout_zero() {
        let mut config = valid_config();
        config.dual_leg_config.execution_timeout_ms = 0;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("execution_timeout_ms must be positive"));
    }

    #[test]
    fn test_validate_execution_timeout_negative() {
        let mut config = valid_config();
        config.dual_leg_config.execution_timeout_ms = -500;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("execution_timeout_ms must be positive"));
    }

    #[test]
    fn test_pair_id_format() {
        let config = valid_config();
        assert_eq!(config.pair_id(), "BTC-USD-BTC-USDT");
    }
}
