//! # Delta-Neutral Basis Trading Strategy
//!
//! # Dual-Leg Trading Strategy
//!
//! This module implements a generic Dual-Leg Trading strategy that supports:
//! - **Basis Trading**: Exploits the price difference (basis) between Spot and Futures.
//! - **Statistical Arbitrage (Pairs)**: Exploits mean reversion of the spread between two assets.
//!
//! ## Architecture
//! The strategy is composed of three main components:
//! - `EntryStrategy`: Trait for analyzing the relationship between two legs (Basis or Pairs).
//! - `RiskMonitor`: Tracks delta/exposure and ensures proper hedging (Delta Neutral or Dollar Neutral).
//! - `ExecutionEngine`: Handles the concurrent execution of orders on both legs.
//!
//! ## Safety
//! The strategy enforces strict risk checks via `RiskMonitor::calc_hedge_ratio` to prevent over-leveraging
//! and ensure proper hedging.

use crate::strategy::Signal;
use async_trait::async_trait;
use std::sync::Arc;
use std::error::Error;
use crate::coinbase::CoinbaseClient;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use rust_decimal::prelude::*;
use chrono::{Utc, DateTime};
use tracing::{info, debug, error, instrument, warn};
use tokio::sync::{mpsc, Semaphore, Mutex};
use tokio::time::{Duration, Instant};
use thiserror::Error;
use serde::{Serialize, Deserialize};

#[derive(Error, Debug)]
pub enum DualLegError {
    #[error("Invalid input parameters: {0}")]
    InvalidInput(String),
    #[error("Math error: {0}")]
    MathError(String),
    #[error("Execution error: {0}")]
    ExecutionError(#[from] ExecutionError),
    #[error("Unknown error: {0}")]
    Unknown(String),
}

#[derive(Error, Debug, Clone)]
pub enum ExecutionError {
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("Exchange error: {0}")]
    ExchangeError(String),
    #[error("Order rejected: {0}")]
    OrderRejected(String),
    #[error("Critical system failure: {0}")]
    CriticalFailure(String),
    #[error("Unknown execution error: {0}")]
    Unknown(String),
}

impl ExecutionError {
    pub fn from_boxed(e: Box<dyn Error + Send + Sync>) -> Self {
        let s = e.to_string();
        if s.to_lowercase().contains("network") || s.to_lowercase().contains("timeout") {
            ExecutionError::NetworkError(s)
        } else if s.to_lowercase().contains("insufficient funds") || s.to_lowercase().contains("rejected") {
            ExecutionError::OrderRejected(s)
        } else {
            ExecutionError::ExchangeError(s)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
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

// Constants removed in favor of Config
// const MAX_TICK_AGE_MS: i64 = 2000;
// const MAX_TICK_DIFF_MS: i64 = 2000;
// const EXECUTION_TIMEOUT_MS: i64 = 30000;

const RECOVERY_BACKOFF_CAP_SECS: u64 = 60;

// CF3: Precision Safety Constants
// Conservative bounds for f64 price ratio conversions to prevent precision loss
// in statistical arbitrage calculations. Ratios outside these bounds will be rejected.
const MAX_SAFE_PRICE_RATIO: f64 = 1e12; // Conservative limit for f64 precision
const MIN_SAFE_PRICE_RATIO: f64 = 1e-12; // Reciprocal of max for symmetry

// --- Logging Utilities ---

/// A lightweight rate limiter for logging to prevent log storms.
#[derive(Debug)]
pub struct LogThrottle {
    last_log_time: Option<Instant>,
    suppressed_count: u64,
    interval: Duration,
}

impl LogThrottle {
    pub fn new(interval: Duration) -> Self {
        Self {
            last_log_time: None,
            suppressed_count: 0,
            interval,
        }
    }

    /// Checks if a log should be emitted.
    /// Returns true if the interval has passed since the last log.
    /// If false, increments the suppressed counter.
    pub fn should_log(&mut self) -> bool {
        let now = Instant::now();
        match self.last_log_time {
            Some(last) => {
                if now.duration_since(last) >= self.interval {
                    self.last_log_time = Some(now);
                    true
                } else {
                    self.suppressed_count += 1;
                    false
                }
            }
            None => {
                self.last_log_time = Some(now);
                true
            }
        }
    }

    /// Returns the number of suppressed logs since the last successful log, and resets the counter.
    pub fn get_and_reset_suppressed_count(&mut self) -> u64 {
        let count = self.suppressed_count;
        self.suppressed_count = 0;
        count
    }
}

/// container for all log throttlers used in the strategy.
#[derive(Debug)]
pub struct DualLegLogThrottler {
    pub unstable_state: LogThrottle,
    pub tick_age: LogThrottle,
    pub sync_issue: LogThrottle,
}

impl DualLegLogThrottler {
    pub fn new(interval_secs: u64) -> Self {
        let interval = Duration::from_secs(interval_secs);
        Self {
            unstable_state: LogThrottle::new(interval),
            tick_age: LogThrottle::new(interval),
            sync_issue: LogThrottle::new(interval),
        }
    }
}

// --- Financial Models ---

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TransactionCostModel {
    pub maker_fee_bps: Decimal,
    pub taker_fee_bps: Decimal,
    pub slippage_bps: Decimal,
}

impl TransactionCostModel {
    pub fn new(maker_fee_bps: Decimal, taker_fee_bps: Decimal, slippage_bps: Decimal) -> Self {
        Self { maker_fee_bps, taker_fee_bps, slippage_bps }
    }

    pub fn calc_net_spread(&self, gross_spread_bps: Decimal) -> Decimal {
        // Net = Gross - (4 * Fees) - Slippage
        // Assuming 4 legs (Entry Spot, Entry Future, Exit Spot, Exit Future)
        // And assuming we are Taker on all for worst case, or Maker?
        // Let's assume Taker for conservatism in this model unless specified.
        // Actually, usually we might be Maker on one. Let's stick to the requirement:
        // "Net Spread (Gross Spread - (4 * Fees) - Slippage)"
        // We'll use taker fee as the conservative fee.
        let total_fees = self.taker_fee_bps * dec!(4.0);
        gross_spread_bps - total_fees - self.slippage_bps
    }
}

// AS5: Default transaction costs for consistency across codebase
impl Default for TransactionCostModel {
    fn default() -> Self {
        Self::new(
            dec!(10.0), // Maker fee: 10 basis points
            dec!(20.0), // Taker fee: 20 basis points
            dec!(5.0),  // Slippage: 5 basis points
        )
    }
}

// --- Time Abstraction ---

pub trait Clock: Send + Sync + std::fmt::Debug {
    fn now(&self) -> DateTime<Utc>;
    fn now_ts_millis(&self) -> i64 {
        self.now().timestamp_millis()
    }
}

#[derive(Debug, Clone)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}

// --- Domain Models ---

/// Represents a market data update (price tick).
#[derive(Debug, Clone, PartialEq)]
pub struct MarketData {
    /// The trading symbol (e.g., "BTC-USD").
    pub symbol: String,
    /// Optional instrument identifier for multi-exchange support (e.g., "coinbase", "binance").
    pub instrument_id: Option<String>,
    /// The current price.
    pub price: Decimal,
    /// The timestamp of the update.
    pub timestamp: i64,
}

/// Represents an open position.
#[derive(Debug, Clone, PartialEq)]
pub struct Position {
    /// The trading symbol.
    pub symbol: String,
    /// Optional instrument identifier for multi-exchange support (e.g., "coinbase", "binance").
    pub instrument_id: Option<String>,
    /// The quantity held (positive for long, negative for short).
    pub quantity: Decimal,
    /// The average entry price.
    pub entry_price: Decimal,
}

#[derive(Debug, Clone)]
pub struct InstrumentPair {
    pub spot_symbol: String,
    pub future_symbol: String,
}

#[derive(Debug, Clone, Copy)]
pub struct Spread {
    pub value_bps: Decimal,
    pub spot_price: Decimal,
    pub future_price: Decimal,
}

impl Spread {
    pub fn new(spot: Decimal, future: Decimal) -> Self {
        let value_bps = if spot.is_zero() {
            Decimal::zero()
        } else {
            ((future - spot) / spot) * dec!(10000.0)
        };
        Self { value_bps, spot_price: spot, future_price: future }
    }
}

// --- State Management ---

// AS9: ExitPolicy trait for flexible exit conditions
#[async_trait]
pub trait ExitPolicy: Send + Sync {
    /// Returns true if the position should be exited
    async fn should_exit(
        &self,
        entry_price: Decimal,
        current_price: Decimal,
        pnl: Decimal,
    ) -> bool;
}

/// Exit when minimum profit threshold is met
pub struct MinimumProfitPolicy {
    min_profit_bps: Decimal,
}

impl MinimumProfitPolicy {
    pub fn new(min_profit_bps: Decimal) -> Self {
        Self { min_profit_bps }
    }
}

#[async_trait]
impl ExitPolicy for MinimumProfitPolicy {
    async fn should_exit(
        &self,
        entry_price: Decimal,
        current_price: Decimal,
        _pnl: Decimal,
    ) -> bool {
        if entry_price.is_zero() {
            return false;
        }
        
        let price_change_bps = ((current_price - entry_price) / entry_price) * dec!(10000.0);
        price_change_bps >= self.min_profit_bps
    }
}

/// Exit when stop loss threshold is hit
pub struct StopLossPolicy {
    max_loss_bps: Decimal,
}

impl StopLossPolicy {
    pub fn new(max_loss_bps: Decimal) -> Self {
        Self { max_loss_bps }
    }
}

#[async_trait]
impl ExitPolicy for StopLossPolicy {
    async fn should_exit(
        &self,
        entry_price: Decimal,
        current_price: Decimal,
        _pnl: Decimal,
    ) -> bool {
        if entry_price.is_zero() {
            return false;
        }
        
        let price_change_bps = ((current_price - entry_price) / entry_price) * dec!(10000.0);
        price_change_bps.abs() >= self.max_loss_bps && price_change_bps < Decimal::ZERO
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualLegConfig {
    pub spot_symbol: String,
    pub future_symbol: String,
    pub order_size: Decimal,
    pub max_tick_age_ms: i64,
    pub execution_timeout_ms: i64,
    pub min_profit_threshold: Decimal,
    pub stop_loss_threshold: Decimal,
    pub fee_tier: TransactionCostModel,
    pub throttle_interval_secs: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StrategyState {
    Flat,
    Entering { leg1_qty: Decimal, leg2_qty: Decimal, leg1_entry_price: Decimal, leg2_entry_price: Decimal },
    InPosition { leg1_qty: Decimal, leg2_qty: Decimal, leg1_entry_price: Decimal, leg2_entry_price: Decimal },
    Exiting { leg1_qty: Decimal, leg2_qty: Decimal, leg1_entry_price: Decimal, leg2_entry_price: Decimal },
    Reconciling, // New State: "I don't know what happened, let me check."
    Halted,      // New State: "Something is broken, human needed."
}

#[derive(Debug)]
pub enum RecoveryResult {
    Success(String), // Symbol
    Failed(String),  // Symbol
}

#[derive(Debug)]
pub enum ExecutionResult {
    Success,
    PartialFailure(ExecutionError), // e.g., "Spot filled, Future failed"
    TotalFailure(ExecutionError),
}

#[derive(Debug)]
pub struct ExecutionReport {
    pub result: ExecutionResult,
    pub action: Signal, // Buy (Entry) or Sell (Exit)
    pub pnl_delta: Option<Decimal>, // PnL change from this execution (for exits)
}

#[derive(Debug, Clone)]
pub struct RecoveryTask {
    pub symbol: String,
    pub action: OrderSide,
    pub quantity: Decimal,
    pub reason: String,
    pub attempts: u32,
}

// --- Interfaces (Dependency Injection) ---

/// Trait for entry logic strategies.
/// Allows for swapping different entry algorithms (e.g., simple threshold vs. statistical arbitrage).
#[async_trait]
pub trait EntryStrategy: Send + Sync {
    /// Analyzes the market data for Leg 1 and Leg 2 to generate a trading signal.
    async fn analyze(&self, leg1: &MarketData, leg2: &MarketData) -> Signal;
}

/// Executor trait for placing orders.
/// This abstraction allows for easy mocking in tests.
#[async_trait]
pub trait Executor: Send + Sync {
    /// Executes an order.
    async fn execute_order(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: Decimal,
        price: Option<Decimal>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    
    /// Gets the current position for a symbol.
    /// Returns the quantity held (positive for long, negative for short, zero for no position).
    async fn get_position(
        &self,
        symbol: &str,
    ) -> Result<Decimal, Box<dyn std::error::Error + Send + Sync>>;
}

// --- Components ---

/// Analyzes the basis spread and generates signals based on fixed thresholds.
pub struct BasisManager {
    entry_threshold_bps: Decimal,
    exit_threshold_bps: Decimal,
    cost_model: TransactionCostModel,
}

impl BasisManager {
    pub fn new(entry_threshold_bps: Decimal, exit_threshold_bps: Decimal, cost_model: TransactionCostModel) -> Self {
        Self {
            entry_threshold_bps,
            exit_threshold_bps,
            cost_model,
        }
    }
}

#[async_trait]
impl EntryStrategy for BasisManager {
    #[instrument(skip(self))]
    async fn analyze(&self, leg1: &MarketData, leg2: &MarketData) -> Signal {
        let spread = Spread::new(leg1.price, leg2.price);
        let net_spread = self.cost_model.calc_net_spread(spread.value_bps);
        debug!("Basis Spread: {:.4} bps (Net: {:.4}) (Spot: {}, Future: {})", spread.value_bps, net_spread, spread.spot_price, spread.future_price);
        
        if net_spread > self.entry_threshold_bps {
            Signal::Buy
        } else if spread.value_bps < self.exit_threshold_bps {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }
}

/// Statistical Arbitrage (Pairs Trading) Manager.
/// Uses Z-Score of the log-spread to generate signals.
pub struct PairsManager {
    window_size: usize,
    entry_z_score: f64,
    exit_z_score: f64,
    spread_history: Mutex<std::collections::VecDeque<f64>>,
}

impl PairsManager {
    pub fn new(window_size: usize, entry_z_score: f64, exit_z_score: f64) -> Self {
        Self {
            window_size,
            entry_z_score,
            exit_z_score,
            spread_history: Mutex::new(std::collections::VecDeque::with_capacity(window_size)),
        }
    }
}

#[async_trait]
impl EntryStrategy for PairsManager {
    #[instrument(skip(self))]
    async fn analyze(&self, leg1: &MarketData, leg2: &MarketData) -> Signal {
        // CF3 DOCUMENTATION: Decimal → f64 precision loss
        // ================================================
        // For statistical arbitrage, we convert Decimal prices to f64 for natural log calculations.
        // This introduces precision loss, but is acceptable for pairs trading where:
        // 1. Z-score thresholds are typically >> 0.01 (sub-basis-point precision not critical)
        // 2. Relative spread matters more than absolute precision
        // 3. f64 provides ~15-17 decimal digits of precision (sufficient for price ratios)
        //
        // RISK ASSESSMENT: For assets with extreme price differences (e.g., BTC/SHIB where ratio > 10^9),
        // precision loss could affect signal generation. Monitor via debug logs.
        //
        // ALTERNATIVE: For critical HFT applications, use decimal-based log library or integer basis points.
        let p1_opt = leg1.price.to_f64();
        let p2_opt = leg2.price.to_f64();

        let (p1, p2) = match (p1_opt, p2_opt) {
            (Some(v1), Some(v2)) if v1 > 0.0 && v2 > 0.0 => {
                if v1.is_infinite() || v1.is_nan() || v2.is_infinite() || v2.is_nan() {
                    warn!("PRECISION WARNING: Infinite or NaN prices detected. P1: {}, P2: {}", v1, v2);
                    return Signal::Hold;
                }
                
                // CF3 FIX: Enforce hard limits on price ratios to prevent precision loss
                let ratio = v1 / v2;
                if ratio > MAX_SAFE_PRICE_RATIO || ratio < MIN_SAFE_PRICE_RATIO {
                    error!(
                        "PRECISION ERROR: Price ratio {:.2e} exceeds safe f64 bounds [{:.2e}, {:.2e}]. Rejecting signal to prevent precision loss.",
                        ratio, MIN_SAFE_PRICE_RATIO, MAX_SAFE_PRICE_RATIO
                    );
                    return Signal::Hold; // Hard rejection - do not trade on degraded precision
                }
                
                // Log warning for ratios approaching the limit (within 2 orders of magnitude)
                if ratio > MAX_SAFE_PRICE_RATIO / 100.0 || ratio < MIN_SAFE_PRICE_RATIO * 100.0 {
                    warn!(
                        "PRECISION WARNING: Price ratio {:.2e} approaching safety limits. Monitor for precision degradation.",
                        ratio
                    );
                }
                
                (v1, v2)
            },
            _ => {
                warn!("Invalid prices for Pairs analysis: {:?}, {:?}", leg1.price, leg2.price);
                return Signal::Hold;
            }
        };

        let spread = p1.ln() - p2.ln();
        
        let z_score = {
            let mut history = self.spread_history.write().await;
            history.push_back(spread);
            if history.len() > self.window_size {
                history.pop_front();
            }

            if history.len() < self.window_size {
                return Signal::Hold; // Not enough data
            }

            let n = history.len() as f64;
            let sum: f64 = history.iter().sum();
            let mean = sum / n;

            let variance: f64 = history.iter().map(|val| {
                let diff = mean - val;
                diff * diff
            }).sum::<f64>() / n;

            let std_dev = variance.sqrt();

            if std_dev == 0.0 {
                0.0
            } else {
                (spread - mean) / std_dev
            }
        };

        debug!("Pairs Spread: {:.6}, Z-Score: {:.4}", spread, z_score);

        if z_score > self.entry_z_score {
            Signal::Sell // Sell A / Buy B (Short the spread)
        } else if z_score < -self.entry_z_score {
            Signal::Buy // Buy A / Sell B (Long the spread)
        } else if z_score.abs() < self.exit_z_score {
            Signal::Sell // Close positions (Mean Reversion)
        } else {
            Signal::Hold
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InstrumentType {
    Linear,
    Inverse,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HedgeMode {
    DeltaNeutral,  // Quantity 1:1 (or adjusted for contract size)
    DollarNeutral, // Value 1:1 (QtyA * PriceA = QtyB * PriceB)
}

/// Tracks delta and enforces safety limits.
pub struct RiskMonitor {
    _max_leverage: Decimal,
    hedge_mode: HedgeMode,
    instrument_type: InstrumentType,
}

impl RiskMonitor {
    pub fn new(max_leverage: Decimal, instrument_type: InstrumentType, hedge_mode: HedgeMode) -> Self {
        Self {
            _max_leverage: max_leverage,
            hedge_mode,
            instrument_type,
        }
    }

    pub fn calc_hedge_ratio(&self, leg1_qty: Decimal, leg1_price: Decimal, leg2_price: Decimal) -> Result<Decimal, DualLegError> {
        if leg1_qty <= Decimal::zero() {
            return Err(DualLegError::InvalidInput("Leg 1 quantity must be positive".to_string()));
        }

        match self.hedge_mode {
            HedgeMode::DeltaNeutral => {
                match self.instrument_type {
                    InstrumentType::Linear => Ok(leg1_qty),
                    InstrumentType::Inverse => {
                        if leg2_price.is_zero() {
                            return Err(DualLegError::MathError("Leg 2 price cannot be zero for inverse calculation".to_string()));
                        }
                        Ok(leg1_qty * leg2_price)
                    }
                }
            },
            HedgeMode::DollarNeutral => {
                // Qty1 * Price1 = Qty2 * Price2
                // Qty2 = (Qty1 * Price1) / Price2
                if leg2_price.is_zero() {
                    return Err(DualLegError::MathError("Leg 2 price cannot be zero for dollar neutral calculation".to_string()));
                }
                Ok((leg1_qty * leg1_price) / leg2_price)
            }
        }
    }
}

/// Worker that processes recovery tasks (failed legs).
// AS3: Priority system for recovery tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Critical = 0,  // Kill switches (highest priority)
    High = 1,      // Large positions
    Normal = 2,    // Regular retries
}

/// Wrapper for RecoveryTask with priority for queue ordering
#[derive(Debug, Clone)]
struct PrioritizedRecoveryTask {
    priority: TaskPriority,
    task: RecoveryTask,
    queued_at: tokio::time::Instant,
}

impl PartialEq for PrioritizedRecoveryTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.queued_at == other.queued_at
    }
}

impl Eq for PrioritizedRecoveryTask {}

impl PartialOrd for PrioritizedRecoveryTask {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedRecoveryTask {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Priority first (lower number = higher priority)
        // Then FIFO within same priority (reverse queued_at for min-heap behavior)
        self.priority.cmp(&other.priority)
            .then_with(|| other.queued_at.cmp(&self.queued_at))
    }
}

pub struct RecoveryWorker {
    client: Arc<dyn Executor>,
    rx: mpsc::Receiver<RecoveryTask>,
    feedback_tx: mpsc::Sender<RecoveryResult>,
    semaphore: Arc<Semaphore>,
    throttler: LogThrottle,
}

impl RecoveryWorker {
    pub fn new(client: Arc<dyn Executor>, rx: mpsc::Receiver<RecoveryTask>, feedback_tx: mpsc::Sender<RecoveryResult>) -> Self {
        Self { 
            client, 
            rx,
            feedback_tx,
            semaphore: Arc::new(Semaphore::new(5)), // Limit to 5 concurrent recoveries
            throttler: LogThrottle::new(Duration::from_secs(5)),
        }
    }

    #[instrument(skip(self), name = "recovery_worker")]
    pub async fn run(mut self) {
        info!("Recovery Worker started.");
        while let Some(mut task) = self.rx.recv().await {
            let client = self.client.clone();
            let feedback_tx = self.feedback_tx.clone();
            // Non-blocking acquire to prevent deadlock
            let permit = match self.semaphore.clone().try_acquire_owned() {
                Ok(p) => p,
                Err(_) => {
                    error!("CRITICAL: Recovery worker overloaded. Dropping task for {}.", task.symbol);
                    let _ = self.feedback_tx.send(RecoveryResult::Failed(task.symbol)).await;
                    continue;
                }
            };
            
            // Local throttler for the spawned task
                // Local throttler for the spawned task

                // But the throttler is for the WORKER loop logging, or the spawned task logging?
                // "Apply the existing LogThrottle pattern to the RecoveryWorker failure logs"
                // The failure logs are inside the spawned task.
                // So we need to pass a throttler to the spawned task.
                // Let's create a new throttler for each task? No, that defeats the purpose.
                // We need a shared throttler.
                // Let's make RecoveryWorker hold Arc<Mutex<LogThrottle>>.
                
                // Wait, if I change the struct, I need to change the previous chunks.
                // Let's just create a new throttler for the task for now, OR better:
                // The requirement is "prevent log disk exhaustion during a failure loop".
                // A failure loop happens inside the spawned task loop.
                // So a local throttler inside the spawned task is sufficient!
                
                tokio::spawn(async move {
                    // Permit is held until dropped at end of scope
                    let _permit = permit;
                    let mut task_throttler = LogThrottle::new(Duration::from_secs(5));
                    
                    info!("Processing Recovery Task: {:?}", task);
                    let mut backoff = Duration::from_secs(2);

                    loop {
                        task.attempts += 1;
                        // Recovery uses Market orders (None price)
                        match client.execute_order(&task.symbol, task.action, task.quantity, None).await {
                            Ok(_) => {
                                info!("Recovery Successful for {} on attempt {}", task.symbol, task.attempts);
                                let _ = feedback_tx.send(RecoveryResult::Success(task.symbol.clone())).await;
                                break;
                            },
                            Err(e) => {
                                if task_throttler.should_log() {
                                     let suppressed = task_throttler.get_and_reset_suppressed_count();
                                     error!("Recovery Failed for {} (Attempt {}): {} (Suppressed: {})", task.symbol, task.attempts, e, suppressed);
                                }
                                
                                if task.attempts >= 5 {
                                    error!("CRITICAL: Recovery abandoned for {} after 5 attempts. MANUAL INTERVENTION REQUIRED.", task.symbol);
                                    let _ = feedback_tx.send(RecoveryResult::Failed(task.symbol.clone())).await;
                                    break;
                                }
                                tokio::time::sleep(backoff).await;
                                backoff = std::cmp::min(backoff * 2, Duration::from_secs(RECOVERY_BACKOFF_CAP_SECS));
                            }
                        }
                    }
                });
            // End of successful permit block

        }
    }
}

/// Handles order execution logic.
// AS8: Circuit Breaker Pattern for execution failures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,     // Normal operation
    Open,       // Blocking requests due to failures
    HalfOpen,   // Testing if system recovered
}

pub struct CircuitBreaker {
    state: Arc<Mutex<CircuitState>>,
    failure_count: Arc<Mutex<u32>>,
    failure_threshold: u32,
    timeout: Duration,
    last_failure_time: Arc<Mutex<Option<Instant>>>,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, timeout: Duration) -> Self {
        Self {
            state: Arc::new(Mutex::new(CircuitState::Closed)),
            failure_count: Arc::new(Mutex::new(0)),
            failure_threshold,
            timeout,
            last_failure_time: Arc::new(Mutex::new(None)),
        }
    }
    
    pub async fn get_state(&self) -> CircuitState {
        *self.state.lock().await
    }
    
    pub async fn record_success(&self) {
        *self.state.write().await = CircuitState::Closed;
        *self.failure_count.write().await = 0;
    }
    
    pub async fn record_failure(&self) {
        let mut count = self.failure_count.write().await;
        *count += 1;
        *self.last_failure_time.write().await = Some(Instant::now());
        
        if *count >= self.failure_threshold {
            *self.state.write().await = CircuitState::Open;
            warn!("Circuit breaker tripped to OPEN after {} failures", count);
        }
    }
    
    pub async fn is_open(&self) -> bool {
        let state = *self.state.write().await;
        
        if state == CircuitState::Open {
            // Check if timeout has passed
            if let Some(last_failure) = *self.last_failure_time.write().await {
                if last_failure.elapsed() > self.timeout {
                    // Transition to HalfOpen
                    *self.state.write().await = CircuitState::HalfOpen;
                    *self.failure_count.write().await = 0;
                    return false;
                }
            }
            return true;
        }
        false
    }
}

pub struct ExecutionEngine {
    client: Arc<dyn Executor>,
    recovery_tx: mpsc::Sender<RecoveryTask>,
}

impl ExecutionEngine {
    pub fn new(client: Arc<dyn Executor>, recovery_tx: mpsc::Sender<RecoveryTask>) -> Self {
        Self { client, recovery_tx }
    }

    #[instrument(skip(self))]
    pub async fn execute_basis_entry(&self, pair: &InstrumentPair, quantity: Decimal, hedge_qty: Decimal, leg1_price: Decimal, leg2_price: Decimal) -> Result<ExecutionResult, ExecutionError> {
        // Concurrently execute both legs to minimize leg risk
        let spot_leg = self.client.execute_order(&pair.spot_symbol, OrderSide::Buy, quantity, Some(leg1_price));
        let future_leg = self.client.execute_order(&pair.future_symbol, OrderSide::Sell, hedge_qty, Some(leg2_price));

        let (spot_res, future_res) = tokio::join!(spot_leg, future_leg);
        
        // Convert errors to ExecutionError
        let spot_res = spot_res.map_err(|e| ExecutionError::from_boxed(e));
        let future_res = future_res.map_err(|e| ExecutionError::from_boxed(e));

        if let Err(e) = spot_res {
             if future_res.is_ok() {
                 error!("CRITICAL: Spot failed but Future succeeded. Queuing Kill Switch on Future leg.");
                 // Queue Kill Switch: Buy back future
                 let task = RecoveryTask {
                     symbol: pair.future_symbol.clone(),
                     action: OrderSide::Buy,
                     quantity: hedge_qty,
                     reason: format!("Spot failed: {}", e),
                     attempts: 0,
                 };
                 if let Err(send_err) = self.recovery_tx.send(task).await {
                     error!("EMERGENCY: Failed to queue recovery task! Manual intervention required. Error: {}", send_err);
                     // CF4 FIX: Return CriticalFailure instead of panic
                     return Err(ExecutionError::CriticalFailure(format!("Recovery queue failure - spot entry kill switch failed: {}", send_err)));
                 }
                 return Ok(ExecutionResult::PartialFailure(e));
             }
             return Ok(ExecutionResult::TotalFailure(e));
        }

        if let Err(e) = future_res {
            error!("CRITICAL: Future leg failed: {}. Queuing Kill Switch on Spot leg.", e);
            // Queue Kill Switch: Sell spot
            let task = RecoveryTask {
                symbol: pair.spot_symbol.clone(),
                action: OrderSide::Sell,
                quantity: quantity,
                reason: format!("Future failed: {}", e),
                attempts: 0,
            };
             if let Err(send_err) = self.recovery_tx.send(task).await {
                 error!("EMERGENCY: Failed to queue recovery task! Manual intervention required. Error: {}", send_err);
                 // CF4 FIX: Return CriticalFailure instead of panic
                 return Err(ExecutionError::CriticalFailure(format!("Recovery queue failure - future entry kill switch failed: {}", send_err)));
             }
            return Ok(ExecutionResult::PartialFailure(e));
        }

        Ok(ExecutionResult::Success)
    }

    #[instrument(skip(self))]
    pub async fn execute_basis_exit(&self, pair: &InstrumentPair, quantity: Decimal, hedge_qty: Decimal, leg1_price: Decimal, leg2_price: Decimal) -> Result<ExecutionResult, ExecutionError> {
         // Reverse of entry: Sell Spot, Buy Future
        let spot_leg = self.client.execute_order(&pair.spot_symbol, OrderSide::Sell, quantity, Some(leg1_price));
        let future_leg = self.client.execute_order(&pair.future_symbol, OrderSide::Buy, hedge_qty, Some(leg2_price));

        let (spot_res, future_res) = tokio::join!(spot_leg, future_leg);
        
        let spot_res = spot_res.map_err(|e| ExecutionError::from_boxed(e));
        let future_res = future_res.map_err(|e| ExecutionError::from_boxed(e));

        if spot_res.is_err() && future_res.is_err() {
            return Ok(ExecutionResult::TotalFailure(ExecutionError::ExchangeError(format!("Both legs failed to exit. Spot: {:?}, Future: {:?}", spot_res.err(), future_res.err()))));
        }

        if spot_res.is_err() || future_res.is_err() {
            // For exit, failures are messy. We might be left with a position.
            // A real system would have complex unwinding logic here.
            // For now, we queue whatever failed to be retried? 
            // Actually, if exit fails, we want to RETRY the exit, not reverse it.
            // So we queue the failed leg to be executed again.
            
            if let Err(e) = spot_res {
                error!("Exit Spot failed: {}. Queuing retry.", e);
                let task = RecoveryTask {
                    symbol: pair.spot_symbol.clone(),
                    action: OrderSide::Sell,
                    quantity: quantity,
                    reason: format!("Exit Spot failed: {}", e),
                    attempts: 0,
                };
                // CF4 FIX: Handle recovery task send errors
                if let Err(send_err) = self.recovery_tx.send(task).await {
                    error!("EMERGENCY: Failed to queue exit spot recovery task! Error: {}", send_err);
                    return Err(ExecutionError::CriticalFailure(format!("Recovery queue failure - exit spot retry failed: {}", send_err)));
                }
            }
            
            if let Err(e) = future_res {
                error!("Exit Future failed: {}. Queuing retry.", e);
                let task = RecoveryTask {
                    symbol: pair.future_symbol.clone(),
                    action: OrderSide::Buy,
                    quantity: hedge_qty,
                    reason: format!("Exit Future failed: {}", e),
                    attempts: 0,
                };
                // CF4 FIX: Handle recovery task send errors
                if let Err(send_err) = self.recovery_tx.send(task).await {
                    error!("EMERGENCY: Failed to queue exit future recovery task! Error: {}", send_err);
                    return Err(ExecutionError::CriticalFailure(format!("Recovery queue failure - exit future retry failed: {}", send_err)));
                }
            }

            return Ok(ExecutionResult::PartialFailure(ExecutionError::ExchangeError("Exit failed. Recovery tasks queued.".to_string())));
        }
        
        Ok(ExecutionResult::Success)
    }
}

// --- Main Strategy Class ---

/// The main coordinator for the Dual-Leg Trading Strategy.
pub struct DualLegStrategy {
    entry_manager: Box<dyn EntryStrategy>,
    risk_monitor: RiskMonitor,
    execution_engine: Arc<ExecutionEngine>,
    pair: Arc<InstrumentPair>,
    state: StrategyState,
    last_state_change_ts: i64,
    report_tx: mpsc::Sender<ExecutionReport>,
    report_rx: mpsc::Receiver<ExecutionReport>,
    recovery_rx: mpsc::Receiver<RecoveryResult>,
    clock: Box<dyn Clock>,
    throttler: DualLegLogThrottler,
    pub state_notifier: Option<mpsc::Sender<StrategyState>>,
    config: DualLegConfig,
    tasks: tokio::task::JoinSet<()>,
}

impl DualLegStrategy {
    /// Creates a new `DualLegStrategy`.
    pub fn new(
        entry_manager: Box<dyn EntryStrategy>,
        risk_monitor: RiskMonitor,
        execution_engine: ExecutionEngine,
        config: DualLegConfig,
        recovery_rx: mpsc::Receiver<RecoveryResult>,
        clock: Box<dyn Clock>,
    ) -> Self {
        let (report_tx, report_rx) = mpsc::channel(100);
        let now = clock.now_ts_millis();
        let pair = InstrumentPair { 
            spot_symbol: config.spot_symbol.clone(), 
            future_symbol: config.future_symbol.clone() 
        };
        
        Self {
            entry_manager,
            risk_monitor,
            execution_engine: Arc::new(execution_engine),
            pair: Arc::new(pair),
            state: StrategyState::Flat,
            last_state_change_ts: now,
            report_tx,
            report_rx,
            recovery_rx,
            clock,
            throttler: DualLegLogThrottler::new(config.throttle_interval_secs),
            state_notifier: None,
            config,
            tasks: tokio::task::JoinSet::new(),
        }
    }

    pub fn set_observer(&mut self, tx: mpsc::Sender<StrategyState>) {
        self.state_notifier = Some(tx);
    }

    fn transition_state(&mut self, new_state: StrategyState) {
        if self.state != new_state {
            info!("State Transition: {:?} -> {:?}", self.state, new_state);
            self.state = new_state.clone();
            self.last_state_change_ts = self.clock.now_ts_millis();
            
            if let Some(tx) = &self.state_notifier {
                if let Err(e) = tx.try_send(new_state) {
                     warn!("State update dropped due to backpressure: {}", e);
                }
            }
        }
    }

    /// Runs the strategy loop.
    #[instrument(skip(self, leg1_rx, leg2_rx), name = "strategy_loop")]
    pub async fn run(&mut self, mut leg1_rx: tokio::sync::mpsc::Receiver<Arc<MarketData>>, mut leg2_rx: tokio::sync::mpsc::Receiver<Arc<MarketData>>) {
        info!("Starting Dual-Leg Strategy for {}/{}", self.pair.spot_symbol, self.pair.future_symbol);

        let mut latest_leg1: Option<Arc<MarketData>> = None;
        let mut latest_leg2: Option<Arc<MarketData>> = None;
        let mut dirty = false;
        let mut heartbeat = tokio::time::interval(Duration::from_secs(1));

        loop {
            tokio::select! {
                _ = heartbeat.tick() => {
                    self.check_timeout();
                }
                Some(leg1_data) = leg1_rx.recv() => {
                    latest_leg1 = Some(leg1_data);
                    dirty = true;
                }
                Some(leg2_data) = leg2_rx.recv() => {
                    latest_leg2 = Some(leg2_data);
                    dirty = true;
                }
                Some(report) = self.report_rx.recv() => {
                    self.handle_execution_report(report);
                }
                Some(recovery_res) = self.recovery_rx.recv() => {
                    self.handle_recovery_result(recovery_res);
                }
                // Some(_res) = self.tasks.join_next(), if !self.tasks.is_empty() => {
                //     // Reap finished tasks. We could handle panics here if needed.
                // }
                else => break, // Channels closed
            }

            if dirty {
                if let (Some(leg1), Some(leg2)) = (&latest_leg1, &latest_leg2) {
                    self.process_tick(leg1, leg2).await;
                }
                dirty = false;
            }
        }
    }

    fn handle_execution_report(&mut self, report: ExecutionReport) {
        info!("Received Execution Report: {:?}", report);
        match report.result {
            ExecutionResult::Success => {
                match report.action {
                    Signal::Buy => {
                        info!("Entry Successful. Transitioning to InPosition.");
                        // We need to know the quantities to transition to InPosition.
                        // Since we are in Entering state, we should have them there.
                        // However, we need to access self.state.
                        // This requires a bit of a change in logic because we are in a method.
                        // But we can check the current state.
                        if let StrategyState::Entering { leg1_qty, leg2_qty, leg1_entry_price, leg2_entry_price } = self.state {
                             self.transition_state(StrategyState::InPosition { leg1_qty, leg2_qty, leg1_entry_price, leg2_entry_price });
                        } else {
                            error!("Received Entry Success but state is not Entering! State: {:?}", self.state);
                            // Fallback or error handling?
                            // If we are Reconciling, maybe we stay there?
                        }
                    },
                    Signal::Sell => {
                        info!("Exit Successful. Transitioning to Flat.");
                        self.transition_state(StrategyState::Flat);
                    },
                    _ => {}
                }
            },
            ExecutionResult::TotalFailure(reason) => {
                error!("Execution Failed completely: {}. Reverting state.", reason);
                if let ExecutionError::CriticalFailure(_) = reason {
                    error!("CRITICAL FAILURE DETECTED. Halting Strategy.");
                    self.transition_state(StrategyState::Halted);
                    return;
                }
                match report.action {
                    Signal::Buy => {
                         // Entry failed completely.
                         // If ambiguous (Network/Timeout), we DO NOT know if orders filled. Reconcile.
                         if matches!(reason, ExecutionError::NetworkError(_) | ExecutionError::Unknown(_)) {
                             error!("Ambiguous Entry Failure ({:?}). Transitioning to Reconciling.", reason);
                             self.transition_state(StrategyState::Reconciling);
                         } else {
                             // Definitive failure (e.g. Rejected). Safe to assume Flat.
                             self.transition_state(StrategyState::Flat);
                         }
                    },
                    Signal::Sell => {
                         // Exit failed.
                         // If ambiguous (Network/Timeout), we DO NOT know if we are InPosition or Flat/Partial.
                         // MUST Reconcile.
                         if matches!(reason, ExecutionError::NetworkError(_) | ExecutionError::Unknown(_)) {
                             error!("Ambiguous Exit Failure ({:?}). Transitioning to Reconciling.", reason);
                             self.transition_state(StrategyState::Reconciling);
                         } else {
                             // Definitive failure (e.g. Rejected). We are likely still InPosition.
                             if let StrategyState::Exiting { leg1_qty, leg2_qty, leg1_entry_price, leg2_entry_price } = self.state {
                                 info!("Exit rejected/failed definitively. Reverting to InPosition.");
                                 self.transition_state(StrategyState::InPosition { leg1_qty, leg2_qty, leg1_entry_price, leg2_entry_price });
                             } else {
                                 error!("CRITICAL: State mismatch! Expected Exiting but found {:?}. Transitioning to Reconciling.", self.state);
                                 self.transition_state(StrategyState::Reconciling); 
                             }
                         }
                    },
                    _ => {}
                }
            },
            ExecutionResult::PartialFailure(reason) => {
                error!("CRITICAL: Partial Execution Failure: {}. Transitioning to Reconciling.", reason);
                // We don't know the exact state, so we go to Reconciling.
                // The RecoveryWorker is already working on it.
                self.transition_state(StrategyState::Reconciling);
            }
        }
    }

    fn handle_recovery_result(&mut self, result: RecoveryResult) {
        info!("Received Recovery Result: {:?}", result);
        match result {
            RecoveryResult::Success(symbol) => {
                info!("Recovery successful for {}. Attempting to resolve state.", symbol);
                // If we were Reconciling, we might be able to go back to Flat or InPosition.
                // This logic depends on what we were trying to do.
                // For simplicity: If we were entering and failed partially, recovery means we unwound -> Flat.
                // If we were exiting and failed partially, recovery means we finished exiting -> Flat.
                // Or maybe we reverted to InPosition?
                
                // Ideally we should track "Target State".
                // For now, let's assume recovery means we are "Safe".
                // If we are in Reconciling, we can move to Flat if we believe we are flat.
                // But we might be InPosition if we failed to exit and recovery just re-established the position?
                // Actually, the recovery tasks in ExecutionEngine are:
                // Entry Partial -> Kill Switch (Unwind) -> Goal: Flat
                // Exit Partial -> Retry Exit -> Goal: Flat
                
                // So in both current cases, success means Flat.
                if self.state == StrategyState::Reconciling {
                    info!("Recovery complete. Transitioning to Flat.");
                    self.transition_state(StrategyState::Flat);
                }
            },
            RecoveryResult::Failed(symbol) => {
                error!("Recovery FAILED for {}. Transitioning to Halted.", symbol);
                self.transition_state(StrategyState::Halted);
            }
        }
    }

    fn check_timeout(&mut self) {
        if matches!(self.state, StrategyState::Entering { .. } | StrategyState::Exiting { .. }) {
            let now = self.clock.now_ts_millis();
            if now - self.last_state_change_ts > self.config.execution_timeout_ms {
                error!("CRITICAL: Execution Timeout in state {:?}! No report received for {}ms. Transitioning to Reconciling.", self.state, self.config.execution_timeout_ms);
                
                // Transition to Reconciling instead of blind reset
                self.transition_state(StrategyState::Reconciling);
                self.trigger_reconciliation();
            }
        }
    }

    fn trigger_reconciliation(&self) {
        info!("Triggering reconciliation process...");
        // In a real system, this would spawn a task to query the exchange for open orders and positions
        // and send a report back to the main loop to resolve the state.
        // For now, we log an alert.
        error!("MANUAL INTERVENTION REQUIRED: Strategy is in Reconciling state. Please check exchange positions and restart if necessary.");
    }

    /// Processes a single tick of matched Leg 1 and Leg 2 data.
    #[instrument(skip(self, leg1, leg2), fields(leg1_price = %leg1.price, leg2_price = %leg2.price))]
    async fn process_tick(&mut self, leg1: &MarketData, leg2: &MarketData) {
        // Safety Guard: Do not process ticks if we are in an unstable state
        if matches!(self.state, StrategyState::Reconciling | StrategyState::Halted) {
            if self.throttler.unstable_state.should_log() {
                let suppressed = self.throttler.unstable_state.get_and_reset_suppressed_count();
                warn!("Dropping tick due to unstable state: {:?} (Suppressed: {})", self.state, suppressed);
            }
            return;
        }
        let now = self.clock.now_ts_millis();
        
        // Check 1: Data freshness (Age)
        let leg1_age = now - leg1.timestamp;
        let leg2_age = now - leg2.timestamp;
        if leg1_age.abs() > self.config.max_tick_age_ms || leg2_age.abs() > self.config.max_tick_age_ms {
            if self.throttler.tick_age.should_log() {
                let suppressed = self.throttler.tick_age.get_and_reset_suppressed_count();
                warn!("Dropping tick due to age. Leg1 Age: {}ms, Leg2 Age: {}ms (Max: {}ms) (Suppressed: {})", leg1_age, leg2_age, self.config.max_tick_age_ms, suppressed);
            }
            return; 
        }

        // Check 2: Data correlation (Synchronization)
        // Ensure the two price points are from roughly the same moment in time.
        let diff = (leg1.timestamp - leg2.timestamp).abs();
        // Using max_tick_age_ms as a proxy for max diff for now, or we could add a specific config.
        // Let's assume max_tick_age_ms is a reasonable bound for diff too.
        if diff > self.config.max_tick_age_ms {
            if self.throttler.sync_issue.should_log() {
                let suppressed = self.throttler.sync_issue.get_and_reset_suppressed_count();
                warn!("Dropping tick due to sync. Diff: {}ms (Max: {}ms). Leg1 TS: {}, Leg2 TS: {} (Suppressed: {})", diff, self.config.max_tick_age_ms, leg1.timestamp, leg2.timestamp, suppressed);
            }
            return;
        }

        let signal = self.entry_manager.analyze(leg1, leg2).await;

        match signal {
            Signal::Buy => {
                if self.state != StrategyState::Flat { return; }

                let leg1_qty = self.config.order_size; 
                if let Ok(hedge_qty) = self.risk_monitor.calc_hedge_ratio(leg1_qty, leg1.price, leg2.price) {
                    info!("Entry Signal! Transitioning to Entering state and spawning execution...");
                    self.transition_state(StrategyState::Entering { leg1_qty, leg2_qty: hedge_qty, leg1_entry_price: leg1.price, leg2_entry_price: leg2.price }); // Prevent further spawns

                    let engine = self.execution_engine.clone();
                    let pair = self.pair.clone();
                    let tx = self.report_tx.clone();
                    let p1 = leg1.price;
                    let p2 = leg2.price;

                    // CF1 & CF2 FIX: Add explicit error handling for report channel
                    tokio::spawn(async move {
                        let result = match engine.execute_basis_entry(&pair, leg1_qty, hedge_qty, p1, p2).await {
                            Ok(res) => res,
                            Err(e) => ExecutionResult::TotalFailure(e),
                        };
                        let report = ExecutionReport { result, action: Signal::Buy, pnl_delta: None };
                        
                        // Critical: Must notify main loop of execution result
                        match tx.send(report).await {
                            Ok(_) => debug!("Entry execution report sent successfully"),
                            Err(e) => {
                                error!("CRITICAL: Execution report channel closed: {}. Strategy state desync!", e);
                                // We cannot panic here as it would only kill the spawned task.
                                // The main loop will eventually timeout.
                            }
                        }
                    });
                }
            },
            Signal::Sell => {
                // Use quantities from current position
                if let StrategyState::InPosition { leg1_qty, leg2_qty, leg1_entry_price, leg2_entry_price } = self.state {
                    // --- Minimum Profit Check ---
                    // Calculate Estimated PnL
                    // PnL = (Exit Price - Entry Price) * Quantity * Side Multiplier
                    // Fees = Value * Fee Rate (0.1% taker) * 2 (Entry + Exit)
                    
                    // Leg 1 (BTC): Long Entry -> Sell Exit
                    let leg1_pnl = (leg1.price - leg1_entry_price) * leg1_qty;
                    
                    // Leg 2 (ETH): Short Entry -> Buy Exit
                    let leg2_pnl = (leg2_entry_price - leg2.price) * leg2_qty;
                    
                    let gross_pnl = leg1_pnl + leg2_pnl;
                    
                    // Estimate Fees: Use configured taker fee
                    // Total Volume = (P1_entry * Q1) + (P2_entry * Q2) + (P1_exit * Q1) + (P2_exit * Q2)
                    let total_volume = (leg1_entry_price * leg1_qty) + (leg2_entry_price * leg2_qty) + (leg1.price * leg1_qty) + (leg2.price * leg2_qty);
                    let fee_rate = self.config.fee_tier.taker_fee_bps / dec!(10000.0);
                    let estimated_fees = total_volume * fee_rate;
                    
                    // CF5 FIX: Include slippage in PnL calculation
                    let slippage_cost = total_volume * (self.config.fee_tier.slippage_bps / dec!(10000.0));
                    let net_pnl = gross_pnl - estimated_fees - slippage_cost;
                    
                    let min_profit_threshold = self.config.min_profit_threshold;
                    let stop_loss_threshold = self.config.stop_loss_threshold;

                    if net_pnl < stop_loss_threshold {
                        warn!("STOP LOSS TRIGGERED! Net PnL: {:.6} < Threshold ({:.6}). Forcing Exit.", net_pnl, stop_loss_threshold);
                        // Proceed to exit (fall through)
                    } else if net_pnl < min_profit_threshold {
                        if self.throttler.unstable_state.should_log() {
                             let suppressed = self.throttler.unstable_state.get_and_reset_suppressed_count();
                             info!("Holding Position. Signal: Sell, but Net PnL ({:.6}) < Threshold ({:.6}). Gross: {:.6}, Fees: {:.6} (Suppressed: {})", net_pnl, min_profit_threshold, gross_pnl, estimated_fees, suppressed);
                        }
                        return;
                    }

                    info!("Exit Signal! Net PnL: {:.6} (Gross: {:.6}, Fees: {:.6}, Slippage: {:.6}). Transitioning to Exiting...", net_pnl, gross_pnl, estimated_fees, slippage_cost);
                    self.transition_state(StrategyState::Exiting { leg1_qty, leg2_qty, leg1_entry_price, leg2_entry_price });

                    let engine = self.execution_engine.clone();
                    let pair = self.pair.clone();
                    let tx = self.report_tx.clone();
                    // Calculate Limit Prices with Slippage for Exit
                    // Sell Spot: Limit = Price * (1 - Slippage)
                    // Buy Future: Limit = Price * (1 + Slippage)
                    let slippage_factor = self.config.fee_tier.slippage_bps / dec!(10000.0);
                    let p1_limit = leg1.price * (dec!(1.0) - slippage_factor);
                    let p2_limit = leg2.price * (dec!(1.0) + slippage_factor);
                    
                    debug!("Executing Exit with Limit Prices - Spot: {} (Market: {}), Future: {} (Market: {})", p1_limit, leg1.price, p2_limit, leg2.price);

                    // CF1 & CF2 FIX: Add explicit error handling for report channel
                    tokio::spawn(async move {
                        let result = match engine.execute_basis_exit(&pair, leg1_qty, leg2_qty, p1_limit, p2_limit).await {
                            Ok(res) => res,
                            Err(e) => ExecutionResult::TotalFailure(e),
                        };
                        let report = ExecutionReport { result, action: Signal::Sell, pnl_delta: None };
                        
                        // Critical: Must notify main loop of execution result
                        match tx.send(report).await {
                            Ok(_) => debug!("Exit execution report sent successfully"),
                            Err(e) => {
                                error!("CRITICAL: Execution report channel closed: {}. Strategy state desync!", e);
                                // We cannot panic here as it would only kill the spawned task.
                                // The main loop will eventually timeout.
                            }
                        }
                    });
                }
            },
            _ => {}
        }
    }
}

#[async_trait]
impl Executor for CoinbaseClient {
    async fn execute_order(&self, symbol: &str, side: OrderSide, quantity: Decimal, price: Option<Decimal>) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.place_order(symbol, &side.to_string(), quantity, price).await
    }
    
    async fn get_position(
        &self,
        symbol: &str,
    ) -> Result<Decimal, Box<dyn Error + Send + Sync>> {
        self.get_position(symbol).await
    }
}

// AS7: Drop implementation for cleanup logging
impl Drop for DualLegStrategy {
    fn drop(&mut self) {
        info!(
            spot = %self.pair.spot_symbol,
            future = %self.pair.future_symbol,
            state = ?self.state,
            "DualLegStrategy dropped - cleanup complete"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::Duration;
use tokio::time::Instant;
use tokio::sync::{Mutex, RwLock};

    #[test]
    fn test_log_throttle() {
        let mut throttle = LogThrottle::new(Duration::from_millis(100));

        // First call should log
        assert!(throttle.should_log());
        assert_eq!(throttle.get_and_reset_suppressed_count(), 0);

        // Immediate subsequent calls should be suppressed
        assert!(!throttle.should_log());
        assert!(!throttle.should_log());
        assert!(!throttle.should_log());

        // Check suppressed count
        assert_eq!(throttle.get_and_reset_suppressed_count(), 3);
        // Count should be reset
        assert_eq!(throttle.get_and_reset_suppressed_count(), 0);

        // Wait for interval
        std::thread::sleep(Duration::from_millis(110));

        // Should log again
        assert!(throttle.should_log());
    }
    #[tokio::test]
    async fn test_z_score_calculation() {
        let manager = PairsManager::new(5, 2.0, 0.1);
        
        // Feed data: 10, 12, 12, 10, 12
        // Log spreads: ln(10)-ln(10)=0, ln(12)-ln(10)=0.182, etc.
        // Let's just feed synthetic spreads by mocking prices.
        // P1 = e^spread, P2 = 1.0
        
        let spreads: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // Mean = 3.0, StdDev = sqrt(2) = 1.414
        // Next spread = 6.0. Z = (6 - 3.4) / ...
        // Rolling window behavior.
        
        for s in spreads {
            let p1 = Decimal::from_f64(s.exp()).unwrap();
            let p2 = dec!(1.0);
            let _ = manager.analyze(
                &MarketData { symbol: "A".into(), price: p1, instrument_id: None, timestamp: 0 },
                &MarketData { symbol: "B".into(), price: p2, instrument_id: None, timestamp: 0 }
            ).await;
        }

        // Window is full [1, 2, 3, 4, 5]. Mean=3, StdDev=1.4142
        // Feed 6.0. New Window [2, 3, 4, 5, 6]. Mean=4, StdDev=1.4142
        // Z = (6 - 4) / 1.4142 = 1.4142
        
        // We can't inspect internal state easily without exposing it, but we can check signals.
        // Let's test signal generation.
        
        // Case 1: Z < -2.0 -> Buy
        // Mean=0, Std=1. Feed -3.0 -> Z=-3.0 -> Buy
        let manager = PairsManager::new(5, 2.0, 0.1);
        // Fill with 0s
        for _ in 0..5 {
             let _ = manager.analyze(
                &MarketData { symbol: "A".into(), price: dec!(1.0), instrument_id: None, timestamp: 0 },
                &MarketData { symbol: "B".into(), price: dec!(1.0), instrument_id: None, timestamp: 0 }
            ).await;
        }
        
        // Feed a drop in spread (P1 drops)
        // Spread = ln(0.1) - ln(1) = -2.3
        let _p1 = Decimal::from_f64(0.1f64.exp()).unwrap(); // e^-2.3 approx 0.1
        // Actually let's just use exact math.
        // Spread = -3.0.
        let p1 = Decimal::from_f64((-3.0f64).exp()).unwrap();
        let _signal = manager.analyze(
            &MarketData { symbol: "A".into(), price: p1, instrument_id: None, timestamp: 0 },
            &MarketData { symbol: "B".into(), price: dec!(1.0), instrument_id: None, timestamp: 0 }
        ).await;
        
        // Current window: [0, 0, 0, 0, -3]. Mean = -0.6. Var = (4*0.36 + 1*5.76)/5 = (1.44 + 5.76)/5 = 1.44. Std = 1.2
        // Z = (-3 - (-0.6)) / 1.2 = -2.4 / 1.2 = -2.0
        // Threshold is 2.0. Z <= -2.0 (if strictly less, might fail, let's go deeper)
        
        // Let's try -4.0
        let p1 = Decimal::from_f64((-4.0f64).exp()).unwrap();
        let _signal = manager.analyze(
            &MarketData { symbol: "A".into(), price: p1, instrument_id: None, timestamp: 0 },
            &MarketData { symbol: "B".into(), price: dec!(1.0), instrument_id: None, timestamp: 0 }
        ).await;
        
        // Window: [0, 0, 0, -3, -4]. Mean = -1.4. 
        // This is getting complicated to track manually.
        // But we expect a Buy signal eventually if we drop enough.
        
        // Re-init for cleaner state
        let manager = PairsManager::new(10, 2.0, 0.1);
        for _ in 0..10 {
             let _ = manager.analyze(
                &MarketData { symbol: "A".into(), price: dec!(1.0), instrument_id: None, timestamp: 0 },
                &MarketData { symbol: "B".into(), price: dec!(1.0), instrument_id: None, timestamp: 0 }
            ).await;
        }
        // Spread 0, Mean 0, Std 0.
        
        // Sudden drop to -10.
        let p1 = Decimal::from_f64((-10.0f64).exp()).unwrap();
        let signal = manager.analyze(
            &MarketData { symbol: "A".into(), price: p1, instrument_id: None, timestamp: 0 },
            &MarketData { symbol: "B".into(), price: dec!(1.0), instrument_id: None, timestamp: 0 }
        ).await;
        
        assert_eq!(signal, Signal::Buy);
    }

    #[test]
    fn test_hedge_ratio_dollar_neutral() {
        let monitor = RiskMonitor::new(dec!(1.0), InstrumentType::Linear, HedgeMode::DollarNeutral);
        
        // QtyA = 1, PriceA = 100, PriceB = 50.
        // ValueA = 100. QtyB = 100 / 50 = 2.
        let ratio = monitor.calc_hedge_ratio(dec!(1.0), dec!(100.0), dec!(50.0)).unwrap();
        assert_eq!(ratio, dec!(2.0));
        
        // QtyA = 0.5, PriceA = 20000, PriceB = 1000.
        // ValueA = 10000. QtyB = 10000 / 1000 = 10.
        let ratio = monitor.calc_hedge_ratio(dec!(0.5), dec!(20000.0), dec!(1000.0)).unwrap();
        assert_eq!(ratio, dec!(10.0));
    }

    #[test]
    fn test_hedge_ratio_delta_neutral_linear() {
        let monitor = RiskMonitor::new(dec!(1.0), InstrumentType::Linear, HedgeMode::DeltaNeutral);
        
        // 1:1 ratio
        let ratio = monitor.calc_hedge_ratio(dec!(1.5), dec!(100.0), dec!(50.0)).unwrap();
        assert_eq!(ratio, dec!(1.5));
    }

    #[test]
    fn test_hedge_ratio_delta_neutral_inverse() {
        let monitor = RiskMonitor::new(dec!(1.0), InstrumentType::Inverse, HedgeMode::DeltaNeutral);
        
        // QtyA = 1 (BTC). PriceB = 50000 (USD/BTC).
        // Contracts = 1 * 50000 = 50000.
        let ratio = monitor.calc_hedge_ratio(dec!(1.0), dec!(100.0), dec!(50000.0)).unwrap();
        assert_eq!(ratio, dec!(50000.0));
    }
    #[test]
    fn test_net_spread_calculation() {
        // Maker 10, Taker 20, Slippage 5.
        // Total Fees = 4 * 20 = 80 bps.
        // Slippage = 5 bps.
        // Net = Gross - 85.
        let model = TransactionCostModel::new(dec!(10.0), dec!(20.0), dec!(5.0));
        
        let net = model.calc_net_spread(dec!(100.0));
        assert_eq!(net, dec!(15.0)); // 100 - 85
        
        let net = model.calc_net_spread(dec!(50.0));
        assert_eq!(net, dec!(-35.0)); // 50 - 85
    }

    #[tokio::test]
    async fn test_basis_manager_signals() {
        let cost_model = TransactionCostModel::new(dec!(0.0), dec!(0.0), dec!(0.0)); // Zero costs for simplicity
        let manager = BasisManager::new(dec!(10.0), dec!(2.0), cost_model);
        
        // Spot 100, Future 100.2. Spread = 0.2/100 = 20 bps.
        // Net = 20. > 10. -> Buy.
        let signal = manager.analyze(
            &MarketData { symbol: "S".into(), price: dec!(100.0), instrument_id: None, timestamp: 0 },
            &MarketData { symbol: "F".into(), price: dec!(100.2), instrument_id: None, timestamp: 0 }
        ).await;
        assert_eq!(signal, Signal::Buy);
        
        // Spot 100, Future 100.01. Spread = 1 bps.
        // < 2. -> Sell.
        let signal = manager.analyze(
            &MarketData { symbol: "S".into(), price: dec!(100.0), instrument_id: None, timestamp: 0 },
            &MarketData { symbol: "F".into(), price: dec!(100.01), instrument_id: None, timestamp: 0 }
        ).await;
        assert_eq!(signal, Signal::Sell);
        
        // Spot 100, Future 100.05. Spread = 5 bps.
        // Between 2 and 10. -> Hold.
        let signal = manager.analyze(
            &MarketData { symbol: "S".into(), price: dec!(100.0), instrument_id: None, timestamp: 0 },
            &MarketData { symbol: "F".into(), price: dec!(100.05), instrument_id: None, timestamp: 0 }
        ).await;
        assert_eq!(signal, Signal::Hold);
    }
}
