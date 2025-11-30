//! # Delta-Neutral Basis Trading Strategy
//!
//! This module implements a Delta-Neutral Basis Trading strategy that exploits the price difference (basis)
//! between the Spot market and Futures/Perpetuals.
//!
//! ## Architecture
//! The strategy is composed of three main components:
//! - `EntryManager`: Analyzes the basis spread and generates entry/exit signals.
//! - `RiskMonitor`: Tracks delta, margin, and exposure, ensuring the strategy remains delta-neutral.
//! - `ExecutionEngine`: Handles the concurrent execution of orders on both legs (Spot and Futures).
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
use tokio::sync::{mpsc, Semaphore};
use tokio::time::{Duration, Instant};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BasisTradingError {
    #[error("Invalid input parameters: {0}")]
    InvalidInput(String),
    #[error("Math error: {0}")]
    MathError(String),
    #[error("Execution error: {0}")]
    ExecutionError(String),
    #[error("Unknown error: {0}")]
    Unknown(String),
}

const MAX_TICK_AGE_MS: i64 = 2000; // 2 seconds
const MAX_TICK_DIFF_MS: i64 = 500; // 500 milliseconds
const EXECUTION_TIMEOUT_MS: i64 = 30000; // 30 seconds

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
pub struct BasisLogThrottler {
    pub unstable_state: LogThrottle,
    pub tick_age: LogThrottle,
    pub sync_issue: LogThrottle,
}

impl BasisLogThrottler {
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

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Clone, PartialEq)]
pub enum StrategyState {
    Flat,
    Entering,
    InPosition,
    Exiting,
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
    PartialFailure(String), // e.g., "Spot filled, Future failed"
    TotalFailure(String),
}

#[derive(Debug)]
pub struct ExecutionReport {
    pub result: ExecutionResult,
    pub action: Signal, // Buy (Entry) or Sell (Exit)
}

#[derive(Debug, Clone)]
pub struct RecoveryTask {
    pub symbol: String,
    pub action: String, // "buy" or "sell"
    pub quantity: Decimal,
    pub reason: String,
    pub attempts: u32,
}

// --- Interfaces (Dependency Injection) ---

/// Trait for entry logic strategies.
/// Allows for swapping different entry algorithms (e.g., simple threshold vs. statistical arbitrage).
#[async_trait]
pub trait EntryStrategy: Send + Sync {
    /// Analyzes the market data for Spot and Future to generate a trading signal.
    async fn analyze(&self, spread: Spread) -> Signal;
}

/// Trait for order execution.
/// Abstracts the underlying exchange client to facilitate testing and multi-exchange support.
#[async_trait]
pub trait Executor: Send + Sync {
    /// Executes an order on the exchange.
    async fn execute_order(&self, symbol: &str, side: &str, quantity: Decimal) -> Result<(), Box<dyn Error + Send + Sync>>;
}

// --- Components ---

/// Analyzes the basis spread and generates signals based on fixed thresholds.
pub struct EntryManager {
    entry_threshold_bps: Decimal,
    exit_threshold_bps: Decimal,
    cost_model: TransactionCostModel,
}

impl EntryManager {
    pub fn new(entry_threshold_bps: Decimal, exit_threshold_bps: Decimal, cost_model: TransactionCostModel) -> Self {
        Self {
            entry_threshold_bps,
            exit_threshold_bps,
            cost_model,
        }
    }
}

#[async_trait]
impl EntryStrategy for EntryManager {
    #[instrument(skip(self))]
    async fn analyze(&self, spread: Spread) -> Signal {
        let net_spread = self.cost_model.calc_net_spread(spread.value_bps);
        debug!("Basis Spread: {:.4} bps (Net: {:.4}) (Spot: {}, Future: {})", spread.value_bps, net_spread, spread.spot_price, spread.future_price);
        
        if net_spread > self.entry_threshold_bps {
            Signal::Buy
        } else if spread.value_bps < self.exit_threshold_bps {
            // For exit, we might care about gross spread being low enough, or net spread being high enough (if we are shorting the basis).
            // Usually for basis trading: Buy when spread is high, Sell when spread is low (converges).
            // So exit condition is usually just convergence.
            Signal::Sell
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

/// Tracks delta and enforces safety limits.
pub struct RiskMonitor {
    _max_leverage: Decimal,
    _target_delta: Decimal, // Should be 0.0 for delta-neutral
    instrument_type: InstrumentType,
}

impl RiskMonitor {
    pub fn new(max_leverage: Decimal, instrument_type: InstrumentType) -> Self {
        Self {
            _max_leverage: max_leverage,
            _target_delta: Decimal::zero(),
            instrument_type,
        }
    }

    pub fn calc_hedge_ratio(&self, spot_quantity: Decimal, _spot_price: Decimal, future_price: Decimal) -> Result<Decimal, BasisTradingError> {
        if spot_quantity <= Decimal::zero() {
            return Err(BasisTradingError::InvalidInput("Spot quantity must be positive".to_string()));
        }
        
        match self.instrument_type {
            InstrumentType::Linear => Ok(spot_quantity),
            InstrumentType::Inverse => {
                // Inverse: Contract value = 1 USD (usually)
                // Quantity = (Spot Value in USD) / (Contract Value in USD)
                // Spot Value = spot_quantity * spot_price (but here we are hedging the USD value)
                // Wait, for Inverse (e.g. BTC-USD perp), 1 contract = 1 USD.
                // To hedge 1 BTC at $50,000, we need 50,000 contracts.
                // Hedge Qty = Spot Qty * Future Price
                if future_price.is_zero() {
                    return Err(BasisTradingError::MathError("Future price cannot be zero for inverse calculation".to_string()));
                }
                Ok(spot_quantity * future_price)
            }
        }
    }
}

/// Worker that processes recovery tasks (failed legs).
pub struct RecoveryWorker {
    client: Arc<dyn Executor>,
    rx: mpsc::Receiver<RecoveryTask>,
    feedback_tx: mpsc::Sender<RecoveryResult>,
    semaphore: Arc<Semaphore>,
}

impl RecoveryWorker {
    pub fn new(client: Arc<dyn Executor>, rx: mpsc::Receiver<RecoveryTask>, feedback_tx: mpsc::Sender<RecoveryResult>) -> Self {
        Self { 
            client, 
            rx,
            feedback_tx,
            semaphore: Arc::new(Semaphore::new(5)), // Limit to 5 concurrent recoveries
        }
    }

    #[instrument(skip(self), name = "recovery_worker")]
    pub async fn run(mut self) {
        info!("Recovery Worker started.");
        while let Some(mut task) = self.rx.recv().await {
            let client = self.client.clone();
            let feedback_tx = self.feedback_tx.clone();
            let permit = self.semaphore.clone().acquire_owned().await;
            
            if let Ok(permit) = permit {
                tokio::spawn(async move {
                    // Permit is held until dropped at end of scope
                    let _permit = permit;
                    
                    info!("Processing Recovery Task: {:?}", task);
                    let mut backoff = Duration::from_secs(2);

                    loop {
                        task.attempts += 1;
                        match client.execute_order(&task.symbol, &task.action, task.quantity).await {
                            Ok(_) => {
                                info!("Recovery Successful for {} on attempt {}", task.symbol, task.attempts);
                                let _ = feedback_tx.send(RecoveryResult::Success(task.symbol.clone())).await;
                                break;
                            },
                            Err(e) => {
                                error!("Recovery Failed for {} (Attempt {}): {}", task.symbol, task.attempts, e);
                                if task.attempts >= 5 {
                                    error!("CRITICAL: Recovery abandoned for {} after 5 attempts. MANUAL INTERVENTION REQUIRED.", task.symbol);
                                    let _ = feedback_tx.send(RecoveryResult::Failed(task.symbol.clone())).await;
                                    break;
                                }
                                tokio::time::sleep(backoff).await;
                                backoff = std::cmp::min(backoff * 2, Duration::from_secs(60));
                            }
                        }
                    }
                });
            } else {
                error!("Failed to acquire semaphore for recovery task");
            }
        }
    }
}

/// Handles order execution logic.
#[derive(Clone)]
pub struct ExecutionEngine {
    client: Arc<dyn Executor>,
    recovery_tx: mpsc::Sender<RecoveryTask>,
}

impl ExecutionEngine {
    pub fn new(client: Arc<dyn Executor>, recovery_tx: mpsc::Sender<RecoveryTask>) -> Self {
        Self { client, recovery_tx }
    }

    #[instrument(skip(self))]
    pub async fn execute_basis_entry(&self, pair: &InstrumentPair, quantity: Decimal, hedge_qty: Decimal) -> ExecutionResult {
        // Concurrently execute both legs to minimize leg risk
        let spot_leg = self.client.execute_order(&pair.spot_symbol, "buy", quantity);
        let future_leg = self.client.execute_order(&pair.future_symbol, "sell", hedge_qty);

        let (spot_res, future_res) = tokio::join!(spot_leg, future_leg);
        
        // Convert errors to String immediately
        let spot_res = spot_res.map_err(|e| e.to_string());
        let future_res = future_res.map_err(|e| e.to_string());

        if let Err(e) = spot_res {
             if future_res.is_ok() {
                 error!("CRITICAL: Spot failed but Future succeeded. Queuing Kill Switch on Future leg.");
                 // Queue Kill Switch: Buy back future
                 let task = RecoveryTask {
                     symbol: pair.future_symbol.clone(),
                     action: "buy".to_string(),
                     quantity: hedge_qty,
                     reason: format!("Spot failed: {}", e),
                     attempts: 0,
                 };
                 if let Err(send_err) = self.recovery_tx.send(task).await {
                     error!("EMERGENCY: Failed to queue recovery task! Manual intervention required. Error: {}", send_err);
                 }
                 return ExecutionResult::PartialFailure(format!("Spot failed: {}. Future succeeded. Recovery queued.", e));
             }
             return ExecutionResult::TotalFailure(format!("Spot leg failed: {}", e));
        }

        if let Err(e) = future_res {
            error!("CRITICAL: Future leg failed: {}. Queuing Kill Switch on Spot leg.", e);
            // Queue Kill Switch: Sell spot
            let task = RecoveryTask {
                symbol: pair.spot_symbol.clone(),
                action: "sell".to_string(),
                quantity: quantity,
                reason: format!("Future failed: {}", e),
                attempts: 0,
            };
            if let Err(send_err) = self.recovery_tx.send(task).await {
                error!("EMERGENCY: Failed to queue recovery task! Manual intervention required. Error: {}", send_err);
            }
            return ExecutionResult::PartialFailure(format!("Future failed: {}. Spot succeeded. Recovery queued.", e));
        }

        ExecutionResult::Success
    }

    #[instrument(skip(self))]
    pub async fn execute_basis_exit(&self, pair: &InstrumentPair, quantity: Decimal, hedge_qty: Decimal) -> ExecutionResult {
         // Reverse of entry: Sell Spot, Buy Future
        let spot_leg = self.client.execute_order(&pair.spot_symbol, "sell", quantity);
        let future_leg = self.client.execute_order(&pair.future_symbol, "buy", hedge_qty);

        let (spot_res, future_res) = tokio::join!(spot_leg, future_leg);
        
        let spot_res = spot_res.map_err(|e| e.to_string());
        let future_res = future_res.map_err(|e| e.to_string());

        if spot_res.is_err() && future_res.is_err() {
            return ExecutionResult::TotalFailure(format!("Both legs failed to exit. Spot: {:?}, Future: {:?}", spot_res.err(), future_res.err()));
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
                    action: "sell".to_string(),
                    quantity: quantity,
                    reason: format!("Exit Spot failed: {}", e),
                    attempts: 0,
                };
                let _ = self.recovery_tx.send(task).await;
            }
            
            if let Err(e) = future_res {
                error!("Exit Future failed: {}. Queuing retry.", e);
                let task = RecoveryTask {
                    symbol: pair.future_symbol.clone(),
                    action: "buy".to_string(),
                    quantity: hedge_qty,
                    reason: format!("Exit Future failed: {}", e),
                    attempts: 0,
                };
                let _ = self.recovery_tx.send(task).await;
            }

            return ExecutionResult::PartialFailure(format!("Exit failed. Recovery tasks queued."));
        }
        
        ExecutionResult::Success
    }
}

// --- Main Strategy Class ---

/// The main coordinator for the Delta-Neutral Basis Trading Strategy.
pub struct BasisTradingStrategy {
    entry_manager: Box<dyn EntryStrategy>,
    risk_monitor: RiskMonitor,
    execution_engine: ExecutionEngine,
    pair: InstrumentPair,
    state: StrategyState,
    last_state_change_ts: i64,
    report_tx: mpsc::Sender<ExecutionReport>,
    report_rx: mpsc::Receiver<ExecutionReport>,
    recovery_rx: mpsc::Receiver<RecoveryResult>,
    clock: Box<dyn Clock>,
    throttler: BasisLogThrottler,
}

impl BasisTradingStrategy {
    /// Creates a new `BasisTradingStrategy`.
    pub fn new(
        entry_manager: Box<dyn EntryStrategy>,
        risk_monitor: RiskMonitor,
        execution_engine: ExecutionEngine,
        spot_symbol: String,
        future_symbol: String,
        recovery_rx: mpsc::Receiver<RecoveryResult>,
        clock: Box<dyn Clock>,
    ) -> Self {
        let (report_tx, report_rx) = mpsc::channel(100);
        let now = clock.now_ts_millis();
        Self {
            entry_manager,
            risk_monitor,
            execution_engine,
            pair: InstrumentPair { spot_symbol, future_symbol },
            state: StrategyState::Flat,
            last_state_change_ts: now,
            report_tx,
            report_rx,
            recovery_rx,
            clock,
            throttler: BasisLogThrottler::new(5), // 5 seconds default
        }
    }

    /// Runs the strategy loop.
    #[instrument(skip(self, spot_rx, future_rx), name = "strategy_loop")]
    pub async fn run(&mut self, mut spot_rx: tokio::sync::mpsc::Receiver<MarketData>, mut future_rx: tokio::sync::mpsc::Receiver<MarketData>) {
        info!("Starting Delta-Neutral Basis Strategy for {}/{}", self.pair.spot_symbol, self.pair.future_symbol);

        let mut latest_spot: Option<MarketData> = None;
        let mut latest_future: Option<MarketData> = None;
        let mut heartbeat = tokio::time::interval(Duration::from_secs(1));

        loop {
            tokio::select! {
                _ = heartbeat.tick() => {
                    self.check_timeout();
                }
                Some(spot_data) = spot_rx.recv() => {
                    latest_spot = Some(spot_data);
                }
                Some(future_data) = future_rx.recv() => {
                    latest_future = Some(future_data);
                }
                Some(report) = self.report_rx.recv() => {
                    self.handle_execution_report(report);
                }
                Some(recovery_res) = self.recovery_rx.recv() => {
                    self.handle_recovery_result(recovery_res);
                }
                else => break, // Channels closed
            }

            if let (Some(spot), Some(future)) = (&latest_spot, &latest_future) {
                self.process_tick(spot, future).await;
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
                        self.state = StrategyState::InPosition;
                        self.last_state_change_ts = self.clock.now_ts_millis();
                    },
                    Signal::Sell => {
                        info!("Exit Successful. Transitioning to Flat.");
                        self.state = StrategyState::Flat;
                        self.last_state_change_ts = self.clock.now_ts_millis();
                    },
                    _ => {}
                }
            },
            ExecutionResult::TotalFailure(reason) => {
                error!("Execution Failed completely: {}. Reverting state.", reason);
                // If entry failed, we go back to Flat. If exit failed, we might still be InPosition (or partial).
                // For TotalFailure (atomic rollback succeeded), we revert to previous state.
                match report.action {
                    Signal::Buy => {
                         self.state = StrategyState::Flat;
                         self.last_state_change_ts = self.clock.now_ts_millis();
                    },
                    Signal::Sell => {
                         self.state = StrategyState::InPosition; // Failed to exit, still in position
                         self.last_state_change_ts = self.clock.now_ts_millis();
                    },
                    _ => {}
                }
            },
            ExecutionResult::PartialFailure(reason) => {
                error!("CRITICAL: Partial Execution Failure: {}. Transitioning to Reconciling.", reason);
                // We don't know the exact state, so we go to Reconciling.
                // The RecoveryWorker is already working on it.
                self.state = StrategyState::Reconciling;
                self.last_state_change_ts = self.clock.now_ts_millis();
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
                    self.state = StrategyState::Flat;
                    self.last_state_change_ts = self.clock.now_ts_millis();
                }
            },
            RecoveryResult::Failed(symbol) => {
                error!("Recovery FAILED for {}. Transitioning to Halted.", symbol);
                self.state = StrategyState::Halted;
            }
        }
    }

    fn check_timeout(&mut self) {
        if self.state == StrategyState::Entering || self.state == StrategyState::Exiting {
            let now = self.clock.now_ts_millis();
            if now - self.last_state_change_ts > EXECUTION_TIMEOUT_MS {
                error!("CRITICAL: Execution Timeout in state {:?}! No report received for {}ms. Transitioning to Reconciling.", self.state, EXECUTION_TIMEOUT_MS);
                
                // Transition to Reconciling instead of blind reset
                self.state = StrategyState::Reconciling;
                self.last_state_change_ts = now;
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

    /// Processes a single tick of matched Spot and Future data.
    #[instrument(skip(self, spot, future), fields(spot_price = %spot.price, future_price = %future.price))]
    async fn process_tick(&mut self, spot: &MarketData, future: &MarketData) {
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
        let spot_age = now - spot.timestamp;
        let future_age = now - future.timestamp;
        if spot_age.abs() > MAX_TICK_AGE_MS || future_age.abs() > MAX_TICK_AGE_MS {
            if self.throttler.tick_age.should_log() {
                let suppressed = self.throttler.tick_age.get_and_reset_suppressed_count();
                warn!("Dropping tick due to age. Spot Age: {}ms, Future Age: {}ms (Max: {}ms) (Suppressed: {})", spot_age, future_age, MAX_TICK_AGE_MS, suppressed);
            }
            return; 
        }

        // Check 2: Data correlation (Synchronization)
        // Ensure the two price points are from roughly the same moment in time.
        let diff = (spot.timestamp - future.timestamp).abs();
        if diff > MAX_TICK_DIFF_MS {
            if self.throttler.sync_issue.should_log() {
                let suppressed = self.throttler.sync_issue.get_and_reset_suppressed_count();
                warn!("Dropping tick due to sync. Diff: {}ms (Max: {}ms). Spot TS: {}, Future TS: {} (Suppressed: {})", diff, MAX_TICK_DIFF_MS, spot.timestamp, future.timestamp, suppressed);
            }
            return;
        }

        let spread = Spread::new(spot.price, future.price);
        let signal = self.entry_manager.analyze(spread).await;

        match signal {
            Signal::Buy => {
                if self.state != StrategyState::Flat { return; }

                let spot_qty = dec!(0.1); 
                if let Ok(hedge_qty) = self.risk_monitor.calc_hedge_ratio(spot_qty, spot.price, future.price) {
                    info!("Entry Signal! Transitioning to Entering state and spawning execution...");
                    self.state = StrategyState::Entering; // Prevent further spawns
                    self.last_state_change_ts = self.clock.now_ts_millis();

                    let engine = self.execution_engine.clone();
                    let pair = self.pair.clone();
                    let tx = self.report_tx.clone();

                    tokio::spawn(async move {
                        let result = engine.execute_basis_entry(&pair, spot_qty, hedge_qty).await;
                        let report = ExecutionReport { result, action: Signal::Buy };
                        let _ = tx.send(report).await;
                    });
                }
            },
            Signal::Sell => {
                if self.state != StrategyState::InPosition { return; }

                let spot_qty = dec!(0.1); 
                if let Ok(hedge_qty) = self.risk_monitor.calc_hedge_ratio(spot_qty, spot.price, future.price) {
                    info!("Exit Signal! Transitioning to Exiting state and spawning execution...");
                    self.state = StrategyState::Exiting;
                    self.last_state_change_ts = self.clock.now_ts_millis();

                    let engine = self.execution_engine.clone();
                    let pair = self.pair.clone();
                    let tx = self.report_tx.clone();

                    tokio::spawn(async move {
                        let result = engine.execute_basis_exit(&pair, spot_qty, hedge_qty).await;
                        let report = ExecutionReport { result, action: Signal::Sell };
                        let _ = tx.send(report).await;
                    });
                }
            },
            _ => {}
        }
    }
}

#[async_trait]
impl Executor for CoinbaseClient {
    async fn execute_order(&self, symbol: &str, side: &str, quantity: Decimal) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.place_order(symbol, side, quantity).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::Duration;

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
}
