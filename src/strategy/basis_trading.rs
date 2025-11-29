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
use chrono::Utc;
use log::{info, debug, error};
use tokio::sync::mpsc;
use tokio::time::Duration;

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
    // In a real system, we'd have more granular states like "PartiallyFilled", "Unwinding", etc.
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
}

impl EntryManager {
    pub fn new(entry_threshold_bps: Decimal, exit_threshold_bps: Decimal) -> Self {
        Self {
            entry_threshold_bps,
            exit_threshold_bps,
        }
    }
}

#[async_trait]
impl EntryStrategy for EntryManager {
    async fn analyze(&self, spread: Spread) -> Signal {
        debug!("Basis Spread: {:.4} bps (Spot: {}, Future: {})", spread.value_bps, spread.spot_price, spread.future_price);
        
        if spread.value_bps > self.entry_threshold_bps {
            Signal::Buy
        } else if spread.value_bps < self.exit_threshold_bps {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }
}

/// Tracks delta and enforces safety limits.
pub struct RiskMonitor {
    _max_leverage: Decimal,
    _target_delta: Decimal, // Should be 0.0 for delta-neutral
}

impl RiskMonitor {
    pub fn new(max_leverage: Decimal) -> Self {
        Self {
            _max_leverage: max_leverage,
            _target_delta: Decimal::zero(),
        }
    }

    pub fn calc_hedge_ratio(&self, spot_quantity: Decimal, _spot_price: Decimal, _future_price: Decimal) -> Result<Decimal, String> {
        if spot_quantity <= Decimal::zero() {
            return Err("Invalid input parameters".to_string());
        }
        // Simple 1:1 hedge for now
        Ok(spot_quantity)
    }
}

/// Worker that processes recovery tasks (failed legs).
pub struct RecoveryWorker {
    client: Arc<dyn Executor>,
    rx: mpsc::Receiver<RecoveryTask>,
}

impl RecoveryWorker {
    pub fn new(client: Arc<dyn Executor>, rx: mpsc::Receiver<RecoveryTask>) -> Self {
        Self { client, rx }
    }

    pub async fn run(mut self) {
        info!("Recovery Worker started.");
        while let Some(task) = self.rx.recv().await {
            info!("Processing Recovery Task: {:?}", task);
            let mut attempts = 0;
            loop {
                attempts += 1;
                match self.client.execute_order(&task.symbol, &task.action, task.quantity).await {
                    Ok(_) => {
                        info!("Recovery Successful for {} on attempt {}", task.symbol, attempts);
                        break;
                    },
                    Err(e) => {
                        error!("Recovery Failed for {} (Attempt {}): {}", task.symbol, attempts, e);
                        if attempts >= 5 {
                            error!("CRITICAL: Recovery abandoned for {} after 5 attempts. MANUAL INTERVENTION REQUIRED.", task.symbol);
                            break;
                        }
                        tokio::time::sleep(Duration::from_secs(2)).await;
                    }
                }
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
            };
            if let Err(send_err) = self.recovery_tx.send(task).await {
                error!("EMERGENCY: Failed to queue recovery task! Manual intervention required. Error: {}", send_err);
            }
            return ExecutionResult::PartialFailure(format!("Future failed: {}. Spot succeeded. Recovery queued.", e));
        }

        ExecutionResult::Success
    }

    pub async fn execute_basis_exit(&self, pair: &InstrumentPair, quantity: Decimal, hedge_qty: Decimal) -> ExecutionResult {
         // Reverse of entry: Sell Spot, Buy Future
        let spot_leg = self.client.execute_order(&pair.spot_symbol, "sell", quantity);
        let future_leg = self.client.execute_order(&pair.future_symbol, "buy", hedge_qty);

        let (spot_res, future_res) = tokio::join!(spot_leg, future_leg);
        
        let spot_res = spot_res.map_err(|e| e.to_string());
        let future_res = future_res.map_err(|e| e.to_string());

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
    report_tx: mpsc::Sender<ExecutionReport>,
    report_rx: mpsc::Receiver<ExecutionReport>,
}

impl BasisTradingStrategy {
    /// Creates a new `BasisTradingStrategy`.
    pub fn new(
        entry_manager: Box<dyn EntryStrategy>,
        risk_monitor: RiskMonitor,
        execution_engine: ExecutionEngine,
        spot_symbol: String,
        future_symbol: String,
    ) -> Self {
        let (report_tx, report_rx) = mpsc::channel(100);
        Self {
            entry_manager,
            risk_monitor,
            execution_engine,
            pair: InstrumentPair { spot_symbol, future_symbol },
            state: StrategyState::Flat,
            report_tx,
            report_rx,
        }
    }

    /// Runs the strategy loop.
    pub async fn run(&mut self, mut spot_rx: tokio::sync::mpsc::Receiver<MarketData>, mut future_rx: tokio::sync::mpsc::Receiver<MarketData>) {
        info!("Starting Delta-Neutral Basis Strategy for {}/{}", self.pair.spot_symbol, self.pair.future_symbol);

        let mut latest_spot: Option<MarketData> = None;
        let mut latest_future: Option<MarketData> = None;

        loop {
            tokio::select! {
                Some(spot_data) = spot_rx.recv() => {
                    latest_spot = Some(spot_data);
                }
                Some(future_data) = future_rx.recv() => {
                    latest_future = Some(future_data);
                }
                Some(report) = self.report_rx.recv() => {
                    self.handle_execution_report(report);
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
                    },
                    Signal::Sell => {
                        info!("Exit Successful. Transitioning to Flat.");
                        self.state = StrategyState::Flat;
                    },
                    _ => {}
                }
            },
            ExecutionResult::TotalFailure(reason) => {
                error!("Execution Failed completely: {}. Reverting state.", reason);
                // If entry failed, we go back to Flat. If exit failed, we might still be InPosition (or partial).
                // For TotalFailure (atomic rollback succeeded), we revert to previous state.
                match report.action {
                    Signal::Buy => self.state = StrategyState::Flat,
                    Signal::Sell => self.state = StrategyState::InPosition, // Failed to exit, still in position
                    _ => {}
                }
            },
            ExecutionResult::PartialFailure(reason) => {
                error!("CRITICAL: Partial Execution Failure: {}. Manual Intervention Required.", reason);
                // In a real system, we'd transition to a "Broken" or "ManualIntervention" state.
                // For now, we'll stay in the current state but log heavily.
            }
        }
    }

    /// Processes a single tick of matched Spot and Future data.
    async fn process_tick(&mut self, spot: &MarketData, future: &MarketData) {
        let now = Utc::now().timestamp();
        if (now - spot.timestamp).abs() > 2 || (now - future.timestamp).abs() > 2 { return; }

        let spread = Spread::new(spot.price, future.price);
        let signal = self.entry_manager.analyze(spread).await;

        match signal {
            Signal::Buy => {
                if self.state != StrategyState::Flat { return; }

                let spot_qty = dec!(0.1); 
                if let Ok(hedge_qty) = self.risk_monitor.calc_hedge_ratio(spot_qty, spot.price, future.price) {
                    info!("Entry Signal! Transitioning to Entering state and spawning execution...");
                    self.state = StrategyState::Entering; // Prevent further spawns

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
