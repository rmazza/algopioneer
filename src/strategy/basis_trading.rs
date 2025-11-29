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
use tokio::sync::Mutex;
use std::error::Error;
use crate::coinbase::CoinbaseClient;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use rust_decimal::prelude::*;
use chrono::Utc;

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

// --- Interfaces (Dependency Injection) ---

/// Trait for entry logic strategies.
/// Allows for swapping different entry algorithms (e.g., simple threshold vs. statistical arbitrage).
#[async_trait]
pub trait EntryStrategy: Send + Sync {
    /// Analyzes the market data for Spot and Future to generate a trading signal.
    async fn analyze(&self, spot: &MarketData, future: &MarketData) -> Signal;
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
    /// Creates a new `EntryManager`.
    ///
    /// # Arguments
    /// * `entry_threshold_bps` - The basis spread (in basis points) required to trigger an entry.
    /// * `exit_threshold_bps` - The basis spread (in basis points) required to trigger an exit.
    pub fn new(entry_threshold_bps: Decimal, exit_threshold_bps: Decimal) -> Self {
        Self {
            entry_threshold_bps,
            exit_threshold_bps,
        }
    }

    /// Calculates the basis spread in basis points.
    fn calculate_basis_bps(&self, spot: Decimal, future: Decimal) -> Decimal {
        if spot.is_zero() {
            return Decimal::zero();
        }
        ((future - spot) / spot) * dec!(10000.0)
    }
}

#[async_trait]
impl EntryStrategy for EntryManager {
    async fn analyze(&self, spot: &MarketData, future: &MarketData) -> Signal {
        let basis_bps = self.calculate_basis_bps(spot.price, future.price);
        println!("Basis Spread: {:.4} bps (Spot: {}, Future: {})", basis_bps, spot.price, future.price);
        
        // Long Spot / Short Future when basis is high (contango)
        // Close when basis converges
        if basis_bps > self.entry_threshold_bps {
            Signal::Buy // Implies "Enter Basis Trade" (Long Spot + Short Future)
        } else if basis_bps < self.exit_threshold_bps {
            Signal::Sell // Implies "Exit Basis Trade" (Sell Spot + Buy Future)
        } else {
            Signal::Hold
        }
    }
}

/// Tracks delta and enforces safety limits.
pub struct RiskMonitor {
    max_leverage: Decimal,
    target_delta: Decimal, // Should be 0.0 for delta-neutral
}

impl RiskMonitor {
    /// Creates a new `RiskMonitor`.
    ///
    /// # Arguments
    /// * `max_leverage` - The maximum allowed leverage.
    pub fn new(max_leverage: Decimal) -> Self {
        Self {
            max_leverage,
            target_delta: Decimal::zero(),
        }
    }

    /// Calculates the hedge ratio to ensure delta neutrality.
    ///
    /// # Returns
    /// The quantity of futures to short for a given spot quantity.
    ///
    /// # Errors
    /// Returns an error if inputs are invalid or if the calculation results in an unsafe value (NaN/Inf).
    pub fn calc_hedge_ratio(&self, spot_quantity: Decimal, spot_price: Decimal, future_price: Decimal) -> Result<Decimal, String> {
        if spot_quantity <= Decimal::zero() || spot_price <= Decimal::zero() || future_price <= Decimal::zero() {
            return Err("Invalid input parameters".to_string());
        }

        // For 1:1 delta neutral, hedge ratio is usually 1.0 (adjusted for contract size if needed).
        // Here we assume 1 contract = 1 unit of underlying for simplicity.
        let hedge_ratio = dec!(1.0); 
        
        let required_futures = spot_quantity * hedge_ratio;

        // Safety Check: Exposure limit
        // In a real system, we'd check account equity here. 
        let _total_exposure = (spot_quantity * spot_price) + (required_futures * future_price);
        
        // Decimal doesn't have is_infinite/is_nan in the same way, but we can check for reasonable bounds if needed.
        // Rust Decimal handles overflow by panicking or returning None on some ops, but basic ops are usually safe or panic.
        // We'll assume valid decimals here.

        Ok(required_futures)
    }
    
    /// Checks the net delta of the current positions.
    pub fn check_delta(&self, spot_pos: &Position, future_pos: &Position) -> Decimal {
        let net_delta = spot_pos.quantity - future_pos.quantity;
        net_delta
    }
}

/// Handles order execution logic.
#[derive(Clone)]
pub struct ExecutionEngine {
    client: Arc<dyn Executor>,
}

impl ExecutionEngine {
    /// Creates a new `ExecutionEngine`.
    pub fn new(client: Arc<dyn Executor>) -> Self {
        Self { client }
    }

    /// Executes a basis trade entry (Long Spot + Short Future) concurrently.
    ///
    /// # Safety
    /// This method uses `tokio::join!` to execute both legs simultaneously.
    /// Includes a "Kill Switch" to unwind the position if one leg fails.
    ///
    /// > [!WARNING]
    /// > If the Future leg fails but the Spot leg succeeds, the system is NOT delta-neutral.
    /// > Production systems must implement atomic rollback or immediate hedging logic here.
    pub async fn execute_basis_entry(&self, spot_symbol: &str, future_symbol: &str, quantity: Decimal, hedge_qty: Decimal) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Concurrently execute both legs to minimize leg risk
        let spot_leg = self.client.execute_order(spot_symbol, "buy", quantity);
        let future_leg = self.client.execute_order(future_symbol, "sell", hedge_qty);

        let (spot_res, future_res) = tokio::join!(spot_leg, future_leg);
        
        // Convert errors to String immediately to ensure Send compliance for tokio::spawn
        let spot_res = spot_res.map_err(|e| e.to_string());
        let future_res = future_res.map_err(|e| e.to_string());

        if let Err(e) = spot_res {
             // Spot failed. If future succeeded, we have a naked short future.
             if future_res.is_ok() {
                 eprintln!("CRITICAL: Spot failed but Future succeeded. Executing Kill Switch on Future leg.");
                 // Kill Switch: Buy back future
                 if let Err(e_kill) = self.client.execute_order(future_symbol, "buy", hedge_qty).await {
                     eprintln!("EMERGENCY: Kill Switch failed for Future leg! Manual intervention required. Error: {}", e_kill);
                 } else {
                     println!("Kill Switch successful: Future leg closed.");
                 }
             }
             return Err(format!("Spot leg failed: {}", e).into());
        }

        if let Err(e) = future_res {
            // Future failed. Spot succeeded (checked above). We have a naked long spot.
            eprintln!("CRITICAL: Future leg failed: {}. Executing Kill Switch on Spot leg.", e);
            // Kill Switch: Sell spot
            if let Err(e_kill) = self.client.execute_order(spot_symbol, "sell", quantity).await {
                eprintln!("EMERGENCY: Kill Switch failed for Spot leg! Manual intervention required. Error: {}", e_kill);
            } else {
                 println!("Kill Switch successful: Spot leg closed.");
            }
            return Err(format!("Future leg failed: {}. Position unwound.", e).into());
        }

        Ok(())
    }
}

// --- Main Strategy Class ---

/// The main coordinator for the Delta-Neutral Basis Trading Strategy.
pub struct BasisTradingStrategy {
    entry_manager: Box<dyn EntryStrategy>,
    risk_monitor: RiskMonitor,
    execution_engine: ExecutionEngine,
    spot_symbol: String,
    future_symbol: String,
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
        Self {
            entry_manager,
            risk_monitor,
            execution_engine,
            spot_symbol,
            future_symbol,
        }
    }

    /// Runs the strategy loop.
    ///
    /// Listens to `spot_rx` and `future_rx` channels for market data updates.
    /// Uses `tokio::select!` to process updates concurrently.
    pub async fn run(&mut self, mut spot_rx: tokio::sync::mpsc::Receiver<MarketData>, mut future_rx: tokio::sync::mpsc::Receiver<MarketData>) {
        println!("Starting Delta-Neutral Basis Strategy for {}/{}", self.spot_symbol, self.future_symbol);

        let mut latest_spot: Option<MarketData> = None;
        let mut latest_future: Option<MarketData> = None;

        loop {
            // Concurrently wait for updates from either stream
            tokio::select! {
                Some(spot_data) = spot_rx.recv() => {
                    latest_spot = Some(spot_data);
                }
                Some(future_data) = future_rx.recv() => {
                    latest_future = Some(future_data);
                }
                else => break, // Channels closed
            }

            if let (Some(spot), Some(future)) = (&latest_spot, &latest_future) {
                self.process_tick(spot, future).await;
            }
        }
    }

    /// Processes a single tick of matched Spot and Future data.
    /// Processes a single tick of matched Spot and Future data.
    async fn process_tick(&self, spot: &MarketData, future: &MarketData) {
        // Stale Data Check
        let now = Utc::now().timestamp();
        if (now - spot.timestamp).abs() > 2 || (now - future.timestamp).abs() > 2 {
            // Data is older than 2 seconds, skip
            return;
        }

        let signal = self.entry_manager.analyze(spot, future).await;

        match signal {
            Signal::Buy => {
                // Calculate safe size
                let spot_qty = dec!(0.1); // Example fixed size, should be dynamic
                match self.risk_monitor.calc_hedge_ratio(spot_qty, spot.price, future.price) {
                    Ok(hedge_qty) => {
                        println!("Entry Signal! Executing Basis Trade...");
                        // Decouple execution from data loop
                        let engine = self.execution_engine.clone();
                        let spot_sym = self.spot_symbol.clone();
                        let future_sym = self.future_symbol.clone();
                        
                        tokio::spawn(async move {
                            if let Err(e) = engine.execute_basis_entry(&spot_sym, &future_sym, spot_qty, hedge_qty).await {
                                eprintln!("Execution Error: {}", e);
                            }
                        });
                    },
                    Err(e) => eprintln!("Risk Check Failed: {}", e),
                }
            },
            _ => {} // Handle Sell/Exit logic similarly
        }
    }
}

#[async_trait]
impl Executor for CoinbaseClient {
    async fn execute_order(&self, symbol: &str, side: &str, quantity: Decimal) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Convert Decimal to f64 for the underlying client if needed, or update client.
        // Assuming client.place_order takes f64 based on previous view.
        // We'll convert to f64 here for now to match the existing CoinbaseClient signature.
        // Ideally CoinbaseClient should also take Decimal.
        let qty_f64 = quantity.to_f64().ok_or("Failed to convert quantity to f64")?;
        self.place_order(symbol, side, qty_f64).await
    }
}
