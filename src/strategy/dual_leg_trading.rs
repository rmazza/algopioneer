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

use crate::coinbase::CoinbaseClient;
use crate::strategy::Signal;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::prelude::*;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::time::Duration;
use tracing::{debug, error, info, instrument, warn};

// Re-export shared types for backward compatibility
pub use crate::types::{MarketData, OrderSide};
// Re-export Executor from exchange for backward compatibility
pub use crate::exchange::Executor;

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
    #[error("Circuit Breaker Open: {0}")]
    CircuitBreakerOpen(String),
}

impl ExecutionError {
    pub fn from_boxed(e: Box<dyn Error + Send + Sync>) -> Self {
        let s = e.to_string();
        if s.to_lowercase().contains("network") || s.to_lowercase().contains("timeout") {
            ExecutionError::NetworkError(s)
        } else if s.to_lowercase().contains("insufficient funds")
            || s.to_lowercase().contains("rejected")
        {
            ExecutionError::OrderRejected(s)
        } else {
            ExecutionError::ExchangeError(s)
        }
    }
}

/// Conversion from ExchangeError to ExecutionError for seamless integration
impl From<crate::exchange::ExchangeError> for ExecutionError {
    fn from(e: crate::exchange::ExchangeError) -> Self {
        use crate::exchange::ExchangeError as EE;
        match e {
            EE::Network(s) => ExecutionError::NetworkError(s),
            EE::RateLimited(ms) => {
                ExecutionError::NetworkError(format!("Rate limited, retry after {}ms", ms))
            }
            EE::OrderRejected(s) => ExecutionError::OrderRejected(s),
            EE::ExchangeInternal(s) => ExecutionError::ExchangeError(s),
            EE::Configuration(s) => ExecutionError::CriticalFailure(s),
            EE::Other(s) => ExecutionError::Unknown(s),
        }
    }
}

// Recovery constants moved to execution.rs

// P1 FIX: Load Shedding Threshold
// Maximum allowed latency (in milliseconds) before a tick is considered stale.
// During high-volatility events or WebSocket floods, processing stale ticks
// leads to execution on prices that no longer exist, causing massive slippage.
// This check is on the hot path and uses simple integer arithmetic.
const MAX_ALLOWED_LATENCY_MS: i64 = 100;

// TASK 4: Ghost Position Recovery Marker
// Used to indicate that an entry price is unknown after state recovery.
// When process_tick sees this value, it will reset to current market price.
// Using NEGATIVE_ONE as a marker since prices cannot be negative.
const UNKNOWN_ENTRY_PRICE: Decimal = dec!(-1.0);

// --- Logging Utilities ---
// LogThrottle and DualLegLogThrottler moved to throttle.rs
pub use crate::strategy::throttle::{DualLegLogThrottler, LogThrottle};

// --- Financial Models ---

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TransactionCostModel {
    pub maker_fee_bps: Decimal,
    pub taker_fee_bps: Decimal,
    pub slippage_bps: Decimal,
}

impl TransactionCostModel {
    pub fn new(maker_fee_bps: Decimal, taker_fee_bps: Decimal, slippage_bps: Decimal) -> Self {
        Self {
            maker_fee_bps,
            taker_fee_bps,
            slippage_bps,
        }
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
// MarketData is now imported from crate::types

// --- AS2: Market Data Validation ---
// Validators moved to validators.rs
pub use crate::strategy::validators::{
    AgeValidator, CompositeValidator, PriceValidator, TickValidator,
};

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
        Self {
            value_bps,
            spot_price: spot,
            future_price: future,
        }
    }
}

// --- State Management ---

// Exit policies moved to exit_policy.rs
pub use crate::strategy::exit_policy::{
    CompositeExitPolicy, ExitPolicy, MinimumProfitPolicy, PnlExitPolicy, StopLossPolicy,
};

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

/// AS1: Builder for DualLegConfig with sensible defaults and validation.
/// Provides a fluent API for constructing strategy configurations consistently.
#[derive(Debug, Clone)]
pub struct DualLegConfigBuilder {
    spot_symbol: Option<String>,
    future_symbol: Option<String>,
    order_size: Decimal,
    max_tick_age_ms: i64,
    execution_timeout_ms: i64,
    min_profit_threshold: Decimal,
    stop_loss_threshold: Decimal,
    fee_tier: TransactionCostModel,
    throttle_interval_secs: u64,
}

impl Default for DualLegConfigBuilder {
    fn default() -> Self {
        Self {
            spot_symbol: None,
            future_symbol: None,
            order_size: dec!(0.001),
            max_tick_age_ms: 2000,
            execution_timeout_ms: 30000,
            min_profit_threshold: dec!(0.005),
            stop_loss_threshold: dec!(-0.05),
            fee_tier: TransactionCostModel::default(),
            throttle_interval_secs: 5,
        }
    }
}

impl DualLegConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the spot symbol (required)
    pub fn spot_symbol(mut self, symbol: impl Into<String>) -> Self {
        self.spot_symbol = Some(symbol.into());
        self
    }

    /// Set the future symbol (required)
    pub fn future_symbol(mut self, symbol: impl Into<String>) -> Self {
        self.future_symbol = Some(symbol.into());
        self
    }

    /// Set the order size in base currency
    pub fn order_size(mut self, size: Decimal) -> Self {
        self.order_size = size;
        self
    }

    /// Set the maximum tick age in milliseconds
    pub fn max_tick_age_ms(mut self, age_ms: i64) -> Self {
        self.max_tick_age_ms = age_ms;
        self
    }

    /// Set the execution timeout in milliseconds
    pub fn execution_timeout_ms(mut self, timeout_ms: i64) -> Self {
        self.execution_timeout_ms = timeout_ms;
        self
    }

    /// Set the minimum profit threshold for exits
    pub fn min_profit_threshold(mut self, threshold: Decimal) -> Self {
        self.min_profit_threshold = threshold;
        self
    }

    /// Set the stop loss threshold (should be negative)
    pub fn stop_loss_threshold(mut self, threshold: Decimal) -> Self {
        self.stop_loss_threshold = threshold;
        self
    }

    /// Set the fee tier model
    pub fn fee_tier(mut self, fee_tier: TransactionCostModel) -> Self {
        self.fee_tier = fee_tier;
        self
    }

    /// Set the log throttle interval in seconds
    pub fn throttle_interval_secs(mut self, secs: u64) -> Self {
        self.throttle_interval_secs = secs;
        self
    }

    /// Build and validate the configuration.
    /// Returns Err if required fields are missing or validation fails.
    pub fn build(self) -> Result<DualLegConfig, String> {
        let spot_symbol = self
            .spot_symbol
            .ok_or_else(|| "spot_symbol is required".to_string())?;
        let future_symbol = self
            .future_symbol
            .ok_or_else(|| "future_symbol is required".to_string())?;

        // Validate required fields
        if spot_symbol.is_empty() {
            return Err("spot_symbol cannot be empty".to_string());
        }
        if future_symbol.is_empty() {
            return Err("future_symbol cannot be empty".to_string());
        }
        if spot_symbol == future_symbol {
            return Err(format!(
                "spot_symbol and future_symbol cannot be the same: {}",
                spot_symbol
            ));
        }

        // Validate numeric fields
        if self.order_size <= Decimal::ZERO {
            return Err(format!(
                "order_size must be positive, got: {}",
                self.order_size
            ));
        }
        if self.max_tick_age_ms <= 0 {
            return Err(format!(
                "max_tick_age_ms must be positive, got: {}",
                self.max_tick_age_ms
            ));
        }
        if self.execution_timeout_ms <= 0 {
            return Err(format!(
                "execution_timeout_ms must be positive, got: {}",
                self.execution_timeout_ms
            ));
        }

        Ok(DualLegConfig {
            spot_symbol,
            future_symbol,
            order_size: self.order_size,
            max_tick_age_ms: self.max_tick_age_ms,
            execution_timeout_ms: self.execution_timeout_ms,
            min_profit_threshold: self.min_profit_threshold,
            stop_loss_threshold: self.stop_loss_threshold,
            fee_tier: self.fee_tier,
            throttle_interval_secs: self.throttle_interval_secs,
        })
    }
}

// AS-1 FIX: Explicit position direction tracking
/// Direction of the position (Long or Short)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionDirection {
    /// Long position: Bought leg1, Sold leg2
    Long,
    /// Short position: Sold leg1, Bought leg2
    Short,
}

impl std::fmt::Display for PositionDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PositionDirection::Long => write!(f, "Long"),
            PositionDirection::Short => write!(f, "Short"),
        }
    }
}

#[must_use]
#[derive(Debug, Clone, PartialEq)]
pub enum StrategyState {
    Flat,
    Entering {
        direction: PositionDirection, // AS-1: Track entry direction
        leg1_qty: Decimal,
        leg2_qty: Decimal,
        leg1_entry_price: Decimal,
        leg2_entry_price: Decimal,
    },
    InPosition {
        direction: PositionDirection, // AS-1: Track position direction
        leg1_qty: Decimal,
        leg2_qty: Decimal,
        leg1_entry_price: Decimal,
        leg2_entry_price: Decimal,
    },
    Exiting {
        direction: PositionDirection, // AS-1: Track exit direction (for proper order sides)
        leg1_qty: Decimal,
        leg2_qty: Decimal,
        leg1_entry_price: Decimal,
        leg2_entry_price: Decimal,
    },
    Reconciling, // New State: "I don't know what happened, let me check."
    Halted,      // New State: "Something is broken, human needed."
}

// N8: Display impl for better logging
impl std::fmt::Display for StrategyState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StrategyState::Flat => write!(f, "Flat"),
            StrategyState::Entering { direction, .. } => write!(f, "Entering({})", direction),
            StrategyState::InPosition { direction, .. } => write!(f, "InPosition({})", direction),
            StrategyState::Exiting { direction, .. } => write!(f, "Exiting({})", direction),
            StrategyState::Halted => write!(f, "Halted"),
            StrategyState::Reconciling => write!(f, "Reconciling"),
        }
    }
}

#[must_use]
#[derive(Debug, Clone)]
pub enum RecoveryResult {
    Success(String), // Symbol
    Failed(String),  // Symbol
}

#[must_use]
#[derive(Debug, Clone)]
pub enum ExecutionResult {
    Success,
    PartialFailure(ExecutionError), // e.g., "Spot filled, Future failed"
    TotalFailure(ExecutionError),
}

#[derive(Debug)]
pub struct ExecutionReport {
    pub result: ExecutionResult,
    pub action: Signal,             // Buy (Entry) or Sell (Exit)
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

// Entry managers moved to entry_manager.rs
pub use crate::strategy::entry_manager::{BasisManager, EntryStrategy, PairsManager};

// Executor trait is now imported from crate::exchange

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
    pub fn new(
        max_leverage: Decimal,
        instrument_type: InstrumentType,
        hedge_mode: HedgeMode,
    ) -> Self {
        Self {
            _max_leverage: max_leverage,
            hedge_mode,
            instrument_type,
        }
    }

    pub fn calc_hedge_ratio(
        &self,
        leg1_qty: Decimal,
        leg1_price: Decimal,
        leg2_price: Decimal,
    ) -> Result<Decimal, DualLegError> {
        if leg1_qty <= Decimal::zero() {
            return Err(DualLegError::InvalidInput(
                "Leg 1 quantity must be positive".to_string(),
            ));
        }

        // TASK 4: Safety check for invalid/unknown prices
        // Reject negative prices (including UNKNOWN_ENTRY_PRICE marker) and zero prices
        if leg1_price <= Decimal::ZERO {
            return Err(DualLegError::InvalidInput(format!(
                "Leg 1 price must be positive, got: {}",
                leg1_price
            )));
        }
        if leg2_price <= Decimal::ZERO {
            return Err(DualLegError::InvalidInput(format!(
                "Leg 2 price must be positive, got: {}",
                leg2_price
            )));
        }

        match self.hedge_mode {
            HedgeMode::DeltaNeutral => match self.instrument_type {
                InstrumentType::Linear => Ok(leg1_qty),
                InstrumentType::Inverse => {
                    if leg2_price.is_zero() {
                        return Err(DualLegError::MathError(
                            "Leg 2 price cannot be zero for inverse calculation".to_string(),
                        ));
                    }
                    Ok(leg1_qty * leg2_price)
                }
            },
            HedgeMode::DollarNeutral => {
                // Qty1 * Price1 = Qty2 * Price2
                // Qty2 = (Qty1 * Price1) / Price2
                if leg2_price.is_zero() {
                    return Err(DualLegError::MathError(
                        "Leg 2 price cannot be zero for dollar neutral calculation".to_string(),
                    ));
                }
                Ok((leg1_qty * leg1_price) / leg2_price)
            }
        }
    }
}

// Execution engine and recovery worker moved to execution.rs
pub use crate::strategy::execution::{
    perform_recovery_with_backoff, ExecutionEngine, RecoveryWorker, TaskPriority,
};

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
    // AS2: Tick validator for market data validation
    validator: Box<dyn TickValidator>,
    // AS9: Exit policy for flexible exit logic
    exit_policy: Box<dyn ExitPolicy>,
    // CB-2 FIX: Track spawned execution tasks for structured concurrency
    active_executions: tokio::task::JoinSet<()>,
    // P-MC-3 FIX: Pre-computed pair name for metrics (avoids format! per-call)
    pair_name: String,
    // STATE PERSISTENCE: Optional state store for surviving restarts
    state_store: Option<std::sync::Arc<dyn crate::logging::StateStore>>,
    // STATE PERSISTENCE: Position ID for state storage (e.g., "pairs:AAPL:MSFT")
    position_id: Option<String>,
    // STATE PERSISTENCE: Strategy type for state records
    strategy_type: String,
    // STATE PERSISTENCE: Paper trading mode flag
    is_paper: bool,
    // MARKET HOURS: Last time we checked market status
    last_market_check: i64,
    // MARKET HOURS: Current market status (true=open, false=closed)
    is_market_open: bool,
    // SAFETY GUARD: Maximum position value per symbol in USD
    max_position_usd: Option<Decimal>,
    // SAFETY GUARD: Maximum allowed imbalance ratio between legs
    max_imbalance_ratio: Decimal,
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
            future_symbol: config.future_symbol.clone(),
        };

        // AS2: Create default validator (age + price validation)
        let validator = Box::new(CompositeValidator::new(vec![
            Box::new(AgeValidator::new(config.max_tick_age_ms)),
            Box::new(PriceValidator),
        ]));

        // AS9: Create default exit policy (Pnl Based)
        let exit_policy = Box::new(PnlExitPolicy::new(
            config.min_profit_threshold,
            config.stop_loss_threshold,
        ));

        // P-MC-3 FIX: Pre-compute pair name once at construction
        let pair_name = format!("{}/{}", config.spot_symbol, config.future_symbol);

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
            validator,                                      // AS2
            exit_policy,                                    // AS9
            active_executions: tokio::task::JoinSet::new(), // CB-2
            pair_name,                                      // P-MC-3
            state_store: None,                              // STATE PERSISTENCE
            position_id: None,                              // STATE PERSISTENCE
            strategy_type: "pairs".to_string(),             // STATE PERSISTENCE (default)
            is_paper: false,                                // STATE PERSISTENCE (default)
            last_market_check: 0,                           // MARKET HOURS
            is_market_open: true,                           // MARKET HOURS (assume open initially)
            max_position_usd: None,                         // SAFETY GUARD (set via builder)
            max_imbalance_ratio: dec!(0.5),                 // SAFETY GUARD (default 50%)
        }
    }

    /// Configure optional state persistence for surviving container restarts.
    ///
    /// When a `StateStore` is provided, the strategy will:
    /// 1. Save state to storage on every state transition
    /// 2. Attempt to load prior state on startup (before reconcile_state)
    ///
    /// # Arguments
    /// * `store` - StateStore implementation (e.g., DynamoDbRecorder)
    /// * `strategy_type` - Type of strategy: "pairs" or "basis"
    /// * `is_paper` - Whether this is paper trading
    #[must_use]
    pub fn with_state_store(
        mut self,
        store: std::sync::Arc<dyn crate::logging::StateStore>,
        strategy_type: impl Into<String>,
        is_paper: bool,
    ) -> Self {
        let position_id = format!(
            "{}:{}:{}",
            strategy_type.into(),
            self.config.spot_symbol,
            self.config.future_symbol
        );
        self.state_store = Some(store);
        self.position_id = Some(position_id.clone());
        self.strategy_type = position_id.split(':').next().unwrap_or("pairs").to_string();
        self.is_paper = is_paper;
        self
    }

    /// Configure safety guards to prevent unhedged position accumulation.
    ///
    /// # Arguments
    /// * `max_position_usd` - Maximum USD value per symbol before blocking new entries
    /// * `max_imbalance_ratio` - Maximum allowed imbalance ratio before halting on reconcile
    #[must_use]
    pub fn with_safety_guards(
        mut self,
        max_position_usd: Option<Decimal>,
        max_imbalance_ratio: Decimal,
    ) -> Self {
        self.max_position_usd = max_position_usd;
        self.max_imbalance_ratio = max_imbalance_ratio;
        self
    }

    pub fn set_observer(&mut self, tx: mpsc::Sender<StrategyState>) {
        self.state_notifier = Some(tx);
    }

    fn transition_state(&mut self, new_state: StrategyState) {
        if self.state != new_state {
            info!("State Transition: {:?} -> {:?}", self.state, new_state);
            self.state = new_state.clone();
            self.last_state_change_ts = self.clock.now_ts_millis();

            // MC-2 FIX: Emit metric for Halted state to enable alerting
            // P-MC-3 FIX: Use pre-computed pair_name
            if new_state == StrategyState::Halted {
                crate::metrics::record_strategy_halted("dual_leg", &self.pair_name);
                error!(
                    pair = %self.pair_name,
                    "CRITICAL: Strategy entered Halted state - manual intervention required"
                );
            }

            if let Some(tx) = &self.state_notifier {
                if let Err(e) = tx.try_send(new_state) {
                    warn!("State update dropped due to backpressure: {}", e);
                }
            }

            // STATE PERSISTENCE: Persist state to durable storage (async, best-effort)
            self.persist_state_async();
        }
    }

    /// STATE PERSISTENCE: Convert current state to a PositionStateRecord for storage.
    fn to_position_state_record(&self) -> Option<crate::logging::PositionStateRecord> {
        let position_id = self.position_id.as_ref()?;

        let (state_str, direction, leg1_qty, leg2_qty, leg1_entry_price, leg2_entry_price) =
            match &self.state {
                StrategyState::Flat => (
                    "flat".to_string(),
                    None,
                    Decimal::ZERO,
                    Decimal::ZERO,
                    Decimal::ZERO,
                    Decimal::ZERO,
                ),
                StrategyState::Entering {
                    direction,
                    leg1_qty,
                    leg2_qty,
                    leg1_entry_price,
                    leg2_entry_price,
                } => (
                    "entering".to_string(),
                    Some(direction.to_string().to_lowercase()),
                    *leg1_qty,
                    *leg2_qty,
                    *leg1_entry_price,
                    *leg2_entry_price,
                ),
                StrategyState::InPosition {
                    direction,
                    leg1_qty,
                    leg2_qty,
                    leg1_entry_price,
                    leg2_entry_price,
                } => (
                    "in_position".to_string(),
                    Some(direction.to_string().to_lowercase()),
                    *leg1_qty,
                    *leg2_qty,
                    *leg1_entry_price,
                    *leg2_entry_price,
                ),
                StrategyState::Exiting {
                    direction,
                    leg1_qty,
                    leg2_qty,
                    leg1_entry_price,
                    leg2_entry_price,
                } => (
                    "exiting".to_string(),
                    Some(direction.to_string().to_lowercase()),
                    *leg1_qty,
                    *leg2_qty,
                    *leg1_entry_price,
                    *leg2_entry_price,
                ),
                StrategyState::Reconciling => (
                    "reconciling".to_string(),
                    None,
                    Decimal::ZERO,
                    Decimal::ZERO,
                    Decimal::ZERO,
                    Decimal::ZERO,
                ),
                StrategyState::Halted => (
                    "halted".to_string(),
                    None,
                    Decimal::ZERO,
                    Decimal::ZERO,
                    Decimal::ZERO,
                    Decimal::ZERO,
                ),
            };

        Some(crate::logging::PositionStateRecord {
            position_id: position_id.clone(),
            strategy_type: self.strategy_type.clone(),
            state: state_str,
            direction,
            leg1_symbol: self.config.spot_symbol.clone(),
            leg2_symbol: self.config.future_symbol.clone(),
            leg1_qty: leg1_qty.to_string(),
            leg2_qty: leg2_qty.to_string(),
            leg1_entry_price: leg1_entry_price.to_string(),
            leg2_entry_price: leg2_entry_price.to_string(),
            updated_at: self.clock.now(),
            is_paper: self.is_paper,
        })
    }

    /// STATE PERSISTENCE: Persist current state asynchronously (fire-and-forget).
    /// Errors are logged but don't block the strategy.
    fn persist_state_async(&self) {
        if let (Some(store), Some(record)) =
            (self.state_store.clone(), self.to_position_state_record())
        {
            tokio::spawn(async move {
                if let Err(e) = store.save_state(&record).await {
                    // Log error but don't fail - state persistence is best-effort
                    warn!(error = %e, "Failed to persist strategy state");
                }
            });
        }
    }

    /// STATE PERSISTENCE: Load prior state from storage.
    /// Returns true if state was restored, false otherwise.
    async fn load_persisted_state(&mut self) -> bool {
        let Some(store) = &self.state_store else {
            return false;
        };
        let Some(position_id) = &self.position_id else {
            return false;
        };

        match store.load_state(position_id).await {
            Ok(Some(record)) => {
                info!(
                    position_id = %record.position_id,
                    saved_state = %record.state,
                    direction = ?record.direction,
                    "Loaded persisted state from storage"
                );

                // Parse quantities and prices from strings
                let leg1_qty = record.leg1_qty.parse::<Decimal>().unwrap_or(Decimal::ZERO);
                let leg2_qty = record.leg2_qty.parse::<Decimal>().unwrap_or(Decimal::ZERO);
                let leg1_entry_price = record
                    .leg1_entry_price
                    .parse::<Decimal>()
                    .unwrap_or(UNKNOWN_ENTRY_PRICE);
                let leg2_entry_price = record
                    .leg2_entry_price
                    .parse::<Decimal>()
                    .unwrap_or(UNKNOWN_ENTRY_PRICE);

                // Parse direction
                let direction = match record.direction.as_deref() {
                    Some("long") => PositionDirection::Long,
                    Some("short") => PositionDirection::Short,
                    _ => PositionDirection::Long, // Default
                };

                // Restore state based on saved state string
                match record.state.as_str() {
                    "in_position" => {
                        if leg1_qty > Decimal::ZERO || leg2_qty > Decimal::ZERO {
                            self.state = StrategyState::InPosition {
                                direction,
                                leg1_qty,
                                leg2_qty,
                                leg1_entry_price,
                                leg2_entry_price,
                            };
                            info!(
                                leg1_qty = %leg1_qty,
                                leg2_qty = %leg2_qty,
                                leg1_entry_price = %leg1_entry_price,
                                leg2_entry_price = %leg2_entry_price,
                                "Restored InPosition state with entry prices from storage"
                            );
                            return true;
                        }
                    }
                    "entering" | "exiting" => {
                        // For transitional states, just restore InPosition and let reconcile handle it
                        if leg1_qty > Decimal::ZERO || leg2_qty > Decimal::ZERO {
                            self.state = StrategyState::InPosition {
                                direction,
                                leg1_qty,
                                leg2_qty,
                                leg1_entry_price,
                                leg2_entry_price,
                            };
                            warn!(
                                saved_state = %record.state,
                                "Restored transitional state as InPosition - will reconcile with exchange"
                            );
                            return true;
                        }
                    }
                    _ => {
                        // Keep Flat state for "flat", "reconciling", "halted", or unknown
                        debug!("Persisted state was Flat or terminal, starting fresh");
                    }
                }
                false
            }
            Ok(None) => {
                debug!("No persisted state found, starting fresh");
                false
            }
            Err(e) => {
                warn!(error = %e, "Failed to load persisted state, starting fresh");
                false
            }
        }
    }

    /// STATE AMNESIA FIX: Reconcile strategy state with actual exchange positions.
    /// Queries exchange for current holdings and updates state accordingly.
    /// Should be called at the start of run() before processing any ticks.
    ///
    /// ## Task 4 (Ghost Position) Fix:
    /// When recovering a position without historical fill data, we use a marker value
    /// (`UNKNOWN_ENTRY_PRICE`) instead of zero. On the first tick, `process_tick`
    /// detects this marker and resets entry prices to current market prices,
    /// effectively starting PnL tracking from zero for the recovered session.
    async fn reconcile_state(&mut self) {
        info!("Reconciling state with exchange positions...");

        // Query positions for both legs
        let leg1_pos = self
            .execution_engine
            .get_position(&self.pair.spot_symbol)
            .await;
        let leg2_pos = self
            .execution_engine
            .get_position(&self.pair.future_symbol)
            .await;

        match (leg1_pos, leg2_pos) {
            (Ok(leg1_qty), Ok(leg2_qty)) => {
                // Threshold for considering a position "open" (to handle dust)
                let threshold = dec!(0.00001);

                if leg1_qty.abs() > threshold || leg2_qty.abs() > threshold {
                    // We have an open position - transition to InPosition

                    // FIXED LOGIC: Infer direction from available legs
                    // If Leg 1 has size, its sign dictates direction (Long Leg1 = Long Strategy)
                    // If Leg 1 is flat, Leg 2's sign dictates direction (Short Leg2 = Long Strategy)
                    let direction = if leg1_qty.abs() > threshold {
                        if leg1_qty >= Decimal::ZERO {
                            PositionDirection::Long
                        } else {
                            PositionDirection::Short
                        }
                    } else {
                        // Leg 1 is flat, infer from Leg 2
                        // Long Strategy = Short Leg 2, so if Leg 2 is Short (neg), we are Long
                        if leg2_qty < Decimal::ZERO {
                            PositionDirection::Long
                        } else {
                            PositionDirection::Short
                        }
                    };

                    info!(
                        "Detected existing position: leg1={}, leg2={}, inferred intent={}. Transitioning to InPosition.",
                        leg1_qty, leg2_qty, direction
                    );

                    // TASK 4 FIX: Use marker value for unknown entry prices
                    // The process_tick method will detect this and reset to current market prices
                    self.state = StrategyState::InPosition {
                        direction,
                        leg1_qty: leg1_qty.abs(), // Store absolute quantities
                        leg2_qty: leg2_qty.abs(),
                        leg1_entry_price: UNKNOWN_ENTRY_PRICE, // Marker for "needs reset"
                        leg2_entry_price: UNKNOWN_ENTRY_PRICE,
                    };
                    self.last_state_change_ts = self.clock.now_ts_millis();

                    warn!(
                        "GHOST POSITION RECOVERY: Entry prices unknown after restart. \
                        PnL will be reset to zero on first market tick. \
                        Manual cost basis check required for accurate accounting."
                    );
                } else {
                    info!("No existing positions detected. Starting in Flat state.");
                    // Already in Flat, no action needed
                }
            }
            (Err(e1), _) => {
                error!(
                    "CRITICAL: Failed to query position for {}: {}. \
                    Transitioning to Halted to prevent trading with unknown state.",
                    self.pair.spot_symbol, e1
                );
                self.state = StrategyState::Halted;
            }
            (_, Err(e2)) => {
                error!(
                    "CRITICAL: Failed to query position for {}: {}. \
                    Transitioning to Halted to prevent trading with unknown state.",
                    self.pair.future_symbol, e2
                );
                self.state = StrategyState::Halted;
            }
        }
    }

    /// Runs the strategy loop.
    #[instrument(skip(self, leg1_rx, leg2_rx), name = "strategy_loop")]
    pub async fn run(
        &mut self,
        mut leg1_rx: tokio::sync::mpsc::Receiver<Arc<MarketData>>,
        mut leg2_rx: tokio::sync::mpsc::Receiver<Arc<MarketData>>,
    ) {
        info!(
            "Starting Dual-Leg Strategy for {}/{}",
            self.pair.spot_symbol, self.pair.future_symbol
        );

        // STATE PERSISTENCE: Try to load prior state from storage first.
        // This restores entry prices from the last session if available.
        let state_restored = self.load_persisted_state().await;

        // STATE AMNESIA FIX: Reconcile position state with exchange.
        // If we restored state from storage, reconcile will validate quantities match.
        // If not, reconcile will detect positions but use UNKNOWN_ENTRY_PRICE.
        if !state_restored {
            self.reconcile_state().await;
        } else {
            // Still verify exchange positions match our restored state
            info!("State restored from storage - verifying with exchange...");
            self.reconcile_state().await;
        }

        let mut latest_leg1: Option<Arc<MarketData>> = None;
        let mut latest_leg2: Option<Arc<MarketData>> = None;
        let mut dirty = false;
        let mut heartbeat = tokio::time::interval(Duration::from_secs(1));
        // Market check interval: 60 seconds
        let market_check_interval_ms = 60_000;

        loop {
            tokio::select! {
                _ = heartbeat.tick() => {
                    self.check_timeout();

                    // Periodic Market Hours Check
                    let now = self.clock.now_ts_millis();
                    if now - self.last_market_check > market_check_interval_ms {
                        self.last_market_check = now;
                        // Don't await the check in the main loop! Spawn a check.
                        // We use a separate task to avoid blocking the heartbeat tick.
                        // We capture `execution_engine` (Arc) and report back via a channel or just rely on next tick?
                        // Actually, simpler: we need to update `self.is_market_open`.
                        // Since `self` is &mut, we can't spawn easily without invalidating references.
                        // FIX: Use timeout to prevent blocking indefinitely, OR accept that 100ms pause is ok?
                        // Let's use a timeout of 2 seconds. If it's slow, we time out and keep old state.

                        let check_future = self.execution_engine.check_market_hours();

                        match tokio::time::timeout(Duration::from_secs(2), check_future).await {
                             Ok(Ok(is_open)) => {
                                if self.is_market_open != is_open {
                                    info!(
                                        "Market Status Changed: Open={} (was {})",
                                        is_open, self.is_market_open
                                    );
                                    self.is_market_open = is_open;
                                }
                             }
                             Ok(Err(e)) => {
                                 warn!("Failed to check market hours: {}", e);
                             }
                             Err(_) => {
                                 warn!("Market hours check timed out after 2s");
                             }
                        }
                    }
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
                        if let StrategyState::Entering {
                            direction,
                            leg1_qty,
                            leg2_qty,
                            leg1_entry_price,
                            leg2_entry_price,
                        } = self.state
                        {
                            self.transition_state(StrategyState::InPosition {
                                direction,
                                leg1_qty,
                                leg2_qty,
                                leg1_entry_price,
                                leg2_entry_price,
                            });
                        } else {
                            error!(
                                "Received Entry Success but state is not Entering! State: {:?}",
                                self.state
                            );
                        }
                    }
                    Signal::Sell => {
                        // CF1 FIX: Short Entry Successful
                        info!("Short Entry Successful. Transitioning to InPosition.");
                        if let StrategyState::Entering {
                            direction,
                            leg1_qty,
                            leg2_qty,
                            leg1_entry_price,
                            leg2_entry_price,
                        } = self.state
                        {
                            self.transition_state(StrategyState::InPosition {
                                direction,
                                leg1_qty,
                                leg2_qty,
                                leg1_entry_price,
                                leg2_entry_price,
                            });
                        } else {
                            error!("Received Short Entry Success but state is not Entering! State: {:?}", self.state);
                        }
                    }
                    Signal::Exit => {
                        // CF2 FIX: Exit Successful
                        info!("Exit Successful. Transitioning to Flat.");
                        self.transition_state(StrategyState::Flat);
                    }
                    _ => {}
                }
            }
            ExecutionResult::TotalFailure(reason) => {
                error!("Execution Failed completely: {}. Reverting state.", reason);
                if let ExecutionError::CriticalFailure(_) = reason {
                    error!("CRITICAL FAILURE DETECTED. Halting Strategy.");
                    self.transition_state(StrategyState::Halted);
                    return;
                }
                match report.action {
                    Signal::Buy => {
                        if matches!(
                            reason,
                            ExecutionError::NetworkError(_) | ExecutionError::Unknown(_)
                        ) {
                            error!(
                                "Ambiguous Entry Failure ({:?}). Transitioning to Reconciling.",
                                reason
                            );
                            self.transition_state(StrategyState::Reconciling);
                        } else {
                            self.transition_state(StrategyState::Flat);
                        }
                    }
                    Signal::Sell => {
                        // CF1 FIX: Short Entry Failure
                        if matches!(
                            reason,
                            ExecutionError::NetworkError(_) | ExecutionError::Unknown(_)
                        ) {
                            error!("Ambiguous Short Entry Failure ({:?}). Transitioning to Reconciling.", reason);
                            self.transition_state(StrategyState::Reconciling);
                        } else {
                            self.transition_state(StrategyState::Flat);
                        }
                    }
                    Signal::Exit => {
                        // CF2 FIX: Exit Failure
                        if matches!(
                            reason,
                            ExecutionError::NetworkError(_) | ExecutionError::Unknown(_)
                        ) {
                            error!(
                                "Ambiguous Exit Failure ({:?}). Transitioning to Reconciling.",
                                reason
                            );
                            self.transition_state(StrategyState::Reconciling);
                        } else if let StrategyState::Exiting {
                            direction,
                            leg1_qty,
                            leg2_qty,
                            leg1_entry_price,
                            leg2_entry_price,
                        } = self.state
                        {
                            info!("Exit rejected/failed definitively. Reverting to InPosition.");
                            self.transition_state(StrategyState::InPosition {
                                direction,
                                leg1_qty,
                                leg2_qty,
                                leg1_entry_price,
                                leg2_entry_price,
                            });
                        } else {
                            error!("CRITICAL: State mismatch! Expected Exiting but found {:?}. Transitioning to Reconciling.", self.state);
                            self.transition_state(StrategyState::Reconciling);
                        }
                    }
                    _ => {}
                }
            }
            ExecutionResult::PartialFailure(reason) => {
                error!(
                    "CRITICAL: Partial Execution Failure: {}. Transitioning to Reconciling.",
                    reason
                );
                self.transition_state(StrategyState::Reconciling);
            }
        }
    }

    fn handle_recovery_result(&mut self, result: RecoveryResult) {
        info!("Received Recovery Result: {:?}", result);
        match result {
            RecoveryResult::Success(symbol) => {
                info!(
                    "Recovery successful for {}. Attempting to resolve state.",
                    symbol
                );
                if self.state == StrategyState::Reconciling {
                    info!("Recovery complete. Transitioning to Flat.");
                    self.transition_state(StrategyState::Flat);
                }
            }
            RecoveryResult::Failed(symbol) => {
                error!("Recovery FAILED for {}. Transitioning to Halted.", symbol);
                self.transition_state(StrategyState::Halted);
            }
        }
    }

    fn check_timeout(&mut self) {
        if matches!(
            self.state,
            StrategyState::Entering { .. } | StrategyState::Exiting { .. }
        ) {
            let now = self.clock.now_ts_millis();
            if now - self.last_state_change_ts > self.config.execution_timeout_ms {
                error!("CRITICAL: Execution Timeout in state {:?}! No report received for {}ms. Transitioning to Reconciling.", self.state, self.config.execution_timeout_ms);
                self.transition_state(StrategyState::Reconciling);
                self.trigger_reconciliation();
            }
        }
    }

    fn trigger_reconciliation(&self) {
        info!("Triggering reconciliation process...");
        error!("MANUAL INTERVENTION REQUIRED: Strategy is in Reconciling state. Please check exchange positions and restart if necessary.");
    }

    /// Processes a single tick of matched Leg 1 and Leg 2 data.
    #[instrument(skip(self, leg1, leg2), fields(leg1_price = %leg1.price, leg2_price = %leg2.price))]
    async fn process_tick(&mut self, leg1: &MarketData, leg2: &MarketData) {
        if matches!(
            self.state,
            StrategyState::Reconciling | StrategyState::Halted
        ) {
            if self.throttler.unstable_state.should_log() {
                let suppressed = self
                    .throttler
                    .unstable_state
                    .get_and_reset_suppressed_count();
                warn!(
                    state = ?self.state,
                    suppressed = suppressed,
                    "Dropping tick due to unstable state"
                );
            }
            return;
        }

        // MARKET HOURS CHECK
        // If market is closed, drop ticks to prevent "hallucinating" signals
        if !self.is_market_open {
            // We can throttle this log if needed, but since we are dropping ticks,
            // maybe we want to know? Let's treat it as "Sync Issue" for throttling purposes?
            // Or just reuse unstable_state throttle or create a new one?
            // Reusing unstable_state simple for now.
            if self.throttler.unstable_state.should_log() {
                let suppressed = self
                    .throttler
                    .unstable_state
                    .get_and_reset_suppressed_count();
                warn!(
                    suppressed = suppressed,
                    "Market Closed - Dropping verification tick"
                );
            }
            return;
        }

        // --- P1 FIX: Load Shedding ---
        // Drop stale ticks to prevent execution on prices that no longer exist.
        // During high-volatility events or WebSocket floods, channel buffers fill and
        // processing lags behind. Executing on 5-second-old prices causes massive slippage.
        //
        // SAFETY: Reconciling/Halted states already returned above, so we only shed load
        // during normal operation when we can afford to skip stale data.
        //
        // HOT PATH: Simple i64 subtraction - single CPU instruction.
        let now = self.clock.now_ts_millis();
        let tick_latency = now - leg1.timestamp.max(leg2.timestamp);

        if tick_latency > MAX_ALLOWED_LATENCY_MS {
            if self.throttler.latency_drop.should_log() {
                let suppressed = self.throttler.latency_drop.get_and_reset_suppressed_count();
                warn!(
                    latency_ms = tick_latency,
                    threshold_ms = MAX_ALLOWED_LATENCY_MS,
                    leg1_ts = leg1.timestamp,
                    leg2_ts = leg2.timestamp,
                    suppressed = suppressed,
                    "LOAD SHEDDING: Dropping stale tick to prevent execution slippage"
                );
            }
            return;
        }

        if let Err(e) = self.validator.validate(leg1, now) {
            if self.throttler.tick_age.should_log() {
                let suppressed = self.throttler.tick_age.get_and_reset_suppressed_count();
                warn!(
                    symbol = %leg1.symbol,
                    error = %e,
                    suppressed = suppressed,
                    "Dropping tick due to validation error"
                );
            }
            return;
        }

        if let Err(e) = self.validator.validate(leg2, now) {
            if self.throttler.tick_age.should_log() {
                let suppressed = self.throttler.tick_age.get_and_reset_suppressed_count();
                warn!(
                    symbol = %leg2.symbol,
                    error = %e,
                    suppressed = suppressed,
                    "Dropping tick due to validation error"
                );
            }
            return;
        }

        let diff = (leg1.timestamp - leg2.timestamp).abs();
        if diff > self.config.max_tick_age_ms {
            if self.throttler.sync_issue.should_log() {
                let suppressed = self.throttler.sync_issue.get_and_reset_suppressed_count();
                warn!("Dropping tick due to sync. Diff: {}ms (Max: {}ms). Leg1 TS: {}, Leg2 TS: {} (Suppressed: {})", diff, self.config.max_tick_age_ms, leg1.timestamp, leg2.timestamp, suppressed);
            }
            return;
        }

        // TASK 4: Ghost Position Recovery - Reset unknown entry prices to current market prices
        // This handles the case where we recovered a position after restart without fill history
        if let StrategyState::InPosition {
            direction,
            leg1_qty,
            leg2_qty,
            leg1_entry_price,
            leg2_entry_price,
        } = &self.state
        {
            // Check if entry prices are the unknown marker
            if *leg1_entry_price == UNKNOWN_ENTRY_PRICE || *leg2_entry_price == UNKNOWN_ENTRY_PRICE
            {
                let new_leg1_price = if *leg1_entry_price == UNKNOWN_ENTRY_PRICE {
                    leg1.price
                } else {
                    *leg1_entry_price
                };
                let new_leg2_price = if *leg2_entry_price == UNKNOWN_ENTRY_PRICE {
                    leg2.price
                } else {
                    *leg2_entry_price
                };

                warn!(
                    "GHOST POSITION PNL RESET: Setting entry prices to current market. \
                    Leg1: {} -> {}, Leg2: {} -> {}. \
                    PnL tracking starts from zero. Manual reconciliation required for accurate accounting.",
                    leg1_entry_price, new_leg1_price,
                    leg2_entry_price, new_leg2_price
                );

                // SAFETY GUARD: Check for imbalance when prices become available
                let leg1_value = *leg1_qty * new_leg1_price;
                let leg2_value = *leg2_qty * new_leg2_price;
                let total_value = leg1_value + leg2_value;

                if total_value > Decimal::ZERO {
                    let imbalance = (leg1_value - leg2_value).abs() / total_value;
                    if imbalance > self.max_imbalance_ratio {
                        error!(
                            "CRITICAL: Position imbalance detected after reconcile! \
                            Leg1 Value: ${:.2}, Leg2 Value: ${:.2}, Imbalance: {:.1}% (max: {:.1}%). \
                            Halting to prevent further damage. Manual intervention required.",
                            leg1_value, leg2_value, imbalance * dec!(100), self.max_imbalance_ratio * dec!(100)
                        );
                        crate::metrics::record_strategy_halted("dual_leg", &self.pair_name);
                        self.state = StrategyState::Halted;
                        return;
                    }
                }

                // Note: derefs needed because if let bindings are references
                self.state = StrategyState::InPosition {
                    direction: *direction,
                    leg1_qty: *leg1_qty,
                    leg2_qty: *leg2_qty,
                    leg1_entry_price: new_leg1_price,
                    leg2_entry_price: new_leg2_price,
                };
            }
        }

        let signal = self.entry_manager.analyze(leg1, leg2).await;

        match signal {
            Signal::Buy => {
                if self.state != StrategyState::Flat {
                    return;
                }

                // FIX: Interpret order_size as USD Allocation.
                // Quantity = Allocation / Price
                if leg1.price.is_zero() {
                    warn!("Invalid price 0.0 for {}", leg1.symbol);
                    return;
                }
                let leg1_qty = self.config.order_size / leg1.price;
                if let Ok(hedge_qty) = self
                    .risk_monitor
                    .calc_hedge_ratio(leg1_qty, leg1.price, leg2.price)
                {
                    info!("Long Entry Signal! Transitioning to Entering state and spawning execution...");
                    self.transition_state(StrategyState::Entering {
                        direction: PositionDirection::Long, // AS-1: Track Long entry
                        leg1_qty,
                        leg2_qty: hedge_qty,
                        leg1_entry_price: leg1.price,
                        leg2_entry_price: leg2.price,
                    });

                    let engine = self.execution_engine.clone();
                    let pair = self.pair.clone();
                    let tx = self.report_tx.clone();
                    let p1 = leg1.price;
                    let p2 = leg2.price;

                    // CB-2 FIX: Use JoinSet for structured concurrency
                    self.active_executions.spawn(async move {
                        let result = match engine
                            .execute_basis_entry(&pair, PositionDirection::Long, leg1_qty, hedge_qty, p1, p2)
                            .await
                        {
                            Ok(res) => res,
                            Err(e) => ExecutionResult::TotalFailure(e),
                        };
                        let report = ExecutionReport {
                            result,
                            action: Signal::Buy,
                            pnl_delta: None,
                        };

                        match tx.send(report).await {
                            Ok(_) => debug!("Entry execution report sent successfully"),
                            Err(e) => {
                                error!("CRITICAL: Execution report channel closed: {}. Strategy state desync!", e);
                            }
                        }
                    });
                }
            }
            Signal::Sell => {
                // CF1 FIX: Handle Short Entry (Sell Leg 1, Buy Leg 2)
                if self.state != StrategyState::Flat {
                    return;
                }

                // FIX: Interpret order_size as USD Allocation.
                // Quantity = Allocation / Price
                if leg1.price.is_zero() {
                    warn!("Invalid price 0.0 for {}", leg1.symbol);
                    return;
                }
                let leg1_qty = self.config.order_size / leg1.price;
                if let Ok(hedge_qty) = self
                    .risk_monitor
                    .calc_hedge_ratio(leg1_qty, leg1.price, leg2.price)
                {
                    info!("Short Entry Signal! Transitioning to Entering state and spawning execution...");
                    self.transition_state(StrategyState::Entering {
                        direction: PositionDirection::Short, // AS-1: Track Short entry
                        leg1_qty,
                        leg2_qty: hedge_qty,
                        leg1_entry_price: leg1.price,
                        leg2_entry_price: leg2.price,
                    });

                    let engine = self.execution_engine.clone();
                    let pair = self.pair.clone();
                    let tx = self.report_tx.clone();
                    let p1 = leg1.price;
                    let p2 = leg2.price;

                    // CB-2 FIX: Use JoinSet for structured concurrency
                    self.active_executions.spawn(async move {
                        let result = match engine
                            .execute_basis_entry(&pair, PositionDirection::Short, leg1_qty, hedge_qty, p1, p2)
                            .await
                        {
                            Ok(res) => res,
                            Err(e) => ExecutionResult::TotalFailure(e),
                        };
                        let report = ExecutionReport {
                            result,
                            action: Signal::Sell,
                            pnl_delta: None,
                        };

                        match tx.send(report).await {
                            Ok(_) => debug!("Short Entry execution report sent successfully"),
                            Err(e) => {
                                error!("CRITICAL: Execution report channel closed: {}. Strategy state desync!", e);
                            }
                        }
                    });
                }
            }
            Signal::Exit => {
                // CF2 FIX: Handle Exit Signal (Mean Reversion)
                if let StrategyState::InPosition {
                    direction,
                    leg1_qty,
                    leg2_qty,
                    leg1_entry_price,
                    leg2_entry_price,
                } = self.state
                {
                    // AS-1: Use direction for PnL calculation
                    // Long: PnL = (exit - entry) for leg1, (entry - exit) for leg2
                    // Short: PnL = (entry - exit) for leg1, (exit - entry) for leg2
                    let (leg1_pnl, leg2_pnl) = match direction {
                        PositionDirection::Long => (
                            (leg1.price - leg1_entry_price) * leg1_qty,
                            (leg2_entry_price - leg2.price) * leg2_qty,
                        ),
                        PositionDirection::Short => (
                            (leg1_entry_price - leg1.price) * leg1_qty,
                            (leg2.price - leg2_entry_price) * leg2_qty,
                        ),
                    };
                    let gross_pnl = leg1_pnl + leg2_pnl;

                    let total_volume = (leg1_entry_price * leg1_qty)
                        + (leg2_entry_price * leg2_qty)
                        + (leg1.price * leg1_qty)
                        + (leg2.price * leg2_qty);
                    let fee_rate = self.config.fee_tier.taker_fee_bps / dec!(10000.0);
                    let estimated_fees = total_volume * fee_rate;
                    let slippage_cost =
                        total_volume * (self.config.fee_tier.slippage_bps / dec!(10000.0));
                    let net_pnl = gross_pnl - estimated_fees - slippage_cost;

                    if !self
                        .exit_policy
                        .should_exit(Decimal::ZERO, Decimal::ZERO, net_pnl)
                        .await
                    {
                        if self.throttler.unstable_state.should_log() {
                            let suppressed = self
                                .throttler
                                .unstable_state
                                .get_and_reset_suppressed_count();
                            info!("Holding Position. Signal: Exit, but Exit Policy says HOLD. Net PnL: {:.6} (Suppressed: {})", net_pnl, suppressed);
                        }
                        return;
                    }

                    info!("Exit Signal! Net PnL: {:.6} (Gross: {:.6}, Fees: {:.6}, Slippage: {:.6}). Transitioning to Exiting...", net_pnl, gross_pnl, estimated_fees, slippage_cost);
                    self.transition_state(StrategyState::Exiting {
                        direction, // AS-1: Carry direction to Exiting state
                        leg1_qty,
                        leg2_qty,
                        leg1_entry_price,
                        leg2_entry_price,
                    });

                    let engine = self.execution_engine.clone();
                    let pair = self.pair.clone();
                    let tx = self.report_tx.clone();
                    let slippage_factor = self.config.fee_tier.slippage_bps / dec!(10000.0);

                    // AS-1 FIX: Use explicit direction for exit order sides
                    // Long: Sell L1, Buy L2
                    // Short: Buy L1, Sell L2
                    let is_long = direction == PositionDirection::Long;

                    let p1_limit = if is_long {
                        leg1.price * (dec!(1.0) - slippage_factor)
                    } else {
                        leg1.price * (dec!(1.0) + slippage_factor)
                    };
                    let p2_limit = if is_long {
                        leg2.price * (dec!(1.0) + slippage_factor)
                    } else {
                        leg2.price * (dec!(1.0) - slippage_factor)
                    };

                    debug!("Executing Exit with Limit Prices - Spot: {} (Market: {}), Future: {} (Market: {})", p1_limit, leg1.price, p2_limit, leg2.price);

                    // CB-2 FIX: Use JoinSet for structured concurrency
                    self.active_executions.spawn(async move {
                        let result = match engine
                            .execute_basis_exit(
                                &pair, direction, leg1_qty, leg2_qty, p1_limit, p2_limit,
                            )
                            .await
                        {
                            Ok(res) => res,
                            Err(e) => ExecutionResult::TotalFailure(e),
                        };
                        let report = ExecutionReport {
                            result,
                            action: Signal::Exit,
                            pnl_delta: None,
                        };

                        match tx.send(report).await {
                            Ok(_) => debug!("Exit execution report sent successfully"),
                            Err(e) => {
                                error!("CRITICAL: Execution report channel closed: {}. Strategy state desync!", e);
                            }
                        }
                    });
                }
            }
            _ => {}
        }
    }
}

#[async_trait]
impl Executor for CoinbaseClient {
    async fn execute_order(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: Decimal,
        price: Option<Decimal>,
    ) -> Result<crate::orders::OrderId, crate::exchange::ExchangeError> {
        self.place_order(symbol, &side.to_string(), quantity, price)
            .await
            .map_err(crate::exchange::ExchangeError::from_boxed)?;

        // MC-2 FIX: Generate order ID for tracking
        let order_id = crate::orders::OrderId::new(format!("cb-{}", uuid::Uuid::new_v4()));
        Ok(order_id)
    }

    async fn get_position(&self, symbol: &str) -> Result<Decimal, crate::exchange::ExchangeError> {
        CoinbaseClient::get_position(self, symbol)
            .await
            .map_err(crate::exchange::ExchangeError::from_boxed)
    }
}

impl Drop for DualLegStrategy {
    fn drop(&mut self) {
        // CB-2 FIX: Abort all pending execution tasks on shutdown
        let pending = self.active_executions.len();
        self.active_executions.abort_all();
        info!(
            spot = %self.pair.spot_symbol,
            future = %self.pair.future_symbol,
            state = ?self.state,
            aborted_tasks = pending,
            "DualLegStrategy dropped - aborted pending executions"
        );
    }
}

// ============================================================================
// LIVE STRATEGY WRAPPER (For StrategySupervisor integration)
// ============================================================================

use crate::strategy::{LiveStrategy, StrategyInput};

/// Configuration for creating a DualLegStrategyLive from config
#[derive(Debug, Clone)]
pub struct DualLegLiveConfig {
    pub dual_leg_config: DualLegConfig,
    pub window_size: usize,
    pub entry_z_score: f64,
    pub exit_z_score: f64,
    pub strategy_type: DualLegStrategyType,
    // N-1 FIX: Extracted magic numbers into config
    /// Circuit breaker failure threshold before tripping (default: 5)
    pub circuit_breaker_threshold: u32,
    /// Circuit breaker timeout in seconds before attempting recovery (default: 60)
    pub circuit_breaker_timeout_secs: u64,
    /// Basis entry threshold in basis points (default: 10.0)
    pub basis_entry_bps: Decimal,
    /// Basis exit threshold in basis points (default: 2.0)
    pub basis_exit_bps: Decimal,
    /// Maximum leverage for risk monitor (default: 3.0)
    pub max_leverage: Decimal,
    // MC-1 FIX: Configurable drift recalculation interval
    /// Interval (in ticks) for recalculating running sums to prevent f64 drift (default: 10_000)
    /// Guidance: For 1000 ticks/sec, use 10_000 (10s). For 1 tick/min, use 1_000 (~17 hours).
    pub drift_recalc_interval: u64,
    // SAFETY GUARD: Maximum position value in USD per symbol before halting
    /// Maximum USD value of position per symbol. If exceeded, new entries are blocked.
    /// Set to None for no limit (DANGEROUS for production). Default: 50_000
    pub max_position_usd: Option<Decimal>,
    // SAFETY GUARD: Maximum allowed imbalance ratio between legs before halting
    /// If |leg1_value - leg2_value| / (leg1_value + leg2_value) > this, halt on reconcile.
    /// Default: 0.5 (50% imbalance triggers halt)
    pub max_imbalance_ratio: Decimal,
}

/// N-1 FIX: Default implementation with production-ready values
impl Default for DualLegLiveConfig {
    fn default() -> Self {
        Self {
            dual_leg_config: DualLegConfigBuilder::new()
                .spot_symbol("BTC-USD")
                .future_symbol("BTC-USDT")
                .build()
                .expect("Default config should be valid"),
            window_size: 100,
            entry_z_score: 2.0,
            exit_z_score: 0.5,
            strategy_type: DualLegStrategyType::Pairs,
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout_secs: 60,
            basis_entry_bps: dec!(10.0),
            basis_exit_bps: dec!(2.0),
            max_leverage: dec!(3.0),
            drift_recalc_interval: 10_000,
            max_position_usd: Some(dec!(50_000)),
            max_imbalance_ratio: dec!(0.5),
        }
    }
}

/// Type of dual-leg strategy to create
#[derive(Debug, Clone, Copy)]
pub enum DualLegStrategyType {
    Basis,
    Pairs,
}

/// Live trading wrapper for DualLegStrategy
/// Implements LiveStrategy trait for use with StrategySupervisor
pub struct DualLegStrategyLive<E: Executor + 'static> {
    id: String,
    config: DualLegLiveConfig,
    executor: Arc<E>,
    pnl: Decimal,
    healthy: bool,
    // STATE PERSISTENCE: Optional state store for surviving restarts
    state_store: Option<std::sync::Arc<dyn crate::logging::StateStore>>,
    // STATE PERSISTENCE: Whether running in paper mode
    is_paper: bool,
}

impl<E: Executor + 'static> DualLegStrategyLive<E> {
    /// Create a new DualLegStrategyLive from configuration
    pub fn new(id: String, config: DualLegLiveConfig, executor: Arc<E>) -> Self {
        Self {
            id,
            config,
            executor,
            pnl: Decimal::ZERO,
            healthy: true,
            state_store: None,
            is_paper: false,
        }
    }

    /// Configure optional state persistence for surviving container restarts.
    #[must_use]
    pub fn with_state_store(
        mut self,
        store: std::sync::Arc<dyn crate::logging::StateStore>,
        is_paper: bool,
    ) -> Self {
        self.state_store = Some(store);
        self.is_paper = is_paper;
        self
    }

    /// Build the internal DualLegStrategy with all dependencies
    ///
    /// # CB-1 NOTE on Panic Safety
    /// The Recovery Worker is spawned as a detached task. Its internal `run()` method uses
    /// `JoinSet` which gracefully handles panics from individual recovery tasks by logging
    /// them without crashing the entire worker. If the worker's top-level `run()` panics,
    /// the spawned task will terminate silently. For production deployments, monitor the
    /// `algopioneer_recovery_attempts_total` metric - if it stops incrementing during active
    /// recoveries, the worker may have crashed.
    fn build_strategy(&self) -> DualLegStrategy {
        let (recovery_tx, recovery_rx) = mpsc::channel(100);
        let (feedback_tx, feedback_rx) = mpsc::channel(100);

        // Create recovery worker (CB-1: see doc comment above for panic handling notes)
        let recovery_worker = RecoveryWorker::new(self.executor.clone(), recovery_rx, feedback_tx);
        tokio::spawn(async move {
            recovery_worker.run().await;
        });

        // N-1 FIX: Use config values instead of magic numbers
        let execution_engine = ExecutionEngine::new(
            self.executor.clone(),
            recovery_tx.clone(),
            self.config.circuit_breaker_threshold,
            self.config.circuit_breaker_timeout_secs,
        );

        // Create entry manager based on strategy type
        let entry_manager: Box<dyn EntryStrategy> = match self.config.strategy_type {
            DualLegStrategyType::Basis => {
                // N-1 FIX: Use config values
                Box::new(BasisManager::new(
                    self.config.basis_entry_bps,
                    self.config.basis_exit_bps,
                    self.config.dual_leg_config.fee_tier,
                ))
            }
            DualLegStrategyType::Pairs => Box::new(PairsManager::new(
                self.config.window_size,
                self.config.entry_z_score,
                self.config.exit_z_score,
            )),
        };

        // N-1 FIX: Use config value for leverage
        let risk_monitor = RiskMonitor::new(
            self.config.max_leverage,
            InstrumentType::Linear,
            HedgeMode::DollarNeutral,
        );

        let mut strategy = DualLegStrategy::new(
            entry_manager,
            risk_monitor,
            execution_engine,
            self.config.dual_leg_config.clone(),
            feedback_rx,
            Box::new(SystemClock),
        );

        // STATE PERSISTENCE: Wire up state store if configured
        if let Some(store) = &self.state_store {
            let strategy_type = match self.config.strategy_type {
                DualLegStrategyType::Basis => "basis",
                DualLegStrategyType::Pairs => "pairs",
            };
            strategy = strategy.with_state_store(store.clone(), strategy_type, self.is_paper);
        }

        // SAFETY GUARD: Wire up position limits and imbalance detection
        strategy = strategy.with_safety_guards(
            self.config.max_position_usd,
            self.config.max_imbalance_ratio,
        );

        strategy
    }
}

#[async_trait]
impl<E: Executor + 'static> LiveStrategy for DualLegStrategyLive<E> {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn subscribed_symbols(&self) -> Vec<String> {
        vec![
            self.config.dual_leg_config.spot_symbol.clone(),
            self.config.dual_leg_config.future_symbol.clone(),
        ]
    }

    fn strategy_type(&self) -> &'static str {
        match self.config.strategy_type {
            DualLegStrategyType::Basis => "DualLeg-Basis",
            DualLegStrategyType::Pairs => "DualLeg-Pairs",
        }
    }

    fn current_pnl(&self) -> Decimal {
        self.pnl
    }

    fn is_healthy(&self) -> bool {
        self.healthy
    }

    async fn run(&mut self, mut data_rx: mpsc::Receiver<StrategyInput>) {
        info!(
            id = %self.id,
            strategy_type = self.strategy_type(),
            spot = %self.config.dual_leg_config.spot_symbol,
            future = %self.config.dual_leg_config.future_symbol,
            "Starting DualLegStrategyLive"
        );

        // Build internal strategy and channels
        let mut strategy = self.build_strategy();

        // Create internal channels for the strategy's run() method
        let (leg1_tx, leg1_rx) = mpsc::channel::<Arc<MarketData>>(100);
        let (leg2_tx, leg2_rx) = mpsc::channel::<Arc<MarketData>>(100);

        // Spawn the internal strategy run loop
        let spot_symbol = self.config.dual_leg_config.spot_symbol.clone();
        let future_symbol = self.config.dual_leg_config.future_symbol.clone();

        let strategy_handle = tokio::spawn(async move {
            strategy.run(leg1_rx, leg2_rx).await;
        });

        // Route incoming StrategyInput to the internal channels
        while let Some(input) = data_rx.recv().await {
            match input {
                StrategyInput::Tick(tick) => {
                    // Route based on symbol - check symbol first, then send
                    let send_failed = if tick.symbol == spot_symbol {
                        leg1_tx.send(tick).await.is_err()
                    } else if tick.symbol == future_symbol {
                        leg2_tx.send(tick).await.is_err()
                    } else {
                        false // Unknown symbol, ignore
                    };
                    if send_failed {
                        break;
                    }
                }
                StrategyInput::PairedTick { leg1, leg2 } => {
                    // MC-2 FIX: Use tokio::join! for concurrent sends to prevent temporal desync
                    // If one channel is full while the other isn't, sequential sends would
                    // create leg timing skew - exactly what PairedTick is designed to prevent
                    let (r1, r2) = tokio::join!(leg1_tx.send(leg1), leg2_tx.send(leg2));
                    if r1.is_err() || r2.is_err() {
                        break;
                    }
                }
            }
        }

        // Cleanup: drop senders to signal strategy to stop
        drop(leg1_tx);
        drop(leg2_tx);

        // Wait for strategy to finish
        let _ = strategy_handle.await;

        info!(id = %self.id, "DualLegStrategyLive stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::Mutex;
    use tokio::time::Duration;

    #[test]
    fn test_log_throttle() {
        let mut throttle = LogThrottle::new(Duration::from_millis(100));
        assert!(throttle.should_log());
        assert!(!throttle.should_log());
        std::thread::sleep(Duration::from_millis(110));
        assert!(throttle.should_log());
    }
    #[tokio::test]
    async fn test_z_score_calculation() {
        let mut manager = PairsManager::new(5, 1.9, 0.1);
        for _ in 0..5 {
            let _ = manager
                .analyze(
                    &MarketData {
                        symbol: "A".into(),
                        price: dec!(1.0),
                        instrument_id: None,
                        timestamp: 0,
                    },
                    &MarketData {
                        symbol: "B".into(),
                        price: dec!(1.0),
                        instrument_id: None,
                        timestamp: 0,
                    },
                )
                .await;
        }

        let p1 = Decimal::from_f64((-10.0f64).exp()).unwrap();
        let signal = manager
            .analyze(
                &MarketData {
                    symbol: "A".into(),
                    price: p1,
                    instrument_id: None,
                    timestamp: 0,
                },
                &MarketData {
                    symbol: "B".into(),
                    price: dec!(1.0),
                    instrument_id: None,
                    timestamp: 0,
                },
            )
            .await;

        assert_eq!(signal, Signal::Buy);
    }

    #[test]
    fn test_hedge_ratio_dollar_neutral() {
        let monitor = RiskMonitor::new(dec!(1.0), InstrumentType::Linear, HedgeMode::DollarNeutral);
        let ratio = monitor
            .calc_hedge_ratio(dec!(1.0), dec!(100.0), dec!(50.0))
            .unwrap();
        assert_eq!(ratio, dec!(2.0));
    }

    #[test]
    fn test_hedge_ratio_delta_neutral_linear() {
        let monitor = RiskMonitor::new(dec!(1.0), InstrumentType::Linear, HedgeMode::DeltaNeutral);
        let ratio = monitor
            .calc_hedge_ratio(dec!(1.5), dec!(100.0), dec!(50.0))
            .unwrap();
        assert_eq!(ratio, dec!(1.5));
    }

    #[test]
    fn test_hedge_ratio_delta_neutral_inverse() {
        let monitor = RiskMonitor::new(dec!(1.0), InstrumentType::Inverse, HedgeMode::DeltaNeutral);
        let ratio = monitor
            .calc_hedge_ratio(dec!(1.0), dec!(100.0), dec!(50000.0))
            .unwrap();
        assert_eq!(ratio, dec!(50000.0));
    }

    #[test]
    fn test_net_spread_calculation() {
        let model = TransactionCostModel::new(dec!(10.0), dec!(20.0), dec!(5.0));
        let net = model.calc_net_spread(dec!(100.0));
        assert_eq!(net, dec!(15.0));
    }

    #[tokio::test]
    async fn test_basis_manager_signals() {
        let cost_model = TransactionCostModel::new(dec!(0.0), dec!(0.0), dec!(0.0));
        let mut manager = BasisManager::new(dec!(10.0), dec!(2.0), cost_model);

        let signal = manager
            .analyze(
                &MarketData {
                    symbol: "S".into(),
                    price: dec!(100.0),
                    instrument_id: None,
                    timestamp: 0,
                },
                &MarketData {
                    symbol: "F".into(),
                    price: dec!(100.2),
                    instrument_id: None,
                    timestamp: 0,
                },
            )
            .await;
        assert_eq!(signal, Signal::Buy);
    }

    struct MockExecutor {
        should_fail: Arc<Mutex<bool>>,
        call_count: Arc<Mutex<usize>>,
        executed_orders: Arc<Mutex<Vec<(String, OrderSide, Decimal)>>>,
    }

    impl MockExecutor {
        fn new() -> Self {
            Self {
                should_fail: Arc::new(Mutex::new(false)),
                call_count: Arc::new(Mutex::new(0)),
                executed_orders: Arc::new(Mutex::new(Vec::new())),
            }
        }

        async fn set_should_fail(&self, fail: bool) {
            *self.should_fail.lock().await = fail;
        }

        async fn get_call_count(&self) -> usize {
            *self.call_count.lock().await
        }
    }

    #[async_trait]
    impl Executor for MockExecutor {
        async fn execute_order(
            &self,
            symbol: &str,
            side: OrderSide,
            quantity: Decimal,
            _price: Option<Decimal>,
        ) -> Result<crate::orders::OrderId, crate::exchange::ExchangeError> {
            let mut count = self.call_count.lock().await;
            *count += 1;

            if *self.should_fail.lock().await {
                return Err(crate::exchange::ExchangeError::Other(
                    "Mock execution failure".to_string(),
                ));
            }

            self.executed_orders
                .lock()
                .await
                .push((symbol.to_string(), side, quantity));

            // MC-2 FIX: Return mock order ID
            Ok(crate::orders::OrderId::new(format!("mock-{}", *count)))
        }

        async fn get_position(
            &self,
            _symbol: &str,
        ) -> Result<Decimal, crate::exchange::ExchangeError> {
            Ok(Decimal::ZERO)
        }
    }

    #[tokio::test]
    async fn test_mock_executor_success() {
        let executor = MockExecutor::new();
        let result = executor
            .execute_order("BTC-USD", OrderSide::Buy, dec!(0.1), Some(dec!(50000.0)))
            .await;
        assert!(result.is_ok());
        assert_eq!(executor.get_call_count().await, 1);
    }

    #[tokio::test]
    async fn test_mock_executor_failure() {
        let executor = MockExecutor::new();
        executor.set_should_fail(true).await;
        let result = executor
            .execute_order("BTC-USD", OrderSide::Buy, dec!(0.1), Some(dec!(50000.0)))
            .await;
        assert!(result.is_err());
        assert_eq!(executor.get_call_count().await, 1);
    }

    #[tokio::test]
    async fn test_execute_exit_zero_quantity() {
        let executor = Arc::new(MockExecutor::new());
        let (recovery_tx, _recovery_rx) = mpsc::channel(100);
        let engine = ExecutionEngine::new(executor.clone(), recovery_tx, 5, 60);

        let pair = InstrumentPair {
            spot_symbol: "SPOT".to_string(),
            future_symbol: "FUTURE".to_string(),
        };

        // Test 1: Only Spot has quantity
        let res = engine
            .execute_basis_exit(
                &pair,
                PositionDirection::Long,
                dec!(10.0), // Spot
                dec!(0),    // Future (Empty)
                dec!(100.0),
                dec!(100.0),
            )
            .await;
        assert!(res.is_ok());
        assert_eq!(executor.get_call_count().await, 1);

        // Verify order details (mock stores them)
        let orders = executor.executed_orders.lock().await;
        assert_eq!(orders.len(), 1);
        assert_eq!(orders[0].0, "SPOT");
        assert_eq!(orders[0].2, dec!(10.0));
        drop(orders);

        // Test 2: Only Future has quantity
        let res = engine
            .execute_basis_exit(
                &pair,
                PositionDirection::Long,
                dec!(0),   // Spot (Empty)
                dec!(5.0), // Future
                dec!(100.0),
                dec!(100.0),
            )
            .await;
        assert!(res.is_ok());
        assert_eq!(executor.get_call_count().await, 2); // 1 previous + 1 new

        // Test 3: Both zero
        let res = engine
            .execute_basis_exit(
                &pair,
                PositionDirection::Long,
                dec!(0),
                dec!(0),
                dec!(100.0),
                dec!(100.0),
            )
            .await;
        assert!(res.is_ok());
        assert_eq!(executor.get_call_count().await, 2); // Unchanged
    }
}
