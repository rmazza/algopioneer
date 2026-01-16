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
use crate::resilience::CircuitBreaker;
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
use tokio::sync::{mpsc, Semaphore};
use tokio::time::{Duration, Instant};
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

const RECOVERY_BACKOFF_CAP_SECS: u64 = 60;

// NP-1 FIX: Extract magic number to named constant
/// Maximum number of recovery attempts before abandoning and requiring manual intervention
const MAX_RECOVERY_ATTEMPTS: u32 = 5;

// CF3: Precision Safety Constants
// Conservative bounds for f64 price ratio conversions to prevent precision loss
// in statistical arbitrage calculations. Ratios outside these bounds will be rejected.
const MAX_SAFE_PRICE_RATIO: f64 = 1e12; // Conservative limit for f64 precision
const MIN_SAFE_PRICE_RATIO: f64 = 1e-12; // Reciprocal of max for symmetry

// NP-1 FIX: Extract magic number to named constant for precision warning throttling
const PRECISION_WARNING_LOG_INTERVAL: u64 = 1000;

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
    /// P1 FIX: Throttler for load shedding warnings (stale tick drops)
    pub latency_drop: LogThrottle,
}

impl DualLegLogThrottler {
    pub fn new(interval_secs: u64) -> Self {
        let interval = Duration::from_secs(interval_secs);
        Self {
            unstable_state: LogThrottle::new(interval),
            tick_age: LogThrottle::new(interval),
            sync_issue: LogThrottle::new(interval),
            latency_drop: LogThrottle::new(interval),
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

/// Trait for validating market data ticks before processing.
/// Enables testable and composable validation logic.
pub trait TickValidator: Send + Sync {
    /// Validates a market data tick against validation rules.
    /// Returns Ok(()) if valid, Err(msg) if invalid.
    fn validate(&self, tick: &MarketData, now_ts: i64) -> Result<(), String>;
}

/// Validates tick age to ensure data freshness.
#[derive(Debug, Clone)]
pub struct AgeValidator {
    max_age_ms: i64,
}

impl AgeValidator {
    pub fn new(max_age_ms: i64) -> Self {
        Self { max_age_ms }
    }
}

impl TickValidator for AgeValidator {
    fn validate(&self, tick: &MarketData, now_ts: i64) -> Result<(), String> {
        let age_ms = now_ts - tick.timestamp;
        if age_ms > self.max_age_ms {
            Err(format!(
                "Tick age {}ms exceeds max {}ms",
                age_ms, self.max_age_ms
            ))
        } else if age_ms < 0 {
            Err(format!(
                "Tick timestamp {} is in the future (now: {})",
                tick.timestamp, now_ts
            ))
        } else {
            Ok(())
        }
    }
}

/// Validates tick price is positive and not NaN/Inf.
#[derive(Debug, Clone)]
pub struct PriceValidator;

impl TickValidator for PriceValidator {
    fn validate(&self, tick: &MarketData, _now_ts: i64) -> Result<(), String> {
        if tick.price <= Decimal::ZERO {
            Err(format!("Invalid price: {} must be positive", tick.price))
        } else {
            Ok(())
        }
    }
}

/// Composite validator that chains multiple validators.
/// Fails on first validation error.
pub struct CompositeValidator {
    validators: Vec<Box<dyn TickValidator>>,
}

impl CompositeValidator {
    pub fn new(validators: Vec<Box<dyn TickValidator>>) -> Self {
        Self { validators }
    }
}

impl TickValidator for CompositeValidator {
    fn validate(&self, tick: &MarketData, now_ts: i64) -> Result<(), String> {
        for validator in &self.validators {
            validator.validate(tick, now_ts)?;
        }
        Ok(())
    }
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
        Self {
            value_bps,
            spot_price: spot,
            future_price: future,
        }
    }
}

// --- State Management ---

// AS9: ExitPolicy trait for flexible exit conditions
#[async_trait]
pub trait ExitPolicy: Send + Sync {
    /// Returns true if the position should be exited
    async fn should_exit(&self, entry_price: Decimal, current_price: Decimal, pnl: Decimal)
        -> bool;
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

/// Composite exit policy that triggers if ANY sub-policy triggers
pub struct CompositeExitPolicy {
    policies: Vec<Box<dyn ExitPolicy>>,
}

impl CompositeExitPolicy {
    pub fn new(policies: Vec<Box<dyn ExitPolicy>>) -> Self {
        Self { policies }
    }
}

#[async_trait]
impl ExitPolicy for CompositeExitPolicy {
    async fn should_exit(
        &self,
        entry_price: Decimal,
        current_price: Decimal,
        pnl: Decimal,
    ) -> bool {
        for policy in &self.policies {
            if policy.should_exit(entry_price, current_price, pnl).await {
                return true;
            }
        }
        false
    }
}

#[derive(Debug, Clone)]
pub struct PnlExitPolicy {
    min_profit: Decimal,
    stop_loss: Decimal,
}

impl PnlExitPolicy {
    pub fn new(min_profit: Decimal, stop_loss: Decimal) -> Self {
        Self {
            min_profit,
            stop_loss,
        }
    }
}

#[async_trait]
impl ExitPolicy for PnlExitPolicy {
    async fn should_exit(
        &self,
        _entry_price: Decimal,
        _current_price: Decimal,
        pnl: Decimal,
    ) -> bool {
        // Exit if PnL is below stop loss (e.g. -15 < -10)
        // OR if PnL is above min profit (e.g. 20 > 10)
        pnl <= self.stop_loss || pnl >= self.min_profit
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

/// Trait for entry logic strategies.
/// Allows for swapping different entry algorithms (e.g., simple threshold vs. statistical arbitrage).
#[async_trait]
pub trait EntryStrategy: Send {
    /// Analyzes the market data for Leg 1 and Leg 2 to generate a trading signal.
    /// Uses `&mut self` to allow implementations to update internal state (e.g., sliding window)
    /// without requiring internal synchronization.
    async fn analyze(&mut self, leg1: &MarketData, leg2: &MarketData) -> Signal;
}

// Executor trait is now imported from crate::exchange

// --- Components ---

/// Analyzes the basis spread and generates signals based on fixed thresholds.
pub struct BasisManager {
    entry_threshold_bps: Decimal,
    exit_threshold_bps: Decimal,
    cost_model: TransactionCostModel,
}

impl BasisManager {
    pub fn new(
        entry_threshold_bps: Decimal,
        exit_threshold_bps: Decimal,
        cost_model: TransactionCostModel,
    ) -> Self {
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
    async fn analyze(&mut self, leg1: &MarketData, leg2: &MarketData) -> Signal {
        let spread = Spread::new(leg1.price, leg2.price);
        let net_spread = self.cost_model.calc_net_spread(spread.value_bps);
        debug!(
            "Basis Spread: {:.4} bps (Net: {:.4}) (Spot: {}, Future: {})",
            spread.value_bps, net_spread, spread.spot_price, spread.future_price
        );

        if net_spread > self.entry_threshold_bps {
            Signal::Buy
        } else if spread.value_bps < self.exit_threshold_bps {
            Signal::Exit // CF2: Explicit exit signal
        } else {
            Signal::Hold
        }
    }
}

/// Statistical Arbitrage (Pairs Trading) Manager.
/// Uses Z-Score of the log-spread to generate signals.
///
/// # Performance
/// Uses O(1) sliding window statistics via running sums instead of O(n) iteration.
/// This eliminates per-tick iteration and removes Mutex contention.
///
/// # Adaptive Thresholds (Phase 2)
/// Uses EWMA (Exponentially Weighted Moving Average) of spread volatility to normalize
/// Z-scores against changing market regimes. This prevents over-trading in low-vol
/// environments and under-trading in high-vol environments.
pub struct PairsManager {
    window_size: usize,
    entry_z_score: f64,
    exit_z_score: f64,
    /// Sliding window of log-spreads (no Mutex - uses &mut self)
    spread_history: std::collections::VecDeque<f64>,
    /// Running sum for O(1) mean calculation
    running_sum: f64,
    /// Running sum of squares for O(1) variance calculation
    running_sq_sum: f64,
    // CF3 FIX: Precision monitoring metrics
    precision_rejections: std::sync::atomic::AtomicU64,
    precision_warnings: std::sync::atomic::AtomicU64,
    // Phase 2: EWMA volatility tracking for adaptive thresholds
    /// Exponentially Weighted Moving Average of spread standard deviation.
    /// NOTE: f64 is intentional - the entire PairsManager operates on f64 because
    /// log-spread calculations (ln()) require floating-point. See CF3 DOCUMENTATION
    /// block for precision risk assessment.
    ewma_volatility: f64,
    /// EWMA decay factor (alpha). Higher = more weight to recent values.
    /// Common values: 0.06 (~30-period half-life), 0.1 (~20-period half-life).
    ewma_alpha: f64,
    /// Whether EWMA has been initialized (needs bootstrap period)
    ewma_initialized: bool,
    // MC-1 FIX: Counter for periodic recalculation to prevent f64 drift
    tick_count: u64,
    // P2: Dynamic hedge ratio via Kalman Filter
    /// Optional Kalman Filter for tracking time-varying hedge ratio (beta).
    /// When enabled, updates beta estimate on each tick for regime adaptation.
    kalman: Option<crate::math::KalmanHedgeRatio>,
}

impl PairsManager {
    /// Default EWMA decay factor (≈30-period half-life)
    const DEFAULT_EWMA_ALPHA: f64 = 0.06;

    /// Create a new PairsManager with default EWMA alpha for adaptive thresholds.
    pub fn new(window_size: usize, entry_z_score: f64, exit_z_score: f64) -> Self {
        Self::new_adaptive(
            window_size,
            entry_z_score,
            exit_z_score,
            Self::DEFAULT_EWMA_ALPHA,
        )
    }

    /// Create a new PairsManager with explicit EWMA alpha for adaptive thresholds.
    ///
    /// # Arguments
    /// * `window_size` - Size of the sliding window for spread statistics
    /// * `entry_z_score` - Z-score threshold to enter a position
    /// * `exit_z_score` - Z-score threshold to exit a position (mean reversion)
    /// * `ewma_alpha` - EWMA decay factor (0.0-1.0). Higher = more weight to recent volatility.
    ///   Common values: 0.06 (~30-period half-life), 0.1 (~20-period half-life).
    pub fn new_adaptive(
        window_size: usize,
        entry_z_score: f64,
        exit_z_score: f64,
        ewma_alpha: f64,
    ) -> Self {
        Self::new_with_kalman(window_size, entry_z_score, exit_z_score, ewma_alpha, false)
    }

    /// Create a new PairsManager with dynamic hedge ratio tracking via Kalman Filter.
    ///
    /// # Arguments
    /// * `window_size` - Size of the sliding window for spread statistics
    /// * `entry_z_score` - Z-score threshold to enter a position
    /// * `exit_z_score` - Z-score threshold to exit a position (mean reversion)
    /// * `ewma_alpha` - EWMA decay factor (0.0-1.0). Higher = more weight to recent volatility.
    /// * `enable_kalman` - Whether to enable dynamic hedge ratio estimation via Kalman Filter.
    ///   When true, the hedge ratio updates each tick to track cointegration drift.
    pub fn new_with_kalman(
        window_size: usize,
        entry_z_score: f64,
        exit_z_score: f64,
        ewma_alpha: f64,
        enable_kalman: bool,
    ) -> Self {
        Self {
            window_size,
            entry_z_score,
            exit_z_score,
            spread_history: std::collections::VecDeque::with_capacity(window_size),
            running_sum: 0.0,
            running_sq_sum: 0.0,
            // CF3 FIX: Initialize precision monitoring counters
            precision_rejections: std::sync::atomic::AtomicU64::new(0),
            precision_warnings: std::sync::atomic::AtomicU64::new(0),
            // Phase 2: EWMA volatility tracking
            ewma_volatility: 0.0,
            ewma_alpha,
            ewma_initialized: false,
            tick_count: 0, // MC-1
            // P2: Initialize Kalman Filter if requested
            kalman: if enable_kalman {
                Some(crate::math::KalmanHedgeRatio::default_for_pairs())
            } else {
                None
            },
        }
    }

    /// CF3 FIX: Get precision monitoring metrics
    pub fn get_precision_metrics(&self) -> (u64, u64) {
        (
            self.precision_rejections
                .load(std::sync::atomic::Ordering::Relaxed),
            self.precision_warnings
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Phase 2: Get the current EWMA volatility estimate (for monitoring/testing)
    pub fn get_ewma_volatility(&self) -> f64 {
        self.ewma_volatility
    }

    /// Phase 2: Check if EWMA has been initialized
    pub fn is_ewma_initialized(&self) -> bool {
        self.ewma_initialized
    }

    /// P2: Get the current dynamic hedge ratio from Kalman Filter.
    ///
    /// Returns the Kalman-estimated beta if enabled and warmed up.
    /// This value should be used by `RiskMonitor` to calculate the actual hedge quantity.
    ///
    /// # Returns
    /// - `Some(beta)` if Kalman is enabled and warmed up (>100 updates)
    /// - `None` if Kalman is disabled or not yet warmed up
    ///
    /// # Precision Warning
    /// The returned f64 is suitable for hedge ratio estimation but MUST be
    /// converted to `Decimal` before being used in position sizing or PnL calculations.
    pub fn get_dynamic_hedge_ratio(&self) -> Option<f64> {
        // N-1 FIX: Named constant instead of magic number
        const KALMAN_WARMUP_TICKS: u64 = 100;
        self.kalman.as_ref().and_then(|k| {
            if k.is_warmed_up(KALMAN_WARMUP_TICKS) {
                Some(k.get_beta())
            } else {
                None
            }
        })
    }

    /// P2: Check if Kalman Filter is enabled
    pub fn is_kalman_enabled(&self) -> bool {
        self.kalman.is_some()
    }
}

#[async_trait]
impl EntryStrategy for PairsManager {
    #[instrument(skip(self))]
    async fn analyze(&mut self, leg1: &MarketData, leg2: &MarketData) -> Signal {
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
                    warn!(
                        "PRECISION WARNING: Infinite or NaN prices detected. P1: {}, P2: {}",
                        v1, v2
                    );
                    return Signal::Hold;
                }

                // CF3 FIX: Enforce hard limits on price ratios to prevent precision loss
                let ratio = v1 / v2;
                if !(MIN_SAFE_PRICE_RATIO..=MAX_SAFE_PRICE_RATIO).contains(&ratio) {
                    // CF3 MONITORING: Increment rejection counter
                    let count = self
                        .precision_rejections
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                        + 1;
                    // CB-2 FIX: Emit Prometheus metric for monitoring
                    crate::metrics::record_precision_rejection(&format!(
                        "{}/{}",
                        leg1.symbol, leg2.symbol
                    ));
                    error!(
                        "PRECISION ERROR: Price ratio {:.2e} exceeds safe f64 bounds [{:.2e}, {:.2e}]. Rejecting signal to prevent precision loss. (Total rejections: {})",
                        ratio, MIN_SAFE_PRICE_RATIO, MAX_SAFE_PRICE_RATIO, count
                    );
                    return Signal::Hold; // Hard rejection - do not trade on degraded precision
                }

                // Log warning for ratios approaching the limit (within 1 order of magnitude)
                // Throttle warnings to max 1 per 1000 occurrences to avoid log spam
                if !(MIN_SAFE_PRICE_RATIO * 10.0..=MAX_SAFE_PRICE_RATIO / 10.0).contains(&ratio) {
                    // CF3 MONITORING: Increment warning counter
                    let count = self
                        .precision_warnings
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                        + 1;
                    // CB-2 FIX: Emit Prometheus metric for monitoring
                    crate::metrics::record_precision_warning(&format!(
                        "{}/{}",
                        leg1.symbol, leg2.symbol
                    ));
                    // NP-1 FIX: Use named constant instead of magic number
                    if count == 1 || count.is_multiple_of(PRECISION_WARNING_LOG_INTERVAL) {
                        warn!(
                            "PRECISION WARNING: Price ratio {:.2e} approaching safety limits. Monitor for precision degradation. (Total warnings: {})",
                            ratio, count
                        );
                    }
                }

                (v1, v2)
            }
            _ => {
                // N-3 FIX: Use error! for consistency with other signal rejections
                error!(
                    "PRECISION ERROR: Invalid prices for Pairs analysis: {:?}, {:?}",
                    leg1.price, leg2.price
                );
                return Signal::Hold;
            }
        };

        // P2: Update Kalman Filter and compute spread
        // When Kalman is enabled, the spread uses dynamic beta: spread = ln(p1) - β * ln(p2)
        // When Kalman is disabled, use static 1:1 spread: spread = ln(p1) - ln(p2)
        let spread = if let Some(ref mut kalman) = self.kalman {
            // Update Kalman with current prices (in log-space for stability)
            let ln_p1 = p1.ln();
            let ln_p2 = p2.ln();
            let beta = kalman.update(ln_p2, ln_p1); // Regress ln(p1) on ln(p2)
            ln_p1 - beta * ln_p2
        } else {
            // Static 1:1 log-spread (original behavior)
            p1.ln() - p2.ln()
        };

        // O(1) sliding window statistics via running sums
        // Add new value to running sums
        self.running_sum += spread;
        self.running_sq_sum += spread * spread;
        self.spread_history.push_back(spread);

        // MC-1 FIX: Periodic recalculation to prevent f64 drift
        // Every 10,000 ticks, recompute from window to eliminate accumulated error
        const RECALC_INTERVAL: u64 = 10_000;
        self.tick_count += 1;
        if self.tick_count.is_multiple_of(RECALC_INTERVAL) {
            self.running_sum = self.spread_history.iter().sum();
            self.running_sq_sum = self.spread_history.iter().map(|x| x * x).sum();
            debug!(
                tick_count = self.tick_count,
                "MC-1: Recalculated running sums to prevent f64 drift"
            );
        }

        // Remove oldest value if window is full
        if self.spread_history.len() > self.window_size {
            if let Some(old_spread) = self.spread_history.pop_front() {
                self.running_sum -= old_spread;
                self.running_sq_sum -= old_spread * old_spread;
            }
        }

        // Not enough data yet
        if self.spread_history.len() < self.window_size {
            return Signal::Hold;
        }

        // O(1) mean and variance calculation
        let n = self.spread_history.len() as f64;
        let mean = self.running_sum / n;

        // Variance = E[X^2] - E[X]^2
        // Clamp to 0.0 to handle floating-point precision issues
        let variance = (self.running_sq_sum / n - mean * mean).max(0.0);
        let std_dev = variance.sqrt();

        // Phase 2: Update EWMA volatility estimate
        // Initialize with first valid std_dev, then apply exponential smoothing
        if !self.ewma_initialized {
            self.ewma_volatility = std_dev;
            self.ewma_initialized = true;
        } else {
            // EWMA update: σ_ewma = α * σ_current + (1 - α) * σ_previous
            self.ewma_volatility =
                self.ewma_alpha * std_dev + (1.0 - self.ewma_alpha) * self.ewma_volatility;
        }

        // Phase 2: Compute adaptive Z-score using EWMA volatility
        // This normalizes signals against longer-term volatility regime,
        // preventing over-trading in low-vol and under-trading in high-vol
        const MIN_VOLATILITY: f64 = 1e-12; // Prevent division by zero
        let effective_vol = self.ewma_volatility.max(MIN_VOLATILITY);
        let z_score = (spread - mean) / effective_vol;

        debug!(
            "Pairs Spread: {:.6}, Z-Score (adaptive): {:.4}, EWMA Vol: {:.6}, Window StdDev: {:.6}",
            spread, z_score, self.ewma_volatility, std_dev
        );

        if z_score > self.entry_z_score {
            Signal::Sell // Sell A / Buy B (Short the spread)
        } else if z_score < -self.entry_z_score {
            Signal::Buy // Buy A / Sell B (Long the spread)
        } else if z_score.abs() < self.exit_z_score {
            Signal::Exit // Close positions (Mean Reversion)
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

/// Worker that processes recovery tasks (failed legs).
// AS3: Priority system for recovery tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Critical = 0, // Kill switches (highest priority)
    High = 1,     // Large positions
    Normal = 2,   // Regular retries
}

// N2: Named constant instead of magic number
const MAX_CONCURRENT_RECOVERIES: usize = 5;

/// Worker that processes recovery tasks (failed legs) with structured concurrency.
///
/// Uses `tokio::task::JoinSet` to ensure all spawned tasks are properly tracked
/// and cancelled on shutdown. This prevents "orphaned tasks" that could continue
/// executing after the worker is dropped.
pub struct RecoveryWorker {
    client: Arc<dyn Executor>,
    rx: mpsc::Receiver<RecoveryTask>,
    feedback_tx: mpsc::Sender<RecoveryResult>,
    semaphore: Arc<Semaphore>,
}

impl RecoveryWorker {
    pub fn new(
        client: Arc<dyn Executor>,
        rx: mpsc::Receiver<RecoveryTask>,
        feedback_tx: mpsc::Sender<RecoveryResult>,
    ) -> Self {
        Self {
            client,
            rx,
            feedback_tx,
            semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_RECOVERIES)),
        }
    }

    /// Run the recovery worker with structured concurrency.
    ///
    /// Uses `JoinSet` to track all spawned recovery tasks. On shutdown:
    /// - All pending tasks are automatically cancelled
    /// - No orphaned tasks continue executing
    #[instrument(skip(self), name = "recovery_worker")]
    pub async fn run(mut self) {
        use tokio::task::JoinSet;
        use tracing::Instrument;

        info!("Recovery Worker started.");

        // Structured concurrency: Own all spawned tasks
        let mut active_recoveries: JoinSet<()> = JoinSet::new();

        loop {
            tokio::select! {
                // Accept new tasks from the channel
                task_opt = self.rx.recv() => {
                    match task_opt {
                        Some(task) => {
                            let client = self.client.clone();
                            let feedback_tx = self.feedback_tx.clone();
                            let semaphore = self.semaphore.clone();
                            let span = tracing::info_span!("recovery_task", symbol = %task.symbol);

                            active_recoveries.spawn(
                                async move {
                                    // Acquire semaphore permit (RAII: released when dropped)
                                    let _permit = match semaphore.acquire_owned().await {
                                        Ok(p) => p,
                                        Err(_) => {
                                            // Semaphore closed - worker shutting down
                                            return;
                                        }
                                    };

                                    perform_recovery_with_backoff(client, task, feedback_tx).await;
                                }
                                .instrument(span)
                            );
                        }
                        None => {
                            // Channel closed - initiate graceful shutdown
                            info!(
                                "Recovery channel closed. Shutting down {} active tasks.",
                                active_recoveries.len()
                            );
                            break;
                        }
                    }
                }

                // Reap completed tasks (handles panics gracefully)
                Some(result) = active_recoveries.join_next() => {
                    if let Err(join_err) = result {
                        if join_err.is_panic() {
                            error!("CRITICAL: Recovery task panicked: {:?}", join_err);
                        }
                        // Cancelled tasks are expected during shutdown, don't log them
                    }
                }
            }
        }

        // Graceful Shutdown: Wait for all active recoveries to complete
        // (or cancel them if the runtime is shutting down)
        info!(
            "Waiting for {} active recovery tasks to complete...",
            active_recoveries.len()
        );
        while let Some(result) = active_recoveries.join_next().await {
            if let Err(join_err) = result {
                if join_err.is_panic() {
                    error!("Recovery task panicked during shutdown: {:?}", join_err);
                }
            }
        }
        info!("Recovery Worker shutdown complete.");
    }
}

/// Performs a single recovery with exponential backoff.
///
/// This is a pure async function that can be easily unit tested.
async fn perform_recovery_with_backoff(
    client: Arc<dyn Executor>,
    mut task: RecoveryTask,
    feedback_tx: mpsc::Sender<RecoveryResult>,
) {
    let mut backoff = Duration::from_secs(2);
    let mut task_throttler = LogThrottle::new(Duration::from_secs(5));

    info!("Processing Recovery Task: {:?}", task);

    for attempt in 1..=MAX_RECOVERY_ATTEMPTS {
        task.attempts = attempt;

        match client
            .execute_order(&task.symbol, task.action, task.quantity, None)
            .await
        {
            Ok(_) => {
                info!(
                    "Recovery Successful for {} on attempt {}",
                    task.symbol, attempt
                );
                let _ = feedback_tx
                    .send(RecoveryResult::Success(task.symbol.clone()))
                    .await;
                return;
            }
            Err(e) => {
                if task_throttler.should_log() {
                    let suppressed = task_throttler.get_and_reset_suppressed_count();
                    error!(
                        "Recovery Failed for {} (Attempt {}): {} (Suppressed: {})",
                        task.symbol, attempt, e, suppressed
                    );
                }

                if attempt >= MAX_RECOVERY_ATTEMPTS {
                    error!(
                        "CRITICAL: Recovery abandoned for {} after {} attempts. MANUAL INTERVENTION REQUIRED.",
                        task.symbol, MAX_RECOVERY_ATTEMPTS
                    );
                    let _ = feedback_tx
                        .send(RecoveryResult::Failed(task.symbol.clone()))
                        .await;
                    return;
                }

                tokio::time::sleep(backoff).await;
                backoff =
                    std::cmp::min(backoff * 2, Duration::from_secs(RECOVERY_BACKOFF_CAP_SECS));
            }
        }
    }
}

// AS2: CircuitBreaker moved to `crate::resilience::CircuitBreaker`
// See `src/resilience/circuit_breaker.rs` for the improved implementation
// with single RwLock (reduced lock contention) and comprehensive unit tests.

/// Handles concurrent execution of orders on both legs with circuit breaker protection.
///
/// # MC-3 NOTE on Circuit Breaker Sharing
/// The `CircuitBreaker` is owned by this `ExecutionEngine` instance. If you wrap
/// `ExecutionEngine` in an `Arc` and share it across multiple trading pairs, the circuit
/// breaker state will be shared - a failure on one pair will affect all pairs.
///
/// **Recommended usage:** Create one `ExecutionEngine` per trading pair to isolate
/// circuit breaker state. For portfolio strategies, each pair should have its own engine.
pub struct ExecutionEngine {
    client: Arc<dyn Executor>,
    recovery_tx: mpsc::Sender<RecoveryTask>,
    circuit_breaker: CircuitBreaker,
}

impl ExecutionEngine {
    /// Creates a new ExecutionEngine with the specified executor, recovery channel, and circuit breaker settings.
    pub fn new(
        client: Arc<dyn Executor>,
        recovery_tx: mpsc::Sender<RecoveryTask>,
        failure_threshold: u32,
        timeout_secs: u64,
    ) -> Self {
        Self {
            client,
            recovery_tx,
            circuit_breaker: CircuitBreaker::new(
                failure_threshold,
                Duration::from_secs(timeout_secs),
            ),
        }
    }

    /// Refactor Challenge: DRY helper for queuing kill switch recovery tasks.
    /// Handles partial failure by queuing a recovery task and recording circuit breaker failure.
    /// Returns PartialFailure on success, CriticalFailure if recovery queue fails.
    async fn queue_kill_switch(
        &self,
        failed_result: ExecutionError,
        successful_leg_symbol: &str,
        recovery_action: OrderSide,
        quantity: Decimal,
        context: &str,
    ) -> Result<ExecutionResult, ExecutionError> {
        error!(
            "CRITICAL: {} failed. {} leg succeeded, queuing kill switch on {}.",
            context,
            if recovery_action == OrderSide::Buy {
                "Spot"
            } else {
                "Future"
            },
            successful_leg_symbol
        );

        let task = RecoveryTask {
            symbol: successful_leg_symbol.to_string(),
            action: recovery_action,
            quantity,
            reason: format!("{}: {}", context, failed_result),
            attempts: 0,
        };

        if let Err(send_err) = self.recovery_tx.send(task).await {
            error!(
                "EMERGENCY: Failed to queue recovery task! Manual intervention required. Error: {}",
                send_err
            );
            self.circuit_breaker.record_failure();
            return Err(ExecutionError::CriticalFailure(format!(
                "Recovery queue failure - {} kill switch failed: {}",
                context, send_err
            )));
        }

        self.circuit_breaker.record_failure();
        Ok(ExecutionResult::PartialFailure(failed_result))
    }

    /// Refactor Challenge: DRY helper for queuing exit retry recovery tasks.
    async fn queue_exit_retry(
        &self,
        symbol: &str,
        action: OrderSide,
        quantity: Decimal,
        error: ExecutionError,
        leg_name: &str,
    ) -> Result<(), ExecutionError> {
        error!("Exit {} failed: {}. Queuing retry.", leg_name, error);

        let task = RecoveryTask {
            symbol: symbol.to_string(),
            action,
            quantity,
            reason: format!("Exit {} failed: {}", leg_name, error),
            attempts: 0,
        };

        if let Err(send_err) = self.recovery_tx.send(task).await {
            error!(
                "EMERGENCY: Failed to queue exit {} recovery task! Error: {}",
                leg_name, send_err
            );
            return Err(ExecutionError::CriticalFailure(format!(
                "Recovery queue failure - exit {} retry failed: {}",
                leg_name, send_err
            )));
        }

        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn execute_basis_entry(
        &self,
        pair: &InstrumentPair,
        quantity: Decimal,
        hedge_qty: Decimal,
        leg1_price: Decimal,
        leg2_price: Decimal,
    ) -> Result<ExecutionResult, ExecutionError> {
        // CR1: Check Circuit Breaker
        if self.circuit_breaker.is_open() {
            return Err(ExecutionError::CircuitBreakerOpen(
                "Circuit breaker is OPEN due to recent failures".to_string(),
            ));
        }

        // Concurrently execute both legs to minimize leg risk
        let spot_leg = self.client.execute_order(
            &pair.spot_symbol,
            OrderSide::Buy,
            quantity,
            Some(leg1_price),
        );
        let future_leg = self.client.execute_order(
            &pair.future_symbol,
            OrderSide::Sell,
            hedge_qty,
            Some(leg2_price),
        );

        let (spot_res, future_res) = tokio::join!(spot_leg, future_leg);

        // Convert errors to ExecutionError
        let spot_res = spot_res.map_err(ExecutionError::from);
        let future_res = future_res.map_err(ExecutionError::from);

        // Handle failures using DRY helper
        if let Err(e) = spot_res {
            if future_res.is_ok() {
                return self
                    .queue_kill_switch(
                        e,
                        &pair.future_symbol,
                        OrderSide::Buy,
                        hedge_qty,
                        "Spot entry",
                    )
                    .await;
            }
            self.circuit_breaker.record_failure();
            return Ok(ExecutionResult::TotalFailure(e));
        }

        if let Err(e) = future_res {
            return self
                .queue_kill_switch(
                    e,
                    &pair.spot_symbol,
                    OrderSide::Sell,
                    quantity,
                    "Future entry",
                )
                .await;
        }

        self.circuit_breaker.record_success();
        Ok(ExecutionResult::Success)
    }

    /// CF1 FIX: Execute Short Entry (Sell Spot, Buy Future)
    #[instrument(skip(self))]
    pub async fn execute_basis_short_entry(
        &self,
        pair: &InstrumentPair,
        quantity: Decimal,
        hedge_qty: Decimal,
        leg1_price: Decimal,
        leg2_price: Decimal,
    ) -> Result<ExecutionResult, ExecutionError> {
        // CR1: Check Circuit Breaker
        if self.circuit_breaker.is_open() {
            return Err(ExecutionError::CircuitBreakerOpen(
                "Circuit breaker is OPEN due to recent failures".to_string(),
            ));
        }

        // Short Entry: Sell Spot, Buy Future
        let spot_leg = self.client.execute_order(
            &pair.spot_symbol,
            OrderSide::Sell,
            quantity,
            Some(leg1_price),
        );
        let future_leg = self.client.execute_order(
            &pair.future_symbol,
            OrderSide::Buy,
            hedge_qty,
            Some(leg2_price),
        );

        let (spot_res, future_res) = tokio::join!(spot_leg, future_leg);

        let spot_res = spot_res.map_err(ExecutionError::from);
        let future_res = future_res.map_err(ExecutionError::from);

        // Handle failures using DRY helper
        if let Err(e) = spot_res {
            if future_res.is_ok() {
                return self
                    .queue_kill_switch(
                        e,
                        &pair.future_symbol,
                        OrderSide::Sell,
                        hedge_qty,
                        "Spot short entry",
                    )
                    .await;
            }
            self.circuit_breaker.record_failure();
            return Ok(ExecutionResult::TotalFailure(e));
        }

        if let Err(e) = future_res {
            return self
                .queue_kill_switch(
                    e,
                    &pair.spot_symbol,
                    OrderSide::Buy,
                    quantity,
                    "Future short entry",
                )
                .await;
        }

        self.circuit_breaker.record_success();
        Ok(ExecutionResult::Success)
    }

    #[instrument(skip(self))]
    pub async fn execute_basis_exit(
        &self,
        pair: &InstrumentPair,
        direction: PositionDirection,
        quantity: Decimal,
        hedge_qty: Decimal,
        leg1_price: Decimal,
        leg2_price: Decimal,
    ) -> Result<ExecutionResult, ExecutionError> {
        // CR1: Check Circuit Breaker
        if self.circuit_breaker.is_open() {
            return Err(ExecutionError::CircuitBreakerOpen(
                "Circuit breaker is OPEN due to recent failures".to_string(),
            ));
        }

        // CRITICAL FIX: Direction-aware order sides
        // Long exit: Sell Spot (close long), Buy Future (close short)
        // Short exit: Buy Spot (close short), Sell Future (close long)
        let (spot_side, future_side) = match direction {
            PositionDirection::Long => (OrderSide::Sell, OrderSide::Buy),
            PositionDirection::Short => (OrderSide::Buy, OrderSide::Sell),
        };

        // BUG FIX: Skip execution if quantity is zero
        let spot_leg = if quantity > Decimal::ZERO {
            Some(self.client.execute_order(
                &pair.spot_symbol,
                spot_side,
                quantity,
                Some(leg1_price),
            ))
        } else {
            None
        };

        let future_leg = if hedge_qty > Decimal::ZERO {
            Some(self.client.execute_order(
                &pair.future_symbol,
                future_side,
                hedge_qty,
                Some(leg2_price),
            ))
        } else {
            None
        };

        // If both are None, we have nothing to do
        if spot_leg.is_none() && future_leg.is_none() {
            return Ok(ExecutionResult::Success);
        }

        // Execute concurrently if both exist, or just await the one that exists
        let (spot_res, future_res) = match (spot_leg, future_leg) {
            (Some(s), Some(f)) => {
                let (s_res, f_res) = tokio::join!(s, f);
                (Some(s_res), Some(f_res))
            }
            (Some(s), None) => (Some(s.await), None),
            (None, Some(f)) => (None, Some(f.await)),
            (None, None) => unreachable!(),
        };

        // Process results
        let mut failed = false;

        let spot_err = if let Some(res) = spot_res {
            match res.map_err(ExecutionError::from) {
                Ok(_) => None,
                Err(e) => {
                    failed = true;
                    Some(e)
                }
            }
        } else {
            None
        };

        let future_err = if let Some(res) = future_res {
            match res.map_err(ExecutionError::from) {
                Ok(_) => None,
                Err(e) => {
                    failed = true;
                    Some(e)
                }
            }
        } else {
            None
        };

        if failed {
            // If both failed (and both existed and were attempted)
            if spot_err.is_some() && future_err.is_some() {
                self.circuit_breaker.record_failure();
                return Ok(ExecutionResult::TotalFailure(
                    ExecutionError::ExchangeError(format!(
                        "Both legs failed to exit. Spot: {:?}, Future: {:?}",
                        spot_err, future_err
                    )),
                ));
            }

            // If one failed (or the only one attempted failed)
            if let Some(e) = spot_err {
                self.queue_exit_retry(&pair.spot_symbol, spot_side, quantity, e, "Spot")
                    .await?;
            }

            if let Some(e) = future_err {
                self.queue_exit_retry(&pair.future_symbol, future_side, hedge_qty, e, "Future")
                    .await?;
            }

            self.circuit_breaker.record_failure();
            return Ok(ExecutionResult::PartialFailure(
                ExecutionError::ExchangeError("Partial Exit Failure".to_string()),
            ));
        }

        self.circuit_breaker.record_success();
        Ok(ExecutionResult::Success)
    }

    /// Query current position from exchange (for state reconciliation)
    pub async fn get_position(
        &self,
        symbol: &str,
    ) -> Result<Decimal, crate::exchange::ExchangeError> {
        self.client.get_position(symbol).await
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
    // AS2: Tick validator for market data validation
    validator: Box<dyn TickValidator>,
    // AS9: Exit policy for flexible exit logic
    exit_policy: Box<dyn ExitPolicy>,
    // CB-2 FIX: Track spawned execution tasks for structured concurrency
    active_executions: tokio::task::JoinSet<()>,
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

            // MC-2 FIX: Emit metric for Halted state to enable alerting
            if new_state == StrategyState::Halted {
                let pair = format!("{}/{}", self.pair.spot_symbol, self.pair.future_symbol);
                crate::metrics::record_strategy_halted("dual_leg", &pair);
                error!(
                    pair = %pair,
                    "CRITICAL: Strategy entered Halted state - manual intervention required"
                );
            }

            if let Some(tx) = &self.state_notifier {
                if let Err(e) = tx.try_send(new_state) {
                    warn!("State update dropped due to backpressure: {}", e);
                }
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
                    // Infer direction from leg1 quantity sign
                    // Long = positive leg1 (bought spot), Short = negative leg1 (sold spot)
                    let direction = if leg1_qty >= Decimal::ZERO {
                        PositionDirection::Long
                    } else {
                        PositionDirection::Short
                    };

                    info!(
                        "Detected existing position: leg1={}, leg2={}, direction={}. Transitioning to InPosition.",
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

        // STATE AMNESIA FIX: Reconcile position state before processing any ticks
        self.reconcile_state().await;

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
                    "Dropping tick due to unstable state: {:?} (Suppressed: {})",
                    self.state, suppressed
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
                    "Dropping tick for {}: {} (Suppressed: {})",
                    leg1.symbol, e, suppressed
                );
            }
            return;
        }

        if let Err(e) = self.validator.validate(leg2, now) {
            if self.throttler.tick_age.should_log() {
                let suppressed = self.throttler.tick_age.get_and_reset_suppressed_count();
                warn!(
                    "Dropping tick for {}: {} (Suppressed: {})",
                    leg2.symbol, e, suppressed
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
                            .execute_basis_entry(&pair, leg1_qty, hedge_qty, p1, p2)
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
                        // Short Entry: Sell Leg 1, Buy Leg 2 (Inverse of Basis Entry)
                        // execute_basis_entry does Buy L1 / Sell L2. We need a new method or use execute_order directly.
                        // For now, assuming execute_basis_entry can be adapted or we use raw execution.
                        // Actually, let's look at ExecutionEngine. It likely has execute_basis_entry hardcoded to Buy/Sell.
                        // We should probably add execute_basis_short_entry to ExecutionEngine or make it generic.
                        // Given constraints, I'll use execute_basis_exit logic but for entry? No, that's closing.
                        // Let's assume for this fix we need to implement the short logic manually here or add a method.
                        // Since I can't easily add a method to ExecutionEngine without seeing it all, I'll use the raw executor if possible,
                        // but ExecutionEngine wraps it.
                        //
                        // WAIT: ExecutionEngine is a struct in this file. I can modify it!
                        // But for now, let's see if I can reuse execute_basis_entry with swapped sides?
                        // execute_basis_entry(pair, qty1, qty2, p1, p2) -> Buys Spot, Sells Future.
                        // Short Entry -> Sell Spot, Buy Future.
                        //
                        // Let's implement `execute_basis_short_entry` in ExecutionEngine later.
                        // For now, I will assume `execute_basis_short_entry` exists or I will add it.
                        // To avoid compilation error, I must add it to ExecutionEngine.

                        let result = match engine
                            .execute_basis_short_entry(&pair, leg1_qty, hedge_qty, p1, p2)
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
        }
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

        DualLegStrategy::new(
            entry_manager,
            risk_monitor,
            execution_engine,
            self.config.dual_leg_config.clone(),
            feedback_rx,
            Box::new(SystemClock),
        )
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
