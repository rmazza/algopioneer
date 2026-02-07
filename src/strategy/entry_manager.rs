//! Entry strategy implementations for dual-leg trading.
//!
//! This module provides the `EntryStrategy` trait and two implementations:
//! - `BasisManager`: Fixed threshold entry/exit based on spot-future spread
//! - `PairsManager`: Statistical arbitrage using Z-score of log-spread

use async_trait::async_trait;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, error, instrument, warn};

use crate::strategy::Signal;
use crate::types::MarketData;

// Re-export TransactionCostModel and Spread as they're used by BasisManager
pub use super::dual_leg_trading::{Spread, TransactionCostModel};

// CF3: Precision Safety Constants
// Conservative bounds for f64 price ratio conversions to prevent precision loss
// in statistical arbitrage calculations. Ratios outside these bounds will be rejected.
const MAX_SAFE_PRICE_RATIO: f64 = 1e12; // Conservative limit for f64 precision
const MIN_SAFE_PRICE_RATIO: f64 = 1e-12; // Reciprocal of max for symmetry

// NP-1 FIX: Extract magic number to named constant for precision warning throttling
const PRECISION_WARNING_LOG_INTERVAL: u64 = 1000;

/// Trait for entry logic strategies.
/// Allows for swapping different entry algorithms (e.g., simple threshold vs. statistical arbitrage).
#[async_trait]
pub trait EntryStrategy: Send {
    /// Analyzes the market data for Leg 1 and Leg 2 to generate a trading signal.
    /// Uses `&mut self` to allow implementations to update internal state (e.g., sliding window)
    /// without requiring internal synchronization.
    async fn analyze(&mut self, leg1: &MarketData, leg2: &MarketData) -> Signal;
}

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
    spread_history: VecDeque<f64>,
    /// Running sum for O(1) mean calculation
    running_sum: f64,
    /// Running sum of squares for O(1) variance calculation
    running_sq_sum: f64,
    // CF3 FIX: Precision monitoring metrics
    precision_rejections: AtomicU64,
    precision_warnings: AtomicU64,
    // Phase 2: EWMA volatility tracking for adaptive thresholds
    /// Exponentially Weighted Moving Average of spread standard deviation.
    ewma_volatility: f64,
    /// EWMA decay factor (alpha). Higher = more weight to recent values.
    ewma_alpha: f64,
    /// Whether EWMA has been initialized (needs bootstrap period)
    ewma_initialized: bool,
    // MC-1 FIX: Counter for periodic recalculation to prevent f64 drift
    tick_count: u64,
    // P2: Dynamic hedge ratio via Kalman Filter
    /// Optional Kalman Filter for tracking time-varying hedge ratio (beta).
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
    pub fn new_adaptive(
        window_size: usize,
        entry_z_score: f64,
        exit_z_score: f64,
        ewma_alpha: f64,
    ) -> Self {
        Self::new_with_kalman(window_size, entry_z_score, exit_z_score, ewma_alpha, false)
    }

    /// Create a new PairsManager with dynamic hedge ratio tracking via Kalman Filter.
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
            spread_history: VecDeque::with_capacity(window_size),
            running_sum: 0.0,
            running_sq_sum: 0.0,
            precision_rejections: AtomicU64::new(0),
            precision_warnings: AtomicU64::new(0),
            ewma_volatility: 0.0,
            ewma_alpha,
            ewma_initialized: false,
            tick_count: 0,
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
            self.precision_rejections.load(Ordering::Relaxed),
            self.precision_warnings.load(Ordering::Relaxed),
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
    pub fn get_dynamic_hedge_ratio(&self) -> Option<f64> {
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

                // CF3 FIX: Enforce hard limits on price ratios
                let ratio = v1 / v2;
                if !(MIN_SAFE_PRICE_RATIO..=MAX_SAFE_PRICE_RATIO).contains(&ratio) {
                    let count = self
                        .precision_rejections
                        .fetch_add(1, Ordering::Relaxed)
                        + 1;
                    crate::metrics::record_precision_rejection(&format!(
                        "{}/{}",
                        leg1.symbol, leg2.symbol
                    ));
                    error!(
                        "PRECISION ERROR: Price ratio {:.2e} exceeds safe f64 bounds [{:.2e}, {:.2e}]. Rejecting signal. (Total rejections: {})",
                        ratio, MIN_SAFE_PRICE_RATIO, MAX_SAFE_PRICE_RATIO, count
                    );
                    return Signal::Hold;
                }

                // Log warning for ratios approaching the limit
                if !(MIN_SAFE_PRICE_RATIO * 10.0..=MAX_SAFE_PRICE_RATIO / 10.0).contains(&ratio) {
                    let count = self
                        .precision_warnings
                        .fetch_add(1, Ordering::Relaxed)
                        + 1;
                    crate::metrics::record_precision_warning(&format!(
                        "{}/{}",
                        leg1.symbol, leg2.symbol
                    ));
                    if count == 1 || count.is_multiple_of(PRECISION_WARNING_LOG_INTERVAL) {
                        warn!(
                            "PRECISION WARNING: Price ratio {:.2e} approaching safety limits. (Total warnings: {})",
                            ratio, count
                        );
                    }
                }

                (v1, v2)
            }
            _ => {
                error!(
                    "PRECISION ERROR: Invalid prices for Pairs analysis: {:?}, {:?}",
                    leg1.price, leg2.price
                );
                return Signal::Hold;
            }
        };

        // P2: Update Kalman Filter and compute spread
        let spread = if let Some(ref mut kalman) = self.kalman {
            let ln_p1 = p1.ln();
            let ln_p2 = p2.ln();
            let beta = kalman.update(ln_p2, ln_p1);
            ln_p1 - beta * ln_p2
        } else {
            p1.ln() - p2.ln()
        };

        // O(1) sliding window statistics via running sums
        self.running_sum += spread;
        self.running_sq_sum += spread * spread;
        self.spread_history.push_back(spread);

        // MC-1 FIX: Periodic recalculation to prevent f64 drift
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
        let variance = (self.running_sq_sum / n - mean * mean).max(0.0);
        let std_dev = variance.sqrt();

        // Phase 2: Update EWMA volatility estimate
        if !self.ewma_initialized {
            self.ewma_volatility = std_dev;
            self.ewma_initialized = true;
        } else {
            self.ewma_volatility =
                self.ewma_alpha * std_dev + (1.0 - self.ewma_alpha) * self.ewma_volatility;
        }

        // Phase 2: Compute adaptive Z-score using EWMA volatility
        const MIN_VOLATILITY: f64 = 1e-12;
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
