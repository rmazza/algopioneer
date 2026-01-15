//! Daily Risk Engine
//!
//! Portfolio-level daily loss limit enforcement.
//!
//! # Architecture
//!
//! - Tracks cumulative realized PnL for the trading day
//! - Halts trading when daily loss threshold is breached
//! - Thread-safe for concurrent strategy updates
//!
//! # Example
//!
//! ```ignore
//! use algopioneer::risk::{DailyRiskEngine, DailyRiskConfig};
//! use rust_decimal_macros::dec;
//!
//! let engine = DailyRiskEngine::new(DailyRiskConfig {
//!     max_daily_loss: dec!(-500),  // Halt at -$500
//!     warning_threshold: dec!(-300),  // Warn at -$300
//! });
//!
//! // After each trade
//! engine.record_pnl(dec!(-50));
//!
//! // Before placing new orders
//! if !engine.is_trading_enabled() {
//!     warn!("Daily loss limit reached - trading halted");
//!     return;
//! }
//! ```

use rust_decimal::Decimal;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use tracing::{error, info, warn};

/// Configuration for daily risk limits.
#[derive(Debug, Clone)]
pub struct DailyRiskConfig {
    /// Maximum allowed daily loss (negative value, e.g., -500.0)
    pub max_daily_loss: Decimal,
    /// Warning threshold before halt (negative value, e.g., -300.0)
    pub warning_threshold: Decimal,
}

impl Default for DailyRiskConfig {
    fn default() -> Self {
        Self {
            max_daily_loss: Decimal::new(-500, 0),    // -$500
            warning_threshold: Decimal::new(-300, 0), // -$300 warning
        }
    }
}

impl DailyRiskConfig {
    /// Conservative config for paper trading
    pub fn paper_trading() -> Self {
        Self {
            max_daily_loss: Decimal::new(-100, 0),   // -$100
            warning_threshold: Decimal::new(-50, 0), // -$50 warning
        }
    }

    /// Config for live trading with larger limits
    pub fn live_trading(max_loss: Decimal) -> Self {
        Self {
            max_daily_loss: max_loss,
            warning_threshold: max_loss / Decimal::new(2, 0), // 50% of max
        }
    }
}

/// Status returned from risk checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskStatus {
    /// Trading allowed, within normal limits
    Normal,
    /// Warning threshold breached, trading still allowed
    Warning,
    /// Daily limit breached, trading halted
    Halted,
}

impl std::fmt::Display for RiskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Normal => write!(f, "Normal"),
            Self::Warning => write!(f, "Warning"),
            Self::Halted => write!(f, "Halted"),
        }
    }
}

/// Thread-safe daily risk engine.
///
/// Tracks cumulative PnL and enforces daily loss limits.
/// Uses atomic operations for lock-free concurrent access.
///
/// # Precision
///
/// Internally stores PnL as i64 micros (1e-6 precision) to enable
/// atomic operations. This supports values up to ±9.2 quadrillion micros
/// (±$9.2 trillion), sufficient for any trading scenario.
pub struct DailyRiskEngine {
    config: DailyRiskConfig,
    /// Cumulative PnL in micros (Decimal * 1_000_000)
    realized_pnl_micros: AtomicI64,
    /// Whether trading is enabled
    trading_enabled: AtomicBool,
    /// Whether warning has been issued (to avoid spam)
    warning_issued: AtomicBool,
}

impl DailyRiskEngine {
    /// Create a new DailyRiskEngine with the given config.
    pub fn new(config: DailyRiskConfig) -> Self {
        info!(
            max_daily_loss = %config.max_daily_loss,
            warning_threshold = %config.warning_threshold,
            "DailyRiskEngine initialized"
        );
        Self {
            config,
            realized_pnl_micros: AtomicI64::new(0),
            trading_enabled: AtomicBool::new(true),
            warning_issued: AtomicBool::new(false),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(DailyRiskConfig::default())
    }

    /// Record a PnL event and check thresholds.
    ///
    /// Returns the new risk status after recording.
    pub fn record_pnl(&self, pnl: Decimal) -> RiskStatus {
        // Convert Decimal to micros (multiply by 1_000_000)
        let pnl_micros = decimal_to_micros(pnl);

        // Atomically add to cumulative PnL
        let new_total_micros = self
            .realized_pnl_micros
            .fetch_add(pnl_micros, Ordering::SeqCst)
            + pnl_micros;
        let new_total = micros_to_decimal(new_total_micros);

        // Check thresholds
        if new_total <= self.config.max_daily_loss {
            // CRITICAL: Daily loss limit breached
            self.trading_enabled.store(false, Ordering::SeqCst);

            error!(
                daily_pnl = %new_total,
                max_daily_loss = %self.config.max_daily_loss,
                "CRITICAL: DAILY LOSS LIMIT BREACHED - TRADING HALTED"
            );

            RiskStatus::Halted
        } else if new_total <= self.config.warning_threshold {
            // Warning threshold
            if !self.warning_issued.swap(true, Ordering::SeqCst) {
                warn!(
                    daily_pnl = %new_total,
                    warning_threshold = %self.config.warning_threshold,
                    max_daily_loss = %self.config.max_daily_loss,
                    "Daily loss warning threshold reached"
                );
            }
            RiskStatus::Warning
        } else {
            RiskStatus::Normal
        }
    }

    /// Check if trading is currently enabled.
    #[must_use]
    pub fn is_trading_enabled(&self) -> bool {
        self.trading_enabled.load(Ordering::SeqCst)
    }

    /// Get current risk status without modifying state.
    #[must_use]
    pub fn status(&self) -> RiskStatus {
        if !self.is_trading_enabled() {
            RiskStatus::Halted
        } else {
            let current = self.current_pnl();
            if current <= self.config.warning_threshold {
                RiskStatus::Warning
            } else {
                RiskStatus::Normal
            }
        }
    }

    /// Get current cumulative PnL for the day.
    #[must_use]
    pub fn current_pnl(&self) -> Decimal {
        micros_to_decimal(self.realized_pnl_micros.load(Ordering::SeqCst))
    }

    /// Reset for a new trading day.
    ///
    /// Call this at market open or midnight to start fresh.
    pub fn reset_daily(&self) {
        let old_pnl = self.current_pnl();
        self.realized_pnl_micros.store(0, Ordering::SeqCst);
        self.trading_enabled.store(true, Ordering::SeqCst);
        self.warning_issued.store(false, Ordering::SeqCst);

        info!(
            previous_day_pnl = %old_pnl,
            "Daily risk counters reset"
        );
    }

    /// Manually re-enable trading (after human review).
    ///
    /// Use with caution - bypasses safety limits.
    pub fn force_enable_trading(&self) {
        warn!("Trading manually re-enabled - daily limits bypassed");
        self.trading_enabled.store(true, Ordering::SeqCst);
    }

    /// Get the configured max daily loss.
    #[must_use]
    pub fn max_daily_loss(&self) -> Decimal {
        self.config.max_daily_loss
    }

    /// Get remaining loss capacity before halt.
    #[must_use]
    pub fn remaining_loss_capacity(&self) -> Decimal {
        self.current_pnl() - self.config.max_daily_loss
    }
}

impl Default for DailyRiskEngine {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Convert Decimal to micros (i64) for atomic storage.
fn decimal_to_micros(d: Decimal) -> i64 {
    // Multiply by 1_000_000 and truncate to i64
    let scaled = d * Decimal::new(1_000_000, 0);
    scaled.mantissa() as i64 / 10i64.pow(scaled.scale())
}

/// Convert micros (i64) back to Decimal.
fn micros_to_decimal(micros: i64) -> Decimal {
    Decimal::new(micros, 6) // 6 decimal places
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_pnl_tracking() {
        let engine = DailyRiskEngine::new(DailyRiskConfig {
            max_daily_loss: dec!(-100),
            warning_threshold: dec!(-50),
        });

        // Record some gains
        assert_eq!(engine.record_pnl(dec!(25)), RiskStatus::Normal);
        assert_eq!(engine.current_pnl(), dec!(25));

        // Record some losses
        assert_eq!(engine.record_pnl(dec!(-30)), RiskStatus::Normal);
        // 25 - 30 = -5, still above warning
        assert!(engine.current_pnl() < dec!(0));
    }

    #[test]
    fn test_warning_threshold() {
        let engine = DailyRiskEngine::new(DailyRiskConfig {
            max_daily_loss: dec!(-100),
            warning_threshold: dec!(-50),
        });

        // Push past warning
        assert_eq!(engine.record_pnl(dec!(-60)), RiskStatus::Warning);
        assert!(engine.is_trading_enabled());
    }

    #[test]
    fn test_halt_on_breach() {
        let engine = DailyRiskEngine::new(DailyRiskConfig {
            max_daily_loss: dec!(-100),
            warning_threshold: dec!(-50),
        });

        // Push past limit
        assert_eq!(engine.record_pnl(dec!(-110)), RiskStatus::Halted);
        assert!(!engine.is_trading_enabled());
    }

    #[test]
    fn test_reset_daily() {
        let engine = DailyRiskEngine::new(DailyRiskConfig {
            max_daily_loss: dec!(-100),
            warning_threshold: dec!(-50),
        });

        // Breach limit
        engine.record_pnl(dec!(-150));
        assert!(!engine.is_trading_enabled());

        // Reset
        engine.reset_daily();
        assert!(engine.is_trading_enabled());
        assert_eq!(engine.current_pnl(), dec!(0));
    }

    #[test]
    fn test_remaining_capacity() {
        let engine = DailyRiskEngine::new(DailyRiskConfig {
            max_daily_loss: dec!(-100),
            warning_threshold: dec!(-50),
        });

        // At 0 PnL, capacity is 100
        assert_eq!(engine.remaining_loss_capacity(), dec!(100));

        // After -30, capacity is 70
        engine.record_pnl(dec!(-30));
        assert_eq!(engine.remaining_loss_capacity(), dec!(70));
    }

    #[test]
    fn test_micros_conversion_roundtrip() {
        let values = vec![dec!(0), dec!(100.123456), dec!(-50.5), dec!(999999.999999)];

        for v in values {
            let micros = decimal_to_micros(v);
            let back = micros_to_decimal(micros);
            // Allow for minor rounding at 6 decimal places
            assert!(
                (v - back).abs() < dec!(0.000001),
                "Failed for {}: got {}",
                v,
                back
            );
        }
    }
}
