//! CLI configuration structs bridging CLI arguments to domain types.
//!
//! These structs decouple the CLI parsing layer from the business logic,
//! allowing command handlers to work with validated, typed configurations.

use crate::exchange::coinbase::AppEnv;
use rust_decimal::Decimal;
use thiserror::Error;

/// Configuration for the simple Moving Average Crossover trading engine.
#[derive(Debug, Clone)]
pub struct SimpleTradingConfig {
    /// Trading product (e.g., "BTC-USD")
    pub product_id: String,
    /// Duration in seconds between trade cycles
    pub duration: u64,
    /// Order size in base currency
    pub order_size: Decimal,
    /// Short moving average window
    pub short_window: usize,
    /// Long moving average window
    pub long_window: usize,
    /// Maximum price history to keep
    pub max_history: usize,
    /// Trading environment (Live or Paper)
    pub env: AppEnv,
}

/// CLI configuration for the Dual-Leg trading strategy.
///
/// This struct captures CLI arguments before conversion to domain types,
/// allowing validation and default handling at the command layer.
#[derive(Debug, Clone)]
pub struct DualLegCliConfig {
    /// Order size in base currency (f64 for CLI compatibility)
    pub order_size: f64,
    /// Maximum age of market ticks in milliseconds
    pub max_tick_age_ms: i64,
    /// Execution timeout in milliseconds
    pub execution_timeout_ms: i64,
    /// Minimum profit threshold to exit positions
    pub min_profit_threshold: f64,
    /// Stop loss threshold (negative value)
    pub stop_loss_threshold: f64,
    /// Log throttle interval in seconds
    pub throttle_interval_secs: u64,
}

/// Strategy types available for backtesting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BacktestStrategyType {
    /// Moving Average Crossover strategy
    MovingAverage,
    /// Dual-leg pairs/basis trading strategy
    DualLeg,
}

impl std::str::FromStr for BacktestStrategyType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "moving-average" | "ma" => Ok(Self::MovingAverage),
            "dual-leg" | "pairs" | "basis" => Ok(Self::DualLeg),
            _ => Err(format!(
                "Unknown strategy: '{}'. Use 'moving-average' or 'dual-leg'",
                s
            )),
        }
    }
}

/// Errors that can occur when parsing backtest configuration.
#[derive(Debug, Error)]
pub enum BacktestConfigError {
    #[error("Invalid duration format: '{0}'. Expected format: 7d, 30d, 1y, 168h")]
    InvalidDurationFormat(String),

    #[error("Invalid number in duration '{0}': {1}")]
    InvalidDurationNumber(String, std::num::ParseIntError),

    #[error("At least one symbol is required")]
    EmptySymbols,

    #[error("Invalid initial capital: {0}")]
    InvalidCapital(String),
}

/// CLI configuration for backtesting.
#[derive(Debug, Clone)]
pub struct BacktestCliConfig {
    /// Strategy type to backtest
    pub strategy: BacktestStrategyType,
    /// Exchange context (coinbase, alpaca)
    pub exchange: String,
    /// Symbols to backtest (parsed from comma-separated string)
    pub symbols: Vec<String>,
    /// Backtest duration string (e.g., "7d", "30d")
    pub duration: String,
    /// Output directory for results
    pub output_dir: String,
    /// Initial capital in USD
    pub initial_capital: Decimal,
    /// Use synthetic data (no CSV files required)
    pub synthetic: bool,
}

impl BacktestCliConfig {
    /// Parse duration string to number of hourly candles.
    ///
    /// # Supported Formats
    /// - `7d` → 7 days × 24 hours = 168 candles
    /// - `1y` → 365 days × 24 hours = 8760 candles
    /// - `168h` → 168 hourly candles
    ///
    /// # Errors
    /// Returns `BacktestConfigError` if format is unrecognized or number invalid.
    pub fn duration_to_candles(&self) -> Result<usize, BacktestConfigError> {
        const HOURS_PER_DAY: usize = 24;
        const DAYS_PER_YEAR: usize = 365;

        let s = self.duration.trim().to_lowercase();

        let (value_str, multiplier) = if let Some(d) = s.strip_suffix('d') {
            (d, HOURS_PER_DAY)
        } else if let Some(y) = s.strip_suffix('y') {
            (y, DAYS_PER_YEAR * HOURS_PER_DAY)
        } else if let Some(h) = s.strip_suffix('h') {
            (h, 1)
        } else {
            return Err(BacktestConfigError::InvalidDurationFormat(
                self.duration.clone(),
            ));
        };

        let value: usize = value_str
            .parse()
            .map_err(|e| BacktestConfigError::InvalidDurationNumber(self.duration.clone(), e))?;

        Ok(value.saturating_mul(multiplier))
    }

    /// Get the primary symbol, returning an error if symbols is empty.
    pub fn primary_symbol(&self) -> Result<&str, BacktestConfigError> {
        self.symbols
            .first()
            .map(|s| s.as_str())
            .ok_or(BacktestConfigError::EmptySymbols)
    }
}
