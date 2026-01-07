//! CLI configuration structs bridging CLI arguments to domain types.
//!
//! These structs decouple the CLI parsing layer from the business logic,
//! allowing command handlers to work with validated, typed configurations.

use crate::exchange::coinbase::AppEnv;
use rust_decimal::Decimal;

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
