//! CLI argument parsing using clap.
//!
//! This module defines the command-line interface for AlgoPioneer,
//! including all subcommands and their arguments.

mod config;

pub use config::{
    BacktestCliConfig, BacktestConfigError, BacktestStrategyType, DualLegCliConfig,
    SimpleTradingConfig,
};

use clap::{Parser, Subcommand};

/// AlgoPioneer - Algorithmic Trading System
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// The subcommand to execute
    #[command(subcommand)]
    pub command: Commands,

    /// Set the verbosity level (error, warn, info, debug, trace)
    #[arg(long, global = true, default_value = "info")]
    pub verbose: String,
}

/// Available CLI commands
#[derive(Subcommand)]
pub enum Commands {
    /// Run the live trading or sandbox bot (Moving Average Crossover)
    Trade {
        /// The product to trade (e.g., "BTC-USD")
        #[arg(short, long)]
        product_id: String,
        /// Duration in seconds to wait between trade cycles
        #[arg(short, long, default_value_t = 60)]
        duration: u64,
        /// Run in paper trading mode (simulated execution)
        #[arg(long, default_value_t = false)]
        paper: bool,
        /// Order size in base currency
        #[arg(long, default_value_t = 0.001)]
        order_size: f64,
        /// Short moving average window
        #[arg(long, default_value_t = 5)]
        short_window: usize,
        /// Long moving average window
        #[arg(long, default_value_t = 20)]
        long_window: usize,
        /// Maximum history to keep
        #[arg(long, default_value_t = 200)]
        max_history: usize,
    },

    /// Run a backtest on historical data
    Backtest {
        /// Strategy to backtest: 'moving-average' or 'dual-leg'
        #[arg(long, default_value = "moving-average")]
        strategy: String,
        /// Exchange context: 'coinbase' or 'alpaca'
        #[arg(long, default_value = "coinbase")]
        exchange: String,
        /// Symbols to backtest (comma-separated, e.g., "BTC-USD" or "BTC-USD,ETH-USD")
        #[arg(long, default_value = "BTC-USD")]
        symbols: String,
        /// Backtest duration (e.g., "7d", "30d", "1y")
        #[arg(long, default_value = "7d")]
        duration: String,
        /// Output directory for results
        #[arg(long, default_value = "backtest_results")]
        output_dir: String,
        /// Initial capital in USD
        #[arg(long, default_value_t = 10000.0)]
        initial_capital: f64,
        /// Use synthetic data for CI (no CSV files required)
        #[arg(long, default_value_t = false)]
        synthetic: bool,
    },

    /// Run the Dual-Leg Trading Strategy (Basis or Pairs)
    DualLeg {
        /// Strategy type: 'basis' or 'pairs'
        #[arg(long)]
        strategy: String,
        /// Trading symbols (comma separated, e.g., "BTC-USD,BTC-USDT")
        #[arg(long)]
        symbols: String,
        /// Exchange to use: coinbase, kraken
        #[arg(long, default_value = "coinbase")]
        exchange: String,
        /// Run in paper trading mode (simulated execution)
        #[arg(long, default_value_t = false)]
        paper: bool,
        /// Order size in base currency
        #[arg(long, default_value_t = 0.00001)]
        order_size: f64,
        /// Maximum tick age in milliseconds before dropping
        #[arg(long, default_value_t = 2000)]
        max_tick_age_ms: i64,
        /// Execution timeout in milliseconds
        #[arg(long, default_value_t = 30000)]
        execution_timeout_ms: i64,
        /// Minimum profit threshold for exits
        #[arg(long, default_value_t = 0.005)]
        min_profit_threshold: f64,
        /// Stop loss threshold (negative value)
        #[arg(long, default_value_t = -0.05)]
        stop_loss_threshold: f64,
        /// Log throttle interval in seconds
        #[arg(long, default_value_t = 5)]
        throttle_interval_secs: u64,
    },

    /// Run the Portfolio Manager (multiple pairs with supervisor)
    Portfolio {
        /// Path to pairs configuration file
        #[arg(long)]
        config: String,
        /// Exchange to use: coinbase, alpaca
        #[arg(long, default_value = "coinbase")]
        exchange: String,
        /// Run in paper trading mode
        #[arg(long, default_value_t = false)]
        paper: bool,
    },

    /// Discover and optimize cointegrated trading pairs automatically
    DiscoverPairs {
        /// Exchange to use: "coinbase" (crypto) or "alpaca" (stocks)
        #[arg(long, default_value = "coinbase")]
        exchange: String,
        /// Symbols to analyze (comma-separated, or "default" for built-in list)
        #[arg(long, default_value = "default")]
        symbols: String,
        /// Minimum Pearson correlation threshold
        #[arg(long, default_value_t = 0.8)]
        min_correlation: f64,
        /// Maximum half-life in hours for mean reversion
        #[arg(long, default_value_t = 48.0)]
        max_half_life: f64,
        /// Minimum Sharpe ratio to include in results
        #[arg(long, default_value_t = 0.5)]
        min_sharpe: f64,
        /// Historical lookback period in days
        #[arg(long, default_value_t = 90)]
        lookback_days: u32,
        /// Maximum number of pairs to output
        #[arg(long, default_value_t = 10)]
        max_pairs: usize,
        /// Output file path for discovered pairs JSON
        #[arg(long, default_value = "discovered_pairs.json")]
        output: String,
        /// Initial capital for backtests in USD
        #[arg(long, default_value_t = 10000.0)]
        initial_capital: f64,
        /// Skip the ADF cointegration test (use for exploratory analysis)
        #[arg(long, default_value_t = false)]
        no_cointegration: bool,
    },
}
