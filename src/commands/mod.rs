//! CLI command handlers.
//!
//! This module contains the implementation for each CLI subcommand,
//! delegating to the appropriate trading engines and pipelines.

mod backtest;
mod discover;
mod dual_leg;
mod portfolio;
mod trade;

pub use backtest::run_backtest;
pub use discover::run_discover_pairs;
pub use dual_leg::run_dual_leg_trading;
pub use portfolio::run_portfolio;
pub use trade::run_trade;
