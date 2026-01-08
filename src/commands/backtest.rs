//! Backtest command handler.
//!
//! Implements the `backtest` subcommand for running backtests
//! on historical data.

use crate::backtest::{self, BacktestConfig};
use crate::strategy::moving_average::MovingAverageCrossover;

use polars::prelude::*;
use rust_decimal_macros::dec;
use std::fs::File;
use tracing::info;

/// Run a backtest on historical data.
///
/// Loads data from `sample_data.csv` and runs the Moving Average
/// Crossover strategy against it.
///
/// # Errors
/// Returns error if data loading or backtest fails.
pub fn run_backtest() -> Result<(), Box<dyn std::error::Error>> {
    info!("--- Running Backtest for Moving Average Crossover Strategy ---");

    // Load historical data from CSV
    let file = File::open("sample_data.csv")?;
    let df = CsvReader::new(file).finish()?;

    // Create the strategy
    let strategy = MovingAverageCrossover::new(5, 10);

    // Run the backtest with configuration
    let config = BacktestConfig::with_capital(dec!(1000));
    let result = backtest::run(&strategy, &df, &config)?;

    // Print the results
    info!("--- Backtest Results ---");
    info!("Initial Capital: ${}", result.initial_capital);
    info!("Final Capital:   ${}", result.final_capital);
    info!("Net Profit:      ${}", result.net_profit);
    info!("Return:          {}%", result.return_percentage());
    info!("Total Trades:    {}", result.total_trades);
    info!("Winning Trades:  {}", result.winning_trades);
    info!("Losing Trades:   {}", result.losing_trades);
    info!("Win Rate:        {}%", result.win_rate() * dec!(100));
    info!("Max Drawdown:    {}%", result.max_drawdown * dec!(100));
    info!("------------------------");

    Ok(())
}
