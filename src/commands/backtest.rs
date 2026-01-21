//! Backtest command handler.
//!
//! Implements the `backtest` subcommand for running backtests
//! on historical data with configurable strategies and data sources.

use crate::backtest::{self, BacktestConfig};
use crate::cli::{BacktestCliConfig, BacktestStrategyType};
use crate::strategy::moving_average::MovingAverageCrossover;

use polars::prelude::*;
use rust_decimal_macros::dec;
use serde::Serialize;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use tracing::{error, info, warn};

/// Backtest results in JSON-serializable format.
#[derive(Debug, Serialize)]
struct BacktestOutput {
    strategy: String,
    exchange: String,
    symbols: Vec<String>,
    duration: String,
    initial_capital: String,
    final_capital: String,
    net_profit: String,
    return_pct: String,
    total_trades: u32,
    winning_trades: u32,
    losing_trades: u32,
    win_rate_pct: String,
    max_drawdown_pct: String,
}

/// Run a backtest with the provided CLI configuration.
///
/// Supports:
/// - Multiple strategies (moving-average, dual-leg)
/// - Multiple data sources (CSV files, synthetic)
/// - JSON output for CI integration
///
/// # Errors
/// Returns error if data loading or backtest fails.
pub fn run_backtest(config: BacktestCliConfig) -> Result<(), Box<dyn std::error::Error>> {
    info!("--- Running Backtest ---");
    info!(
        strategy = %format!("{:?}", config.strategy),
        exchange = %config.exchange,
        symbols = ?config.symbols,
        duration = %config.duration,
        synthetic = config.synthetic,
        "Backtest configuration"
    );

    // Parse duration to candle count
    let candle_count = config.duration_to_candles()?;

    // Get primary symbol for data loading
    let primary_symbol = config.primary_symbol()?;

    // Load data
    let df = if config.synthetic {
        generate_synthetic_data(primary_symbol, candle_count)?
    } else {
        load_csv_data(primary_symbol, candle_count)?
    };

    info!(rows = df.height(), "Data loaded");

    // Run appropriate strategy
    let result = match config.strategy {
        BacktestStrategyType::MovingAverage => {
            let strategy = MovingAverageCrossover::new(5, 20);
            let bt_config = BacktestConfig::with_capital(config.initial_capital);
            backtest::run(&strategy, &df, &bt_config)?
        }
        BacktestStrategyType::DualLeg => {
            // For dual-leg, we use the same MA strategy as a placeholder
            // TODO: Implement a proper pairs backtest strategy
            warn!("dual-leg backtest not fully implemented, using moving-average as fallback");
            let strategy = MovingAverageCrossover::new(5, 20);
            let bt_config = BacktestConfig::with_capital(config.initial_capital);
            backtest::run(&strategy, &df, &bt_config)?
        }
    };

    // Print results
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

    // Write output JSON
    let output = BacktestOutput {
        strategy: format!("{:?}", config.strategy),
        exchange: config.exchange.clone(),
        symbols: config.symbols.clone(),
        duration: config.duration.clone(),
        initial_capital: result.initial_capital.to_string(),
        final_capital: result.final_capital.to_string(),
        net_profit: result.net_profit.to_string(),
        return_pct: result.return_percentage().to_string(),
        total_trades: result.total_trades,
        winning_trades: result.winning_trades,
        losing_trades: result.losing_trades,
        win_rate_pct: (result.win_rate() * dec!(100)).to_string(),
        max_drawdown_pct: (result.max_drawdown * dec!(100)).to_string(),
    };

    // Create output directory and write results
    fs::create_dir_all(&config.output_dir)?;
    let output_path = Path::new(&config.output_dir).join("results.json");
    let mut file = File::create(&output_path)?;
    let json = serde_json::to_string_pretty(&output)?;
    file.write_all(json.as_bytes())?;
    info!(path = %output_path.display(), "Results written");

    Ok(())
}

/// Load historical data from a CSV file.
fn load_csv_data(symbol: &str, max_rows: usize) -> Result<DataFrame, Box<dyn std::error::Error>> {
    // Try multiple paths: data/{symbol}.csv, sample_data.csv
    let paths = [
        format!("data/{}.csv", symbol),
        format!("data/{}.csv", symbol.to_lowercase()),
        "sample_data.csv".to_string(),
    ];

    for path in &paths {
        if Path::new(path).exists() {
            info!(path = %path, "Loading CSV data");
            let file = File::open(path)?;
            let df = CsvReader::new(file).finish()?;

            // Limit rows if necessary
            let df = if df.height() > max_rows {
                df.slice(0, max_rows)
            } else {
                df
            };

            return Ok(df);
        }
    }

    error!("No CSV data found for symbol: {}", symbol);
    Err(format!(
        "No CSV data found. Tried: {:?}. Use --synthetic for CI mode.",
        paths
    )
    .into())
}

/// Generate synthetic price data for testing.
fn generate_synthetic_data(
    symbol: &str,
    candle_count: usize,
) -> Result<DataFrame, Box<dyn std::error::Error>> {
    info!(
        symbol = %symbol,
        candles = candle_count,
        "Generating synthetic data"
    );

    // NOTE: f64 is acceptable for synthetic data generation (test-only code).
    // This data never touches real trading logic. Precision loss is irrelevant.
    let mut prices = Vec::with_capacity(candle_count);
    let mut price = 100.0_f64;

    // Use a simple pseudo-random sequence for reproducibility
    // (seeded by symbol hash for different patterns per symbol)
    let seed: u64 = symbol.bytes().map(|b| b as u64).sum();
    let mut state = seed;

    for _ in 0..candle_count {
        // Simple LCG random number generator
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let rand = ((state >> 33) as f64) / (u32::MAX as f64) - 0.5;

        // Add some trend and mean reversion
        let drift = 0.0001; // Slight upward drift
        let volatility = 0.02; // 2% daily volatility
        let change = drift + volatility * rand;

        price *= 1.0 + change;
        price = price.max(1.0); // Floor at $1
        prices.push(price);
    }

    let df = df! {
        "close" => prices
    }?;

    Ok(df)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_data_generation() {
        let df = generate_synthetic_data("BTC-USD", 100).unwrap();
        assert_eq!(df.height(), 100);
        assert!(df.column("close").is_ok());
    }

    #[test]
    fn test_duration_parsing() {
        let config = BacktestCliConfig {
            strategy: BacktestStrategyType::MovingAverage,
            exchange: "coinbase".to_string(),
            symbols: vec!["BTC-USD".to_string()],
            duration: "7d".to_string(),
            output_dir: "test_output".to_string(),
            initial_capital: dec!(10000),
            synthetic: true,
        };
        assert_eq!(config.duration_to_candles().unwrap(), 7 * 24);

        let config_year = BacktestCliConfig {
            duration: "1y".to_string(),
            ..config.clone()
        };
        assert_eq!(config_year.duration_to_candles().unwrap(), 365 * 24);

        // Test invalid format returns error
        let config_invalid = BacktestCliConfig {
            duration: "invalid".to_string(),
            ..config.clone()
        };
        assert!(config_invalid.duration_to_candles().is_err());
    }
}
