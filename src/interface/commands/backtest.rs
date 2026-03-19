//! Backtest command handler.
//!
//! Implements the `backtest` subcommand for running backtests
//! on historical data with configurable strategies and data sources.

use crate::application::backtest::{self, BacktestConfig, BacktestResult};
use crate::application::strategy::dual_leg::PairsManager;
use crate::application::strategy::moving_average::MovingAverageCrossover;
use crate::interface::cli::{BacktestCliConfig, BacktestStrategyType};

use polars::prelude::*;
use rust_decimal_macros::dec;
use serde::Serialize;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use thiserror::Error;
use tracing::{error, info};

/// Errors that can occur during backtesting.
#[derive(Debug, Error)]
pub enum BacktestError {
    #[error("Configuration error: {0}")]
    Config(#[from] crate::interface::cli::BacktestConfigError),

    #[error("Data loading error: {0}")]
    Data(String),

    #[error("Strategy execution error: {0}")]
    Execution(#[from] crate::application::backtest::BacktestError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("CSV parsing error: {0}")]
    Csv(#[from] polars::error::PolarsError),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Dual-leg strategy backtesting is not yet implemented")]
    NotImplemented,
}

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
    sharpe_ratio: String,
    sortino_ratio: String,
    profit_factor: String,
    expectancy: String,
}

/// Run a backtest with the provided CLI configuration.
pub async fn run_backtest(config: BacktestCliConfig) -> Result<(), BacktestError> {
    info!("--- Running Backtest ---");
    info!(
        strategy = %format!("{:?}", config.strategy),
        exchange = %config.exchange,
        symbols = ?config.symbols,
        duration = %config.duration,
        synthetic = config.synthetic,
        "Backtest configuration"
    );

    let candle_count = config.duration_to_candles()?;

    // Result of the backtest
    let result = match config.strategy {
        BacktestStrategyType::MovingAverage => {
            let primary_symbol = config.primary_symbol()?;
            let df = if config.synthetic {
                generate_synthetic_data(primary_symbol, candle_count)?
            } else {
                load_csv_data(primary_symbol, candle_count)?
            };
            info!(rows = df.height(), "Data loaded for {}", primary_symbol);

            let strategy = MovingAverageCrossover::new(5, 20);
            let bt_config = BacktestConfig::with_capital(config.initial_capital);
            backtest::run(&strategy, &df, &bt_config)?
        }
        BacktestStrategyType::DualLeg => {
            if config.symbols.len() < 2 {
                return Err(BacktestError::Data(
                    "Dual-leg strategy requires exactly 2 symbols".to_string(),
                ));
            }
            let s1 = &config.symbols[0];
            let s2 = &config.symbols[1];

            let df1 = if config.synthetic {
                generate_synthetic_data(s1, candle_count)?
            } else {
                load_csv_data(s1, candle_count)?
            };
            let df2 = if config.synthetic {
                generate_synthetic_data(s2, candle_count)?
            } else {
                load_csv_data(s2, candle_count)?
            };

            info!(
                rows1 = df1.height(),
                rows2 = df2.height(),
                "Data loaded for {} and {}",
                s1,
                s2
            );

            // Create a PairsManager for signal generation
            // Using default parameters for the backtest
            let mut strategy = PairsManager::new("BACKTEST".to_string(), 20, 2.0, 0.5);
            let bt_config = BacktestConfig::with_capital(config.initial_capital);
            backtest::run_dual(&mut strategy, &df1, &df2, &bt_config).await?
        }
    };

    // Print results
    print_backtest_results(&result);

    // Write output files
    write_backtest_outputs(&result, &config)?;

    Ok(())
}

fn print_backtest_results(result: &BacktestResult) {
    info!("--- Backtest Results ---");
    info!("Initial Capital: ${}", result.initial_capital);
    info!("Final Capital:   ${}", result.final_capital);
    info!("Net Profit:      ${}", result.net_profit);
    info!("Return:          {}%", result.return_percentage());
    info!("Sharpe Ratio:    {}", result.sharpe_ratio);
    info!("Sortino Ratio:   {}", result.sortino_ratio);
    info!("Profit Factor:   {}", result.profit_factor);
    info!("Expectancy:      {}", result.expectancy);
    info!("Total Trades:    {}", result.total_trades);
    info!("Win Rate:        {}%", result.win_rate() * dec!(100));
    info!("Max Drawdown:    {}%", result.max_drawdown * dec!(100));
    info!("------------------------");
}

fn write_backtest_outputs(
    result: &BacktestResult,
    config: &BacktestCliConfig,
) -> Result<(), BacktestError> {
    fs::create_dir_all(&config.output_dir)?;

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
        sharpe_ratio: result.sharpe_ratio.to_string(),
        sortino_ratio: result.sortino_ratio.to_string(),
        profit_factor: result.profit_factor.to_string(),
        expectancy: result.expectancy.to_string(),
    };

    let summary_path = Path::new(&config.output_dir).join("results.json");
    let mut summary_file = File::create(&summary_path)?;
    summary_file.write_all(serde_json::to_string_pretty(&output)?.as_bytes())?;
    info!(path = %summary_path.display(), "Summary written");

    let trades_path = Path::new(&config.output_dir).join("trades.csv");
    let mut trades_file = File::create(&trades_path)?;
    writeln!(
        trades_file,
        "entry_idx,exit_idx,entry_price,exit_price,size,pnl,pnl_pct"
    )?;
    for t in &result.trades {
        writeln!(
            trades_file,
            "{},{},{},{},{},{},{}",
            t.entry_idx, t.exit_idx, t.entry_price, t.exit_price, t.size, t.pnl, t.pnl_pct
        )?;
    }
    info!(path = %trades_path.display(), "Trades log written");

    let equity_path = Path::new(&config.output_dir).join("equity.csv");
    let mut equity_file = File::create(&equity_path)?;
    writeln!(equity_file, "step,equity")?;
    for (i, e) in result.equity_curve.iter().enumerate() {
        writeln!(equity_file, "{},{}", i, e)?;
    }
    info!(path = %equity_path.display(), "Equity curve written");

    Ok(())
}

fn load_csv_data(symbol: &str, max_rows: usize) -> Result<DataFrame, BacktestError> {
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

            let df = if df.height() > max_rows {
                df.slice(0, max_rows)
            } else {
                df
            };

            return Ok(df);
        }
    }

    error!("No CSV data found for symbol: {}", symbol);
    Err(BacktestError::Data(format!(
        "No CSV data found. Tried: {:?}. Use --synthetic for CI mode.",
        paths
    )))
}

fn generate_synthetic_data(symbol: &str, candle_count: usize) -> Result<DataFrame, BacktestError> {
    info!(
        symbol = %symbol,
        candles = candle_count,
        "Generating synthetic data"
    );

    let mut prices = Vec::with_capacity(candle_count);
    let mut price = 100.0_f64;

    let seed: u64 = symbol.bytes().map(|b| b as u64).sum();
    let mut state = seed;

    for _ in 0..candle_count {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let rand = ((state >> 33) as f64) / (u32::MAX as f64) - 0.5;

        let drift = 0.0001;
        let volatility = 0.02;
        let change = drift + volatility * rand;

        price *= 1.0 + change;
        price = price.max(1.0);
        prices.push(price);
    }

    let df = df! {
        "close" => prices,
        "timestamp" => (0..candle_count).map(|i| i as i64).collect::<Vec<_>>()
    }?;

    Ok(df)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_data_generation() {
        let df =
            generate_synthetic_data("BTC-USD", 100).expect("failed to generate synthetic data");
        assert_eq!(df.height(), 100);
        assert!(df.column("close").is_ok());
        assert!(df.column("timestamp").is_ok());
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

        let config_invalid = BacktestCliConfig {
            duration: "invalid".to_string(),
            ..config.clone()
        };
        assert!(config_invalid.duration_to_candles().is_err());
    }
}
