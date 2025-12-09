//! Backtesting engine for trading strategies.

use polars::prelude::*;
use crate::strategy::{Strategy, Signal};
use tracing::{info, debug};

/// Slippage in basis points (0.05% = 5 bps)
const SLIPPAGE_BPS: f64 = 5.0;

/// Holds the results of a backtest.
#[derive(Debug)]
pub struct BacktestResult {
    pub initial_capital: f64,
    pub final_capital: f64,
    pub net_profit: f64,
    pub total_trades: u32,
    pub winning_trades: u32,
    pub losing_trades: u32,
}

impl BacktestResult {
    /// Calculates the percentage return.
    pub fn return_percentage(&self) -> f64 {
        (self.net_profit / self.initial_capital) * 100.0
    }
}

/// Runs a backtest for a given strategy.
///
/// # Arguments
/// * `strategy` - The trading strategy to backtest.
/// * `data` - DataFrame with historical price data, must include a 'close' column.
/// * `initial_capital` - The starting capital for the simulation.
///
/// # Returns
/// A `Result` containing the `BacktestResult` or an error.
///
/// # Look-Ahead Bias Fix
/// Trades are executed at the NEXT candle's price (i+1), not the signal candle (i).
/// This prevents the unrealistic assumption of executing at the same price that triggered the signal.
pub fn run(
    strategy: &impl Strategy,
    data: &DataFrame,
    initial_capital: f64,
) -> Result<BacktestResult, Box<dyn std::error::Error>> {
    let signals = strategy.generate_signals(data)?;
    let close_prices = data.column("close")?.f64()?;
    let data_len = data.height();

    let mut capital = initial_capital;
    let mut position = 0.0; // Represents the amount of asset held
    let mut total_trades = 0;
    let mut winning_trades = 0;
    let mut losing_trades = 0;
    let mut last_buy_price = 0.0;

    for (i, signal) in signals.iter().enumerate() {
        // LOOK-AHEAD BIAS FIX: Execute at NEXT candle price, not signal candle
        let next_idx = i + 1;
        if next_idx >= data_len {
            // Can't execute at end of data - no next candle available
            continue;
        }
        let execution_price = close_prices.get(next_idx).unwrap_or_default();

        match signal {
            Signal::Buy if capital > 0.0 => {
                // Apply slippage: buy at HIGHER price (unfavorable)
                let slipped_price = execution_price * (1.0 + SLIPPAGE_BPS / 10000.0);
                position = capital / slipped_price;
                last_buy_price = slipped_price;
                capital = 0.0;
                total_trades += 1;
                debug!("BUY at {:.2} (signal at candle {}, executed at {})", slipped_price, i, next_idx);
            }
            Signal::Sell if position > 0.0 => {
                // Apply slippage: sell at LOWER price (unfavorable)
                let slipped_price = execution_price * (1.0 - SLIPPAGE_BPS / 10000.0);
                capital = position * slipped_price;
                position = 0.0;
                if slipped_price > last_buy_price {
                    winning_trades += 1;
                } else {
                    losing_trades += 1;
                }
                debug!("SELL at {:.2} (signal at candle {}, executed at {})", slipped_price, i, next_idx);
            }
            _ => {}
        }
    }

    // If still holding a position at the end, liquidate it at the last close price
    if position > 0.0 {
        let last_price = close_prices.get(data_len - 1).unwrap_or_default();
        let slipped_price = last_price * (1.0 - SLIPPAGE_BPS / 10000.0);
        capital = position * slipped_price;
        info!("Liquidating remaining position at {:.2}", slipped_price);
    }

    let final_capital = capital;
    let net_profit = final_capital - initial_capital;

    Ok(BacktestResult {
        initial_capital,
        final_capital,
        net_profit,
        total_trades,
        winning_trades,
        losing_trades,
    })
}