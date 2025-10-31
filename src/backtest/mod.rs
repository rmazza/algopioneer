//! Backtesting engine for trading strategies.

use polars::prelude::*;
use crate::strategy::{Strategy, Signal};

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
pub fn run(
    strategy: &impl Strategy,
    data: &DataFrame,
    initial_capital: f64,
) -> Result<BacktestResult, Box<dyn std::error::Error>> {
    let signals = strategy.generate_signals(data)?;
    let close_prices = data.column("close")?.f64()?;

    let mut capital = initial_capital;
    let mut position = 0.0; // Represents the amount of asset held
    let mut total_trades = 0;
    let mut winning_trades = 0;
    let mut losing_trades = 0;
    let mut last_buy_price = 0.0;

    for (i, signal) in signals.iter().enumerate() {
        let close_price = close_prices.get(i).unwrap_or_default();

        match signal {
            Signal::Buy if capital > 0.0 => {
                position = capital / close_price;
                last_buy_price = close_price;
                capital = 0.0;
                total_trades += 1;
                println!("Buying at {:.2}", close_price);
            }
            Signal::Sell if position > 0.0 => {
                capital = position * close_price;
                position = 0.0;
                if close_price > last_buy_price {
                    winning_trades += 1;
                } else {
                    losing_trades += 1;
                }
                println!("Selling at {:.2}", close_price);
            }
            _ => {}
        }
    }

    // If still holding a position at the end, liquidate it at the last close price
    if position > 0.0 {
        capital = position * close_prices.get(data.height() - 1).unwrap_or_default();
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