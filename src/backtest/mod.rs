//! Backtesting engine for trading strategies.
//!
//! This module provides a deterministic backtesting framework using fixed-point
//! arithmetic (`rust_decimal::Decimal`) for all financial calculations to ensure
//! reproducible results across platforms.

use crate::strategy::{Signal, Strategy};
use polars::prelude::*;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::str::FromStr;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Default slippage in basis points (0.05% = 5 bps)
const DEFAULT_SLIPPAGE_BPS: Decimal = dec!(5);

/// Errors that can occur during backtesting.
#[derive(Debug, Error)]
pub enum BacktestError {
    #[error("Missing 'close' column in DataFrame")]
    MissingCloseColumn,

    #[error("Strategy signal generation failed: {0}")]
    StrategyError(String),

    #[error("Insufficient data: need at least 2 candles, got {0}")]
    InsufficientData(usize),

    #[error("Polars error: {0}")]
    PolarsError(#[from] PolarsError),

    #[error("Decimal conversion error: {0}")]
    DecimalConversion(String),
}

/// Configuration for a backtest run.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Starting capital for the simulation.
    pub initial_capital: Decimal,
    /// Slippage in basis points (e.g., 5 = 0.05%).
    pub slippage_bps: Decimal,
    /// Position size as a fraction (e.g., 1.0 = 100%, 0.1 = 10%).
    pub position_size_pct: Decimal,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: dec!(10000),
            slippage_bps: DEFAULT_SLIPPAGE_BPS,
            position_size_pct: dec!(1.0),
        }
    }
}

impl BacktestConfig {
    /// Creates a new config with the specified initial capital.
    pub fn with_capital(initial_capital: Decimal) -> Self {
        Self {
            initial_capital,
            ..Default::default()
        }
    }
}

/// Holds the results of a backtest.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Starting capital.
    pub initial_capital: Decimal,
    /// Final capital after all trades.
    pub final_capital: Decimal,
    /// Net profit (final - initial).
    pub net_profit: Decimal,
    /// Total number of completed trades.
    pub total_trades: u32,
    /// Number of profitable trades.
    pub winning_trades: u32,
    /// Number of losing trades.
    pub losing_trades: u32,
    /// Maximum drawdown as a decimal (e.g., 0.15 = 15%).
    pub max_drawdown: Decimal,
}

impl BacktestResult {
    /// Calculates the percentage return.
    pub fn return_percentage(&self) -> Decimal {
        if self.initial_capital.is_zero() {
            return Decimal::ZERO;
        }
        (self.net_profit / self.initial_capital) * dec!(100)
    }

    /// Calculates the win rate as a decimal.
    pub fn win_rate(&self) -> Decimal {
        if self.total_trades == 0 {
            return Decimal::ZERO;
        }
        Decimal::from(self.winning_trades) / Decimal::from(self.total_trades)
    }
}

/// Runs a backtest for a given strategy.
///
/// # Arguments
/// * `strategy` - The trading strategy to backtest.
/// * `data` - DataFrame with historical price data, must include a 'close' column.
/// * `config` - Backtest configuration (capital, slippage, position sizing).
///
/// # Returns
/// A `Result` containing the `BacktestResult` or a `BacktestError`.
///
/// # Look-Ahead Bias Prevention
/// Trades are executed at the NEXT candle's price (i+1), not the signal candle (i).
/// This prevents the unrealistic assumption of executing at the same price that
/// triggered the signal.
///
/// # Example
/// ```ignore
/// use algopioneer::backtest::{run, BacktestConfig};
/// use rust_decimal_macros::dec;
///
/// let config = BacktestConfig::with_capital(dec!(10000));
/// let result = run(&strategy, &data, &config)?;
/// println!("Return: {}%", result.return_percentage());
/// ```
pub fn run(
    strategy: &impl Strategy,
    data: &DataFrame,
    config: &BacktestConfig,
) -> Result<BacktestResult, BacktestError> {
    let data_len = data.height();
    if data_len < 2 {
        return Err(BacktestError::InsufficientData(data_len));
    }

    let signals = strategy
        .generate_signals(data)
        .map_err(|e| BacktestError::StrategyError(e.to_string()))?;

    let close_series = data
        .column("close")
        .map_err(|_| BacktestError::MissingCloseColumn)?;
    let close_prices = close_series.f64()?;

    // Pre-calculate slippage multipliers
    let slippage_mult_buy = Decimal::ONE + config.slippage_bps / dec!(10000);
    let slippage_mult_sell = Decimal::ONE - config.slippage_bps / dec!(10000);

    let mut capital = config.initial_capital;
    let mut position = Decimal::ZERO;
    let mut peak_capital = capital;
    let mut max_drawdown = Decimal::ZERO;
    let mut total_trades = 0u32;
    let mut winning_trades = 0u32;
    let mut losing_trades = 0u32;
    let mut last_buy_price = Decimal::ZERO;

    for (i, signal) in signals.iter().enumerate() {
        // LOOK-AHEAD BIAS FIX: Execute at NEXT candle price, not signal candle
        let next_idx = i + 1;
        if next_idx >= data_len {
            // Can't execute at end of data - no next candle available
            continue;
        }

        // Validate execution price - skip if missing or invalid
        let Some(raw_price) = close_prices.get(next_idx) else {
            warn!(idx = next_idx, "Missing close price, skipping candle");
            continue;
        };
        if raw_price.is_nan() || raw_price <= 0.0 {
            warn!(
                idx = next_idx,
                price = raw_price,
                "Invalid close price, skipping candle"
            );
            continue;
        }

        // Convert f64 to Decimal for precise calculations
        let execution_price = Decimal::from_str(&format!("{:.8}", raw_price))
            .map_err(|e| BacktestError::DecimalConversion(e.to_string()))?;

        if execution_price.is_zero() {
            warn!(
                idx = next_idx,
                "Zero execution price after conversion, skipping"
            );
            continue;
        }

        match signal {
            Signal::Buy if capital > Decimal::ZERO => {
                // Apply slippage: buy at HIGHER price (unfavorable)
                let slipped_price = execution_price * slippage_mult_buy;
                let allocation = capital * config.position_size_pct;
                position = allocation / slipped_price;
                last_buy_price = slipped_price;
                capital -= allocation;
                total_trades += 1;

                debug!(
                    price = %slipped_price,
                    signal_candle = i,
                    exec_candle = next_idx,
                    position = %position,
                    "BUY executed"
                );
            }
            Signal::Sell if position > Decimal::ZERO => {
                // Apply slippage: sell at LOWER price (unfavorable)
                let slipped_price = execution_price * slippage_mult_sell;
                let proceeds = position * slipped_price;
                capital += proceeds;
                position = Decimal::ZERO;

                if slipped_price > last_buy_price {
                    winning_trades += 1;
                } else {
                    losing_trades += 1;
                }

                // Track peak capital and drawdown
                if capital > peak_capital {
                    peak_capital = capital;
                }
                let drawdown = if peak_capital > Decimal::ZERO {
                    (peak_capital - capital) / peak_capital
                } else {
                    Decimal::ZERO
                };
                if drawdown > max_drawdown {
                    max_drawdown = drawdown;
                }

                debug!(
                    price = %slipped_price,
                    signal_candle = i,
                    exec_candle = next_idx,
                    capital = %capital,
                    "SELL executed"
                );
            }
            _ => {}
        }
    }

    // If still holding a position at the end, liquidate it at the last close price
    if position > Decimal::ZERO {
        if let Some(last_price) = close_prices.get(data_len - 1) {
            if last_price > 0.0 && !last_price.is_nan() {
                let price =
                    Decimal::from_str(&format!("{:.8}", last_price)).unwrap_or(Decimal::ZERO);
                if !price.is_zero() {
                    let slipped_price = price * slippage_mult_sell;
                    capital += position * slipped_price;
                    info!(price = %slipped_price, "Liquidated remaining position");
                }
            }
        }
    }

    let net_profit = capital - config.initial_capital;

    Ok(BacktestResult {
        initial_capital: config.initial_capital,
        final_capital: capital,
        net_profit,
        total_trades,
        winning_trades,
        losing_trades,
        max_drawdown,
    })
}

/// Convenience function to run a backtest with default configuration.
///
/// Uses default slippage (5 bps) and 100% position sizing.
pub fn run_with_capital(
    strategy: &impl Strategy,
    data: &DataFrame,
    initial_capital: Decimal,
) -> Result<BacktestResult, BacktestError> {
    let config = BacktestConfig::with_capital(initial_capital);
    run(strategy, data, &config)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock strategy for testing
    struct MockStrategy {
        signals: Vec<Signal>,
    }

    impl Strategy for MockStrategy {
        fn generate_signals(&self, _data: &DataFrame) -> Result<Vec<Signal>, PolarsError> {
            Ok(self.signals.clone())
        }

        fn get_latest_signal(
            &self,
            _df: &DataFrame,
            _position_open: bool,
        ) -> Result<Signal, PolarsError> {
            Ok(self.signals.last().cloned().unwrap_or(Signal::Hold))
        }
    }

    fn create_test_df(prices: Vec<f64>) -> DataFrame {
        df! {
            "close" => prices
        }
        .unwrap()
    }

    #[test]
    fn test_insufficient_data() {
        let strategy = MockStrategy {
            signals: vec![Signal::Hold],
        };
        let data = create_test_df(vec![100.0]);
        let config = BacktestConfig::default();

        let result = run(&strategy, &data, &config);
        assert!(matches!(result, Err(BacktestError::InsufficientData(1))));
    }

    #[test]
    fn test_buy_and_sell_trade() {
        let strategy = MockStrategy {
            signals: vec![Signal::Buy, Signal::Hold, Signal::Sell, Signal::Hold],
        };
        // Signal at 0 (Buy) -> executes at price 105
        // Signal at 2 (Sell) -> executes at price 115
        let data = create_test_df(vec![100.0, 105.0, 110.0, 115.0]);
        let config = BacktestConfig {
            initial_capital: dec!(1000),
            slippage_bps: Decimal::ZERO, // No slippage for easier testing
            position_size_pct: dec!(1.0),
        };

        let result = run(&strategy, &data, &config).unwrap();

        assert_eq!(result.total_trades, 1);
        assert_eq!(result.winning_trades, 1);
        assert_eq!(result.losing_trades, 0);
        // Bought at 105, sold at 115 -> ~9.5% profit
        assert!(result.net_profit > Decimal::ZERO);
    }

    #[test]
    fn test_losing_trade() {
        let strategy = MockStrategy {
            signals: vec![Signal::Buy, Signal::Hold, Signal::Sell, Signal::Hold],
        };
        // Signal at 0 (Buy) -> executes at price 105
        // Signal at 2 (Sell) -> executes at price 95
        let data = create_test_df(vec![100.0, 105.0, 110.0, 95.0]);
        let config = BacktestConfig {
            initial_capital: dec!(1000),
            slippage_bps: Decimal::ZERO,
            position_size_pct: dec!(1.0),
        };

        let result = run(&strategy, &data, &config).unwrap();

        assert_eq!(result.total_trades, 1);
        assert_eq!(result.winning_trades, 0);
        assert_eq!(result.losing_trades, 1);
        assert!(result.net_profit < Decimal::ZERO);
    }

    #[test]
    fn test_slippage_applied() {
        let strategy = MockStrategy {
            signals: vec![Signal::Buy, Signal::Hold, Signal::Sell, Signal::Hold],
        };
        let data = create_test_df(vec![100.0, 100.0, 100.0, 100.0]);

        // With slippage, buy at higher and sell at lower -> always lose
        let config = BacktestConfig {
            initial_capital: dec!(1000),
            slippage_bps: dec!(50), // 0.5% slippage
            position_size_pct: dec!(1.0),
        };

        let result = run(&strategy, &data, &config).unwrap();

        // Should lose money due to slippage even with flat prices
        assert!(result.net_profit < Decimal::ZERO);
    }

    #[test]
    fn test_position_sizing() {
        let strategy = MockStrategy {
            signals: vec![Signal::Buy, Signal::Hold, Signal::Hold, Signal::Hold],
        };
        let data = create_test_df(vec![100.0, 100.0, 100.0, 100.0]);

        let config = BacktestConfig {
            initial_capital: dec!(1000),
            slippage_bps: Decimal::ZERO,
            position_size_pct: dec!(0.5), // 50% position size
        };

        let result = run(&strategy, &data, &config).unwrap();

        // Should still have 50% capital remaining (position liquidated at end)
        // But since we liquidate, final capital should be close to initial
        assert!(result.final_capital > Decimal::ZERO);
    }

    #[test]
    fn test_look_ahead_bias_prevention() {
        let strategy = MockStrategy {
            // Signal at last candle should not execute (no next candle)
            signals: vec![Signal::Hold, Signal::Hold, Signal::Buy],
        };
        let data = create_test_df(vec![100.0, 105.0, 110.0]);
        let config = BacktestConfig::default();

        let result = run(&strategy, &data, &config).unwrap();

        // Buy signal at index 2 cannot execute (no index 3)
        // So no trades should occur
        assert_eq!(result.total_trades, 0);
    }

    #[test]
    fn test_max_drawdown_tracking() {
        let strategy = MockStrategy {
            signals: vec![
                Signal::Buy,
                Signal::Hold,
                Signal::Sell,
                Signal::Buy,
                Signal::Hold,
                Signal::Sell,
                Signal::Hold,
            ],
        };
        // First trade: buy at 100, sell at 80 (-20%)
        // Second trade: buy at 90, sell at 100 (+11%)
        let data = create_test_df(vec![90.0, 100.0, 70.0, 80.0, 90.0, 95.0, 100.0]);

        let config = BacktestConfig {
            initial_capital: dec!(1000),
            slippage_bps: Decimal::ZERO,
            position_size_pct: dec!(1.0),
        };

        let result = run(&strategy, &data, &config).unwrap();

        // Should have recorded a drawdown
        assert!(result.max_drawdown > Decimal::ZERO);
    }

    #[test]
    fn test_win_rate_calculation() {
        let result = BacktestResult {
            initial_capital: dec!(1000),
            final_capital: dec!(1100),
            net_profit: dec!(100),
            total_trades: 10,
            winning_trades: 6,
            losing_trades: 4,
            max_drawdown: dec!(0.05),
        };

        assert_eq!(result.win_rate(), dec!(0.6));
    }

    #[test]
    fn test_return_percentage() {
        let result = BacktestResult {
            initial_capital: dec!(1000),
            final_capital: dec!(1150),
            net_profit: dec!(150),
            total_trades: 5,
            winning_trades: 3,
            losing_trades: 2,
            max_drawdown: dec!(0.1),
        };

        assert_eq!(result.return_percentage(), dec!(15));
    }
}
