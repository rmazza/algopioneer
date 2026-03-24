//! Backtesting engine for trading strategies.
//!
//! This module provides a deterministic backtesting framework using fixed-point
//! arithmetic (`rust_decimal::Decimal`) for all financial calculations to ensure
//! reproducible results across platforms.

use crate::application::strategy::{Signal, Strategy};
use polars::prelude::*;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::str::FromStr;
use thiserror::Error;
use tracing::debug;

/// Holds synchronized data for two legs.
#[derive(Debug, Clone)]
pub struct DualLegData {
    pub timestamps: Vec<i64>,
    pub close_a: Vec<Decimal>,
    pub close_b: Vec<Decimal>,
}

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
    /// Annualized risk-free rate (e.g., 0.02 = 2%) for Sharpe/Sortino.
    pub risk_free_rate: Decimal,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: dec!(10000),
            slippage_bps: DEFAULT_SLIPPAGE_BPS,
            position_size_pct: dec!(1.0),
            risk_free_rate: dec!(0.02),
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

/// A single completed trade record.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TradeRecord {
    pub entry_idx: usize,
    pub exit_idx: usize,
    pub entry_price: Decimal,
    pub exit_price: Decimal,
    pub size: Decimal,
    pub pnl: Decimal,
    pub pnl_pct: Decimal,
}

/// Holds the results of a backtest.
#[derive(Debug, Clone, serde::Serialize)]
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
    /// Annualized Sharpe Ratio.
    pub sharpe_ratio: Decimal,
    /// Annualized Sortino Ratio.
    pub sortino_ratio: Decimal,
    /// Gross Profit / Gross Loss.
    pub profit_factor: Decimal,
    /// Expected value per trade.
    pub expectancy: Decimal,
    /// Complete trade history.
    pub trades: Vec<TradeRecord>,
    /// Daily (or per-candle) equity curve.
    #[serde(skip)]
    pub equity_curve: Vec<Decimal>,
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

    let slippage_mult_buy = Decimal::ONE + config.slippage_bps / dec!(10000);
    let slippage_mult_sell = Decimal::ONE - config.slippage_bps / dec!(10000);

    let mut capital = config.initial_capital;
    let mut position = Decimal::ZERO;
    let mut peak_capital = capital;
    let mut max_drawdown = Decimal::ZERO;
    let mut winning_trades = 0u32;
    let mut losing_trades = 0u32;
    let mut last_buy_price = Decimal::ZERO;
    let mut last_buy_idx = 0;

    let mut trades = Vec::new();
    let mut equity_curve = Vec::with_capacity(data_len);
    let mut returns = Vec::with_capacity(data_len);
    let mut last_equity = capital;

    for (i, signal) in signals.iter().enumerate() {
        let next_idx = i + 1;
        if next_idx >= data_len {
            equity_curve.push(last_equity);
            continue;
        }

        let Some(raw_price) = close_prices.get(next_idx) else {
            equity_curve.push(last_equity);
            continue;
        };

        let execution_price = Decimal::from_str(&format!("{:.8}", raw_price))
            .map_err(|e| BacktestError::DecimalConversion(e.to_string()))?;

        if execution_price.is_zero() {
            equity_curve.push(last_equity);
            continue;
        }

        match signal {
            Signal::Buy if capital > Decimal::ZERO && position.is_zero() => {
                let slipped_price = execution_price * slippage_mult_buy;
                let allocation = capital * config.position_size_pct;
                position = allocation / slipped_price;
                last_buy_price = slipped_price;
                last_buy_idx = next_idx;
                capital -= allocation;

                debug!(price = %slipped_price, "BUY executed");
            }
            Signal::Sell if position > Decimal::ZERO => {
                let slipped_price = execution_price * slippage_mult_sell;
                let proceeds = position * slipped_price;
                let pnl = proceeds - (position * last_buy_price);
                let pnl_pct = if !last_buy_price.is_zero() {
                    (slipped_price - last_buy_price) / last_buy_price * dec!(100)
                } else {
                    Decimal::ZERO
                };

                trades.push(TradeRecord {
                    entry_idx: last_buy_idx,
                    exit_idx: next_idx,
                    entry_price: last_buy_price,
                    exit_price: slipped_price,
                    size: position,
                    pnl,
                    pnl_pct,
                });

                capital += proceeds;
                position = Decimal::ZERO;

                if slipped_price > last_buy_price {
                    winning_trades += 1;
                } else {
                    losing_trades += 1;
                }

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

                debug!(price = %slipped_price, capital = %capital, "SELL executed");
            }
            _ => {}
        }

        let current_equity = capital + (position * execution_price);
        equity_curve.push(current_equity);

        if last_equity > Decimal::ZERO {
            let ret = (current_equity - last_equity) / last_equity;
            returns.push(ret);
        }
        last_equity = current_equity;
    }

    if position > Decimal::ZERO {
        if let Some(last_price) = close_prices.get(data_len - 1) {
            let price = Decimal::from_str(&format!("{:.8}", last_price)).unwrap_or(Decimal::ZERO);
            if !price.is_zero() {
                let slipped_price = price * slippage_mult_sell;
                let proceeds = position * slipped_price;
                let pnl = proceeds - (position * last_buy_price);
                let pnl_pct = if !last_buy_price.is_zero() {
                    (slipped_price - last_buy_price) / last_buy_price * dec!(100)
                } else {
                    Decimal::ZERO
                };

                trades.push(TradeRecord {
                    entry_idx: last_buy_idx,
                    exit_idx: data_len - 1,
                    entry_price: last_buy_price,
                    exit_price: slipped_price,
                    size: position,
                    pnl,
                    pnl_pct,
                });

                capital += proceeds;
                if slipped_price > last_buy_price {
                    winning_trades += 1;
                } else {
                    losing_trades += 1;
                }
            }
        }
    }

    let total_trades = trades.len() as u32;
    let (sharpe, sortino) = calculate_risk_adjusted_ratios(&returns, config.risk_free_rate);
    let profit_factor = calculate_profit_factor(&trades);
    let expectancy = calculate_expectancy(win_rate(winning_trades, total_trades), &trades);

    Ok(BacktestResult {
        initial_capital: config.initial_capital,
        final_capital: capital,
        net_profit: capital - config.initial_capital,
        total_trades,
        winning_trades,
        losing_trades,
        max_drawdown,
        sharpe_ratio: sharpe,
        sortino_ratio: sortino,
        profit_factor,
        expectancy,
        trades,
        equity_curve,
    })
}

fn win_rate(wins: u32, total: u32) -> Decimal {
    if total == 0 {
        Decimal::ZERO
    } else {
        Decimal::from(wins) / Decimal::from(total)
    }
}

fn calculate_risk_adjusted_ratios(returns: &[Decimal], rf: Decimal) -> (Decimal, Decimal) {
    if returns.len() < 2 {
        return (Decimal::ZERO, Decimal::ZERO);
    }

    let n = returns.len() as f64;
    let returns_f64: Vec<f64> = returns.iter().filter_map(|r| r.to_f64()).collect();
    let mean = returns_f64.iter().sum::<f64>() / n;

    let variance = returns_f64.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();

    let rf_daily = rf.to_f64().unwrap_or(0.0) / 252.0;

    let sharpe = if std_dev > f64::EPSILON {
        let annual_return = mean * 252.0;
        let annual_vol = std_dev * 252.0f64.sqrt();
        (annual_return - rf_daily * 252.0) / annual_vol
    } else {
        0.0
    };

    let downside_returns: Vec<f64> = returns_f64.iter().filter(|&&r| r < 0.0).cloned().collect();
    let sortino = if !downside_returns.is_empty() {
        let downside_variance = downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / n;
        let downside_std = downside_variance.sqrt();
        if downside_std > f64::EPSILON {
            (mean * 252.0 - rf_daily * 252.0) / (downside_std * 252.0f64.sqrt())
        } else {
            0.0
        }
    } else {
        0.0
    };

    (
        Decimal::from_f64_retain(sharpe)
            .unwrap_or(Decimal::ZERO)
            .round_dp(4),
        Decimal::from_f64_retain(sortino)
            .unwrap_or(Decimal::ZERO)
            .round_dp(4),
    )
}

fn calculate_profit_factor(trades: &[TradeRecord]) -> Decimal {
    let mut gross_profit = Decimal::ZERO;
    let mut gross_loss = Decimal::ZERO;

    for trade in trades {
        if trade.pnl > Decimal::ZERO {
            gross_profit += trade.pnl;
        } else {
            gross_loss += trade.pnl.abs();
        }
    }

    if gross_loss.is_zero() {
        if gross_profit.is_zero() {
            Decimal::ZERO
        } else {
            dec!(999)
        }
    } else {
        gross_profit / gross_loss
    }
}

fn calculate_expectancy(win_rate: Decimal, trades: &[TradeRecord]) -> Decimal {
    if trades.is_empty() {
        return Decimal::ZERO;
    }

    let mut avg_win = Decimal::ZERO;
    let mut win_count = 0;
    let mut avg_loss = Decimal::ZERO;
    let mut loss_count = 0;

    for trade in trades {
        if trade.pnl > Decimal::ZERO {
            avg_win += trade.pnl;
            win_count += 1;
        } else {
            avg_loss += trade.pnl.abs();
            loss_count += 1;
        }
    }

    if win_count > 0 {
        avg_win /= Decimal::from(win_count);
    }
    if loss_count > 0 {
        avg_loss /= Decimal::from(loss_count);
    }

    (win_rate * avg_win) - ((Decimal::ONE - win_rate) * avg_loss)
}

use crate::application::strategy::dual_leg::EntryStrategy;
use crate::domain::types::MarketData;

/// Runs a backtest for a dual-leg strategy (e.g., Pairs or Basis trading).
pub async fn run_dual(
    strategy: &mut impl EntryStrategy,
    leg_a: &DataFrame,
    leg_b: &DataFrame,
    config: &BacktestConfig,
) -> Result<BacktestResult, BacktestError> {
    // 1. Synchronize Data
    let data = synchronize_dual_data(leg_a, leg_b)?;
    let data_len = data.timestamps.len();
    if data_len < 2 {
        return Err(BacktestError::InsufficientData(data_len));
    }

    // 2. Execution Loop
    let mut capital = config.initial_capital;
    let mut position_size = Decimal::ZERO; // Shares/Units of the spread
    let mut last_entry_price_a = Decimal::ZERO;
    let mut last_entry_price_b = Decimal::ZERO;
    let mut current_direction: i8 = 0; // 1 = Long A/Short B, -1 = Short A/Long B, 0 = Flat

    let slippage_mult_buy = Decimal::ONE + config.slippage_bps / dec!(10000);
    let slippage_mult_sell = Decimal::ONE - config.slippage_bps / dec!(10000);

    let mut peak_capital = capital;
    let mut max_drawdown = Decimal::ZERO;
    let mut winning_trades = 0u32;
    let mut losing_trades = 0u32;
    let mut trades = Vec::new();
    let mut equity_curve = Vec::with_capacity(data_len);
    let mut returns = Vec::with_capacity(data_len);
    let mut last_equity = capital;

    // OW2: Pre-allocate MarketData shells
    let mut m1 = MarketData {
        symbol: "A".to_string(),
        price: Decimal::ZERO,
        instrument_id: None,
        timestamp: 0,
    };
    let mut m2 = MarketData {
        symbol: "B".to_string(),
        price: Decimal::ZERO,
        instrument_id: None,
        timestamp: 0,
    };

    for i in 0..data_len {
        let ts = data.timestamps[i];
        let p_a = data.close_a[i];
        let p_b = data.close_b[i];

        // Update shells for strategy
        m1.price = p_a;
        m1.timestamp = ts;
        m2.price = p_b;
        m2.timestamp = ts;

        // Generate signal from entry strategy
        let signal = strategy.analyze(&m1, &m2).await;

        let next_idx = i + 1;
        if next_idx >= data_len {
            equity_curve.push(last_equity);
            continue;
        }

        // Execution happens at NEXT candle
        let exec_p_a = data.close_a[next_idx];
        let exec_p_b = data.close_b[next_idx];

        match signal {
            Signal::Buy if current_direction == 0 && capital > Decimal::ZERO => {
                let slipped_a = exec_p_a * slippage_mult_buy;
                let slipped_b = exec_p_b * slippage_mult_sell;

                // Dollar-neutral allocation: Spend half capital on each leg
                let total_allocation = capital * config.position_size_pct;
                let leg_allocation = total_allocation / dec!(2);
                
                let size_a = leg_allocation / slipped_a;
                let size_b = leg_allocation / slipped_b;
                
                // We track position as a "unit" of the spread. 
                // To keep it simple and consistent with previous logic, 
                // we'll store the sizes separately or use an average size.
                // Re-implementing to track actual shares per leg for accuracy.
                last_entry_price_a = slipped_a;
                last_entry_price_b = slipped_b;
                
                // Use a custom struct or just use position_size as a multiplier for the ratio
                // Let's use position_size as the 'base' units and track leg sizes
                position_size = leg_allocation; // Using allocated dollars as the 'size'
                current_direction = 1;

                capital -= total_allocation;
                debug!(price_a = %slipped_a, price_b = %slipped_b, "LONG SPREAD executed");
            }

            Signal::Sell if current_direction == 0 && capital > Decimal::ZERO => {
                let slipped_a = exec_p_a * slippage_mult_sell;
                let slipped_b = exec_p_b * slippage_mult_buy;

                let total_allocation = capital * config.position_size_pct;
                let leg_allocation = total_allocation / dec!(2);
                
                last_entry_price_a = slipped_a;
                last_entry_price_b = slipped_b;
                
                position_size = leg_allocation; 
                current_direction = -1;

                capital -= total_allocation;
                debug!(price_a = %slipped_a, price_b = %slipped_b, "SHORT SPREAD executed");
            }

            Signal::Exit if current_direction != 0 => {
                let (exit_a, exit_b) = if current_direction == 1 {
                    (exec_p_a * slippage_mult_sell, exec_p_b * slippage_mult_buy)
                } else {
                    (exec_p_a * slippage_mult_buy, exec_p_b * slippage_mult_sell)
                };

                // PnL calculated based on dollar-neutral legs
                // size_a = position_size / last_entry_price_a
                // pnl_a = (exit - entry) * size_a
                let pnl_a = if current_direction == 1 {
                    (exit_a - last_entry_price_a) * (position_size / last_entry_price_a)
                } else {
                    (last_entry_price_a - exit_a) * (position_size / last_entry_price_a)
                };
                
                let pnl_b = if current_direction == 1 {
                    (last_entry_price_b - exit_b) * (position_size / last_entry_price_b)
                } else {
                    (exit_b - last_entry_price_b) * (position_size / last_entry_price_b)
                };

                let total_pnl = pnl_a + pnl_b;
                let original_cost = position_size * dec!(2);
                let pnl_pct = (total_pnl / original_cost) * dec!(100);

                trades.push(TradeRecord {
                    entry_idx: i,
                    exit_idx: next_idx,
                    entry_price: last_entry_price_a,
                    exit_price: exit_a,
                    size: position_size, // Dollar size per leg
                    pnl: total_pnl,
                    pnl_pct,
                });

                capital += original_cost + total_pnl;
                if total_pnl > Decimal::ZERO {
                    winning_trades += 1;
                } else {
                    losing_trades += 1;
                }

                position_size = Decimal::ZERO;
                current_direction = 0;

                if capital > peak_capital {
                    peak_capital = capital;
                }
                let dd = if peak_capital > Decimal::ZERO {
                    (peak_capital - capital) / peak_capital
                } else {
                    Decimal::ZERO
                };
                if dd > max_drawdown {
                    max_drawdown = dd;
                }
            }
            _ => {}
        }

        let current_equity = if current_direction == 0 {
            capital
        } else {
            let unrealized_a = if current_direction == 1 {
                (exec_p_a - last_entry_price_a) * position_size
            } else {
                (last_entry_price_a - exec_p_a) * position_size
            };
            let unrealized_b = if current_direction == 1 {
                (last_entry_price_b - exec_p_b) * position_size
            } else {
                (exec_p_b - last_entry_price_b) * position_size
            };
            let cost = (last_entry_price_a + last_entry_price_b) * position_size;
            capital + cost + unrealized_a + unrealized_b
        };

        equity_curve.push(current_equity);
        if last_equity > Decimal::ZERO {
            returns.push((current_equity - last_equity) / last_equity);
        }
        last_equity = current_equity;
    }

    // Final liquidation if position is still open
    if current_direction != 0 {
        if let (Some(&exec_p_a), Some(&exec_p_b)) = (data.close_a.last(), data.close_b.last()) {
            let (exit_a, exit_b) = if current_direction == 1 {
                (exec_p_a * slippage_mult_sell, exec_p_b * slippage_mult_buy)
            } else {
                (exec_p_a * slippage_mult_buy, exec_p_b * slippage_mult_sell)
            };

            let pnl_a = if current_direction == 1 {
                (exit_a - last_entry_price_a) * (position_size / last_entry_price_a)
            } else {
                (last_entry_price_a - exit_a) * (position_size / last_entry_price_a)
            };
            let pnl_b = if current_direction == 1 {
                (last_entry_price_b - exit_b) * (position_size / last_entry_price_b)
            } else {
                (exit_b - last_entry_price_b) * (position_size / last_entry_price_b)
            };

            let total_pnl = pnl_a + pnl_b;
            let original_cost = position_size * dec!(2);
            let pnl_pct = (total_pnl / original_cost) * dec!(100);

            trades.push(TradeRecord {
                entry_idx: data_len - 1, 
                exit_idx: data_len - 1,
                entry_price: last_entry_price_a,
                exit_price: exit_a,
                size: position_size,
                pnl: total_pnl,
                pnl_pct,
            });

            capital += original_cost + total_pnl;
            if total_pnl > Decimal::ZERO {
                winning_trades += 1;
            } else {
                losing_trades += 1;
            }
            current_direction = 0;
        }
    }

    let total_trades = trades.len() as u32;
    let (sharpe, sortino) = calculate_risk_adjusted_ratios(&returns, config.risk_free_rate);
    let profit_factor = calculate_profit_factor(&trades);
    let expectancy = calculate_expectancy(win_rate(winning_trades, total_trades), &trades);

    Ok(BacktestResult {
        initial_capital: config.initial_capital,
        final_capital: capital,
        net_profit: capital - config.initial_capital,
        total_trades,
        winning_trades,
        losing_trades,
        max_drawdown,
        sharpe_ratio: sharpe,
        sortino_ratio: sortino,
        profit_factor,
        expectancy,
        trades,
        equity_curve,
    })
}

fn synchronize_dual_data(
    leg_a: &DataFrame,
    leg_b: &DataFrame,
) -> Result<DualLegData, BacktestError> {
    let joined = leg_a.join(
        leg_b,
        ["timestamp"],
        ["timestamp"],
        JoinArgs::new(JoinType::Inner),
        None, // missing arg in previous turn
    )?;

    let ts = joined.column("timestamp")?.i64()?;
    let close_a = joined.column("close")?.f64()?;
    let close_b = joined.column("close_right")?.f64()?;

    let height = joined.height();
    let mut out_ts = Vec::with_capacity(height);
    let mut out_a = Vec::with_capacity(height);
    let mut out_b = Vec::with_capacity(height);

    for i in 0..height {
        if let (Some(t), Some(a), Some(b)) = (ts.get(i), close_a.get(i), close_b.get(i)) {
            out_ts.push(t);
            out_a.push(Decimal::from_f64_retain(a).unwrap_or(Decimal::ZERO));
            out_b.push(Decimal::from_f64_retain(b).unwrap_or(Decimal::ZERO));
        }
    }

    Ok(DualLegData {
        timestamps: out_ts,
        close_a: out_a,
        close_b: out_b,
    })
}

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
        let data = create_test_df(vec![100.0, 105.0, 110.0, 115.0]);
        let config = BacktestConfig {
            initial_capital: dec!(1000),
            slippage_bps: Decimal::ZERO,
            position_size_pct: dec!(1.0),
            risk_free_rate: Decimal::ZERO,
        };

        let result = run(&strategy, &data, &config).unwrap();

        assert_eq!(result.total_trades, 1);
        assert_eq!(result.winning_trades, 1);
        assert_eq!(result.losing_trades, 0);
        assert!(result.net_profit > Decimal::ZERO);
        assert_eq!(result.trades.len(), 1);
    }

    #[test]
    fn test_losing_trade() {
        let strategy = MockStrategy {
            signals: vec![Signal::Buy, Signal::Hold, Signal::Sell, Signal::Hold],
        };
        let data = create_test_df(vec![100.0, 105.0, 110.0, 95.0]);
        let config = BacktestConfig {
            initial_capital: dec!(1000),
            slippage_bps: Decimal::ZERO,
            position_size_pct: dec!(1.0),
            risk_free_rate: Decimal::ZERO,
        };

        let result = run(&strategy, &data, &config).unwrap();

        assert_eq!(result.total_trades, 1);
        assert_eq!(result.winning_trades, 0);
        assert_eq!(result.losing_trades, 1);
        assert!(result.net_profit < Decimal::ZERO);
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
        let data = create_test_df(vec![90.0, 100.0, 70.0, 80.0, 90.0, 95.0, 100.0]);

        let config = BacktestConfig {
            initial_capital: dec!(1000),
            slippage_bps: Decimal::ZERO,
            position_size_pct: dec!(1.0),
            risk_free_rate: Decimal::ZERO,
        };

        let result = run(&strategy, &data, &config).unwrap();
        assert!(result.max_drawdown > Decimal::ZERO);
    }

    #[test]
    fn test_profit_factor() {
        let trades = vec![
            TradeRecord {
                entry_idx: 0,
                exit_idx: 1,
                entry_price: dec!(100),
                exit_price: dec!(110),
                size: dec!(1),
                pnl: dec!(10),
                pnl_pct: dec!(10),
            },
            TradeRecord {
                entry_idx: 2,
                exit_idx: 3,
                entry_price: dec!(100),
                exit_price: dec!(95),
                size: dec!(1),
                pnl: dec!(-5),
                pnl_pct: dec!(-5),
            },
        ];

        let pf = calculate_profit_factor(&trades);
        assert_eq!(pf, dec!(2));
    }
}
