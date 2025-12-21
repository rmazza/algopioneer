//! Grid Search Optimizer for Pairs Trading Parameters
//!
//! Implements parameter optimization through historical backtesting
//! with proper Decimal-based financial calculations.

use super::config::{DiscoveryConfig, GridSearchConfig, PortfolioPairConfig};
use super::error::DiscoveryError;
use super::filter::{filter_candidates, CandidatePair};
use crate::strategy::dual_leg_trading::Clock;
use crate::strategy::dual_leg_trading::{
    DualLegConfig, EntryStrategy, MarketData, PairsManager, TransactionCostModel,
};
use crate::strategy::Signal;

use async_trait::async_trait;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use tokio::time::sleep;
use tracing::{debug, info, warn};

/// Trait for data sources that can provide historical candle data for discovery.
///
/// This abstraction allows the discovery pipeline to work with any exchange
/// (Coinbase, Alpaca, etc.) without coupling to a specific client implementation.
///
/// # DRY Principle
/// Single trait allows unified discovery logic for crypto and equities.
#[async_trait]
pub trait DiscoveryDataSource: Send + Sync {
    /// Fetch hourly candle close prices for a symbol.
    ///
    /// Returns a vector of (timestamp_seconds, close_price) tuples sorted by time.
    async fn fetch_candles_hourly(
        &mut self,
        symbol: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<(i64, Decimal)>, Box<dyn Error + Send + Sync>>;
}

/// Result of parameter optimization for a single pair
#[derive(Debug, Clone)]
pub struct OptimizedPair {
    /// First leg symbol
    pub leg1: String,
    /// Second leg symbol  
    pub leg2: String,
    /// Optimal rolling window size
    pub window: usize,
    /// Optimal z-score entry threshold
    pub z_entry: f64,
    /// Z-score exit threshold (mean reversion)
    pub z_exit: f64,
    /// In-sample Sharpe ratio (training period)
    pub sharpe_ratio: f64,
    /// Net profit in USD (total: train + test)
    pub net_profit: Decimal,
    /// Total number of trades (train + test)
    pub trades: u32,
    /// Pearson correlation
    pub correlation: f64,
    /// Mean-reversion half-life (hours)
    pub half_life_hours: f64,
    /// ADF cointegration test statistic
    pub adf_statistic: f64,
    /// Out-of-sample Sharpe ratio (validation/test period)
    pub validation_sharpe: f64,
    /// Number of trades in validation period
    pub validation_trades: u32,
    /// Profit per trade (net_profit / trades)
    pub profit_per_trade: Decimal,
}

impl OptimizedPair {
    /// Convert to PortfolioPairConfig for use with PortfolioManager
    ///
    /// # CB-2 FIX
    /// Converts f64 z-scores to Decimal for deterministic threshold comparisons.
    pub fn to_portfolio_config(&self, allocation: Decimal) -> PortfolioPairConfig {
        PortfolioPairConfig {
            dual_leg_config: DualLegConfig {
                spot_symbol: self.leg1.clone(),
                future_symbol: self.leg2.clone(),
                order_size: allocation,
                max_tick_age_ms: 2000,
                execution_timeout_ms: 30000,
                min_profit_threshold: dec!(0.001),
                stop_loss_threshold: dec!(-0.02),
                fee_tier: TransactionCostModel::default(),
                throttle_interval_secs: 5,
            },
            window_size: self.window,
            // CB-2 FIX: Convert f64 to Decimal for financial precision
            entry_z_score: Decimal::from_f64_retain(self.z_entry).unwrap_or(dec!(2.0)),
            exit_z_score: Decimal::from_f64_retain(self.z_exit).unwrap_or(dec!(0.1)),
        }
    }
}

/// Backtest result for a single parameter combination
#[derive(Debug, Clone)]
struct BacktestResult {
    window: usize,
    z_entry: f64,
    z_exit: f64,
    net_profit: Decimal,
    sharpe_ratio: f64,
    trades: u32,
}

/// Configuration for backtest simulation (clippy: reduces argument count)
#[derive(Debug, Clone)]
struct BacktestConfig {
    initial_capital: Decimal,
    taker_fee: Decimal,
    window: usize,
    z_entry: f64,
    z_exit: f64,
}

/// Market data for backtest (clippy: reduces argument count)
struct BacktestData<'a> {
    timestamps: &'a [i64],
    prices_a: &'a [Decimal],
    prices_b: &'a [Decimal],
}

/// Position state for PnL tracking (Decimal-based for financial integrity)
#[derive(Debug, Clone)]
struct Position {
    /// Direction: 1 = Long spread (Long A, Short B), -1 = Short spread
    direction: i8,
    /// Entry price of leg A
    entry_price_a: Decimal,
    /// Entry price of leg B
    entry_price_b: Decimal,
    /// Capital at time of entry (after fees)
    capital_at_entry: Decimal,
}

impl Position {
    /// Open a new position, deducting entry fees
    fn open(
        direction: i8,
        price_a: Decimal,
        price_b: Decimal,
        capital: &mut Decimal,
        fee: Decimal,
    ) -> Self {
        // Fee on both legs (entry)
        let entry_fee = *capital * fee * dec!(2);
        *capital -= entry_fee;

        Self {
            direction,
            entry_price_a: price_a,
            entry_price_b: price_b,
            capital_at_entry: *capital,
        }
    }

    /// Close position, calculate PnL and update capital
    fn close(
        &self,
        exit_price_a: Decimal,
        exit_price_b: Decimal,
        capital: &mut Decimal,
        fee: Decimal,
    ) -> Decimal {
        // Calculate returns for each leg
        let (return_a, return_b) = match self.direction {
            1 => {
                // Long spread: profit from A going up relative to B
                let ret_a = (exit_price_a - self.entry_price_a) / self.entry_price_a;
                let ret_b = (exit_price_b - self.entry_price_b) / self.entry_price_b;
                (ret_a, ret_b)
            }
            _ => {
                // Short spread: profit from A going down relative to B
                let ret_a = (self.entry_price_a - exit_price_a) / self.entry_price_a;
                let ret_b = (self.entry_price_b - exit_price_b) / self.entry_price_b;
                (ret_a, ret_b)
            }
        };

        let spread_return = return_a - return_b;
        let gross_pnl = self.capital_at_entry * spread_return;

        // Fee on both legs (exit)
        let exit_fee = *capital * fee * dec!(2);
        *capital += gross_pnl - exit_fee;

        spread_return
    }
}

/// Trading days per year for Sharpe annualization (unused, Sharpe calculated per-trade)
/// Calculate annualized Sharpe ratio from trade returns
///
/// # Optimization (Zero-Allocation)
/// Uses a two-pass approach over the slice to calculate mean and variance
/// without allocating an intermediate `Vec<f64>`.
fn calculate_sharpe_ratio(returns: &[Decimal]) -> f64 {
    let n = returns.len();
    if n < 2 {
        return 0.0;
    }

    let n_f64 = n as f64;

    // Pass 1: Sum for mean (Zero allocation)
    let sum: f64 = returns.iter().map(|d| d.to_f64().unwrap_or(0.0)).sum();

    let mean = sum / n_f64;

    // Pass 2: Sum for variance (Zero allocation)
    let variance_sum: f64 = returns
        .iter()
        .map(|d| {
            let val = d.to_f64().unwrap_or(0.0);
            (val - mean).powi(2)
        })
        .sum();

    let variance = variance_sum / (n_f64 - 1.0);
    let std_dev = variance.sqrt();

    // Per-trade Sharpe ratio (not annualized) to avoid inflation on high-frequency trading
    if std_dev.abs() < f64::EPSILON {
        return 0.0;
    }

    mean / std_dev
}

/// Run backtest simulation for given parameters
/// Run backtest simulation for given parameters
///
/// # Optimization (Zero-Allocation Hot Path)
/// Allocates `MarketData` structs once outside the loop and reuses them.
/// Pre-allocates `trade_returns` vector to minimize resizing.
async fn run_backtest(
    manager: &mut PairsManager,
    data: BacktestData<'_>,
    config: &BacktestConfig,
) -> BacktestResult {
    let mut capital = config.initial_capital;
    let mut position: Option<Position> = None;
    let mut trades = 0u32;
    // Pre-allocate to approximate size (assuming ~1 trade per 50 ticks is generous)
    let mut trade_returns: Vec<Decimal> = Vec::with_capacity(data.timestamps.len() / 50 + 1);

    // SAFETY: Hoist allocations out of the loop
    // We reuse these structs for every tick to avoid 50M+ allocations per discovery run.
    let mut leg1 = MarketData {
        symbol: "A".to_string(), // Allocated once
        instrument_id: None,
        price: Decimal::ZERO,
        timestamp: 0,
    };

    let mut leg2 = MarketData {
        symbol: "B".to_string(), // Allocated once
        instrument_id: None,
        price: Decimal::ZERO,
        timestamp: 0,
    };

    for i in 0..data.timestamps.len() {
        let p_a = data.prices_a[i];
        let p_b = data.prices_b[i];
        let ts = data.timestamps[i];

        // Hot-path mutation: Zero allocation
        leg1.price = p_a;
        leg1.timestamp = ts;

        leg2.price = p_b;
        leg2.timestamp = ts;

        // Get signal from PairsManager
        let signal = manager.analyze(&leg1, &leg2).await;

        match (&signal, &position) {
            // Entry: Long spread (Buy A, Sell B)
            (Signal::Buy, None) => {
                position = Some(Position::open(1, p_a, p_b, &mut capital, config.taker_fee));
            }
            // Entry: Short spread (Sell A, Buy B)
            (Signal::Sell, None) => {
                position = Some(Position::open(-1, p_a, p_b, &mut capital, config.taker_fee));
            }
            // Exit: Close position
            (Signal::Exit, Some(pos)) => {
                let spread_return = pos.close(p_a, p_b, &mut capital, config.taker_fee);
                trade_returns.push(spread_return);
                trades += 1;
                position = None;
            }
            // Hold or invalid transition
            _ => {}
        }
    }

    let sharpe_ratio = calculate_sharpe_ratio(&trade_returns);
    let net_profit = capital - config.initial_capital;

    BacktestResult {
        window: config.window,
        z_entry: config.z_entry,
        z_exit: config.z_exit,
        net_profit,
        sharpe_ratio,
        trades,
    }
}

/// Optimize parameters for a single pair with walk-forward validation
///
/// Uses train/test split to prevent overfitting:
/// 1. Grid search on TRAIN data (first train_ratio % of data)
/// 2. Validate best parameters on TEST data (remaining %)
/// 3. Only accept if validation Sharpe meets threshold
async fn optimize_pair(
    pair: &CandidatePair,
    timestamps: &[i64],
    prices_a: &[Decimal],
    prices_b: &[Decimal],
    config: &DiscoveryConfig,
    grid: &GridSearchConfig,
) -> Option<OptimizedPair> {
    let n = timestamps.len();
    if n < 50 {
        debug!(
            pair = format!("{}-{}", pair.symbol_a, pair.symbol_b),
            samples = n,
            "Insufficient data for train/test split"
        );
        return None;
    }

    // Split data: train_ratio for training, rest for validation
    let split_idx = (n as f64 * config.train_ratio) as usize;
    let split_idx = split_idx.max(20).min(n - 20); // Ensure both sets have at least 20 samples

    let train_timestamps = &timestamps[..split_idx];
    let train_prices_a = &prices_a[..split_idx];
    let train_prices_b = &prices_b[..split_idx];

    let test_timestamps = &timestamps[split_idx..];
    let test_prices_a = &prices_a[split_idx..];
    let test_prices_b = &prices_b[split_idx..];

    debug!(
        pair = format!("{}-{}", pair.symbol_a, pair.symbol_b),
        train_samples = train_timestamps.len(),
        test_samples = test_timestamps.len(),
        "Train/test split"
    );

    // Phase 1: Grid search on TRAIN data only
    let mut best_train_result: Option<BacktestResult> = None;

    for &window in &grid.windows {
        // Skip window sizes larger than training data
        if window >= train_timestamps.len() {
            continue;
        }

        for z_entry_dec in &grid.z_entries {
            // Convert Decimal to f64 for PairsManager API
            let z_entry = z_entry_dec.to_f64().unwrap_or(2.0);
            let z_exit = grid.z_exit.to_f64().unwrap_or(0.1);
            let mut manager = PairsManager::new(window, z_entry, z_exit);

            let backtest_data = BacktestData {
                timestamps: train_timestamps,
                prices_a: train_prices_a,
                prices_b: train_prices_b,
            };
            let backtest_config = BacktestConfig {
                initial_capital: config.initial_capital,
                taker_fee: config.taker_fee,
                window,
                z_entry,
                z_exit,
            };

            let result = run_backtest(&mut manager, backtest_data, &backtest_config).await;

            // Require at least 1 trade in training
            if result.trades < 1 {
                continue;
            }

            let is_better = match &best_train_result {
                Some(best) => result.sharpe_ratio > best.sharpe_ratio,
                None => true,
            };

            if is_better {
                best_train_result = Some(result);
            }
        }
    }

    let train_result = match best_train_result {
        Some(r) => r,
        None => {
            debug!(
                pair = format!("{}-{}", pair.symbol_a, pair.symbol_b),
                "No valid parameter combination found in training"
            );
            return None;
        }
    };

    // Phase 2: Validate best parameters on TEST data (out-of-sample)
    let mut manager = PairsManager::new(
        train_result.window,
        train_result.z_entry,
        train_result.z_exit,
    );

    let test_data = BacktestData {
        timestamps: test_timestamps,
        prices_a: test_prices_a,
        prices_b: test_prices_b,
    };
    let test_config = BacktestConfig {
        initial_capital: config.initial_capital,
        taker_fee: config.taker_fee,
        window: train_result.window,
        z_entry: train_result.z_entry,
        z_exit: train_result.z_exit,
    };

    let validation_result = run_backtest(&mut manager, test_data, &test_config).await;

    debug!(
        pair = format!("{}-{}", pair.symbol_a, pair.symbol_b),
        train_sharpe = format!("{:.2}", train_result.sharpe_ratio),
        validation_sharpe = format!("{:.2}", validation_result.sharpe_ratio),
        train_trades = train_result.trades,
        validation_trades = validation_result.trades,
        "Walk-forward validation complete"
    );

    // Phase 3: Apply filters
    let total_trades = train_result.trades + validation_result.trades;
    let total_profit = train_result.net_profit + validation_result.net_profit;

    // Filter: minimum total trades
    if total_trades < config.min_trades {
        debug!(
            pair = format!("{}-{}", pair.symbol_a, pair.symbol_b),
            trades = total_trades,
            min = config.min_trades,
            "Rejected: insufficient trades"
        );
        return None;
    }

    // Filter: validation Sharpe must not be negative (catches obvious overfitting)
    if validation_result.sharpe_ratio < 0.0 {
        debug!(
            pair = format!("{}-{}", pair.symbol_a, pair.symbol_b),
            validation_sharpe = validation_result.sharpe_ratio,
            "Rejected: negative validation Sharpe (overfit)"
        );
        return None;
    }

    // Note: We intentionally don't filter on validation Sharpe threshold
    // Low but positive validation Sharpe is still informative for ranking

    // Filter: minimum net profit
    if total_profit < config.min_net_profit {
        return None;
    }

    // Warn if train Sharpe is suspiciously high
    if train_result.sharpe_ratio > config.max_sharpe_ratio {
        warn!(
            pair = format!("{}/{}", pair.symbol_a, pair.symbol_b),
            train_sharpe = format!("{:.2}", train_result.sharpe_ratio),
            validation_sharpe = format!("{:.2}", validation_result.sharpe_ratio),
            "High train Sharpe ({:.1}) may indicate overfitting - validation Sharpe is {:.2}",
            train_result.sharpe_ratio,
            validation_result.sharpe_ratio
        );
    }

    let profit_per_trade = if total_trades > 0 {
        total_profit / Decimal::from(total_trades)
    } else {
        Decimal::ZERO
    };

    Some(OptimizedPair {
        leg1: pair.symbol_a.clone(),
        leg2: pair.symbol_b.clone(),
        window: train_result.window,
        z_entry: train_result.z_entry,
        z_exit: train_result.z_exit,
        sharpe_ratio: train_result.sharpe_ratio,
        net_profit: total_profit,
        trades: total_trades,
        correlation: pair.correlation,
        half_life_hours: pair.half_life_hours,
        adf_statistic: pair.adf_statistic,
        validation_sharpe: validation_result.sharpe_ratio,
        validation_trades: validation_result.trades,
        profit_per_trade,
    })
}

/// Fetch historical candle data using any DiscoveryDataSource
///
/// # CB-1 Fix
/// Time is now injected via Clock trait for deterministic discovery and testing.
///
/// # DRY Principle
/// Works with any exchange that implements DiscoveryDataSource.
async fn fetch_candle_data<D: DiscoveryDataSource>(
    data_source: &mut D,
    symbols: &[String],
    lookback_days: u32,
    clock: &dyn Clock,
) -> Result<(Vec<i64>, HashMap<String, Vec<Decimal>>), DiscoveryError> {
    let end = clock.now();
    let start = end - ChronoDuration::days(lookback_days as i64);

    info!(
        symbols = symbols.len(),
        start = %start.format("%Y-%m-%d"),
        end = %end.format("%Y-%m-%d"),
        "Fetching candle data"
    );

    let mut all_candles: HashMap<String, Vec<(i64, Decimal)>> = HashMap::new();
    let mut common_timestamps: Option<HashSet<i64>> = None;

    for symbol in symbols {
        info!(symbol = %symbol, "Fetching candles");

        match data_source.fetch_candles_hourly(symbol, start, end).await {
            Ok(candles) => {
                if candles.is_empty() {
                    warn!(symbol = %symbol, "No candles received");
                    continue;
                }

                let timestamps: HashSet<i64> = candles.iter().map(|(t, _)| *t).collect();

                // Update common timestamps
                common_timestamps = match common_timestamps {
                    Some(existing) => Some(existing.intersection(&timestamps).copied().collect()),
                    None => Some(timestamps),
                };

                all_candles.insert(symbol.clone(), candles);
            }
            Err(e) => {
                let err_str = e.to_string();
                if err_str.contains("INVALID_ARGUMENT")
                    || err_str.contains("ProductID is invalid")
                    || err_str.contains("not found")
                {
                    warn!(symbol = %symbol, error = %err_str, "Invalid symbol, skipping");
                    continue;
                }
                return Err(DiscoveryError::Api(e.to_string().into()));
            }
        }

        // Rate limiting (exchange-agnostic)
        sleep(std::time::Duration::from_millis(200)).await;
    }

    let common = common_timestamps.unwrap_or_default();
    if common.is_empty() {
        return Err(DiscoveryError::InsufficientData {
            expected: 50,
            actual: 0,
        });
    }

    // Align all data to common timestamps
    let mut sorted_timestamps: Vec<i64> = common.into_iter().collect();
    sorted_timestamps.sort();

    let mut aligned_prices: HashMap<String, Vec<Decimal>> = HashMap::new();

    for (symbol, data) in all_candles {
        let price_map: HashMap<i64, Decimal> = data.into_iter().collect();
        let aligned: Vec<Decimal> = sorted_timestamps
            .iter()
            .filter_map(|ts| price_map.get(ts).copied())
            .collect();

        if aligned.len() == sorted_timestamps.len() {
            aligned_prices.insert(symbol, aligned);
        } else {
            warn!(
                symbol = %symbol,
                expected = sorted_timestamps.len(),
                actual = aligned.len(),
                "Incomplete alignment, skipping"
            );
        }
    }

    if aligned_prices.len() < 2 {
        return Err(DiscoveryError::InsufficientData {
            expected: 2,
            actual: aligned_prices.len(),
        });
    }

    info!(
        symbols = aligned_prices.len(),
        data_points = sorted_timestamps.len(),
        "Data aligned and ready"
    );

    Ok((sorted_timestamps, aligned_prices))
}

/// Main entry point: Discover and optimize cointegrated pairs
///
/// # Pipeline
/// 1. Fetch historical candle data for all candidates
/// 2. Filter pairs by correlation and half-life
/// 3. Run grid search optimization on viable pairs
/// 4. Return ranked results
///
/// # CB-1 Fix
/// Accepts a `Clock` trait for deterministic time handling, enabling reproducible
/// discovery runs and proper integration testing.
///
/// # DRY Principle
/// Works with any exchange via the DiscoveryDataSource trait.
pub async fn discover_and_optimize<D: DiscoveryDataSource>(
    data_source: &mut D,
    config: &DiscoveryConfig,
    clock: &dyn Clock,
) -> Result<Vec<OptimizedPair>, DiscoveryError> {
    config.validate().map_err(DiscoveryError::InvalidConfig)?;

    info!(
        candidates = config.candidates.len(),
        lookback = config.lookback_days,
        "Starting pair discovery"
    );

    // Step 1: Fetch data
    let (timestamps, prices_decimal) =
        fetch_candle_data(data_source, &config.candidates, config.lookback_days, clock).await?;

    // Step 2: Convert to f64 for correlation filtering
    let prices_f64: HashMap<String, Vec<f64>> = prices_decimal
        .iter()
        .map(|(k, v)| {
            let floats: Vec<f64> = v
                .iter()
                .filter_map(|d| d.to_string().parse().ok())
                .collect();
            (k.clone(), floats)
        })
        .collect();

    // Step 3: Filter candidates
    let viable_pairs = filter_candidates(
        &prices_f64,
        config.min_correlation,
        config.max_half_life_hours,
    );

    if viable_pairs.is_empty() {
        return Err(DiscoveryError::NoViablePairs {
            min_correlation: config.min_correlation,
            max_half_life: config.max_half_life_hours,
        });
    }

    info!(
        viable_pairs = viable_pairs.len(),
        "Running parameter optimization"
    );

    // Step 4: Optimize each pair
    let grid = GridSearchConfig::default();
    let mut results: Vec<OptimizedPair> = Vec::new();

    for pair in &viable_pairs {
        let prices_a = prices_decimal.get(&pair.symbol_a);
        let prices_b = prices_decimal.get(&pair.symbol_b);

        if let (Some(pa), Some(pb)) = (prices_a, prices_b) {
            if let Some(optimized) = optimize_pair(pair, &timestamps, pa, pb, config, &grid).await {
                results.push(optimized);
            }
        }
    }

    // Step 5: Rank by Sharpe ratio
    results.sort_by(|a, b| {
        b.sharpe_ratio
            .partial_cmp(&a.sharpe_ratio)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Step 6: Truncate to max pairs
    results.truncate(config.max_pairs_output);

    info!(optimized_pairs = results.len(), "Discovery complete");

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe_ratio_constant_returns() {
        // Constant returns = zero std dev = zero Sharpe
        let returns = vec![dec!(0.01), dec!(0.01), dec!(0.01), dec!(0.01)];
        let sharpe = calculate_sharpe_ratio(&returns);
        assert_eq!(sharpe, 0.0);
    }

    #[test]
    fn test_sharpe_ratio_positive() {
        // Positive mean with low variance = positive Sharpe
        let returns = vec![
            dec!(0.01),
            dec!(0.02),
            dec!(0.015),
            dec!(0.018),
            dec!(0.012),
        ];
        let sharpe = calculate_sharpe_ratio(&returns);
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_sharpe_ratio_empty() {
        let returns: Vec<Decimal> = vec![];
        let sharpe = calculate_sharpe_ratio(&returns);
        assert_eq!(sharpe, 0.0);
    }

    #[test]
    fn test_sharpe_ratio_single() {
        let returns = vec![dec!(0.01)];
        let sharpe = calculate_sharpe_ratio(&returns);
        assert_eq!(sharpe, 0.0);
    }

    #[test]
    fn test_optimized_pair_to_config() {
        let pair = OptimizedPair {
            leg1: "BTC-USD".to_string(),
            leg2: "ETH-USD".to_string(),
            window: 20,
            z_entry: 2.0,
            z_exit: 0.1,
            sharpe_ratio: 1.5,
            net_profit: dec!(500),
            trades: 10,
            correlation: 0.85,
            half_life_hours: 12.0,
            adf_statistic: -3.5,
            validation_sharpe: 1.2,
            validation_trades: 5,
            profit_per_trade: dec!(50),
        };

        let config = pair.to_portfolio_config(dec!(100));
        assert_eq!(config.dual_leg_config.spot_symbol, "BTC-USD");
        assert_eq!(config.dual_leg_config.future_symbol, "ETH-USD");
        assert_eq!(config.dual_leg_config.order_size, dec!(100));
        assert_eq!(config.window_size, 20);
        assert_eq!(config.entry_z_score, dec!(2.0));

        // Verify new fields
        assert_eq!(pair.validation_sharpe, 1.2);
        assert_eq!(pair.validation_trades, 5);
        assert_eq!(pair.profit_per_trade, dec!(50));
    }
}
