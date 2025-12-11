//! Grid Search Optimizer for Pairs Trading Parameters
//!
//! Implements parameter optimization through historical backtesting
//! with proper Decimal-based financial calculations.

use crate::coinbase::CoinbaseClient;
use crate::strategy::dual_leg_trading::{
    DualLegConfig, EntryStrategy, MarketData, PairsManager, TransactionCostModel,
};
use crate::strategy::portfolio::PortfolioPairConfig;
use crate::strategy::Signal;

use super::config::{DiscoveryConfig, GridSearchConfig};
use super::error::DiscoveryError;
use super::filter::{filter_candidates, CandidatePair};

use cbadv::time::Granularity;
use chrono::{Duration as ChronoDuration, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::time::sleep;
use tracing::{debug, info, warn};

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
    /// Annualized Sharpe ratio
    pub sharpe_ratio: f64,
    /// Net profit in USD
    pub net_profit: Decimal,
    /// Total number of trades
    pub trades: u32,
    /// Pearson correlation
    pub correlation: f64,
    /// Mean-reversion half-life (hours)
    pub half_life_hours: f64,
}

impl OptimizedPair {
    /// Convert to PortfolioPairConfig for use with PortfolioManager
    pub fn to_portfolio_config(&self) -> PortfolioPairConfig {
        PortfolioPairConfig {
            dual_leg_config: DualLegConfig {
                spot_symbol: self.leg1.clone(),
                future_symbol: self.leg2.clone(),
                order_size: dec!(10.0), // Default $10 allocation
                max_tick_age_ms: 2000,
                execution_timeout_ms: 30000,
                min_profit_threshold: dec!(0.001),
                stop_loss_threshold: dec!(-0.02),
                fee_tier: TransactionCostModel::default(),
                throttle_interval_secs: 5,
            },
            window_size: self.window,
            entry_z_score: self.z_entry,
            exit_z_score: self.z_exit,
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
    fn open(direction: i8, price_a: Decimal, price_b: Decimal, capital: &mut Decimal, fee: Decimal) -> Self {
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
    fn close(&self, exit_price_a: Decimal, exit_price_b: Decimal, capital: &mut Decimal, fee: Decimal) -> Decimal {
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

/// Trading days per year for Sharpe annualization
const ANNUALIZATION_FACTOR: f64 = 252.0;

/// Calculate annualized Sharpe ratio from trade returns
fn calculate_sharpe_ratio(returns: &[Decimal]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    // Convert to f64 for statistical calculations
    let float_returns: Vec<f64> = returns
        .iter()
        .filter_map(|d| d.to_string().parse::<f64>().ok())
        .collect();

    if float_returns.len() < 2 {
        return 0.0;
    }

    let n = float_returns.len() as f64;
    let mean = float_returns.iter().sum::<f64>() / n;
    
    // Sample variance (n-1 denominator)
    let variance = float_returns
        .iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / (n - 1.0);
    
    let std_dev = variance.sqrt();

    if std_dev.abs() < f64::EPSILON {
        return 0.0;
    }

    // Annualized Sharpe ratio
    (mean / std_dev) * ANNUALIZATION_FACTOR.sqrt()
}

/// Run backtest simulation for given parameters
async fn run_backtest(
    manager: &PairsManager,
    timestamps: &[i64],
    prices_a: &[Decimal],
    prices_b: &[Decimal],
    initial_capital: Decimal,
    taker_fee: Decimal,
    window: usize,
    z_entry: f64,
    z_exit: f64,
) -> BacktestResult {
    let mut capital = initial_capital;
    let mut position: Option<Position> = None;
    let mut trades = 0u32;
    let mut trade_returns: Vec<Decimal> = Vec::with_capacity(timestamps.len() / 10);

    // Pre-allocate symbol strings (optimization)
    let symbol_a: Arc<str> = Arc::from("A");
    let symbol_b: Arc<str> = Arc::from("B");

    for i in 0..timestamps.len() {
        let p_a = prices_a[i];
        let p_b = prices_b[i];

        let leg1 = MarketData {
            symbol: symbol_a.to_string(),
            price: p_a,
            instrument_id: None,
            timestamp: timestamps[i],
        };
        let leg2 = MarketData {
            symbol: symbol_b.to_string(),
            price: p_b,
            instrument_id: None,
            timestamp: timestamps[i],
        };

        // Get signal from PairsManager
        let signal = manager.analyze(&leg1, &leg2).await;

        match (&signal, &position) {
            // Entry: Long spread (Buy A, Sell B)
            (Signal::Buy, None) => {
                position = Some(Position::open(1, p_a, p_b, &mut capital, taker_fee));
            }
            // Entry: Short spread (Sell A, Buy B)
            (Signal::Sell, None) => {
                position = Some(Position::open(-1, p_a, p_b, &mut capital, taker_fee));
            }
            // Exit: Close position
            (Signal::Exit, Some(pos)) => {
                let spread_return = pos.close(p_a, p_b, &mut capital, taker_fee);
                trade_returns.push(spread_return);
                trades += 1;
                position = None;
            }
            // Hold or invalid transition
            _ => {}
        }
    }

    let sharpe_ratio = calculate_sharpe_ratio(&trade_returns);
    let net_profit = capital - initial_capital;

    BacktestResult {
        window,
        z_entry,
        z_exit,
        net_profit,
        sharpe_ratio,
        trades,
    }
}

/// Optimize parameters for a single pair
async fn optimize_pair(
    pair: &CandidatePair,
    timestamps: &[i64],
    prices_a: &[Decimal],
    prices_b: &[Decimal],
    config: &DiscoveryConfig,
    grid: &GridSearchConfig,
) -> Option<OptimizedPair> {
    let mut best_result: Option<BacktestResult> = None;

    for &window in &grid.windows {
        for &z_entry in &grid.z_entries {
            // Create fresh manager for each parameter combination
            let manager = PairsManager::new(window, z_entry, grid.z_exit);

            let result = run_backtest(
                &manager,
                timestamps,
                prices_a,
                prices_b,
                config.initial_capital,
                config.taker_fee,
                window,
                z_entry,
                grid.z_exit,
            )
            .await;

            // Filter by minimum trades
            if result.trades < config.min_trades {
                continue;
            }

            // Update best if better Sharpe (or first valid result)
            let is_better = match &best_result {
                Some(best) => result.sharpe_ratio > best.sharpe_ratio,
                None => true,
            };

            if is_better {
                best_result = Some(result);
            }
        }
    }

    debug!(
        pair = format!("{}-{}", pair.symbol_a, pair.symbol_b),
        "Grid search complete for pair"
    );

    best_result.and_then(|result| {
        // Filter by minimum Sharpe ratio
        if result.sharpe_ratio < config.min_sharpe_ratio {
            return None;
        }

        Some(OptimizedPair {
            leg1: pair.symbol_a.clone(),
            leg2: pair.symbol_b.clone(),
            window: result.window,
            z_entry: result.z_entry,
            z_exit: result.z_exit,
            sharpe_ratio: result.sharpe_ratio,
            net_profit: result.net_profit,
            trades: result.trades,
            correlation: pair.correlation,
            half_life_hours: pair.half_life_hours,
        })
    })
}

/// Fetch historical candle data from Coinbase API
async fn fetch_candle_data(
    client: &mut CoinbaseClient,
    symbols: &[String],
    lookback_days: u32,
) -> Result<(Vec<i64>, HashMap<String, Vec<Decimal>>), DiscoveryError> {
    let end = Utc::now();
    let start = end - ChronoDuration::days(lookback_days as i64);

    info!(
        symbols = symbols.len(),
        start = %start.format("%Y-%m-%d"),
        end = %end.format("%Y-%m-%d"),
        "Fetching candle data"
    );

    let mut all_candles: HashMap<String, Vec<(i64, f64)>> = HashMap::new();
    let mut common_timestamps: Option<HashSet<i64>> = None;

    for symbol in symbols {
        info!(symbol = %symbol, "Fetching candles");
        
        match client
            .get_product_candles_paginated(symbol, &start, &end, Granularity::OneHour)
            .await
        {
            Ok(candles) => {
                if candles.is_empty() {
                    warn!(symbol = %symbol, "No candles received");
                    continue;
                }

                // Extract timestamps and close prices
                let data: Vec<(i64, f64)> = candles
                    .iter()
                    .map(|c| (c.start as i64, c.close))
                    .collect();

                let timestamps: HashSet<i64> = data.iter().map(|(t, _)| *t).collect();

                // Update common timestamps
                common_timestamps = match common_timestamps {
                    Some(existing) => Some(existing.intersection(&timestamps).copied().collect()),
                    None => Some(timestamps),
                };

                all_candles.insert(symbol.clone(), data);
            }
            Err(e) => {
                let err_str = e.to_string();
                if err_str.contains("INVALID_ARGUMENT") || err_str.contains("ProductID is invalid") {
                    warn!(symbol = %symbol, "Invalid product ID, skipping");
                    continue;
                }
                return Err(DiscoveryError::Api(e.to_string().into()));
            }
        }

        // Rate limiting
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
        let price_map: HashMap<i64, f64> = data.into_iter().collect();
        let aligned: Vec<Decimal> = sorted_timestamps
            .iter()
            .filter_map(|ts| {
                price_map.get(ts).and_then(|p| {
                    Decimal::from_f64_retain(*p)
                })
            })
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
pub async fn discover_and_optimize(
    client: &mut CoinbaseClient,
    config: &DiscoveryConfig,
) -> Result<Vec<OptimizedPair>, DiscoveryError> {
    config.validate().map_err(DiscoveryError::InvalidConfig)?;

    info!(
        candidates = config.candidates.len(),
        lookback = config.lookback_days,
        "Starting pair discovery"
    );

    // Step 1: Fetch data
    let (timestamps, prices_decimal) = fetch_candle_data(client, &config.candidates, config.lookback_days).await?;

    // Step 2: Convert to f64 for correlation filtering
    let prices_f64: HashMap<String, Vec<f64>> = prices_decimal
        .iter()
        .map(|(k, v)| {
            let floats: Vec<f64> = v.iter()
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

    info!(viable_pairs = viable_pairs.len(), "Running parameter optimization");

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

    info!(
        optimized_pairs = results.len(),
        "Discovery complete"
    );

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
        let returns = vec![dec!(0.01), dec!(0.02), dec!(0.015), dec!(0.018), dec!(0.012)];
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
        };

        let config = pair.to_portfolio_config();
        assert_eq!(config.dual_leg_config.spot_symbol, "BTC-USD");
        assert_eq!(config.dual_leg_config.future_symbol, "ETH-USD");
        assert_eq!(config.window_size, 20);
        assert_eq!(config.entry_z_score, 2.0);
    }
}
