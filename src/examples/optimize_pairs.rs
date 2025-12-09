//! Pairs Trading Parameter Optimizer
//!
//! Grid-search optimizer for pairs trading parameters (window size, z-score thresholds).
//! Simulates historical trading with proper PnL and Sharpe ratio calculation.
//!
//! Usage:
//!   cargo run --example optimize_pairs -- --leg1 BTC-USD --leg2 ETH-USD --days 7
//!   cargo run --example optimize_pairs -- --mock  # Use mock data

use algopioneer::coinbase::{AppEnv, CoinbaseClient};
use algopioneer::strategy::dual_leg_trading::{EntryStrategy, MarketData, PairsManager};
use algopioneer::strategy::Signal;
use cbadv::time::Granularity;
use chrono::{Duration as ChronoDuration, Utc};
use clap::Parser;
use rust_decimal::Decimal;
use std::collections::HashSet;
use tracing::{debug, info, warn};

// --- Configuration Constants ---
const INITIAL_CAPITAL: f64 = 10_000.0;
const TAKER_FEE: f64 = 0.002; // 0.2%
const MIN_TRADES_FILTER: u32 = 5;
const ANNUALIZATION_FACTOR: f64 = 252.0; // Trading days per year
const MOCK_DATA_SIZE: usize = 1000;

// --- CLI Arguments ---

#[derive(Parser, Debug)]
#[command(name = "optimize_pairs")]
#[command(about = "Grid-search optimizer for pairs trading parameters")]
struct Args {
    /// First leg product ID (e.g., BTC-USD)
    #[arg(long, default_value = "BTC-USD")]
    leg1: String,

    /// Second leg product ID (e.g., ETH-USD)
    #[arg(long, default_value = "ETH-USD")]
    leg2: String,

    /// Lookback period in days
    #[arg(long, default_value_t = 7)]
    days: u32,

    /// End date for backtest (YYYY-MM-DD). Defaults to today.
    #[arg(long)]
    end_date: Option<String>,

    /// Candle granularity: 1h, 6h, or 1d
    #[arg(long, default_value = "1h")]
    granularity: String,

    /// Use mock data instead of Coinbase API
    #[arg(long)]
    mock: bool,
}

/// Parse granularity string to cbadv Granularity enum
fn parse_granularity(s: &str) -> Granularity {
    match s.to_lowercase().as_str() {
        "5m" => Granularity::FiveMinute,
        "15m" => Granularity::FifteenMinute,
        "30m" => Granularity::ThirtyMinute,
        "6h" => Granularity::SixHour,
        "1d" | "d" => Granularity::OneDay,
        _ => Granularity::OneHour, // default
    }
}

// --- Data Structures ---

#[derive(Debug, Clone)]
#[allow(dead_code)] // z_exit kept for debugging/logging purposes
struct BacktestResult {
    window: usize,
    z_entry: f64,
    z_exit: f64,
    net_profit: f64,
    sharpe_ratio: f64,
    trades: u32,
}

/// Tracks open position state for PnL calculation
#[derive(Debug, Clone)]
struct Position {
    direction: i8, // 1 = Long Spread (Long A, Short B), -1 = Short Spread
    entry_price_a: f64,
    entry_price_b: f64,
    capital_at_entry: f64,
}

impl Position {
    /// Opens a new position, deducting entry fees from capital.
    fn open(direction: i8, price_a: f64, price_b: f64, capital: &mut f64) -> Self {
        let entry_fee = *capital * TAKER_FEE * 2.0; // Fee on both legs
        *capital -= entry_fee;
        Self {
            direction,
            entry_price_a: price_a,
            entry_price_b: price_b,
            capital_at_entry: *capital,
        }
    }

    /// Closes the position, returning the spread return and updating capital.
    fn close(&self, exit_price_a: f64, exit_price_b: f64, capital: &mut f64) -> f64 {
        let (return_a, return_b) = match self.direction {
            1 => (
                (exit_price_a - self.entry_price_a) / self.entry_price_a,
                (exit_price_b - self.entry_price_b) / self.entry_price_b,
            ),
            _ => (
                (self.entry_price_a - exit_price_a) / self.entry_price_a,
                (self.entry_price_b - exit_price_b) / self.entry_price_b,
            ),
        };
        let spread_return = return_a - return_b;
        let gross_pnl = self.capital_at_entry * spread_return;
        let exit_fee = *capital * TAKER_FEE * 2.0;
        *capital += gross_pnl - exit_fee;
        spread_return
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    dotenv::dotenv().ok();

    let args = Args::parse();

    // 1. Load Data
    info!("Loading market data...");
    let (dates, prices_a, prices_b) = if args.mock {
        info!("Using mock random walk data");
        load_mock_data()
    } else {
        let granularity = parse_granularity(&args.granularity);
        info!(granularity = args.granularity, end_date = ?args.end_date, "Using Coinbase API data");
        load_coinbase_data(
            &args.leg1,
            &args.leg2,
            args.days,
            args.end_date.as_deref(),
            granularity,
        )
        .await?
    };
    let n_rows = dates.len();

    info!(data_points = n_rows, "Starting Grid Search");

    // 2. Define Parameter Ranges
    let windows = (10..=60).step_by(5); // Test windows: 10, 15, 20... 60
    let z_entries = (15..=30).map(|i| i as f64 / 10.0); // Test Zs: 1.5, 1.6... 3.0
    let z_exit = 0.1; // Keep exit constant (Mean Reversion)

    let mut results = Vec::new();

    // 3. The Optimization Loop
    for window in windows {
        for z_entry in z_entries.clone() {
            let manager = PairsManager::new(window, z_entry, z_exit);

            let result = run_simulation(
                &manager, &dates, &prices_a, &prices_b, window, z_entry, z_exit,
            )
            .await;

            if result.trades > MIN_TRADES_FILTER {
                results.push(result);
            }
        }
        debug!(window, "Completed window iteration");
    }

    info!("Optimization Complete");

    // 4. Sort and Display Top 5 Results by Net Profit
    results.sort_by(|a, b| {
        b.net_profit
            .partial_cmp(&a.net_profit)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!(
        "\n{:<10} | {:<10} | {:<10} | {:<15} | {:<10}",
        "Window", "Z-Entry", "Trades", "Net Profit ($)", "Sharpe"
    );
    println!("{}", "-".repeat(70));

    for res in results.iter().take(5) {
        println!(
            "{:<10} | {:<10.1} | {:<10} | ${:<14.2} | {:<10.2}",
            res.window, res.z_entry, res.trades, res.net_profit, res.sharpe_ratio
        );
    }

    Ok(())
}

/// Simulates the Pairs Trading strategy on historical arrays
async fn run_simulation(
    manager: &PairsManager,
    dates: &[i64],
    prices_a: &[f64],
    prices_b: &[f64],
    window: usize,
    z_entry: f64,
    z_exit: f64,
) -> BacktestResult {
    let mut capital = INITIAL_CAPITAL;
    let mut position: Option<Position> = None;
    let mut trades = 0u32;
    let mut trade_returns: Vec<f64> = Vec::with_capacity(dates.len() / 10);

    // OW2: Hoist String allocation outside the hot loop
    let symbol_a: String = "A".into();
    let symbol_b: String = "B".into();

    for i in 0..dates.len() {
        let p_a = prices_a[i];
        let p_b = prices_b[i];

        // Convert to Decimal for PairsManager (required by interface)
        let Some(dec_a) = Decimal::from_f64_retain(p_a) else {
            continue;
        };
        let Some(dec_b) = Decimal::from_f64_retain(p_b) else {
            continue;
        };

        let leg1 = MarketData {
            symbol: symbol_a.clone(),
            price: dec_a,
            instrument_id: None,
            timestamp: dates[i],
        };
        let leg2 = MarketData {
            symbol: symbol_b.clone(),
            price: dec_b,
            instrument_id: None,
            timestamp: dates[i],
        };

        // Get Signal from PairsManager
        let signal = manager.analyze(&leg1, &leg2).await;

        match (&signal, &position) {
            // --- ENTRY: Long Spread (Buy A, Sell B) ---
            (Signal::Buy, None) => {
                position = Some(Position::open(1, p_a, p_b, &mut capital));
            }

            // --- ENTRY: Short Spread (Sell A, Buy B) ---
            (Signal::Sell, None) => {
                position = Some(Position::open(-1, p_a, p_b, &mut capital));
            }

            // --- EXIT: Close position ---
            (Signal::Exit, Some(pos)) => {
                let spread_return = pos.close(p_a, p_b, &mut capital);
                trade_returns.push(spread_return);
                trades += 1;
                position = None;
            }

            // --- HOLD or invalid state transition ---
            _ => {}
        }
    }

    // Calculate Sharpe Ratio from trade returns
    let sharpe_ratio = calculate_sharpe_ratio(&trade_returns);
    let net_profit = capital - INITIAL_CAPITAL;

    BacktestResult {
        window,
        z_entry,
        z_exit,
        net_profit,
        sharpe_ratio,
        trades,
    }
}

/// Calculate annualized Sharpe Ratio from returns
fn calculate_sharpe_ratio(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;

    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();

    if std_dev.abs() < f64::EPSILON {
        return 0.0;
    }

    // Annualized Sharpe (assuming each trade is ~1 day, adjust as needed)
    (mean / std_dev) * ANNUALIZATION_FACTOR.sqrt()
}

/// Generate mock correlated random walk data for testing
fn load_mock_data() -> (Vec<i64>, Vec<f64>, Vec<f64>) {
    use rand::Rng;

    let mut dates = Vec::with_capacity(MOCK_DATA_SIZE);
    let mut a = Vec::with_capacity(MOCK_DATA_SIZE);
    let mut b = Vec::with_capacity(MOCK_DATA_SIZE);

    let mut price_a = 100.0;
    let mut price_b = 10.0;
    let mut rng = rand::rng();

    for i in 0..MOCK_DATA_SIZE {
        dates.push(i as i64);

        // Correlated move + idiosyncratic component
        let move_common = rng.random_range(-0.01..0.01);
        let move_a = rng.random_range(-0.002..0.002);
        let move_b = rng.random_range(-0.002..0.002);

        price_a *= 1.0 + move_common + move_a;
        price_b *= 1.0 + move_common + move_b;

        a.push(price_a);
        b.push(price_b);
    }

    (dates, a, b)
}

/// Load historical candle data from Coinbase API
async fn load_coinbase_data(
    leg1: &str,
    leg2: &str,
    days: u32,
    end_date: Option<&str>,
    granularity: Granularity,
) -> Result<(Vec<i64>, Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
    use chrono::NaiveDate;

    info!("Connecting to Coinbase API...");

    let mut client = CoinbaseClient::new(AppEnv::Live)?;

    // Parse end date or use now
    let end = match end_date {
        Some(date_str) => {
            let naive = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")?;
            naive.and_hms_opt(23, 59, 59).unwrap().and_utc()
        }
        None => Utc::now(),
    };
    let start = end - ChronoDuration::days(days as i64);

    info!(
        leg1 = leg1,
        leg2 = leg2,
        start = %start.format("%Y-%m-%d %H:%M"),
        end = %end.format("%Y-%m-%d %H:%M"),
        "Fetching candles (paginated)"
    );

    // Fetch candles for both legs using paginated method
    let candles_a = client
        .get_product_candles_paginated(leg1, &start, &end, granularity.clone())
        .await?;
    let candles_b = client
        .get_product_candles_paginated(leg2, &start, &end, granularity)
        .await?;

    info!(
        leg1_candles = candles_a.len(),
        leg2_candles = candles_b.len(),
        "Candles received"
    );

    if candles_a.is_empty() || candles_b.is_empty() {
        return Err("No candle data received from API".into());
    }

    // Align by timestamp (use intersection of available data)
    let timestamps_a: HashSet<_> = candles_a.iter().map(|c| c.start).collect();
    let timestamps_b: HashSet<_> = candles_b.iter().map(|c| c.start).collect();
    let common: HashSet<_> = timestamps_a.intersection(&timestamps_b).copied().collect();

    if common.is_empty() {
        return Err("No overlapping timestamps between the two pairs".into());
    }

    // Build aligned data vectors
    let mut data: Vec<(i64, f64, f64)> = candles_a
        .iter()
        .filter(|c| common.contains(&c.start))
        .filter_map(|c| {
            candles_b
                .iter()
                .find(|b| b.start == c.start)
                .map(|b| (c.start as i64, c.close, b.close))
        })
        .collect();

    // Sort by timestamp (oldest first)
    data.sort_by_key(|(t, _, _)| *t);

    let dates: Vec<i64> = data.iter().map(|(t, _, _)| *t).collect();
    let prices_a: Vec<f64> = data.iter().map(|(_, a, _)| *a).collect();
    let prices_b: Vec<f64> = data.iter().map(|(_, _, b)| *b).collect();

    if dates.len() < 50 {
        warn!(
            data_points = dates.len(),
            "Low data point count may affect optimization quality"
        );
    }

    info!(
        aligned_data_points = dates.len(),
        first_price_a = prices_a.first().unwrap_or(&0.0),
        first_price_b = prices_b.first().unwrap_or(&0.0),
        "Data aligned and ready"
    );

    Ok((dates, prices_a, prices_b))
}
