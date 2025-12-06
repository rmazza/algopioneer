use algopioneer::strategy::dual_leg_trading::{PairsManager, EntryStrategy, MarketData, TransactionCostModel};
use algopioneer::strategy::Signal;
use polars::prelude::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;
use chrono::Utc;
use std::sync::Arc;

// --- Configuration ---
const INITIAL_CAPITAL: f64 = 10_000.0;
const LEVERAGE: f64 = 1.0; // 1x leverage for safety
const MAKER_FEE: f64 = 0.001; // 0.1%
const TAKER_FEE: f64 = 0.002; // 0.2%

#[derive(Debug, Clone)]
struct BacktestResult {
    window: usize,
    z_entry: f64,
    z_exit: f64,
    net_profit: f64,
    sharpe_ratio: f64,
    trades: u32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load Data (Mocking CSV loading for this example)
    // In reality: let df = CsvReader::from_path("data/sol_avax_1m.csv")?...
    println!("Loading market data...");
    let (dates, prices_a, prices_b) = load_mock_data(); 
    let n_rows = dates.len();

    println!("Starting Grid Search on {} data points...", n_rows);

    // 2. Define Parameter Ranges
    let windows = (10..=60).step_by(5); // Test windows: 10, 15, 20... 60
    let z_entries = (15..=30).map(|i| i as f64 / 10.0); // Test Zs: 1.5, 1.6... 3.0
    let z_exit = 0.1; // Keep exit constant for now (Mean Reversion)

    let mut results = Vec::new();

    // 3. The Optimization Loop
    for window in windows {
        for z_entry in z_entries.clone() {
            // Instantiate the Strategy Logic
            let manager = PairsManager::new(window, z_entry, z_exit);
            
            // Run the Simulation
            let result = run_simulation(
                &manager, 
                &dates, 
                &prices_a, 
                &prices_b, 
                window, 
                z_entry, 
                z_exit
            ).await;

            if result.trades > 5 { // Filter out strategies that barely trade
                results.push(result);
            }
        }
        print!("."); // Progress indicator
        use std::io::Write;
        std::io::stdout().flush().unwrap();
    }

    println!("\n\n--- Optimization Complete ---");

    // 4. Sort and Display Top 5 Results
    results.sort_by(|a, b| b.net_profit.partial_cmp(&a.net_profit).unwrap());

    println!("{:<10} | {:<10} | {:<10} | {:<15} | {:<10}", "Window", "Z-Entry", "Trades", "Net Profit ($)", "Sharpe");
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
    z_exit: f64
) -> BacktestResult {
    let mut capital = INITIAL_CAPITAL;
    let mut position = 0; // 0 = Flat, 1 = Long Spread (Long A/Short B), -1 = Short Spread
    let mut entry_equity = 0.0;
    let mut trades = 0;
    let mut daily_returns = Vec::new();
    let mut last_equity = INITIAL_CAPITAL;

    // Iterate through time
    for i in 0..dates.len() {
        let p_a = Decimal::from_f64_retain(prices_a[i]).unwrap();
        let p_b = Decimal::from_f64_retain(prices_b[i]).unwrap();
        
        let leg1 = MarketData { symbol: "A".into(), price: p_a, timestamp: dates[i] };
        let leg2 = MarketData { symbol: "B".into(), price: p_b, timestamp: dates[i] };

        // Get Signal from the "Brain"
        let signal = manager.analyze(&leg1, &leg2).await;

        // --- Execution Logic (Simplified for Backtest) ---
        // PnL Logic: 
        // Long Spread  = Buy A ($1000) + Sell B ($1000)
        // Short Spread = Sell A ($1000) + Buy B ($1000)
        
        match signal {
            Signal::Buy if position == 0 => {
                // Enter Long Spread
                position = 1;
                entry_equity = capital;
                capital -= capital * TAKER_FEE * 2.0; // Fee on both legs
            },
            Signal::Sell if position == 0 => {
                // Enter Short Spread
                position = -1;
                entry_equity = capital;
                capital -= capital * TAKER_FEE * 2.0;
            },
            Signal::Sell if position == 1 => {
                // Close Long Spread (Mean Reverted)
                let pnl_a = (prices_a[i] / prices_a[i-1]) - 1.0; // Simplified % return tracking needed here
                // For speed, let's just assume we close at current prices
                // In a real backtest, you track quantity.
                // Reverting to simplified "Trade Counter" logic for brevity in this example.
                position = 0;
                trades += 1;
                capital -= capital * TAKER_FEE * 2.0; 
                // Add Profit (Simulated for this snippet)
                // Need precise Quantity tracking for real PnL
            },
            Signal::Buy if position == -1 => {
                // Close Short Spread
                position = 0;
                trades += 1;
                capital -= capital * TAKER_FEE * 2.0;
            },
            _ => {}
        }
        
        // PnL Tracking (Mark-to-Market would go here)
    }

    // Mocking PnL for the example runner since full accounting requires the `Position` struct logic
    // In your real implementation, copy the logic from `DualLegStrategy::process_tick`
    let simulated_profit = (trades as f64) * 5.0 - (trades as f64 * 2.0); 

    BacktestResult {
        window,
        z_entry,
        z_exit,
        net_profit: simulated_profit, // Placeholder
        sharpe_ratio: 1.5, // Placeholder
        trades,
    }
}

// --- Helper to generate dummy data ---
fn load_mock_data() -> (Vec<i64>, Vec<f64>, Vec<f64>) {
    let mut dates = Vec::new();
    let mut a = Vec::new();
    let mut b = Vec::new();
    
    // Generate correlated random walk
    let mut price_a = 100.0;
    let mut price_b = 10.0;
    
    use rand::prelude::*;
    let mut rng = rand::thread_rng();

    for i in 0..1000 {
        dates.push(i);
        
        let move_common = rng.gen_range(-0.01..0.01);
        let move_a = rng.gen_range(-0.002..0.002);
        let move_b = rng.gen_range(-0.002..0.002);
        
        price_a *= 1.0 + move_common + move_a;
        price_b *= 1.0 + move_common + move_b;
        
        a.push(price_a);
        b.push(price_b);
    }
    
    (dates, a, b)
}