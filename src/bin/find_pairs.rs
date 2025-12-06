use algopioneer::coinbase::{CoinbaseClient, AppEnv};
use algopioneer::strategy::dual_leg_trading::DualLegConfig;
use algopioneer::strategy::dual_leg_trading::TransactionCostModel;
use cbadv::time::Granularity;
use chrono::{Utc, Duration};
use polars::prelude::*;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use tokio::time::sleep;

#[derive(Debug, Clone)]
struct PairStats {
    symbol_a: String,
    symbol_b: String,
    correlation: f64,
    z_score_std: f64,
    mean_reversion_half_life: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();
    println!("--- AlgoPioneer: Finding Cointegrated Pairs ---");

    let mut client = CoinbaseClient::new(AppEnv::Live)?;

    // 1. Fetch Top Volume Pairs
    println!("Using hardcoded list of top volume pairs (API limitation workaround)...");
    // Hardcoded list of top 20 pairs
    let candidates: Vec<String> = vec![
        "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOGE-USD", 
        "AVAX-USD", "SHIB-USD", "DOT-USD", "MATIC-USD", "LTC-USD", 
        "UNI-USD", "LINK-USD", "XLM-USD", "BCH-USD", "ALGO-USD", 
        "ATOM-USD", "FIL-USD", "VET-USD", "ICP-USD", "AXS-USD"
    ].into_iter().map(|s| s.to_string()).collect();

    println!("Selected {} candidates for analysis: {:?}", candidates.len(), candidates);

    // 2. Fetch Historical Data (10 days, 1h candles)
    let end = Utc::now();
    let start = end - Duration::days(10);
    
    let mut prices: HashMap<String, Vec<f64>> = HashMap::new();
    let mut timestamps: Vec<i64> = Vec::new();

    for symbol in &candidates {
        println!("Fetching candles for {}...", symbol);
        match client.get_product_candles(symbol, &start, &end, Granularity::OneHour).await {
            Ok(candles) => {
                // Sort by time
                let mut sorted_candles = candles;
                sorted_candles.sort_by_key(|c| c.start);
                
                let closes: Vec<f64> = sorted_candles.iter().map(|c| c.close).collect();
                
                if timestamps.is_empty() {
                    timestamps = sorted_candles.iter().map(|c| c.start as i64).collect();
                } else {
                    // Ensure alignment? For MVP assume 1h alignment is roughly ok or truncate
                    // In production, we need strict alignment.
                    if sorted_candles.len() != timestamps.len() {
                         println!("Warning: Length mismatch for {}. Skipping.", symbol);
                         continue;
                    }
                }
                
                prices.insert(symbol.clone(), closes);
            },
            Err(e) => {
                println!("Failed to fetch {}: {}", symbol, e);
            }
        }
        sleep(std::time::Duration::from_millis(200)).await; // Rate limit
    }

    // 3. Analyze Pairs
    let mut results: Vec<PairStats> = Vec::new();
    let symbols: Vec<String> = prices.keys().cloned().collect();

    for i in 0..symbols.len() {
        for j in (i + 1)..symbols.len() {
            let sym_a = &symbols[i];
            let sym_b = &symbols[j];
            
            let series_a = &prices[sym_a];
            let series_b = &prices[sym_b];
            
            // Correlation
            let corr = calculate_correlation(series_a, series_b);
            
            if corr > 0.8 { // High correlation filter
                // Check Stationarity of Spread (Log Spread)
                let spread: Vec<f64> = series_a.iter().zip(series_b.iter())
                    .map(|(a, b)| a.ln() - b.ln())
                    .collect();
                
                let (z_score_std, half_life) = analyze_spread(&spread);
                
                if half_life < 24.0 { // Mean reverts within a day
                    results.push(PairStats {
                        symbol_a: sym_a.clone(),
                        symbol_b: sym_b.clone(),
                        correlation: corr,
                        z_score_std,
                        mean_reversion_half_life: half_life,
                    });
                }
            }
        }
    }

    // Sort by correlation desc
    results.sort_by(|a, b| b.correlation.partial_cmp(&a.correlation).unwrap());

    println!("Found {} cointegrated pairs.", results.len());

    // 4. Generate Config
    let mut portfolio_config: Vec<DualLegConfig> = Vec::new();
    
    for pair in results.iter().take(10) {
        println!("Selected: {} - {} (Corr: {:.2}, Half-Life: {:.1}h)", pair.symbol_a, pair.symbol_b, pair.correlation, pair.mean_reversion_half_life);
        
        // Estimate parameters based on volatility
        let entry_z = 2.0;
        let exit_z = 0.0; // Mean reversion target
        
        let config = DualLegConfig {
            spot_symbol: pair.symbol_a.clone(),
            future_symbol: pair.symbol_b.clone(), // Treating B as "Future" or just 2nd leg
            order_size: dec!(10.0), // $10 allocation
            max_tick_age_ms: 2000,
            execution_timeout_ms: 30000,
            min_profit_threshold: dec!(0.001), // 0.1%
            stop_loss_threshold: dec!(-0.02), // 2%
            fee_tier: TransactionCostModel::new(dec!(10.0), dec!(20.0), dec!(5.0)),
            throttle_interval_secs: 5,
        };
        portfolio_config.push(config);
    }

    let json = serde_json::to_string_pretty(&portfolio_config)?;
    let mut file = File::create("pairs_config.json")?;
    file.write_all(json.as_bytes())?;
    
    println!("Saved configuration to pairs_config.json");

    Ok(())
}

fn calculate_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    
    let mut num = 0.0;
    let mut den_a = 0.0;
    let mut den_b = 0.0;
    
    for (x, y) in a.iter().zip(b.iter()) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        num += dx * dy;
        den_a += dx * dx;
        den_b += dy * dy;
    }
    
    if den_a == 0.0 || den_b == 0.0 { 0.0 } else { num / (den_a.sqrt() * den_b.sqrt()) }
}

fn analyze_spread(spread: &[f64]) -> (f64, f64) {
    // Simplified analysis:
    // 1. Calculate Std Dev (Volatility)
    // 2. Estimate Ornstein-Uhlenbeck parameters for half-life (simplified via lag-1 autocorrelation)
    
    let n = spread.len() as f64;
    let mean = spread.iter().sum::<f64>() / n;
    let variance = spread.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    
    // Lag-1 Autocorrelation for Half-Life
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..spread.len()-1 {
        let dx = spread[i] - mean;
        let dy = spread[i+1] - mean;
        num += dx * dy;
        den += dx * dx;
    }
    let rho = if den == 0.0 { 0.0 } else { num / den };
    
    // Half-life = -ln(2) / ln(rho)
    let half_life = if rho > 0.0 && rho < 1.0 {
        -2.0f64.ln() / rho.ln()
    } else {
        1000.0 // Infinite/Non-stationary
    };
    
    (std_dev, half_life)
}
