//! Pair discovery command handler.
//!
//! Implements the `discover-pairs` subcommand for automatically finding
//! and optimizing cointegrated trading pairs.

use crate::discovery::config::{
    PortfolioPairConfig, DEFAULT_CANDIDATES, DEFAULT_EQUITY_CANDIDATES,
};
use crate::discovery::{discover_and_optimize, DiscoveryConfig};
use crate::exchange::alpaca::AlpacaClient;
use crate::exchange::coinbase::{AppEnv, CoinbaseClient};
use crate::strategy::dual_leg_trading::SystemClock;

use rust_decimal::prelude::*;
use rust_decimal_macros::dec;
use tracing::{error, info, warn};

/// Run the pair discovery and optimization pipeline.
///
/// # Arguments
/// * `exchange_arg` - Exchange: "coinbase" or "alpaca"
/// * `symbols_arg` - Comma-separated symbols or "default"
/// * `min_correlation` - Minimum Pearson correlation threshold
/// * `max_half_life` - Maximum half-life in hours
/// * `min_sharpe` - Minimum Sharpe ratio
/// * `lookback_days` - Historical lookback period in days
/// * `max_pairs` - Maximum number of pairs to output
/// * `output_path` - Output file path for JSON
/// * `initial_capital` - Initial capital for backtests
/// * `no_cointegration` - Whether to skip ADF test
///
/// # Errors
/// Returns error if discovery pipeline fails.
#[allow(clippy::too_many_arguments)]
pub async fn run_discover_pairs(
    exchange_arg: &str,
    symbols_arg: &str,
    min_correlation: f64,
    max_half_life: f64,
    min_sharpe: f64,
    lookback_days: u32,
    max_pairs: usize,
    output_path: &str,
    initial_capital: f64,
    no_cointegration: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let is_alpaca = exchange_arg.to_lowercase() == "alpaca";

    if is_alpaca {
        info!("--- AlgoPioneer: Pair Discovery Pipeline (Alpaca - Equities) ---");
    } else {
        info!("--- AlgoPioneer: Pair Discovery Pipeline (Coinbase - Crypto) ---");
    }

    // Parse symbols - use appropriate defaults based on exchange
    let candidates: Vec<String> = if symbols_arg == "default" {
        if is_alpaca {
            info!("Using default equity candidates (40+ stocks)");
            DEFAULT_EQUITY_CANDIDATES
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            info!("Using default crypto candidates (50+ crypto pairs)");
            DEFAULT_CANDIDATES.iter().map(|s| s.to_string()).collect()
        }
    } else {
        symbols_arg
            .split(',')
            .map(|s| s.trim().to_string())
            .collect()
    };

    info!(
        exchange = exchange_arg,
        candidates = candidates.len(),
        min_corr = min_correlation,
        max_hl = max_half_life,
        min_sharpe = min_sharpe,
        lookback = lookback_days,
        require_cointegration = !no_cointegration,
        "Configuration loaded"
    );

    // Build discovery config
    // CB-1/CB-2 FIX: Alpaca uses daily bars, so thresholds need adjustment:
    // - half_life is in SAMPLES not hours, so 48.0 = 48 days with daily data
    // - min_trades must be realistic for daily data (63 samples with 90-day lookback)
    let (adjusted_max_half_life, adjusted_min_trades, adjusted_train_ratio) = if is_alpaca {
        // Alpaca: daily bars
        // max_half_life = 30 samples (30 days) is reasonable for daily mean reversion
        // min_trades = 3 is achievable with 63 daily samples
        // train_ratio = 0.8 to maximize training data
        (30.0, 3u32, 0.8)
    } else {
        // Coinbase: hourly bars (default)
        (max_half_life, 25, 0.7)
    };

    let config = DiscoveryConfig {
        candidates,
        min_correlation,
        max_half_life_hours: adjusted_max_half_life,
        min_sharpe_ratio: min_sharpe,
        lookback_days,
        max_pairs_output: max_pairs,
        initial_capital: Decimal::from_f64_retain(initial_capital).unwrap_or(dec!(10000)),
        require_cointegration: !no_cointegration,
        min_trades: adjusted_min_trades,
        train_ratio: adjusted_train_ratio,
        ..Default::default()
    };

    // Run discovery pipeline with appropriate client
    info!("Starting discovery and optimization...");
    let clock = SystemClock;

    let results = if is_alpaca {
        // Alpaca path - use AlpacaClient for equities
        let mut client = AlpacaClient::new(AppEnv::Live, None)?;
        discover_and_optimize(&mut client, &config, &clock).await
    } else {
        // Coinbase path - use CoinbaseClient for crypto
        let mut client = CoinbaseClient::new(AppEnv::Live, None)?;
        discover_and_optimize(&mut client, &config, &clock).await
    };

    let results = match results {
        Ok(pairs) => pairs,
        Err(e) => {
            error!("Discovery failed: {}", e);
            return Err(e.into());
        }
    };

    if results.is_empty() {
        warn!("No pairs found matching criteria");
        return Ok(());
    }

    // Display results
    info!("\n=== DISCOVERED PAIRS (with walk-forward validation) ===");
    println!(
        "\n{:<20} | {:>6} | {:>7} | {:>8} | {:>8} | {:>6} | {:>10}",
        "Pair", "Window", "Z-Entry", "Train-SR", "Val-SR", "Trades", "Net Profit"
    );
    println!("{}", "-".repeat(85));

    for pair in &results {
        println!(
            "{:<20} | {:>6} | {:>7.1} | {:>8.2} | {:>8.2} | {:>6} | ${:>9.2}",
            format!("{}/{}", pair.leg1, pair.leg2),
            pair.window,
            pair.z_entry,
            pair.sharpe_ratio,
            pair.validation_sharpe,
            pair.trades,
            pair.net_profit
        );

        // Warn about large gap between train and validation Sharpe (overfitting indicator)
        let sharpe_gap = pair.sharpe_ratio - pair.validation_sharpe;
        if sharpe_gap > 2.0 {
            warn!(
                pair = format!("{}/{}", pair.leg1, pair.leg2),
                train_sharpe = format!("{:.2}", pair.sharpe_ratio),
                validation_sharpe = format!("{:.2}", pair.validation_sharpe),
                "Large train/validation Sharpe gap ({:.1}) - possible overfitting",
                sharpe_gap
            );
        }
    }

    // Summary: use validation Sharpe threshold instead of unrealistic 10.0
    if results.iter().any(|p| p.sharpe_ratio > 4.0) {
        warn!("Some pairs have train Sharpe > 4.0. Check validation Sharpe for true performance.");
        warn!("A large gap between train and validation Sharpe indicates overfitting.");
    }

    // Calculate allocation per pair (Equal Weight)
    // Allocation = Initial Capital / Max Pairs (to ensure safety even if fewer pairs are found)
    let initial_capital_dec = Decimal::from_f64_retain(initial_capital).unwrap_or(dec!(10000));
    let allocation = initial_capital_dec / Decimal::from(max_pairs);

    info!(
        capital = %initial_capital_dec,
        max_pairs = max_pairs,
        per_pair = %allocation,
        "Calculated portfolio allocation"
    );

    // Convert to PortfolioPairConfig format
    let portfolio_configs: Vec<PortfolioPairConfig> = results
        .iter()
        .map(|p| p.to_portfolio_config(allocation))
        .collect();

    // Write output file
    let json = serde_json::to_string_pretty(&portfolio_configs)?;
    std::fs::write(output_path, &json)?;

    info!(
        output = output_path,
        pairs = results.len(),
        "Configuration saved"
    );

    println!("\n✓ Saved {} pairs to {}", results.len(), output_path);
    println!(
        "  Run with: cargo run -- portfolio --config {}",
        output_path
    );

    Ok(())
}
