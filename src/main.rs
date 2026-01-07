//! AlgoPioneer - Algorithmic Trading System
//!
//! Entry point for CLI commands. This file handles:
//! - Global allocator configuration (jemalloc)
//! - CLI parsing
//! - Logging initialization
//! - Command dispatch to handlers

// --- Global Allocator (Jemalloc for reduced fragmentation in long-running async apps) ---
#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use algopioneer::cli::{Cli, Commands, DualLegCliConfig};
use algopioneer::commands;
use algopioneer::exchange::coinbase::AppEnv;
use clap::Parser;
use dotenvy::dotenv;
use tracing::error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from the .env file
    dotenv().ok();

    let cli = Cli::parse();

    // Initialize Logger
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(&cli.verbose))
        .init();

    match cli.command {
        Commands::Trade {
            product_id,
            duration,
            paper,
            order_size,
            short_window,
            long_window,
            max_history,
        } => {
            commands::run_trade(
                product_id,
                duration,
                paper,
                order_size,
                short_window,
                long_window,
                max_history,
            )
            .await?;
        }
        Commands::Backtest => {
            commands::run_backtest()?;
        }
        Commands::DualLeg {
            strategy,
            symbols,
            exchange,
            paper,
            order_size,
            max_tick_age_ms,
            execution_timeout_ms,
            min_profit_threshold,
            stop_loss_threshold,
            throttle_interval_secs,
        } => {
            // Parse exchange ID
            let exchange_id: algopioneer::exchange::ExchangeId =
                exchange.parse().map_err(|e: String| {
                    error!("{}", e);
                    std::io::Error::other(e)
                })?;

            let env = if paper { AppEnv::Paper } else { AppEnv::Live };
            let parts: Vec<&str> = symbols.split(',').collect();
            if parts.len() != 2 {
                error!("Error: --symbols must contain exactly two symbols separated by a comma (e.g., BTC-USD,BTC-USDT)");
                return Ok(());
            }

            let dual_leg_config = DualLegCliConfig {
                order_size,
                max_tick_age_ms,
                execution_timeout_ms,
                min_profit_threshold,
                stop_loss_threshold,
                throttle_interval_secs,
            };

            commands::run_dual_leg_trading(
                &strategy,
                parts[0],
                parts[1],
                env,
                exchange_id,
                dual_leg_config,
            )
            .await?;
        }
        Commands::Portfolio {
            config,
            exchange,
            paper,
        } => {
            commands::run_portfolio(&config, &exchange, paper).await?;
        }
        Commands::DiscoverPairs {
            exchange,
            symbols,
            min_correlation,
            max_half_life,
            min_sharpe,
            lookback_days,
            max_pairs,
            output,
            initial_capital,
            no_cointegration,
        } => {
            commands::run_discover_pairs(
                &exchange,
                &symbols,
                min_correlation,
                max_half_life,
                min_sharpe,
                lookback_days,
                max_pairs,
                &output,
                initial_capital,
                no_cointegration,
            )
            .await?;
        }
    }

    Ok(())
}
