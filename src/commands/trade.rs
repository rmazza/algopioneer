//! Trade command handler.
//!
//! Implements the `trade` subcommand for running the Moving Average
//! Crossover strategy on Coinbase.

use crate::cli::SimpleTradingConfig;
use crate::exchange::coinbase::AppEnv;
use crate::logging::{CsvRecorder, TradeRecorder};
use crate::state::TradeState;
use crate::trading::SimpleTradingEngine;

use rust_decimal::prelude::*;
use rust_decimal_macros::dec;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{error, warn};

/// Run the simple Moving Average Crossover trading strategy.
///
/// # Arguments
/// * `product_id` - Trading product (e.g., "BTC-USD")
/// * `duration` - Seconds between trade cycles
/// * `paper` - Whether to run in paper trading mode
/// * `order_size` - Order size in base currency
/// * `short_window` - Short MA window
/// * `long_window` - Long MA window
/// * `max_history` - Maximum price history to keep
///
/// # Errors
/// Returns error if trading engine fails to initialize or run.
pub async fn run_trade(
    product_id: String,
    duration: u64,
    paper: bool,
    order_size: f64,
    short_window: usize,
    long_window: usize,
    max_history: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let env = if paper { AppEnv::Paper } else { AppEnv::Live };
    let config = SimpleTradingConfig {
        product_id,
        duration,
        order_size: match Decimal::from_f64(order_size) {
            Some(d) => d,
            None => {
                warn!("Invalid order_size '{}', defaulting to 0.001", order_size);
                dec!(0.001)
            }
        },
        short_window,
        long_window,
        max_history,
        env,
    };

    // Create async state persistence channel
    let (state_tx, mut state_rx) = tokio::sync::mpsc::unbounded_channel::<TradeState>();

    // Spawn background task for async state persistence
    tokio::spawn(async move {
        while let Some(state) = state_rx.recv().await {
            let _ = tokio::task::spawn_blocking(move || {
                if let Err(e) = state.save() {
                    tracing::error!("Background state save failed: {}", e);
                }
            })
            .await;
        }
    });

    // Create paper logger if in paper mode
    let recorder: Option<Arc<dyn TradeRecorder>> = if paper {
        match CsvRecorder::new(PathBuf::from("paper_trades.csv")) {
            Ok(recorder) => Some(Arc::new(recorder)),
            Err(e) => {
                error!("Failed to initialize CSV recorder: {}", e);
                None
            }
        }
    } else {
        None
    };

    let mut engine = SimpleTradingEngine::new(config, state_tx, recorder).await?;
    engine.run().await?;

    Ok(())
}
