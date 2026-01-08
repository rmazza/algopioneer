//! Dual-leg trading command handler.
//!
//! Implements the `dual-leg` subcommand for running Basis or Pairs
//! trading strategies with dual-leg execution.

use crate::cli::DualLegCliConfig;
use crate::coinbase::websocket::CoinbaseWebsocket;
use crate::exchange::coinbase::{AppEnv, CoinbaseClient};
use crate::logging::{CsvRecorder, TradeRecorder};
use crate::strategy::dual_leg_trading::{
    BasisManager, DualLegConfig, DualLegStrategy, ExecutionEngine, HedgeMode, InstrumentType,
    PairsManager, RecoveryWorker, RiskMonitor, SystemClock, TransactionCostModel,
};

use rust_decimal::prelude::*;
use rust_decimal_macros::dec;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{error, info, warn};

/// Run the dual-leg trading strategy (Basis or Pairs).
///
/// # Arguments
/// * `strategy_type` - "basis" or "pairs"
/// * `leg1_id` - First leg symbol
/// * `leg2_id` - Second leg symbol
/// * `env` - Trading environment (Live or Paper)
/// * `exchange_id` - Exchange to use
/// * `cli_config` - CLI configuration
///
/// # Errors
/// Returns error if strategy initialization or execution fails.
pub async fn run_dual_leg_trading(
    strategy_type: &str,
    leg1_id: &str,
    leg2_id: &str,
    env: AppEnv,
    exchange_id: crate::exchange::ExchangeId,
    cli_config: DualLegCliConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    info!(
        "--- AlgoPioneer: Initializing Dual-Leg Strategy ({}) on {} ---",
        strategy_type, exchange_id
    );

    // Initialize exchange client using factory
    let _exchange_config = crate::exchange::ExchangeConfig::from_env(exchange_id)?;

    // For now, we still use CoinbaseClient for the execution engine since strategies
    // depend on the Executor trait from dual_leg_trading module.
    // The abstraction allows switching once Kraken is fully implemented.
    let paper = matches!(env, AppEnv::Paper);
    if exchange_id == crate::exchange::ExchangeId::Kraken {
        warn!("Kraken exchange selected but not fully implemented yet. Falling back to Coinbase for execution.");
    }

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

    let client = Arc::new(CoinbaseClient::new(env, recorder)?);

    // CF1 FIX: Create bounded Recovery Channel (capacity 20) to apply backpressure
    // This prevents unbounded queuing and ensures recovery tasks are never dropped
    let (recovery_tx, recovery_rx) = tokio::sync::mpsc::channel(20);
    // Create Feedback Channel for Recovery Worker -> Strategy
    let (feedback_tx, feedback_rx) = tokio::sync::mpsc::channel(100);

    // Spawn Recovery Worker
    let recovery_worker = RecoveryWorker::new(client.clone(), recovery_rx, feedback_tx);
    tokio::spawn(async move {
        recovery_worker.run().await;
    });

    let execution_engine = ExecutionEngine::new(client.clone(), recovery_tx, 5, 60);

    // Dependency Injection based on Strategy Type
    let (entry_strategy, risk_monitor) = match strategy_type {
        "basis" => {
            // Initialize Cost Model (e.g., 10 bps maker, 20 bps taker, 5 bps slippage)
            let cost_model = TransactionCostModel::default();
            let manager = Box::new(BasisManager::new(dec!(10.0), dec!(2.0), cost_model)); // 10 bps entry, 2 bps exit
            let monitor =
                RiskMonitor::new(dec!(3.0), InstrumentType::Linear, HedgeMode::DeltaNeutral);
            (
                manager as Box<dyn crate::strategy::dual_leg_trading::EntryStrategy>,
                monitor,
            )
        }
        "pairs" => {
            // Pairs Trading: Z-Score based
            // Window 20, Entry Z=2.0, Exit Z=0.1
            let manager = Box::new(PairsManager::new(500, 4.0, 0.1));
            // Pairs is usually Dollar Neutral
            let monitor =
                RiskMonitor::new(dec!(3.0), InstrumentType::Linear, HedgeMode::DollarNeutral);
            (
                manager as Box<dyn crate::strategy::dual_leg_trading::EntryStrategy>,
                monitor,
            )
        }
        _ => {
            error!(
                "Error: Unknown strategy type '{}'. Use 'basis' or 'pairs'.",
                strategy_type
            );
            return Ok(());
        }
    };

    let order_size = match Decimal::from_f64(cli_config.order_size) {
        Some(d) => d,
        None => {
            warn!(
                "Invalid order_size '{}', defaulting to 0.00001",
                cli_config.order_size
            );
            dec!(0.00001)
        }
    };
    let min_profit = match Decimal::from_f64(cli_config.min_profit_threshold) {
        Some(d) => d,
        None => {
            warn!(
                "Invalid min_profit_threshold '{}', defaulting to 0.005",
                cli_config.min_profit_threshold
            );
            dec!(0.005)
        }
    };
    let stop_loss = match Decimal::from_f64(cli_config.stop_loss_threshold) {
        Some(d) => d,
        None => {
            warn!(
                "Invalid stop_loss_threshold '{}', defaulting to -0.05",
                cli_config.stop_loss_threshold
            );
            dec!(-0.05)
        }
    };

    let config = DualLegConfig {
        spot_symbol: leg1_id.to_string(),
        future_symbol: leg2_id.to_string(),
        order_size,
        max_tick_age_ms: cli_config.max_tick_age_ms,
        execution_timeout_ms: cli_config.execution_timeout_ms,
        min_profit_threshold: min_profit,
        stop_loss_threshold: stop_loss,
        fee_tier: TransactionCostModel::default(),
        throttle_interval_secs: cli_config.throttle_interval_secs,
    };

    let mut strategy = DualLegStrategy::new(
        entry_strategy,
        risk_monitor,
        execution_engine,
        config,
        feedback_rx,
        Box::new(SystemClock),
    );

    // Create channels for market data
    let (leg1_tx, leg1_rx) = tokio::sync::mpsc::channel(100);
    let (leg2_tx, leg2_rx) = tokio::sync::mpsc::channel(100);

    // WebSocket Integration
    let ws_client = CoinbaseWebsocket::new()?;
    let products = vec![leg1_id.to_string(), leg2_id.to_string()];
    let (ws_tx, mut ws_rx) = tokio::sync::mpsc::channel(100);

    // Spawn WebSocket Client
    tokio::spawn(async move {
        if let Err(e) = ws_client.connect_and_subscribe(products, ws_tx).await {
            tracing::error!("WebSocket connection failed: {}", e);
        }
    });

    // Demultiplexer: Route WS messages to appropriate strategy channels
    let leg1_id_clone = leg1_id.to_string();
    let leg2_id_clone = leg2_id.to_string();

    tokio::spawn(async move {
        while let Some(data) = ws_rx.recv().await {
            tracing::debug!("Demux received: {} at {}", data.symbol, data.price);
            let arc_data = Arc::new(data);
            // P-4 FIX: Use clone only when we need to continue using arc_data
            // In if-else chain, only one branch executes, so second clone was unnecessary
            if arc_data.symbol == leg1_id_clone {
                if leg1_tx.send(arc_data).await.is_err() {
                    break;
                }
            } else if arc_data.symbol == leg2_id_clone {
                if leg2_tx.send(arc_data).await.is_err() {
                    break;
                }
            } else {
                tracing::warn!("Demux received unknown symbol: {}", arc_data.symbol);
            }
        }
    });

    // Run strategy
    strategy.run(leg1_rx, leg2_rx).await;

    Ok(())
}
