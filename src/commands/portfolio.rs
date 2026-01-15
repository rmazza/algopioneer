//! Portfolio command handler.
//!
//! Implements the `portfolio` subcommand for running multiple pairs
//! with the strategy supervisor.

use crate::discovery::config::PortfolioPairConfig;
use crate::exchange::coinbase::{AppEnv, CoinbaseClient, CoinbaseWebSocketProvider};
use crate::exchange::ExchangeId;
use crate::logging::{CsvRecorder, TradeRecorder};
use crate::strategy::dual_leg_trading::{
    DualLegLiveConfig, DualLegStrategyLive, DualLegStrategyType,
};
use crate::strategy::supervisor::StrategySupervisor;

use rust_decimal::prelude::ToPrimitive;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{error, info, warn};

/// Run the portfolio supervisor with multiple pairs.
///
/// # Arguments
/// * `config_path` - Path to pairs configuration JSON file
/// * `exchange` - Exchange to use ("coinbase" or "alpaca")
/// * `paper` - Whether to run in paper trading mode
///
/// # Errors
/// Returns error if configuration loading or supervisor fails.
pub async fn run_portfolio(
    config_path: &str,
    exchange: &str,
    paper: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let env = if paper { AppEnv::Paper } else { AppEnv::Live };

    // Parse exchange
    let exchange_id: ExchangeId = exchange.parse().unwrap_or_else(|e| {
        warn!(
            "Invalid exchange '{}': {}. Defaulting to Coinbase.",
            exchange, e
        );
        ExchangeId::Coinbase
    });

    info!("--- AlgoPioneer: Portfolio Supervisor Mode ---");
    info!("Exchange: {}, Paper: {}", exchange_id, paper);
    info!("Loading configuration from: {}", config_path);

    // Load Config
    let file = File::open(config_path)?;
    let reader = BufReader::new(file);
    let config_list: Vec<PortfolioPairConfig> = serde_json::from_reader(reader)?;

    if config_list.is_empty() {
        error!("No pairs found in configuration.");
        return Ok(());
    }

    // Handle exchange-specific initialization
    match exchange_id {
        ExchangeId::Alpaca => {
            // Alpaca path - use AlpacaClient and AlpacaWebSocketProvider
            info!("Initializing Alpaca exchange for Portfolio mode");

            use crate::exchange::alpaca::{AlpacaClient, AlpacaWebSocketProvider};

            // Create paper logger if in paper mode
            let recorder: Option<Arc<dyn TradeRecorder>> = if paper {
                match CsvRecorder::new(PathBuf::from("paper_trades_alpaca.csv")) {
                    Ok(rec) => Some(Arc::new(rec)),
                    Err(e) => {
                        error!("Failed to initialize CSV recorder: {}", e);
                        None
                    }
                }
            } else {
                None
            };

            let alpaca_client = Arc::new(AlpacaClient::new(env, recorder)?);

            let risk_config = if paper {
                crate::risk::DailyRiskConfig::paper_trading()
            } else {
                crate::risk::DailyRiskConfig::default()
            };
            let risk_engine = Arc::new(crate::risk::DailyRiskEngine::new(risk_config.clone()));
            let alpaca_client = Arc::new(crate::risk::RiskManagedExecutor::new(
                alpaca_client,
                risk_engine,
            ));

            let ws_client = Box::new(AlpacaWebSocketProvider::from_env()?);

            // Initialize Supervisor
            let mut supervisor = StrategySupervisor::new().with_risk_config(risk_config);

            for (idx, json_config) in config_list.into_iter().enumerate() {
                let pair_id = format!(
                    "{}-{}",
                    json_config.dual_leg_config.spot_symbol,
                    json_config.dual_leg_config.future_symbol
                );

                let live_config = DualLegLiveConfig {
                    dual_leg_config: json_config.dual_leg_config,
                    window_size: json_config.window_size,
                    entry_z_score: json_config.entry_z_score.to_f64().unwrap_or(2.0),
                    exit_z_score: json_config.exit_z_score.to_f64().unwrap_or(0.1),
                    strategy_type: DualLegStrategyType::Pairs,
                    // N-1 FIX: Use default values for new config fields
                    circuit_breaker_threshold: 5,
                    circuit_breaker_timeout_secs: 60,
                    basis_entry_bps: rust_decimal_macros::dec!(10.0),
                    basis_exit_bps: rust_decimal_macros::dec!(2.0),
                    max_leverage: rust_decimal_macros::dec!(3.0),
                    drift_recalc_interval: 10_000,
                };

                let strategy =
                    DualLegStrategyLive::new(pair_id.clone(), live_config, alpaca_client.clone());

                supervisor.add_strategy(Box::new(strategy));
                info!("Added Alpaca strategy #{} ({})", idx + 1, pair_id);
            }

            // Run Supervisor (blocks until completion)
            if let Err(e) = supervisor.run(ws_client).await {
                error!("Alpaca supervisor terminated with error: {}", e);
            }
            return Ok(());
        }
        ExchangeId::Kraken => {
            warn!("Kraken exchange not fully implemented. Defaulting to Coinbase.");
        }
        ExchangeId::Coinbase => {
            // Default path - continue with Coinbase
        }
    }

    // Initialize Shared Resources (Coinbase path)
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
    let risk_config = if paper {
        crate::risk::DailyRiskConfig::paper_trading()
    } else {
        crate::risk::DailyRiskConfig::default()
    };
    let risk_engine = Arc::new(crate::risk::DailyRiskEngine::new(risk_config.clone()));
    let client = Arc::new(crate::risk::RiskManagedExecutor::new(client, risk_engine));

    // Use CoinbaseWebSocketProvider which implements WebSocketProvider trait
    let ws_client = Box::new(CoinbaseWebSocketProvider::from_env()?);

    // Initialize Supervisor with risk config
    let mut supervisor = StrategySupervisor::new().with_risk_config(risk_config);

    for (idx, json_config) in config_list.into_iter().enumerate() {
        let pair_id = format!(
            "{}-{}",
            json_config.dual_leg_config.spot_symbol, json_config.dual_leg_config.future_symbol
        );

        // Convert to Live Config
        // CB-2 FIX: Convert Decimal z-scores from config to f64 for internal use
        let live_config = DualLegLiveConfig {
            dual_leg_config: json_config.dual_leg_config,
            window_size: json_config.window_size,
            entry_z_score: json_config.entry_z_score.to_f64().unwrap_or(2.0),
            exit_z_score: json_config.exit_z_score.to_f64().unwrap_or(0.1),
            strategy_type: DualLegStrategyType::Pairs, // Legacy portfolio only supported Pairs
            // N-1 FIX: Use default values for new config fields
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout_secs: 60,
            basis_entry_bps: rust_decimal_macros::dec!(10.0),
            basis_exit_bps: rust_decimal_macros::dec!(2.0),
            max_leverage: rust_decimal_macros::dec!(3.0),
            drift_recalc_interval: 10_000,
        };

        // Create Strategy
        // StrategySupervisor requires Box<dyn LiveStrategy>
        // DualLegStrategyLive<CoinbaseClient> implements LiveStrategy
        let strategy = DualLegStrategyLive::new(pair_id.clone(), live_config, client.clone());

        supervisor.add_strategy(Box::new(strategy));
        info!("Added strategy #{} ({})", idx + 1, pair_id);
    }

    // Run Supervisor (blocks until completion)
    if let Err(e) = supervisor.run(ws_client).await {
        error!("Supervisor terminated with error: {}", e);
    }

    Ok(())
}
