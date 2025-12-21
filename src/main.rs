// --- Global Allocator (Jemalloc for reduced fragmentation in long-running async apps) ---
#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use algopioneer::coinbase::websocket::CoinbaseWebsocket;
use algopioneer::discovery::{discover_and_optimize, DiscoveryConfig};
use algopioneer::exchange::coinbase::{AppEnv, CoinbaseClient};
use algopioneer::strategy::dual_leg_trading::SystemClock;
use algopioneer::strategy::dual_leg_trading::{
    BasisManager, DualLegConfig, DualLegStrategy, ExecutionEngine, HedgeMode, InstrumentType,
    PairsManager, RecoveryWorker, RiskMonitor, TransactionCostModel,
};
use algopioneer::strategy::moving_average::MovingAverageCrossover;
use algopioneer::strategy::Signal;
use cbadv::time::Granularity;
use chrono::{Duration as ChronoDuration, Utc};
use clap::Parser;
use dotenvy::dotenv;
use polars::prelude::*;
use rust_decimal::prelude::FromPrimitive;
use rust_decimal::prelude::*;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use tokio::time::Duration;
use tracing::{error, info, warn};

// Paper trade logging
use algopioneer::logging::{CsvRecorder, TradeRecorder};
use std::path::PathBuf;

// --- Constants ---
const STATE_FILE: &str = "trade_state.json";

// --- Position Tracking ---
/// Detailed position information for reconciliation
#[derive(Serialize, Deserialize, Debug, Clone)]
struct PositionDetail {
    symbol: String,
    side: String,
    quantity: Decimal,
    entry_price: Decimal,
}

/// State persistence with proper position tracking (fixes Ghost Position risk)
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
struct TradeState {
    /// Key: Symbol (e.g., "BTC-USD"), Value: Position details
    positions: std::collections::HashMap<String, PositionDetail>,
}

impl TradeState {
    fn load() -> Self {
        if let Ok(data) = fs::read_to_string(STATE_FILE) {
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Self::default()
        }
    }

    fn save(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let json = serde_json::to_string_pretty(self)?;
        let temp_path = format!("{}.tmp", STATE_FILE);

        // Write to temporary file first
        let mut file = fs::File::create(&temp_path)?;
        file.write_all(json.as_bytes())?;

        // Sync data to disk before rename (fsync for durability on Linux/Unix)
        // This ensures the write is fully committed before we make it visible
        file.sync_all()?;

        // Atomic rename: POSIX guarantees rename is atomic on the same filesystem
        // If we crash here, either the old file or new file exists - never a partial file
        fs::rename(&temp_path, STATE_FILE)?;

        Ok(())
    }

    /// Check if we have an open position for a symbol
    fn has_position(&self, symbol: &str) -> bool {
        self.positions.contains_key(symbol)
    }

    /// Open a new position
    fn open_position(&mut self, detail: PositionDetail) {
        self.positions.insert(detail.symbol.clone(), detail);
    }

    /// Close a position and return its details
    fn close_position(&mut self, symbol: &str) -> Option<PositionDetail> {
        self.positions.remove(symbol)
    }
}

// --- CLI Argument Parsing ---
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Set the verbosity level (error, warn, info, debug, trace)
    #[arg(long, global = true, default_value = "info")]
    verbose: String,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Run the live trading or sandbox bot
    Trade {
        /// The product to trade (e.g., "BTC-USD")
        #[arg(short, long)]
        product_id: String,
        /// Duration in seconds to wait between trade cycles
        #[arg(short, long, default_value_t = 60)]
        duration: u64,
        /// Run in paper trading mode (simulated execution)
        #[arg(long, default_value_t = false)]
        paper: bool,
        /// Order size in base currency
        #[arg(long, default_value_t = 0.001)]
        order_size: f64,
        /// Short moving average window
        #[arg(long, default_value_t = 5)]
        short_window: usize,
        /// Long moving average window
        #[arg(long, default_value_t = 20)]
        long_window: usize,
        /// Maximum history to keep
        #[arg(long, default_value_t = 200)]
        max_history: usize,
    },
    /// Run a backtest on historical data
    Backtest,
    /// Run the Dual-Leg Trading Strategy (Basis or Pairs)
    DualLeg {
        /// Strategy type: 'basis' or 'pairs'
        #[arg(long)]
        strategy: String,
        /// Trading symbols (comma separated, e.g., "BTC-USD,BTC-USDT")
        #[arg(long)]
        symbols: String,
        /// Exchange to use: coinbase, kraken
        #[arg(long, default_value = "coinbase")]
        exchange: String,
        /// Run in paper trading mode (simulated execution)
        #[arg(long, default_value_t = false)]
        paper: bool,
        /// Order size in base currency
        #[arg(long, default_value_t = 0.00001)]
        order_size: f64,
        /// Maximum tick age in milliseconds before dropping
        #[arg(long, default_value_t = 2000)]
        max_tick_age_ms: i64,
        /// Execution timeout in milliseconds
        #[arg(long, default_value_t = 30000)]
        execution_timeout_ms: i64,
        /// Minimum profit threshold for exits
        #[arg(long, default_value_t = 0.005)]
        min_profit_threshold: f64,
        /// Stop loss threshold (negative value)
        #[arg(long, default_value_t = -0.05)]
        stop_loss_threshold: f64,
        /// Log throttle interval in seconds
        #[arg(long, default_value_t = 5)]
        throttle_interval_secs: u64,
    },
    /// Run the Portfolio Manager
    Portfolio {
        /// Path to pairs configuration file
        #[arg(long)]
        config: String,
        /// Exchange to use: coinbase, alpaca
        #[arg(long, default_value = "coinbase")]
        exchange: String,
        /// Run in paper trading mode
        #[arg(long, default_value_t = false)]
        paper: bool,
    },
    /// Discover and optimize cointegrated trading pairs automatically
    DiscoverPairs {
        /// Symbols to analyze (comma-separated, or "default" for top 20 pairs)
        #[arg(long, default_value = "default")]
        symbols: String,
        /// Minimum Pearson correlation threshold
        #[arg(long, default_value_t = 0.8)]
        min_correlation: f64,
        /// Maximum half-life in hours for mean reversion
        #[arg(long, default_value_t = 48.0)]
        max_half_life: f64,
        /// Minimum Sharpe ratio to include in results
        #[arg(long, default_value_t = 0.5)]
        min_sharpe: f64,
        /// Historical lookback period in days
        #[arg(long, default_value_t = 90)]
        lookback_days: u32,
        /// Maximum number of pairs to output
        #[arg(long, default_value_t = 10)]
        max_pairs: usize,
        /// Output file path for discovered pairs JSON
        #[arg(long, default_value = "discovered_pairs.json")]
        output: String,
        /// Initial capital for backtests in USD
        #[arg(long, default_value_t = 10000.0)]
        initial_capital: f64,
        /// Skip the ADF cointegration test (use for exploratory analysis)
        #[arg(long, default_value_t = false)]
        no_cointegration: bool,
    },
}

// --- Main Application Logic ---
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from the .env file
    dotenv().ok();

    let cli = Cli::parse();

    // Initialize Logger
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(&cli.verbose))
        .init();

    match &cli.command {
        Commands::Trade {
            product_id,
            duration,
            paper,
            order_size,
            short_window,
            long_window,
            max_history,
        } => {
            let env = if *paper { AppEnv::Paper } else { AppEnv::Live };
            let config = SimpleTradingConfig {
                product_id: product_id.clone(),
                duration: *duration,
                order_size: match Decimal::from_f64(*order_size) {
                    Some(d) => d,
                    None => {
                        warn!("Invalid order_size '{}', defaulting to 0.001", order_size);
                        dec!(0.001)
                    }
                },
                short_window: *short_window,
                long_window: *long_window,
                max_history: *max_history,
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
            let recorder: Option<Arc<dyn TradeRecorder>> = if *paper {
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
        }
        Commands::Backtest => {
            run_backtest()?;
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

            let env = if *paper { AppEnv::Paper } else { AppEnv::Live };
            let parts: Vec<&str> = symbols.split(',').collect();
            if parts.len() != 2 {
                error!("Error: --symbols must contain exactly two symbols separated by a comma (e.g., BTC-USD,BTC-USDT)");
                return Ok(());
            }
            let dual_leg_config = DualLegCliConfig {
                order_size: *order_size,
                max_tick_age_ms: *max_tick_age_ms,
                execution_timeout_ms: *execution_timeout_ms,
                min_profit_threshold: *min_profit_threshold,
                stop_loss_threshold: *stop_loss_threshold,
                throttle_interval_secs: *throttle_interval_secs,
            };
            run_dual_leg_trading(
                strategy,
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
            use algopioneer::discovery::config::PortfolioPairConfig;
            use algopioneer::exchange::coinbase::CoinbaseWebSocketProvider;
            use algopioneer::exchange::ExchangeId;
            use algopioneer::strategy::dual_leg_trading::{
                DualLegLiveConfig, DualLegStrategyLive, DualLegStrategyType,
            };
            use algopioneer::strategy::supervisor::StrategySupervisor;
            use std::fs::File;
            use std::io::BufReader;

            let env = if *paper { AppEnv::Paper } else { AppEnv::Live };

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
            info!("Loading configuration from: {}", config);

            // Load Config
            let file = File::open(config)?;
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

                    use algopioneer::exchange::alpaca::{AlpacaClient, AlpacaWebSocketProvider};

                    // Create paper logger if in paper mode
                    let recorder: Option<Arc<dyn TradeRecorder>> = if *paper {
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
                    let ws_client = Box::new(AlpacaWebSocketProvider::from_env()?);

                    // Initialize Supervisor
                    let mut supervisor = StrategySupervisor::new();

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
                        };

                        let strategy = DualLegStrategyLive::new(
                            pair_id.clone(),
                            live_config,
                            alpaca_client.clone(),
                        );

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
            let recorder: Option<Arc<dyn TradeRecorder>> = if *paper {
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
            // Use CoinbaseWebSocketProvider which implements WebSocketProvider trait
            let ws_client = Box::new(CoinbaseWebSocketProvider::from_env()?);

            // Initialize Supervisor
            let mut supervisor = StrategySupervisor::new();

            for (idx, json_config) in config_list.into_iter().enumerate() {
                let pair_id = format!(
                    "{}-{}",
                    json_config.dual_leg_config.spot_symbol,
                    json_config.dual_leg_config.future_symbol
                );

                // Convert to Live Config
                // CB-2 FIX: Convert Decimal z-scores from config to f64 for internal use
                let live_config = DualLegLiveConfig {
                    dual_leg_config: json_config.dual_leg_config,
                    window_size: json_config.window_size,
                    entry_z_score: json_config.entry_z_score.to_f64().unwrap_or(2.0),
                    exit_z_score: json_config.exit_z_score.to_f64().unwrap_or(0.1),
                    strategy_type: DualLegStrategyType::Pairs, // Legacy portfolio only supported Pairs
                };

                // Create Strategy
                // StrategySupervisor requires Box<dyn LiveStrategy>
                // DualLegStrategyLive<CoinbaseClient> implements LiveStrategy
                let strategy =
                    DualLegStrategyLive::new(pair_id.clone(), live_config, client.clone());

                supervisor.add_strategy(Box::new(strategy));
                info!("Added strategy #{} ({})", idx + 1, pair_id);
            }

            // Run Supervisor (blocks until completion)
            if let Err(e) = supervisor.run(ws_client).await {
                error!("Supervisor terminated with error: {}", e);
            }
        }
        Commands::DiscoverPairs {
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
            run_discover_pairs(
                symbols,
                *min_correlation,
                *max_half_life,
                *min_sharpe,
                *lookback_days,
                *max_pairs,
                output,
                *initial_capital,
                *no_cointegration,
            )
            .await?;
        }
    }

    Ok(())
}

struct SimpleTradingConfig {
    product_id: String,
    duration: u64,
    order_size: Decimal,
    short_window: usize,
    long_window: usize,
    max_history: usize,
    env: AppEnv,
}

struct SimpleTradingEngine {
    client: CoinbaseClient,
    strategy: MovingAverageCrossover,
    state: TradeState,
    config: SimpleTradingConfig,
    state_tx: tokio::sync::mpsc::UnboundedSender<TradeState>,
}

impl SimpleTradingEngine {
    async fn new(
        config: SimpleTradingConfig,
        state_tx: tokio::sync::mpsc::UnboundedSender<TradeState>,
        recorder: Option<Arc<dyn TradeRecorder>>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let client = CoinbaseClient::new(config.env, recorder)?;
        let strategy = MovingAverageCrossover::new(config.short_window, config.long_window);
        let state = TradeState::load();
        Ok(Self {
            client,
            strategy,
            state,
            config,
            state_tx,
        })
    }

    async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("--- AlgoPioneer: Initializing ---");

        info!("Loaded state: {:?}", self.state);

        // --- Warm up the strategy with historical data ---
        info!("Fetching historical data to warm up the strategy...");
        let end = Utc::now();
        let start = end - ChronoDuration::minutes((self.config.long_window * 5) as i64);
        let initial_candles = self
            .client
            .get_product_candles(
                &self.config.product_id,
                &start,
                &end,
                Granularity::OneMinute,
            )
            .await?;

        let mut closes: Vec<f64> = initial_candles.iter().map(|c| c.close).collect();

        // Initial truncation
        if closes.len() > self.config.max_history {
            let remove_count = closes.len() - self.config.max_history;
            closes.drain(0..remove_count);
        }

        self.strategy.warmup(&closes);

        info!("Strategy warmed up with {} data points.", closes.len());

        // --- Main Trading Loop ---
        let mut interval = tokio::time::interval(Duration::from_secs(self.config.duration));

        loop {
            interval.tick().await;
            info!("--- AlgoPioneer: Running Trade Cycle ---");

            let end = Utc::now();
            let start = end - ChronoDuration::minutes(1);
            let latest_candles = self
                .client
                .get_product_candles(
                    &self.config.product_id,
                    &start,
                    &end,
                    Granularity::OneMinute,
                )
                .await?;

            if let Some(latest_candle) = latest_candles.first() {
                let has_position = self.state.has_position(&self.config.product_id);
                let signal = self.strategy.update(latest_candle.close, has_position);
                info!("Latest Signal: {:?}", signal);

                match signal {
                    Signal::Buy => {
                        info!("Buy signal received. Placing order.");
                        self.client
                            .place_order(
                                &self.config.product_id,
                                "buy",
                                self.config.order_size,
                                None,
                            )
                            .await
                            .map_err(|e| e as Box<dyn std::error::Error>)?;

                        // Track position with full details for reconciliation
                        let detail = PositionDetail {
                            symbol: self.config.product_id.clone(),
                            side: "buy".to_string(),
                            quantity: self.config.order_size,
                            entry_price: Decimal::from_f64(latest_candle.close)
                                .unwrap_or(Decimal::ZERO),
                        };
                        self.state.open_position(detail);

                        // Non-blocking async state persistence
                        if let Err(e) = self.state_tx.send(self.state.clone()) {
                            warn!("Failed to queue state save: {}", e);
                        }
                    }
                    Signal::Sell => {
                        info!("Sell signal received. Placing order.");
                        self.client
                            .place_order(
                                &self.config.product_id,
                                "sell",
                                self.config.order_size,
                                None,
                            )
                            .await
                            .map_err(|e| e as Box<dyn std::error::Error>)?;

                        // Close position and log details
                        if let Some(closed) = self.state.close_position(&self.config.product_id) {
                            let exit_price =
                                Decimal::from_f64(latest_candle.close).unwrap_or(Decimal::ZERO);
                            let pnl = (exit_price - closed.entry_price) * closed.quantity;
                            info!(
                                "Closed position: entry={}, exit={}, pnl={}",
                                closed.entry_price, exit_price, pnl
                            );
                        }

                        // Non-blocking async state persistence
                        if let Err(e) = self.state_tx.send(self.state.clone()) {
                            warn!("Failed to queue state save: {}", e);
                        }
                    }
                    Signal::Hold => {
                        info!("Hold signal received. No action taken.");
                    }
                    Signal::Exit => {
                        info!("Exit signal received (unexpected for MA). Treating as Hold.");
                    }
                }
            } else {
                warn!("Warning: No data received for this interval.");
            }
        }
    }
}

// --- Backtesting Logic ---
use algopioneer::backtest;

/// Loads data and runs the backtest.
fn run_backtest() -> Result<(), Box<dyn std::error::Error>> {
    info!("--- Running Backtest for Moving Average Crossover Strategy ---");
    // Load historical data from CSV
    let file = File::open("sample_data.csv")?;
    let df = CsvReader::new(file).finish()?;

    // Create the strategy
    let strategy = MovingAverageCrossover::new(5, 10);

    // Run the backtest
    let initial_capital = 1000.0;
    let result = backtest::run(&strategy, &df, initial_capital)?;

    // Print the results
    info!("--- Backtest Results ---");
    info!("Initial Capital: ${:.2}", result.initial_capital);
    info!("Final Capital:   ${:.2}", result.final_capital);
    info!("Net Profit:      ${:.2}", result.net_profit);
    info!("Return:          {:.2}%", result.return_percentage());
    info!("Total Trades:    {}", result.total_trades);
    info!("Winning Trades:  {}", result.winning_trades);
    info!("Losing Trades:   {}", result.losing_trades);
    info!("------------------------");

    Ok(())
}

// --- Basis Trading Logic ---

/// CLI configuration for DualLeg strategy (externalized from hardcoded values)
struct DualLegCliConfig {
    order_size: f64,
    max_tick_age_ms: i64,
    execution_timeout_ms: i64,
    min_profit_threshold: f64,
    stop_loss_threshold: f64,
    throttle_interval_secs: u64,
}

async fn run_dual_leg_trading(
    strategy_type: &str,
    leg1_id: &str,
    leg2_id: &str,
    env: AppEnv,
    exchange_id: algopioneer::exchange::ExchangeId,
    cli_config: DualLegCliConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    info!(
        "--- AlgoPioneer: Initializing Dual-Leg Strategy ({}) on {} ---",
        strategy_type, exchange_id
    );

    // Initialize exchange client using factory
    let _exchange_config = algopioneer::exchange::ExchangeConfig::from_env(exchange_id)?;

    // For now, we still use CoinbaseClient for the execution engine since strategies
    // depend on the Executor trait from dual_leg_trading module.
    // The abstraction allows switching once Kraken is fully implemented.
    let _paper = matches!(env, AppEnv::Paper);
    if exchange_id == algopioneer::exchange::ExchangeId::Kraken {
        warn!("Kraken exchange selected but not fully implemented yet. Falling back to Coinbase for execution.");
    }

    // Create paper logger if in paper mode
    let recorder: Option<Arc<dyn TradeRecorder>> = if _paper {
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
                manager as Box<dyn algopioneer::strategy::dual_leg_trading::EntryStrategy>,
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
                manager as Box<dyn algopioneer::strategy::dual_leg_trading::EntryStrategy>,
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

// --- Pair Discovery Logic ---

/// Run pair discovery pipeline.
/// Note: Many arguments are acceptable for CLI entry points as they map to CLI flags.
#[allow(clippy::too_many_arguments)]
async fn run_discover_pairs(
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
    use algopioneer::discovery::config::{PortfolioPairConfig, DEFAULT_CANDIDATES};

    info!("--- AlgoPioneer: Pair Discovery Pipeline ---");

    // Parse symbols
    let candidates: Vec<String> = if symbols_arg == "default" {
        info!("Using default top-20 candidate pairs");
        DEFAULT_CANDIDATES.iter().map(|s| s.to_string()).collect()
    } else {
        symbols_arg
            .split(',')
            .map(|s| s.trim().to_string())
            .collect()
    };

    info!(
        candidates = candidates.len(),
        min_corr = min_correlation,
        max_hl = max_half_life,
        min_sharpe = min_sharpe,
        lookback = lookback_days,
        require_cointegration = !no_cointegration,
        "Configuration loaded"
    );

    // Build discovery config
    let config = DiscoveryConfig {
        candidates,
        min_correlation,
        max_half_life_hours: max_half_life,
        min_sharpe_ratio: min_sharpe,
        lookback_days,
        max_pairs_output: max_pairs,
        initial_capital: Decimal::from_f64_retain(initial_capital).unwrap_or(dec!(10000)),
        require_cointegration: !no_cointegration,
        ..Default::default()
    };

    // Initialize Coinbase client (discovery always uses Live mode, no paper trading)
    let mut client = CoinbaseClient::new(AppEnv::Live, None)?;

    // Run discovery pipeline
    info!("Starting discovery and optimization...");
    let clock = SystemClock;
    let results = match discover_and_optimize(&mut client, &config, &clock).await {
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
