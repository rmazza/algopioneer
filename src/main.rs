// --- Global Allocator (Jemalloc for reduced fragmentation in long-running async apps) ---
#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use algopioneer::coinbase::websocket::CoinbaseWebsocket;
use algopioneer::coinbase::{AppEnv, CoinbaseClient};
use algopioneer::strategy::dual_leg_trading::{
    BasisManager, DualLegConfig, DualLegStrategy, ExecutionEngine, HedgeMode, InstrumentType,
    PairsManager, RecoveryWorker, RiskMonitor, SystemClock, TransactionCostModel,
};
use algopioneer::strategy::moving_average::MovingAverageCrossover;
use algopioneer::strategy::portfolio::PortfolioManager;
use algopioneer::strategy::Signal;
use cbadv::time::Granularity;
use chrono::{Duration as ChronoDuration, Utc};
use clap::Parser;
use dotenv::dotenv;
use polars::prelude::*;
use rust_decimal::prelude::FromPrimitive;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use tokio::time::Duration;
use tracing::{error, info, warn};

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
        let mut file = fs::File::create(STATE_FILE)?;
        file.write_all(json.as_bytes())?;
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
        /// Run in paper trading mode
        #[arg(long, default_value_t = false)]
        paper: bool,
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

            let mut engine = SimpleTradingEngine::new(config, state_tx).await?;
            engine.run().await?;
        }
        Commands::Backtest => {
            run_backtest()?;
        }
        Commands::DualLeg {
            strategy,
            symbols,
            paper,
            order_size,
            max_tick_age_ms,
            execution_timeout_ms,
            min_profit_threshold,
            stop_loss_threshold,
            throttle_interval_secs,
        } => {
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
            run_dual_leg_trading(strategy, parts[0], parts[1], env, dual_leg_config).await?;
        }
        Commands::Portfolio { config, paper } => {
            let env = if *paper { AppEnv::Paper } else { AppEnv::Live };
            let mut manager = PortfolioManager::new(config, env);
            manager.run().await?;
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
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let client = CoinbaseClient::new(config.env.clone())?;
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
                            entry_price: Decimal::from_f64(latest_candle.close).unwrap_or(dec!(0)),
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
                                Decimal::from_f64(latest_candle.close).unwrap_or(dec!(0));
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
    cli_config: DualLegCliConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    info!(
        "--- AlgoPioneer: Initializing Dual-Leg Strategy ({}) ---",
        strategy_type
    );

    // Initialize components
    let client = Arc::new(CoinbaseClient::new(env)?);

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
            if arc_data.symbol == leg1_id_clone {
                if let Err(_) = leg1_tx.send(arc_data.clone()).await {
                    break;
                }
            } else if arc_data.symbol == leg2_id_clone {
                if let Err(_) = leg2_tx.send(arc_data.clone()).await {
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
