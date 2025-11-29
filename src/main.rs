use algopioneer::coinbase::{CoinbaseClient, AppEnv};
use algopioneer::coinbase::websocket::CoinbaseWebsocket;
use dotenv::dotenv;
use polars::prelude::*;
use std::fs::File;
use clap::Parser;
use tokio::time::Duration;
use chrono::{Utc, Duration as ChronoDuration};
use algopioneer::strategy::moving_average::MovingAverageCrossover;
use algopioneer::strategy::basis_trading::{BasisTradingStrategy, EntryManager, RiskMonitor, ExecutionEngine, RecoveryWorker, SystemClock};
use algopioneer::strategy::Signal;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use rust_decimal_macros::dec;
use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;

// --- Constants ---
const MAX_HISTORY: usize = 200;
const ORDER_SIZE: f64 = 0.001;
const SHORT_WINDOW: usize = 5;
const LONG_WINDOW: usize = 20;
const STATE_FILE: &str = "trade_state.json";

// --- State Persistence ---
#[derive(Serialize, Deserialize, Debug, Default)]
struct TradeState {
    position_open: bool,
}

impl TradeState {
    fn load() -> Self {
        if let Ok(data) = fs::read_to_string(STATE_FILE) {
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Self::default()
        }
    }

    fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        let mut file = fs::File::create(STATE_FILE)?;
        file.write_all(json.as_bytes())?;
        Ok(())
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
    },
    /// Run a backtest on historical data
    Backtest,
    /// Run the Delta-Neutral Basis Trading Strategy
    BasisTrade {
        /// Spot product (e.g., "BTC-USD")
        #[arg(long, default_value = "BTC-USD")]
        spot_id: String,
        /// Future product (e.g., "BTC-USDT")
        #[arg(long, default_value = "BTC-USDT")]
        future_id: String,
        /// Run in paper trading mode (simulated execution)
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
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(&cli.verbose)).init();

    match &cli.command {
        Commands::Trade { product_id, duration, paper } => {
            let env = if *paper { AppEnv::Paper } else { AppEnv::Live };
            run_trading(product_id, *duration, env).await?;
        }
        Commands::Backtest => {
            run_backtest()?;
        }
        Commands::BasisTrade { spot_id, future_id, paper } => {
            let env = if *paper { AppEnv::Paper } else { AppEnv::Live };
            run_basis_trading(spot_id, future_id, env).await?;
        }
    }

    Ok(())
}

/// Connects to Coinbase and executes the trading strategy.
async fn run_trading(product_id: &str, duration: u64, env: AppEnv) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- AlgoPioneer: Initializing ---");

    let mut client = CoinbaseClient::new(env)?;
    let strategy = MovingAverageCrossover::new(SHORT_WINDOW, LONG_WINDOW);
    
    // Load state
    let mut state = TradeState::load();
    println!("Loaded state: {:?}", state);

    // --- Warm up the strategy with historical data ---
    println!("Fetching historical data to warm up the strategy...");
    let end = Utc::now();
    let start = end - ChronoDuration::minutes((LONG_WINDOW * 5) as i64); // Fetch enough data
    let initial_candles = client.get_product_candles(product_id, &start, &end).await?;

    let mut times: Vec<i64> = initial_candles.iter().map(|c| c.start as i64).collect();
    let mut closes: Vec<f64> = initial_candles.iter().map(|c| c.close).collect();

    // Initial truncation if needed
    if times.len() > MAX_HISTORY {
        let remove_count = times.len() - MAX_HISTORY;
        times.drain(0..remove_count);
        closes.drain(0..remove_count);
    }

    let mut df = df! {
        "time" => &times,
        "close" => &closes,
    }?;

    println!("Strategy warmed up with {} data points.", df.height());

    // --- Main Trading Loop ---
    let mut interval = tokio::time::interval(Duration::from_secs(duration));
    
    loop {
        // Wait for the next tick
        interval.tick().await;
        println!("\n--- AlgoPioneer: Running Trade Cycle ---");

        // Fetch the latest candle
        let end = Utc::now();
        let start = end - ChronoDuration::minutes(1);
        let latest_candles = client.get_product_candles(product_id, &start, &end).await?;

        if let Some(latest_candle) = latest_candles.first() {
            // Append the latest candle
            times.push(latest_candle.start as i64);
            closes.push(latest_candle.close);

            // Fix Memory Leak: Keep vectors fixed size
            if times.len() > MAX_HISTORY {
                let remove_count = times.len() - MAX_HISTORY;
                times.drain(0..remove_count);
                closes.drain(0..remove_count);
            }

            df = df! {
                "time" => &times,
                "close" => &closes,
            }?;

            // Generate the latest signal
            let signal = strategy.get_latest_signal(&df, state.position_open)?;
            println!("Latest Signal: {:?}", signal);

            match signal {
                Signal::Buy => {
                    println!("Buy signal received. Placing order.");
                    let size = Decimal::from_f64(ORDER_SIZE).unwrap();
                    client.place_order(product_id, "buy", size).await.map_err(|e| e as Box<dyn std::error::Error>)?;
                    state.position_open = true;
                    state.save()?;
                }
                Signal::Sell => {
                    println!("Sell signal received. Placing order.");
                    let size = Decimal::from_f64(ORDER_SIZE).unwrap();
                    client.place_order(product_id, "sell", size).await.map_err(|e| e as Box<dyn std::error::Error>)?;
                    state.position_open = false;
                    state.save()?;
                }
                Signal::Hold => {
                    println!("Hold signal received. No action taken.");
                }
            }
        } else {
            eprintln!("Warning: No data received for this interval.");
        }
    }
}


// --- Backtesting Logic ---
use algopioneer::backtest;

/// Loads data and runs the backtest.
fn run_backtest() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Running Backtest for Moving Average Crossover Strategy ---");
    // Load historical data from CSV
    let file = File::open("sample_data.csv")?;
    let df = CsvReader::new(file).finish()?;

    // Create the strategy
    let strategy = MovingAverageCrossover::new(5, 10);

    // Run the backtest
    let initial_capital = 1000.0;
    let result = backtest::run(&strategy, &df, initial_capital)?;

    // Print the results
    println!("\n--- Backtest Results ---");
    println!("Initial Capital: ${:.2}", result.initial_capital);
    println!("Final Capital:   ${:.2}", result.final_capital);
    println!("Net Profit:      ${:.2}", result.net_profit);
    println!("Return:          {:.2}%", result.return_percentage());
    println!("Total Trades:    {}", result.total_trades);
    println!("Winning Trades:  {}", result.winning_trades);
    println!("Losing Trades:   {}", result.losing_trades);
    println!("------------------------");

    Ok(())
}


// --- Basis Trading Logic ---



async fn run_basis_trading(spot_id: &str, future_id: &str, env: AppEnv) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- AlgoPioneer: Initializing Basis Trading Strategy ---");

    // Initialize components
    // Initialize components
    let client = Arc::new(CoinbaseClient::new(env)?);

    // Create Recovery Channel
    let (recovery_tx, recovery_rx) = tokio::sync::mpsc::channel(100);

    // Spawn Recovery Worker
    let recovery_worker = RecoveryWorker::new(client.clone(), recovery_rx);
    tokio::spawn(async move {
        recovery_worker.run().await;
    });

    let entry_manager = Box::new(EntryManager::new(dec!(10.0), dec!(2.0))); // 10 bps entry, 2 bps exit
    let risk_monitor = RiskMonitor::new(dec!(3.0)); // 3x max leverage
    let execution_engine = ExecutionEngine::new(client.clone(), recovery_tx);

    let mut strategy = BasisTradingStrategy::new(
        entry_manager,
        risk_monitor,
        execution_engine,
        spot_id.to_string(),
        future_id.to_string(),
        Box::new(SystemClock),
    );

    // Create channels for market data
    let (spot_tx, spot_rx) = tokio::sync::mpsc::channel(100);
    let (future_tx, future_rx) = tokio::sync::mpsc::channel(100);

    // WebSocket Integration
    let ws_client = CoinbaseWebsocket::new()?;
    let products = vec![spot_id.to_string(), future_id.to_string()];
    let (ws_tx, mut ws_rx) = tokio::sync::mpsc::channel(100);

    // Spawn WebSocket Client
    tokio::spawn(async move {
        if let Err(e) = ws_client.connect_and_subscribe(products, ws_tx).await {
            eprintln!("WebSocket Error: {}", e);
        }
    });

    // Demultiplexer: Route WS messages to appropriate strategy channels
    let spot_id_clone = spot_id.to_string();
    let future_id_clone = future_id.to_string();
    
    tokio::spawn(async move {
        while let Some(data) = ws_rx.recv().await {
            if data.symbol == spot_id_clone {
                if let Err(_) = spot_tx.send(data).await { break; }
            } else if data.symbol == future_id_clone {
                if let Err(_) = future_tx.send(data).await { break; }
            }
        }
    });

    // Run strategy
    strategy.run(spot_rx, future_rx).await;

    Ok(())
}
