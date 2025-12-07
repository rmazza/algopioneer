use crate::coinbase::{CoinbaseClient, AppEnv};
use crate::coinbase::websocket::CoinbaseWebsocket;
use crate::strategy::dual_leg_trading::{
    DualLegStrategy, DualLegConfig, ExecutionEngine, RecoveryWorker, 
    SystemClock, PairsManager, 
    RiskMonitor, InstrumentType, HedgeMode, MarketData
};

use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio::time::Duration;
use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use tracing::{info, error, debug, warn};

use rust_decimal_macros::dec;
use std::fs::File;
use std::io::BufReader;
use serde::{Serialize, Deserialize};

// Channel buffer size constants
const WS_CHANNEL_BUFFER: usize = 1000; // ~1 sec of ticks at 1000 ticks/sec
const STRATEGY_CHANNEL_BUFFER: usize = 100; // ~100ms buffer per leg
const RECOVERY_CHANNEL_BUFFER: usize = 100;
const FEEDBACK_CHANNEL_BUFFER: usize = 100;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioPairConfig {
    #[serde(flatten)]
    pub dual_leg_config: DualLegConfig,
    pub window_size: usize,
    pub entry_z_score: f64,
    pub exit_z_score: f64,
}

impl PortfolioPairConfig {
    pub fn pair_id(&self) -> String {
        format!("{}-{}", self.dual_leg_config.spot_symbol, self.dual_leg_config.future_symbol)
    }
}

pub struct PortfolioManager {
    config_path: String,
    env: AppEnv,
    ws_task_handle: Option<tokio::task::JoinHandle<()>>,
}

impl PortfolioManager {
    pub fn new(config_path: &str, env: AppEnv) -> Self {
        Self {
            config_path: config_path.to_string(),
            env,
            ws_task_handle: None,
        }
    }

    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Starting Portfolio Manager with config: {}", self.config_path);

        // 1. Load Configuration
        let file = File::open(&self.config_path)?;
        let reader = BufReader::new(file);
        let config_list: Vec<PortfolioPairConfig> = serde_json::from_reader(reader)?;

        if config_list.is_empty() {
            return Err("No pairs found in configuration.".into());
        }

        info!("Loaded {} pairs from config.", config_list.len());

        // Store configs for restart capability
        let config_map: HashMap<String, Arc<PortfolioPairConfig>> = config_list
            .into_iter()
            .map(|cfg| {
                let pair_id = cfg.pair_id();
                (pair_id, Arc::new(cfg))
            })
            .collect();

        // 2. Initialize Shared Resources
        let client = Arc::new(CoinbaseClient::new(self.env.clone())?); 
        
        // 3. Spawn Strategy Actors
        // We use JoinSet to monitor actors. They return their pair_id on exit.
        let mut join_set: JoinSet<String> = JoinSet::new();
        
        // Symbol Map: Symbol -> List of (Sender, PairID)
        // We need PairID to remove stale senders on restart.
        let mut symbol_map: HashMap<String, Vec<(mpsc::Sender<Arc<MarketData>>, String)>> = HashMap::new();
        
        // OW1: Use HashSet for O(n) instead of O(n²) duplicate checking
        let mut symbols_set: HashSet<String> = HashSet::new();
        for config in config_map.values() {
            symbols_set.insert(config.dual_leg_config.spot_symbol.clone());
            symbols_set.insert(config.dual_leg_config.future_symbol.clone());
        }
        let symbols_to_subscribe: Vec<String> = symbols_set.into_iter().collect();

        // Spawn all strategies
        for config in config_map.values() {
            Self::spawn_strategy(
                &mut join_set, 
                &mut symbol_map, 
                client.clone(), 
                config.clone()
            ).await;
        }

        // 4. WebSocket Connection & Demultiplexer
        let ws_client = CoinbaseWebsocket::new()?;
        let (ws_tx, mut ws_rx) = mpsc::channel(WS_CHANNEL_BUFFER); 

        let symbols_clone = symbols_to_subscribe.clone();
        let ws_handle = tokio::spawn(async move {
            if let Err(e) = ws_client.connect_and_subscribe(symbols_clone, ws_tx).await {
                error!("WebSocket Error: {}", e);
            }
        });
        self.ws_task_handle = Some(ws_handle);

        info!("Portfolio Manager running. Monitoring {} strategies.", config_map.len());

        // Supervisor Loop
        loop {
            tokio::select! {
                // Handle WebSocket Ticks
                res = ws_rx.recv() => {
                    match res {
                        Some(data) => {
                            let arc_data = Arc::new(data);
                            // CF5: Clean up closed senders before routing
                            if let Some(senders) = symbol_map.get_mut(&arc_data.symbol) {
                                senders.retain(|(sender, _)| !sender.is_closed());
                                
                                for (sender, _pair_id) in senders {
                                    if let Err(e) = sender.send(arc_data.clone()).await {
                                        debug!("Failed to route tick to strategy: {}", e);
                                    }
                                }
                            }
                        }
                        None => {
                            // CF4: Implement WebSocket reconnection instead of fatal error
                            error!("WebSocket channel closed. Attempting reconnection...");
                            
                            let mut reconnect_attempts = 0;
                            const MAX_RECONNECT_ATTEMPTS: u32 = 5;
                            let mut backoff = Duration::from_secs(1);
                            
                            loop {
                                reconnect_attempts += 1;
                                if reconnect_attempts > MAX_RECONNECT_ATTEMPTS {
                                    error!("FATAL: WebSocket reconnection failed after {} attempts", MAX_RECONNECT_ATTEMPTS);
                                    return Err("WebSocket permanent failure".into());
                                }
                                
                                warn!("Reconnection attempt {} of {}", reconnect_attempts, MAX_RECONNECT_ATTEMPTS);
                                tokio::time::sleep(backoff).await;
                                
                                match CoinbaseWebsocket::new() {
                                    Ok(new_ws_client) => {
                                        let (new_ws_tx, new_ws_rx) = mpsc::channel(WS_CHANNEL_BUFFER);
                                        
                                        // NI1 Fix: Abort old WebSocket task before spawning new one
                                        if let Some(old_handle) = self.ws_task_handle.take() {
                                            old_handle.abort();
                                            debug!("Aborted old WebSocket task before reconnection");
                                        }
                                        
                                        let symbols_clone = symbols_to_subscribe.clone();
                                        let new_handle = tokio::spawn(async move {
                                            if let Err(e) = new_ws_client.connect_and_subscribe(symbols_clone, new_ws_tx).await {
                                                error!("WebSocket reconnection error: {}", e);
                                            }
                                        });
                                        self.ws_task_handle = Some(new_handle);
                                        
                                        ws_rx = new_ws_rx;
                                        info!("WebSocket reconnected successfully!");
                                        break;
                                    }
                                    Err(e) => {
                                        error!("Reconnection failed: {}", e);
                                        backoff = std::cmp::min(backoff * 2, Duration::from_secs(30));
                                    }
                                }
                            }
                        }
                    }
                }

                // Handle Actor Termination (Supervisor Pattern)
                Some(res) = join_set.join_next() => {
                    match res {
                        Ok(pair_id) => {
                            error!(pair_id = %pair_id, "Strategy exited unexpectedly, initiating restart");
                            
                            // 1. Clean up old routing entries for this pair
                            for (_, list) in symbol_map.iter_mut() {
                                list.retain(|(_, pid)| pid != &pair_id);
                            }

                            // 2. Retrieve Config and Respawn
                            if let Some(config) = config_map.get(&pair_id) {
                                // CF2: Add backoff to prevent tight crash loops (removed duplicate)
                                tokio::time::sleep(Duration::from_secs(1)).await;
                                
                                Self::spawn_strategy(
                                    &mut join_set, 
                                    &mut symbol_map, 
                                    client.clone(), 
                                    config.clone()
                                ).await;
                                
                                info!(pair_id = %pair_id, "Strategy restarted successfully");
                            } else {
                                error!(pair_id = %pair_id, "CRITICAL: Could not find config for exited pair");
                            }
                        }
                        Err(e) => {
                            // CF3: Handle panic case - we can't identify which strategy crashed
                            // without additional tracking. Log detailed error for manual intervention.
                            let error_type = if e.is_panic() {
                                "panicked"
                            } else if e.is_cancelled() {
                                "cancelled"
                            } else {
                                "failed"
                            };
                            
                            error!(
                                error_type = error_type,
                                error = %e,
                                "CRITICAL: Strategy task {} but pair_id unknown. Cannot auto-restart.",
                                error_type
                            );
                            error!("Hint: Check symbol_map for stale channels or review logs for last strategy activity");
                            
                            // In production, this should trigger alerts for manual intervention
                            // since we cannot safely restart without knowing which strategy failed
                        }
                    }
                }
            }
        }
    }

    async fn spawn_strategy(
        join_set: &mut JoinSet<String>,
        symbol_map: &mut HashMap<String, Vec<(mpsc::Sender<Arc<MarketData>>, String)>>,
        client: Arc<CoinbaseClient>,
        config: Arc<PortfolioPairConfig>
    ) {
        let pair_id = config.pair_id();
        info!("Initializing strategy for {}", pair_id);

        // Create Channels
        let (leg1_tx, leg1_rx) = mpsc::channel(STRATEGY_CHANNEL_BUFFER);
        let (leg2_tx, leg2_rx) = mpsc::channel(STRATEGY_CHANNEL_BUFFER);
        
        // Map symbols to channels (1-to-N)
        symbol_map.entry(config.dual_leg_config.spot_symbol.clone()).or_default().push((leg1_tx, pair_id.clone()));
        symbol_map.entry(config.dual_leg_config.future_symbol.clone()).or_default().push((leg2_tx, pair_id.clone()));

        // Setup Strategy Components
        let (recovery_tx, recovery_rx) = mpsc::channel(RECOVERY_CHANNEL_BUFFER);
        let (feedback_tx, feedback_rx) = mpsc::channel(FEEDBACK_CHANNEL_BUFFER);

        let recovery_worker = RecoveryWorker::new(client.clone(), recovery_rx, feedback_tx);
        tokio::spawn(async move {
            recovery_worker.run().await;
        });

        let execution_engine = ExecutionEngine::new(client.clone(), recovery_tx);

        // Strategy Logic (Pairs Trading for Portfolio)
        // Use dynamic parameters from config
        let manager = Box::new(PairsManager::new(
            config.window_size, 
            config.entry_z_score, 
            config.exit_z_score
        ));
        let monitor = RiskMonitor::new(dec!(3.0), InstrumentType::Linear, HedgeMode::DollarNeutral);

        let mut strategy = DualLegStrategy::new(
            manager,
            monitor,
            execution_engine,
            config.dual_leg_config.clone(),
            feedback_rx,
            Box::new(SystemClock),
        );

        let pid_clone = pair_id.clone();
        join_set.spawn(async move {
            strategy.run(leg1_rx, leg2_rx).await;
            pid_clone // Return ID on exit
        });
    }
}
