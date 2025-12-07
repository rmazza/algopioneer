use crate::coinbase::{CoinbaseClient, AppEnv};
use crate::coinbase::websocket::CoinbaseWebsocket;
    DualLegStrategy, DualLegConfig, ExecutionEngine, RecoveryWorker, 
    SystemClock, PairsManager, 
    RiskMonitor, InstrumentType, HedgeMode, MarketData
};

use tokio::sync::mpsc;
use tokio::task::JoinSet;
use std::sync::Arc;
use std::collections::HashMap;
use tracing::{info, error, debug};

use rust_decimal_macros::dec;
use std::fs::File;
use std::io::BufReader;
use serde::{Serialize, Deserialize};

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
}

impl PortfolioManager {
    pub fn new(config_path: &str, env: AppEnv) -> Self {
        Self {
            config_path: config_path.to_string(),
            env,
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
        let mut config_map: HashMap<String, Arc<PortfolioPairConfig>> = HashMap::new();
        for cfg in config_list {
            let pair_id = cfg.pair_id();
            config_map.insert(pair_id, Arc::new(cfg));
        }

        // 2. Initialize Shared Resources
        let client = Arc::new(CoinbaseClient::new(self.env.clone())?); 
        
        // 3. Spawn Strategy Actors
        // We use JoinSet to monitor actors. They return their pair_id on exit.
        let mut join_set: JoinSet<String> = JoinSet::new();
        
        // Symbol Map: Symbol -> List of (Sender, PairID)
        // We need PairID to remove stale senders on restart.
        let mut symbol_map: HashMap<String, Vec<(mpsc::Sender<Arc<MarketData>>, String)>> = HashMap::new();
        let mut symbols_to_subscribe: Vec<String> = Vec::new();

        for config in config_map.values() {
            // Register symbols first
            if !symbols_to_subscribe.contains(&config.dual_leg_config.spot_symbol) {
                symbols_to_subscribe.push(config.dual_leg_config.spot_symbol.clone());
            }
            if !symbols_to_subscribe.contains(&config.dual_leg_config.future_symbol) {
                symbols_to_subscribe.push(config.dual_leg_config.future_symbol.clone());
            }

            Self::spawn_strategy(
                &mut join_set, 
                &mut symbol_map, 
                client.clone(), 
                config.clone()
            ).await;
        }

        // 4. WebSocket Connection & Demultiplexer
        let ws_client = CoinbaseWebsocket::new()?;
        let (ws_tx, mut ws_rx) = mpsc::channel(1000); 

        tokio::spawn(async move {
            if let Err(e) = ws_client.connect_and_subscribe(symbols_to_subscribe, ws_tx).await {
                error!("WebSocket Error: {}", e);
            }
        });

        info!("Portfolio Manager running. Monitoring {} strategies.", config_map.len());

        // Supervisor Loop
        loop {
            tokio::select! {
                // Handle WebSocket Ticks
                // Handle WebSocket Ticks
                res = ws_rx.recv() => {
                    match res {
                        Some(data) => {
                            let arc_data = Arc::new(data);
                            if let Some(senders) = symbol_map.get(&arc_data.symbol) {
                                // We need to iterate and send. 
                                // If send fails (channel closed), it means actor died. 
                                // We could clean up here, but `join_next` handles the death event.
                                for (sender, _pair_id) in senders {
                                    if let Err(e) = sender.send(arc_data.clone()).await {
                                        // Log debug, but rely on join_set to handle the crash
                                        debug!("Failed to route tick to strategy: {}", e);
                                    }
                                }
                            }
                        }
                        None => {
                            error!("CRITICAL: WebSocket channel closed. Connection lost.");
                            return Err("WebSocket connection lost".into());
                        }
                    }
                }

                // Handle Actor Termination (Supervisor Pattern)
                Some(res) = join_set.join_next() => {
                    match res {
                        Ok(pair_id) => {
                            error!("Strategy {} exited unexpectedly! Restarting...", pair_id);
                            
                            // 1. Clean up old routing entries for this pair
                            for (_, list) in symbol_map.iter_mut() {
                                list.retain(|(_, pid)| pid != &pair_id);
                            }

                            // 2. Retrieve Config
                            if let Some(config) = config_map.get(&pair_id) {
                                // 3. Respawn
                                // Add a small backoff to prevent tight loop crashing
                                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                                
                                // 3. Respawn
                                // Add a small backoff to prevent tight loop crashing
                                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                                
                                Self::spawn_strategy(
                                    &mut join_set, 
                                    &mut symbol_map, 
                                    client.clone(), 
                                    config.clone()
                                ).await;
                                
                                info!("Strategy {} restarted successfully.", pair_id);
                            } else {
                                error!("CRITICAL: Could not find config for exited pair {}", pair_id);
                            }
                        }
                        Err(e) => {
                            if e.is_panic() {
                                error!("Strategy task panicked!");
                            } else {
                                error!("Strategy task cancelled or failed: {}", e);
                            }
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
        let (leg1_tx, leg1_rx) = mpsc::channel(100);
        let (leg2_tx, leg2_rx) = mpsc::channel(100);
        
        // Map symbols to channels (1-to-N)
        symbol_map.entry(config.dual_leg_config.spot_symbol.clone()).or_default().push((leg1_tx, pair_id.clone()));
        symbol_map.entry(config.dual_leg_config.future_symbol.clone()).or_default().push((leg2_tx, pair_id.clone()));

        // Setup Strategy Components
        let (recovery_tx, recovery_rx) = mpsc::channel(100);
        let (feedback_tx, feedback_rx) = mpsc::channel(100);

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
