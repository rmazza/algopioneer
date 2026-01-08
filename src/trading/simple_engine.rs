//! Moving Average Crossover trading engine.
//!
//! This engine implements a simple moving average crossover strategy
//! with support for both live and paper trading modes.

use crate::cli::SimpleTradingConfig;
use crate::exchange::coinbase::CoinbaseClient;
use crate::logging::TradeRecorder;
use crate::state::{PositionDetail, TradeState};
use crate::strategy::moving_average::MovingAverageCrossover;
use crate::strategy::Signal;

use cbadv::time::Granularity;
use chrono::{Duration as ChronoDuration, Utc};
use rust_decimal::prelude::*;
use std::sync::Arc;
use tokio::time::Duration;
use tracing::{info, warn};

/// Simple trading engine using Moving Average Crossover strategy.
///
/// This engine:
/// 1. Warms up the strategy with historical data
/// 2. Runs a continuous trading loop fetching new candles
/// 3. Executes buy/sell orders based on MA crossover signals
/// 4. Tracks positions with persistent state
pub struct SimpleTradingEngine {
    client: CoinbaseClient,
    strategy: MovingAverageCrossover,
    state: TradeState,
    config: SimpleTradingConfig,
    state_tx: tokio::sync::mpsc::UnboundedSender<TradeState>,
}

impl SimpleTradingEngine {
    /// Create a new trading engine with the given configuration.
    ///
    /// # Arguments
    /// * `config` - Trading configuration
    /// * `state_tx` - Channel for async state persistence
    /// * `recorder` - Optional trade recorder for paper trading
    ///
    /// # Errors
    /// Returns error if Coinbase client initialization fails.
    pub async fn new(
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

    /// Run the trading loop.
    ///
    /// This method:
    /// 1. Fetches historical data to warm up the strategy
    /// 2. Enters an infinite loop fetching new candles and generating signals
    /// 3. Executes orders based on signals
    ///
    /// # Errors
    /// Returns error if API calls fail.
    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
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

                        // CB-1 FIX: Fail loudly on price conversion failure
                        let entry_price = match Decimal::from_f64(latest_candle.close) {
                            Some(p) => p,
                            None => {
                                tracing::error!(
                                    price = latest_candle.close,
                                    "CRITICAL: Cannot convert entry price to Decimal (NaN/Inf). Skipping trade cycle."
                                );
                                continue; // Skip this cycle, don't corrupt state
                            }
                        };

                        // Track position with full details for reconciliation
                        let detail = PositionDetail {
                            symbol: self.config.product_id.clone(),
                            side: "buy".to_string(),
                            quantity: self.config.order_size,
                            entry_price,
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

                        // CB-1 FIX: Fail loudly on price conversion failure
                        let exit_price = match Decimal::from_f64(latest_candle.close) {
                            Some(p) => p,
                            None => {
                                tracing::error!(
                                    price = latest_candle.close,
                                    "CRITICAL: Cannot convert exit price to Decimal (NaN/Inf). Position closed without PnL tracking."
                                );
                                // Still close position but skip PnL calculation
                                self.state.close_position(&self.config.product_id);
                                continue;
                            }
                        };

                        // Close position and log details
                        if let Some(closed) = self.state.close_position(&self.config.product_id) {
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
