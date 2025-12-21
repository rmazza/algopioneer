//! Alpaca WebSocket Provider
//!
//! Market data provider for Alpaca using polling (reliable for equities).
//!
//! Note: The apca crate's streaming API has complex generics that are difficult
//! to work with. This polling implementation provides reliable market data
//! updates which are sufficient for equity trading (not latency-sensitive like crypto).

use async_trait::async_trait;

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{debug, info, warn};

use apca::data::v2::bars as alpaca_bars;
use apca::{ApiInfo, Client};

use crate::exchange::alpaca::utils;
use crate::exchange::{ExchangeConfig, ExchangeError, WebSocketProvider};
use crate::strategy::dual_leg_trading::{Clock, SystemClock};
use crate::types::MarketData;

/// Alpaca data provider using polling for reliable market data
///
/// Uses 1-minute bar polling instead of WebSocket streaming.
/// This is simpler and more reliable for equity trading.
///
/// Note: Stock data only available during market hours (9:30 AM - 4 PM ET)
pub struct AlpacaWebSocketProvider {
    api_key: String,
    api_secret: String,
    sandbox: bool,
    /// Polling interval in seconds
    poll_interval_secs: u64,
    /// Injected clock for deterministic time
    clock: Arc<dyn Clock>,
}

impl AlpacaWebSocketProvider {
    /// Create a new Alpaca WebSocket provider
    pub fn new(config: &ExchangeConfig) -> Result<Self, ExchangeError> {
        info!(
            sandbox = config.sandbox,
            "Creating Alpaca data provider (polling mode)"
        );
        Ok(Self {
            api_key: config.api_key.clone(),
            api_secret: config.api_secret.clone(),
            sandbox: config.sandbox,
            poll_interval_secs: 5, // 5 second polling interval
            clock: Arc::new(SystemClock),
        })
    }

    /// Create from environment variables
    pub fn from_env() -> Result<Self, ExchangeError> {
        let api_key = std::env::var("ALPACA_API_KEY")
            .map_err(|_| ExchangeError::Configuration("ALPACA_API_KEY must be set".to_string()))?;
        let api_secret = std::env::var("ALPACA_API_SECRET").map_err(|_| {
            ExchangeError::Configuration("ALPACA_API_SECRET must be set".to_string())
        })?;

        // N-3 FIX: Read sandbox from env, default to paper (true) for safety
        let sandbox = std::env::var("ALPACA_SANDBOX")
            .map(|v| v != "false" && v != "0")
            .unwrap_or(true);

        Ok(Self {
            api_key,
            api_secret,
            sandbox,
            poll_interval_secs: 5,
            clock: Arc::new(SystemClock),
        })
    }

    /// Create with injected clock for testing
    pub fn with_clock(
        config: &ExchangeConfig,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, ExchangeError> {
        info!(
            sandbox = config.sandbox,
            "Creating Alpaca data provider with injected clock"
        );
        Ok(Self {
            api_key: config.api_key.clone(),
            api_secret: config.api_secret.clone(),
            sandbox: config.sandbox,
            poll_interval_secs: 5,
            clock,
        })
    }
}

#[async_trait]
impl WebSocketProvider for AlpacaWebSocketProvider {
    async fn connect_and_subscribe(
        &self,
        symbols: Vec<String>,
        sender: mpsc::Sender<MarketData>,
    ) -> Result<(), ExchangeError> {
        // Convert symbols to Alpaca format (into_owned() for Cow -> String)
        let alpaca_symbols: Vec<String> = symbols
            .iter()
            .map(|s| utils::to_alpaca_symbol(s).into_owned())
            .collect();

        info!(
            symbols = ?alpaca_symbols,
            poll_interval = self.poll_interval_secs,
            "Starting Alpaca data provider (polling mode)"
        );

        // SAFETY: We construct ApiInfo directly to avoid mutating global environment variables
        let base_url = if self.sandbox {
            "https://paper-api.alpaca.markets"
        } else {
            "https://api.alpaca.markets"
        };

        let api_info =
            ApiInfo::from_parts(base_url, &self.api_key, &self.api_secret).map_err(|e| {
                ExchangeError::Configuration(format!("Failed to create Alpaca API info: {}", e))
            })?;

        let client = Client::new(api_info);

        // Polling loop
        let mut poll_timer = interval(Duration::from_secs(self.poll_interval_secs));

        info!("Alpaca polling started");

        loop {
            poll_timer.tick().await;

            // Fetch latest bar for each symbol
            for (orig_symbol, alpaca_symbol) in symbols.iter().zip(alpaca_symbols.iter()) {
                let now = self.clock.now();
                let start = now - chrono::Duration::minutes(2);

                let request = alpaca_bars::ListReqInit {
                    limit: Some(1),
                    ..Default::default()
                }
                .init(alpaca_symbol, start, now, alpaca_bars::TimeFrame::OneMinute);

                match client.issue::<alpaca_bars::List>(&request).await {
                    Ok(result) => {
                        if let Some(bar) = result.bars.first() {
                            // CB-2 FIX: Propagate conversion error instead of silent zero
                            let price = match utils::num_to_decimal(&bar.close) {
                                Ok(p) => p,
                                Err(e) => {
                                    warn!(error = %e, symbol = %alpaca_symbol, "Skipping bar due to price conversion error");
                                    continue;
                                }
                            };
                            let timestamp = bar.time.timestamp_millis();

                            debug!(
                                symbol = %alpaca_symbol,
                                price = %price,
                                "Price update"
                            );

                            let market_data = MarketData {
                                symbol: orig_symbol.clone(),
                                instrument_id: Some("alpaca".to_string()),
                                price,
                                timestamp,
                            };

                            if sender.send(market_data).await.is_err() {
                                info!("Channel closed, stopping polling");
                                return Ok(());
                            }
                        }
                    }
                    Err(e) => {
                        warn!(
                            symbol = %alpaca_symbol,
                            error = %e,
                            "Failed to fetch bar (market may be closed)"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let config = ExchangeConfig {
            api_key: "test".to_string(),
            api_secret: "test".to_string(),
            sandbox: true,
        };

        let provider = AlpacaWebSocketProvider::new(&config);
        assert!(provider.is_ok());
    }

    #[test]
    fn test_provider_with_clock() {
        let config = ExchangeConfig {
            api_key: "test".to_string(),
            api_secret: "test".to_string(),
            sandbox: true,
        };
        let clock = Arc::new(SystemClock);
        let provider = AlpacaWebSocketProvider::with_clock(&config, clock);
        assert!(provider.is_ok());
    }
}
