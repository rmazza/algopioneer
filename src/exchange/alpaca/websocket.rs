//! Alpaca WebSocket Provider
//!
//! Market data provider for Alpaca using polling (reliable for equities).
//!
//! Note: The apca crate's streaming API has complex generics that are difficult
//! to work with. This polling implementation provides reliable market data
//! updates which are sufficient for equity trading (not latency-sensitive like crypto).

use async_trait::async_trait;
use futures_util::stream::{self, StreamExt};
use governor::{Quota, RateLimiter};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

use tracing::{debug, info, warn};

use apca::data::v2::bars as alpaca_bars;
use apca::{ApiInfo, Client};

use crate::exchange::alpaca::utils;
use crate::exchange::{ExchangeConfig, ExchangeError, WebSocketProvider};
use crate::strategy::dual_leg_trading::{Clock, SystemClock};
use crate::types::MarketData;

/// MC-1 FIX: Static instrument_id to avoid heap allocation per tick
static ALPACA_INSTRUMENT_ID: &str = "alpaca";

/// Alpaca free tier rate limit: 200 req/min ≈ 3.3 req/sec
/// Using 3 req/sec to stay safely under the limit
const RATE_LIMIT_PER_SECOND: u32 = 3;

/// CB-2 FIX: Compile-time NonZeroU32 to avoid runtime .expect()
const RATE_LIMIT_NZ: NonZeroU32 = match NonZeroU32::new(RATE_LIMIT_PER_SECOND) {
    Some(v) => v,
    None => panic!("RATE_LIMIT_PER_SECOND must be > 0"),
};

/// Maximum number of concurrent poll requests
const MAX_CONCURRENT_POLLS: usize = 5;

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
        info!(
            symbols_count = symbols.len(),
            poll_interval = self.poll_interval_secs,
            "Starting Alpaca data provider (Actor-Lite Poll Mode)"
        );

        // NOTE: We construct ApiInfo directly to avoid abusing global environment variables
        let base_url = if self.sandbox {
            "https://paper-api.alpaca.markets"
        } else {
            "https://api.alpaca.markets"
        };

        let api_info =
            ApiInfo::from_parts(base_url, &self.api_key, &self.api_secret).map_err(|e| {
                ExchangeError::Configuration(format!("Failed to create Alpaca API info: {}", e))
            })?;

        let client = Arc::new(Client::new(api_info));
        let rate_limiter = Arc::new(RateLimiter::direct(Quota::per_second(RATE_LIMIT_NZ)));
        let clock = Arc::clone(&self.clock);

        // Pre-compute symbol mappings once
        // FIX: Use Arc<str> for cheap cloning in hot loops
        let monitored_symbols: Arc<[(Arc<str>, Arc<str>)]> = symbols
            .iter()
            .map(|s| (Arc::from(s.as_str()), Arc::from(utils::to_alpaca_symbol(s))))
            .collect::<Vec<_>>()
            .into();

        let poll_interval = Duration::from_secs(self.poll_interval_secs);

        // Spawn a dedicated poller task to decouple scheduling from execution
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(poll_interval);
            // set missed tick behavior to delay to avoid bursts if we get behind
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

            loop {
                // Wait for next tick
                interval.tick().await;

                if sender.is_closed() {
                    info!("Channel closed, stopping polling task");
                    break;
                }

                let start_poll = std::time::Instant::now();

                // Fan-out: Create a stream of concurrent fetch futures
                let fetches = stream::iter(monitored_symbols.iter().cloned())
                    .map(|(orig, alpaca)| {
                        let client = client.clone();
                        let rl = rate_limiter.clone();
                        let clock = clock.clone();

                        async move {
                            rl.until_ready().await;
                            Self::fetch_single_bar(&client, &clock, &orig, &alpaca).await
                        }
                    })
                    // Limit concurrency to avoid overwhelming the runtime or hitting rate limits too hard
                    .buffer_unordered(MAX_CONCURRENT_POLLS);

                // Fan-in: Process results
                let mut results = fetches;
                let mut processed_count = 0;

                while let Some(result) = results.next().await {
                    match result {
                        Ok(Some(data)) => {
                            let symbol = data.symbol.clone();
                            if let Err(_e) = sender.send(data).await {
                                info!("Channel closed during send, stopping polling");
                                return;
                            }
                            crate::metrics::record_ws_tick(&symbol);
                            processed_count += 1;
                        }
                        Ok(None) => {
                            // No new data
                        }
                        Err(e) => {
                            warn!(error = %e, "Poll failed");
                        }
                    }
                }

                let elapsed = start_poll.elapsed();
                if elapsed > poll_interval {
                    warn!(
                        elapsed_ms = elapsed.as_millis(),
                        interval_ms = poll_interval.as_millis(),
                        "Polling cycle saturated interval - system falling behind"
                    );
                } else {
                    debug!(
                        processed = processed_count,
                        elapsed_ms = elapsed.as_millis(),
                        "Poll cycle complete"
                    );
                }
            }
        });

        Ok(())
    }
}

impl AlpacaWebSocketProvider {
    // Helper for fetching a single bar (isolated logic)
    // Returns Result<Option<MarketData>>: Ok(None) means success but no data.
    async fn fetch_single_bar(
        client: &Client,
        clock: &Arc<dyn Clock>,
        orig_symbol: &str,
        alpaca_symbol: &str,
    ) -> Result<Option<MarketData>, ExchangeError> {
        let now = clock.now();
        let start = now - chrono::Duration::minutes(2);

        let request = alpaca_bars::ListReqInit {
            limit: Some(1),
            feed: Some(apca::data::v2::Feed::IEX),
            ..Default::default()
        }
        .init(alpaca_symbol, start, now, alpaca_bars::TimeFrame::OneMinute);

        let fetch_start = std::time::Instant::now();
        let result = client.issue::<alpaca_bars::List>(&request).await;
        let fetch_elapsed = fetch_start.elapsed().as_secs_f64();

        match result {
            Ok(result) => {
                crate::metrics::record_poll_latency(orig_symbol, "success", fetch_elapsed);

                if let Some(bar) = result.bars.first() {
                    let price = match utils::num_to_decimal(&bar.close) {
                        Ok(p) => p,
                        Err(e) => {
                            warn!(error = %e, symbol = %alpaca_symbol, "Price conversion error");
                            crate::metrics::record_dropped_tick(orig_symbol, "conversion_error");
                            return Ok(None);
                        }
                    };
                    let timestamp = bar.time.timestamp_millis();

                    Ok(Some(MarketData {
                        symbol: orig_symbol.to_string(),
                        instrument_id: Some(ALPACA_INSTRUMENT_ID.to_owned()),
                        price,
                        timestamp,
                    }))
                } else {
                    Ok(None)
                }
            }
            Err(e) => {
                crate::metrics::record_poll_latency(orig_symbol, "error", fetch_elapsed);
                warn!(symbol = %alpaca_symbol, error = %e, "Failed to fetch bar");
                crate::metrics::record_dropped_tick(orig_symbol, "fetch_error");
                Err(ExchangeError::Network(format!("Fetch failed: {}", e)))
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
