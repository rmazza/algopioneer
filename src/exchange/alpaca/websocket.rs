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
        // MC-3 FIX: Arc for zero-copy sharing across ticks
        // Store as Arc<[(Arc<original>, Arc<alpaca>)]> so we can clone individual items zero-copy
        let symbol_pairs: Arc<[(Arc<str>, Arc<str>)]> = symbols
            .iter()
            .map(|s| (Arc::from(s.as_str()), Arc::from(utils::to_alpaca_symbol(s))))
            .collect::<Vec<_>>()
            .into();

        info!(
            symbols_count = symbol_pairs.len(),
            poll_interval = self.poll_interval_secs,
            rate_limit_per_sec = RATE_LIMIT_PER_SECOND,
            "Starting Alpaca data provider (polling mode with concurrent requests)"
        );

        // SAFETY: We construct ApiInfo directly to avoid abusing global environment variables
        let base_url = if self.sandbox {
            "https://paper-api.alpaca.markets"
        } else {
            "https://api.alpaca.markets"
        };

        let api_info =
            ApiInfo::from_parts(base_url, &self.api_key, &self.api_secret).map_err(|e| {
                ExchangeError::Configuration(format!("Failed to create Alpaca API info: {}", e))
            })?;

        // Wrap client in Arc for cheap cloning across tasks
        let client = Arc::new(Client::new(api_info));

        // Rate limiter to prevent API abuse (Alpaca free tier: 200 req/min ~ 3.3 req/sec)
        // CB-2 FIX: Uses compile-time RATE_LIMIT_NZ constant
        let rate_limiter = Arc::new(RateLimiter::direct(Quota::per_second(RATE_LIMIT_NZ)));

        // Polling loop
        // FIX: Replaced fixed interval with adaptive polling to preventing "Latency Spiral"
        // The loop will now run as fast as the rate limiter allows, but never Faster than poll_interval_secs.
        // If the work takes longer than poll_interval_secs (due to rate limits), the next cycle starts immediately.
        info!("Alpaca concurrent polling started (Adaptive Mode)");

        loop {
            let cycle_start = tokio::time::Instant::now();

            // MC-3 FIX: Arc::clone is O(1)
            let pairs = Arc::clone(&symbol_pairs);
            let client = Arc::clone(&client);
            let rate_limiter = Arc::clone(&rate_limiter);
            let stream_sender = sender.clone();
            let clock = Arc::clone(&self.clock);

            // Create a stream of futures to fetch data concurrently
            // Using buffer_unordered to execute up to MAX_CONCURRENT_POLLS requests in parallel
            let mut stream = stream::iter(pairs.iter().cloned())
                .map(move |(orig_symbol, alpaca_symbol)| {
                    let client = client.clone();
                    let rate_limiter = rate_limiter.clone();
                    let sender = stream_sender.clone();
                    let clock = clock.clone();
                    // Clone Arcs to strings (cheap) for the async block
                    let orig_symbol = orig_symbol.clone();
                    let alpaca_symbol = alpaca_symbol.clone();

                    async move {
                        // Wait for rate limiter - this provides the pacing
                        rate_limiter.until_ready().await;

                        let now = clock.now();
                        let start = now - chrono::Duration::minutes(2);

                        let request = alpaca_bars::ListReqInit {
                            limit: Some(1),
                            feed: Some(apca::data::v2::Feed::IEX),
                            ..Default::default()
                        }
                        .init(alpaca_symbol.as_ref(), start, now, alpaca_bars::TimeFrame::OneMinute);

                        let fetch_start = std::time::Instant::now();
                        let result = client.issue::<alpaca_bars::List>(&request).await;
                        let fetch_elapsed = fetch_start.elapsed().as_secs_f64();

                        match result {
                            Ok(result) => {
                                crate::metrics::record_poll_latency(&orig_symbol, "success", fetch_elapsed);

                                if let Some(bar) = result.bars.first() {
                                    let price = match utils::num_to_decimal(&bar.close) {
                                        Ok(p) => p,
                                        Err(e) => {
                                            warn!(error = %e, symbol = %alpaca_symbol, "Price conversion error");
                                            crate::metrics::record_dropped_tick(&orig_symbol, "conversion_error");
                                            return;
                                        }
                                    };
                                    let timestamp = bar.time.timestamp_millis();

                                    debug!(symbol = %alpaca_symbol, price = %price, "Price update");

                                    let market_data = MarketData {
                                        symbol: orig_symbol.to_string(),
                                        instrument_id: Some(ALPACA_INSTRUMENT_ID.to_owned()),
                                        price,
                                        timestamp,
                                    };

                                    if sender.send(market_data).await.is_err() {
                                        // Channel receiver dropped
                                    } else {
                                        crate::metrics::record_ws_tick(&orig_symbol);
                                    }
                                }
                            }
                            Err(e) => {
                                crate::metrics::record_poll_latency(&orig_symbol, "error", fetch_elapsed);
                                warn!(symbol = %alpaca_symbol, error = %e, "Failed to fetch bar");
                                crate::metrics::record_dropped_tick(&orig_symbol, "fetch_error");
                            }
                        }
                    }
                })
                .buffer_unordered(MAX_CONCURRENT_POLLS);

            // Drive the stream to completion
            while stream.next().await.is_some() {
                // Check for channel closure inside the loop to fail fast(er)
                if sender.is_closed() {
                    info!("Channel closed, stopping polling");
                    return Ok(());
                }
            }

            // Adaptive Sleep:
            // Calculate how long the work took
            let elapsed = cycle_start.elapsed();
            let target_interval = Duration::from_secs(self.poll_interval_secs);

            if elapsed < target_interval {
                // If we finished early, sleep for the remainder to maintain the target interval
                tokio::time::sleep(target_interval - elapsed).await;
            } else {
                // If we took longer (e.g. strict rate limits for many symbols),
                // we log a warning (once per minute to avoid spam) and start immediately.
                // This prevents the spiral by acknowledging we are maxed out.
                static NEXT_LOG: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(0);
                // Simple rate limited logging
                let now_secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let last = NEXT_LOG.load(std::sync::atomic::Ordering::Relaxed);

                if now_secs > last {
                    warn!(
                        elapsed_ms = elapsed.as_millis(),
                        target_ms = target_interval.as_millis(),
                        "Polling cycle took longer than interval (system saturated, running at max speed)"
                    );
                    NEXT_LOG.store(now_secs + 60, std::sync::atomic::Ordering::Relaxed);
                }
            }

            // Final check
            if sender.is_closed() {
                info!("Channel closed, stopping polling");
                return Ok(());
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
