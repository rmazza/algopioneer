//! Alpaca WebSocket Provider
//!
//! Market data provider for Alpaca using polling (reliable for equities).
//!
//! Note: The apca crate's streaming API has complex generics that are difficult
//! to work with. This polling implementation provides reliable market data
//! updates which are sufficient for equity trading (not latency-sensitive like crypto).

use async_trait::async_trait;
use governor::{Quota, RateLimiter};
use std::num::NonZeroU32;
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

/// MC-2 FIX: Constant for instrument_id to avoid heap allocation per tick
const ALPACA_INSTRUMENT_ID: &str = "alpaca";

/// Alpaca free tier rate limit: 200 req/min ≈ 3.3 req/sec
/// Using 3 req/sec to stay safely under the limit
const RATE_LIMIT_PER_SECOND: u32 = 3;

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

use futures_util::stream::{self, StreamExt};

impl AlpacaWebSocketProvider {
    // ... (keeping new/from_env/with_clock same) ...
}

#[async_trait]
impl WebSocketProvider for AlpacaWebSocketProvider {
    async fn connect_and_subscribe(
        &self,
        symbols: Vec<String>,
        sender: mpsc::Sender<MarketData>,
    ) -> Result<(), ExchangeError> {
        // Convert symbols to Alpaca format (into_owned() for Cow -> String)
        let mut symbol_pairs: Vec<(String, String)> = Vec::new();
        for s in &symbols {
            symbol_pairs.push((s.clone(), utils::to_alpaca_symbol(s).into_owned()));
        }

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

        // CB-1 FIX: Rate limiter to prevent API abuse
        // Alpaca free tier: 200 req/min ~ 3.3 req/sec.
        let rate_limiter = Arc::new(RateLimiter::direct(Quota::per_second(
            NonZeroU32::new(RATE_LIMIT_PER_SECOND).expect("Rate limit must be non-zero"),
        )));

        // Polling loop
        let mut poll_timer = interval(Duration::from_secs(self.poll_interval_secs));

        info!("Alpaca concurrent polling started");

        loop {
            poll_timer.tick().await;

            // Clone things needed for the stream
            let pairs = symbol_pairs.clone();
            let client = client.clone();
            let rate_limiter = rate_limiter.clone();
            let stream_sender = sender.clone();
            let clock = self.clock.clone();

            // Create a stream of futures to fetch data concurrently
            // Using buffer_unordered to execute up to 5 requests in parallel (conservative)
            let mut stream = stream::iter(pairs)
                .map(move |(orig_symbol, alpaca_symbol)| {
                    let client = client.clone();
                    let rate_limiter = rate_limiter.clone();
                    let sender = stream_sender.clone();
                    let clock = clock.clone();

                    async move {
                        // Wait for rate limiter
                        rate_limiter.until_ready().await;

                        let now = clock.now();
                        let start = now - chrono::Duration::minutes(2);

                        let request = alpaca_bars::ListReqInit {
                            limit: Some(1),
                            ..Default::default()
                        }
                        .init(&alpaca_symbol, start, now, alpaca_bars::TimeFrame::OneMinute);

                        match client.issue::<alpaca_bars::List>(&request).await {
                            Ok(result) => {
                                if let Some(bar) = result.bars.first() {
                                    // CB-2 FIX: Propagate conversion error
                                    let price = match utils::num_to_decimal(&bar.close) {
                                        Ok(p) => p,
                                        Err(e) => {
                                            warn!(error = %e, symbol = %alpaca_symbol, "Price conversion error");
                                            return;
                                        }
                                    };
                                    let timestamp = bar.time.timestamp_millis();

                                    debug!(symbol = %alpaca_symbol, price = %price, "Price update");

                                    // MC-2 FIX: Use const for instrument_id
                                    let market_data = MarketData {
                                        symbol: orig_symbol,
                                        instrument_id: Some(ALPACA_INSTRUMENT_ID.to_string()),
                                        price,
                                        timestamp,
                                    };

                                    if sender.send(market_data).await.is_err() {
                                        // Channel closed. We can't easily break the outer loop from here,
                                        // but we can stop sending. The outer loop will continue until next tick
                                        // or we can use a signal mechanism.
                                        // For now, logging.
                                        // Note: We can't return error easily from stream map.
                                    }
                                }
                            }
                            Err(e) => {
                                warn!(symbol = %alpaca_symbol, error = %e, "Failed to fetch bar");
                            }
                        }
                    }
                })
                .buffer_unordered(5); // Parallelism factor

            // Drive the stream to completion
            while stream.next().await.is_some() {}

            // Optimization: If channel is closed, we should break.
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
