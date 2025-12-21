//! Alpaca Client Wrapper
//!
//! A higher-level client that wraps `AlpacaExchangeClient` and provides
//! an interface compatible with `CoinbaseClient` for use with strategies.

use crate::coinbase::AppEnv;
use crate::logging::{TradeRecord, TradeRecorder, TradeSide};
use crate::strategy::dual_leg_trading::{Clock, SystemClock};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use governor::{clock::DefaultClock, state::InMemoryState, Quota, RateLimiter};
use num_decimal::Num;
use rust_decimal::Decimal;
use std::num::NonZeroU32;
use std::sync::{Arc, Once};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use apca::api::v2::asset;
use apca::api::v2::order as alpaca_order;
use apca::api::v2::position as alpaca_position;
use apca::data::v2::bars as alpaca_bars;
use apca::{ApiInfo, Client, RequestError};

use crate::exchange::{Candle, ExchangeConfig, ExchangeError, Executor, Granularity};
use crate::types::OrderSide;

/// Thread-safe initialization guard for environment variables.
/// CB-1 FIX: Environment variables are set exactly once at startup.
static ENV_INIT: Once = Once::new();

/// Alpaca client with same interface as CoinbaseClient
///
/// Provides order execution, position tracking, and candle fetching
/// with paper trading support and rate limiting.
pub struct AlpacaClient {
    client: Arc<RwLock<Client>>,
    mode: AppEnv,
    rate_limiter: Arc<RateLimiter<governor::state::direct::NotKeyed, InMemoryState, DefaultClock>>,
    recorder: Option<Arc<dyn TradeRecorder>>,
    /// CB-3 FIX: Injected clock for deterministic timestamps
    clock: Arc<dyn Clock>,
}

impl AlpacaClient {
    /// Create a new AlpacaClient from environment variables
    ///
    /// # Environment Variables
    /// - `ALPACA_API_KEY`: Your Alpaca API key
    /// - `ALPACA_API_SECRET`: Your Alpaca API secret
    pub fn new(
        env: AppEnv,
        recorder: Option<Arc<dyn TradeRecorder>>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let api_key = std::env::var("ALPACA_API_KEY")
            .map_err(|_| "ALPACA_API_KEY must be set in .env file or environment")?;
        let api_secret = std::env::var("ALPACA_API_SECRET")
            .map_err(|_| "ALPACA_API_SECRET must be set in .env file or environment")?;

        Self::with_credentials_and_clock(api_key, api_secret, env, recorder, Arc::new(SystemClock))
    }

    /// Create a new AlpacaClient with explicit credentials (uses SystemClock)
    pub fn with_credentials(
        api_key: String,
        api_secret: String,
        env: AppEnv,
        recorder: Option<Arc<dyn TradeRecorder>>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_credentials_and_clock(api_key, api_secret, env, recorder, Arc::new(SystemClock))
    }

    /// Create a new AlpacaClient with explicit credentials and clock (CB-3 FIX)
    ///
    /// This constructor allows injecting a custom clock for deterministic testing.
    pub fn with_credentials_and_clock(
        api_key: String,
        api_secret: String,
        env: AppEnv,
        recorder: Option<Arc<dyn TradeRecorder>>,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let is_paper = matches!(env, AppEnv::Paper | AppEnv::Sandbox);

        let base_url = if is_paper {
            "https://paper-api.alpaca.markets"
        } else {
            "https://api.alpaca.markets"
        };

        info!(
            paper = is_paper,
            "Initializing Alpaca client (paper={})", is_paper
        );

        // CB-1 FIX: Set environment variables exactly once using thread-safe guard.
        // This prevents data races in multi-threaded async contexts.
        ENV_INIT.call_once(|| {
            // SAFETY: This runs exactly once before any other code can observe the env vars.
            // The Once guard ensures thread-safe initialization.
            std::env::set_var("APCA_API_KEY_ID", &api_key);
            std::env::set_var("APCA_API_SECRET_KEY", &api_secret);
            std::env::set_var("APCA_API_BASE_URL", base_url);
        });

        let api_info =
            ApiInfo::from_env().map_err(|e| format!("Failed to create Alpaca API info: {}", e))?;

        let client = Client::new(api_info);

        // CB-4 FIX: Use safe NonZeroU32 constant (now safe in const context since Rust 1.70+)
        const RATE_LIMIT_PER_SECOND: NonZeroU32 = NonZeroU32::new(3).unwrap();
        let quota = Quota::per_second(RATE_LIMIT_PER_SECOND);
        let rate_limiter = Arc::new(RateLimiter::direct(quota));

        Ok(Self {
            client: Arc::new(RwLock::new(client)),
            mode: env,
            rate_limiter,
            recorder,
            clock,
        })
    }

    /// Test connection to Alpaca API
    pub async fn test_connection(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.rate_limiter.until_ready().await;

        let client = self.client.read().await;
        let account = client
            .issue::<apca::api::v2::account::Get>(&())
            .await
            .map_err(|e| format!("Failed to connect to Alpaca: {}", e))?;

        info!(
            account_id = %account.id.as_hyphenated(),
            status = ?account.status,
            buying_power = %account.buying_power,
            "Connected to Alpaca"
        );

        Ok(())
    }

    /// Place an order on Alpaca
    pub async fn place_order(
        &self,
        product_id: &str,
        side: &str,
        size: Decimal,
        price: Option<Decimal>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.rate_limiter.until_ready().await;

        let symbol = product_id.replace('-', ""); // Convert BTC-USD -> BTCUSD

        match self.mode {
            AppEnv::Live => {
                info!(
                    symbol = %symbol,
                    side = side,
                    size = %size,
                    price = ?price,
                    "Placing LIVE Alpaca order"
                );

                let order_side = match side.to_lowercase().as_str() {
                    "buy" => alpaca_order::Side::Buy,
                    _ => alpaca_order::Side::Sell,
                };

                // CB-2 FIX: Propagate conversion errors instead of silent fallback
                let qty_num = Self::decimal_to_num(size)
                    .map_err(|e| format!("Failed to convert quantity: {}", e))?;

                let request = match price {
                    Some(limit_price) => {
                        let limit_num = Self::decimal_to_num(limit_price)
                            .map_err(|e| format!("Failed to convert limit price: {}", e))?;
                        alpaca_order::CreateReqInit {
                            type_: alpaca_order::Type::Limit,
                            limit_price: Some(limit_num),
                            time_in_force: alpaca_order::TimeInForce::Day,
                            ..Default::default()
                        }
                        .init(
                            &symbol,
                            order_side,
                            alpaca_order::Amount::quantity(qty_num),
                        )
                    }
                    None => alpaca_order::CreateReqInit {
                        type_: alpaca_order::Type::Market,
                        time_in_force: alpaca_order::TimeInForce::Day,
                        ..Default::default()
                    }
                    .init(
                        &symbol,
                        order_side,
                        alpaca_order::Amount::quantity(qty_num),
                    ),
                };

                let client = self.client.read().await;
                let order = client
                    .issue::<alpaca_order::Create>(&request)
                    .await
                    .map_err(|e| format!("Order failed: {}", e))?;

                info!(
                    order_id = %order.id.as_hyphenated(),
                    symbol = %symbol,
                    status = ?order.status,
                    "Alpaca order created"
                );

                Ok(())
            }
            AppEnv::Paper | AppEnv::Sandbox => {
                let price_str = price
                    .map(|p| p.to_string())
                    .unwrap_or_else(|| "MARKET".to_string());

                info!(
                    product_id = product_id,
                    side = side,
                    size = %size,
                    price = %price_str,
                    "PAPER TRADE executed (Alpaca)"
                );

                // Log using the recorder
                if let Some(ref recorder) = self.recorder {
                    let trade_side = match side.to_lowercase().as_str() {
                        "buy" => TradeSide::Buy,
                        _ => TradeSide::Sell,
                    };

                    // CB-3 FIX: Use injected clock for deterministic timestamps
                    let record = TradeRecord::with_timestamp(
                        product_id.to_string(),
                        trade_side,
                        size,
                        price,
                        true,
                        self.clock.now(),
                    );

                    if let Err(e) = recorder.record(&record).await {
                        error!("Failed to record paper trade: {}", e);
                    }
                } else {
                    warn!("Paper trading without recorder - trades not being recorded");
                }

                Ok(())
            }
        }
    }

    /// Get position for a symbol
    pub async fn get_position(
        &self,
        product_id: &str,
    ) -> Result<Decimal, Box<dyn std::error::Error + Send + Sync>> {
        self.rate_limiter.until_ready().await;

        let symbol = product_id.replace('-', "");

        match self.mode {
            AppEnv::Live => {
                let client = self.client.read().await;
                let sym = asset::Symbol::Sym(symbol.clone());

                match client.issue::<alpaca_position::Get>(&sym).await {
                    Ok(position) => {
                        // CB-2 FIX: Propagate conversion error
                        let qty = Self::num_to_decimal(&position.quantity)
                            .map_err(|e| format!("Failed to convert position quantity: {}", e))?;
                        debug!(symbol = %symbol, quantity = %qty, "Position found");
                        Ok(qty)
                    }
                    Err(e) => {
                        if let RequestError::Endpoint(alpaca_position::GetError::NotFound(_)) = e {
                            debug!(symbol = %symbol, "No position found");
                            Ok(Decimal::ZERO)
                        } else {
                            Err(format!("Failed to get position: {}", e).into())
                        }
                    }
                }
            }
            AppEnv::Paper | AppEnv::Sandbox => Ok(Decimal::ZERO),
        }
    }

    /// Get candles for a symbol
    pub async fn get_product_candles(
        &mut self,
        product_id: &str,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
        granularity: Granularity,
    ) -> Result<Vec<Candle>, Box<dyn std::error::Error>> {
        self.get_product_candles_paginated(product_id, start, end, granularity)
            .await
    }

    /// Get candles with pagination
    pub async fn get_product_candles_paginated(
        &mut self,
        product_id: &str,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
        granularity: Granularity,
    ) -> Result<Vec<Candle>, Box<dyn std::error::Error>> {
        self.rate_limiter.until_ready().await;

        let symbol = product_id.replace('-', "");
        let timeframe = Self::granularity_to_timeframe(granularity);

        info!(
            symbol = %symbol,
            start = %start,
            end = %end,
            timeframe = ?timeframe,
            "Fetching Alpaca bars"
        );

        let client = self.client.read().await;

        let request = alpaca_bars::ListReqInit {
            limit: Some(10000),
            ..Default::default()
        }
        .init(&symbol, *start, *end, timeframe);

        let bars_result = client
            .issue::<alpaca_bars::List>(&request)
            .await
            .map_err(|e| format!("Failed to fetch bars: {}", e))?;

        // CB-2 FIX: Collect candles with proper error handling for conversions
        let candles: Result<Vec<Candle>, Box<dyn std::error::Error>> = bars_result
            .bars
            .iter()
            .map(|bar| {
                Ok(Candle {
                    timestamp: bar.time,
                    open: Self::num_to_decimal(&bar.open)
                        .map_err(|e| format!("Failed to convert open: {}", e))?,
                    high: Self::num_to_decimal(&bar.high)
                        .map_err(|e| format!("Failed to convert high: {}", e))?,
                    low: Self::num_to_decimal(&bar.low)
                        .map_err(|e| format!("Failed to convert low: {}", e))?,
                    close: Self::num_to_decimal(&bar.close)
                        .map_err(|e| format!("Failed to convert close: {}", e))?,
                    volume: Decimal::from(bar.volume),
                })
            })
            .collect();

        let candles = candles?;

        info!(
            symbol = %symbol,
            bars_count = candles.len(),
            "Fetched Alpaca bars"
        );

        Ok(candles)
    }

    // Helper functions

    /// CB-2 FIX: Convert Decimal to Num with proper error handling.
    /// Returns ExchangeError instead of silently falling back to zero.
    fn decimal_to_num(d: Decimal) -> Result<Num, ExchangeError> {
        d.to_string().parse::<Num>().map_err(|e| {
            tracing::error!(
                decimal = %d,
                error = %e,
                "Failed to convert Decimal to Num"
            );
            ExchangeError::Other(format!("Decimal to Num conversion failed for {}: {}", d, e))
        })
    }

    /// CB-2 FIX: Convert Num to Decimal with proper error handling.
    /// Returns ExchangeError instead of silently falling back to zero.
    fn num_to_decimal(n: &Num) -> Result<Decimal, ExchangeError> {
        n.to_string().parse::<Decimal>().map_err(|e| {
            tracing::error!(
                num = %n,
                error = %e,
                "Failed to convert Num to Decimal"
            );
            ExchangeError::Other(format!("Num to Decimal conversion failed for {}: {}", n, e))
        })
    }

    /// MC-4 FIX: Convert granularity with warning when falling back.
    fn granularity_to_timeframe(g: Granularity) -> alpaca_bars::TimeFrame {
        match g {
            Granularity::OneMinute => alpaca_bars::TimeFrame::OneMinute,
            Granularity::FiveMinute => {
                warn!(requested = ?g, actual = "1m", "Alpaca does not support 5m bars, using 1m");
                alpaca_bars::TimeFrame::OneMinute
            }
            Granularity::FifteenMinute => {
                warn!(requested = ?g, actual = "1m", "Alpaca does not support 15m bars, using 1m");
                alpaca_bars::TimeFrame::OneMinute
            }
            Granularity::ThirtyMinute => {
                warn!(requested = ?g, actual = "1h", "Alpaca does not support 30m bars, using 1h");
                alpaca_bars::TimeFrame::OneHour
            }
            Granularity::OneHour => alpaca_bars::TimeFrame::OneHour,
            Granularity::TwoHour => {
                warn!(requested = ?g, actual = "1h", "Alpaca does not support 2h bars, using 1h");
                alpaca_bars::TimeFrame::OneHour
            }
            Granularity::SixHour => {
                warn!(requested = ?g, actual = "1h", "Alpaca does not support 6h bars, using 1h");
                alpaca_bars::TimeFrame::OneHour
            }
            Granularity::OneDay => alpaca_bars::TimeFrame::OneDay,
        }
    }
}

// Implement From<ExchangeConfig> for convenience
impl TryFrom<(ExchangeConfig, AppEnv, Option<Arc<dyn TradeRecorder>>)> for AlpacaClient {
    type Error = Box<dyn std::error::Error>;

    fn try_from(
        (config, env, recorder): (ExchangeConfig, AppEnv, Option<Arc<dyn TradeRecorder>>),
    ) -> Result<Self, Self::Error> {
        Self::with_credentials(config.api_key, config.api_secret, env, recorder)
    }
}

// Implement Executor trait for strategy compatibility
#[async_trait]
impl Executor for AlpacaClient {
    async fn execute_order(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: Decimal,
        price: Option<Decimal>,
    ) -> Result<(), ExchangeError> {
        let side_str = match side {
            OrderSide::Buy => "buy",
            OrderSide::Sell => "sell",
        };

        self.place_order(symbol, side_str, quantity, price)
            .await
            .map_err(|e| ExchangeError::Other(e.to_string()))
    }

    async fn get_position(&self, symbol: &str) -> Result<Decimal, ExchangeError> {
        AlpacaClient::get_position(self, symbol)
            .await
            .map_err(|e| ExchangeError::Other(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decimal_num_conversion() {
        let d = Decimal::new(12345, 2);
        let n = AlpacaClient::decimal_to_num(d).expect("conversion should succeed");
        assert_eq!(n.to_string(), "123.45");

        let d2 = AlpacaClient::num_to_decimal(&n).expect("conversion should succeed");
        assert_eq!(d2, d);
    }
}
