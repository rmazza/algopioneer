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
use rust_decimal::Decimal;
use std::num::NonZeroU32;
use std::sync::Arc;

use tracing::{debug, error, info};

use apca::api::v2::asset;
use apca::api::v2::order as alpaca_order;
use apca::api::v2::position as alpaca_position;
use apca::data::v2::bars as alpaca_bars;
use apca::{ApiInfo, Client, RequestError};

use crate::exchange::alpaca::utils;
use crate::exchange::{Candle, ExchangeConfig, ExchangeError, Executor, Granularity};
use crate::types::OrderSide;

/// Alpaca client with same interface as CoinbaseClient
///
/// Provides order execution, position tracking, and candle fetching
/// with paper trading support and rate limiting.
pub struct AlpacaClient {
    client: Arc<Client>,
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
    ) -> Result<Self, ExchangeError> {
        let api_key = std::env::var("ALPACA_API_KEY").map_err(|_| {
            ExchangeError::Configuration(
                "ALPACA_API_KEY must be set in .env file or environment".to_string(),
            )
        })?;
        let api_secret = std::env::var("ALPACA_API_SECRET").map_err(|_| {
            ExchangeError::Configuration(
                "ALPACA_API_SECRET must be set in .env file or environment".to_string(),
            )
        })?;

        Self::with_credentials_and_clock(api_key, api_secret, env, recorder, Arc::new(SystemClock))
            .map_err(|e| ExchangeError::Configuration(e.to_string())) // Simplify error mapping for now, ideally with_credentials_and_clock should return ExchangeError
    }

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

        // NOTE: We construct ApiInfo directly to avoid mutating global environment variables
        let api_info = ApiInfo::from_parts(base_url, &api_key, &api_secret)
            .map_err(|e| format!("Failed to create Alpaca API info: {}", e))?;

        let client = Client::new(api_info);

        // MC-1 FIX: Safe unwrapping of hardcoded constant
        let quota = Quota::per_second(NonZeroU32::new(3).unwrap_or(NonZeroU32::MIN));
        let rate_limiter = Arc::new(RateLimiter::direct(quota));

        Ok(Self {
            client: Arc::new(client),
            mode: env,
            rate_limiter,
            recorder,
            clock,
        })
    }

    /// Create from ExchangeConfig (for factory/DI pattern - MC-3 FIX)
    ///
    /// Used by `create_exchange_client` factory function.
    pub fn from_config(config: ExchangeConfig) -> Result<Self, ExchangeError> {
        let env = if config.sandbox {
            AppEnv::Paper
        } else {
            AppEnv::Live
        };

        Self::with_credentials(config.api_key, config.api_secret, env, None)
            .map_err(|e| ExchangeError::Configuration(e.to_string()))
    }

    /// Test connection to Alpaca API
    ///
    /// MC-3 FIX: Uses the ExchangeClient trait implementation to avoid duplication
    pub async fn test_connection(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        <Self as crate::exchange::ExchangeClient>::test_connection(self)
            .await
            .map_err(|e| e.to_string().into())
    }

    /// Place an order on Alpaca
    pub async fn place_order(
        &self,
        product_id: &str,
        side: &str,
        size: Decimal,
        price: Option<Decimal>,
    ) -> Result<(), ExchangeError> {
        self.rate_limiter.until_ready().await;

        match self.mode {
            AppEnv::Live => {
                let symbol = utils::to_alpaca_symbol(product_id); // Convert BTC-USD -> BTCUSD
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
                let qty_num = utils::decimal_to_num(size).map_err(|e| {
                    ExchangeError::Other(format!("Failed to convert quantity: {}", e))
                })?;

                let request = match price {
                    Some(limit_price) => {
                        let limit_num = utils::decimal_to_num(limit_price).map_err(|e| {
                            ExchangeError::Other(format!("Failed to convert limit price: {}", e))
                        })?;
                        alpaca_order::CreateReqInit {
                            type_: alpaca_order::Type::Limit,
                            limit_price: Some(limit_num),
                            time_in_force: alpaca_order::TimeInForce::Day,
                            ..Default::default()
                        }
                        .init(
                            symbol.as_ref(),
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
                        symbol.as_ref(),
                        order_side,
                        alpaca_order::Amount::quantity(qty_num),
                    ),
                };

                let client = &self.client;
                let order = client
                    .issue::<alpaca_order::Create>(&request)
                    .await
                    .map_err(|e| ExchangeError::OrderRejected(format!("Order failed: {}", e)))?;

                info!(
                    order_id = %order.id.as_hyphenated(),
                    symbol = %symbol,
                    status = ?order.status,
                    "Alpaca order created"
                );

                Ok(())
            }
            AppEnv::Paper | AppEnv::Sandbox => {
                // In Paper/Sandbox, we DO execute the order against the Paper API.
                // The client is already configured with the Paper URL in the constructor.
                // We reuse the exact same logic as Live, but we also keep the CSV recording
                // for local verification/debugging if a recorder is configured.

                let symbol = utils::to_alpaca_symbol(product_id); // Convert BTC-USD -> BTCUSD
                info!(
                    symbol = %symbol,
                    side = side,
                    size = %size,
                    price = ?price,
                    "Placing PAPER Alpaca order"
                );

                let order_side = match side.to_lowercase().as_str() {
                    "buy" => alpaca_order::Side::Buy,
                    _ => alpaca_order::Side::Sell,
                };

                // Propagate conversion errors
                let qty_num = utils::decimal_to_num(size).map_err(|e| {
                    ExchangeError::Other(format!("Failed to convert quantity: {}", e))
                })?;

                let request = match price {
                    Some(limit_price) => {
                        let limit_num = utils::decimal_to_num(limit_price).map_err(|e| {
                            ExchangeError::Other(format!("Failed to convert limit price: {}", e))
                        })?;
                        alpaca_order::CreateReqInit {
                            type_: alpaca_order::Type::Limit,
                            limit_price: Some(limit_num),
                            time_in_force: alpaca_order::TimeInForce::Day,
                            ..Default::default()
                        }
                        .init(
                            symbol.as_ref(),
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
                        symbol.as_ref(),
                        order_side,
                        alpaca_order::Amount::quantity(qty_num),
                    ),
                };

                let client = &self.client;
                let order = client
                    .issue::<alpaca_order::Create>(&request)
                    .await
                    .map_err(|e| {
                        ExchangeError::OrderRejected(format!("Paper Order failed: {}", e))
                    })?;

                info!(
                    order_id = %order.id.as_hyphenated(),
                    symbol = %symbol,
                    status = ?order.status,
                    "Alpaca PAPER order created successfully"
                );

                // --- CSV Recording for Paper Mode ---
                if let Some(ref recorder) = self.recorder {
                    let trade_side = match side.to_lowercase().as_str() {
                        "buy" => TradeSide::Buy,
                        _ => TradeSide::Sell,
                    };

                    // Use injected clock for deterministic timestamps
                    let record = TradeRecord::with_timestamp(
                        product_id.to_string(),
                        trade_side,
                        size,
                        price, // Note: For market orders, this might be None. Ideally we'd valid fill price later.
                        true,
                        self.clock.now(),
                    );

                    if let Err(e) = recorder.record(&record).await {
                        error!("Failed to record paper trade to CSV: {}", e);
                    }
                }

                Ok(())
            }
        }
    }

    /// Get position for a symbol
    pub async fn get_position(&self, product_id: &str) -> Result<Decimal, ExchangeError> {
        self.rate_limiter.until_ready().await;

        match self.mode {
            // Paper/Sandbox now behaves exactly like Live for querying positions
            // because we are using the real Paper/Sandbox API.
            AppEnv::Live | AppEnv::Paper | AppEnv::Sandbox => {
                let symbol = utils::to_alpaca_symbol(product_id);
                let client = &self.client;
                let sym = asset::Symbol::Sym(symbol.clone().into_owned());

                match client.issue::<alpaca_position::Get>(&sym).await {
                    Ok(position) => {
                        // CB-2 FIX: Propagate conversion error
                        let qty = utils::num_to_decimal(&position.quantity).map_err(|e| {
                            ExchangeError::Other(format!(
                                "Failed to convert position quantity: {}",
                                e
                            ))
                        })?;
                        debug!(symbol = %symbol, quantity = %qty, "Position found");
                        Ok(qty)
                    }
                    Err(e) => {
                        if let RequestError::Endpoint(alpaca_position::GetError::NotFound(_)) = e {
                            debug!(symbol = %symbol, "No position found");
                            Ok(Decimal::ZERO)
                        } else {
                            Err(ExchangeError::Network(format!(
                                "Failed to get position: {}",
                                e
                            )))
                        }
                    }
                }
            } // Removed dedicated Paper/Sandbox block because it's now merged with Live
        }
    }

    /// Get candles for a symbol
    ///
    /// MC-1 FIX: Delegates to ExchangeClient trait to avoid code duplication
    pub async fn get_product_candles(
        &mut self,
        product_id: &str,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
        granularity: Granularity,
    ) -> Result<Vec<Candle>, ExchangeError> {
        <Self as crate::exchange::ExchangeClient>::get_candles(
            self,
            product_id,
            start,
            end,
            granularity,
        )
        .await
    }

    /// Get candles with pagination
    ///
    /// MC-1 FIX: Delegates to ExchangeClient trait to avoid code duplication
    pub async fn get_product_candles_paginated(
        &mut self,
        product_id: &str,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
        granularity: Granularity,
    ) -> Result<Vec<Candle>, ExchangeError> {
        <Self as crate::exchange::ExchangeClient>::get_candles_paginated(
            self,
            product_id,
            start,
            end,
            granularity,
        )
        .await
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

        self.place_order(symbol, side_str, quantity, price).await
    }

    async fn get_position(&self, symbol: &str) -> Result<Decimal, ExchangeError> {
        AlpacaClient::get_position(self, symbol).await
    }
}

// MC-3 FIX: Implement ExchangeClient trait for unified client (DRY principle)
// This consolidates functionality from the deleted AlpacaExchangeClient.
#[async_trait]
impl crate::exchange::ExchangeClient for AlpacaClient {
    async fn test_connection(&mut self) -> Result<(), ExchangeError> {
        self.rate_limiter.until_ready().await;

        let client = &self.client;
        let account = client
            .issue::<apca::api::v2::account::Get>(&())
            .await
            .map_err(|e| ExchangeError::Network(format!("Failed to connect to Alpaca: {}", e)))?;

        info!(
            account_id = %account.id.as_hyphenated(),
            status = ?account.status,
            buying_power = %account.buying_power,
            "Connected to Alpaca"
        );

        Ok(())
    }

    async fn get_candles(
        &mut self,
        product_id: &str,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
        granularity: Granularity,
    ) -> Result<Vec<Candle>, ExchangeError> {
        self.get_candles_paginated(product_id, start, end, granularity)
            .await
    }

    async fn get_candles_paginated(
        &mut self,
        product_id: &str,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
        granularity: Granularity,
    ) -> Result<Vec<Candle>, ExchangeError> {
        self.rate_limiter.until_ready().await;

        let symbol = utils::to_alpaca_symbol(product_id);
        let timeframe = utils::granularity_to_timeframe(granularity);

        info!(
            symbol = %symbol,
            start = %start,
            end = %end,
            timeframe = ?timeframe,
            "Fetching Alpaca bars"
        );

        let client = &self.client;

        let request = alpaca_bars::ListReqInit {
            limit: Some(10000),
            ..Default::default()
        }
        .init(symbol.as_ref(), *start, *end, timeframe);

        let bars_result = client
            .issue::<alpaca_bars::List>(&request)
            .await
            .map_err(|e| ExchangeError::Other(format!("Failed to fetch bars: {}", e)))?;

        // Collect candles with proper error handling for conversions
        let candles: Result<Vec<Candle>, ExchangeError> = bars_result
            .bars
            .iter()
            .map(|bar| {
                Ok(Candle {
                    timestamp: bar.time,
                    open: utils::num_to_decimal(&bar.open)?,
                    high: utils::num_to_decimal(&bar.high)?,
                    low: utils::num_to_decimal(&bar.low)?,
                    close: utils::num_to_decimal(&bar.close)?,
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

    fn normalize_symbol(&self, symbol: &str) -> String {
        utils::to_alpaca_symbol(symbol).into_owned()
    }

    fn exchange_id(&self) -> crate::exchange::ExchangeId {
        crate::exchange::ExchangeId::Alpaca
    }
}

// Implement DiscoveryDataSource for pair discovery pipeline
use crate::discovery::DiscoveryDataSource;

#[async_trait]
impl DiscoveryDataSource for AlpacaClient {
    async fn fetch_candles_hourly(
        &mut self,
        symbol: &str,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<(i64, Decimal)>, Box<dyn std::error::Error + Send + Sync>> {
        self.rate_limiter.until_ready().await;

        let alpaca_symbol = utils::to_alpaca_symbol(symbol);
        let timeframe = alpaca_bars::TimeFrame::OneHour;

        info!(
            symbol = %alpaca_symbol,
            start = %start,
            end = %end,
            timeframe = "1h",
            "Fetching Alpaca candles for discovery"
        );

        let client = &self.client;

        // Use IEX feed for free tier access (SIP requires paid subscription)
        let request = alpaca_bars::ListReqInit {
            limit: Some(10000),
            feed: Some(apca::data::v2::Feed::IEX),
            ..Default::default()
        }
        .init(alpaca_symbol.as_ref(), start, end, timeframe);

        let bars_result = client
            .issue::<alpaca_bars::List>(&request)
            .await
            .map_err(|e| format!("Failed to fetch bars: {}", e))?;

        // Convert to (timestamp_seconds, close_price) tuples
        let result: Result<Vec<(i64, Decimal)>, _> = bars_result
            .bars
            .iter()
            .map(|bar| {
                let close = utils::num_to_decimal(&bar.close)?;
                Ok((bar.time.timestamp(), close))
            })
            .collect();

        result.map_err(|e: ExchangeError| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }

    async fn fetch_candles_daily(
        &mut self,
        symbol: &str,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Result<Vec<(i64, Decimal)>, Box<dyn std::error::Error + Send + Sync>> {
        self.rate_limiter.until_ready().await;

        let alpaca_symbol = utils::to_alpaca_symbol(symbol);
        let timeframe = alpaca_bars::TimeFrame::OneDay;

        info!(
            symbol = %alpaca_symbol,
            start = %start,
            end = %end,
            timeframe = "1d",
            "Fetching Alpaca daily candles for discovery"
        );

        let client = &self.client;

        // Use IEX feed for free tier access
        let request = alpaca_bars::ListReqInit {
            limit: Some(10000),
            feed: Some(apca::data::v2::Feed::IEX),
            ..Default::default()
        }
        .init(alpaca_symbol.as_ref(), start, end, timeframe);

        let bars_result = client
            .issue::<alpaca_bars::List>(&request)
            .await
            .map_err(|e| format!("Failed to fetch daily bars: {}", e))?;

        let result: Result<Vec<(i64, Decimal)>, _> = bars_result
            .bars
            .iter()
            .map(|bar| {
                let close = utils::num_to_decimal(&bar.close)?;
                Ok((bar.time.timestamp(), close))
            })
            .collect();

        result.map_err(|e: ExchangeError| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }

    /// Alpaca free tier has limited hourly data; prefer daily bars for discovery
    fn prefers_daily_bars(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {

    // decimal_to_num and num_to_decimal tests moved to utils.rs
}
