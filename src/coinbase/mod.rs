pub mod market_data_provider;
pub mod websocket;
use crate::sandbox;
use cbadv::models::product::{Candle, ProductCandleQuery};
use cbadv::time::Granularity;
use cbadv::{RestClient, RestClientBuilder};
use chrono::{DateTime, Utc};
use std::env;
use std::sync::Arc;
// AS5: Rate limiting
use governor::{clock::DefaultClock, state::InMemoryState, Quota, RateLimiter};
use std::num::NonZeroU32;

#[derive(Clone, Copy)]
pub enum AppEnv {
    Live,
    Sandbox,
    Paper,
}

pub struct CoinbaseClient {
    client: RestClient,
    mode: AppEnv,
    // AS5: Rate limiter for API calls (10 req/sec for Coinbase Advanced Trade)
    rate_limiter: Arc<RateLimiter<governor::state::direct::NotKeyed, InMemoryState, DefaultClock>>,
}

impl CoinbaseClient {
    /// Creates a new CoinbaseClient.
    ///
    /// It initializes the connection to the Coinbase Advanced Trade API
    /// using credentials from environment variables.
    ///
    /// # Errors
    /// Returns an error if COINBASE_API_KEY or COINBASE_API_SECRET environment
    /// variables are not set, or if the REST client fails to build.
    pub fn new(env: AppEnv) -> Result<Self, Box<dyn std::error::Error>> {
        // Retrieve API Key and Secret from environment variables
        let api_key = std::env::var("COINBASE_API_KEY")
            .map_err(|_| "COINBASE_API_KEY must be set in .env file or environment")?;
        let api_secret = std::env::var("COINBASE_API_SECRET")
            .map_err(|_| "COINBASE_API_SECRET must be set in .env file or environment")?;

        // Build the REST client, wiring API credentials from the environment
        let client: RestClient = RestClientBuilder::new()
            .with_authentication(&api_key, &api_secret)
            .build()?;

        // AS5: Initialize rate limiter (Coinbase Advanced Trade: 10 requests/second)
        // SAFETY: 10 is a non-zero constant, this will never fail
        const RATE_LIMIT: u32 = 10;
        let quota = Quota::per_second(
            NonZeroU32::new(RATE_LIMIT).expect("RATE_LIMIT is non-zero constant"),
        );
        let rate_limiter = Arc::new(RateLimiter::direct(quota));

        Ok(Self {
            client,
            mode: env,
            rate_limiter,
        })
    }

    /// Pings the Coinbase server to test the API connection.
    pub async fn test_connection(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        match self.client.public.time().await {
            Ok(server_time) => {
                tracing::info!("Successfully connected to Coinbase!");
                tracing::info!("Server Time (ISO): {}", server_time.iso);
                Ok(())
            }
            Err(e) => {
                tracing::error!("Error during test API call: {}", e);
                Err(Box::new(e) as Box<dyn std::error::Error>)
            }
        }
    }

    /// Places an order.
    pub async fn place_order(
        &self,
        product_id: &str,
        side: &str,
        size: rust_decimal::Decimal,
        price: Option<rust_decimal::Decimal>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // AS5: Wait for rate limit
        self.rate_limiter.until_ready().await;

        match self.mode {
            AppEnv::Live => {
                tracing::info!(
                    "-- Live Mode: Placing order for {} {} of {} --",
                    side,
                    size,
                    product_id
                );
                // Here you would add the actual call to the Coinbase API to place an order
                // For now, we just print a message.
                Ok(())
            }
            AppEnv::Sandbox => {
                let trade_details = format!("{},{},{}", product_id, side, size);
                sandbox::save_trade(&trade_details)
            }
            AppEnv::Paper => {
                let price_str = price
                    .map(|p| p.to_string())
                    .unwrap_or_else(|| "MARKET".to_string());
                let msg = format!(
                    "-- PAPER TRADE: {} {} of {} @ {} --",
                    side, size, product_id, price_str
                );
                tracing::info!("{}", msg);

                // Log to CSV
                use std::fs::OpenOptions;
                use std::io::Write;

                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("paper_trades.csv")?;

                writeln!(
                    file,
                    "{},{},{},{},{}",
                    Utc::now(),
                    product_id,
                    side,
                    size,
                    price_str
                )?;
                Ok(())
            }
        }
    }

    /// Gets product candles (candlestick data).
    pub async fn get_product_candles(
        &mut self,
        product_id: &str,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
        granularity: Granularity,
    ) -> Result<Vec<Candle>, Box<dyn std::error::Error>> {
        // AS5: Wait for rate limit
        self.rate_limiter.until_ready().await;

        let start_timestamp = start.timestamp() as u64;
        let end_timestamp = end.timestamp() as u64;

        let query = ProductCandleQuery {
            start: start_timestamp,
            end: end_timestamp,
            granularity,
            limit: 300,
        };

        let candles = self.client.product.candles(product_id, &query).await?;
        Ok(candles)
    }

    /// Gets product candles with automatic pagination (bypasses 300 limit).
    /// Fetches multiple batches and stitches them together.
    pub async fn get_product_candles_paginated(
        &mut self,
        product_id: &str,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
        granularity: Granularity,
    ) -> Result<Vec<Candle>, Box<dyn std::error::Error>> {
        use chrono::Duration as ChronoDuration;

        let mut all_candles = Vec::new();
        let mut current_start = *start;

        // Calculate hours per candle based on granularity
        let hours_per_candle: f64 = match granularity {
            Granularity::OneMinute => 1.0 / 60.0,
            Granularity::FiveMinute => 5.0 / 60.0,
            Granularity::FifteenMinute => 0.25,
            Granularity::ThirtyMinute => 0.5,
            Granularity::OneHour => 1.0,
            Granularity::TwoHour => 2.0,
            Granularity::SixHour => 6.0,
            Granularity::OneDay => 24.0,
            _ => 1.0, // Default to 1 hour for unknown
        };

        // 300 candles per batch
        let batch_hours = (300.0 * hours_per_candle) as i64;
        let batch_duration = ChronoDuration::hours(batch_hours);

        tracing::debug!(
            product_id = product_id,
            batch_hours = batch_hours,
            "Starting paginated candle fetch"
        );

        let mut batch_count = 0;
        while current_start < *end {
            let batch_end = (current_start + batch_duration).min(*end);

            let candles = self
                .get_product_candles(product_id, &current_start, &batch_end, granularity.clone())
                .await?;

            if candles.is_empty() {
                break;
            }

            batch_count += 1;
            tracing::debug!(
                batch = batch_count,
                candles = candles.len(),
                "Fetched candle batch"
            );

            all_candles.extend(candles);
            current_start = batch_end;
        }

        tracing::info!(
            product_id = product_id,
            total_candles = all_candles.len(),
            batches = batch_count,
            "Paginated fetch complete"
        );

        Ok(all_candles)
    }

    /// Gets the current position for a symbol.
    /// In paper/sandbox mode returns zero, in live mode would query actual positions.
    pub async fn get_position(
        &self,
        _product_id: &str,
    ) -> Result<rust_decimal::Decimal, Box<dyn std::error::Error + Send + Sync>> {
        // AS5: Wait for rate limit
        self.rate_limiter.until_ready().await;

        match self.mode {
            AppEnv::Live => Ok(rust_decimal::Decimal::ZERO),
            AppEnv::Sandbox | AppEnv::Paper => Ok(rust_decimal::Decimal::ZERO),
        }
    }
}
