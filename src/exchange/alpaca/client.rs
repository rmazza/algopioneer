//! Alpaca Exchange Client Implementation
//!
//! Provides order execution and market data access for US stocks/ETFs.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

use apca::api::v2::asset;
use apca::api::v2::order as alpaca_order;
use apca::api::v2::position as alpaca_position;
use apca::data::v2::bars as alpaca_bars;
use apca::{ApiInfo, Client, RequestError};

use crate::exchange::alpaca::utils;
use crate::exchange::{
    Candle, ExchangeClient, ExchangeConfig, ExchangeError, ExchangeId, Executor, Granularity,
};
use crate::strategy::dual_leg_trading::OrderSide;

/// Alpaca exchange client for US equities
///
/// Implements the `Executor` and `ExchangeClient` traits for
/// seamless integration with existing trading strategies.
pub struct AlpacaExchangeClient {
    config: ExchangeConfig,
    client: Arc<RwLock<Client>>,
}

impl AlpacaExchangeClient {
    /// Create a new Alpaca client from configuration
    ///
    /// # Environment Variables
    /// The apca crate reads from these env vars by default:
    /// - `APCA_API_KEY_ID`: Your Alpaca API key
    /// - `APCA_API_SECRET_KEY`: Your Alpaca API secret
    /// - `APCA_API_BASE_URL`: API endpoint (paper or live)
    ///
    /// This implementation also supports:
    /// - `ALPACA_API_KEY`: Alternative key name
    /// - `ALPACA_API_SECRET`: Alternative secret name
    ///
    /// # Paper Trading
    /// Set `sandbox: true` in config to use paper trading endpoint
    pub fn new(config: ExchangeConfig) -> Result<Self, ExchangeError> {
        info!(
            sandbox = config.sandbox,
            "Initializing Alpaca client (paper={})", config.sandbox
        );

        // Build API info from config
        let base_url = if config.sandbox {
            "https://paper-api.alpaca.markets"
        } else {
            "https://api.alpaca.markets"
        };

        // Try from_parts constructor
        let api_info =
            ApiInfo::from_parts(base_url, &config.api_key, &config.api_secret).map_err(|e| {
                ExchangeError::Configuration(format!("Failed to create ApiInfo: {}", e))
            })?;

        let client = Client::new(api_info);

        Ok(Self {
            config,
            client: Arc::new(RwLock::new(client)),
        })
    }
}

#[async_trait]
impl Executor for AlpacaExchangeClient {
    async fn execute_order(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: Decimal,
        price: Option<Decimal>,
    ) -> Result<(), ExchangeError> {
        let alpaca_symbol = utils::to_alpaca_symbol(symbol);

        // Convert side
        let order_side = match side {
            OrderSide::Buy => alpaca_order::Side::Buy,
            OrderSide::Sell => alpaca_order::Side::Sell,
        };

        // Convert quantity to Num (apca uses num_decimal)
        let qty_num = utils::decimal_to_num(quantity)?;

        info!(
            symbol = %alpaca_symbol,
            side = ?side,
            quantity = %quantity,
            price = ?price,
            sandbox = self.config.sandbox,
            "Submitting Alpaca order"
        );

        // Build order request based on order type
        let request = match price {
            Some(limit_price) => {
                let limit_num = utils::decimal_to_num(limit_price)?;
                // Limit order
                alpaca_order::CreateReqInit {
                    type_: alpaca_order::Type::Limit,
                    limit_price: Some(limit_num),
                    time_in_force: alpaca_order::TimeInForce::Day,
                    ..Default::default()
                }
                .init(
                    &alpaca_symbol,
                    order_side,
                    alpaca_order::Amount::quantity(qty_num),
                )
            }
            None => {
                // Market order
                alpaca_order::CreateReqInit {
                    type_: alpaca_order::Type::Market,
                    time_in_force: alpaca_order::TimeInForce::Day,
                    ..Default::default()
                }
                .init(
                    &alpaca_symbol,
                    order_side,
                    alpaca_order::Amount::quantity(qty_num),
                )
            }
        };

        // Execute order
        let client = self.client.read().await;
        let order = client
            .issue::<alpaca_order::Create>(&request)
            .await
            .map_err(|e| {
                let err_str = e.to_string();
                if err_str.contains("insufficient") {
                    ExchangeError::OrderRejected(format!("Insufficient funds: {}", err_str))
                } else if err_str.contains("not found") || err_str.contains("invalid symbol") {
                    ExchangeError::OrderRejected(format!(
                        "Invalid symbol {}: {}",
                        alpaca_symbol, err_str
                    ))
                } else {
                    ExchangeError::Other(format!("Order failed: {}", err_str))
                }
            })?;

        info!(
            order_id = %order.id.as_hyphenated(),
            symbol = %alpaca_symbol,
            status = ?order.status,
            "Alpaca order created"
        );

        Ok(())
    }

    async fn get_position(&self, symbol: &str) -> Result<Decimal, ExchangeError> {
        let alpaca_symbol = utils::to_alpaca_symbol(symbol);
        debug!(symbol = %alpaca_symbol, "Fetching Alpaca position");

        let client = self.client.read().await;

        // Create symbol reference for apca
        let sym = asset::Symbol::Sym(alpaca_symbol.clone());

        // Try to get position, return 0 if not found
        match client.issue::<alpaca_position::Get>(&sym).await {
            Ok(position) => {
                let qty = utils::num_to_decimal(&position.quantity)?;
                debug!(symbol = %alpaca_symbol, quantity = %qty, "Position found");
                Ok(qty)
            }
            Err(e) => {
                // Check if it's a "not found" error
                if let RequestError::Endpoint(alpaca_position::GetError::NotFound(_)) = e {
                    debug!(symbol = %alpaca_symbol, "No position found, returning 0");
                    Ok(Decimal::ZERO)
                } else {
                    Err(ExchangeError::Other(format!(
                        "Failed to get position: {}",
                        e
                    )))
                }
            }
        }
    }
}

#[async_trait]
impl ExchangeClient for AlpacaExchangeClient {
    async fn test_connection(&mut self) -> Result<(), ExchangeError> {
        info!("Testing Alpaca API connection");

        let client = self.client.read().await;

        // Test by fetching account info
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
        let alpaca_symbol = utils::to_alpaca_symbol(product_id);
        let timeframe = utils::granularity_to_timeframe(granularity);

        info!(
            symbol = %alpaca_symbol,
            start = %start,
            end = %end,
            timeframe = ?timeframe,
            "Fetching Alpaca bars"
        );

        let client = self.client.read().await;

        // Build bars request - init takes (symbol, start, end, timeframe)
        let request = alpaca_bars::ListReqInit {
            limit: Some(10000),
            ..Default::default()
        }
        .init(&alpaca_symbol, *start, *end, timeframe);

        // Fetch bars
        let bars_result = client
            .issue::<alpaca_bars::List>(&request)
            .await
            .map_err(|e| ExchangeError::Other(format!("Failed to fetch bars: {}", e)))?;

        // Convert to our Candle format
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
            symbol = %alpaca_symbol,
            bars_count = candles.len(),
            "Fetched Alpaca bars"
        );

        Ok(candles)
    }

    fn normalize_symbol(&self, symbol: &str) -> String {
        utils::to_alpaca_symbol(symbol)
    }

    fn exchange_id(&self) -> ExchangeId {
        ExchangeId::Alpaca
    }
}
