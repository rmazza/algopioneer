//! Alpaca Exchange Client Implementation
//!
//! Provides order execution and market data access for US stocks/ETFs.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use std::sync::{Arc, Once};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use apca::api::v2::asset;
use apca::api::v2::order as alpaca_order;
use apca::api::v2::position as alpaca_position;
use apca::data::v2::bars as alpaca_bars;
use apca::{ApiInfo, Client, RequestError};

// Re-export Num from apca's dependency for internal use
use num_decimal::Num;

use crate::exchange::{
    Candle, ExchangeClient, ExchangeConfig, ExchangeError, ExchangeId, Executor, Granularity,
};
use crate::strategy::dual_leg_trading::OrderSide;

/// Thread-safe initialization guard for environment variables.
/// CB-1 FIX: Environment variables are set exactly once at startup.
static CLIENT_ENV_INIT: Once = Once::new();

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

        // CB-1 FIX: Set environment variables exactly once using thread-safe guard.
        // This prevents data races in multi-threaded async contexts.
        CLIENT_ENV_INIT.call_once(|| {
            std::env::set_var("APCA_API_KEY_ID", &config.api_key);
            std::env::set_var("APCA_API_SECRET_KEY", &config.api_secret);
            std::env::set_var("APCA_API_BASE_URL", base_url);
        });

        let api_info = ApiInfo::from_env().map_err(|e| {
            ExchangeError::Configuration(format!("Failed to create Alpaca API info: {}", e))
        })?;

        let client = Client::new(api_info);

        Ok(Self {
            config,
            client: Arc::new(RwLock::new(client)),
        })
    }

    /// Convert internal symbol format to Alpaca format
    ///
    /// Examples:
    /// - "AAPL" -> "AAPL" (no change for stocks)
    /// - "BTC-USD" -> "BTCUSD" (crypto format)
    pub fn to_alpaca_symbol(symbol: &str) -> String {
        // Remove dashes for Alpaca format
        symbol.replace('-', "")
    }

    /// Convert Alpaca symbol back to internal format
    #[allow(dead_code)]
    pub fn from_alpaca_symbol(symbol: &str) -> String {
        // For crypto, insert dash before USD
        if symbol.ends_with("USD") && symbol.len() > 3 {
            let base = &symbol[..symbol.len() - 3];
            format!("{}-USD", base)
        } else {
            symbol.to_string()
        }
    }

    /// CB-2 FIX: Convert Decimal to num_decimal::Num with error handling.
    fn decimal_to_num(d: Decimal) -> Result<Num, ExchangeError> {
        d.to_string().parse::<Num>().map_err(|e| {
            tracing::error!(
                decimal = %d,
                error = %e,
                "Failed to convert Decimal to Num"
            );
            ExchangeError::Other(format!(
                "Decimal to Num conversion failed for '{}': {}",
                d, e
            ))
        })
    }

    /// CB-2 FIX: Convert num_decimal::Num to Decimal with error handling.
    fn num_to_decimal(n: &Num) -> Result<Decimal, ExchangeError> {
        n.to_string().parse::<Decimal>().map_err(|e| {
            tracing::error!(
                num = %n,
                error = %e,
                "Failed to convert Num to Decimal"
            );
            ExchangeError::Other(format!(
                "Num to Decimal conversion failed for '{}': {}",
                n, e
            ))
        })
    }

    /// MC-4 FIX: Convert Granularity to Alpaca TimeFrame with warnings.
    fn granularity_to_timeframe(g: Granularity) -> alpaca_bars::TimeFrame {
        match g {
            Granularity::OneMinute => alpaca_bars::TimeFrame::OneMinute,
            Granularity::FiveMinute => {
                warn!(requested = ?g, actual = "1m", "Alpaca doesn't support 5m bars, using 1m");
                alpaca_bars::TimeFrame::OneMinute
            }
            Granularity::FifteenMinute => {
                warn!(requested = ?g, actual = "1m", "Alpaca doesn't support 15m bars, using 1m");
                alpaca_bars::TimeFrame::OneMinute
            }
            Granularity::ThirtyMinute => {
                warn!(requested = ?g, actual = "1h", "Alpaca doesn't support 30m bars, using 1h");
                alpaca_bars::TimeFrame::OneHour
            }
            Granularity::OneHour => alpaca_bars::TimeFrame::OneHour,
            Granularity::TwoHour => {
                warn!(requested = ?g, actual = "1h", "Alpaca doesn't support 2h bars, using 1h");
                alpaca_bars::TimeFrame::OneHour
            }
            Granularity::SixHour => {
                warn!(requested = ?g, actual = "1h", "Alpaca doesn't support 6h bars, using 1h");
                alpaca_bars::TimeFrame::OneHour
            }
            Granularity::OneDay => alpaca_bars::TimeFrame::OneDay,
        }
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
        let alpaca_symbol = Self::to_alpaca_symbol(symbol);

        // Convert side
        let order_side = match side {
            OrderSide::Buy => alpaca_order::Side::Buy,
            OrderSide::Sell => alpaca_order::Side::Sell,
        };

        // Convert quantity to Num (apca uses num_decimal)
        // CB-2 FIX: Propagate conversion error
        let qty_num = Self::decimal_to_num(quantity)?;

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
                // CB-2 FIX: Propagate conversion error
                let limit_num = Self::decimal_to_num(limit_price)?;
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
        let alpaca_symbol = Self::to_alpaca_symbol(symbol);
        debug!(symbol = %alpaca_symbol, "Fetching Alpaca position");

        let client = self.client.read().await;

        // Create symbol reference for apca
        let sym = asset::Symbol::Sym(alpaca_symbol.clone());

        // Try to get position, return 0 if not found
        match client.issue::<alpaca_position::Get>(&sym).await {
            Ok(position) => {
                // CB-2 FIX: Propagate conversion error
                let qty = Self::num_to_decimal(&position.quantity)?;
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
        let alpaca_symbol = Self::to_alpaca_symbol(product_id);
        let timeframe = Self::granularity_to_timeframe(granularity);

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
        // CB-2 FIX: Collect with error handling for conversions
        let candles: Result<Vec<Candle>, ExchangeError> = bars_result
            .bars
            .iter()
            .map(|bar| {
                Ok(Candle {
                    timestamp: bar.time,
                    open: Self::num_to_decimal(&bar.open)?,
                    high: Self::num_to_decimal(&bar.high)?,
                    low: Self::num_to_decimal(&bar.low)?,
                    close: Self::num_to_decimal(&bar.close)?,
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
        Self::to_alpaca_symbol(symbol)
    }

    fn exchange_id(&self) -> ExchangeId {
        ExchangeId::Alpaca
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_conversion() {
        assert_eq!(AlpacaExchangeClient::to_alpaca_symbol("AAPL"), "AAPL");
        assert_eq!(AlpacaExchangeClient::to_alpaca_symbol("BTC-USD"), "BTCUSD");
        assert_eq!(
            AlpacaExchangeClient::from_alpaca_symbol("BTCUSD"),
            "BTC-USD"
        );
        assert_eq!(AlpacaExchangeClient::from_alpaca_symbol("AAPL"), "AAPL");
    }

    #[test]
    fn test_decimal_to_num_conversion() {
        let d = Decimal::new(12345, 2); // 123.45
        let n = AlpacaExchangeClient::decimal_to_num(d).expect("conversion should succeed");
        assert_eq!(n.to_string(), "123.45");
    }

    #[test]
    fn test_num_to_decimal_conversion() {
        let n: Num = "123.45".parse().unwrap();
        let d = AlpacaExchangeClient::num_to_decimal(&n).expect("conversion should succeed");
        assert_eq!(d, Decimal::new(12345, 2));
    }

    #[test]
    fn test_granularity_to_timeframe() {
        assert!(matches!(
            AlpacaExchangeClient::granularity_to_timeframe(Granularity::OneMinute),
            alpaca_bars::TimeFrame::OneMinute
        ));
        assert!(matches!(
            AlpacaExchangeClient::granularity_to_timeframe(Granularity::OneDay),
            alpaca_bars::TimeFrame::OneDay
        ));
    }
}
