//! Coinbase Exchange Client Adapter
//!
//! Wraps the existing CoinbaseClient to implement the ExchangeClient trait.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

use crate::coinbase::{AppEnv, CoinbaseClient};
use crate::exchange::{
    Candle, ExchangeClient, ExchangeConfig, ExchangeError, ExchangeId, Executor, Granularity,
};
use crate::strategy::dual_leg_trading::OrderSide;

/// Coinbase exchange client implementing the ExchangeClient trait
pub struct CoinbaseExchangeClient {
    inner: CoinbaseClient,
    /// NP-3 FIX: Config retained for future features:
    /// - Dynamic API key rotation
    /// - Sandbox mode switching at runtime  
    /// - Rate limit configuration per-instance
    #[allow(dead_code)]
    config: ExchangeConfig,
}

impl CoinbaseExchangeClient {
    /// Create a new CoinbaseExchangeClient from configuration
    pub fn new(config: ExchangeConfig) -> Result<Self, ExchangeError> {
        let env = if config.sandbox {
            AppEnv::Sandbox
        } else {
            AppEnv::Live
        };

        // Use direct credential injection (thread-safe, no env var mutation)
        let inner = CoinbaseClient::with_credentials(
            config.api_key.clone(),
            config.api_secret.clone(),
            env,
            None,
        )
        .map_err(|e| ExchangeError::Configuration(e.to_string()))?;

        Ok(Self { inner, config })
    }

    /// Create with Paper trading mode
    pub fn new_paper(config: ExchangeConfig) -> Result<Self, ExchangeError> {
        // Use direct credential injection (thread-safe, no env var mutation)
        let inner = CoinbaseClient::with_credentials(
            config.api_key.clone(),
            config.api_secret.clone(),
            AppEnv::Paper,
            None,
        )
        .map_err(|e| ExchangeError::Configuration(e.to_string()))?;

        Ok(Self { inner, config })
    }

    /// Get reference to inner client for backward compatibility
    pub fn inner(&self) -> &CoinbaseClient {
        &self.inner
    }

    /// Get mutable reference to inner client
    pub fn inner_mut(&mut self) -> &mut CoinbaseClient {
        &mut self.inner
    }
}

#[async_trait]
impl Executor for CoinbaseExchangeClient {
    async fn execute_order(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: Decimal,
        price: Option<Decimal>,
    ) -> Result<(), ExchangeError> {
        self.inner
            .place_order(symbol, &side.to_string(), quantity, price)
            .await
            .map_err(ExchangeError::from_boxed)
    }

    async fn get_position(&self, symbol: &str) -> Result<Decimal, ExchangeError> {
        self.inner
            .get_position(symbol)
            .await
            .map_err(ExchangeError::from_boxed)
    }
}

#[async_trait]
impl ExchangeClient for CoinbaseExchangeClient {
    async fn test_connection(&mut self) -> Result<(), ExchangeError> {
        self.inner
            .test_connection()
            .await
            .map_err(|e| ExchangeError::Network(e.to_string()))
    }

    async fn get_candles(
        &mut self,
        product_id: &str,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
        granularity: Granularity,
    ) -> Result<Vec<Candle>, ExchangeError> {
        let cbadv_granularity: cbadv::time::Granularity = granularity.into();
        let candles = self
            .inner
            .get_product_candles(product_id, start, end, cbadv_granularity)
            .await
            .map_err(|e| ExchangeError::Network(e.to_string()))?;

        Ok(candles.into_iter().map(Candle::from).collect())
    }

    async fn get_candles_paginated(
        &mut self,
        product_id: &str,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
        granularity: Granularity,
    ) -> Result<Vec<Candle>, ExchangeError> {
        let cbadv_granularity: cbadv::time::Granularity = granularity.into();
        let candles = self
            .inner
            .get_product_candles_paginated(product_id, start, end, cbadv_granularity)
            .await
            .map_err(|e| ExchangeError::Network(e.to_string()))?;

        Ok(candles.into_iter().map(Candle::from).collect())
    }

    fn normalize_symbol(&self, symbol: &str) -> String {
        // Coinbase uses symbols as-is (e.g., "BTC-USD")
        symbol.to_string()
    }

    fn exchange_id(&self) -> ExchangeId {
        ExchangeId::Coinbase
    }
}
