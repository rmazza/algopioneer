use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use tokio::sync::mpsc;

use crate::domain::types::{MarketData, OrderSide};
use crate::domain::exchange::{ExchangeError, ExchangeId, Candle, Granularity};
use crate::domain::orders::{OrderId, OrderState};

/// Core trait for order execution - exchange implementations must provide this
#[async_trait]
pub trait Executor: Send + Sync {
    /// Execute an order on the exchange.
    async fn execute_order(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: Decimal,
        price: Option<Decimal>,
    ) -> Result<OrderId, ExchangeError>;

    /// Get current position for a symbol
    async fn get_position(&self, symbol: &str) -> Result<Decimal, ExchangeError>;

    /// Poll order status (for exchanges without WebSocket updates).
    async fn get_order_status(
        &self,
        _order_id: &OrderId,
    ) -> Result<(OrderState, Decimal, Option<Decimal>), ExchangeError> {
        // Default: assume order was filled (for backward compat with simple strategies)
        Ok((OrderState::Filled, Decimal::ZERO, None))
    }

    /// Check if the market is currently open for trading.
    async fn check_market_hours(&self) -> Result<bool, ExchangeError> {
        Ok(true)
    }

    /// Cancel an order on the exchange.
    async fn cancel_order(&self, _order_id: &OrderId) -> Result<(), ExchangeError> {
        Err(ExchangeError::Other("cancel_order not implemented".to_string()))
    }

    /// Cancel ALL open orders for a symbol.
    async fn cancel_all_orders(&self, _symbol: &str) -> Result<(), ExchangeError> {
        Ok(())
    }
}

/// Extended exchange client trait with full capabilities
#[async_trait]
pub trait ExchangeClient: Executor + Send + Sync {
    /// Test API connectivity
    async fn test_connection(&mut self) -> Result<(), ExchangeError>;

    /// Get historical candles
    async fn get_candles(
        &mut self,
        product_id: &str,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
        granularity: Granularity,
    ) -> Result<Vec<Candle>, ExchangeError>;

    /// Get candles with pagination
    async fn get_candles_paginated(
        &mut self,
        product_id: &str,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
        granularity: Granularity,
    ) -> Result<Vec<Candle>, ExchangeError>;

    /// Normalize a symbol to exchange-specific format
    fn normalize_symbol(&self, symbol: &str) -> String;

    /// Get the exchange identifier
    fn exchange_id(&self) -> ExchangeId;
}

/// Trait for WebSocket market data providers
#[async_trait]
pub trait WebSocketProvider: Send + Sync {
    /// Connect and subscribe to market data for given symbols.
    async fn connect_and_subscribe(
        &self,
        symbols: Vec<String>,
        sender: mpsc::Sender<MarketData>,
    ) -> Result<(), ExchangeError>;

    /// Spawn WebSocket task and return handle for structured concurrency.
    async fn spawn_and_subscribe(
        &self,
        symbols: Vec<String>,
        sender: mpsc::Sender<MarketData>,
    ) -> Result<WebSocketHandle, ExchangeError> {
        self.connect_and_subscribe(symbols, sender).await?;
        Ok(WebSocketHandle::new(
            tokio::spawn(async {}),
            ExchangeId::Coinbase,
        ))
    }
}

/// Handle to a WebSocket background task for structured concurrency.
pub struct WebSocketHandle {
    pub handle: tokio::task::JoinHandle<()>,
    pub exchange: ExchangeId,
}

impl WebSocketHandle {
    pub fn new(handle: tokio::task::JoinHandle<()>, exchange: ExchangeId) -> Self {
        Self { handle, exchange }
    }

    pub fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }

    pub async fn join(self) -> Result<(), tokio::task::JoinError> {
        self.handle.await
    }

    pub fn abort(&self) {
        self.handle.abort();
    }
}
