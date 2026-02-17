//! Exchange Abstraction Layer
//!
//! This module provides exchange-agnostic traits and factories for trading.
//! New exchanges can be added by implementing the core traits without
//! modifying business logic in strategies.

pub mod alpaca;
pub mod coinbase;
pub mod kraken;

use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use rust_decimal::prelude::FromPrimitive;
use rust_decimal::Decimal;
use std::error::Error;
use std::sync::Arc;
use thiserror::Error as ThisError;
use tokio::sync::mpsc;

// Re-export shared types for convenience
pub use crate::types::{MarketData, OrderSide};

// Re-export commonly used types (MC-3 FIX: single unified Alpaca client)
pub use alpaca::AlpacaClient;
pub use coinbase::CoinbaseExchangeClient;
pub use kraken::KrakenExchangeClient;

/// Exchange-agnostic candle data
///
/// All price fields use `Decimal` for financial precision.
/// Timestamp is derived from the source candle's start time.
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: DateTime<Utc>,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
}

/// Default candle with zero values (used as fallback for conversion failures)
impl Default for Candle {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            open: Decimal::ZERO,
            high: Decimal::ZERO,
            low: Decimal::ZERO,
            close: Decimal::ZERO,
            volume: Decimal::ZERO,
        }
    }
}

/// Convert from cbadv Candle to our Candle
///
/// CB-1 FIX: Uses Decimal instead of f64 for financial precision
/// CB-2 FIX: Uses source candle's `start` timestamp instead of Utc::now()
/// MC-2 FIX: Logs warnings for conversion failures instead of silent fallback
impl From<cbadv::models::product::Candle> for Candle {
    fn from(c: cbadv::models::product::Candle) -> Self {
        // CB-2 FIX: Convert unix timestamp (seconds) to DateTime
        let timestamp = Utc
            .timestamp_opt(c.start as i64, 0)
            .single()
            .unwrap_or_else(Utc::now); // Fallback only for truly invalid timestamps

        // MC-2 FIX: Helper to convert f64 to Decimal with warning on failure
        let convert_with_warning = |value: f64, field: &str| -> Decimal {
            match Decimal::from_f64(value) {
                Some(d) => d,
                None => {
                    tracing::warn!(
                        field = field,
                        value = value,
                        "Failed to convert candle {} to Decimal, using ZERO",
                        field
                    );
                    Decimal::ZERO
                }
            }
        };

        Self {
            timestamp,
            open: convert_with_warning(c.open, "open"),
            high: convert_with_warning(c.high, "high"),
            low: convert_with_warning(c.low, "low"),
            close: convert_with_warning(c.close, "close"),
            volume: convert_with_warning(c.volume, "volume"),
        }
    }
}

/// Unified granularity enum (maps to exchange-specific values)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Granularity {
    OneMinute,
    FiveMinute,
    FifteenMinute,
    ThirtyMinute,
    OneHour,
    TwoHour,
    SixHour,
    OneDay,
}

impl From<Granularity> for cbadv::time::Granularity {
    fn from(g: Granularity) -> Self {
        match g {
            Granularity::OneMinute => cbadv::time::Granularity::OneMinute,
            Granularity::FiveMinute => cbadv::time::Granularity::FiveMinute,
            Granularity::FifteenMinute => cbadv::time::Granularity::FifteenMinute,
            Granularity::ThirtyMinute => cbadv::time::Granularity::ThirtyMinute,
            Granularity::OneHour => cbadv::time::Granularity::OneHour,
            Granularity::TwoHour => cbadv::time::Granularity::TwoHour,
            Granularity::SixHour => cbadv::time::Granularity::SixHour,
            Granularity::OneDay => cbadv::time::Granularity::OneDay,
        }
    }
}

/// Configuration for exchange connections
#[derive(Debug, Clone)]
pub struct ExchangeConfig {
    pub api_key: String,
    pub api_secret: String,
    pub sandbox: bool,
}

impl ExchangeConfig {
    /// Create config from environment variables for the specified exchange
    ///
    /// MC-2 FIX: Returns typed `ExchangeError` instead of `Box<dyn Error>`
    pub fn from_env(exchange: ExchangeId) -> Result<Self, ExchangeError> {
        let (key_var, secret_var) = match exchange {
            ExchangeId::Coinbase => ("COINBASE_API_KEY", "COINBASE_API_SECRET"),
            ExchangeId::Kraken => ("KRAKEN_API_KEY", "KRAKEN_API_SECRET"),
            ExchangeId::Alpaca => ("ALPACA_API_KEY", "ALPACA_API_SECRET"),
        };

        let api_key = std::env::var(key_var).map_err(|_| {
            ExchangeError::Configuration(format!("{} must be set in environment", key_var))
        })?;
        let api_secret = std::env::var(secret_var).map_err(|_| {
            ExchangeError::Configuration(format!("{} must be set in environment", secret_var))
        })?;

        Ok(Self {
            api_key,
            api_secret,
            sandbox: false,
        })
    }
}

// --- Typed Error Handling ---

/// Exchange-layer error type for actionable error handling.
/// Enables strategies to distinguish between retryable and non-retryable errors.
#[derive(ThisError, Debug, Clone)]
pub enum ExchangeError {
    /// Network connectivity issues (retryable)
    #[error("Network error: {0}")]
    Network(String),

    /// Rate limited by exchange (retryable after delay)
    #[error("Rate limited, retry after {0}ms")]
    RateLimited(u64),

    /// Order rejected by exchange (not retryable without modification)
    #[error("Order rejected: {0}")]
    OrderRejected(String),

    /// Exchange internal error (may be retryable)
    #[error("Exchange internal error: {0}")]
    ExchangeInternal(String),

    /// Configuration or setup errors
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Other errors (fallback category)
    #[error("{0}")]
    Other(String),
}

impl ExchangeError {
    /// Determines if this error type is potentially retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ExchangeError::Network(_)
                | ExchangeError::RateLimited(_)
                | ExchangeError::ExchangeInternal(_)
        )
    }

    /// Suggested retry delay in milliseconds (if retryable)
    pub fn retry_delay_ms(&self) -> Option<u64> {
        match self {
            ExchangeError::RateLimited(ms) => Some(*ms),
            ExchangeError::Network(_) => Some(1000), // 1 second for network errors
            ExchangeError::ExchangeInternal(_) => Some(5000), // 5 seconds for exchange issues
            _ => None,
        }
    }

    /// Convert from a boxed error (for legacy compatibility)
    pub fn from_boxed(e: Box<dyn Error + Send + Sync>) -> Self {
        let s = e.to_string();
        let lower = s.to_lowercase();

        if lower.contains("network") || lower.contains("timeout") || lower.contains("connection") {
            ExchangeError::Network(s)
        } else if lower.contains("rate limit") || lower.contains("too many requests") {
            ExchangeError::RateLimited(1000) // Default 1 second
        } else if lower.contains("insufficient funds")
            || lower.contains("rejected")
            || lower.contains("invalid")
        {
            ExchangeError::OrderRejected(s)
        } else if lower.contains("internal") || lower.contains("server error") {
            ExchangeError::ExchangeInternal(s)
        } else {
            ExchangeError::Other(s)
        }
    }
}

/// Convenience conversion from string errors
impl From<String> for ExchangeError {
    fn from(s: String) -> Self {
        ExchangeError::Other(s)
    }
}

impl From<&str> for ExchangeError {
    fn from(s: &str) -> Self {
        ExchangeError::Other(s.to_string())
    }
}

// --- CB-2 FIX: Structured Concurrency for WebSocket Tasks ---

/// Handle to a WebSocket background task for structured concurrency.
///
/// This type wraps a `JoinHandle` and provides methods for:
/// - Checking if the task is still running
/// - Gracefully awaiting task completion
/// - Detecting panics in the background task
///
/// # Example
///
/// ```ignore
/// let handle = provider.spawn_and_subscribe(symbols, sender).await?;
///
/// // Later, on shutdown:
/// if let Err(e) = handle.join().await {
///     error!("WebSocket task panicked: {:?}", e);
/// }
/// ```
pub struct WebSocketHandle {
    handle: tokio::task::JoinHandle<()>,
    exchange: ExchangeId,
}

impl WebSocketHandle {
    /// Create a new WebSocket handle
    pub fn new(handle: tokio::task::JoinHandle<()>, exchange: ExchangeId) -> Self {
        Self { handle, exchange }
    }

    /// Check if the background task has finished
    pub fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }

    /// Await the task completion.
    ///
    /// Returns `Ok(())` if task completed normally, `Err` if it panicked.
    pub async fn join(self) -> Result<(), tokio::task::JoinError> {
        self.handle.await
    }

    /// Abort the background task.
    ///
    /// The task will be cancelled at the next await point.
    pub fn abort(&self) {
        tracing::info!(exchange = %self.exchange, "Aborting WebSocket task");
        self.handle.abort();
    }

    /// Get the exchange this handle is for
    pub fn exchange(&self) -> ExchangeId {
        self.exchange
    }
}

/// Core trait for order execution - exchange implementations must provide this
///
/// MC-2 FIX: Returns `OrderId` instead of `()` to enable order lifecycle tracking.
#[async_trait]
pub trait Executor: Send + Sync {
    /// Execute an order on the exchange.
    ///
    /// Returns the exchange-assigned order ID for tracking the order lifecycle.
    async fn execute_order(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: Decimal,
        price: Option<Decimal>,
    ) -> Result<crate::orders::OrderId, ExchangeError>;

    /// Get current position for a symbol
    async fn get_position(&self, symbol: &str) -> Result<Decimal, ExchangeError>;

    /// Poll order status (for exchanges without WebSocket updates).
    ///
    /// Returns (state, filled_qty, avg_fill_price).
    ///
    /// # Default Implementation
    ///
    /// Returns `Filled` with unknown quantities. Implementations should override
    /// this to provide actual fill information from the exchange.
    async fn get_order_status(
        &self,
        _order_id: &crate::orders::OrderId,
    ) -> Result<(crate::orders::OrderState, Decimal, Option<Decimal>), ExchangeError> {
        // Default: assume order was filled (for backward compat with simple strategies)
        // Fill qty is unknown - callers should check state.is_terminal() rather than qty
        Ok((crate::orders::OrderState::Filled, Decimal::ZERO, None))
    }

    /// Check if the market is currently open for trading.
    ///
    /// # Returns
    /// - `Ok(true)`: Market is open, trading allowed.
    /// - `Ok(false)`: Market is closed, trading should pause.
    /// - `Err(e)`: Failed to check status (network error, etc).
    ///
    /// # Default Implementation
    /// Returns `true` (market always open), suitable for crypto exchanges.
    /// Equity exchanges (Alpaca) MUST override this.
    async fn check_market_hours(&self) -> Result<bool, ExchangeError> {
        Ok(true)
    }

    /// Cancel an order on the exchange.
    ///
    /// # Default Implementation
    /// Returns `Err(NotImplemented)` to avoid breaking existing clients.
    async fn cancel_order(&self, _order_id: &crate::orders::OrderId) -> Result<(), ExchangeError> {
        Err(ExchangeError::Other(
            "cancel_order not implemented".to_string(),
        ))
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

    /// Get candles with pagination (for large date ranges)
    async fn get_candles_paginated(
        &mut self,
        product_id: &str,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
        granularity: Granularity,
    ) -> Result<Vec<Candle>, ExchangeError>;

    /// Normalize a symbol to exchange-specific format
    /// e.g., "BTC-USD" -> "XXBTZUSD" for Kraken
    fn normalize_symbol(&self, symbol: &str) -> String;

    /// Get the exchange identifier
    fn exchange_id(&self) -> ExchangeId;
}

/// Trait for WebSocket market data providers
#[async_trait]
pub trait WebSocketProvider: Send + Sync {
    /// Connect and subscribe to market data for given symbols.
    ///
    /// **Deprecated**: Use `spawn_and_subscribe` instead for proper task lifecycle management.
    /// This method exists for backward compatibility but orphans the background task.
    async fn connect_and_subscribe(
        &self,
        symbols: Vec<String>,
        sender: mpsc::Sender<MarketData>,
    ) -> Result<(), ExchangeError>;

    /// CB-2 FIX: Spawn WebSocket task and return handle for structured concurrency.
    ///
    /// This is the preferred method for WebSocket connections. The returned handle
    /// enables proper shutdown and panic detection.
    ///
    /// # Returns
    ///
    /// A `WebSocketHandle` that can be used to:
    /// - Check if the task is still running
    /// - Await graceful completion
    /// - Abort the task on shutdown
    ///
    /// # Default Implementation
    ///
    /// Delegates to `connect_and_subscribe` and returns a **dummy handle**.
    /// The dummy handle does NOT track the actual WebSocket task.
    ///
    /// **Providers should override this method** to return a real handle
    /// that tracks the spawned task. Currently only Alpaca implements this.
    /// Coinbase/Kraken use the default (backward compatible but no lifecycle mgmt).
    async fn spawn_and_subscribe(
        &self,
        symbols: Vec<String>,
        sender: mpsc::Sender<MarketData>,
    ) -> Result<WebSocketHandle, ExchangeError> {
        // Default: call legacy method (for backward compatibility)
        self.connect_and_subscribe(symbols, sender).await?;
        // KNOWN LIMITATION: This dummy handle does NOT track the real task.
        // Providers that need proper shutdown should override this method.
        Ok(WebSocketHandle::new(
            tokio::spawn(async {}),
            ExchangeId::Coinbase, // Placeholder - providers should override
        ))
    }
}

/// Exchange identifier for factory/registry
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExchangeId {
    Coinbase,
    Kraken,
    Alpaca,
}

impl std::fmt::Display for ExchangeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExchangeId::Coinbase => write!(f, "coinbase"),
            ExchangeId::Kraken => write!(f, "kraken"),
            ExchangeId::Alpaca => write!(f, "alpaca"),
        }
    }
}

impl std::str::FromStr for ExchangeId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "coinbase" => Ok(ExchangeId::Coinbase),
            "kraken" => Ok(ExchangeId::Kraken),
            "alpaca" => Ok(ExchangeId::Alpaca),
            _ => Err(format!(
                "Unknown exchange: {}. Valid options: coinbase, kraken, alpaca",
                s
            )),
        }
    }
}

/// Factory function to create exchange clients
pub fn create_exchange_client(
    exchange: ExchangeId,
    config: ExchangeConfig,
) -> Result<Arc<dyn ExchangeClient>, ExchangeError> {
    match exchange {
        ExchangeId::Coinbase => {
            let client = coinbase::CoinbaseExchangeClient::new(config)?;
            Ok(Arc::new(client))
        }
        ExchangeId::Kraken => {
            let client = kraken::KrakenExchangeClient::new(config)?;
            Ok(Arc::new(client))
        }
        ExchangeId::Alpaca => {
            // MC-3 FIX: Use unified AlpacaClient
            let client = alpaca::AlpacaClient::from_config(config)?;
            Ok(Arc::new(client))
        }
    }
}

/// Factory function to create WebSocket providers
pub fn create_websocket_provider(
    exchange: ExchangeId,
    config: &ExchangeConfig,
) -> Result<Box<dyn WebSocketProvider>, ExchangeError> {
    match exchange {
        ExchangeId::Coinbase => {
            let provider = coinbase::CoinbaseWebSocketProvider::new(config)?;
            Ok(Box::new(provider))
        }
        ExchangeId::Kraken => {
            let provider = kraken::KrakenWebSocketProvider::new(config)?;
            Ok(Box::new(provider))
        }
        ExchangeId::Alpaca => {
            let provider = alpaca::AlpacaWebSocketProvider::new(config)?;
            Ok(Box::new(provider))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exchange_id_from_str() {
        assert_eq!(
            "coinbase".parse::<ExchangeId>().unwrap(),
            ExchangeId::Coinbase
        );
        assert_eq!(
            "Coinbase".parse::<ExchangeId>().unwrap(),
            ExchangeId::Coinbase
        );
        assert_eq!("kraken".parse::<ExchangeId>().unwrap(), ExchangeId::Kraken);
        assert!("binance".parse::<ExchangeId>().is_err());
    }

    #[test]
    fn test_exchange_id_display() {
        assert_eq!(ExchangeId::Coinbase.to_string(), "coinbase");
        assert_eq!(ExchangeId::Kraken.to_string(), "kraken");
    }
}
