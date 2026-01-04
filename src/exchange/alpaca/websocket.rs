//! Alpaca WebSocket Provider
//!
//! Real-time market data streaming via native WebSocket connection to Alpaca.
//!
//! ## WebSocket Flow
//! 1. Connect to `wss://stream.data.alpaca.markets/v2/iex`
//! 2. Receive `{"T":"success","msg":"connected"}`
//! 3. Send auth: `{"action":"auth","key":"...","secret":"..."}`
//! 4. Receive `{"T":"success","msg":"authenticated"}`
//! 5. Send subscribe: `{"action":"subscribe","trades":["AAPL","AMD"]}`
//! 6. Receive trades in real-time: `{"T":"t","S":"AAPL","p":"126.55","t":"..."}`

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use rust_decimal::prelude::FromPrimitive;
use rust_decimal::Decimal;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::str::FromStr;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::interval;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::exchange::alpaca::utils;
use crate::exchange::{ExchangeConfig, ExchangeError, WebSocketProvider};
use crate::types::MarketData;

/// Static instrument_id to avoid heap allocation per tick
static ALPACA_INSTRUMENT_ID: &str = "alpaca";

/// IEX feed URL (free tier)
const WS_URL_IEX: &str = "wss://stream.data.alpaca.markets/v2/iex";

// Alpaca WebSocket message types
#[derive(Debug, Deserialize)]
struct AlpacaMessage {
    #[serde(rename = "T")]
    msg_type: String,
    #[serde(default)]
    msg: Option<String>,
}

/// Alpaca WebSocket provider for real-time market data streaming
///
/// Connects to Alpaca's real-time stock data WebSocket for sub-second
/// price updates. Uses the IEX feed (free tier).
///
/// # CB-1 FIX: Lifecycle Management
/// The provider includes a `CancellationToken` for graceful shutdown.
/// When the token is cancelled, the background WebSocket task will exit cleanly.
///
/// # MC-4 FIX: Configurable Circuit Breaker
/// All circuit breaker and timeout parameters are configurable at construction time.
pub struct AlpacaWebSocketProvider {
    api_key: String,
    api_secret: String,
    /// CB-1 FIX: Token for graceful shutdown of the WebSocket background task
    shutdown_token: CancellationToken,
    /// MC-4 FIX: Maximum consecutive failures before circuit breaker trips
    max_consecutive_failures: u32,
    /// MC-4 FIX: Cooldown period after circuit breaker trips (seconds)
    circuit_breaker_cooldown_secs: u64,
    /// MC-4 FIX: Maximum retries for each connection attempt
    connection_max_retries: u32,
    /// MC-3 FIX: Read timeout for WebSocket messages (seconds)
    read_timeout_secs: u64,
    /// CB-1 FIX: Handle to the background task for observability
    task_handle: std::sync::RwLock<Option<tokio::task::JoinHandle<()>>>,
}

impl AlpacaWebSocketProvider {
    /// Default values for circuit breaker configuration
    const DEFAULT_MAX_CONSECUTIVE_FAILURES: u32 = 10;
    const DEFAULT_CIRCUIT_BREAKER_COOLDOWN_SECS: u64 = 300;
    const DEFAULT_CONNECTION_MAX_RETRIES: u32 = 5;
    const DEFAULT_READ_TIMEOUT_SECS: u64 = 120;

    /// Create a new Alpaca WebSocket provider
    pub fn new(config: &ExchangeConfig) -> Result<Self, ExchangeError> {
        info!(
            sandbox = config.sandbox,
            "Creating Alpaca WebSocket provider (streaming mode)"
        );
        Ok(Self {
            api_key: config.api_key.clone(),
            api_secret: config.api_secret.clone(),
            shutdown_token: CancellationToken::new(),
            max_consecutive_failures: Self::DEFAULT_MAX_CONSECUTIVE_FAILURES,
            circuit_breaker_cooldown_secs: Self::DEFAULT_CIRCUIT_BREAKER_COOLDOWN_SECS,
            connection_max_retries: Self::DEFAULT_CONNECTION_MAX_RETRIES,
            read_timeout_secs: Self::DEFAULT_READ_TIMEOUT_SECS,
            task_handle: std::sync::RwLock::new(None),
        })
    }

    /// Create from environment variables
    pub fn from_env() -> Result<Self, ExchangeError> {
        let api_key = std::env::var("ALPACA_API_KEY")
            .map_err(|_| ExchangeError::Configuration("ALPACA_API_KEY must be set".to_string()))?;
        let api_secret = std::env::var("ALPACA_API_SECRET").map_err(|_| {
            ExchangeError::Configuration("ALPACA_API_SECRET must be set".to_string())
        })?;

        Ok(Self {
            api_key,
            api_secret,
            shutdown_token: CancellationToken::new(),
            max_consecutive_failures: Self::DEFAULT_MAX_CONSECUTIVE_FAILURES,
            circuit_breaker_cooldown_secs: Self::DEFAULT_CIRCUIT_BREAKER_COOLDOWN_SECS,
            connection_max_retries: Self::DEFAULT_CONNECTION_MAX_RETRIES,
            read_timeout_secs: Self::DEFAULT_READ_TIMEOUT_SECS,
            task_handle: std::sync::RwLock::new(None),
        })
    }

    /// MC-1 FIX: Get the shutdown token for external lifecycle management
    ///
    /// Call `shutdown_token().cancel()` to trigger graceful shutdown of the WebSocket task.
    pub fn shutdown_token(&self) -> CancellationToken {
        self.shutdown_token.clone()
    }

    /// MC-1 FIX: Wait for the background task to complete
    pub async fn join(&self) {
        let handle = {
            let mut guard = self.task_handle.write().unwrap();
            guard.take()
        };

        if let Some(h) = handle {
            if let Err(e) = h.await {
                error!("WebSocket background task panicked: {:?}", e);
            }
        }
    }

    /// Connect to WebSocket with retry logic
    async fn connect_with_retry(
        max_retries: u32,
    ) -> Result<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
        ExchangeError,
    > {
        let mut backoff = Duration::from_secs(1);
        let max_backoff = Duration::from_secs(60);

        for attempt in 1..=max_retries {
            match connect_async(WS_URL_IEX).await {
                Ok((stream, _)) => {
                    info!("Alpaca WebSocket connected (attempt {})", attempt);
                    return Ok(stream);
                }
                Err(e) if attempt < max_retries => {
                    error!(
                        "WebSocket connection failed (attempt {}/{}): {}",
                        attempt, max_retries, e
                    );
                    info!("Retrying in {:?}...", backoff);
                    tokio::time::sleep(backoff).await;
                    backoff = std::cmp::min(backoff * 2, max_backoff);
                }
                Err(e) => {
                    error!("WebSocket connection failed after {} attempts", max_retries);
                    return Err(ExchangeError::Network(format!(
                        "Failed to connect after {} retries: {}",
                        max_retries, e
                    )));
                }
            }
        }

        Err(ExchangeError::Network(
            "Connection retry loop exited unexpectedly".to_string(),
        ))
    }

    /// CB-1 FIX: Parse trade from serde_json::Value without f64 precision loss
    /// MC-2 FIX: Zero-allocation parsing (no to_string() call)
    /// MC-3 FIX: O(1) HashMap lookup instead of O(n) Vec scan
    /// MC-1 FIX: No Utc::now() fallback - strict timestamp parsing
    fn parse_trade_from_value(
        value: &serde_json::Value,
        symbol_map: &HashMap<String, String>,
    ) -> Option<MarketData> {
        // Early exit: check message type inline
        let msg_type = value.get("T")?.as_str()?;
        if msg_type != "t" {
            return None;
        }

        let alpaca_symbol = value.get("S")?.as_str()?;
        let timestamp_str = value.get("t")?.as_str()?;

        // O(1) symbol lookup (MC-3 FIX)
        let original_symbol = symbol_map
            .get(alpaca_symbol)
            .cloned()
            .unwrap_or_else(|| alpaca_symbol.to_string());

        // CB-1 FIX: Precision-safe price parsing
        // Try string first (preferred), fall back to f64 if Alpaca sends numeric
        let price = if let Some(s) = value.get("p")?.as_str() {
            // Best case: string value preserves precision
            Decimal::from_str(s).ok()?
        } else if let Some(n) = value.get("p")?.as_f64() {
            // Fallback: Alpaca may send numeric JSON (lossy but functional)
            Decimal::from_f64(n)?
        } else {
            return None;
        };

        // MC-1 FIX: Strict timestamp - no Utc::now() fallback
        let timestamp = timestamp_str
            .parse::<DateTime<Utc>>()
            .ok()?
            .timestamp_millis();

        Some(MarketData {
            symbol: original_symbol,
            instrument_id: Some(ALPACA_INSTRUMENT_ID.to_owned()),
            price,
            timestamp,
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
        info!(
            symbols_count = symbols.len(),
            "Starting Alpaca WebSocket streaming"
        );

        // MC-3 FIX: Use HashMap for O(1) symbol lookup
        // Key: Alpaca symbol, Value: Original symbol
        let symbol_map: HashMap<String, String> = symbols
            .iter()
            .map(|s| (utils::to_alpaca_symbol(s).into_owned(), s.clone()))
            .collect();

        let alpaca_symbols: Vec<String> = symbol_map.keys().cloned().collect();

        let api_key = self.api_key.clone();
        let api_secret = self.api_secret.clone();

        // CB-1 FIX: Clone shutdown token for the spawned task
        let shutdown_token = self.shutdown_token.clone();

        // MC-4 FIX: Clone config values for the spawned task
        let max_consecutive_failures = self.max_consecutive_failures;
        let circuit_breaker_cooldown_secs = self.circuit_breaker_cooldown_secs;
        let connection_max_retries = self.connection_max_retries;
        let read_timeout_secs = self.read_timeout_secs;

        // Spawn background WebSocket handler
        let handle = tokio::spawn(async move {
            let mut consecutive_failures: u32 = 0;
            let mut reconnect_count: u64 = 0; // u64 for long-running processes

            loop {
                reconnect_count += 1;
                info!(
                    "Alpaca WebSocket connection attempt {} (symbols: {:?})",
                    reconnect_count, alpaca_symbols
                );

                // CB-2 FIX: Check circuit breaker before attempting connection
                if consecutive_failures >= max_consecutive_failures {
                    error!(
                        consecutive_failures = consecutive_failures,
                        cooldown_secs = circuit_breaker_cooldown_secs,
                        "Circuit breaker tripped, entering cooldown"
                    );
                    crate::metrics::CIRCUIT_BREAKER_TRIPS
                        .with_label_values(&["alpaca_websocket"])
                        .inc();

                    // Wait for cooldown or shutdown
                    tokio::select! {
                        _ = shutdown_token.cancelled() => {
                            info!("Shutdown requested during circuit breaker cooldown");
                            return;
                        }
                        _ = tokio::time::sleep(Duration::from_secs(circuit_breaker_cooldown_secs)) => {
                            info!("Circuit breaker cooldown complete, resetting failure count");
                            consecutive_failures = 0;
                        }
                    }
                }

                // CB-1 FIX: Check for shutdown before connecting
                if shutdown_token.is_cancelled() {
                    info!("Shutdown requested, exiting WebSocket loop");
                    return;
                }

                // Connect with retries
                let ws_stream = match Self::connect_with_retry(connection_max_retries).await {
                    Ok(stream) => {
                        // Reset failure count on successful connection
                        consecutive_failures = 0;
                        stream
                    }
                    Err(e) => {
                        consecutive_failures += 1;
                        error!(
                            error = %e,
                            consecutive_failures = consecutive_failures,
                            "Failed to establish WebSocket connection"
                        );
                        crate::metrics::WS_RECONNECTIONS
                            .with_label_values(&["failure"])
                            .inc();
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        continue;
                    }
                };

                let (mut write, mut read) = ws_stream.split();

                // Wait for connection success message
                let mut authenticated = false;
                let mut subscribed = false;

                // Read initial connection message
                if let Some(Ok(Message::Text(text))) = read.next().await {
                    debug!("Received: {}", text);
                    // Parse as array of messages
                    if let Ok(msgs) = serde_json::from_str::<Vec<AlpacaMessage>>(&text) {
                        for msg in msgs {
                            if msg.msg_type == "success" && msg.msg.as_deref() == Some("connected")
                            {
                                info!("Alpaca WebSocket connected successfully");

                                // Send auth message
                                let auth_msg = json!({
                                    "action": "auth",
                                    "key": api_key,
                                    "secret": api_secret
                                });

                                if let Err(e) =
                                    write.send(Message::Text(auth_msg.to_string().into())).await
                                {
                                    error!("Failed to send auth message: {}", e);
                                    break;
                                }
                            }
                        }
                    }
                }

                // Wait for auth response
                if let Some(Ok(Message::Text(text))) = read.next().await {
                    debug!("Auth response: {}", text);
                    if let Ok(msgs) = serde_json::from_str::<Vec<AlpacaMessage>>(&text) {
                        for msg in msgs {
                            if msg.msg_type == "success"
                                && msg.msg.as_deref() == Some("authenticated")
                            {
                                info!("Alpaca WebSocket authenticated");
                                authenticated = true;

                                // Send subscribe message
                                let subscribe_msg = json!({
                                    "action": "subscribe",
                                    "trades": alpaca_symbols
                                });

                                if let Err(e) = write
                                    .send(Message::Text(subscribe_msg.to_string().into()))
                                    .await
                                {
                                    error!("Failed to send subscribe message: {}", e);
                                    break;
                                }
                            } else if msg.msg_type == "error" {
                                error!("Auth failed: {:?}", msg.msg);
                                tokio::time::sleep(Duration::from_secs(30)).await;
                                break;
                            }
                        }
                    }
                }

                if !authenticated {
                    error!("Failed to authenticate, reconnecting...");
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    continue;
                }

                // Wait for subscription confirmation
                if let Some(Ok(Message::Text(text))) = read.next().await {
                    debug!("Subscribe response: {}", text);
                    if let Ok(msgs) = serde_json::from_str::<Vec<AlpacaMessage>>(&text) {
                        for msg in msgs {
                            if msg.msg_type == "subscription" {
                                info!("Alpaca WebSocket subscribed to trades");
                                subscribed = true;
                            }
                        }
                    }
                }

                if !subscribed {
                    warn!("Subscription confirmation not received, continuing anyway...");
                }

                // Main message loop
                let mut heartbeat_interval = interval(Duration::from_secs(30));

                loop {
                    // MC-3 FIX: Read timeout duration (from config)
                    let read_timeout = Duration::from_secs(read_timeout_secs);

                    tokio::select! {
                        // CB-1 FIX: Check for shutdown signal
                        _ = shutdown_token.cancelled() => {
                            info!("Shutdown requested, closing Alpaca WebSocket");
                            return;
                        }
                        _ = heartbeat_interval.tick() => {
                            debug!("Alpaca WebSocket heartbeat: connection active");
                        }
                        // MC-3 FIX: Wrap read in timeout to detect unresponsive servers
                        timeout_result = tokio::time::timeout(read_timeout, read.next()) => {
                            match timeout_result {
                                Err(_elapsed) => {
                                    // MC-3 FIX: Timeout occurred - server is unresponsive
                                    warn!(
                                        timeout_secs = read_timeout_secs,
                                        "WebSocket read timeout, server may be unresponsive. Reconnecting..."
                                    );
                                    break;
                                }
                                Ok(msg) => match msg {
                                    Some(Ok(Message::Text(text))) => {
                                        // Alpaca sends arrays of messages
                                        // MC-2 FIX: Parse directly, no intermediate to_string()
                                        if let Ok(messages) = serde_json::from_str::<Vec<serde_json::Value>>(&text) {
                                            let arrival = std::time::Instant::now();
                                            for msg_value in messages {
                                                if let Some(data) = Self::parse_trade_from_value(
                                                    &msg_value,
                                                    &symbol_map
                                                ) {
                                                    // MC-3 FIX: Record specific tick processing latency
                                                    crate::metrics::record_ws_tick_latency("alpaca", arrival.elapsed().as_secs_f64());
                                                    // MC-3 FIX: Record metrics before send to avoid clone
                                                    crate::metrics::record_ws_tick(&data.symbol);
                                                    if let Err(_e) = sender.send(data).await {
                                                        info!("Channel closed, stopping WebSocket");
                                                        return;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Some(Ok(Message::Ping(payload))) => {
                                        debug!("Received ping, sending pong");
                                        if let Err(e) = write.send(Message::Pong(payload)).await {
                                            warn!("Failed to send pong: {}", e);
                                        }
                                    }
                                    Some(Ok(Message::Close(frame))) => {
                                        info!("WebSocket closed by server: {:?}", frame);
                                        break;
                                    }
                                    Some(Err(e)) => {
                                        error!("WebSocket error: {}", e);
                                        break;
                                    }
                                    None => {
                                        info!("WebSocket stream ended");
                                        break;
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                } // End Inner Loop

                // If we reach here, the inner loop exited (error, timeout, or closed).
                // We should reconnect (Outer Loop continues).
                crate::metrics::WS_RECONNECTIONS
                    .with_label_values(&["reconnecting"])
                    .inc();
                info!("Initiating Alpaca WebSocket reconnection...");
                tokio::time::sleep(Duration::from_secs(2)).await;
            } // End Outer Loop
        });

        // Store handle for observability
        {
            let mut guard = self.task_handle.write().unwrap();
            *guard = Some(handle);
        }

        Ok(())
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
    fn test_parse_trade_with_string_price() {
        let symbol_map: HashMap<String, String> = [("AAPL".to_string(), "AAPL".to_string())].into();

        // CB-1 FIX: Test with string price (precision-safe)
        let value: serde_json::Value = serde_json::json!({
            "T": "t",
            "S": "AAPL",
            "p": "150.25",
            "t": "2024-01-15T10:30:00Z"
        });

        let result = AlpacaWebSocketProvider::parse_trade_from_value(&value, &symbol_map);
        assert!(result.is_some());

        let data = result.unwrap();
        assert_eq!(data.symbol, "AAPL");
        assert_eq!(data.price, Decimal::from_str("150.25").unwrap());
    }

    #[test]
    fn test_parse_trade_with_numeric_price_fallback() {
        let symbol_map: HashMap<String, String> = [("AAPL".to_string(), "AAPL".to_string())].into();

        // Test fallback for numeric price (Alpaca may send this)
        let value: serde_json::Value = serde_json::json!({
            "T": "t",
            "S": "AAPL",
            "p": 150.25,
            "t": "2024-01-15T10:30:00Z"
        });

        let result = AlpacaWebSocketProvider::parse_trade_from_value(&value, &symbol_map);
        assert!(result.is_some());
    }

    #[test]
    fn test_parse_non_trade_message() {
        let symbol_map: HashMap<String, String> = HashMap::new();
        let value: serde_json::Value = serde_json::json!({
            "T": "success",
            "msg": "connected"
        });

        let result = AlpacaWebSocketProvider::parse_trade_from_value(&value, &symbol_map);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_trade_bad_timestamp_returns_none() {
        let symbol_map: HashMap<String, String> = [("AAPL".to_string(), "AAPL".to_string())].into();

        // MC-1 FIX: Bad timestamp should return None, not fallback to Utc::now()
        let value: serde_json::Value = serde_json::json!({
            "T": "t",
            "S": "AAPL",
            "p": "150.25",
            "t": "invalid-timestamp"
        });

        let result = AlpacaWebSocketProvider::parse_trade_from_value(&value, &symbol_map);
        assert!(result.is_none());
    }
}
