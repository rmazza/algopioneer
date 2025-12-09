use crate::coinbase::websocket::CoinbaseWebsocket;
use crate::strategy::dual_leg_trading::MarketData;
use async_trait::async_trait;
use rust_decimal::Decimal;
use std::error::Error;
use tokio::sync::mpsc;

/// Trait for market data providers. Enables swapping between different data sources
/// (WebSocket, synthetic, replay) without changing strategy code.
#[async_trait]
pub trait MarketDataProvider: Send + Sync {
    /// Subscribe to market data for the given symbols.
    /// Returns a receiver that will yield MarketData ticks.
    async fn subscribe(
        &self,
        symbols: Vec<String>,
    ) -> Result<mpsc::Receiver<MarketData>, Box<dyn Error + Send + Sync>>;

    /// Unsubscribe from market data for the given symbols.
    async fn unsubscribe(&self, symbols: Vec<String>) -> Result<(), Box<dyn Error + Send + Sync>>;
}

// ===== Coinbase WebSocket Provider =====

/// Production market data provider using Coinbase WebSocket
pub struct CoinbaseWebsocketProvider;

impl CoinbaseWebsocketProvider {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl MarketDataProvider for CoinbaseWebsocketProvider {
    async fn subscribe(
        &self,
        symbols: Vec<String>,
    ) -> Result<mpsc::Receiver<MarketData>, Box<dyn Error + Send + Sync>> {
        let (tx, rx) = mpsc::channel(1000);

        // Create new WebSocket connection - convert error to String for Send+Sync
        let ws = CoinbaseWebsocket::new().map_err(|e| -> Box<dyn Error + Send + Sync> {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })?;

        // Spawn task to connect and stream data
        tokio::spawn(async move {
            if let Err(e) = ws.connect_and_subscribe(symbols, tx).await {
                eprintln!("Coinbase WebSocket error: {}", e);
            }
        });

        Ok(rx)
    }

    async fn unsubscribe(&self, _symbols: Vec<String>) -> Result<(), Box<dyn Error + Send + Sync>> {
        // WebSocket connection is managed per-subscription, so unsubscribe is a no-op
        // In a more advanced implementation, this could send unsubscribe messages
        Ok(())
    }
}

// ===== Synthetic Provider (for testing) =====

/// Synthetic market data provider for testing without live exchange connection
pub struct SyntheticProvider {
    base_price: Decimal,
    volatility: f64,
    tick_interval_ms: u64,
}

impl SyntheticProvider {
    /// Create a new synthetic provider
    ///
    /// # Arguments
    /// * `base_price` - Starting price (e.g., 50000 for BTC)
    /// * `volatility` - Price movement per tick as a fraction (e.g., 0.001 = 0.1%)
    /// * `tick_interval_ms` - Time between ticks in milliseconds
    pub fn new(base_price: Decimal, volatility: f64, tick_interval_ms: u64) -> Self {
        Self {
            base_price,
            volatility,
            tick_interval_ms,
        }
    }
}

#[async_trait]
impl MarketDataProvider for SyntheticProvider {
    async fn subscribe(
        &self,
        symbols: Vec<String>,
    ) -> Result<mpsc::Receiver<MarketData>, Box<dyn Error + Send + Sync>> {
        let (tx, rx) = mpsc::channel(1000);

        let base_price = self.base_price;
        let volatility = self.volatility;
        let interval_ms = self.tick_interval_ms;

        // Spawn task to generate synthetic ticks
        tokio::spawn(async move {
            let mut current_prices: std::collections::HashMap<String, Decimal> =
                symbols.iter().map(|s| (s.clone(), base_price)).collect();

            let mut tick_count: i64 = 0;

            loop {
                tokio::time::sleep(tokio::time::Duration::from_millis(interval_ms)).await;

                for symbol in &symbols {
                    // Generate random walk price using sine wave for predictable variation
                    let current_price = current_prices.get(symbol).unwrap();
                    let change_pct = (tick_count as f64 * 0.01).sin() * volatility;
                    let change =
                        current_price * Decimal::try_from(change_pct).unwrap_or(Decimal::ZERO);
                    let new_price = current_price + change;

                    current_prices.insert(symbol.clone(), new_price);

                    let tick = MarketData {
                        symbol: symbol.clone(),
                        price: new_price,
                        instrument_id: None,
                        timestamp: chrono::Utc::now().timestamp_millis(),
                    };

                    // Non-blocking send with backpressure handling
                    // In HFT, it's better to drop stale data than process lagged data
                    match tx.try_send(tick) {
                        Ok(_) => {}
                        Err(mpsc::error::TrySendError::Full(_)) => {
                            // Channel full: drop tick to maintain real-time sync
                            // This prevents latency buildup when consumer is slower than producer
                        }
                        Err(mpsc::error::TrySendError::Closed(_)) => {
                            // Receiver dropped, stop generating
                            return;
                        }
                    }
                }

                tick_count += 1;
            }
        });

        Ok(rx)
    }

    async fn unsubscribe(&self, _symbols: Vec<String>) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Synthetic provider doesn't need explicit unsubscribe
        Ok(())
    }
}
