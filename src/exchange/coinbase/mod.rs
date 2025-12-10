//! Coinbase Exchange Implementation
//!
//! Adapters that wrap the existing CoinbaseClient and CoinbaseWebsocket
//! to implement the exchange abstraction traits.

mod client;
mod websocket;

pub use client::CoinbaseExchangeClient;
pub use websocket::CoinbaseWebSocketProvider;

// Re-export legacy types for backward compatibility
pub use crate::coinbase::{AppEnv, CoinbaseClient};
