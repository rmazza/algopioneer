//! Kraken Exchange Implementation (Stub)
//!
//! Placeholder implementation for Kraken exchange.
//! Methods return `unimplemented!()` - ready for future implementation.

mod client;
mod websocket;

pub use client::KrakenExchangeClient;
pub use websocket::KrakenWebSocketProvider;
