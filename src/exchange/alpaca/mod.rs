//! Alpaca Exchange Client
//!
//! Implementation of the exchange abstraction for Alpaca Markets.
//! Enables trading US stocks and ETFs with zero commission.
//!
//! ## Features
//! - Commission-free stock/ETF trading
//! - Paper trading mode for testing
//! - WebSocket streaming for real-time quotes
//!
//! ## Usage
//! ```ignore
//! use algopioneer::exchange::{ExchangeConfig, ExchangeId};
//! use algopioneer::exchange::alpaca::AlpacaExchangeClient;
//!
//! let config = ExchangeConfig::from_env(ExchangeId::Alpaca)?;
//! let client = AlpacaExchangeClient::new(config)?;
//! ```

mod alpaca_client;
mod client;
pub mod utils;
mod websocket;

pub use alpaca_client::AlpacaClient;
pub use client::AlpacaExchangeClient;
pub use websocket::AlpacaWebSocketProvider;
