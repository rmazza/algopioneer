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
//! use algopioneer::infrastructure::exchange::{ExchangeConfig, ExchangeId};
//! use algopioneer::infrastructure::exchange::alpaca::AlpacaClient;
//!
//! let config = ExchangeConfig::from_env(ExchangeId::Alpaca)?;
//! let client = AlpacaClient::from_config(config)?;
//! ```

mod alpaca_client;
pub mod utils;
mod websocket;

// Single unified client export (DRY principle)
pub use alpaca_client::AlpacaClient;
pub use websocket::AlpacaWebSocketProvider;
