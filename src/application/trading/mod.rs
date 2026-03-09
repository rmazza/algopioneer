//! Trading engine implementations.
//!
//! This module contains trading engines that combine strategies with
//! exchange clients for live or paper trading.

mod simple_engine;

pub use simple_engine::SimpleTradingEngine;
