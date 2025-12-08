//! # Resilience Module
//!
//! This module provides reusable resilience patterns for the trading system.
//!
//! ## Components
//! - `CircuitBreaker`: Prevents cascading failures by blocking requests after threshold failures.

pub mod circuit_breaker;

// Re-export for convenience
pub use circuit_breaker::{CircuitBreaker, CircuitState};
