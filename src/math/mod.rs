//! Mathematical utilities for trading strategies.
//!
//! This module provides statistical and mathematical primitives used
//! by trading strategies, including Kalman filtering for dynamic
//! hedge ratio estimation.

pub mod kalman;

pub use kalman::KalmanHedgeRatio;
