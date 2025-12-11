//! Pair Discovery and Optimization Module
//!
//! Automatically discovers cointegrated trading pairs and optimizes parameters
//! through grid search backtesting.
//!
//! # Example
//!
//! ```ignore
//! use algopioneer::discovery::{discover_and_optimize, DiscoveryConfig};
//! use algopioneer::coinbase::CoinbaseClient;
//!
//! let config = DiscoveryConfig::default();
//! let mut client = CoinbaseClient::new(AppEnv::Live)?;
//! let results = discover_and_optimize(&mut client, &config).await?;
//! ```

pub mod config;
pub mod error;
pub mod filter;
pub mod optimizer;

pub use config::DiscoveryConfig;
pub use error::DiscoveryError;
pub use filter::CandidatePair;
pub use optimizer::{discover_and_optimize, OptimizedPair};
