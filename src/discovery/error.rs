//! Error types for the discovery module

use thiserror::Error;

/// Errors that can occur during pair discovery and optimization
#[derive(Error, Debug)]
pub enum DiscoveryError {
    /// API error from data provider
    #[error("API error: {0}")]
    Api(#[from] Box<dyn std::error::Error + Send + Sync>),

    /// Insufficient historical data for analysis
    #[error("Insufficient data: expected at least {expected} data points, got {actual}")]
    InsufficientData { expected: usize, actual: usize },

    /// No pairs passed the filtering criteria
    #[error("No viable pairs found matching criteria (correlation >= {min_correlation}, half-life <= {max_half_life}h)")]
    NoViablePairs {
        min_correlation: f64,
        max_half_life: f64,
    },

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// I/O error (file operations)
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Date parsing error
    #[error("Date parsing error: {0}")]
    DateParse(String),
}
