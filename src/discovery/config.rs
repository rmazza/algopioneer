//! Configuration for pair discovery and optimization

use crate::strategy::dual_leg_trading::DualLegConfig;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};

/// Configuration for a specific pair in the portfolio (matching legacy format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioPairConfig {
    #[serde(flatten)]
    pub dual_leg_config: DualLegConfig,
    /// Rolling window size for z-score calculation (in ticks)
    pub window_size: usize,
    /// Z-score threshold to enter a position (must be positive)
    pub entry_z_score: f64,
    /// Z-score threshold to exit a position (must be < entry_z_score)
    pub exit_z_score: f64,
}

/// Default top-volume trading pairs on Coinbase
pub const DEFAULT_CANDIDATES: &[&str] = &[
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "ADA-USD",
    "DOGE-USD",
    "AVAX-USD",
    "SHIB-USD",
    "DOT-USD",
    "MATIC-USD",
    "LTC-USD",
    "UNI-USD",
    "LINK-USD",
    "XLM-USD",
    "BCH-USD",
    "ALGO-USD",
    "ATOM-USD",
    "FIL-USD",
    "VET-USD",
    "ICP-USD",
    "AXS-USD",
];

/// Configuration for the discovery pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// Candidate symbols to analyze
    pub candidates: Vec<String>,

    /// Minimum Pearson correlation threshold (0.0-1.0)
    #[serde(default = "default_min_correlation")]
    pub min_correlation: f64,

    /// Maximum half-life for mean reversion (hours)
    #[serde(default = "default_max_half_life")]
    pub max_half_life_hours: f64,

    /// Minimum Sharpe ratio to include pair in results
    #[serde(default = "default_min_sharpe")]
    pub min_sharpe_ratio: f64,

    /// Historical lookback period in days
    #[serde(default = "default_lookback_days")]
    pub lookback_days: u32,

    /// Maximum number of pairs to output
    #[serde(default = "default_max_pairs")]
    pub max_pairs_output: usize,

    /// Initial capital for backtests (in USD)
    #[serde(default = "default_initial_capital")]
    pub initial_capital: Decimal,

    /// Taker fee as decimal (e.g., 0.002 = 0.2%)
    #[serde(default = "default_taker_fee")]
    pub taker_fee: Decimal,

    /// Minimum number of trades required for valid backtest
    #[serde(default = "default_min_trades")]
    pub min_trades: u32,

    /// Minimum net profit required (in USD)
    #[serde(default = "default_min_net_profit")]
    pub min_net_profit: Decimal,

    /// Maximum Sharpe ratio cap (flag likely overfitting above this)
    #[serde(default = "default_max_sharpe")]
    pub max_sharpe_ratio: f64,

    /// Train/test split ratio (0.0-1.0, e.g., 0.7 = 70% train, 30% test)
    #[serde(default = "default_train_ratio")]
    pub train_ratio: f64,

    /// Require ADF cointegration test to pass
    #[serde(default = "default_require_cointegration")]
    pub require_cointegration: bool,
}

// Default value functions for serde
fn default_min_correlation() -> f64 {
    0.8
}
fn default_max_half_life() -> f64 {
    24.0
}
fn default_min_sharpe() -> f64 {
    1.0 // Raised from 0.5 for statistical significance
}
fn default_lookback_days() -> u32 {
    14
}
fn default_max_pairs() -> usize {
    10
}
fn default_initial_capital() -> Decimal {
    dec!(10_000)
}
fn default_taker_fee() -> Decimal {
    dec!(0.002)
}
fn default_min_trades() -> u32 {
    30 // Raised from 5 for statistical significance
}
fn default_min_net_profit() -> Decimal {
    dec!(0)
}
fn default_max_sharpe() -> f64 {
    4.0 // Cap to flag overfitting (HFT rarely exceeds 3-4)
}
fn default_train_ratio() -> f64 {
    0.7 // 70% train, 30% test for walk-forward validation
}
fn default_require_cointegration() -> bool {
    true // Require ADF test to pass by default
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            candidates: DEFAULT_CANDIDATES.iter().map(|s| s.to_string()).collect(),
            min_correlation: default_min_correlation(),
            max_half_life_hours: default_max_half_life(),
            min_sharpe_ratio: default_min_sharpe(),
            lookback_days: default_lookback_days(),
            max_pairs_output: default_max_pairs(),
            initial_capital: default_initial_capital(),
            taker_fee: default_taker_fee(),
            min_trades: default_min_trades(),
            min_net_profit: default_min_net_profit(),
            max_sharpe_ratio: default_max_sharpe(),
            train_ratio: default_train_ratio(),
            require_cointegration: default_require_cointegration(),
        }
    }
}

impl DiscoveryConfig {
    /// Create a config with custom candidates
    pub fn with_candidates(candidates: Vec<String>) -> Self {
        Self {
            candidates,
            ..Default::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.candidates.is_empty() {
            return Err("candidates list cannot be empty".to_string());
        }
        if self.candidates.len() < 2 {
            return Err("need at least 2 candidates to form pairs".to_string());
        }
        if !(0.0..=1.0).contains(&self.min_correlation) {
            return Err(format!(
                "min_correlation must be between 0.0 and 1.0, got {}",
                self.min_correlation
            ));
        }
        if self.max_half_life_hours <= 0.0 {
            return Err(format!(
                "max_half_life_hours must be positive, got {}",
                self.max_half_life_hours
            ));
        }
        if self.lookback_days == 0 {
            return Err("lookback_days must be at least 1".to_string());
        }
        if self.initial_capital <= Decimal::ZERO {
            return Err("initial_capital must be positive".to_string());
        }
        if self.taker_fee < Decimal::ZERO {
            return Err("taker_fee cannot be negative".to_string());
        }
        Ok(())
    }
}

/// Grid search parameter ranges
#[derive(Debug, Clone)]
pub struct GridSearchConfig {
    /// Window sizes to test
    pub windows: Vec<usize>,
    /// Z-score entry thresholds to test
    pub z_entries: Vec<f64>,
    /// Fixed Z-score exit threshold (mean reversion)
    pub z_exit: f64,
}

impl Default for GridSearchConfig {
    fn default() -> Self {
        Self {
            windows: (10..=60).step_by(5).collect(),
            z_entries: (15..=30).map(|i| i as f64 / 10.0).collect(),
            z_exit: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = DiscoveryConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_empty_candidates_invalid() {
        let config = DiscoveryConfig {
            candidates: vec![],
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_single_candidate_invalid() {
        let config = DiscoveryConfig {
            candidates: vec!["BTC-USD".to_string()],
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_correlation() {
        let config = DiscoveryConfig {
            min_correlation: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
}
