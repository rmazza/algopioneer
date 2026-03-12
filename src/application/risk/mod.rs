//! Risk Management Module
//!
//! Provides portfolio-level risk controls including daily loss limits.

mod daily_limit;
mod executor;

pub use daily_limit::{DailyRiskConfig, DailyRiskEngine, RiskStatus};
pub use executor::RiskManagedExecutor;
