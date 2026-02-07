//! Tick validation for market data.
//!
//! Provides composable validators to ensure market data quality before processing.

use rust_decimal::Decimal;

// Import MarketData from parent module
pub use crate::types::MarketData;

/// Trait for validating market data ticks before processing.
/// Enables testable and composable validation logic.
pub trait TickValidator: Send + Sync {
    /// Validates a market data tick against validation rules.
    /// Returns Ok(()) if valid, Err(msg) if invalid.
    fn validate(&self, tick: &MarketData, now_ts: i64) -> Result<(), String>;
}

/// Validates tick age to ensure data freshness.
#[derive(Debug, Clone)]
pub struct AgeValidator {
    max_age_ms: i64,
}

impl AgeValidator {
    pub fn new(max_age_ms: i64) -> Self {
        Self { max_age_ms }
    }
}

impl TickValidator for AgeValidator {
    fn validate(&self, tick: &MarketData, now_ts: i64) -> Result<(), String> {
        let age_ms = now_ts - tick.timestamp;
        if age_ms > self.max_age_ms {
            Err(format!(
                "Tick age {}ms exceeds max {}ms",
                age_ms, self.max_age_ms
            ))
        } else if age_ms < 0 {
            Err(format!(
                "Tick timestamp {} is in the future (now: {})",
                tick.timestamp, now_ts
            ))
        } else {
            Ok(())
        }
    }
}

/// Validates tick price is positive and not NaN/Inf.
#[derive(Debug, Clone)]
pub struct PriceValidator;

impl TickValidator for PriceValidator {
    fn validate(&self, tick: &MarketData, _now_ts: i64) -> Result<(), String> {
        if tick.price <= Decimal::ZERO {
            Err(format!("Invalid price: {} must be positive", tick.price))
        } else {
            Ok(())
        }
    }
}

/// Composite validator that chains multiple validators.
/// Fails on first validation error.
pub struct CompositeValidator {
    validators: Vec<Box<dyn TickValidator>>,
}

impl CompositeValidator {
    pub fn new(validators: Vec<Box<dyn TickValidator>>) -> Self {
        Self { validators }
    }
}

impl TickValidator for CompositeValidator {
    fn validate(&self, tick: &MarketData, now_ts: i64) -> Result<(), String> {
        for validator in &self.validators {
            validator.validate(tick, now_ts)?;
        }
        Ok(())
    }
}
