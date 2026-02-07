//! Exit policy implementations for dual-leg trading strategies.
//!
//! This module provides flexible exit conditions through the `ExitPolicy` trait
//! and several implementations for different exit strategies.

use async_trait::async_trait;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

// AS9: ExitPolicy trait for flexible exit conditions
#[async_trait]
pub trait ExitPolicy: Send + Sync {
    /// Returns true if the position should be exited
    async fn should_exit(&self, entry_price: Decimal, current_price: Decimal, pnl: Decimal)
        -> bool;
}

/// Exit when minimum profit threshold is met
pub struct MinimumProfitPolicy {
    min_profit_bps: Decimal,
}

impl MinimumProfitPolicy {
    pub fn new(min_profit_bps: Decimal) -> Self {
        Self { min_profit_bps }
    }
}

#[async_trait]
impl ExitPolicy for MinimumProfitPolicy {
    async fn should_exit(
        &self,
        entry_price: Decimal,
        current_price: Decimal,
        _pnl: Decimal,
    ) -> bool {
        if entry_price.is_zero() {
            return false;
        }

        let price_change_bps = ((current_price - entry_price) / entry_price) * dec!(10000.0);
        price_change_bps >= self.min_profit_bps
    }
}

/// Exit when stop loss threshold is hit
pub struct StopLossPolicy {
    max_loss_bps: Decimal,
}

impl StopLossPolicy {
    pub fn new(max_loss_bps: Decimal) -> Self {
        Self { max_loss_bps }
    }
}

#[async_trait]
impl ExitPolicy for StopLossPolicy {
    async fn should_exit(
        &self,
        entry_price: Decimal,
        current_price: Decimal,
        _pnl: Decimal,
    ) -> bool {
        if entry_price.is_zero() {
            return false;
        }

        let price_change_bps = ((current_price - entry_price) / entry_price) * dec!(10000.0);
        price_change_bps.abs() >= self.max_loss_bps && price_change_bps < Decimal::ZERO
    }
}

/// Composite exit policy that triggers if ANY sub-policy triggers
pub struct CompositeExitPolicy {
    policies: Vec<Box<dyn ExitPolicy>>,
}

impl CompositeExitPolicy {
    pub fn new(policies: Vec<Box<dyn ExitPolicy>>) -> Self {
        Self { policies }
    }
}

#[async_trait]
impl ExitPolicy for CompositeExitPolicy {
    async fn should_exit(
        &self,
        entry_price: Decimal,
        current_price: Decimal,
        pnl: Decimal,
    ) -> bool {
        for policy in &self.policies {
            if policy.should_exit(entry_price, current_price, pnl).await {
                return true;
            }
        }
        false
    }
}

/// PnL-based exit policy that exits on profit target or stop loss
#[derive(Debug, Clone)]
pub struct PnlExitPolicy {
    min_profit: Decimal,
    stop_loss: Decimal,
}

impl PnlExitPolicy {
    pub fn new(min_profit: Decimal, stop_loss: Decimal) -> Self {
        Self {
            min_profit,
            stop_loss,
        }
    }
}

#[async_trait]
impl ExitPolicy for PnlExitPolicy {
    async fn should_exit(
        &self,
        _entry_price: Decimal,
        _current_price: Decimal,
        pnl: Decimal,
    ) -> bool {
        // Exit if PnL is below stop loss (e.g. -15 < -10)
        // OR if PnL is above min profit (e.g. 20 > 10)
        pnl <= self.stop_loss || pnl >= self.min_profit
    }
}
