pub mod dual_leg_trading;
pub mod moving_average;

pub mod supervisor;
pub mod tick_router;

use async_trait::async_trait;
use polars::prelude::*;
use rust_decimal::Decimal;
use std::sync::Arc;
use tokio::sync::mpsc;

// Re-export commonly used types
pub use dual_leg_trading::MarketData;

/// Represents a trading signal.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Signal {
    Buy,
    Sell,
    Exit, // CF2: Explicit exit signal for mean reversion
    Hold,
}

// ============================================================================
// BACKTESTING TRAIT (Existing - DataFrame-based)
// ============================================================================

/// Strategy trait contract for backtesting.
pub trait Strategy {
    /// Given a DataFrame with at least a `close` column, generate a vector of Signals
    /// aligned with the rows of the DataFrame (same length).
    fn generate_signals(&self, df: &DataFrame) -> Result<Vec<Signal>, PolarsError>;

    /// Generates a trading signal for the latest data point.
    fn get_latest_signal(&self, df: &DataFrame, position_open: bool)
        -> Result<Signal, PolarsError>;
}

// Implement Strategy for the existing MovingAverageCrossover by delegating to its method.
impl Strategy for moving_average::MovingAverageCrossover {
    fn generate_signals(&self, df: &DataFrame) -> Result<Vec<Signal>, PolarsError> {
        // Call the inherent method implemented on the type.
        moving_average::MovingAverageCrossover::generate_signals(self, df)
    }

    fn get_latest_signal(
        &self,
        df: &DataFrame,
        position_open: bool,
    ) -> Result<Signal, PolarsError> {
        moving_average::MovingAverageCrossover::get_latest_signal(self, df, position_open)
    }
}

// ============================================================================
// LIVE TRADING TRAIT (New - Real-time streaming)
// ============================================================================

/// Unified market data input for live strategies
#[derive(Debug, Clone)]
pub enum StrategyInput {
    /// Single ticker update (for single-product strategies like MA)
    Tick(Arc<MarketData>),
    /// Paired ticker updates (for dual-leg strategies)
    PairedTick {
        leg1: Arc<MarketData>,
        leg2: Arc<MarketData>,
    },
}

/// Trait for live trading strategies that can be supervised
#[async_trait]
pub trait LiveStrategy: Send {
    /// Unique identifier for this strategy instance
    fn id(&self) -> String;

    /// Symbols this strategy needs market data for
    fn subscribed_symbols(&self) -> Vec<String>;

    /// Run the strategy with market data receiver
    /// Strategy runs until the receiver is closed
    async fn run(&mut self, data_rx: mpsc::Receiver<StrategyInput>);

    /// Current unrealized PnL for aggregation (returns 0 if not tracked)
    fn current_pnl(&self) -> Decimal {
        Decimal::ZERO
    }

    /// Check if strategy is in a healthy state
    fn is_healthy(&self) -> bool {
        true
    }

    /// Get strategy type for logging/display
    fn strategy_type(&self) -> &'static str;
}
