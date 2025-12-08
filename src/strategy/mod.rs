pub mod moving_average;
pub mod dual_leg_trading;
pub mod portfolio;

use polars::prelude::*;

/// Represents a trading signal.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Signal {
    Buy,
    Sell,
    Exit, // CF2: Explicit exit signal for mean reversion
    Hold,
}


/// Strategy trait contract for backtesting.
pub trait Strategy {
	/// Given a DataFrame with at least a `close` column, generate a vector of Signals
	/// aligned with the rows of the DataFrame (same length).
	fn generate_signals(&self, df: &DataFrame) -> Result<Vec<Signal>, PolarsError>;

    /// Generates a trading signal for the latest data point.
    fn get_latest_signal(&self, df: &DataFrame, position_open: bool) -> Result<Signal, PolarsError>;
}

// Implement Strategy for the existing MovingAverageCrossover by delegating to its method.
impl Strategy for moving_average::MovingAverageCrossover {
	fn generate_signals(&self, df: &DataFrame) -> Result<Vec<Signal>, PolarsError> {
		// Call the inherent method implemented on the type.
		moving_average::MovingAverageCrossover::generate_signals(self, df)
	}

    fn get_latest_signal(&self, df: &DataFrame, position_open: bool) -> Result<Signal, PolarsError> {
        moving_average::MovingAverageCrossover::get_latest_signal(self, df, position_open)
    }
}
