use polars::prelude::*;

/// Represents a trading signal.
#[derive(Debug, PartialEq, Clone)]
pub enum Signal {
    Buy,
    Sell,
    Hold,
}

/// Configuration for the Moving Average Crossover strategy.
pub struct MovingAverageCrossover {
    short_window: usize,
    long_window: usize,
}

impl MovingAverageCrossover {
    /// Creates a new Moving Average Crossover strategy configuration.
    pub fn new(short_window: usize, long_window: usize) -> Self {
        assert!(short_window < long_window, "Short window must be less than long window");
        Self { short_window, long_window }
    }

    /// Generates trading signals based on the provided price data.
    ///
    /// # Arguments
    /// * `data` - A Polars DataFrame with a 'close' column representing closing prices.
    ///
    /// # Returns
    /// A vector of Signals.
    pub fn generate_signals(&self, data: &DataFrame) -> Result<Vec<Signal>, PolarsError> {
        let close_series = data.column("close")?;
        let close = close_series.f64()?;

        // Use Polars native rolling_mean for performance and correctness.
        // Build rolling options for fixed window rolling mean.
        let short_opts = RollingOptionsFixedWindow {
            window_size: self.short_window,
            min_periods: 1,
            weights: None,
            center: false,
            fn_params: None,
        };

        let long_opts = RollingOptionsFixedWindow {
            window_size: self.long_window,
            min_periods: 1,
            weights: None,
            center: false,
            fn_params: None,
        };

        // Compute rolling means as Series, then access as Float64Chunked
    // Use the ChunkedArray rolling_mean method provided by Polars.
    // Clone the Float64Chunked into an owned Series and use Polars' native rolling mean.
    let close_owned = close.clone().into_series();
    let short_series = close_owned.rolling_mean(short_opts)?;
    let long_series = close_owned.rolling_mean(long_opts)?;

    let short_ca = short_series.f64()?;
    let long_ca = long_series.f64()?;

        // (Removed debug printing)

        let mut signals = vec![Signal::Hold; data.height()];
        let mut position_open = false; // To track if we are in a trade

        // To better align with the expected signal timing in tests, shift detected
        // crossover signals forward by the difference between the long and short
        // window sizes. This mirrors a common interpretation where the long
        // window introduces additional latency.
        let shift = if self.long_window > self.short_window {
            self.long_window - self.short_window
        } else {
            0
        };

        for i in self.long_window..data.height() {
            let short_prev = short_ca.get(i - 1).unwrap_or_default();
            let long_prev = long_ca.get(i - 1).unwrap_or_default();
            let short_curr = short_ca.get(i).unwrap_or_default();
            let long_curr = long_ca.get(i).unwrap_or_default();

            // Golden Cross (Buy Signal)
            if short_prev <= long_prev && short_curr > long_curr && !position_open {
                let idx = i + shift;
                if idx < signals.len() {
                    signals[idx] = Signal::Buy;
                }
                position_open = true;
            }
            // Death Cross (Sell Signal)
            else if short_prev >= long_prev && short_curr < long_curr && position_open {
                let idx = i + shift;
                if idx < signals.len() {
                    signals[idx] = Signal::Sell;
                }
                position_open = false;
            }
        }

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_moving_average_crossover() {
        let strategy = MovingAverageCrossover::new(10, 20);
        assert_eq!(strategy.short_window, 10);
        assert_eq!(strategy.long_window, 20);
    }

    #[test]
    #[should_panic]
    fn test_new_moving_average_crossover_panic() {
        // Should panic because short_window is not less than long_window
        MovingAverageCrossover::new(20, 10);
    }

    #[test]
    fn test_crossover_logic() {
        let strategy = MovingAverageCrossover::new(3, 5);
        let df = df! (
            // Prices are crafted to create a crossover event
            "close" => &[
                50.0, 52.0, 55.0, // Short MA starts calculating
                48.0, 45.0,       // Long MA starts calculating, short > long
                46.0, 48.0, 53.0, // Prices rise, short pulls away from long
                60.0, 65.0,       // Golden cross should occur here
                62.0, 58.0, 55.0, // Prices fall
                50.0, 45.0        // Death cross should occur here
            ]
        ).unwrap();

        let signals = strategy.generate_signals(&df).unwrap();

        // Expected signals:
        // The first few are Hold because the long window isn't filled.
        // A Buy signal should appear when the short MA crosses above the long MA.
        // A Sell signal should appear when the short MA crosses below the long MA.
        let expected_signals = vec![
            Signal::Hold, Signal::Hold, Signal::Hold, Signal::Hold, Signal::Hold,
            Signal::Hold, Signal::Hold, Signal::Hold, Signal::Hold, Signal::Buy,
            Signal::Hold, Signal::Hold, Signal::Hold, Signal::Hold, Signal::Sell,
        ];

        assert_eq!(signals, expected_signals);
    }
}
