use polars::prelude::*;
use crate::strategy::Signal;



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

    /// Generates a trading signal for the latest data point.
    pub fn get_latest_signal(&self, data: &DataFrame, position_open: bool) -> Result<Signal, PolarsError> {
        let close_series = data.column("close")?;
        let close = close_series.f64()?;

        let short_opts = RollingOptionsFixedWindow {
            window_size: self.short_window,
            min_periods: self.short_window,
            weights: None,
            center: false,
            fn_params: None,
        };

        let long_opts = RollingOptionsFixedWindow {
            window_size: self.long_window,
            min_periods: self.long_window,
            weights: None,
            center: false,
            fn_params: None,
        };

        let close_owned = close.clone().into_series();
        let short_series = close_owned.rolling_mean(short_opts)?;
        let long_series = close_owned.rolling_mean(long_opts)?;

        let short_ca = short_series.f64()?;
        let long_ca = long_series.f64()?;

        let i = data.height() - 1;
        if i < 1 {
            return Ok(Signal::Hold);
        }

        let short_prev = short_ca.get(i - 1).unwrap_or_default();
        let long_prev = long_ca.get(i - 1).unwrap_or_default();
        let short_curr = short_ca.get(i).unwrap_or_default();
        let long_curr = long_ca.get(i).unwrap_or_default();

        // Golden Cross (Buy Signal)
        if short_prev <= long_prev && short_curr > long_curr && !position_open {
            return Ok(Signal::Buy);
        }
        // Death Cross (Sell Signal)
        else if short_prev >= long_prev && short_curr < long_curr && position_open {
            return Ok(Signal::Sell);
        }

        Ok(Signal::Hold)
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
        //
        // UPDATE: Removed shift to align with live execution. Signals are now generated
        // immediately when the crossover is detected.

        for i in self.long_window..data.height() {
            let short_prev = short_ca.get(i - 1).unwrap_or_default();
            let long_prev = long_ca.get(i - 1).unwrap_or_default();
            let short_curr = short_ca.get(i).unwrap_or_default();
            let long_curr = long_ca.get(i).unwrap_or_default();

            // Golden Cross (Buy Signal)
            if short_prev <= long_prev && short_curr > long_curr && !position_open {
                signals[i] = Signal::Buy;
                position_open = true;
            }
            // Death Cross (Sell Signal)
            else if short_prev >= long_prev && short_curr < long_curr && position_open {
                signals[i] = Signal::Sell;
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
        //
        // With shift removed:
        // Buy at index 7 (Price 53.0)
        // Sell at index 12 (Price 55.0)
        let expected_signals = vec![
            Signal::Hold, Signal::Hold, Signal::Hold, Signal::Hold, Signal::Hold,
            Signal::Hold, Signal::Hold, Signal::Buy, Signal::Hold, Signal::Hold,
            Signal::Hold, Signal::Hold, Signal::Sell, Signal::Hold, Signal::Hold,
        ];

        assert_eq!(signals, expected_signals);
    }

    #[test]
    fn test_no_crossover() {
        let strategy = MovingAverageCrossover::new(3, 5);
        let df = df!("close" => &[50.0, 52.0, 55.0, 58.0, 60.0, 62.0, 65.0, 68.0, 70.0, 72.0]).unwrap();

        let signals = strategy.generate_signals(&df).unwrap();
        // No crossover, so all signals should be Hold.
        let expected_signals = vec![Signal::Hold; 10];

        assert_eq!(signals, expected_signals);
    }

    #[test]
    fn test_data_too_short() {
        let strategy = MovingAverageCrossover::new(5, 10);
        let df = df!("close" => &[50.0, 52.0, 55.0, 58.0]).unwrap(); // Length is 4, less than long_window of 10

        let signals = strategy.generate_signals(&df).unwrap();
        // Data is too short for any crossover logic to run, so all signals should be Hold.
        let expected_signals = vec![Signal::Hold; 4];

        assert_eq!(signals, expected_signals);
    }

    #[test]
    fn test_missing_close_column() {
        let strategy = MovingAverageCrossover::new(3, 5);
        let df = df!("price" => &[50.0, 52.0, 55.0, 48.0, 45.0]).unwrap();

        let result = strategy.generate_signals(&df);
        // Expect an error because the 'close' column is missing.
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_crossovers() {
        let strategy = MovingAverageCrossover::new(3, 5);
        let df = df!(
            "close" => &[
                // Initial data
                50.0, 52.0, 55.0, 48.0, 45.0,
                // First crossover (Buy)
                60.0, 65.0, 70.0, 75.0, 80.0, // Golden cross
                // Second crossover (Sell)
                70.0, 65.0, 60.0, 55.0, 50.0, // Death cross
                // Third crossover (Buy)
                60.0, 65.0, 70.0, 75.0, 80.0, // Golden cross again
            ]
        )
        .unwrap();

        let signals = strategy.generate_signals(&df).unwrap();

        // Check properties rather than a fixed vector, which is brittle.
        let buy_count = signals.iter().filter(|&s| *s == Signal::Buy).count();
        let sell_count = signals.iter().filter(|&s| *s == Signal::Sell).count();

        assert_eq!(buy_count, 2, "Should have two buy signals");
        assert_eq!(sell_count, 1, "Should have one sell signal");

        // Find first buy and first sell positions
        let first_buy_pos = signals.iter().position(|s| *s == Signal::Buy);
        let first_sell_pos = signals.iter().position(|s| *s == Signal::Sell);
        let last_buy_pos = signals.iter().rposition(|s| *s == Signal::Buy);

        assert!(first_buy_pos.is_some(), "First buy signal not found");
        assert!(first_sell_pos.is_some(), "First sell signal not found");
        assert!(last_buy_pos.is_some(), "Second buy signal not found");

        // The first buy must happen before the first sell.
        assert!(first_buy_pos < first_sell_pos, "Buy should happen before sell");
        // The first sell must happen before the second buy.
        assert!(first_sell_pos < last_buy_pos, "Sell should happen before the second buy");
    }
}
