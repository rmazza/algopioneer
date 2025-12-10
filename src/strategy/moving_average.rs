use crate::strategy::Signal;
use polars::prelude::*;
use std::collections::VecDeque;

/// Configuration for the Moving Average Crossover strategy.
pub struct MovingAverageCrossover {
    short_window: usize,
    long_window: usize,
    // OW1: State for incremental calculation
    history: VecDeque<f64>,
    short_sum: f64,
    long_sum: f64,
    prev_short_ma: f64,
    prev_long_ma: f64,
}

impl MovingAverageCrossover {
    /// Creates a new Moving Average Crossover strategy configuration.
    pub fn new(short_window: usize, long_window: usize) -> Self {
        assert!(
            short_window < long_window,
            "Short window must be less than long window"
        );
        Self {
            short_window,
            long_window,
            history: VecDeque::with_capacity(long_window + 1),
            short_sum: 0.0,
            long_sum: 0.0,
            prev_short_ma: 0.0,
            prev_long_ma: 0.0,
        }
    }

    /// Warms up the strategy with historical data.
    pub fn warmup(&mut self, prices: &[f64]) {
        for &price in prices {
            self.update(price, false);
        }
    }

    /// Updates the strategy with a new price and returns the signal.
    /// Uses incremental calculation for O(1) complexity.
    pub fn update(&mut self, price: f64, position_open: bool) -> Signal {
        // 1. Update History
        self.history.push_back(price);

        // 2. Update Sums
        self.short_sum += price;
        self.long_sum += price;

        // 3. Remove old values if window exceeded
        if self.history.len() > self.short_window {
            let removed = self.history[self.history.len() - 1 - self.short_window];
            self.short_sum -= removed;
        }
        if self.history.len() > self.long_window {
            let removed = self.history.pop_front().unwrap_or(0.0);
            self.long_sum -= removed;
        }

        // 4. Check if we have enough data
        if self.history.len() < self.long_window {
            return Signal::Hold;
        }

        // 5. Calculate MAs
        let short_ma = self.short_sum / self.short_window as f64;
        let long_ma = self.long_sum / self.long_window as f64;

        // 6. Check Crossover
        let mut signal = Signal::Hold;

        // Only check if we have a valid previous MA (not the first tick)
        if self.prev_long_ma != 0.0 {
            // Golden Cross (Buy Signal)
            if self.prev_short_ma <= self.prev_long_ma && short_ma > long_ma && !position_open {
                signal = Signal::Buy;
            }
            // Death Cross (Sell Signal)
            else if self.prev_short_ma >= self.prev_long_ma && short_ma < long_ma && position_open
            {
                signal = Signal::Sell;
            }
        }

        // 7. Update Previous MAs
        self.prev_short_ma = short_ma;
        self.prev_long_ma = long_ma;

        signal
    }

    /// Generates a trading signal for the latest data point.
    pub fn get_latest_signal(
        &self,
        data: &DataFrame,
        position_open: bool,
    ) -> Result<Signal, PolarsError> {
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

        #[allow(clippy::needless_range_loop)]
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

// ============================================================================
// LIVE TRADING WRAPPER
// ============================================================================

use async_trait::async_trait;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::exchange::Executor;
use crate::strategy::dual_leg_trading::OrderSide;
use crate::strategy::{LiveStrategy, StrategyInput};

/// Live trading wrapper for MovingAverageCrossover
/// Implements LiveStrategy trait for supervision
pub struct MovingAverageStrategy<E: Executor> {
    /// Unique identifier for this strategy instance
    id: String,
    /// The underlying signal generator
    strategy: MovingAverageCrossover,
    /// Symbol to trade
    symbol: String,
    /// Order size for trades
    order_size: Decimal,
    /// Executor for placing orders
    executor: Arc<E>,
    /// Current position status
    position_open: bool,
    /// Realized PnL
    realized_pnl: Decimal,
    /// Entry price (for PnL calculation)
    entry_price: Option<Decimal>,
}

impl<E: Executor + 'static> MovingAverageStrategy<E> {
    pub fn new(
        id: String,
        symbol: String,
        short_window: usize,
        long_window: usize,
        order_size: Decimal,
        executor: Arc<E>,
    ) -> Self {
        Self {
            id,
            strategy: MovingAverageCrossover::new(short_window, long_window),
            symbol,
            order_size,
            executor,
            position_open: false,
            realized_pnl: Decimal::ZERO,
            entry_price: None,
        }
    }

    /// Warmup the strategy with historical data
    pub fn warmup(&mut self, prices: &[f64]) {
        self.strategy.warmup(prices);
    }
}

#[async_trait]
impl<E: Executor + 'static> LiveStrategy for MovingAverageStrategy<E> {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn subscribed_symbols(&self) -> Vec<String> {
        vec![self.symbol.clone()]
    }

    fn strategy_type(&self) -> &'static str {
        "MovingAverageCrossover"
    }

    fn current_pnl(&self) -> Decimal {
        self.realized_pnl
    }

    fn is_healthy(&self) -> bool {
        true
    }

    async fn run(&mut self, mut data_rx: mpsc::Receiver<StrategyInput>) {
        info!(
            strategy_id = %self.id,
            symbol = %self.symbol,
            "MovingAverageStrategy starting"
        );

        while let Some(input) = data_rx.recv().await {
            let tick = match input {
                StrategyInput::Tick(t) => t,
                StrategyInput::PairedTick { .. } => {
                    warn!("MovingAverageStrategy received PairedTick, expected single Tick");
                    continue;
                }
            };

            // Only process ticks for our symbol
            if tick.symbol != self.symbol {
                continue;
            }

            // CR-Refactor: Explicit handling instead of unwrap_or(0.0) sentinel
            let price_f64 = match tick.price.to_f64() {
                Some(p) if p > 0.0 => p,
                _ => {
                    warn!(
                        strategy_id = %self.id,
                        price = %tick.price,
                        "Price conversion failed or invalid, skipping tick"
                    );
                    continue;
                }
            };
            let signal = self.strategy.update(price_f64, self.position_open);

            match signal {
                Signal::Buy if !self.position_open => {
                    info!(
                        strategy_id = %self.id,
                        symbol = %self.symbol,
                        price = %tick.price,
                        "Buy signal - placing order"
                    );

                    match self.executor.execute_order(
                        &self.symbol,
                        OrderSide::Buy,
                        self.order_size,
                        Some(tick.price),
                    ).await {
                        Ok(_) => {
                            self.position_open = true;
                            self.entry_price = Some(tick.price);
                            info!(strategy_id = %self.id, "Buy order executed");
                        }
                        Err(e) => {
                            warn!(strategy_id = %self.id, error = %e, "Buy order failed");
                        }
                    }
                }
                Signal::Sell if self.position_open => {
                    info!(
                        strategy_id = %self.id,
                        symbol = %self.symbol,
                        price = %tick.price,
                        "Sell signal - placing order"
                    );

                    match self.executor.execute_order(
                        &self.symbol,
                        OrderSide::Sell,
                        self.order_size,
                        Some(tick.price),
                    ).await {
                        Ok(_) => {
                            // Calculate PnL
                            if let Some(entry) = self.entry_price {
                                let pnl = (tick.price - entry) * self.order_size;
                                self.realized_pnl += pnl;
                                info!(
                                    strategy_id = %self.id,
                                    pnl = %pnl,
                                    total_pnl = %self.realized_pnl,
                                    "Position closed"
                                );
                            }
                            self.position_open = false;
                            self.entry_price = None;
                        }
                        Err(e) => {
                            warn!(strategy_id = %self.id, error = %e, "Sell order failed");
                        }
                    }
                }
                _ => {
                    debug!(
                        strategy_id = %self.id,
                        signal = ?signal,
                        price = %tick.price,
                        "No action"
                    );
                }
            }
        }

        info!(
            strategy_id = %self.id,
            final_pnl = %self.realized_pnl,
            "MovingAverageStrategy stopped"
        );
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
        )
        .unwrap();

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
            Signal::Hold,
            Signal::Hold,
            Signal::Hold,
            Signal::Hold,
            Signal::Hold,
            Signal::Hold,
            Signal::Hold,
            Signal::Buy,
            Signal::Hold,
            Signal::Hold,
            Signal::Hold,
            Signal::Hold,
            Signal::Sell,
            Signal::Hold,
            Signal::Hold,
        ];

        assert_eq!(signals, expected_signals);
    }

    #[test]
    fn test_no_crossover() {
        let strategy = MovingAverageCrossover::new(3, 5);
        let df =
            df!("close" => &[50.0, 52.0, 55.0, 58.0, 60.0, 62.0, 65.0, 68.0, 70.0, 72.0]).unwrap();

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

        // The first sell must happen before the second buy.
        assert!(
            first_sell_pos < last_buy_pos,
            "Sell should happen before the second buy"
        );
    }

    #[test]
    fn test_incremental_update() {
        let mut strategy = MovingAverageCrossover::new(3, 5);
        let prices = vec![
            50.0, 52.0, 55.0, // Short MA starts calculating
            48.0, 45.0, // Long MA starts calculating, short > long
            46.0, 48.0, 53.0, // Prices rise, short pulls away from long
            60.0, 65.0, // Golden cross should occur here
            62.0, 58.0, 55.0, // Prices fall
            50.0, 45.0, // Death cross should occur here
        ];

        let mut signals = Vec::new();
        let mut position_open = false;
        for &price in &prices {
            let signal = strategy.update(price, position_open);
            signals.push(signal);
            if signal == Signal::Buy {
                position_open = true;
            } else if signal == Signal::Sell {
                position_open = false;
            }
        }

        // Expected signals (same as batch test but incremental)
        // Note: update() returns signal for current tick.
        // Batch test output was:
        // Hold, Hold, Hold, Hold, Hold, Hold, Hold, Buy, Hold, Hold, Hold, Hold, Sell, Hold, Hold

        // Let's verify specific indices
        assert_eq!(signals[7], Signal::Buy, "Should buy at index 7");
        assert_eq!(signals[12], Signal::Sell, "Should sell at index 12");
    }
}

