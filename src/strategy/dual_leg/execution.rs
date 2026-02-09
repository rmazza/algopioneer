//! Execution engine and recovery worker for dual-leg trading.
//!
//! This module contains:
//! - `ExecutionEngine`: Handles concurrent execution of orders on both legs with circuit breaker protection
//! - `RecoveryWorker`: Processes recovery tasks (failed legs) with structured concurrency
//! - `perform_recovery_with_backoff`: Retry logic for failed order recovery

use rust_decimal::Decimal;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Semaphore};
use tracing::{error, info, instrument};

use crate::exchange::Executor;
use crate::resilience::CircuitBreaker;
use crate::types::OrderSide;

// Re-export types from dual_leg_trading that are closely coupled
// Re-export types from parent module that are closely coupled
pub use super::{
    ExecutionError, ExecutionResult, InstrumentPair, PositionDirection, RecoveryResult,
    RecoveryTask,
};

// Import LogThrottle for recovery task logging
use crate::logging::throttle::LogThrottle;

// AS3: Priority system for recovery tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Critical = 0, // Kill switches (highest priority)
    High = 1,     // Large positions
    Normal = 2,   // Regular retries
}

// N2: Named constant instead of magic number
const MAX_CONCURRENT_RECOVERIES: usize = 5;

// Maximum recovery attempts before abandoning
const MAX_RECOVERY_ATTEMPTS: u32 = 5;

// Recovery backoff cap in seconds
const RECOVERY_BACKOFF_CAP_SECS: u64 = 60;

/// Worker that processes recovery tasks (failed legs) with structured concurrency.
///
/// Uses `tokio::task::JoinSet` to ensure all spawned tasks are properly tracked
/// and cancelled on shutdown. This prevents "orphaned tasks" that could continue
/// executing after the worker is dropped.
pub struct RecoveryWorker {
    client: Arc<dyn Executor>,
    rx: mpsc::Receiver<RecoveryTask>,
    feedback_tx: mpsc::Sender<RecoveryResult>,
    semaphore: Arc<Semaphore>,
}

impl RecoveryWorker {
    pub fn new(
        client: Arc<dyn Executor>,
        rx: mpsc::Receiver<RecoveryTask>,
        feedback_tx: mpsc::Sender<RecoveryResult>,
    ) -> Self {
        Self {
            client,
            rx,
            feedback_tx,
            semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_RECOVERIES)),
        }
    }

    /// Run the recovery worker with structured concurrency.
    ///
    /// Uses `JoinSet` to track all spawned recovery tasks. On shutdown:
    /// - All pending tasks are automatically cancelled
    /// - No orphaned tasks continue executing
    #[instrument(skip(self), name = "recovery_worker")]
    pub async fn run(mut self) {
        use tokio::task::JoinSet;
        use tracing::Instrument;

        info!("Recovery Worker started.");

        // Structured concurrency: Own all spawned tasks
        let mut active_recoveries: JoinSet<()> = JoinSet::new();

        loop {
            tokio::select! {
                // Accept new tasks from the channel
                task_opt = self.rx.recv() => {
                    match task_opt {
                        Some(task) => {
                            let client = self.client.clone();
                            let feedback_tx = self.feedback_tx.clone();
                            let semaphore = self.semaphore.clone();
                            let span = tracing::info_span!("recovery_task", symbol = %task.symbol);

                            // MC-2 FIX: Acquire semaphore BEFORE spawning to prevent task explosion.
                            // Previously, unlimited tasks could be spawned and all wait for permits.
                            // Now we apply backpressure at the channel level by blocking here.
                            let permit = match semaphore.acquire_owned().await {
                                Ok(p) => p,
                                Err(_) => {
                                    // Semaphore closed - worker shutting down
                                    info!("Semaphore closed, dropping recovery task for {}", task.symbol);
                                    continue;
                                }
                            };

                            active_recoveries.spawn(
                                async move {
                                    // Permit is moved into the task and released when dropped
                                    let _permit = permit;
                                    perform_recovery_with_backoff(client, task, feedback_tx).await;
                                }
                                .instrument(span)
                            );
                        }
                        None => {
                            // Channel closed - initiate graceful shutdown
                            info!(
                                "Recovery channel closed. Shutting down {} active tasks.",
                                active_recoveries.len()
                            );
                            break;
                        }
                    }
                }

                // Reap completed tasks (handles panics gracefully)
                Some(result) = active_recoveries.join_next() => {
                    if let Err(join_err) = result {
                        if join_err.is_panic() {
                            error!("CRITICAL: Recovery task panicked: {:?}", join_err);
                        }
                        // Cancelled tasks are expected during shutdown, don't log them
                    }
                }
            }
        }

        // Graceful Shutdown: Wait for all active recoveries to complete
        // (or cancel them if the runtime is shutting down)
        info!(
            "Waiting for {} active recovery tasks to complete...",
            active_recoveries.len()
        );
        while let Some(result) = active_recoveries.join_next().await {
            if let Err(join_err) = result {
                if join_err.is_panic() {
                    error!("Recovery task panicked during shutdown: {:?}", join_err);
                }
            }
        }
        info!("Recovery Worker shutdown complete.");
    }
}

/// Perform recovery with exponential backoff
pub async fn perform_recovery_with_backoff(
    client: Arc<dyn Executor>,
    mut task: RecoveryTask,
    feedback_tx: mpsc::Sender<RecoveryResult>,
) {
    let mut backoff = Duration::from_secs(2);
    let mut task_throttler = LogThrottle::new(Duration::from_secs(5));

    info!("Processing Recovery Task: {:?}", task);

    for attempt in 1..=MAX_RECOVERY_ATTEMPTS {
        task.attempts = attempt;

        // MC-1 FIX: Use limit_price if available to prevent slippage during market crashes.
        // If limit_price is None, falls back to market order (legacy behavior).
        match client
            .execute_order(&task.symbol, task.action, task.quantity, task.limit_price)
            .await
        {
            Ok(_) => {
                info!(
                    "Recovery Successful for {} on attempt {}",
                    task.symbol, attempt
                );
                let _ = feedback_tx
                    .send(RecoveryResult::Success(task.symbol.clone()))
                    .await;
                return;
            }
            Err(e) => {
                if task_throttler.should_log() {
                    let suppressed = task_throttler.get_and_reset_suppressed_count();
                    error!(
                        "Recovery Failed for {} (Attempt {}): {} (Suppressed: {})",
                        task.symbol, attempt, e, suppressed
                    );
                }

                if attempt >= MAX_RECOVERY_ATTEMPTS {
                    error!(
                        "CRITICAL: Recovery abandoned for {} after {} attempts. MANUAL INTERVENTION REQUIRED.",
                        task.symbol, MAX_RECOVERY_ATTEMPTS
                    );
                    let _ = feedback_tx
                        .send(RecoveryResult::Failed(task.symbol.clone()))
                        .await;
                    return;
                }

                tokio::time::sleep(backoff).await;
                backoff =
                    std::cmp::min(backoff * 2, Duration::from_secs(RECOVERY_BACKOFF_CAP_SECS));
            }
        }
    }
}

/// Handles concurrent execution of orders on both legs with circuit breaker protection.
///
/// # MC-3 NOTE on Circuit Breaker Sharing
/// The `CircuitBreaker` is owned by this `ExecutionEngine` instance. If you wrap
/// `ExecutionEngine` in an `Arc` and share it across multiple trading pairs, the circuit
/// breaker state will be shared - a failure on one pair will affect all pairs.
///
/// **Recommended usage:** Create one `ExecutionEngine` per trading pair to isolate
/// circuit breaker state. For portfolio strategies, each pair should have its own engine.
pub struct ExecutionEngine {
    client: Arc<dyn Executor>,
    recovery_tx: mpsc::Sender<RecoveryTask>,
    circuit_breaker: CircuitBreaker,
}

impl ExecutionEngine {
    /// Creates a new ExecutionEngine with the specified executor, recovery channel, and circuit breaker settings.
    pub fn new(
        client: Arc<dyn Executor>,
        recovery_tx: mpsc::Sender<RecoveryTask>,
        failure_threshold: u32,
        timeout_secs: u64,
    ) -> Self {
        Self {
            client,
            recovery_tx,
            circuit_breaker: CircuitBreaker::new(
                failure_threshold,
                Duration::from_secs(timeout_secs),
            ),
        }
    }

    /// Refactor Challenge: DRY helper for queuing kill switch recovery tasks.
    /// Handles partial failure by queuing a recovery task and recording circuit breaker failure.
    /// Returns PartialFailure on success, CriticalFailure if recovery queue fails.
    async fn queue_kill_switch(
        &self,
        failed_result: ExecutionError,
        successful_leg_symbol: &str,
        recovery_action: OrderSide,
        quantity: Decimal,
        limit_price: Option<Decimal>,
        context: &str,
    ) -> Result<ExecutionResult, ExecutionError> {
        error!(
            "CRITICAL: {} failed. {} leg succeeded, queuing kill switch on {}.",
            context,
            if recovery_action == OrderSide::Buy {
                "Spot"
            } else {
                "Future"
            },
            successful_leg_symbol
        );

        let task = RecoveryTask {
            symbol: successful_leg_symbol.to_string(),
            action: recovery_action,
            quantity,
            limit_price,
            reason: format!("{}: {}", context, failed_result),
            attempts: 0,
        };

        if let Err(send_err) = self.recovery_tx.send(task).await {
            error!(
                "EMERGENCY: Failed to queue recovery task! Manual intervention required. Error: {}",
                send_err
            );
            self.circuit_breaker.record_failure();
            return Err(ExecutionError::CriticalFailure(format!(
                "Recovery queue failure - {} kill switch failed: {}",
                context, send_err
            )));
        }

        self.circuit_breaker.record_failure();
        Ok(ExecutionResult::PartialFailure(failed_result))
    }

    /// Refactor Challenge: DRY helper for queuing exit retry recovery tasks.
    async fn queue_exit_retry(
        &self,
        symbol: &str,
        action: OrderSide,
        quantity: Decimal,
        limit_price: Option<Decimal>,
        error: ExecutionError,
        leg_name: &str,
    ) -> Result<(), ExecutionError> {
        error!("Exit {} failed: {}. Queuing retry.", leg_name, error);

        let task = RecoveryTask {
            symbol: symbol.to_string(),
            action,
            quantity,
            limit_price,
            reason: format!("Exit {} failed: {}", leg_name, error),
            attempts: 0,
        };

        if let Err(send_err) = self.recovery_tx.send(task).await {
            error!(
                "EMERGENCY: Failed to queue exit {} recovery task! Error: {}",
                leg_name, send_err
            );
            return Err(ExecutionError::CriticalFailure(format!(
                "Recovery queue failure - exit {} retry failed: {}",
                leg_name, send_err
            )));
        }

        Ok(())
    }

    /// Execute a dual-leg entry (Long or Short) based on direction.
    ///
    /// - **Long Entry**: Buy Spot, Sell Future
    /// - **Short Entry**: Sell Spot, Buy Future
    ///
    /// This is a unified method consolidating execute_basis_entry and execute_basis_short_entry.
    #[instrument(skip(self))]
    pub async fn execute_basis_entry(
        &self,
        pair: &InstrumentPair,
        direction: PositionDirection,
        quantity: Decimal,
        hedge_qty: Decimal,
        leg1_price: Decimal,
        leg2_price: Decimal,
    ) -> Result<ExecutionResult, ExecutionError> {
        // CR1: Check Circuit Breaker
        if self.circuit_breaker.is_open() {
            return Err(ExecutionError::CircuitBreakerOpen(
                "Circuit breaker is OPEN due to recent failures".to_string(),
            ));
        }

        // N-1 FIX: Skip execution if quantity is zero
        if quantity.is_zero() || hedge_qty.is_zero() {
            tracing::warn!(
                quantity = %quantity,
                hedge_qty = %hedge_qty,
                direction = %direction,
                "Skipping entry with zero quantity"
            );
            return Ok(ExecutionResult::Success);
        }

        // Direction-aware order sides:
        // Long Entry: Buy Spot (leg1), Sell Future (leg2)
        // Short Entry: Sell Spot (leg1), Buy Future (leg2)
        let (spot_side, future_side) = match direction {
            PositionDirection::Long => (OrderSide::Buy, OrderSide::Sell),
            PositionDirection::Short => (OrderSide::Sell, OrderSide::Buy),
        };

        // Concurrently execute both legs to minimize leg risk
        let spot_leg =
            self.client
                .execute_order(&pair.spot_symbol, spot_side, quantity, Some(leg1_price));
        let future_leg = self.client.execute_order(
            &pair.future_symbol,
            future_side,
            hedge_qty,
            Some(leg2_price),
        );

        let (spot_res, future_res) = tokio::join!(spot_leg, future_leg);

        // Convert errors to ExecutionError
        let spot_res = spot_res.map_err(ExecutionError::from);
        let future_res = future_res.map_err(ExecutionError::from);

        // CRITICAL FIX: Kill switch action must be OPPOSITE of entry action to unwind.
        // If spot failed and we SOLD future, we must BUY future to close the short.
        // If future failed and we BOUGHT spot, we must SELL spot to close the long.
        let unwind_future_side = future_side.opposite(); // If spot fails, unwind future
        let unwind_spot_side = spot_side.opposite(); // If future fails, unwind spot

        let direction_name = match direction {
            PositionDirection::Long => "entry",
            PositionDirection::Short => "short entry",
        };

        // CB-2 FIX: Handle double failure explicitly to preserve both error messages.
        // This ensures complete telemetry when both legs fail simultaneously.
        match (&spot_res, &future_res) {
            (Err(spot_e), Err(fut_e)) => {
                // Both legs failed - no recovery needed, but record both errors
                error!(
                    spot_error = %spot_e,
                    future_error = %fut_e,
                    "Both legs failed during {} - no positions opened",
                    direction_name
                );
                self.circuit_breaker.record_failure();
                return Ok(ExecutionResult::TotalFailure(ExecutionError::ExchangeError(
                    format!("Both legs failed. Spot: {}; Future: {}", spot_e, fut_e),
                )));
            }
            (Err(e), Ok(_)) => {
                // Spot failed, future succeeded - unwind future
                // Calculate aggressive limit price (1% worse for slippage tolerance)
                let unwind_limit = if unwind_future_side == OrderSide::Buy {
                    Some(leg2_price * rust_decimal_macros::dec!(1.01)) // Buy at up to 1% higher
                } else {
                    Some(leg2_price * rust_decimal_macros::dec!(0.99)) // Sell at up to 1% lower
                };
                return self
                    .queue_kill_switch(
                        e.clone(),
                        &pair.future_symbol,
                        unwind_future_side,
                        hedge_qty,
                        unwind_limit,
                        &format!("Spot {}", direction_name),
                    )
                    .await;
            }
            (Ok(_), Err(e)) => {
                // Future failed, spot succeeded - unwind spot
                // Calculate aggressive limit price (1% worse for slippage tolerance)
                let unwind_limit = if unwind_spot_side == OrderSide::Buy {
                    Some(leg1_price * rust_decimal_macros::dec!(1.01)) // Buy at up to 1% higher
                } else {
                    Some(leg1_price * rust_decimal_macros::dec!(0.99)) // Sell at up to 1% lower
                };
                return self
                    .queue_kill_switch(
                        e.clone(),
                        &pair.spot_symbol,
                        unwind_spot_side,
                        quantity,
                        unwind_limit,
                        &format!("Future {}", direction_name),
                    )
                    .await;
            }
            (Ok(_), Ok(_)) => {
                // Both succeeded - happy path handled below
            }
        }

        self.circuit_breaker.record_success();
        Ok(ExecutionResult::Success)
    }

    #[instrument(skip(self))]
    pub async fn execute_basis_exit(
        &self,
        pair: &InstrumentPair,
        direction: PositionDirection,
        quantity: Decimal,
        hedge_qty: Decimal,
        leg1_price: Decimal,
        leg2_price: Decimal,
    ) -> Result<ExecutionResult, ExecutionError> {
        // CR1: Check Circuit Breaker
        if self.circuit_breaker.is_open() {
            return Err(ExecutionError::CircuitBreakerOpen(
                "Circuit breaker is OPEN due to recent failures".to_string(),
            ));
        }

        // CRITICAL FIX: Direction-aware order sides
        // Long exit: Sell Spot (close long), Buy Future (close short)
        // Short exit: Buy Spot (close short), Sell Future (close long)
        let (spot_side, future_side) = match direction {
            PositionDirection::Long => (OrderSide::Sell, OrderSide::Buy),
            PositionDirection::Short => (OrderSide::Buy, OrderSide::Sell),
        };

        // BUG FIX: Skip execution if quantity is zero
        let spot_leg = if quantity > Decimal::ZERO {
            Some(self.client.execute_order(
                &pair.spot_symbol,
                spot_side,
                quantity,
                Some(leg1_price),
            ))
        } else {
            None
        };

        let future_leg = if hedge_qty > Decimal::ZERO {
            Some(self.client.execute_order(
                &pair.future_symbol,
                future_side,
                hedge_qty,
                Some(leg2_price),
            ))
        } else {
            None
        };

        // If both are None, we have nothing to do
        if spot_leg.is_none() && future_leg.is_none() {
            return Ok(ExecutionResult::Success);
        }

        // Execute concurrently if both exist, or just await the one that exists
        let (spot_res, future_res) = match (spot_leg, future_leg) {
            (Some(s), Some(f)) => {
                let (s_res, f_res) = tokio::join!(s, f);
                (Some(s_res), Some(f_res))
            }
            (Some(s), None) => (Some(s.await), None),
            (None, Some(f)) => (None, Some(f.await)),
            (None, None) => unreachable!(),
        };

        // Process results
        let mut failed = false;

        let spot_err = if let Some(res) = spot_res {
            match res.map_err(ExecutionError::from) {
                Ok(_) => None,
                Err(e) => {
                    failed = true;
                    Some(e)
                }
            }
        } else {
            None
        };

        let future_err = if let Some(res) = future_res {
            match res.map_err(ExecutionError::from) {
                Ok(_) => None,
                Err(e) => {
                    failed = true;
                    Some(e)
                }
            }
        } else {
            None
        };

        if failed {
            // If both failed (and both existed and were attempted)
            if spot_err.is_some() && future_err.is_some() {
                self.circuit_breaker.record_failure();
                return Ok(ExecutionResult::TotalFailure(
                    ExecutionError::ExchangeError(format!(
                        "Both legs failed to exit. Spot: {:?}, Future: {:?}",
                        spot_err, future_err
                    )),
                ));
            }

            // If one failed (or the only one attempted failed)
            if let Some(e) = spot_err {
                // Calculate aggressive limit price for spot recovery
                let spot_limit = if spot_side == OrderSide::Buy {
                    Some(leg1_price * rust_decimal_macros::dec!(1.01)) // Buy at up to 1% higher
                } else {
                    Some(leg1_price * rust_decimal_macros::dec!(0.99)) // Sell at up to 1% lower
                };
                self.queue_exit_retry(&pair.spot_symbol, spot_side, quantity, spot_limit, e, "Spot")
                    .await?;
            }

            if let Some(e) = future_err {
                // Calculate aggressive limit price for future recovery
                let future_limit = if future_side == OrderSide::Buy {
                    Some(leg2_price * rust_decimal_macros::dec!(1.01)) // Buy at up to 1% higher
                } else {
                    Some(leg2_price * rust_decimal_macros::dec!(0.99)) // Sell at up to 1% lower
                };
                self.queue_exit_retry(&pair.future_symbol, future_side, hedge_qty, future_limit, e, "Future")
                    .await?;
            }

            self.circuit_breaker.record_failure();
            return Ok(ExecutionResult::PartialFailure(
                ExecutionError::ExchangeError("Partial Exit Failure".to_string()),
            ));
        }

        self.circuit_breaker.record_success();
        Ok(ExecutionResult::Success)
    }

    /// Query current position from exchange (for state reconciliation)
    pub async fn get_position(
        &self,
        symbol: &str,
    ) -> Result<Decimal, crate::exchange::ExchangeError> {
        self.client.get_position(symbol).await
    }

    /// Check market hours status (for Alpaca equity trading)
    pub fn check_market_hours(
        &self,
    ) -> impl std::future::Future<Output = Result<bool, crate::exchange::ExchangeError>> + Send + '_
    {
        self.client.check_market_hours()
    }
}
