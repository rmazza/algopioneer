//! # Circuit Breaker Pattern
//!
//! The Circuit Breaker pattern prevents cascading failures by temporarily blocking
//! requests to a failing service, allowing it time to recover.
//!
//! ## States
//! - **Closed**: Normal operation, requests pass through.
//! - **Open**: Requests are blocked after `failure_threshold` consecutive failures.
//! - **HalfOpen**: After `timeout`, allows one request to test recovery.
//!
//! ## Usage
//! ```ignore
//! let breaker = CircuitBreaker::new(5, Duration::from_secs(60));
//!
//! if breaker.is_open().await {
//!     return Err("Service unavailable");
//! }
//!
//! match do_risky_operation().await {
//!     Ok(_) => breaker.record_success().await,
//!     Err(_) => breaker.record_failure().await,
//! }
//! ```

use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{Duration, Instant};
use tracing::warn;

/// State of the circuit breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation - requests pass through
    Closed,
    /// Blocking requests due to recent failures
    Open,
    /// Testing if system recovered
    HalfOpen,
}

/// Internal state of the circuit breaker, consolidated for reduced lock contention.
#[derive(Debug)]
struct CircuitBreakerState {
    state: CircuitState,
    failure_count: u32,
    last_failure_time: Option<Instant>,
}

/// Circuit Breaker implementation for resilience.
///
/// Prevents cascading failures by temporarily blocking requests
/// after a threshold of consecutive failures is reached.
pub struct CircuitBreaker {
    inner: Arc<RwLock<CircuitBreakerState>>,
    failure_threshold: u32,
    timeout: Duration,
}

impl CircuitBreaker {
    /// Creates a new CircuitBreaker with the specified failure threshold and reset timeout.
    ///
    /// # Arguments
    /// * `failure_threshold` - Number of consecutive failures to trip the breaker.
    /// * `timeout` - Duration to wait before transitioning to HalfOpen state.
    ///
    /// # Example
    /// ```ignore
    /// let breaker = CircuitBreaker::new(5, Duration::from_secs(60));
    /// ```
    pub fn new(failure_threshold: u32, timeout: Duration) -> Self {
        Self {
            inner: Arc::new(RwLock::new(CircuitBreakerState {
                state: CircuitState::Closed,
                failure_count: 0,
                last_failure_time: None,
            })),
            failure_threshold,
            timeout,
        }
    }

    /// Returns the current state of the circuit breaker.
    pub async fn get_state(&self) -> CircuitState {
        self.inner.read().await.state
    }

    /// Returns the current failure count.
    pub async fn get_failure_count(&self) -> u32 {
        self.inner.read().await.failure_count
    }

    /// Records a successful operation, resetting the failure count and closing the circuit.
    pub async fn record_success(&self) {
        let mut state = self.inner.write().await;
        state.state = CircuitState::Closed;
        state.failure_count = 0;
    }

    /// Records a failed operation. Trips the breaker if the threshold is reached.
    pub async fn record_failure(&self) {
        let mut state = self.inner.write().await;
        state.failure_count += 1;
        state.last_failure_time = Some(Instant::now());

        if state.failure_count >= self.failure_threshold {
            state.state = CircuitState::Open;
            warn!(
                "Circuit breaker tripped to OPEN after {} failures",
                state.failure_count
            );
        }
    }

    /// Checks if the circuit is currently open.
    ///
    /// Returns `true` if the circuit is Open (and timeout hasn't passed).
    /// Returns `false` if the circuit is Closed or HalfOpen (ready to try).
    /// Automatically transitions from Open to HalfOpen if the timeout has elapsed.
    pub async fn is_open(&self) -> bool {
        // Fast path: check with read lock first
        {
            let state = self.inner.read().await;
            if state.state != CircuitState::Open {
                return false;
            }

            // Check if timeout has NOT passed
            if let Some(last_failure) = state.last_failure_time {
                if last_failure.elapsed() <= self.timeout {
                    return true;
                }
            }
        }

        // Slow path: timeout has passed, need to transition to HalfOpen
        let mut state = self.inner.write().await;
        if state.state == CircuitState::Open {
            if let Some(last_failure) = state.last_failure_time {
                if last_failure.elapsed() > self.timeout {
                    state.state = CircuitState::HalfOpen;
                    state.failure_count = 0;
                    return false;
                }
            }
            return true;
        }
        false
    }

    /// Resets the circuit breaker to its initial closed state.
    pub async fn reset(&self) {
        let mut state = self.inner.write().await;
        state.state = CircuitState::Closed;
        state.failure_count = 0;
        state.last_failure_time = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_circuit_breaker_starts_closed() {
        let breaker = CircuitBreaker::new(3, Duration::from_secs(10));
        assert_eq!(breaker.get_state().await, CircuitState::Closed);
        assert!(!breaker.is_open().await);
    }

    #[tokio::test]
    async fn test_circuit_breaker_trips_after_threshold() {
        let breaker = CircuitBreaker::new(3, Duration::from_secs(10));

        breaker.record_failure().await;
        assert_eq!(breaker.get_state().await, CircuitState::Closed);

        breaker.record_failure().await;
        assert_eq!(breaker.get_state().await, CircuitState::Closed);

        breaker.record_failure().await;
        assert_eq!(breaker.get_state().await, CircuitState::Open);
        assert!(breaker.is_open().await);
    }

    #[tokio::test]
    async fn test_circuit_breaker_success_resets() {
        let breaker = CircuitBreaker::new(3, Duration::from_secs(10));

        breaker.record_failure().await;
        breaker.record_failure().await;
        breaker.record_success().await;

        assert_eq!(breaker.get_state().await, CircuitState::Closed);
        assert_eq!(breaker.get_failure_count().await, 0);
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_transition() {
        tokio::time::pause();

        let breaker = CircuitBreaker::new(2, Duration::from_millis(100));

        // Trip the breaker
        breaker.record_failure().await;
        breaker.record_failure().await;
        assert!(breaker.is_open().await);

        // Advance past timeout
        tokio::time::advance(Duration::from_millis(150)).await;

        // Should transition to HalfOpen
        assert!(!breaker.is_open().await);
        assert_eq!(breaker.get_state().await, CircuitState::HalfOpen);
    }
}
