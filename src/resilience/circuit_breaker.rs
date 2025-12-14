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
//! ## Performance
//! This implementation uses lock-free atomics for the hot path (`is_open()`).
//! The `is_open()` method never acquires a mutex, making it suitable for HFT.
//!
//! ## Usage
//! ```ignore
//! let breaker = CircuitBreaker::new(5, Duration::from_secs(60));
//!
//! if breaker.is_open() {
//!     return Err("Service unavailable");
//! }
//!
//! match do_risky_operation().await {
//!     Ok(_) => breaker.record_success(),
//!     Err(_) => breaker.record_failure(),
//! }
//! ```

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;
use tracing::warn;

/// State of the circuit breaker (encoded as u32 for atomic operations).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CircuitState {
    /// Normal operation - requests pass through
    Closed = 0,
    /// Blocking requests due to recent failures
    Open = 1,
    /// Testing if system recovered
    HalfOpen = 2,
}

impl CircuitState {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            2 => CircuitState::HalfOpen,
            _ => CircuitState::Closed, // Default to closed for safety
        }
    }
}

/// Lock-free Circuit Breaker implementation for HFT resilience.
///
/// Prevents cascading failures by temporarily blocking requests
/// after a threshold of consecutive failures is reached.
///
/// # Performance Characteristics
/// - `is_open()`: Lock-free, O(1) using atomic loads
/// - `record_success()`: Lock-free, O(1) using atomic stores
/// - `record_failure()`: Lock-free with CAS for state transition
pub struct CircuitBreaker {
    /// Current state: 0=Closed, 1=Open, 2=HalfOpen
    state: AtomicU32,
    /// Consecutive failure count
    failure_count: AtomicU32,
    /// Last failure time as nanoseconds since `creation_time`
    last_failure_nanos: AtomicU64,
    /// Reference time point for computing elapsed time
    creation_time: Instant,
    /// Number of consecutive failures to trip the breaker (immutable after construction)
    failure_threshold: u32,
    /// Timeout in nanoseconds before transitioning to HalfOpen (immutable after construction)
    timeout_nanos: u64,
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
    /// use std::time::Duration;
    /// let breaker = CircuitBreaker::new(5, Duration::from_secs(60));
    /// ```
    pub fn new(failure_threshold: u32, timeout: std::time::Duration) -> Self {
        Self {
            state: AtomicU32::new(CircuitState::Closed as u32),
            failure_count: AtomicU32::new(0),
            last_failure_nanos: AtomicU64::new(0),
            creation_time: Instant::now(),
            failure_threshold,
            timeout_nanos: timeout.as_nanos() as u64,
        }
    }

    /// Returns the current elapsed nanoseconds since creation.
    #[inline]
    fn elapsed_nanos(&self) -> u64 {
        self.creation_time.elapsed().as_nanos() as u64
    }

    /// Returns the current state of the circuit breaker.
    /// Lock-free read.
    pub fn get_state(&self) -> CircuitState {
        CircuitState::from_u32(self.state.load(Ordering::Acquire))
    }

    /// Returns the current failure count.
    /// Lock-free read.
    pub fn get_failure_count(&self) -> u32 {
        self.failure_count.load(Ordering::Acquire)
    }

    /// Records a successful operation, resetting the failure count and closing the circuit.
    /// Lock-free operation.
    pub fn record_success(&self) {
        self.state
            .store(CircuitState::Closed as u32, Ordering::Release);
        self.failure_count.store(0, Ordering::Release);
    }

    /// Records a failed operation. Trips the breaker if the threshold is reached.
    /// Lock-free with atomic compare-and-swap for state transition.
    pub fn record_failure(&self) {
        // Atomically increment failure count
        let new_count = self.failure_count.fetch_add(1, Ordering::AcqRel) + 1;

        // Record the failure timestamp
        self.last_failure_nanos
            .store(self.elapsed_nanos(), Ordering::Release);

        if new_count >= self.failure_threshold {
            // Use CAS to atomically transition to Open state
            // Only transition if we're in Closed or HalfOpen state
            let current = self.state.load(Ordering::Acquire);
            if current != CircuitState::Open as u32
                && self
                    .state
                    .compare_exchange(
                        current,
                        CircuitState::Open as u32,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
            {
                warn!(
                    "Circuit breaker tripped to OPEN after {} failures",
                    new_count
                );
            }
        }
    }

    /// Checks if the circuit is currently open.
    ///
    /// Returns `true` if the circuit is Open (and timeout hasn't passed).
    /// Returns `false` if the circuit is Closed or HalfOpen (ready to try).
    /// Automatically transitions from Open to HalfOpen if the timeout has elapsed.
    ///
    /// **Lock-free**: This method uses only atomic operations and never acquires a mutex.
    #[inline]
    pub fn is_open(&self) -> bool {
        let state = self.state.load(Ordering::Acquire);

        match state {
            0 | 2 => false, // Closed or HalfOpen - allow traffic
            1 => {
                // Open - check if timeout has elapsed
                let last_failure = self.last_failure_nanos.load(Ordering::Acquire);
                let now = self.elapsed_nanos();
                let elapsed = now.saturating_sub(last_failure);

                if elapsed > self.timeout_nanos {
                    // Timeout elapsed, try to transition to HalfOpen
                    // Use CAS to avoid race conditions - only one thread wins
                    if self
                        .state
                        .compare_exchange(
                            CircuitState::Open as u32,
                            CircuitState::HalfOpen as u32,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        // Successfully transitioned, reset failure count
                        self.failure_count.store(0, Ordering::Release);
                    }
                    false // Allow this request through (either we transitioned or another thread did)
                } else {
                    true // Still open, block traffic
                }
            }
            _ => true, // Unknown state, be conservative and block
        }
    }

    /// Resets the circuit breaker to its initial closed state.
    /// Lock-free operation.
    pub fn reset(&self) {
        self.state
            .store(CircuitState::Closed as u32, Ordering::Release);
        self.failure_count.store(0, Ordering::Release);
        self.last_failure_nanos.store(0, Ordering::Release);
    }

    // Async wrappers for backward compatibility with existing async code
    // These simply call the synchronous methods

    /// Async wrapper for `get_state()` (for backward compatibility).
    pub async fn get_state_async(&self) -> CircuitState {
        self.get_state()
    }

    /// Async wrapper for `get_failure_count()` (for backward compatibility).
    pub async fn get_failure_count_async(&self) -> u32 {
        self.get_failure_count()
    }

    /// Async wrapper for `record_success()` (for backward compatibility).
    pub async fn record_success_async(&self) {
        self.record_success()
    }

    /// Async wrapper for `record_failure()` (for backward compatibility).
    pub async fn record_failure_async(&self) {
        self.record_failure()
    }

    /// Async wrapper for `is_open()` (for backward compatibility).
    pub async fn is_open_async(&self) -> bool {
        self.is_open()
    }

    /// Async wrapper for `reset()` (for backward compatibility).
    pub async fn reset_async(&self) {
        self.reset()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_circuit_breaker_starts_closed() {
        let breaker = CircuitBreaker::new(3, Duration::from_secs(10));
        assert_eq!(breaker.get_state(), CircuitState::Closed);
        assert!(!breaker.is_open());
    }

    #[test]
    fn test_circuit_breaker_trips_after_threshold() {
        let breaker = CircuitBreaker::new(3, Duration::from_secs(10));

        breaker.record_failure();
        assert_eq!(breaker.get_state(), CircuitState::Closed);

        breaker.record_failure();
        assert_eq!(breaker.get_state(), CircuitState::Closed);

        breaker.record_failure();
        assert_eq!(breaker.get_state(), CircuitState::Open);
        assert!(breaker.is_open());
    }

    #[test]
    fn test_circuit_breaker_success_resets() {
        let breaker = CircuitBreaker::new(3, Duration::from_secs(10));

        breaker.record_failure();
        breaker.record_failure();
        breaker.record_success();

        assert_eq!(breaker.get_state(), CircuitState::Closed);
        assert_eq!(breaker.get_failure_count(), 0);
    }

    #[test]
    fn test_circuit_breaker_half_open_transition() {
        // Use a very short timeout for testing
        let breaker = CircuitBreaker::new(2, Duration::from_millis(1));

        // Trip the breaker
        breaker.record_failure();
        breaker.record_failure();
        assert!(breaker.is_open());
        assert_eq!(breaker.get_state(), CircuitState::Open);

        // Wait for timeout to elapse
        std::thread::sleep(Duration::from_millis(10));

        // Should transition to HalfOpen
        assert!(!breaker.is_open());
        assert_eq!(breaker.get_state(), CircuitState::HalfOpen);
    }

    #[test]
    fn test_circuit_breaker_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let breaker = Arc::new(CircuitBreaker::new(100, Duration::from_secs(60)));

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let b = Arc::clone(&breaker);
                thread::spawn(move || {
                    for _ in 0..50 {
                        b.record_failure();
                        let _ = b.is_open();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // With 10 threads * 50 failures = 500 failures, should definitely be open
        assert!(breaker.is_open());
        assert!(breaker.get_failure_count() >= 100);
    }
}
