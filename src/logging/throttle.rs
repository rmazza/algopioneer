//! Rate-limited logging utilities.
//!
//! Provides `LogThrottle` to prevent log storms while still tracking suppressed messages.

use std::time::{Duration, Instant};

/// A lightweight rate limiter for logging to prevent log storms.
#[derive(Debug)]
pub struct LogThrottle {
    last_log_time: Option<Instant>,
    suppressed_count: u64,
    interval: Duration,
}

impl LogThrottle {
    pub fn new(interval: Duration) -> Self {
        Self {
            last_log_time: None,
            suppressed_count: 0,
            interval,
        }
    }

    /// Checks if a log should be emitted.
    /// Returns true if the interval has passed since the last log.
    /// If false, increments the suppressed counter.
    pub fn should_log(&mut self) -> bool {
        let now = Instant::now();
        match self.last_log_time {
            Some(last) => {
                if now.duration_since(last) >= self.interval {
                    self.last_log_time = Some(now);
                    true
                } else {
                    self.suppressed_count += 1;
                    false
                }
            }
            None => {
                self.last_log_time = Some(now);
                true
            }
        }
    }

    /// Returns the number of suppressed logs since the last successful log, and resets the counter.
    pub fn get_and_reset_suppressed_count(&mut self) -> u64 {
        let count = self.suppressed_count;
        self.suppressed_count = 0;
        count
    }
}

/// Container for all log throttlers used in the dual-leg strategy.
#[derive(Debug)]
pub struct DualLegLogThrottler {
    pub unstable_state: LogThrottle,
    pub tick_age: LogThrottle,
    pub sync_issue: LogThrottle,
    /// P1 FIX: Throttler for load shedding warnings (stale tick drops)
    pub latency_drop: LogThrottle,
}

impl DualLegLogThrottler {
    pub fn new(interval_secs: u64) -> Self {
        let interval = Duration::from_secs(interval_secs);
        Self {
            unstable_state: LogThrottle::new(interval),
            tick_age: LogThrottle::new(interval),
            sync_issue: LogThrottle::new(interval),
            latency_drop: LogThrottle::new(interval),
        }
    }
}
