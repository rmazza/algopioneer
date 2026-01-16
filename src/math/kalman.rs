//! Kalman Filter for dynamic hedge ratio estimation.
//!
//! Implements a simple 1D Kalman Filter that tracks the optimal hedge ratio
//! (beta) between two assets in a pairs trading strategy. This allows the
//! strategy to adapt to changing cointegration relationships in real-time.
//!
//! # Mathematical Model
//!
//! **State equation** (random walk):
//! ```text
//! β[t] = β[t-1] + w,  where w ~ N(0, Q)
//! ```
//!
//! **Observation equation**:
//! ```text
//! y[t] = β[t] * x[t] + v,  where v ~ N(0, R)
//! ```
//!
//! Where:
//! - `y[t]` is the dependent asset price (e.g., leg 2)
//! - `x[t]` is the independent asset price (e.g., leg 1)
//! - `β[t]` is the hedge ratio we're estimating
//! - `Q` is process noise (how fast beta drifts)
//! - `R` is observation noise (measurement uncertainty)
//!
//! # Usage
//!
//! ```rust
//! use algopioneer::math::KalmanHedgeRatio;
//!
//! let mut kalman = KalmanHedgeRatio::new(1.0, 1e-5, 1e-3);
//!
//! // Update with each new price pair
//! let beta = kalman.update(100.0, 98.5);  // x=100, y=98.5
//! println!("Current hedge ratio: {}", beta);
//! ```
//!
//! # References
//!
//! - Avellaneda, M. & Lee, J.H. (2010). "Statistical Arbitrage in the US Equities Market"
//! - Chan, E. (2013). "Algorithmic Trading: Winning Strategies and Their Rationale"

/// Kalman Filter for estimating dynamic hedge ratios.
///
/// Tracks the optimal hedge ratio β between two assets, adapting to
/// regime changes and cointegration drift in real-time.
///
/// # Performance
///
/// - O(1) per update (constant time, no historical data storage)
/// - Suitable for high-frequency tick-by-tick updates
#[derive(Debug, Clone)]
pub struct KalmanHedgeRatio {
    /// Current hedge ratio estimate (β)
    beta: f64,
    /// State estimation error covariance (P)
    variance: f64,
    /// Process noise covariance (Q) - controls how fast β can drift
    /// Higher Q = faster adaptation but more noise sensitivity
    process_noise: f64,
    /// Observation noise covariance (R) - measurement uncertainty
    /// Higher R = smoother estimates but slower adaptation
    obs_noise: f64,
    /// Number of updates received (for diagnostics)
    update_count: u64,
}

impl KalmanHedgeRatio {
    /// Create a new Kalman Filter for hedge ratio estimation.
    ///
    /// # Arguments
    ///
    /// * `initial_beta` - Starting hedge ratio estimate. Use 1.0 for equal-weight
    ///   or pre-compute from historical OLS regression.
    /// * `process_noise` - Q parameter. Typical range: 1e-6 to 1e-4.
    ///   Higher values allow faster adaptation to regime changes.
    /// * `obs_noise` - R parameter. Typical range: 1e-4 to 1e-2.
    ///   Higher values produce smoother but slower estimates.
    ///
    /// # Recommended Defaults
    ///
    /// For pairs trading with minute-level updates:
    /// - `process_noise = 1e-5` (moderate drift tolerance)
    /// - `obs_noise = 1e-3` (balance smoothness/responsiveness)
    pub fn new(initial_beta: f64, process_noise: f64, obs_noise: f64) -> Self {
        Self {
            beta: initial_beta,
            // Initialize with high uncertainty to allow rapid initial convergence
            variance: 1.0,
            process_noise,
            obs_noise,
            update_count: 0,
        }
    }

    /// Create a Kalman Filter with recommended defaults for pairs trading.
    ///
    /// Uses `Q = 1e-5`, `R = 1e-3` which provides a good balance for
    /// intraday pairs trading on crypto or equities.
    pub fn default_for_pairs() -> Self {
        Self::new(1.0, 1e-5, 1e-3)
    }

    /// Update the hedge ratio estimate with a new price observation.
    ///
    /// This is the core Kalman Filter update step, running in O(1) time.
    ///
    /// # Arguments
    ///
    /// * `x` - Independent variable (leg 1 price)
    /// * `y` - Dependent variable (leg 2 price)
    ///
    /// # Returns
    ///
    /// Updated hedge ratio estimate β, where ideally: `y ≈ β * x`
    ///
    /// # Numerical Stability
    ///
    /// - Guards against NaN/Inf inputs (returns current beta unchanged)
    /// - Clamps beta to [-10, 10] to prevent extreme hedge ratios during regime breaks
    /// - Floors variance at 1e-12 to prevent negative covariance from f64 errors
    pub fn update(&mut self, x: f64, y: f64) -> f64 {
        // Guard 1: Degenerate input protection
        const MIN_X: f64 = 1e-12;
        if !x.is_finite() || !y.is_finite() || x.abs() < MIN_X {
            return self.beta;
        }

        self.update_count += 1;

        // === PREDICT STEP ===
        // State prediction: β stays the same (random walk)
        // β_predicted = β_previous (no state transition matrix needed for 1D)

        // Covariance prediction: P = P + Q
        let p_predicted = self.variance + self.process_noise;

        // === UPDATE STEP ===
        // Observation matrix H = x (since y = β * x)
        // Innovation (measurement residual): y - H * β = y - β * x
        let innovation = y - self.beta * x;

        // Innovation covariance: S = H * P * H' + R = x² * P + R
        let s = x * x * p_predicted + self.obs_noise;

        // Guard 2: Prevent division instability
        if s.abs() < f64::EPSILON {
            return self.beta;
        }

        // Kalman gain: K = P * H' * S^(-1) = P * x / S
        let kalman_gain = p_predicted * x / s;

        // State update with bounds: β = β + K * innovation
        // MC-2 FIX: Clamp beta to prevent extreme hedge ratios during regime breaks
        self.beta = (self.beta + kalman_gain * innovation).clamp(-10.0, 10.0);

        // Covariance update: P = (I - K * H) * P = (1 - K * x) * P
        // Guard 3: Variance must stay positive (numerical stability floor)
        self.variance = ((1.0 - kalman_gain * x) * p_predicted).max(1e-12);

        self.beta
    }

    /// Get the current hedge ratio estimate.
    #[inline]
    pub fn get_beta(&self) -> f64 {
        self.beta
    }

    /// Get the current estimation uncertainty (variance).
    ///
    /// Lower values indicate higher confidence in the estimate.
    /// Useful for:
    /// - Confidence-weighted position sizing
    /// - Detecting when the filter has converged
    #[inline]
    pub fn get_variance(&self) -> f64 {
        self.variance
    }

    /// Get the number of updates processed.
    #[inline]
    pub fn get_update_count(&self) -> u64 {
        self.update_count
    }

    /// Check if the filter has received enough updates to be considered "warm".
    ///
    /// Returns `true` after at least `min_updates` have been processed.
    pub fn is_warmed_up(&self, min_updates: u64) -> bool {
        self.update_count >= min_updates
    }

    /// Reset the filter to initial state.
    ///
    /// Useful when a regime change is detected externally and you want
    /// to re-initialize with high uncertainty.
    pub fn reset(&mut self, new_beta: f64) {
        self.beta = new_beta;
        self.variance = 1.0;
        self.update_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_converges_to_true_beta() {
        // Simulate: y = 0.8 * x + noise
        let true_beta = 0.8;
        let mut kalman = KalmanHedgeRatio::new(1.0, 1e-5, 1e-3); // Start at wrong value

        // Feed 1000 observations with small noise
        for i in 0..1000 {
            let x = 100.0 + (i as f64 * 0.1); // Slowly rising x
            let noise = ((i * 17) % 11) as f64 / 100.0 - 0.05; // Deterministic pseudo-noise
            let y = true_beta * x + noise;
            kalman.update(x, y);
        }

        let estimated_beta = kalman.get_beta();
        assert!(
            (estimated_beta - true_beta).abs() < 0.05,
            "Kalman should converge to true beta. Expected ~{}, got {}",
            true_beta,
            estimated_beta
        );
    }

    #[test]
    fn test_kalman_tracks_drifting_beta() {
        let mut kalman = KalmanHedgeRatio::new(1.0, 1e-4, 1e-3); // Higher Q for faster tracking

        // First regime: beta = 1.0
        for i in 0..500 {
            let x = 100.0 + (i as f64 * 0.01);
            let y = 1.0 * x;
            kalman.update(x, y);
        }
        assert!(
            (kalman.get_beta() - 1.0).abs() < 0.1,
            "Should track beta=1.0, got {}",
            kalman.get_beta()
        );

        // Second regime: beta = 1.5 (sudden shift)
        for i in 0..500 {
            let x = 100.0 + (i as f64 * 0.01);
            let y = 1.5 * x;
            kalman.update(x, y);
        }
        assert!(
            (kalman.get_beta() - 1.5).abs() < 0.1,
            "Should adapt to beta=1.5, got {}",
            kalman.get_beta()
        );
    }

    #[test]
    fn test_kalman_handles_zero_x() {
        let mut kalman = KalmanHedgeRatio::new(1.0, 1e-5, 1e-3);
        let original_beta = kalman.get_beta();

        // Zero x should not crash or change beta
        let result = kalman.update(0.0, 100.0);
        assert_eq!(result, original_beta);
    }

    #[test]
    fn test_kalman_handles_nan_inf() {
        let mut kalman = KalmanHedgeRatio::new(1.0, 1e-5, 1e-3);
        let original_beta = kalman.get_beta();

        // NaN should not crash or change beta
        assert_eq!(kalman.update(f64::NAN, 100.0), original_beta);
        assert_eq!(kalman.update(100.0, f64::NAN), original_beta);

        // Infinity should not crash or change beta
        assert_eq!(kalman.update(f64::INFINITY, 100.0), original_beta);
        assert_eq!(kalman.update(100.0, f64::NEG_INFINITY), original_beta);
    }

    #[test]
    fn test_kalman_warmup() {
        let kalman = KalmanHedgeRatio::new(1.0, 1e-5, 1e-3);
        assert!(!kalman.is_warmed_up(100));

        let mut kalman = kalman;
        for i in 0..100 {
            kalman.update(100.0 + i as f64, 100.0 + i as f64);
        }
        assert!(kalman.is_warmed_up(100));
    }

    #[test]
    fn test_kalman_variance_decreases() {
        let mut kalman = KalmanHedgeRatio::new(1.0, 1e-5, 1e-3);
        let initial_variance = kalman.get_variance();

        // Feed consistent data
        for i in 0..100 {
            let x = 100.0 + i as f64;
            let y = x; // Perfect 1:1 relationship
            kalman.update(x, y);
        }

        assert!(
            kalman.get_variance() < initial_variance,
            "Variance should decrease with consistent data"
        );
    }

    #[test]
    fn test_kalman_reset() {
        let mut kalman = KalmanHedgeRatio::new(1.0, 1e-5, 1e-3);

        // Run some updates
        for _ in 0..100 {
            kalman.update(100.0, 80.0);
        }

        // Reset
        kalman.reset(2.0);
        assert_eq!(kalman.get_beta(), 2.0);
        assert_eq!(kalman.get_variance(), 1.0);
        assert_eq!(kalman.get_update_count(), 0);
    }

    #[test]
    fn test_default_for_pairs() {
        let kalman = KalmanHedgeRatio::default_for_pairs();
        assert_eq!(kalman.get_beta(), 1.0);
        assert_eq!(kalman.get_update_count(), 0);
    }
}
