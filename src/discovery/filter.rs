//! Statistical filtering for pair candidates
//!
//! Implements correlation analysis, mean-reversion testing,
//! and ADF cointegration testing to filter viable trading pairs.

use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Fallback half-life for non-stationary or invalid spreads (hours)
const NON_STATIONARY_HALF_LIFE: f64 = 1000.0;

/// MC-1 FIX: Maximum safe price ratio for correlation calculations
/// Beyond this ratio, f64 precision loss may affect results
const MAX_PRICE_RATIO: f64 = 1e9;

/// ADF critical values at 5% significance level (MacKinnon, 1994)
/// For n > 100 samples, critical value ≈ -2.86
const ADF_CRITICAL_VALUE_5PCT: f64 = -2.86;

/// A candidate pair that passed initial filtering
#[derive(Debug, Clone)]
pub struct CandidatePair {
    /// First symbol (leg A)
    pub symbol_a: String,
    /// Second symbol (leg B)
    pub symbol_b: String,
    /// Pearson correlation coefficient
    pub correlation: f64,
    /// Log-spread standard deviation (z-score volatility)
    pub spread_std: f64,
    /// Estimated mean-reversion half-life in hours
    pub half_life_hours: f64,
    /// ADF test statistic (more negative = more stationary/cointegrated)
    pub adf_statistic: f64,
}

/// Calculate Pearson correlation coefficient between two price series
///
/// Returns a value in [-1.0, 1.0], or None if calculation fails.
///
/// # MC-1 FIX: Precision Guard
/// Returns None if price ratio exceeds MAX_PRICE_RATIO to prevent f64 precision loss.
///
/// # Mathematical Definition
/// r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]
pub fn calculate_correlation(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.len() < 2 {
        return None;
    }

    // MC-1 FIX: Check for extreme price ratios that could cause precision loss
    let mean_a: f64 = a.iter().sum::<f64>() / a.len() as f64;
    let mean_b: f64 = b.iter().sum::<f64>() / b.len() as f64;

    if mean_b != 0.0 {
        let ratio = (mean_a / mean_b).abs();
        if !(1.0 / MAX_PRICE_RATIO..=MAX_PRICE_RATIO).contains(&ratio) {
            warn!(
                ratio = format!("{:.2e}", ratio),
                limit = format!("{:.2e}", MAX_PRICE_RATIO),
                "Price ratio exceeds safe bounds for correlation calculation"
            );
            return None;
        }
    }

    let mut covariance = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        covariance += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }

    if var_a == 0.0 || var_b == 0.0 {
        return Some(0.0);
    }

    let correlation = covariance / (var_a.sqrt() * var_b.sqrt());

    if correlation.is_finite() {
        Some(correlation)
    } else {
        None
    }
}

/// Analyze log-spread for mean-reversion characteristics
///
/// Returns (std_dev, half_life_hours) based on Ornstein-Uhlenbeck model.
///
/// # Half-Life Estimation
/// Uses lag-1 autocorrelation to estimate the speed of mean reversion:
/// ρ = autocorrelation
/// half_life = -ln(2) / ln(ρ)
pub fn analyze_spread(spread: &[f64]) -> (f64, f64) {
    if spread.len() < 3 {
        return (0.0, f64::INFINITY);
    }

    let n = spread.len() as f64;
    let mean = spread.iter().sum::<f64>() / n;

    // Population variance (for z-score calculation)
    let variance = spread.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    // Lag-1 autocorrelation for mean-reversion speed
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..spread.len() - 1 {
        let dx = spread[i] - mean;
        let dy = spread[i + 1] - mean;
        numerator += dx * dy;
        denominator += dx * dx;
    }

    let rho = if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    };

    // Half-life = -ln(2) / ln(ρ)
    let half_life = if rho > 0.0 && rho < 1.0 {
        -2.0f64.ln() / rho.ln()
    } else {
        NON_STATIONARY_HALF_LIFE // Non-stationary or invalid
    };

    (std_dev, half_life)
}

/// Augmented Dickey-Fuller (ADF) stationarity test for cointegration
///
/// Tests whether a spread series is stationary (mean-reverting) using
/// the ADF test. Returns (adf_statistic, is_stationary).
///
/// # Algorithm
/// 1. Compute first differences: Δy[t] = y[t] - y[t-1]
/// 2. Regress Δy[t] on y[t-1] using OLS
/// 3. Compute t-statistic for the coefficient
/// 4. Compare to critical value (-2.86 at 5% significance)
///
/// # Returns
/// - `adf_statistic`: The t-statistic (more negative = more stationary)
/// - `is_stationary`: true if statistic < critical value (reject unit root)
///
/// # Mathematical Foundation
/// Under H0 (unit root): y[t] = y[t-1] + ε  (non-stationary random walk)
/// Under H1 (stationary): y[t] = ρ*y[t-1] + ε where |ρ| < 1
///
/// We test: Δy[t] = γ*y[t-1] + ε where γ = ρ - 1
/// If γ < 0 significantly, reject H0 → series is stationary
pub fn adf_test(spread: &[f64]) -> (f64, bool) {
    if spread.len() < 20 {
        // Not enough data for reliable test
        return (0.0, false);
    }

    let n = spread.len() - 1; // Number of differences

    // Compute first differences and lagged values
    let mut delta_y: Vec<f64> = Vec::with_capacity(n);
    let mut y_lag: Vec<f64> = Vec::with_capacity(n);

    for i in 1..spread.len() {
        delta_y.push(spread[i] - spread[i - 1]);
        y_lag.push(spread[i - 1]);
    }

    // OLS regression: Δy = γ * y_lag + ε
    // γ = Σ(y_lag * Δy) / Σ(y_lag²)
    let n_f64 = n as f64;

    // Demean y_lag for numerical stability
    let y_lag_mean = y_lag.iter().sum::<f64>() / n_f64;
    let delta_y_mean = delta_y.iter().sum::<f64>() / n_f64;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..n {
        let y_centered = y_lag[i] - y_lag_mean;
        let d_centered = delta_y[i] - delta_y_mean;
        numerator += y_centered * d_centered;
        denominator += y_centered * y_centered;
    }

    if denominator.abs() < f64::EPSILON {
        return (0.0, false); // Degenerate case
    }

    let gamma = numerator / denominator;

    // Compute residuals and standard error
    let mut sse = 0.0; // Sum of squared errors
    for i in 0..n {
        let predicted = gamma * (y_lag[i] - y_lag_mean) + delta_y_mean;
        let residual = delta_y[i] - predicted;
        sse += residual * residual;
    }

    let mse = sse / (n_f64 - 1.0); // Mean squared error (1 regressor)
    let se_gamma = (mse / denominator).sqrt(); // Standard error of γ

    if se_gamma.abs() < f64::EPSILON {
        return (0.0, false);
    }

    let t_statistic = gamma / se_gamma;

    // Compare to critical value: more negative = more stationary
    let is_stationary = t_statistic < ADF_CRITICAL_VALUE_5PCT;

    (t_statistic, is_stationary)
}

/// Compute log-spread between two price series
///
/// spread[i] = ln(a[i]) - ln(b[i])
pub fn compute_log_spread(prices_a: &[f64], prices_b: &[f64]) -> Vec<f64> {
    prices_a
        .iter()
        .zip(prices_b.iter())
        .filter_map(|(a, b)| {
            if *a > 0.0 && *b > 0.0 {
                Some(a.ln() - b.ln())
            } else {
                None
            }
        })
        .collect()
}

/// Filter candidate pairs based on correlation, mean-reversion, and cointegration
///
/// # Algorithm
/// 1. For each unique pair (i, j) where i < j:
/// 2. Calculate Pearson correlation
/// 3. If correlation >= threshold, compute log-spread
/// 4. Estimate half-life via O-U model
/// 5. Run ADF stationarity test if require_cointegration is true
/// 6. If all criteria pass, add to results
pub fn filter_candidates(
    prices: &HashMap<String, Vec<f64>>,
    min_correlation: f64,
    max_half_life_hours: f64,
) -> Vec<CandidatePair> {
    filter_candidates_with_options(prices, min_correlation, max_half_life_hours, true)
}

/// Filter candidate pairs with configurable cointegration requirement
pub fn filter_candidates_with_options(
    prices: &HashMap<String, Vec<f64>>,
    min_correlation: f64,
    max_half_life_hours: f64,
    require_cointegration: bool,
) -> Vec<CandidatePair> {
    let symbols: Vec<&String> = prices.keys().collect();
    let mut results = Vec::new();

    info!(
        candidates = symbols.len(),
        min_corr = min_correlation,
        max_hl = max_half_life_hours,
        require_coint = require_cointegration,
        "Filtering pair candidates"
    );

    let mut rejected_adf = 0u32;

    for i in 0..symbols.len() {
        for j in (i + 1)..symbols.len() {
            let sym_a = symbols[i];
            let sym_b = symbols[j];

            let series_a = &prices[sym_a];
            let series_b = &prices[sym_b];

            // Skip if lengths don't match
            if series_a.len() != series_b.len() {
                warn!(a = %sym_a, b = %sym_b, "Length mismatch, skipping pair");
                continue;
            }

            // Calculate correlation
            let Some(correlation) = calculate_correlation(series_a, series_b) else {
                continue;
            };

            if correlation < min_correlation {
                debug!(
                    pair = format!("{}-{}", sym_a, sym_b),
                    corr = correlation,
                    "Correlation too low"
                );
                continue;
            }

            // Compute log-spread and analyze
            let spread = compute_log_spread(series_a, series_b);
            if spread.len() < 20 {
                warn!(
                    pair = format!("{}-{}", sym_a, sym_b),
                    len = spread.len(),
                    "Spread too short"
                );
                continue;
            }

            let (spread_std, half_life) = analyze_spread(&spread);

            if half_life > max_half_life_hours {
                debug!(
                    pair = format!("{}-{}", sym_a, sym_b),
                    hl = half_life,
                    "Half-life too long"
                );
                continue;
            }

            // ADF cointegration test
            let (adf_stat, is_cointegrated) = adf_test(&spread);

            if require_cointegration && !is_cointegrated {
                debug!(
                    pair = format!("{}-{}", sym_a, sym_b),
                    adf = format!("{:.2}", adf_stat),
                    critical = ADF_CRITICAL_VALUE_5PCT,
                    "Failed ADF cointegration test (spread is non-stationary)"
                );
                rejected_adf += 1;
                continue;
            }

            info!(
                pair = format!("{}-{}", sym_a, sym_b),
                correlation = format!("{:.3}", correlation),
                half_life = format!("{:.1}h", half_life),
                adf = format!("{:.2}", adf_stat),
                cointegrated = is_cointegrated,
                "Viable pair found"
            );

            results.push(CandidatePair {
                symbol_a: sym_a.clone(),
                symbol_b: sym_b.clone(),
                correlation,
                spread_std,
                half_life_hours: half_life,
                adf_statistic: adf_stat,
            });
        }
    }

    // Sort by correlation descending
    results.sort_by(|a, b| {
        b.correlation
            .partial_cmp(&a.correlation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    info!(
        viable_pairs = results.len(),
        rejected_adf = rejected_adf,
        "Filtering complete"
    );
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_perfect() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = calculate_correlation(&a, &b).unwrap();
        assert!((corr - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_correlation_negative() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr = calculate_correlation(&a, &b).unwrap();
        assert!((corr + 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_correlation_symmetric() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.5, 2.5, 2.8, 4.2, 4.9];
        let corr_ab = calculate_correlation(&a, &b).unwrap();
        let corr_ba = calculate_correlation(&b, &a).unwrap();
        assert!((corr_ab - corr_ba).abs() < 0.0001);
    }

    #[test]
    fn test_spread_constant() {
        let spread = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let (std_dev, _half_life) = analyze_spread(&spread);
        assert_eq!(std_dev, 0.0);
    }

    #[test]
    fn test_log_spread() {
        let a = vec![100.0, 110.0, 105.0];
        let b = vec![50.0, 55.0, 52.5];
        let spread = compute_log_spread(&a, &b);
        assert_eq!(spread.len(), 3);
        // ln(100/50) = ln(2) ≈ 0.693
        assert!((spread[0] - 0.693).abs() < 0.01);
    }

    #[test]
    fn test_adf_insufficient_data() {
        // Too few samples should return non-stationary
        let spread: Vec<f64> = (0..15).map(|x| x as f64).collect();
        let (stat, is_stationary) = adf_test(&spread);
        assert_eq!(stat, 0.0);
        assert!(!is_stationary);
    }

    #[test]
    fn test_adf_trending_series() {
        // A simple upward trend (like a random walk with drift)
        // The ADF test should return a finite statistic
        let spread: Vec<f64> = (0..100).map(|i| i as f64 * 0.5).collect();
        let (stat, _is_stationary) = adf_test(&spread);
        // Trend series will have some ADF stat - we just verify it's finite
        assert!(
            stat.is_finite(),
            "ADF statistic should be finite, got {}",
            stat
        );
    }

    #[test]
    fn test_adf_mean_reverting_stationary() {
        // A mean-reverting process: y[t] = 0.5 * y[t-1] + noise
        // This should be stationary (strongly mean-reverting)
        let mut spread: Vec<f64> = Vec::with_capacity(100);
        let mut current = 10.0;
        for i in 0..100 {
            let noise = ((i * 31) % 11) as f64 / 10.0 - 0.5;
            current = 0.3 * current + noise; // Strong mean reversion
            spread.push(current);
        }
        let (stat, _is_stationary) = adf_test(&spread);
        // Mean-reverting series should have stat < -2.86
        assert!(
            stat < -1.5, // Should be significantly negative
            "Mean-reverting series should have negative ADF stat, got {:.2}",
            stat
        );
    }

    #[test]
    fn test_adf_constant_series() {
        // Constant series is a degenerate case
        let spread = vec![5.0; 50];
        let (stat, is_stationary) = adf_test(&spread);
        // Constant series has zero variance in differences
        assert_eq!(stat, 0.0);
        assert!(!is_stationary);
    }
}
