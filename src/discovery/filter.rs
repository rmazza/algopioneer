//! Statistical filtering for pair candidates
//!
//! Implements correlation analysis and mean-reversion testing
//! to filter viable trading pairs.

use std::collections::HashMap;
use tracing::{debug, info, warn};

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
}

/// Calculate Pearson correlation coefficient between two price series
///
/// Returns a value in [-1.0, 1.0], or None if calculation fails.
///
/// # Mathematical Definition
/// r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]
pub fn calculate_correlation(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.len() < 2 {
        return None;
    }

    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;

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
        1000.0 // Non-stationary or invalid
    };

    (std_dev, half_life)
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

/// Filter candidate pairs based on correlation and mean-reversion criteria
///
/// # Algorithm
/// 1. For each unique pair (i, j) where i < j:
/// 2. Calculate Pearson correlation
/// 3. If correlation >= threshold, compute log-spread
/// 4. Estimate half-life via O-U model
/// 5. If half-life <= threshold, add to results
pub fn filter_candidates(
    prices: &HashMap<String, Vec<f64>>,
    min_correlation: f64,
    max_half_life_hours: f64,
) -> Vec<CandidatePair> {
    let symbols: Vec<&String> = prices.keys().collect();
    let mut results = Vec::new();

    info!(
        candidates = symbols.len(),
        min_corr = min_correlation,
        max_hl = max_half_life_hours,
        "Filtering pair candidates"
    );

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

            info!(
                pair = format!("{}-{}", sym_a, sym_b),
                correlation = format!("{:.3}", correlation),
                half_life = format!("{:.1}h", half_life),
                "Viable pair found"
            );

            results.push(CandidatePair {
                symbol_a: sym_a.clone(),
                symbol_b: sym_b.clone(),
                correlation,
                spread_std,
                half_life_hours: half_life,
            });
        }
    }

    // Sort by correlation descending
    results.sort_by(|a, b| {
        b.correlation
            .partial_cmp(&a.correlation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    info!(viable_pairs = results.len(), "Filtering complete");
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
}
