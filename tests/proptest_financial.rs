//! Property-based tests for financial calculations
//!
//! These tests use proptest to verify invariants across many random inputs,
//! catching edge cases that unit tests might miss.

use proptest::prelude::*;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

/// Helper function to compute z-score (mirrors implementation in PairsManager)
fn compute_z_score(history: &[f64], current_spread: f64) -> Option<f64> {
    if history.len() < 2 {
        return None;
    }

    let n = history.len() as f64;
    let sum: f64 = history.iter().sum();
    let mean = sum / n;

    let variance: f64 = history
        .iter()
        .map(|val| {
            let diff = mean - val;
            diff * diff
        })
        .sum::<f64>()
        / n;

    let std_dev = variance.sqrt();

    if std_dev == 0.0 || !std_dev.is_finite() {
        return Some(0.0);
    }

    let z = (current_spread - mean) / std_dev;
    if z.is_finite() {
        Some(z)
    } else {
        None
    }
}

/// Helper function to compute hedge ratio (dollar neutral)
fn compute_hedge_ratio_dollar_neutral(
    _quantity: Decimal,
    leg1_price: Decimal,
    leg2_price: Decimal,
) -> Option<Decimal> {
    if leg2_price == Decimal::ZERO {
        return None;
    }
    Some(leg1_price / leg2_price)
}

proptest! {
    /// Z-score should always be finite for valid input
    #[test]
    fn zscore_is_finite_for_valid_input(
        history in prop::collection::vec(-1000.0f64..1000.0f64, 10..100),
        current in -1000.0f64..1000.0f64
    ) {
        if let Some(z) = compute_z_score(&history, current) {
            prop_assert!(z.is_finite(), "Z-score should be finite: {}", z);
        }
    }

    /// Z-score of the mean should be approximately 0
    #[test]
    fn zscore_of_mean_is_near_zero(
        history in prop::collection::vec(1.0f64..100.0f64, 10..50)
    ) {
        let mean: f64 = history.iter().sum::<f64>() / history.len() as f64;

        if let Some(z) = compute_z_score(&history, mean) {
            prop_assert!(z.abs() < 0.01, "Z-score of mean should be ~0, got: {}", z);
        }
    }

    /// Z-score should be symmetric: z(x) = -z(2*mean - x)
    #[test]
    fn zscore_is_symmetric(
        history in prop::collection::vec(50.0f64..150.0f64, 10..30),
        offset in 1.0f64..50.0f64
    ) {
        let mean: f64 = history.iter().sum::<f64>() / history.len() as f64;

        let z_above = compute_z_score(&history, mean + offset);
        let z_below = compute_z_score(&history, mean - offset);

        if let (Some(za), Some(zb)) = (z_above, z_below) {
            prop_assert!(
                (za + zb).abs() < 0.01,
                "Z-scores should be symmetric: {} vs {}", za, zb
            );
        }
    }

    /// Hedge ratio is always positive for positive prices
    #[test]
    fn hedge_ratio_is_positive(
        qty in 1i64..1000i64,
        p1 in 1i64..100000i64,
        p2 in 1i64..100000i64
    ) {
        let quantity = Decimal::new(qty, 3);
        let leg1_price = Decimal::new(p1, 2);
        let leg2_price = Decimal::new(p2, 2);

        if let Some(ratio) = compute_hedge_ratio_dollar_neutral(quantity, leg1_price, leg2_price) {
            prop_assert!(ratio > Decimal::ZERO, "Hedge ratio should be positive: {}", ratio);
        }
    }

    /// Dollar-neutral hedge: leg1_price * qty = leg2_price * qty * ratio
    #[test]
    fn hedge_ratio_preserves_dollar_value(
        p1 in 100i64..100000i64,
        p2 in 100i64..100000i64
    ) {
        let leg1_price = Decimal::new(p1, 2);
        let leg2_price = Decimal::new(p2, 2);

        if let Some(ratio) = compute_hedge_ratio_dollar_neutral(dec!(1.0), leg1_price, leg2_price) {
            // For dollar-neutral: leg1_price * 1 should equal leg2_price * ratio
            let leg1_value = leg1_price;
            let leg2_value = leg2_price * ratio;

            // They should be equal (within precision)
            let diff = (leg1_value - leg2_value).abs();
            prop_assert!(
                diff < dec!(0.0001),
                "Values should match: {} vs {} (diff: {})",
                leg1_value, leg2_value, diff
            );
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_zscore_constant_history() {
        // All same values should give z-score of 0
        let history = vec![50.0, 50.0, 50.0, 50.0, 50.0];
        let z = compute_z_score(&history, 50.0);
        assert_eq!(z, Some(0.0));
    }

    #[test]
    fn test_zscore_extreme_value() {
        let history = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let z = compute_z_score(&history, 100.0);
        assert!(z.is_some());
        assert!(z.unwrap() > 10.0); // Should be a large positive z-score
    }

    #[test]
    fn test_hedge_ratio_equal_prices() {
        let ratio = compute_hedge_ratio_dollar_neutral(dec!(1.0), dec!(100.0), dec!(100.0));
        assert_eq!(ratio, Some(dec!(1.0)));
    }
}
