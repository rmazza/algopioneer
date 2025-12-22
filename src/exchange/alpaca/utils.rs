//! Shared utilities for Alpaca module

use crate::exchange::{ExchangeError, Granularity};
use apca::data::v2::bars as alpaca_bars;
use num_decimal::Num;
use rust_decimal::Decimal;
use std::borrow::Cow;
use tracing::{error, warn};

/// Convert internal symbol format to Alpaca format (zero-alloc when possible)
///
/// # Performance (N-2 FIX)
/// Uses `Cow<str>` to avoid allocation when symbol has no dash.
/// For equities (most Alpaca usage), this is the common case.
///
/// # Examples
/// - `"AAPL"` -> `Cow::Borrowed("AAPL")` (zero alloc)
/// - `"BTC-USD"` -> `Cow::Owned("BTCUSD")` (one alloc)
#[inline]
pub fn to_alpaca_symbol(symbol: &str) -> Cow<'_, str> {
    // Fast path: no dash = no allocation needed
    if !symbol.contains('-') {
        return Cow::Borrowed(symbol);
    }

    // Slow path: preallocate exact capacity
    Cow::Owned(symbol.replace('-', ""))
}

/// Convert Decimal to num_decimal::Num with error handling.
/// N-4 FIX: #[inline] for hot path (called per-tick)
#[inline]
pub fn decimal_to_num(d: Decimal) -> Result<Num, ExchangeError> {
    d.to_string().parse::<Num>().map_err(|e| {
        error!(
            decimal = %d,
            error = %e,
            "Failed to convert Decimal to Num"
        );
        ExchangeError::Other(format!(
            "Decimal to Num conversion failed for '{}': {}",
            d, e
        ))
    })
}

/// Convert num_decimal::Num to Decimal with error handling.
/// N-4 FIX: #[inline] for hot path (called per-tick)
#[inline]
pub fn num_to_decimal(n: &Num) -> Result<Decimal, ExchangeError> {
    n.to_string().parse::<Decimal>().map_err(|e| {
        error!(
            num = %n,
            error = %e,
            "Failed to convert Num to Decimal"
        );
        ExchangeError::Other(format!(
            "Num to Decimal conversion failed for '{}': {}",
            n, e
        ))
    })
}

/// Convert Granularity to Alpaca TimeFrame with warnings.
pub fn granularity_to_timeframe(g: Granularity) -> alpaca_bars::TimeFrame {
    match g {
        Granularity::OneMinute => alpaca_bars::TimeFrame::OneMinute,
        Granularity::FiveMinute => {
            warn!(requested = ?g, actual = "1m", "Alpaca doesn't support 5m bars, using 1m");
            alpaca_bars::TimeFrame::OneMinute
        }
        Granularity::FifteenMinute => {
            warn!(requested = ?g, actual = "1m", "Alpaca doesn't support 15m bars, using 1m");
            alpaca_bars::TimeFrame::OneMinute
        }
        Granularity::ThirtyMinute => {
            warn!(requested = ?g, actual = "1h", "Alpaca doesn't support 30m bars, using 1h");
            alpaca_bars::TimeFrame::OneHour
        }
        Granularity::OneHour => alpaca_bars::TimeFrame::OneHour,
        Granularity::TwoHour => {
            warn!(requested = ?g, actual = "1h", "Alpaca doesn't support 2h bars, using 1h");
            alpaca_bars::TimeFrame::OneHour
        }
        Granularity::SixHour => {
            warn!(requested = ?g, actual = "1h", "Alpaca doesn't support 6h bars, using 1h");
            alpaca_bars::TimeFrame::OneHour
        }
        Granularity::OneDay => alpaca_bars::TimeFrame::OneDay,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only: Convert Alpaca symbol back to internal format
    fn from_alpaca_symbol(symbol: &str) -> String {
        // For crypto, insert dash before USD
        if symbol.ends_with("USD") && symbol.len() > 3 {
            let base = &symbol[..symbol.len() - 3];
            format!("{}-USD", base)
        } else {
            symbol.to_string()
        }
    }

    #[test]
    fn test_symbol_zero_alloc() {
        // Test that equities return borrowed (zero alloc)
        let cow = to_alpaca_symbol("AAPL");
        assert!(matches!(cow, Cow::Borrowed(_)));
        assert_eq!(cow.as_ref(), "AAPL");
    }

    #[test]
    fn test_symbol_with_dash() {
        // Test that crypto symbols allocate
        let cow = to_alpaca_symbol("BTC-USD");
        assert!(matches!(cow, Cow::Owned(_)));
        assert_eq!(cow.as_ref(), "BTCUSD");
    }

    #[test]
    fn test_symbol_roundtrip() {
        assert_eq!(from_alpaca_symbol("BTCUSD"), "BTC-USD");
        assert_eq!(from_alpaca_symbol("AAPL"), "AAPL");
    }

    #[test]
    fn test_decimal_to_num_conversion() {
        let d = Decimal::new(12345, 2); // 123.45
        let n = decimal_to_num(d).expect("conversion should succeed");
        assert_eq!(n.to_string(), "123.45");
    }

    #[test]
    fn test_num_to_decimal_conversion() {
        let n: Num = "123.45".parse().unwrap();
        let d = num_to_decimal(&n).expect("conversion should succeed");
        assert_eq!(d, Decimal::new(12345, 2));
    }

    #[test]
    fn test_granularity_to_timeframe() {
        assert!(matches!(
            granularity_to_timeframe(Granularity::OneMinute),
            alpaca_bars::TimeFrame::OneMinute
        ));
        assert!(matches!(
            granularity_to_timeframe(Granularity::OneDay),
            alpaca_bars::TimeFrame::OneDay
        ));
    }
}
