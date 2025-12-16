//! Tracing-based Trade Recorder
//!
//! Emits structured logs for trades that can be captured by CloudWatch Logs
//! or any tracing subscriber. Zero additional dependencies.

use super::recorder::{RecordError, TradeRecord, TradeRecorder};
use async_trait::async_trait;
use tracing::info;

/// Recorder that emits structured tracing logs
///
/// In AWS, these logs are automatically captured by CloudWatch when using
/// the tracing-subscriber with JSON formatting.
pub struct TracingRecorder;

impl TracingRecorder {
    /// Create a new tracing recorder
    pub fn new() -> Self {
        Self
    }
}

impl Default for TracingRecorder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TradeRecorder for TracingRecorder {
    async fn record(&self, trade: &TradeRecord) -> Result<(), RecordError> {
        // Emit structured log that CloudWatch can parse
        info!(
            target: "trades",
            trade_type = "EXECUTED",
            trade_id = %trade.trade_id,
            timestamp = %trade.timestamp.to_rfc3339(),
            symbol = %trade.symbol,
            side = %trade.side,
            size = %trade.size,
            price = trade.price.map(|p| p.to_string()).unwrap_or_else(|| "MARKET".to_string()),
            strategy = trade.strategy.as_deref().unwrap_or("default"),
            is_paper = trade.is_paper,
            "Trade executed"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::recorder::TradeSide;
    use chrono::Utc;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_tracing_recorder_does_not_error() {
        let recorder = TracingRecorder::new();

        // Use with_timestamp for deterministic testing (avoids deprecated now())
        let trade = TradeRecord::with_timestamp(
            "ETH-USD".to_string(),
            TradeSide::Sell,
            dec!(1.0),
            None,
            false,
            Utc::now(), // Explicit timestamp injection
        );

        // Should not error
        recorder.record(&trade).await.unwrap();
    }
}
