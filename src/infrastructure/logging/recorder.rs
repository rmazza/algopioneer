//! Trade Recording Implementation
//!
//! Provides concrete implementations and helpers for trade recording.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
pub use crate::domain::logging::{TradeRecord, TradeSide, RecordError, PositionStateRecord};
pub use crate::application::ports::logging::{TradeRecorder, StateStore};

impl TradeRecord {
    pub fn with_timestamp(
        symbol: String,
        side: TradeSide,
        size: Decimal,
        price: Option<Decimal>,
        is_paper: bool,
        timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            trade_id: uuid::Uuid::new_v4().to_string(),
            timestamp,
            symbol,
            side,
            size,
            price,
            strategy: None,
            is_paper,
        }
    }

    pub fn to_csv_line(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{}",
            self.trade_id,
            self.timestamp.to_rfc3339(),
            self.symbol,
            self.side,
            self.size,
            self.price.map(|p| p.to_string()).unwrap_or_default(),
            self.strategy.as_deref().unwrap_or(""),
            self.is_paper,
        )
    }

    pub fn write_csv_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        write!(
            writer,
            "{},{},{},{},{},{},{},{}",
            self.trade_id,
            self.timestamp.to_rfc3339(),
            self.symbol,
            self.side,
            self.size,
            self.price.map(|p| p.to_string()).unwrap_or_default(),
            self.strategy.as_deref().unwrap_or(""),
            self.is_paper,
        )
    }

    pub fn csv_header() -> &'static str {
        "trade_id,timestamp,symbol,side,size,price,strategy,is_paper"
    }
}

impl PositionStateRecord {
    pub fn flat(
        position_id: impl Into<String>,
        strategy_type: impl Into<String>,
        leg1_symbol: impl Into<String>,
        leg2_symbol: impl Into<String>,
        is_paper: bool,
    ) -> Self {
        Self {
            position_id: position_id.into(),
            strategy_type: strategy_type.into(),
            state: "flat".to_string(),
            direction: None,
            leg1_symbol: leg1_symbol.into(),
            leg2_symbol: leg2_symbol.into(),
            leg1_qty: "0".to_string(),
            leg2_qty: "0".to_string(),
            leg1_entry_price: "0".to_string(),
            leg2_entry_price: "0".to_string(),
            updated_at: Utc::now(),
            is_paper,
        }
    }
}

/// A recorder that fans out to multiple backends
pub struct MultiRecorder {
    recorders: Vec<Box<dyn TradeRecorder>>,
}

impl MultiRecorder {
    pub fn new(recorders: Vec<Box<dyn TradeRecorder>>) -> Self {
        Self { recorders }
    }

    pub fn add(&mut self, recorder: Box<dyn TradeRecorder>) {
        self.recorders.push(recorder);
    }
}

#[async_trait]
impl TradeRecorder for MultiRecorder {
    async fn record(&self, trade: &TradeRecord) -> Result<(), RecordError> {
        let mut error_count = 0;
        let mut last_error = None;

        for recorder in &self.recorders {
            if let Err(e) = recorder.record(trade).await {
                tracing::error!(error = %e, "Failed to record trade to backend");
                last_error = Some(e);
                error_count += 1;
            }
        }

        if error_count > 0 && error_count == self.recorders.len() {
            if let Some(e) = last_error {
                return Err(e);
            }
        }

        Ok(())
    }

    async fn flush(&self) -> Result<(), RecordError> {
        for recorder in &self.recorders {
            recorder.flush().await?;
        }
        Ok(())
    }
}
