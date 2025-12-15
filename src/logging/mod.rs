//! Logging and Trade Recording Module
//!
//! Provides multiple backends for recording trades:
//! - `PaperTradeLogger` - Legacy channel-based CSV logger
//! - `TradeRecorder` trait - Pluggable recorder interface
//! - `CsvRecorder` - Synchronous CSV file recorder
//! - `TracingRecorder` - CloudWatch-compatible structured logs
//! - `DynamoDbRecorder` - AWS DynamoDB persistence (requires `dynamodb` feature)

// New recorder infrastructure
pub mod csv_recorder;
pub mod recorder;
pub mod tracing_recorder;

#[cfg(feature = "dynamodb")]
pub mod dynamodb_recorder;

// Re-exports for convenience
pub use csv_recorder::CsvRecorder;
pub use recorder::{MultiRecorder, RecordError, TradeRecord, TradeRecorder, TradeSide};
pub use tracing_recorder::TracingRecorder;

#[cfg(feature = "dynamodb")]
pub use dynamodb_recorder::DynamoDbRecorder;

// Legacy logger (existing code below)
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

/// A single paper trade record
#[derive(Debug, Clone)]
pub struct PaperTradeRecord {
    pub timestamp: DateTime<Utc>,
    pub product_id: String,
    pub side: String,
    pub size: Decimal,
    pub price: String,
}

impl PaperTradeRecord {
    /// Format as CSV line
    pub fn to_csv_line(&self) -> String {
        format!(
            "{},{},{},{},{}",
            self.timestamp, self.product_id, self.side, self.size, self.price
        )
    }
}

/// Handle for sending paper trade records to the logger
#[derive(Clone)]
pub struct PaperTradeLogger {
    sender: mpsc::UnboundedSender<PaperTradeRecord>,
}

impl PaperTradeLogger {
    /// Create a new paper trade logger that writes to the specified file.
    ///
    /// Returns the logger handle and a future that must be spawned to run the writer task.
    pub fn new(file_path: PathBuf) -> (Self, PaperTradeLoggerTask) {
        let (sender, receiver) = mpsc::unbounded_channel();
        let logger = Self { sender };
        let task = PaperTradeLoggerTask {
            receiver,
            file_path,
        };
        (logger, task)
    }

    /// Log a paper trade record.
    ///
    /// This is non-blocking and will never fail (logs are dropped if the writer task is dead).
    pub fn log(&self, record: PaperTradeRecord) {
        if let Err(e) = self.sender.send(record) {
            warn!("Paper trade logger channel closed, dropping record: {}", e);
        }
    }

    /// Convenience method to log a trade
    pub fn log_trade(&self, product_id: &str, side: &str, size: Decimal, price: Option<Decimal>) {
        let record = PaperTradeRecord {
            timestamp: Utc::now(),
            product_id: product_id.to_string(),
            side: side.to_string(),
            size,
            price: price
                .map(|p| p.to_string())
                .unwrap_or_else(|| "MARKET".to_string()),
        };
        self.log(record);
    }
}

/// Background task that handles all file writes serially
pub struct PaperTradeLoggerTask {
    receiver: mpsc::UnboundedReceiver<PaperTradeRecord>,
    file_path: PathBuf,
}

impl PaperTradeLoggerTask {
    /// Run the logger task. This should be spawned with `tokio::spawn`.
    pub async fn run(mut self) {
        info!(path = ?self.file_path, "Paper trade logger started");

        while let Some(record) = self.receiver.recv().await {
            if let Err(e) = self.write_record(&record) {
                error!(
                    error = %e,
                    record = ?record,
                    "Failed to write paper trade record"
                );
            }
        }

        info!("Paper trade logger shutting down");
    }

    fn write_record(&self, record: &PaperTradeRecord) -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)?;

        writeln!(file, "{}", record.to_csv_line())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    use std::fs;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_paper_trade_logger_writes_records() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_trades.csv");

        let (logger, task) = PaperTradeLogger::new(file_path.clone());

        // Spawn the writer task
        let handle = tokio::spawn(task.run());

        // Log some trades
        logger.log_trade("BTC-USD", "buy", dec!(0.5), Some(dec!(50000.0)));
        logger.log_trade("ETH-USD", "sell", dec!(1.0), None);

        // Give time for writes
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        // Drop logger to close channel and stop task
        drop(logger);
        handle.await.unwrap();

        // Verify file contents
        let contents = fs::read_to_string(&file_path).unwrap();
        assert!(contents.contains("BTC-USD,buy,0.5,50000.0"));
        assert!(contents.contains("ETH-USD,sell,1.0,MARKET"));
    }

    #[tokio::test]
    async fn test_concurrent_logging() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("concurrent_trades.csv");

        let (logger, task) = PaperTradeLogger::new(file_path.clone());
        let handle = tokio::spawn(task.run());

        // Simulate concurrent logging from multiple tasks
        let mut handles = vec![];
        for i in 0..10 {
            let logger_clone = logger.clone();
            handles.push(tokio::spawn(async move {
                for j in 0..10 {
                    logger_clone.log_trade(
                        &format!("PAIR-{}", i),
                        "buy",
                        Decimal::new(j, 0),
                        Some(Decimal::new(1000 + j, 0)),
                    );
                }
            }));
        }

        // Wait for all logging tasks
        for h in handles {
            h.await.unwrap();
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        drop(logger);
        handle.await.unwrap();

        // Verify all 100 records were written correctly
        let contents = fs::read_to_string(&file_path).unwrap();
        let line_count = contents.lines().count();
        assert_eq!(line_count, 100, "Expected 100 records, got {}", line_count);

        // Verify no corruption (each line should have exactly 5 comma-separated fields)
        for (i, line) in contents.lines().enumerate() {
            let fields: Vec<&str> = line.split(',').collect();
            assert_eq!(
                fields.len(),
                5,
                "Line {} has {} fields instead of 5: {}",
                i,
                fields.len(),
                line
            );
        }
    }
}
