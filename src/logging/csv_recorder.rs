//! CSV Trade Recorder
//!
//! Writes trades to a CSV file. Suitable for development and testing.

use super::recorder::{RecordError, TradeRecord, TradeRecorder};
use async_trait::async_trait;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// CSV file recorder for development/testing
///
/// Uses `spawn_blocking` to avoid blocking the async runtime during file I/O.
/// CSV file recorder for development/testing
///
/// Uses a persistent `BufWriter` protected by a `Mutex` to ensure atomic writes
/// and minimize syscall overhead.
pub struct CsvRecorder {
    /// Mutex protecting the buffered writer.
    /// We use std::sync::Mutex because we run inside spawn_blocking.
    writer: Arc<Mutex<std::io::BufWriter<std::fs::File>>>,
}

impl CsvRecorder {
    /// Create a new CSV recorder
    pub fn new(file_path: PathBuf) -> Result<Self, std::io::Error> {
        let file_exists = file_path.exists();
        let file_empty = if file_exists {
            std::fs::metadata(&file_path)?.len() == 0
        } else {
            true
        };

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)?;

        let mut writer = std::io::BufWriter::new(file);

        // Write header if new file or empty
        if file_empty {
            writeln!(writer, "{}", TradeRecord::csv_header())?;
            writer.flush()?;
        }

        Ok(Self {
            writer: Arc::new(Mutex::new(writer)),
        })
    }
}

#[async_trait]
impl TradeRecorder for CsvRecorder {
    async fn record(&self, trade: &TradeRecord) -> Result<(), RecordError> {
        let writer = Arc::clone(&self.writer);
        let csv_line = trade.to_csv_line();

        // Use spawn_blocking to avoid blocking the async runtime with IO or mutex contention
        tokio::task::spawn_blocking(move || {
            let mut guard = writer
                .lock()
                .map_err(|_| RecordError::Io(std::io::Error::other("CSV writer mutex poisoned")))?;

            writeln!(guard, "{}", csv_line).map_err(RecordError::Io)?;

            // Flush periodically or rely on OS/buffer.
            // For trading logs, we prefer safety over raw throughput, so we flush.
            guard.flush().map_err(RecordError::Io)?;

            Ok::<(), RecordError>(())
        })
        .await
        .map_err(|e| RecordError::Io(std::io::Error::other(e)))??;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::recorder::TradeSide;
    use rust_decimal_macros::dec;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_csv_recorder_writes_header_and_records() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_trades.csv");

        let recorder = CsvRecorder::new(file_path.clone()).unwrap();

        let trade = TradeRecord::now(
            "BTC-USD".to_string(),
            TradeSide::Buy,
            dec!(0.5),
            Some(dec!(50000)),
            true,
        );

        recorder.record(&trade).await.unwrap();

        let contents = std::fs::read_to_string(&file_path).unwrap();
        assert!(contents.starts_with("trade_id,timestamp"));
        assert!(contents.contains("BTC-USD"));
    }
}
