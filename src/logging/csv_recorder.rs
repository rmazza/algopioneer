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
pub struct CsvRecorder {
    file_path: Arc<PathBuf>,
    /// Mutex to serialize writes and track header state
    state: Arc<Mutex<CsvState>>,
}

struct CsvState {
    header_written: bool,
}

impl CsvRecorder {
    /// Create a new CSV recorder
    pub fn new(file_path: PathBuf) -> Self {
        Self {
            file_path: Arc::new(file_path),
            state: Arc::new(Mutex::new(CsvState {
                header_written: false,
            })),
        }
    }
}

#[async_trait]
impl TradeRecorder for CsvRecorder {
    async fn record(&self, trade: &TradeRecord) -> Result<(), RecordError> {
        let file_path = Arc::clone(&self.file_path);
        let state = Arc::clone(&self.state);
        let csv_line = trade.to_csv_line();

        // Use spawn_blocking to avoid blocking the async runtime
        tokio::task::spawn_blocking(move || {
            // Handle mutex poisoning gracefully
            let mut guard = state.lock().unwrap_or_else(|e| e.into_inner());

            // Check if header needs to be written
            if !guard.header_written {
                let needs_header = !file_path.exists()
                    || std::fs::metadata(&*file_path)
                        .map(|m| m.len() == 0)
                        .unwrap_or(true);

                if needs_header {
                    let mut file = OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&*file_path)?;
                    writeln!(file, "{}", TradeRecord::csv_header())?;
                }
                guard.header_written = true;
            }

            // Write the trade record
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&*file_path)?;
            writeln!(file, "{}", csv_line)?;

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

        let recorder = CsvRecorder::new(file_path.clone());

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
