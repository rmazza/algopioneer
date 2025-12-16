//! Logging and Trade Recording Module
//!
//! Provides multiple backends for recording trades:
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
