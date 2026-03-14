use crate::domain::logging::{PositionStateRecord, RecordError, TradeRecord};
use async_trait::async_trait;

/// Trait for recording trades to various backends
#[async_trait]
pub trait TradeRecorder: Send + Sync {
    async fn record(&self, trade: &TradeRecord) -> Result<(), RecordError>;

    async fn flush(&self) -> Result<(), RecordError> {
        Ok(())
    }
}

/// Trait for persisting and retrieving strategy position state.
#[async_trait]
pub trait StateStore: Send + Sync {
    async fn save_state(&self, state: &PositionStateRecord) -> Result<(), RecordError>;

    async fn load_state(
        &self,
        position_id: &str,
    ) -> Result<Option<PositionStateRecord>, RecordError>;

    async fn delete_state(&self, position_id: &str) -> Result<(), RecordError>;
}
