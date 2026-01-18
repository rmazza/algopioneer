//! DynamoDB Trade Recorder
//!
//! Writes trades to AWS DynamoDB for persistent, queryable storage.
//! Requires the `dynamodb` feature flag.

use super::recorder::{RecordError, TradeRecord, TradeRecorder};
use async_trait::async_trait;
use aws_sdk_dynamodb::types::AttributeValue;
use aws_sdk_dynamodb::Client;
use tracing::{debug, error};

/// DynamoDB trade recorder
///
/// Writes trades to a DynamoDB table with the following schema:
/// - pk (Partition Key): symbol (e.g., "BTC-USD")
/// - sk (Sort Key): timestamp in ISO format
/// - trade_id, side, size, price, strategy, is_paper
pub struct DynamoDbRecorder {
    client: Client,
    table_name: String,
}

impl DynamoDbRecorder {
    /// Create a new DynamoDB recorder
    ///
    /// # Arguments
    /// * `client` - AWS SDK DynamoDB client
    /// * `table_name` - Name of the DynamoDB table
    pub fn new(client: Client, table_name: impl Into<String>) -> Self {
        Self {
            client,
            table_name: table_name.into(),
        }
    }

    /// Create a recorder with default configuration from environment
    ///
    /// Loads AWS credentials from environment variables or IAM role.
    pub async fn from_env(table_name: impl Into<String>) -> Self {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        let client = Client::new(&config);
        Self::new(client, table_name)
    }
}

#[async_trait]
impl TradeRecorder for DynamoDbRecorder {
    async fn record(&self, trade: &TradeRecord) -> Result<(), RecordError> {
        let mut item = std::collections::HashMap::new();

        // Keys
        item.insert("pk".to_string(), AttributeValue::S(trade.symbol.clone()));
        item.insert(
            "sk".to_string(),
            AttributeValue::S(trade.timestamp.to_rfc3339()),
        );

        // Attributes
        item.insert(
            "trade_id".to_string(),
            AttributeValue::S(trade.trade_id.clone()),
        );
        item.insert(
            "side".to_string(),
            AttributeValue::S(trade.side.to_string()),
        );
        item.insert(
            "size".to_string(),
            AttributeValue::N(trade.size.to_string()),
        );

        if let Some(price) = trade.price {
            item.insert("price".to_string(), AttributeValue::N(price.to_string()));
        }

        if let Some(ref strategy) = trade.strategy {
            item.insert("strategy".to_string(), AttributeValue::S(strategy.clone()));
        }

        item.insert("is_paper".to_string(), AttributeValue::Bool(trade.is_paper));

        debug!(
            table = %self.table_name,
            trade_id = %trade.trade_id,
            "Writing trade to DynamoDB"
        );

        self.client
            .put_item()
            .table_name(&self.table_name)
            .set_item(Some(item))
            .send()
            .await
            .map_err(|e| {
                error!(error = %e, "DynamoDB put_item failed");
                RecordError::DynamoDb(e.to_string())
            })?;

        Ok(())
    }
}

// ============================================================================
// StateStore Implementation for Position State Persistence
// ============================================================================

use super::recorder::{PositionStateRecord, StateStore};

#[async_trait]
impl StateStore for DynamoDbRecorder {
    async fn save_state(&self, state: &PositionStateRecord) -> Result<(), RecordError> {
        let mut item = std::collections::HashMap::new();

        // Keys: Use "state:" prefix to separate from trade records
        // pk = "state:{position_id}", sk = "LATEST" (single-item, always overwritten)
        item.insert(
            "pk".to_string(),
            AttributeValue::S(format!("state:{}", state.position_id)),
        );
        item.insert("sk".to_string(), AttributeValue::S("LATEST".to_string()));

        // State attributes
        item.insert(
            "position_id".to_string(),
            AttributeValue::S(state.position_id.clone()),
        );
        item.insert(
            "strategy_type".to_string(),
            AttributeValue::S(state.strategy_type.clone()),
        );
        item.insert("state".to_string(), AttributeValue::S(state.state.clone()));

        if let Some(ref direction) = state.direction {
            item.insert(
                "direction".to_string(),
                AttributeValue::S(direction.clone()),
            );
        }

        item.insert(
            "leg1_symbol".to_string(),
            AttributeValue::S(state.leg1_symbol.clone()),
        );
        item.insert(
            "leg2_symbol".to_string(),
            AttributeValue::S(state.leg2_symbol.clone()),
        );
        item.insert(
            "leg1_qty".to_string(),
            AttributeValue::S(state.leg1_qty.clone()),
        );
        item.insert(
            "leg2_qty".to_string(),
            AttributeValue::S(state.leg2_qty.clone()),
        );
        item.insert(
            "leg1_entry_price".to_string(),
            AttributeValue::S(state.leg1_entry_price.clone()),
        );
        item.insert(
            "leg2_entry_price".to_string(),
            AttributeValue::S(state.leg2_entry_price.clone()),
        );
        item.insert(
            "updated_at".to_string(),
            AttributeValue::S(state.updated_at.to_rfc3339()),
        );
        item.insert("is_paper".to_string(), AttributeValue::Bool(state.is_paper));

        debug!(
            table = %self.table_name,
            position_id = %state.position_id,
            state = %state.state,
            "Saving position state to DynamoDB"
        );

        self.client
            .put_item()
            .table_name(&self.table_name)
            .set_item(Some(item))
            .send()
            .await
            .map_err(|e| {
                error!(error = %e, "DynamoDB save_state failed");
                RecordError::DynamoDb(e.to_string())
            })?;

        Ok(())
    }

    async fn load_state(
        &self,
        position_id: &str,
    ) -> Result<Option<PositionStateRecord>, RecordError> {
        debug!(
            table = %self.table_name,
            position_id = %position_id,
            "Loading position state from DynamoDB"
        );

        let result = self
            .client
            .get_item()
            .table_name(&self.table_name)
            .key("pk", AttributeValue::S(format!("state:{}", position_id)))
            .key("sk", AttributeValue::S("LATEST".to_string()))
            .send()
            .await
            .map_err(|e| {
                error!(error = %e, "DynamoDB load_state failed");
                RecordError::DynamoDb(e.to_string())
            })?;

        let Some(item) = result.item else {
            debug!(position_id = %position_id, "No saved state found");
            return Ok(None);
        };

        // Parse the item into PositionStateRecord
        let get_string = |key: &str| -> Result<String, RecordError> {
            item.get(key)
                .and_then(|v| v.as_s().ok())
                .map(|s| s.to_string())
                .ok_or_else(|| RecordError::DynamoDb(format!("Missing field: {}", key)))
        };

        let get_bool = |key: &str| -> Result<bool, RecordError> {
            item.get(key)
                .and_then(|v| v.as_bool().ok())
                .copied()
                .ok_or_else(|| RecordError::DynamoDb(format!("Missing field: {}", key)))
        };

        let direction = item
            .get("direction")
            .and_then(|v| v.as_s().ok())
            .map(|s| s.to_string());

        let updated_at_str = get_string("updated_at")?;
        let updated_at = chrono::DateTime::parse_from_rfc3339(&updated_at_str)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .map_err(|e| RecordError::DynamoDb(format!("Invalid timestamp: {}", e)))?;

        let record = PositionStateRecord {
            position_id: get_string("position_id")?,
            strategy_type: get_string("strategy_type")?,
            state: get_string("state")?,
            direction,
            leg1_symbol: get_string("leg1_symbol")?,
            leg2_symbol: get_string("leg2_symbol")?,
            leg1_qty: get_string("leg1_qty")?,
            leg2_qty: get_string("leg2_qty")?,
            leg1_entry_price: get_string("leg1_entry_price")?,
            leg2_entry_price: get_string("leg2_entry_price")?,
            updated_at,
            is_paper: get_bool("is_paper")?,
        };

        debug!(
            position_id = %record.position_id,
            state = %record.state,
            "Loaded position state from DynamoDB"
        );

        Ok(Some(record))
    }

    async fn delete_state(&self, position_id: &str) -> Result<(), RecordError> {
        debug!(
            table = %self.table_name,
            position_id = %position_id,
            "Deleting position state from DynamoDB"
        );

        self.client
            .delete_item()
            .table_name(&self.table_name)
            .key("pk", AttributeValue::S(format!("state:{}", position_id)))
            .key("sk", AttributeValue::S("LATEST".to_string()))
            .send()
            .await
            .map_err(|e| {
                error!(error = %e, "DynamoDB delete_state failed");
                RecordError::DynamoDb(e.to_string())
            })?;

        Ok(())
    }
}
