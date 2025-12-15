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
