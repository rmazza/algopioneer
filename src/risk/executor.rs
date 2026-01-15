use crate::exchange::{ExchangeError, Executor};
use crate::orders::{OrderId, OrderState};
use crate::risk::DailyRiskEngine;
use crate::types::OrderSide;
use async_trait::async_trait;
use rust_decimal::Decimal;
use std::sync::Arc;
use tracing::warn;

/// Executor wrapper that enforces daily risk limits.
///
/// Blocks new order execution if the daily risk limit has been breached.
/// Position queries and order status checks are passed through to allow for
/// closing positions or monitoring.
pub struct RiskManagedExecutor<E> {
    inner: Arc<E>,
    risk_engine: Arc<DailyRiskEngine>,
}

impl<E: Executor> RiskManagedExecutor<E> {
    /// Create a new risk-managed executor.
    pub fn new(inner: Arc<E>, risk_engine: Arc<DailyRiskEngine>) -> Self {
        Self { inner, risk_engine }
    }
}

#[async_trait]
impl<E: Executor + Send + Sync> Executor for RiskManagedExecutor<E> {
    async fn execute_order(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: Decimal,
        price: Option<Decimal>,
    ) -> Result<OrderId, ExchangeError> {
        // Enforce risk limits
        if !self.risk_engine.is_trading_enabled() {
            warn!(
                symbol = symbol,
                side = %side,
                quantity = %quantity,
                "Order blocked by Daily Risk Limit"
            );
            return Err(ExchangeError::Other(
                "Order blocked: Daily Risk Limit breached".to_string(),
            ));
        }

        // Delegate to inner executor
        self.inner
            .execute_order(symbol, side, quantity, price)
            .await
    }

    async fn get_position(&self, symbol: &str) -> Result<Decimal, ExchangeError> {
        self.inner.get_position(symbol).await
    }

    async fn get_order_status(
        &self,
        order_id: &OrderId,
    ) -> Result<(OrderState, Decimal, Option<Decimal>), ExchangeError> {
        self.inner.get_order_status(order_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::risk::{DailyRiskConfig, RiskStatus};
    use rust_decimal_macros::dec;

    struct MockExecutor;

    #[async_trait]
    impl Executor for MockExecutor {
        async fn execute_order(
            &self,
            _symbol: &str,
            _side: OrderSide,
            _quantity: Decimal,
            _price: Option<Decimal>,
        ) -> Result<OrderId, ExchangeError> {
            Ok(OrderId::new("mock-order"))
        }

        async fn get_position(&self, _symbol: &str) -> Result<Decimal, ExchangeError> {
            Ok(Decimal::ZERO)
        }
    }

    #[tokio::test]
    async fn test_risk_enforcement() {
        let risk_engine = Arc::new(DailyRiskEngine::new(DailyRiskConfig {
            max_daily_loss: dec!(-100),
            warning_threshold: dec!(-50),
        }));
        let inner = Arc::new(MockExecutor);
        let executor = RiskManagedExecutor::new(inner, risk_engine.clone());

        // 1. Initial order should succeed
        let result = executor
            .execute_order("BTC-USD", OrderSide::Buy, dec!(1), None)
            .await;
        assert!(result.is_ok());

        // 2. Breach limit
        risk_engine.record_pnl(dec!(-150));
        assert_eq!(risk_engine.status(), RiskStatus::Halted);

        // 3. Subsequent order should be blocked
        let result = executor
            .execute_order("BTC-USD", OrderSide::Buy, dec!(1), None)
            .await;

        match result {
            Err(ExchangeError::Other(msg)) => {
                assert!(msg.contains("Order blocked"));
                assert!(msg.contains("Daily Risk Limit"));
            }
            _ => panic!("Expected ExchangeError::Other, got {:?}", result),
        }
    }
}
