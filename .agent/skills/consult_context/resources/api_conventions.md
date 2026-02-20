# API Conventions

Coding patterns and conventions used in algopioneer.

## Error Handling

### Pattern: Typed Errors
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OrderError {
    #[error("insufficient funds: need {needed}, have {available}")]
    InsufficientFunds { needed: Decimal, available: Decimal },
    
    #[error("invalid quantity: {0}")]
    InvalidQuantity(Decimal),
    
    #[error("exchange error: {0}")]
    Exchange(#[from] ExchangeError),
}
```

### Pattern: Result Propagation
```rust
pub async fn place_order(&self, order: Order) -> Result<OrderId, OrderError> {
    self.validate_order(&order)?;
    let id = self.executor.submit(order).await?;
    Ok(id)
}
```

## Async Patterns

### Pattern: Graceful Shutdown
```rust
tokio::select! {
    _ = shutdown_rx.recv() => {
        info!("Shutdown signal received");
        break;
    }
    tick = data_rx.recv() => {
        // Process tick
    }
}
```

### Pattern: Timeout
```rust
use tokio::time::timeout;

let result = timeout(Duration::from_secs(5), api_call())
    .await
    .map_err(|_| Error::Timeout)?;
```

## Trait Patterns

### Pattern: Dependency Injection
```rust
pub trait Executor: Send + Sync {
    async fn place_order(&self, order: Order) -> Result<OrderId, Error>;
    async fn cancel_order(&self, id: OrderId) -> Result<(), Error>;
}

// Use in strategy
pub struct MyStrategy<E: Executor> {
    executor: Arc<E>,
}
```

### Pattern: Mock for Testing
```rust
#[cfg(test)]
mod tests {
    use mockall::mock;
    
    mock! {
        pub Executor {}
        impl Executor for Executor {
            async fn place_order(&self, order: Order) -> Result<OrderId, Error>;
        }
    }
}
```

## Decimal Conventions

### Creating Decimals
```rust
// From string (compile-time checked)
use rust_decimal_macros::dec;
let price = dec!(100.50);

// From runtime string
let price = Decimal::from_str("100.50")?;

// From integer
let qty = Decimal::from(10);
```

### Rounding
```rust
use rust_decimal::RoundingStrategy;

// Currency (2 decimal places)
price.round_dp_with_strategy(2, RoundingStrategy::MidpointAwayFromZero);

// Crypto quantity (8 decimal places)
qty.round_dp_with_strategy(8, RoundingStrategy::ToZero);
```

## Logging Conventions

### Structured Fields
```rust
use tracing::{info, warn, error, instrument};

#[instrument(skip(self), fields(symbol = %order.symbol))]
async fn place_order(&self, order: Order) -> Result<OrderId, Error> {
    info!(side = ?order.side, qty = %order.quantity, "Placing order");
    
    let id = self.executor.submit(order).await.map_err(|e| {
        error!(error = %e, "Order submission failed");
        e
    })?;
    
    info!(order_id = %id, "Order placed successfully");
    Ok(id)
}
```
