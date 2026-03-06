# Error Codes Reference

Error types and handling patterns in algopioneer.

## Strategy Errors

| Code | Type | Description | Recovery |
|------|------|-------------|----------|
| `STRAT_001` | `SignalError` | Invalid signal calculation | Log and skip tick |
| `STRAT_002` | `PositionError` | Position limit exceeded | Reject new entries |
| `STRAT_003` | `DataError` | Insufficient data for calculation | Wait for more ticks |
| `STRAT_004` | `StateError` | Invalid state transition | Reset to Idle |

## Exchange Errors

| Code | Type | Description | Recovery |
|------|------|-------------|----------|
| `EXCH_001` | `AuthenticationError` | Invalid API credentials | Halt, alert operator |
| `EXCH_002` | `RateLimitError` | Too many requests | Exponential backoff |
| `EXCH_003` | `InsufficientFunds` | Not enough balance | Cancel order, alert |
| `EXCH_004` | `InvalidOrder` | Order rejected by exchange | Log, do not retry |
| `EXCH_005` | `NetworkError` | Connection failed | Retry with backoff |
| `EXCH_006` | `TimeoutError` | Request timed out | Retry once, then alert |

## Order Errors

| Code | Type | Description | Recovery |
|------|------|-------------|----------|
| `ORD_001` | `InvalidQuantity` | Quantity below minimum | Reject order |
| `ORD_002` | `InvalidPrice` | Price outside valid range | Reject order |
| `ORD_003` | `OrderNotFound` | Order ID not found | Log, no action |
| `ORD_004` | `PartialFill` | Order partially filled | Track fill, log |
| `ORD_005` | `OrderCanceled` | Order was canceled | Log, update state |

## Discovery Errors

| Code | Type | Description | Recovery |
|------|------|-------------|----------|
| `DISC_001` | `DataFetchError` | Failed to fetch historical data | Skip symbol |
| `DISC_002` | `CointegrationError` | Cointegration test failed | Skip pair |
| `DISC_003` | `OptimizationError` | No valid parameters found | Skip pair |

## Circuit Breaker States

| State | Description | Transition |
|-------|-------------|------------|
| `Closed` | Normal operation | → Open on failure threshold |
| `Open` | Rejecting all requests | → HalfOpen after timeout |
| `HalfOpen` | Testing recovery | → Closed on success, Open on failure |

## Error Handling Patterns

### Retryable vs Non-Retryable

```rust
impl Error {
    pub fn is_retryable(&self) -> bool {
        matches!(self, 
            Error::Network(_) | 
            Error::Timeout | 
            Error::RateLimit
        )
    }
}
```

### Exponential Backoff
```rust
let delays = [100, 200, 400, 800, 1600]; // ms
for (attempt, delay) in delays.iter().enumerate() {
    match operation().await {
        Ok(result) => return Ok(result),
        Err(e) if e.is_retryable() => {
            tokio::time::sleep(Duration::from_millis(*delay)).await;
        }
        Err(e) => return Err(e),
    }
}
```
