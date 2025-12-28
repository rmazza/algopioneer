# Technical Roadmap

Technical debt and enhancement tracking for algopioneer.

## High Priority

### NP-5: Strategy Restart Policy Implementation
**Location**: `src/strategy/supervisor.rs:31-57`

`RestartPolicy` struct exists but restart logic is not implemented. Panicked strategies remain stopped and require manual intervention.

**Planned Features**:
- Exponential backoff with jitter
- Per-strategy restart budgets
- Cooldown periods after repeated failures
- Integration with health monitoring

---

## Medium Priority

### N-4: SmallVec for Trade Returns
**Location**: `src/discovery/optimizer.rs:275`

Consider using `SmallVec<[Decimal; 32]>` for trade returns in backtests with few trades to avoid heap allocation.

```rust
// Current
let mut trade_returns: Vec<Decimal> = Vec::with_capacity(...);

// Suggested
let mut trade_returns: SmallVec<[Decimal; 32]> = SmallVec::new();
```

---

## Low Priority / Cleanup

### Inconsistent Error Suffix Convention
Some modules use `Error` suffix (e.g., `ExecutionError`, `ExchangeError`) while others use different patterns (`DiscoveryError`, `DualLegError`). Minor but inconsistent.

---

## Completed ✅

- [x] **MC-1**: Remove dead code fields (`sandbox`, `clock`) in `AlpacaWebSocketProvider`
- [x] **MC-2**: Add numerical stability threshold (1e-12) to ADF test in `filter.rs`
- [x] **CB-1**: Precision-safe price parsing in Alpaca WebSocket
- [x] **MC-3**: O(1) symbol lookup with HashMap in WebSocket handler
- [x] **CB-2**: Alpaca Discovery Fixes (Decimal z-scores, daily bars support)
- [x] **CB-3**: Reliability improvements (Clock injection, graceful error handling)
- [x] **CB-5**: Safe access in Market Data Provider (removed unwrap)
- [x] **N-2**: Performance optimization in Alpaca Utils (pre-computed powers of 10)
