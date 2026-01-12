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

### MC-3: Portfolio-Level Risk Limits
**Status**: Strategy-level limits exist (1 position per pair/symbol), but no portfolio-level controls.

**Existing Safeguards** (no action needed):
- `DualLegStrategy` uses `StrategyState::Flat` gate - prevents double entry
- `MovingAverageStrategy` uses `position_open: bool` flag - same effect

**Actual Gap** (future enhancement):
- No cap on total notional deployed across all pairs
- No pre-trade check against account buying power
- No limit on concurrent active positions

**Deferred**: Acceptable for paper trading. Required before significant live capital.

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

### MC-2 (Alpaca): Optimize String Allocations in WebSocket Hot Path
**Location**: `src/exchange/alpaca/websocket.rs`

Deferred from Alpaca Code Review. The `handle_message` function performs unnecessary string allocations during JSON parsing/deserialization which is a hot path.

**Planned Improvement**:
- Use `Cow<str>` or `&str` with `serde_json::from_slice` where possible.
- Avoid cloning strings when passing to `MarketData`.

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

### Jan 2026 - Alpaca Module Review
- [x] **MC-1**: Fix `JoinHandle` leak in `AlpacaWebSocketProvider`
- [x] **MC-3**: Add high-resolution tick latency metrics
- [x] **MC-4**: Optimize hot-path clone in market data routing
- [x] **CB-1**: Implement graceful shutdown for WebSocket task
- [x] **CB-2**: Add circuit breaker for reconnection logic
- [x] **CB-3**: Fix unsafe boolean initialization (UB)
- [x] **N-1**: Use `expect` with context instead of `unwrap_or_default` for `NonZeroU32`
- [x] **Refactor**: Unified `place_order` logic for live/paper trading
