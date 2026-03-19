# Technical Roadmap

Technical debt and enhancement tracking for algopioneer.

## High Priority

### ARC-1: Enforce Architectural Boundaries
**Status**: The audit in v1.9.2 identified that the `application` layer (e.g., `simple_engine.rs`, `dual_leg/mod.rs`) is directly importing concrete types from `infrastructure` (e.g., `CoinbaseClient`, `LogThrottle`) instead of using traits from `src/application/ports/`.

**Impact**: This violates Clean Architecture principles, making the application logic tightly coupled to specific infrastructure implementations. It hinders testability and exchange-agnostic execution.

**Required Action**:
- Refactor all `application` modules to use the `ExchangeClient`, `TradeRecorder`, and `MarketDataProvider` traits defined in `src/application/ports/`.
- Use Dependency Injection in `main.rs` to wire implementation details into the application services.

---

### MC-4: State Reconciliation for Overlapping Pairs
**Status**: **MITIGATED (v1.9.3 Configuration Hotfix)**. Current strategies use isolated symbol pairs (F-JNJ, WFC-JPM) to prevent aggregate position misattribution.

**Impact**: Successfully prevents "Ghost Positions" for the current run.

**Long-Term Action**:
- Implement `Client Order IDs` tracking for structural resolution.
- Add sub-account support to isolate positions at the broker level.

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

### Alpaca: Market Order Fallback for Small Quantities
**Status**: Deferred (workaround: use sufficient `order_size`)

**Problem**: Alpaca limit orders require whole shares. When `order_size / price < 1`, limit orders fail.

**Current Workaround**: Set `order_size` >= max stock price in pair (e.g., $500+ for AXP at $360).

**Future Enhancement**: Modify `alpaca_client.rs` to automatically fall back to market orders when floored quantity < 1 share. This would allow smaller notional sizes without failures.

**Location**: [alpaca_client.rs](file:///home/bob/dev/algopioneer/src/exchange/alpaca/alpaca_client.rs#L231-L245)

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
**Location**: `src/infrastructure/exchange/alpaca/websocket.rs`

**Status**: **PARTIALLY ADDRESSED** (Zero-allocation parsing implemented in `parse_trade_from_value`, but `s.clone()` remains for symbol mapping).

**Remaining Improvement**:
- Use `Cow<'static, str>` or reference-based symbol mapping to avoid clones.

---

## Low Priority / Cleanup

### Inconsistent Error Suffix Convention
Some modules use `Error` suffix (e.g., `ExecutionError`, `ExchangeError`) while others use different patterns (`DiscoveryError`, `DualLegError`). Minor but inconsistent.

---

## Completed ✅

### March 2026 - Clean Architecture & Backtesting (v1.9.0/1.9.1/1.9.2)
- [x] **Project Reorganization**: Consolidated domain logic, application services, and infrastructure details.
- [x] **Dual-Leg Backtesting**: Advanced spread trading simulation with risk metrics (Sharpe, Sortino).
- [x] **Security Maintenance**: Upgrade `quinn-proto` and resolve integration test syntax issues.
- [x] **Exchange Refactor**: Fix `exchange_id` duplication and `RiskManagedExecutor` implementation.

### Jan 2026 - Alpaca Module Review
- [x] **MC-1**: Fix `JoinHandle` leak in `AlpacaWebSocketProvider`
- [x] **MC-3**: Add high-resolution tick latency metrics
- [x] **MC-4**: Optimize hot-path clone in market data routing (Consolidated with MC-2)
- [x] **CB-1**: Implement graceful shutdown for WebSocket task
- [x] **CB-2**: Add circuit breaker for reconnection logic
- [x] **CB-3**: Fix unsafe boolean initialization (UB)
- [x] **N-1**: Use `expect` with context instead of `unwrap_or_default` for `NonZeroU32`
- [x] **Refactor**: Unified `place_order` logic for live/paper trading

### Jan 2026 - Supervisor Resilience
- [x] **NP-5**: Strategy Restart Policy - Exponential backoff with jitter, per-strategy restart budgets, cooldown reset

### Prior Technical Debt
- [x] **MC-1**: Remove dead code fields (`sandbox`, `clock`) in `AlpacaWebSocketProvider`
- [x] **MC-2**: Add numerical stability threshold (1e-12) to ADF test in `filter.rs`
- [x] **CB-1**: Precision-safe price parsing in Alpaca WebSocket
- [x] **MC-3**: O(1) symbol lookup with HashMap in WebSocket handler
- [x] **CB-2**: Alpaca Discovery Fixes (Decimal z-scores, daily bars support)
- [x] **CB-3**: Reliability improvements (Clock injection, graceful error handling)
- [x] **CB-5**: Safe access in Market Data Provider (removed unwrap)
- [x] **N-2**: Performance optimization in Alpaca Utils (pre-computed powers of 10)
