# Code Quality Checklist

Standards for all code changes in algopioneer.

## Rust Standards
- [ ] No `unwrap()` or `expect()` in production code paths
- [ ] All errors are typed with `thiserror`
- [ ] `Result<T, E>` used for fallible operations
- [ ] No `clone()` in hot paths without justification
- [ ] All `unsafe` blocks have `// SAFETY:` comments

## Financial Correctness
- [ ] `rust_decimal::Decimal` used for all money/prices
- [ ] No `f64` for financial calculations
- [ ] Explicit rounding with documented strategy
- [ ] PnL calculations verified manually

## Concurrency
- [ ] No `Mutex` held across `.await`
- [ ] Deadlock-free lock ordering documented
- [ ] Channels preferred over shared state
- [ ] Graceful shutdown handles all tasks

## Observability
- [ ] Structured logging with `tracing`
- [ ] Key operations have spans
- [ ] Errors include context
- [ ] Metrics for critical operations

## Testing
- [ ] Unit tests for business logic
- [ ] Edge cases covered
- [ ] Mock implementations for external services
- [ ] Integration tests where applicable

## Documentation
- [ ] Public functions have doc comments
- [ ] Complex logic explained
- [ ] Examples in doc comments for key functions
- [ ] CHANGELOG updated for user-facing changes
