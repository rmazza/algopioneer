# Pre-Deployment Checklist

Complete all items before deploying to production or paper trading.

## Code Quality
- [ ] `cargo fmt --all -- --check` passes
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` passes
- [ ] `cargo test` passes
- [ ] `cargo deny check` passes
- [ ] No `TODO` or `FIXME` in critical paths

## Configuration
- [ ] Correct environment variables in `.env`
- [ ] Config file validated (JSON syntax, required fields)
- [ ] `max_tick_age_ms` appropriate for exchange (2000 crypto, 120000 equity)
- [ ] Position sizes within limits
- [ ] Stop loss and take profit thresholds set

## Infrastructure
- [ ] EC2 instance running and accessible
- [ ] Docker daemon running on EC2
- [ ] SSH key accessible (`~/.ssh/trading-key.pem`)
- [ ] Sufficient disk space on EC2

## Trading Specific
- [ ] Paper trading tested successfully
- [ ] Pairs still show cointegration (if pairs trading)
- [ ] Market hours appropriate (equities: 9:30-16:00 ET)
- [ ] No scheduled maintenance on exchange

## Monitoring
- [ ] Health endpoint accessible (`/health`)
- [ ] Metrics endpoint accessible (`/metrics`)
- [ ] CloudWatch logs configured (if applicable)
- [ ] Alert thresholds set

## Rollback Plan
- [ ] Previous image tagged and available
- [ ] Rollback command documented
- [ ] Data backup if needed

## Sign-off
- [ ] Reviewed by: _______________
- [ ] Date: _______________
- [ ] Mode: Paper / Live
