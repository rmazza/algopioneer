# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.7.4] - 2026-02-13
### Fixed
- **Alpaca API**: Fixed "Endpoint Error" on order submission by rounding quantities to 9 decimal places (Alpaca's precision limit). This prevents rejections for fractional shares with excessive precision (e.g., `11.514767...`).

## [1.7.3] - 2026-02-12
### Fixed
- **Recovery Logic**: Implemented robust verification loop in `RecoveryWorker` to fix "Fire and Forget" issue. Now polls for order completion and cancels stuck orders.
- **Alpaca API**: Fixed compilation errors with `apca` 0.30.0 (Tuple struct construction for `Get`/`Delete` and correct field names `filled_quantity`/`average_fill_price`).
- **Executor Trait**: Added `cancel_order` method to `Executor` trait.

## [1.7.2] - 2026-02-11
### Changed
- **Pairs Config**: Reduced active pairs from 8 to 4 (JNJ/ABBV, GIS/KHC, BAC/AXP, MS/GS) to reduce instance load and latency warnings.
- **DynamoDB**: Confirmed `dynamodb` feature enabled by default for production trade logging.

## [1.7.1] - 2026-02-09
### Changed
- **DynamoDB**: Enabled `dynamodb` feature by default in `Cargo.toml` to ensure trade logging is active in production builds.

## [1.7.0] - 2026-02-06
### Added
- **DynamoDB Logging**: Enabled `dynamodb` feature in Docker build to support trade persistence.
- **Autopilot**: Added `autopilot.sh` and `compare_pairs.py` for autonomous pair rebalancing.
- **Monitoring**: Added Terraform monitoring config (`monitoring.tf`) for CloudWatch alarms.

## [1.6.1] - 2026-01-26
### Fixed
- **Alpaca**: Fixed fractional order rejection by automatically falling back to Market orders for quantities < 1.

## [1.6.0] - 2026-01-22
### Added
- **Backtest CLI**: Enhanced backtest command with support for multi-strategy and synthetic data generation.
- **Deployment**: Improved EC2 deployment script `deploy_alpaca.sh` with correct CSV volume mounts for trade logging.
- **Trading Log**: Updated TRADING_LOG.md with recent pair health check results.

## [1.5.0] - 2026-01-21
### Added
- **Order Tracking**: Comprehensive order tracking and position reconciliation system.
- **Risk Management**: Daily Loss Limit implementation (`DailyRiskEngine`) to halt trading on excessive losses.
- **Persistence**: DynamoDB integration for position state persistence.
- **Observability**: New trading journal and tracking logs.

### Fixed
- **Critical**: Zero-quantity order guards in `execute_basis_exit` preventing API rejections.
- **Alpaca**: Fixed `size=0` issue causing circuit breaker trips.
- **Validation**: Improved OrderId validation and error handling.

## [1.4.3] - 2026-01-07
### Added
- Monthly pairs health check workflow.

### Fixed
- Resolved integration test deadlock issues.

## [1.4.2] - 2026-01-06
### Changed
- Refined backoff strategy for rate limits.

## [1.4.1] - 2026-01-06
### Fixed
- Minor bug fixes in moving average strategy.

## [1.4.0] - 2025-12-28
### Fixed
- Resolved deadlock in integration tests.

## [1.3.0] - 2025-12-27
### Added
- Alpaca paper trading support.

## [1.2.0] - 2025-12-22
### Added
- Dual-leg strategy improvements.

## [1.1.0] - 2025-12-17
### Added
- Initial Moving Average strategy implementation.

## [1.0.0] - 2025-12-15
### Added
- Initial release of AlgoPioneer.
- Coinbase and Kraken exchange support.
