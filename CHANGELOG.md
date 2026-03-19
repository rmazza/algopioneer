# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.9.4] - 2026-03-19
### Added
- **Observability**: Promoted Z-score logging from `debug` to `info` level and added pair identification to log messages.

## [1.9.3] - 2026-03-18
### Fixed
- **Strategy**: Resolved 'Ghost Position' issue in `dual_leg` by isolating symbols across active pairs.
- **Strategy**: Mitigated execution 'churning' by increasing `exit_z_score` from 0.2 to 0.8 and `entry_cooldown_ms` to 300s.
- **Infrastructure**: Completed `Executor` trait implementation for `CoinbaseExchangeClient` and `RiskManagedExecutor`.
- **Reliability**: Hardened strategy stability against Alpaca 'wash trade' rejections.

## [1.9.2] - 2026-03-14
### Changed
- **Maintenance**: Performed workspace-wide formatting and fixed several Clippy warnings for improved code quality.
- **Refactor**: Cleaned up the root directory by removing temporary artifacts and legacy configuration files.
- **Project Structure**: Consolidated agent skills into `.gemini/skills/` and standardized command configurations.

### Fixed
- **Security**: Upgraded `quinn-proto` to v0.11.14 to address RUSTSEC-2026-0037.
- **Exchange**: Removed duplicate `exchange_id` from `ExchangeClient` and corrected `RiskManagedExecutor` implementation.
- **Tests**: Resolved syntax errors in `integration_test.rs` and stabilized integration testing suite.

## [1.9.1] - 2026-03-12
### Fixed
- **CI**: Removed immutable 'latest' tag from ECR push to comply with repository security policy.
- **Executor**: Implemented 'exchange_id' for all Executor implementations to improve strategy identification.
- **Alpaca**: Resolved wash trade rejections by adding a 500ms execution delay and improved recovery success rates.

## [1.9.0] - 2026-03-11
### Added
- **Clean Architecture**: Reorganized the entire project into `domain`, `application`, `infrastructure`, and `interface` layers for better separation of concerns and maintainability.
- **Backtesting**: Implemented a dual-leg backtesting engine with support for spread trading simulation and advanced risk metrics (Sharpe, Sortino, Profit Factor).
- **Domain Models**: Introduced dedicated domain entities for orders, exchanges, and events.

### Changed
- **Project Structure**: Relocated core logic to `src/domain` and `src/application`, moving external integrations to `src/infrastructure`.
- **CLI**: Standardized command handlers and configuration in `src/interface`.

### Fixed
- **Alpaca Client**: Improved order handling and stability, including proactive order cancellation to prevent wash trade errors (merged from v1.8.2 candidate).
- **Test Stability**: Fixed several integration test race conditions and improved overall reliability.

## [1.8.1] - 2026-03-04
### Fixed
- **Alpaca API**: Resolved stale ticks load-shedding by replacing blocking WebSocket senders with non-blocking logic and increasing latency threshold.
- **Alpaca API**: Corrected strategy fractional share logic to treat `order_size` as USD allocation and floor final quantities to whole shares.

## [1.8.0] - 2026-02-17
### Added
- **Safety Guards**: Position imbalance detection with automatic safety guards to prevent runaway positions.

### Changed
- **Architecture**: Restructured strategy module into submodules — extracted execution engine, recovery worker, entry managers, exit policies, throttle utilities, and validators into dedicated files.

### Fixed
- **Code Review**: Addressed code review findings for `dual_leg`, `supervisor`, and `tick_router` modules.
- **Kill Switch**: Fixed kill switch to use opposite side for unwinding positions.
- **Error Handling**: Preserved both errors on dual-leg double failure instead of swallowing the first.
- **Recovery**: Use limit orders instead of market orders for recovery to prevent slippage.
- **Recovery**: Acquire semaphore before spawn to prevent task explosion under load.

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
