# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
