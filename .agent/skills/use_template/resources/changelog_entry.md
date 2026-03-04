# Changelog Entry Format

This project follows [Keep a Changelog](https://keepachangelog.com/).

## Format

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features

### Changed
- Changes to existing functionality

### Deprecated
- Features that will be removed in future

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Security fixes
```

## Example Entry

```markdown
## [1.4.3] - 2026-01-10

### Added
- Alpaca equity pairs trading support
- WebSocket tick latency metrics for observability
- Circuit breaker for Alpaca API rate limiting

### Changed
- Increased `max_tick_age_ms` default from 2000 to 120000 for equity polling
- Refactored `place_order` to eliminate live/paper code duplication

### Fixed
- JoinHandle leak in WebSocket task
- f64 usage in warmup data replaced with Decimal
- String allocation in hot path websocket handler

### Security
- Updated dependencies per `cargo deny` advisories
```

## Guidelines

1. **User-focused**: Write for someone upgrading, not for code reviewers
2. **Actionable**: What does the user need to know or do?
3. **Grouped**: All changes for a version in one section
4. **Chronological**: Newest version at the top
