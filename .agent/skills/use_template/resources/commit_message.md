# Commit Message Format

This project follows [Conventional Commits](https://www.conventionalcommits.org/).

## Format

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

## Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no logic change |
| `refactor` | Code change that neither fixes nor adds |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `chore` | Maintenance, deps updates |
| `ci` | CI/CD changes |

## Scopes (Project-Specific)

| Scope | Area |
|-------|------|
| `strategy` | Trading strategies |
| `coinbase` | Coinbase integration |
| `alpaca` | Alpaca integration |
| `kraken` | Kraken integration |
| `discovery` | Pair discovery |
| `metrics` | Prometheus metrics |
| `health` | Health checks |
| `cli` | Command-line interface |
| `trading` | Trading engine |
| `exchange` | Exchange abstraction |

## Examples

```
feat(alpaca): add WebSocket reconnection with exponential backoff

fix(strategy): prevent division by zero in z-score calculation

perf(discovery): parallelize cointegration tests with rayon

chore(deps): update tokio to 1.35.0

docs(readme): add Alpaca paper trading setup instructions
```

## Breaking Changes

Append `!` after type for breaking changes:

```
feat(cli)!: rename --paper flag to --mode paper
```
