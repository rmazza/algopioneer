---
description: Standardized backtesting commands for sanity checks and deep analysis
---

# Run Backtest Suite

This workflow provides standardized commands for running backtests, ensuring consistency and making it easier to verify strategy performance.

## Prerequisites

- Ensure you have historical data available or internet access for the backtester to fetch it.
- Build the release binary for performance: `cargo build --release`.

## Steps

### 1. Sanity Check (Quick)
Run a short backtest on a single pair to verify that the system is functioning without crashing.

```bash
cargo run --release -- backtest --strategy moving-average --symbols BTC-USD --start-date 2024-01-01 --end-date 2024-01-02
```

### 2. Standard Performance Test
Run a longer backtest (e.g., 1 month) to gauge strategy metrics.

```bash
cargo run --release -- backtest --strategy moving-average --symbols BTC-USD --duration 30d
```

### 3. Deep Analysis (Multi-Pair)
Run a backtest across multiple pairs to test portfolio interactions or comparative performance.

```bash
cargo run --release -- backtest --strategy dual-leg --symbols BTC-USD,ETH-USD --duration 60d
```

### 4. Verify Results
Check the output for:
- **Total Return**: Is it reasonable?
- **Sharpe Ratio**: Is it positive?
- **Max Drawdown**: Is it within limits?
- **Trades**: Are trades actually being executed?
