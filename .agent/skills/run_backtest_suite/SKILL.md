---
name: Run Backtest Suite
description: Standardized backtesting wizard for sanity checks and deep analysis.
---

# Run Backtest Suite

This skill simplifies running backtests by constructing the complex CLI commands for you. It supports quick sanity checks, standard performance tests, and multi-pair analysis.

## Instructions

### 1. Configure Backtest
Decide on the parameters:
1.  **Strategy**: `moving-average` | `dual-leg`
2.  **Symbols**: `BTC-USD` | `ETH-USD` | `BTC-USD,ETH-USD`
3.  **Duration**: e.g., `30d` (30 days), `2024-01-01` (start date)

### 2. Construct Command
Build the `cargo run` command based on the configuration.

*   **Sanity Check (Quick)**:
    ```bash
    cargo run --release -- backtest --strategy moving-average --symbols BTC-USD --duration 1d
    ```

*   **Standard Test (30 Days)**:
    ```bash
    cargo run --release -- backtest --strategy moving-average --symbols BTC-USD --duration 30d
    ```

*   **Multi-Leg / Arbitrage**:
    ```bash
    cargo run --release -- backtest --strategy dual-leg --symbols BTC-USD,ETH-USD --duration 60d
    ```

### 3. Execute
Run the command. Ensure you are in the project root and `cargo build --release` has been run recently for best performance.

### 4. Analyze Results
Review the output metrics:
*   **Total Return**: Net profit/loss.
*   **Sharpe Ratio**: Risk-adjusted return (> 1.0 is good).
*   **Max Drawdown**: Maximum peak-to-valley decline.
*   **Trade Count**: Ensure statistical significance (> 30 trades).
