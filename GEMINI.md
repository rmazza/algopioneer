# Project Overview

This is a Rust-based algorithmic trading project named `algopioneer`. It is designed to interact with the Coinbase Advanced Trade API for trading research and execution, supporting both live and paper trading modes.

**Key Technologies:**

*   **Language:** Rust
*   **Core Libraries:**
    *   `cbadv`: For interacting with the Coinbase Advanced Trade API.
    *   `polars`: For high-performance data manipulation and analysis.
    *   `ta`: For technical analysis indicators.
    *   `tokio`: Asynchronous runtime for handling API requests.
    *   `dotenv`: For managing environment variables (API keys).
    *   `clap`: For command-line argument parsing.
    *   `rust_decimal`: For high-precision financial calculations.
    *   `tokio-tungstenite`: For WebSocket connections to Coinbase.

**Architecture:**

The project is structured as a modular application with a CLI entry point defined in `src/main.rs`.

*   **Entry Point (`src/main.rs`):** Handles CLI argument parsing and dispatches commands to the appropriate execution logic.
*   **Strategies (`src/strategy/`):** Contains the trading logic.
    *   `moving_average.rs`: Implements a Moving Average Crossover strategy.
    *   `basis_trading.rs`: Implements a Basis Trading strategy (Spot vs Future arbitrage) with risk management and a **Queue-Based Recovery System** for handling failed execution legs.
*   **Coinbase Integration (`src/coinbase/`):** Handles API interactions and WebSocket data streams.
*   **Backtesting (`src/backtest/`):** Provides a framework for testing strategies against historical data.

**Features:**

*   **Live Trading:** Execute trades on Coinbase Advanced Trade.
*   **Paper Trading:** Simulate trade execution for testing strategies without real funds.
*   **Backtesting:** Evaluate strategy performance using historical data.
*   **Basis Trading:** specialized strategy for arbitrage between spot and future markets.

# Building and Running

**Prerequisites:**

*   Rust and Cargo installed.
*   A `.env` file in the root directory with the following variables:
    *   `COINBASE_API_KEY`
    *   `COINBASE_API_SECRET`

**Build:**

```bash
cargo build --release
```

**Run:**

The application uses subcommands to control its behavior.

*   **Trade (Moving Average Strategy):**
    ```bash
    cargo run --release -- trade --product-id BTC-USD --duration 60
    ```
    *   Use `--paper` for paper trading mode.

*   **Basis Trading:**
    ```bash
    cargo run --release -- basistrade --spot-id BTC-USD --future-id BTC-USDT
    ```
    *   Use `--paper` for paper trading mode.

*   **Backtest:**
    ```bash
    cargo run --release -- backtest
    ```

**Test:**

```bash
cargo test
```

**Integration Test:**

```bash
cargo test --test integration_test
```

# Development Conventions

*   **Configuration:** API keys and other secrets are managed through a `.env` file.
*   **Asynchronous Operations:** The project uses the `tokio` runtime for asynchronous operations.
*   **Error Handling:** The code uses `Result` and `Box<dyn std::error::Error>` for error handling.
*   **Precision:** Financial calculations should use `rust_decimal` to avoid floating-point errors.
