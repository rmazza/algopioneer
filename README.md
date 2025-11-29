# algopioneer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**AlgoPioneer** is a robust Rust toolkit designed for algorithmic trading research and execution on Coinbase Advanced Trade. It provides a high-performance foundation for building, testing, and deploying trading strategies, featuring real-time data streaming, backtesting capabilities, and support for complex strategies like Basis Trading.

## Features

*   **Coinbase Advanced Trade Integration**: Seamless interaction with the Coinbase Advanced Trade API via the `cbadv` crate.
*   **Real-time Data**: WebSocket integration for streaming market data (ticker, orderbook).
*   **Strategy Engine**:
    *   **Moving Average Crossover**: A classic trend-following strategy with configurable windows.
    *   **Basis Trading**: A sophisticated delta-neutral strategy exploiting price differences between Spot and Futures markets.
*   **Execution Modes**:
    *   **Live Trading**: Execute real orders on Coinbase.
    *   **Paper Trading**: Simulate execution with real-time data to test strategies without financial risk.
    *   **Backtesting**: Validate strategies against historical data using a built-in backtesting engine.
*   **High Performance**: Built on `tokio` for asynchronous I/O and `polars` for fast data manipulation.

## Prerequisites

*   **Rust**: Stable release (install via [rustup](https://rustup.rs/)).
*   **Coinbase API Credentials**: You need an API Key and Secret from Coinbase Advanced Trade.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/algopioneer.git
    cd algopioneer
    ```

2.  **Configure Environment Variables**:
    Create a `.env` file in the project root based on the example:
    ```bash
    cp .env.example .env
    ```
    Edit `.env` and add your API credentials:
    ```env
    COINBASE_API_KEY=your_api_key
    COINBASE_API_SECRET=your_api_secret
    # Optional: Set to true if using Sandbox API (if supported)
    # COINBASE_USE_SANDBOX=true
    ```

3.  **Build the Project**:
    ```bash
    cargo build --release
    ```

## Usage

AlgoPioneer uses a CLI interface to manage different modes and strategies.

### 1. Standard Trading (Moving Average Crossover)

Run the standard trading bot on a specific product.

**Live Mode:**
```bash
cargo run --release -- trade --product-id BTC-USD --duration 60
```

**Paper Trading Mode (Simulated):**
```bash
cargo run --release -- trade --product-id BTC-USD --paper
```

### 2. Basis Trading (Spot vs Future)

Run the delta-neutral basis trading strategy.

```bash
cargo run --release -- basis-trade --spot-id BTC-USD --future-id BTC-USDT --paper
```

### 3. Backtesting

Run a backtest using historical data (currently configured for the Moving Average strategy).

```bash
cargo run --release -- backtest
```

## Project Structure

*   `src/main.rs`: Application entry point and CLI command orchestration.
*   `src/coinbase/`: Coinbase API client wrapper and WebSocket implementation.
*   `src/strategy/`: Strategy implementations.
    *   `basis_trading.rs`: Logic for the Basis Trading strategy.
    *   `moving_average.rs`: Logic for the Moving Average Crossover strategy.
*   `src/backtest/`: Backtesting engine logic.
*   `src/sandbox/`: Utilities for simulated environments.

## Development

**Run Unit Tests:**
```bash
cargo test
```

**Linting:**
```bash
cargo clippy
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
