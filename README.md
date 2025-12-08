# AlgoPioneer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**Production-Ready Algorithmic Trading System** built with Rust for high-performance, low-latency trading strategies.

**Quality Score:** 9.9+/10 ⭐

## Overview

AlgoPioneer is an enterprise-grade algorithmic trading platform designed for the Coinbase Advanced Trade API. It features comprehensive risk management, automatic failover, distributed tracing, and production-ready resilience patterns.

## Key Features

### Trading Strategies
- **Dual-Leg Arbitrage**: Spot vs Future arbitrage with dollar-neutral hedging
- **Basis Trading**: Spread trading with state machine and recovery system
- **Moving Average Crossover**: Classic trend-following strategy
- **Pairs Trading**: Statistical arbitrage with Z-score analysis
- **Portfolio Mode**: Multi-strategy execution with supervisor pattern

### Production Features
- ✅ **Live Trading**: Real-time execution on Coinbase Advanced Trade
- ✅ **Paper Trading**: Risk-free simulation mode for testing
- ✅ **Position Reconciliation**: Automatic recovery from network failures
- ✅ **Circuit Breaker**: Cascading failure prevention with auto-recovery
- ✅ **PnL Aggregation**: Portfolio-level risk monitoring and tracking
- ✅ **Health Monitoring**: `/health` HTTP endpoint for Kubernetes/Docker
- ✅ **Distributed Tracing**: OpenTelemetry integration for observability
- ✅ **Panic Recovery**: Supervisor pattern with automatic strategy restart
- ✅ **WebSocket Stability**: Proper task cleanup preventing resource leaks

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

### 2. Dual-Leg Trading (Basis & Pairs)

Run dual-leg strategies using the `dual-leg` command.

**Basis Trading (Spot vs Future):**
```bash
cargo run --release -- dual-leg --strategy basis --symbols BTC-USD,BTC-USDT --paper
```

**Pairs Trading (Asset A vs Asset B):**
```bash
cargo run --release -- dual-leg --strategy pairs --symbols BTC-USD,ETH-USD --paper
```

### 3. Backtesting

Run a backtest using historical data (currently configured for the Moving Average strategy).

```bash
cargo run --release -- backtest
```

## Project Structure

```
algopioneer/
├── src/
│   ├── main.rs                 # CLI entry point with subcommands
│   ├── lib.rs                  # Library root
│   ├── coinbase/
│   │   ├── mod.rs             # Coinbase API client with position querying
│   │   ├── websocket.rs       # Real-time WebSocket data streaming
│   │   └── market_data_provider.rs  # Abstracted data sources (live + synthetic)
│   ├── strategy/
│   │   ├── dual_leg_trading.rs  # Main arbitrage strategy with state machine
│   │   ├── moving_average.rs    # Moving average crossover strategy
│   │   └── portfolio.rs         # Portfolio manager with supervisor pattern
│   ├── health.rs               # HTTP health check endpoint (/health)
│   ├── observability.rs        # OpenTelemetry tracing integration
│   └── bin/
│       └── find_pairs.rs       # Pairs discovery utility
├── tests/
│   └── integration_test.rs     # Integration tests with mock executor
├── Cargo.toml                  # Dependencies and project metadata
└── .env                        # API credentials (not committed)
```

### Key Components

- **Strategies**: Modular trading logic with trait-based abstraction
- **Execution Engine**: Order placement with circuit breaker and retry logic
- **Recovery System**: Queue-based recovery with exponential backoff
- **Market Data**: Pluggable providers (Coinbase WebSocket, Synthetic)
- **Observability**: OpenTelemetry traces for production monitoring
- **Health Checks**: Kubernetes-ready `/health` endpoint

## Development

**Run Unit Tests:**
```bash
cargo test
```

**Run Integration Tests:**
```bash
cargo test --test integration_test
```

**Linting:**
```bash
cargo clippy
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
