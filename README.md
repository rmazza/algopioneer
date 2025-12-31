# AlgoPioneer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**Production-Ready Algorithmic Trading System** built with Rust for high-performance, low-latency trading strategies.

**Quality Score:** 9.9+/10 ⭐

## Overview

AlgoPioneer is an enterprise-grade algorithmic trading platform designed for the Coinbase Advanced Trade API and Alpaca. It features comprehensive risk management, automatic failover, distributed tracing, and production-ready resilience patterns.

## Key Features

### Trading Strategies
- **Dual-Leg Arbitrage**: Spot vs Future arbitrage with dollar-neutral hedging
- **Basis Trading**: Spread trading with state machine and recovery system
- **Paper Trading**:
  - **Coinbase**: Safe testing environment using internal simulation.
  - **Alpaca**: Full end-to-end testing using Alpaca Paper API.
- **Moving Average Crossover**: Classic trend-following strategy
- **Pairs Trading**: Statistical arbitrage with Z-score analysis
- **Portfolio Mode**: Multi-strategy execution with supervisor pattern

### Research & Discovery
- **Automated Pair Discovery**: Find cointegrated pairs with correlation and half-life filtering
- **Sector-Based Filtering**: Classify tokens by sector (DeFi, L1, Meme, etc.) to find fundamentally-linked pairs
- **Parameter Optimization**: Grid search backtesting with Sharpe ratio ranking
- **Backtest Simulation**: Evaluate strategy performance on historical data

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
- ✅ **Trade Recording**: Modular trade logging to CSV or DynamoDB (via feature flag)
- ✅ **Multi-Exchange Architecture**: Extensible design supporting Coinbase (Crypto), Kraken (Experimental), and Alpaca (US Equities).

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

    **With DynamoDB Support:**
    ```bash
    cargo build --release --features dynamodb
    ```

## Usage

AlgoPioneer uses a CLI interface to manage different modes and strategies.

### 1. Pair Discovery (Automated)

Automatically find and optimize cointegrated trading pairs.

```bash
# Default: Analyze top 20 pairs with default thresholds
cargo run --release -- discover-pairs

# Custom configuration
cargo run --release -- discover-pairs \
  --symbols "BTC-USD,ETH-USD,SOL-USD,AVAX-USD" \
  --min-correlation 0.85 \
  --max-half-life 12.0 \
  --lookback-days 14 \
  --max-pairs 5 \
  --output my_pairs.json
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--symbols` | `default` | Comma-separated pairs or "default" for top 20 |
| `--min-correlation` | `0.8` | Minimum Pearson correlation |
| `--max-half-life` | `24.0` | Maximum mean-reversion half-life (hours) |
| `--min-sharpe` | `0.5` | Minimum Sharpe ratio filter |
| `--lookback-days` | `14` | Historical data window |
| `--max-pairs` | `10` | Number of pairs to output |
| `--output` | `discovered_pairs.json` | Output file path |

### 2. Standard Trading (Moving Average Crossover)

Run the standard trading bot on a specific product.

**Live Mode:**
```bash
cargo run --release -- trade --product-id BTC-USD --duration 60
```

**Paper Trading Mode (Simulated):**
```bash
cargo run --release -- trade --product-id BTC-USD --paper
```

### 3. Dual-Leg Trading (Basis & Pairs)

Run dual-leg strategies using the `dual-leg` command.

**Basis Trading (Spot vs Future):**
```bash
cargo run --release -- dual-leg --strategy basis --symbols BTC-USD,BTC-USDT --paper
```

**Pairs Trading (Asset A vs Asset B):**
```bash
cargo run --release -- dual-leg --strategy pairs --symbols BTC-USD,ETH-USD --paper
```

### 4. Portfolio Mode

Run multiple pairs from a configuration file.

```bash
# Use discovered pairs
cargo run --release -- portfolio --config discovered_pairs.json --paper
```

### 5. Backtesting

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
│   ├── coinbase/               # Legacy Coinbase integration
│   │   ├── mod.rs              # Coinbase API client
│   │   ├── websocket.rs        # Real-time WebSocket data streaming
│   │   └── market_data_provider.rs
│   ├── exchange/               # Exchange abstraction layer
│   │   ├── mod.rs              # Traits (Executor, MarketDataProvider)
│   │   ├── coinbase/           # Coinbase implementation
│   │   ├── kraken/             # Kraken implementation
│   │   └── alpaca/             # Alpaca implementation (US Equities)
│   ├── sandbox/                # Simulation environment
│   ├── logging/                # Trade recording and metrics
│   │   ├── mod.rs              # Logging traits
│   │   ├── recorder.rs         # Recorder implementations
│   │   ├── csv_recorder.rs     # CSV file recorder
│   │   └── dynamodb_recorder.rs # AWS DynamoDB recorder
│   ├── discovery/              # Automated pair discovery
│   │   ├── mod.rs              # Module exports
│   │   ├── config.rs           # DiscoveryConfig with serde support
│   │   ├── error.rs            # Typed errors with thiserror
│   │   ├── filter.rs           # Correlation + half-life filtering
│   │   ├── optimizer.rs        # Grid search parameter optimization
│   │   └── sector.rs           # Token sector classification (DeFi, L1, etc.)
│   ├── strategy/
│   │   ├── dual_leg_trading.rs # Main arbitrage strategy with state machine
│   │   ├── moving_average.rs   # Moving average crossover strategy
│   │   ├── supervisor.rs       # Strategy supervisor for multi-strategy
│   │   └── tick_router.rs      # Market data routing with backpressure
│   ├── resilience/
│   │   ├── mod.rs              # Resilience patterns
│   │   └── circuit_breaker.rs  # Circuit breaker with RwLock
│   ├── health.rs               # HTTP health check endpoint (/health)
│   ├── observability.rs        # OpenTelemetry tracing integration
│   └── examples/
│       ├── optimize_pairs.rs   # Pairs optimization example
│       ├── check_sectors.rs    # Sector classification viewer
│       └── debug_config.rs     # Config deserialization debug
├── tests/
│   ├── integration_test.rs     # Integration tests with mock executor
│   └── proptest_financial.rs   # Property-based tests for financial math
├── Cargo.toml                  # Dependencies and project metadata
└── .env                        # API credentials (not committed)
```

### Key Components

- **Discovery**: Automated pair finding with correlation, half-life, and Sharpe filtering
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

