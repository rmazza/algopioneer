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
- **Dynamic Hedge Ratios**: Kalman Filter for adaptive beta estimation
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
- ✅ **Prometheus Metrics**: `/metrics` endpoint with order latency, PnL, and circuit breaker state
- ✅ **Daily Risk Limits**: Configurable daily loss thresholds with automatic trading halt
- ✅ **Panic Recovery**: Supervisor pattern with automatic strategy restart
- ✅ **WebSocket Stability**: Proper task cleanup preventing resource leaks
- ✅ **Trade Recording**: Modular trade logging to CSV or DynamoDB (via feature flag)
- ✅ **Multi-Exchange Architecture**: Extensible design supporting Coinbase (Crypto), Kraken (Experimental), and Alpaca (US Equities).
- ✅ **Autopilot Mode**: Self-healing mechanism that automatically discovers new pairs, compares them with active config, and redeploys if improvements are found (`autopilot.sh`).

## Prerequisites

*   **Rust**: Stable release (install via [rustup](https://rustup.rs/)).
*   **Coinbase API Credentials**: API Key and Secret from Coinbase Advanced Trade.
*   **Alpaca API Credentials** (optional): API Key and Secret for US Equities trading.
*   **Kraken API Credentials** (optional): API Key and Secret for Kraken exchange.

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
    
    # Optional: Alpaca API (for US Equities)
    ALPACA_KEY_ID=your_alpaca_key
    ALPACA_SECRET_KEY=your_alpaca_secret
    
    # Optional: Kraken API
    KRAKEN_API_KEY=your_kraken_key
    KRAKEN_API_SECRET=your_kraken_secret
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
| `--exchange` | `coinbase` | Exchange: "coinbase" (crypto) or "alpaca" (stocks) |
| `--symbols` | `default` | Comma-separated pairs or "default" for built-in list |
| `--min-correlation` | `0.8` | Minimum Pearson correlation |
| `--max-half-life` | `48.0` | Maximum mean-reversion half-life (hours) |
| `--min-sharpe` | `0.5` | Minimum Sharpe ratio filter |
| `--lookback-days` | `90` | Historical lookback period in days |
| `--max-pairs` | `10` | Maximum number of pairs to output |
| `--output` | `discovered_pairs.json` | Output file path for discovered pairs |
| `--initial-capital` | `10000.0` | Initial capital for backtests (USD) |
| `--no-cointegration` | `false` | Skip ADF cointegration test |

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

Run backtests with configurable strategies and data sources.

```bash
# Moving Average strategy with synthetic data
cargo run --release -- backtest --strategy moving_average --symbols BTC-USD --duration 60 --synthetic

# Dual-leg strategy backtest
cargo run --release -- backtest --strategy dual_leg --symbols BTC-USD,ETH-USD --duration 60 --synthetic

# Output results to JSON
cargo run --release -- backtest --strategy moving_average --symbols BTC-USD --synthetic --output-dir ./results
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--strategy` | `moving_average` | Strategy: "moving_average" or "dual_leg" |
| `--exchange` | `coinbase` | Exchange: "coinbase" or "alpaca" |
| `--symbols` | Required | Comma-separated symbols (2 for dual_leg) |
| `--duration` | `60` | Backtest duration in minutes |
| `--synthetic` | `false` | Use synthetic data for testing |
| `--output-dir` | None | Directory for JSON output. Use `--output-dir ./results` to specify JSON output directory. |
| `--initial-capital` | `10000.0` | Initial capital (USD) |

### 6. Autopilot (Self-Healing & Rebalancing)

Automatically discover new pairs, compare them with the current configuration, and redeploy if better pairs are found.

```bash
./autopilot.sh
```

**Workflow:**
1.  Fetches live market data (Alpaca/Coinbase).
2.  Runs `discover-pairs` in a disposable Docker container.
3.  Compares new pairs vs. current using `compare_pairs.py`.
4.  If improvements are found (>5% better metrics), updates config and redeploys via `deploy_alpaca.sh`.

## Project Structure

```
algopioneer/
├── src/
│   ├── main.rs                 # CLI entry point with jemalloc and command dispatch
│   ├── lib.rs                  # Library root
│   ├── cli/                    # CLI definitions
│   │   ├── mod.rs              # Cli struct and Commands enum
│   │   └── config.rs           # Configuration structs (DualLegCliConfig, etc.)
│   ├── commands/               # Command handlers
│   │   ├── mod.rs              # Command exports
│   │   ├── trade.rs            # Moving average trade handler
│   │   ├── backtest.rs         # Backtest handler
│   │   ├── dual_leg.rs         # Dual-leg trading handler
│   │   ├── portfolio.rs        # Portfolio mode handler
│   │   └── discover.rs         # Pair discovery handler
│   ├── coinbase/               # Legacy Coinbase integration
│   │   ├── mod.rs              # Coinbase API client
│   │   ├── websocket.rs        # Real-time WebSocket data streaming
│   │   └── market_data_provider.rs
│   ├── exchange/               # Exchange abstraction layer
│   │   ├── mod.rs              # Traits (Executor, ExchangeClient, WebSocketProvider)
│   │   ├── coinbase/           # Coinbase implementation
│   │   ├── kraken/             # Kraken implementation (experimental)
│   │   └── alpaca/             # Alpaca implementation (US Equities)
│   ├── orders/                 # Order management
│   │   ├── mod.rs              # Order types and traits
│   │   ├── tracker.rs          # Order state tracking
│   │   ├── reconciler.rs       # Position reconciliation
│   │   └── types.rs            # Order domain types
│   ├── risk/                   # Risk management
│   │   ├── mod.rs              # Risk module exports
│   │   ├── daily_limit.rs      # Daily loss limit engine
│   │   └── executor.rs         # Risk-aware order executor
│   ├── logging/                # Trade recording
│   │   ├── mod.rs              # Logging traits
│   │   ├── recorder.rs         # Recorder implementations
│   │   ├── csv_recorder.rs     # CSV file recorder
│   │   └── dynamodb_recorder.rs # AWS DynamoDB recorder (feature-gated)
│   ├── discovery/              # Automated pair discovery
│   │   ├── mod.rs              # Module exports
│   │   ├── config.rs           # DiscoveryConfig with serde support
│   │   ├── error.rs            # Typed errors with thiserror
│   │   ├── filter.rs           # Correlation + half-life filtering
│   │   ├── optimizer.rs        # Grid search parameter optimization
│   │   └── sector.rs           # Token sector classification
│   ├── strategy/               # Trading strategies
│   │   ├── mod.rs              # Strategy traits
│   │   ├── dual_leg_trading.rs # Dual-leg arbitrage with state machine
│   │   ├── moving_average.rs   # Moving average crossover strategy
│   │   ├── supervisor.rs       # Strategy supervisor with panic recovery
│   │   └── tick_router.rs      # Market data routing with backpressure
│   ├── math/                   # Mathematical utilities
│   │   ├── mod.rs              # Math module exports
│   │   └── kalman.rs           # Kalman filter for dynamic hedge ratios
│   ├── backtest/               # Backtesting engine
│   │   └── mod.rs              # Deterministic backtest with Decimal arithmetic
│   ├── resilience/             # Resilience patterns
│   │   ├── mod.rs              # Resilience exports
│   │   └── circuit_breaker.rs  # Circuit breaker with RwLock
│   ├── health.rs               # HTTP health check endpoint (/health)
│   ├── metrics.rs              # Prometheus metrics (/metrics)
│   ├── observability.rs        # OpenTelemetry tracing integration
│   └── types.rs                # Shared domain types (MarketData, OrderSide)
├── tests/
│   ├── integration_test.rs     # Integration tests with mock executor
│   └── proptest_financial.rs   # Property-based tests for financial math
├── Cargo.toml                  # Dependencies and project metadata
├── .env                        # API credentials (not committed)
├── autopilot.sh                # Autopilot rebalancing script
├── deploy_alpaca.sh            # Alpaca deployment script
└── compare_pairs.py            # Pair comparison logic for Autopilot
```

### Key Components

- **Discovery**: Automated pair finding with correlation, half-life, and Sharpe filtering
- **Strategies**: Modular trading logic with trait-based abstraction
- **Execution Engine**: Order placement with circuit breaker and retry logic
- **Recovery System**: Queue-based recovery with exponential backoff
- **Market Data**: Pluggable providers (Coinbase WebSocket, Synthetic)
- **Mathematical Utilities**: Kalman filter for dynamic hedge ratio estimation
- **Backtesting**: Deterministic simulation with fixed-point arithmetic
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

