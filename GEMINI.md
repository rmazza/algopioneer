# Project Overview

This is a production-ready, enterprise-grade Rust-based algorithmic trading system named `algopioneer`. It is designed to interact with multiple cryptocurrency exchanges (Coinbase, Kraken) and equity brokers (Alpaca) for trading research and execution, supporting both live and paper trading modes with comprehensive risk management and observability.

**Quality Score:** 9.9+/10 (Production-Ready)

**Key Technologies:**

*   **Language:** Rust (Edition 2021)
*   **Core Libraries:**
    *   `cbadv`: Coinbase Advanced Trade API client
    *   `apca`: Alpaca Markets API client for equities
    *   `polars`: High-performance data manipulation and analysis
    *   `ta`: Technical analysis indicators
    *   `tokio`: Asynchronous runtime for concurrent operations
    *   `axum`: HTTP server for health checks and metrics
    *   `opentelemetry`: Distributed tracing and observability
    *   `prometheus`: Metrics collection and exposure
    *   `dashmap`: Lock-free concurrent hash maps
    *   `rust_decimal`: High-precision financial calculations
    *   `num-decimal`: Additional decimal number handling for Alpaca integration
    *   `tokio-tungstenite`: WebSocket connections
    *   `tikv-jemallocator`: Optimized memory allocation for long-running processes
    *   `governor`: Rate limiting for API calls
    *   `thiserror`: Typed error handling
    *   `bytes`: Zero-copy string handling
    *   `uuid`: Trade ID generation for logging/recording
    *   `proptest`: Property-based testing (dev dependency)
    *   `aws-sdk-dynamodb` / `aws-config`: Optional AWS DynamoDB trade recording

**Architecture:**

Production-grade modular application with dependency injection, trait-based abstractions, and comprehensive error handling.

*   **Entry Point (`src/main.rs`):** CLI with subcommands for trading, backtesting, portfolio management, and pair discovery
*   **CLI (`src/cli/`):**
    *   `mod.rs`: CLI struct and command definitions (Clap-based)
    *   `config.rs`: CLI configuration structs (`SimpleTradingConfig`, `DualLegCliConfig`)
*   **Commands (`src/commands/`):**
    *   `mod.rs`: Command re-exports
    *   `trade.rs`: Moving Average strategy trade handler
    *   `backtest.rs`: Backtesting evaluation handler
    *   `dual_leg.rs`: Dual-leg (basis/pairs) trading handler
    *   `portfolio.rs`: Multi-pair portfolio trading handler
    *   `discover.rs`: Pair discovery command handler
*   **Trading (`src/trading/`):**
    *   `mod.rs`: Trading module re-exports
    *   `simple_engine.rs`: Simple trading engine for Moving Average strategy
*   **Strategies (`src/strategy/`):**
    *   `mod.rs`: Strategy trait definitions and re-exports
    *   `dual_leg_trading.rs`: Main dual-leg arbitrage strategy with state machine, recovery system, and circuit breaker
    *   `moving_average.rs`: Moving Average Crossover strategy
    *   `supervisor.rs`: Generic strategy supervisor with panic recovery, PnL aggregation, and health monitoring
    *   `tick_router.rs`: Lock-free market data distribution with backpressure handling
*   **Exchange Abstraction (`src/exchange/`):**
    *   `mod.rs`: Exchange-agnostic traits (`Executor`, `ExchangeClient`, `WebSocketProvider`) and factory functions
    *   `coinbase/`: Coinbase-specific implementation (client, websocket, market data provider)
    *   `kraken/`: Kraken-specific implementation (client, websocket, market data provider)
    *   `alpaca/`: Alpaca-specific implementation
        *   `alpaca_client.rs`: Alpaca API client with order execution and position tracking
        *   `websocket.rs`: Real-time equity market data streaming via Alpaca WebSocket
        *   `utils.rs`: Conversion utilities for Alpaca types
*   **Coinbase Integration (`src/coinbase/`):**
    *   `mod.rs`: API client with position querying
    *   `websocket.rs`: Real-time market data streaming
    *   `market_data_provider.rs`: Abstracted data sources (live + synthetic)
*   **Risk Management (`src/risk/`):**
    *   `mod.rs`: Risk module re-exports
    *   `daily_limit.rs`: Daily loss limit enforcement (`DailyRiskEngine`)
    *   `executor.rs`: Risk-aware trade executor wrapper
*   **Orders (`src/orders/`):**
    *   `mod.rs`: Order module re-exports
    *   `tracker.rs`: Order state tracking with TOCTOU-safe operations
    *   `reconciler.rs`: Position reconciliation after network failures
    *   `types.rs`: Order-related types (OrderRecord, ExecutionReport, etc.)
*   **Discovery (`src/discovery/`):**
    *   `mod.rs`: Pair discovery and optimization module
    *   `config.rs`: Discovery configuration
    *   `filter.rs`: Candidate pair filtering with cointegration tests
    *   `optimizer.rs`: Grid search parameter optimization
    *   `sector.rs`: Sector classification for pair grouping
    *   `error.rs`: Typed discovery errors
*   **Logging (`src/logging/`):**
    *   `mod.rs`: Trade recording module re-exports
    *   `recorder.rs`: Base trade recording logic
    *   `csv_recorder.rs`: CSV file trade recording
    *   `dynamodb_recorder.rs`: AWS DynamoDB trade recording (feature-gated)
    *   `tracing_recorder.rs`: Tracing-based trade recording
*   **Resilience (`src/resilience/`):**
    *   `mod.rs`: Resilience module re-exports
    *   `circuit_breaker.rs`: Circuit breaker pattern with RwLock-based implementation
*   **Math (`src/math/`):**
    *   `mod.rs`: Mathematical utilities module re-exports
    *   `kalman.rs`: Kalman Filter for dynamic hedge ratio estimation in pairs trading
*   **Backtest (`src/backtest/`):**
    *   `mod.rs`: Deterministic backtesting engine with fixed-point arithmetic (`Decimal`) for reproducible results
*   **State (`src/state/`):**
    *   `mod.rs`: Application state types (`PositionDetail`, `TradeState`)
*   **Observability (`src/observability.rs`):** OpenTelemetry integration for distributed tracing
*   **Metrics (`src/metrics.rs`):** Prometheus metrics with lock-free atomics
*   **Health (`src/health.rs`):** HTTP health check endpoint for Kubernetes/Docker
*   **Events (`src/events.rs`):** Event types for strategy communication
*   **Types (`src/types.rs`):** Shared domain types (MarketData, OrderSide, etc.)

**Production Features:**

*   **Live Trading:** Execute trades on Coinbase/Kraken/Alpaca with full error handling
*   **Paper Trading:**
    *   **Coinbase:** Simulate trades without real funds (Internal Simulation)
    *   **Alpaca:** Uses real **Alpaca Paper API** for full end-to-end verification
*   **Backtesting:** Evaluate strategy performance using historical data
*   **Basis Trading:** Spot vs Future arbitrage with dollar-neutral hedging
*   **Pairs Trading:** Cointegration-based statistical arbitrage
*   **Pair Discovery:** Automatic discovery and optimization of cointegrated pairs
*   **Daily Risk Limits:** Configurable daily loss thresholds with automatic trading halt
*   **Position Reconciliation:** Automatic recovery from network failures
*   **Circuit Breaker:** Cascading failure prevention with auto-recovery
*   **Strategy Supervisor:** Panic recovery and automatic restart with configurable policies
*   **PnL Aggregation:** Portfolio-level risk monitoring via DashMap
*   **Trade Recording:** CSV, DynamoDB (optional), and tracing-based trade logging
*   **Health Monitoring:** `/health` endpoint for orchestration
*   **Prometheus Metrics:** `/metrics` endpoint with order latency, PnL, tick latency, and circuit breaker state
*   **Distributed Tracing:** OpenTelemetry/Jaeger integration for production observability
*   **Rate Limiting:** `governor` for API rate limit compliance
*   **Memory Optimization:** jemalloc allocator for reduced fragmentation
*   **Dynamic Hedge Ratios:** Kalman Filter for adaptive pairs trading beta estimation

# Building and Running

**Prerequisites:**

*   Rust and Cargo installed.
*   A `.env` file in the root directory with the following variables:
    *   `COINBASE_API_KEY`
    *   `COINBASE_API_SECRET`
    *   `KRAKEN_API_KEY` (optional, for Kraken support)
    *   `KRAKEN_API_SECRET` (optional, for Kraken support)
    *   `APCA_API_KEY_ID` (optional, for Alpaca support)
    *   `APCA_API_SECRET_KEY` (optional, for Alpaca support)

**Build:**

```bash
cargo build --release
```

**Build with DynamoDB support:**

```bash
cargo build --release --features dynamodb
```

**Run:**

The application uses subcommands to control its behavior.

*   **Trade (Moving Average Strategy):**
    ```bash
    cargo run --release -- trade --product-id BTC-USD --duration 60
    ```
    *   Use `--paper` for paper trading mode.

*   **Dual-Leg Trading (Basis/Pairs):**
    ```bash
    cargo run --release -- dual-leg --strategy pairs --symbols BTC-USD,ETH-USD
    ```
    *   Use `--paper` for paper trading mode.
    *   Use `--exchange alpaca` for Alpaca equities (e.g., `--symbols AAPL,MSFT`).

*   **Portfolio Mode:**
    ```bash
    cargo run --release -- portfolio --config pairs_config.json
    ```
    *   Use `--paper` for paper trading mode.
    *   Use `--exchange alpaca` for Alpaca equities.

*   **Discover Pairs:**
    ```bash
    cargo run --release -- discover-pairs --symbols default --min-correlation 0.8
    ```
    *   Use `--exchange alpaca` for Alpaca equity pair discovery.

*   **Backtest:**
    ```bash
    cargo run --release -- backtest --strategy moving_average --symbols BTC-USD --duration 60 --synthetic
    ```
    *   Use `--strategy dual_leg` for dual-leg backtesting with two symbols.
    *   Use `--synthetic` to generate synthetic data for CI testing.
    *   Use `--output-dir ./results` to specify JSON output directory.

**Test:**

```bash
cargo test
```

**Integration Test:**

```bash
cargo test --test integration_test
```

**Examples:**

```bash
cargo run --example optimize_pairs
```

# Development Conventions

*   **Configuration:** API keys and secrets managed through `.env` file
*   **Asynchronous Operations:** `tokio` runtime for all async operations
*   **Error Handling:** Typed errors with `thiserror` and `Result<T, Box<dyn Error + Send + Sync>>`
*   **Precision:** `rust_decimal::Decimal` for all financial calculations
*   **Dependency Injection:** Trait-based abstractions (`Executor`, `ExchangeClient`, `MarketDataProvider`, `ExitPolicy`, `WebSocketProvider`)
*   **Testing:** Mock implementations via `mockall` for offline testing; `proptest` for property-based testing
*   **Observability:** OpenTelemetry tracing + Prometheus metrics for production monitoring
*   **Resilience:** Circuit breakers, panic recovery, position reconciliation, rate limiting, daily loss limits
*   **Performance:** RwLock for concurrent reads, DashMap for lock-free maps, jemalloc allocator, pre-allocated data structures, zero-copy with `bytes`
*   **Logging:** `tracing` crate with structured logging (no `println!`)
*   **Feature Flags:** Optional `dynamodb` feature for AWS DynamoDB trade recording
