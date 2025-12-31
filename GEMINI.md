# Project Overview

This is a production-ready, enterprise-grade Rust-based algorithmic trading system named `algopioneer`. It is designed to interact with multiple cryptocurrency exchanges (Coinbase, Kraken) for trading research and execution, supporting both live and paper trading modes with comprehensive risk management and observability.

**Quality Score:** 9.9+/10 (Production-Ready)

**Key Technologies:**

*   **Language:** Rust (Edition 2021)
*   **Core Libraries:**
    *   `cbadv`: Coinbase Advanced Trade API client
    *   `polars`: High-performance data manipulation and analysis
    *   `ta`: Technical analysis indicators
    *   `tokio`: Asynchronous runtime for concurrent operations
    *   `axum`: HTTP server for health checks and metrics
    *   `opentelemetry`: Distributed tracing and observability
    *   `prometheus`: Metrics collection and exposure
    *   `dashmap`: Lock-free concurrent hash maps
    *   `rust_decimal`: High-precision financial calculations
    *   `tokio-tungstenite`: WebSocket connections
    *   `tikv-jemallocator`: Optimized memory allocation for long-running processes
    *   `governor`: Rate limiting for API calls
    *   `thiserror`: Typed error handling
    *   `proptest`: Property-based testing (dev dependency)

**Architecture:**

Production-grade modular application with dependency injection, trait-based abstractions, and comprehensive error handling.

*   **Entry Point (`src/main.rs`):** CLI with subcommands for trading, backtesting, portfolio management, and pair discovery
*   **Strategies (`src/strategy/`):**
    *   `dual_leg_trading.rs`: Main dual-leg arbitrage strategy with state machine, recovery system, and circuit breaker
    *   `moving_average.rs`: Moving Average Crossover strategy
    *   `supervisor.rs`: Generic strategy supervisor with panic recovery, PnL aggregation, and health monitoring
    *   `tick_router.rs`: Lock-free market data distribution with backpressure handling
*   **Coinbase Integration (`src/coinbase/`):**
    *   `mod.rs`: API client with position querying
    *   `websocket.rs`: Real-time market data streaming
    *   `market_data_provider.rs`: Abstracted data sources (live + synthetic)
*   **Exchange Abstraction (`src/exchange/`):**
    *   `mod.rs`: Exchange-agnostic traits (`Executor`, `ExchangeClient`, `WebSocketProvider`) and factory functions
    *   `coinbase/`: Coinbase-specific implementation
    *   `kraken/`: Kraken-specific implementation
*   **Discovery (`src/discovery/`):**
    *   `mod.rs`: Pair discovery and optimization module
    *   `config.rs`: Discovery configuration
    *   `filter.rs`: Candidate pair filtering
    *   `optimizer.rs`: Grid search parameter optimization
    *   `error.rs`: Typed discovery errors
*   **Resilience (`src/resilience/`):**
    *   `circuit_breaker.rs`: Circuit breaker pattern with RwLock-based implementation
*   **Observability (`src/observability.rs`):** OpenTelemetry integration for distributed tracing
*   **Metrics (`src/metrics.rs`):** Prometheus metrics with lock-free atomics
*   **Health (`src/health.rs`):** HTTP health check endpoint for Kubernetes/Docker
*   **Types (`src/types.rs`):** Shared domain types (MarketData, OrderSide, etc.)

**Production Features:**

*   **Live Trading:** Execute trades on Coinbase/Kraken with full error handling
*   **Paper Trading:**
    *   **Coinbase:** Simulate trades without real funds (Internal Simulation)
    *   **Alpaca:** Uses real **Alpaca Paper API** for full end-to-end verification
*   **Backtesting:** Evaluate strategy performance using historical data
*   **Basis Trading:** Spot vs Future arbitrage with dollar-neutral hedging
*   **Pairs Trading:** Cointegration-based statistical arbitrage
*   **Pair Discovery:** Automatic discovery and optimization of cointegrated pairs
*   **Position Reconciliation:** Automatic recovery from network failures
*   **Circuit Breaker:** Cascading failure prevention with auto-recovery
*   **Strategy Supervisor:** Panic recovery and automatic restart with configurable policies
*   **PnL Aggregation:** Portfolio-level risk monitoring via DashMap
*   **Health Monitoring:** `/health` endpoint for orchestration
*   **Prometheus Metrics:** `/metrics` endpoint with order latency, PnL, and circuit breaker state
*   **Distributed Tracing:** OpenTelemetry/Jaeger integration for production observability
*   **Rate Limiting:** `governor` for API rate limit compliance
*   **Memory Optimization:** jemalloc allocator for reduced fragmentation

# Building and Running

**Prerequisites:**

*   Rust and Cargo installed.
*   A `.env` file in the root directory with the following variables:
    *   `COINBASE_API_KEY`
    *   `COINBASE_API_SECRET`
    *   `KRAKEN_API_KEY` (optional, for Kraken support)
    *   `KRAKEN_API_SECRET` (optional, for Kraken support)

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

*   **Dual-Leg Trading (Basis/Pairs):**
    ```bash
    cargo run --release -- dual-leg --strategy pairs --symbols BTC-USD,ETH-USD
    ```
    *   Use `--paper` for paper trading mode.

*   **Portfolio Mode:**
    ```bash
    cargo run --release -- portfolio --config pairs_config.json
    ```
    *   Use `--paper` for paper trading mode.

*   **Discover Pairs:**
    ```bash
    cargo run --release -- discover-pairs --symbols default --min-correlation 0.8
    ```

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
*   **Resilience:** Circuit breakers, panic recovery, position reconciliation, rate limiting
*   **Performance:** RwLock for concurrent reads, DashMap for lock-free maps, jemalloc allocator, pre-allocated data structures
*   **Logging:** `tracing` crate with structured logging (no `println!`)
