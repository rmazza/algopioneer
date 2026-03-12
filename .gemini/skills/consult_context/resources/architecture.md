# System Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           CLI (main.rs)                              │
│  trade | dual-leg | portfolio | backtest | discover-pairs           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │ Strategy  │   │ Discovery │   │ Backtest  │
            │ Supervisor│   │  Engine   │   │  Engine   │
            └───────────┘   └───────────┘   └───────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│  Tick Router  │       │   Strategies  │
│ (lock-free)   │       │  (per-pair)   │
└───────────────┘       └───────────────┘
        │                       │
        ▼                       ▼
┌───────────────────────────────────────┐
│         Exchange Abstraction          │
│  Executor | ExchangeClient | WebSocket│
└───────────────────────────────────────┘
        │               │               │
        ▼               ▼               ▼
┌───────────┐   ┌───────────┐   ┌───────────┐
│ Coinbase  │   │  Alpaca   │   │  Kraken   │
└───────────┘   └───────────┘   └───────────┘
```

## Key Components

### Strategy Layer
- **Supervisor**: Manages strategy lifecycle, panic recovery, PnL aggregation
- **Tick Router**: Distributes market data to subscribed strategies
- **DualLegTrading**: Pairs/basis trading state machine
- **MovingAverage**: Simple MA crossover strategy

### Exchange Layer
- **Executor trait**: Order execution abstraction
- **ExchangeClient trait**: Account/position queries
- **WebSocketProvider trait**: Real-time data streams

### Resilience
- **Circuit Breaker**: Prevents cascading failures
- **Position Reconciliation**: Recovery after restarts
- **Graceful Shutdown**: Clean task termination

### Observability
- **Prometheus Metrics**: `/metrics` endpoint
- **Health Checks**: `/health` endpoint
- **Tracing**: OpenTelemetry integration

## Data Flow

1. **Market Data**: Exchange → WebSocket → Tick Router → Strategies
2. **Signals**: Strategy → Executor → Exchange
3. **Metrics**: All components → Prometheus → `/metrics`
