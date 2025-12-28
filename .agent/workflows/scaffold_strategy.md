---
description: Scaffold a new strategy implementation with boilerplate code
---

# Scaffold New Strategy

This workflow creates a new Rust source file in `src/strategy/` with the necessary imports and trait implementations to get started quickly.

## Steps

### 1. Determine Strategy Name
Choose a name for your strategy (e.g., `MeanReversion`, `RsiBreakout`).
Convert it to snake_case for the filename (e.g., `mean_reversion.rs`) and CamelCase for the struct (e.g., `MeanReversion`).

### 2. Creates File
Run the following command to create the file (replace `YOUR_STRATEGY_NAME` and `your_strategy_filename.rs`):

```bash
cat <<EOF > src/strategy/your_strategy_filename.rs
use crate::strategy::Signal;
use crate::strategy::{LiveStrategy, StrategyInput};
use crate::exchange::Executor;
use async_trait::async_trait;
use rust_decimal::Decimal;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, warn, debug};

/// Configuration and state for the strategy.
pub struct YourStrategyName<E: Executor> {
    id: String,
    symbol: String,
    executor: Arc<E>,
    // Add custom fields here
}

impl<E: Executor + 'static> YourStrategyName<E> {
    pub fn new(id: String, symbol: String, executor: Arc<E>) -> Self {
        Self {
            id,
            symbol,
            executor,
        }
    }
}

#[async_trait]
impl<E: Executor + 'static> LiveStrategy for YourStrategyName<E> {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn subscribed_symbols(&self) -> Vec<String> {
        vec![self.symbol.clone()]
    }

    fn strategy_type(&self) -> &'static str {
        "YourStrategyName"
    }

    fn current_pnl(&self) -> Decimal {
        Decimal::ZERO // Implement PnL tracking
    }

    fn is_healthy(&self) -> bool {
        true
    }

    async fn run(&mut self, mut data_rx: mpsc::Receiver<StrategyInput>) {
        info!(strategy_id = %self.id, "Strategy starting");

        while let Some(input) = data_rx.recv().await {
            match input {
                StrategyInput::Tick(tick) => {
                    // Implement logic here
                    debug!(price = %tick.price, "Received tick");
                }
                _ => {}
            }
        }
    }
}
EOF
```

### 3. Register Module
Add the new module to `src/strategy/mod.rs`.

```rust
// In src/strategy/mod.rs
pub mod your_strategy_filename;
```

### 4. Implement Logic
Open the new file and start implementing your logic!
