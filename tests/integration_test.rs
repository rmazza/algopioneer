use algopioneer::strategy::basis_trading::*;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio::time::Duration;
use chrono::{DateTime, Utc, TimeZone};
use mockall::predicate::*;
use mockall::mock;
use std::pin::Pin;
use std::future::Future;
use async_trait::async_trait;

// --- Mocks ---

// We use an adapter pattern to handle the lifetime complexity of mocking async_trait methods
// that need to return a custom pending future (to simulate a hang).
// 1. We define a mock struct with a method that returns a 'static Future.
// 2. We implement the Executor trait for this mock struct, delegating to the mock method.

mock! {
    pub ExecutorImpl {
        fn execute_order_mock(&self, symbol: &str, side: &str, quantity: Decimal) -> Pin<Box<dyn Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>> + Send + 'static>>;
    }
}

#[async_trait]
impl Executor for MockExecutorImpl {
    async fn execute_order(&self, symbol: &str, side: &str, quantity: Decimal) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Delegate to the mock expectation
        self.execute_order_mock(symbol, side, quantity).await
    }
}

#[derive(Debug, Clone)]
struct MockClock {
    current_time: Arc<Mutex<i64>>,
}

impl MockClock {
    fn new(start_ts: i64) -> Self {
        Self {
            current_time: Arc::new(Mutex::new(start_ts)),
        }
    }

    fn advance_millis(&self, millis: i64) {
        let mut time = self.current_time.lock().unwrap();
        *time += millis;
    }
}

impl Clock for MockClock {
    fn now(&self) -> DateTime<Utc> {
        let ts = *self.current_time.lock().unwrap();
        Utc.timestamp_millis_opt(ts).unwrap()
    }
}

// --- Test ---

#[tokio::test]
async fn test_phoenix_recovery() {
    // 0. Setup Time Control
    tokio::time::pause(); 
    let start_ts = 1_600_000_000_000;
    let clock = MockClock::new(start_ts);
    
    // 1. Setup Dependencies
    let mut mock_executor = MockExecutorImpl::new();
    
    // Simulate a "Hang" on execution
    // We return a Future that sleeps for 1 hour.
    // Since we are using tokio::time::sleep and time is paused, this future will remain pending
    // until we advance time past 1 hour (which we won't).
    mock_executor
        .expect_execute_order_mock()
        .with(eq("BTC-USD"), eq("buy"), always())
        .times(1)
        .returning(|_, _, _| {
            Box::pin(async {
                tokio::time::sleep(Duration::from_secs(3600)).await; // Hang
                Ok(())
            })
        });

    mock_executor
        .expect_execute_order_mock()
        .with(eq("BTC-USDT"), eq("sell"), always())
        .times(1)
        .returning(|_, _, _| {
             Box::pin(async {
                tokio::time::sleep(Duration::from_secs(3600)).await; // Hang
                Ok(())
            })
        });

    let mock_executor = Arc::new(mock_executor);
    
    let (recovery_tx, recovery_rx) = mpsc::channel(10);
    
    // Cost Model: High fees to ensure we test logic correctly
    let cost_model = TransactionCostModel::new(dec!(5.0), dec!(10.0), dec!(1.0)); 
    let entry_manager = Box::new(EntryManager::new(dec!(10.0), dec!(5.0), cost_model));
    let risk_monitor = RiskMonitor::new(dec!(1.0), InstrumentType::Linear);
    
    let (engine_recovery_tx, _engine_recovery_rx) = mpsc::channel(10);
    let execution_engine = ExecutionEngine::new(mock_executor.clone(), engine_recovery_tx);
    
    let mut strategy = BasisTradingStrategy::new(
        entry_manager,
        risk_monitor,
        execution_engine,
        "BTC-USD".to_string(),
        "BTC-USDT".to_string(),
        recovery_rx,
        Box::new(clock.clone()),
    );
    
    // Setup State Observer
    let (state_tx, mut state_rx) = mpsc::channel(10);
    strategy.set_observer(state_tx);

    // Channels for Market Data
    let (spot_tx, spot_rx) = mpsc::channel(10);
    let (future_tx, future_rx) = mpsc::channel(10);
    
    // Spawn Strategy
    tokio::spawn(async move {
        strategy.run(spot_rx, future_rx).await;
    });

    // Step 1: Send MarketData tick that triggers Buy
    let spot_tick = MarketData {
        symbol: "BTC-USD".to_string(),
        price: dec!(50000),
        timestamp: start_ts,
    };
    let future_tick = MarketData {
        symbol: "BTC-USDT".to_string(),
        price: dec!(51000),
        timestamp: start_ts,
    };

    spot_tx.send(spot_tick).await.unwrap();
    future_tx.send(future_tick).await.unwrap();

    // Step 2: Assert Strategy enters Entering state
    let state = state_rx.recv().await.expect("Failed to receive state");
    assert_eq!(state, StrategyState::Entering, "Step 2: Should enter Entering state");

    // Step 3: Simulate "Hang"
    // Advance time by 31 seconds.
    clock.advance_millis(31000);
    tokio::time::advance(Duration::from_millis(31000)).await;

    // Step 4: Trigger Strategy heartbeat -> Assert Reconciling
    tokio::task::yield_now().await;
    
    let state = state_rx.recv().await.expect("Failed to receive state");
    assert_eq!(state, StrategyState::Reconciling, "Step 4: Should transition to Reconciling on timeout");

    // Step 5: Manually inject RecoveryResult::Success
    recovery_tx.send(RecoveryResult::Success("BTC-USD".to_string())).await.unwrap();

    // Step 6: Process message -> Assert Flat
    let state = state_rx.recv().await.expect("Failed to receive state");
    assert_eq!(state, StrategyState::Flat, "Step 6: Should transition to Flat after recovery");
}
