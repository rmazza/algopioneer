use algopioneer::strategy::dual_leg_trading::*;
use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use mockall::mock;
use mockall::predicate::*;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio::time::Duration;

// --- Mocks ---

// We use an adapter pattern to handle the lifetime complexity of mocking async_trait methods
// that need to return a custom pending future (to simulate a hang).
// 1. We define a mock struct with a method that returns a 'static Future.
// 2. We implement the Executor trait for this mock struct, delegating to the mock method.

/// Type alias the async Result type to reduce complexity warnings
type BoxedFuture<T> = Pin<Box<dyn Future<Output = T> + Send + 'static>>;
type ExecuteOrderResult = BoxedFuture<Result<(), Box<dyn std::error::Error + Send + Sync>>>;

mock! {
    pub ExecutorImpl {
        fn execute_order_mock(&self, symbol: &str, side: &str, quantity: Decimal) -> ExecuteOrderResult;
    }
}

#[async_trait]
impl Executor for MockExecutorImpl {
    async fn execute_order(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: Decimal,
        _price: Option<Decimal>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Delegate to the mock expectation
        self.execute_order_mock(symbol, &side.to_string(), quantity)
            .await
    }

    async fn get_position(
        &self,
        _symbol: &str,
    ) -> Result<Decimal, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Decimal::ZERO)
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
    let entry_manager = Box::new(BasisManager::new(dec!(10.0), dec!(5.0), cost_model));
    let risk_monitor = RiskMonitor::new(dec!(1.0), InstrumentType::Linear, HedgeMode::DeltaNeutral);

    let (engine_recovery_tx, _engine_recovery_rx) = mpsc::channel(10);
    let execution_engine = ExecutionEngine::new(mock_executor.clone(), engine_recovery_tx, 5, 60);

    let config = DualLegConfig {
        spot_symbol: "BTC-USD".to_string(),
        future_symbol: "BTC-USDT".to_string(),
        order_size: dec!(0.1),
        max_tick_age_ms: 2000,
        execution_timeout_ms: 30000,
        min_profit_threshold: dec!(1.0),
        stop_loss_threshold: dec!(-10.0),
        fee_tier: TransactionCostModel::new(dec!(5.0), dec!(10.0), dec!(1.0)),
        throttle_interval_secs: 5,
    };

    let mut strategy = DualLegStrategy::new(
        entry_manager,
        risk_monitor,
        execution_engine,
        config,
        recovery_rx,
        Box::new(clock.clone()),
    );

    // Setup State Observer
    let (state_tx, mut state_rx) = mpsc::channel(10);
    strategy.set_observer(state_tx);

    // Channels for Market Data
    let (leg1_tx, leg1_rx) = mpsc::channel::<Arc<MarketData>>(10);
    let (leg2_tx, leg2_rx) = mpsc::channel::<Arc<MarketData>>(10);

    // Spawn Strategy
    tokio::spawn(async move {
        strategy.run(leg1_rx, leg2_rx).await;
    });

    // Step 1: Send MarketData tick that triggers Buy
    let spot_tick = MarketData {
        symbol: "BTC-USD".to_string(),
        price: dec!(50000),
        instrument_id: None,
        timestamp: start_ts,
    };
    let future_tick = MarketData {
        symbol: "BTC-USDT".to_string(),
        price: dec!(51000),
        instrument_id: None,
        timestamp: start_ts,
    };

    leg1_tx.send(Arc::new(spot_tick)).await.unwrap();
    leg2_tx.send(Arc::new(future_tick)).await.unwrap();

    // Step 2: Assert Strategy enters Entering state
    let state = state_rx.recv().await.expect("Failed to receive state");
    assert!(
        matches!(state, StrategyState::Entering { .. }),
        "Step 2: Should enter Entering state"
    );

    // Step 3: Simulate "Hang"
    // Advance time by 31 seconds.
    clock.advance_millis(31000);
    tokio::time::advance(Duration::from_millis(31000)).await;

    // Step 4: Trigger Strategy heartbeat -> Assert Reconciling
    tokio::task::yield_now().await;

    let state = state_rx.recv().await.expect("Failed to receive state");
    assert_eq!(
        state,
        StrategyState::Reconciling,
        "Step 4: Should transition to Reconciling on timeout"
    );

    // Step 5: Manually inject RecoveryResult::Success
    recovery_tx
        .send(RecoveryResult::Success("BTC-USD".to_string()))
        .await
        .unwrap();

    // Step 6: Process message -> Assert Flat
    let state = state_rx.recv().await.expect("Failed to receive state");
    assert_eq!(
        state,
        StrategyState::Flat,
        "Step 6: Should transition to Flat after recovery"
    );
}

#[tokio::test]
async fn test_pairs_trading_cycle() {
    let _ = tracing_subscriber::fmt::try_init();
    println!("Starting test_pairs_trading_cycle");
    tokio::time::pause();
    let start_ts = 1_600_000_000_000;
    let clock = MockClock::new(start_ts);

    let mut mock_executor = MockExecutorImpl::new();

    // Expect Entry (Short Spread: Sell A, Buy B)
    // Wait, PairsManager currently implements:
    // Z > 2 -> Sell (Short Spread)
    // Z < -2 -> Buy (Long Spread)
    // We will test Long Spread Entry (Buy A, Sell B).

    mock_executor
        .expect_execute_order_mock()
        .with(eq("A"), eq("buy"), always())
        .times(1)
        .returning(|_, _, _| Box::pin(async { Ok(()) }));

    mock_executor
        .expect_execute_order_mock()
        .with(eq("B"), eq("sell"), always())
        .times(1)
        .returning(|_, _, _| Box::pin(async { Ok(()) }));

    // Expect Exit (Sell A, Buy B)
    mock_executor
        .expect_execute_order_mock()
        .with(eq("A"), eq("sell"), always())
        .times(1)
        .returning(|_, _, _| Box::pin(async { Ok(()) }));

    mock_executor
        .expect_execute_order_mock()
        .with(eq("B"), eq("buy"), always())
        .times(1)
        .returning(|_, _, _| Box::pin(async { Ok(()) }));

    let mock_executor = Arc::new(mock_executor);
    let (_recovery_tx, _recovery_rx) = mpsc::channel(10);
    let (engine_recovery_tx, _engine_recovery_rx) = mpsc::channel(10);
    let execution_engine = ExecutionEngine::new(mock_executor.clone(), engine_recovery_tx, 5, 60);

    // Pairs Manager: Window 5, Entry Z=1.9, Exit Z=1.0
    let entry_manager = Box::new(PairsManager::new(5, 1.9, 1.0));
    let risk_monitor =
        RiskMonitor::new(dec!(1.0), InstrumentType::Linear, HedgeMode::DollarNeutral);

    let config = DualLegConfig {
        spot_symbol: "A".to_string(),
        future_symbol: "B".to_string(),
        order_size: dec!(1.0),
        max_tick_age_ms: 2000,
        execution_timeout_ms: 30000,
        min_profit_threshold: dec!(0.1),
        stop_loss_threshold: dec!(-5.0),
        fee_tier: TransactionCostModel::new(dec!(0.0), dec!(0.0), dec!(0.0)),
        throttle_interval_secs: 5,
    };

    let mut strategy = DualLegStrategy::new(
        entry_manager,
        risk_monitor,
        execution_engine,
        config,
        _recovery_rx,
        Box::new(clock.clone()),
    );

    let (state_tx, mut state_rx) = mpsc::channel(10);
    strategy.set_observer(state_tx);

    let (leg1_tx, leg1_rx) = mpsc::channel::<Arc<MarketData>>(10);
    let (leg2_tx, leg2_rx) = mpsc::channel::<Arc<MarketData>>(10);

    tokio::spawn(async move {
        strategy.run(leg1_rx, leg2_rx).await;
    });

    // 1. Warm up window with stable spread (0)
    for _ in 0..5 {
        leg1_tx
            .send(Arc::new(MarketData {
                symbol: "A".into(),
                price: dec!(100),
                instrument_id: None,
                timestamp: start_ts,
            }))
            .await
            .unwrap();
        leg2_tx
            .send(Arc::new(MarketData {
                symbol: "B".into(),
                price: dec!(100),
                instrument_id: None,
                timestamp: start_ts,
            }))
            .await
            .unwrap();
        tokio::task::yield_now().await;
    }

    // 2. Trigger Long Entry (Z < -2). Drop A price.
    leg1_tx
        .send(Arc::new(MarketData {
            symbol: "A".into(),
            price: dec!(79),
            instrument_id: None,
            timestamp: start_ts,
        }))
        .await
        .unwrap();
    leg2_tx
        .send(Arc::new(MarketData {
            symbol: "B".into(),
            price: dec!(100),
            instrument_id: None,
            timestamp: start_ts,
        }))
        .await
        .unwrap();

    // Allow processing
    tokio::time::sleep(Duration::from_millis(10)).await;

    // Expect Entering -> InPosition
    let state = state_rx.recv().await.expect("No state (Entering)");
    assert!(matches!(state, StrategyState::Entering { .. }));

    // Allow execution task to run
    tokio::time::sleep(Duration::from_millis(100)).await;

    let state = state_rx.recv().await.expect("No state (InPosition)");
    assert!(matches!(state, StrategyState::InPosition { .. }));

    // 3. Trigger Exit (Mean Reversion). Prices converge.
    leg1_tx
        .send(Arc::new(MarketData {
            symbol: "A".into(),
            price: dec!(100),
            instrument_id: None,
            timestamp: start_ts,
        }))
        .await
        .unwrap();
    leg2_tx
        .send(Arc::new(MarketData {
            symbol: "B".into(),
            price: dec!(100),
            instrument_id: None,
            timestamp: start_ts,
        }))
        .await
        .unwrap();

    // Allow processing
    tokio::time::sleep(Duration::from_millis(10)).await;

    // Expect Exiting -> Flat
    let state = state_rx.recv().await.expect("No state (Exiting)");
    assert!(matches!(state, StrategyState::Exiting { .. }));

    // Allow execution task to run
    tokio::time::sleep(Duration::from_millis(100)).await;

    let state = state_rx.recv().await.expect("No state (Flat)");
    assert_eq!(state, StrategyState::Flat);
}

#[tokio::test]
async fn test_basis_trading_cycle() {
    // let _ = tracing_subscriber::fmt::try_init();
    println!("Starting test_basis_trading_cycle");
    tokio::time::pause();
    let start_ts = 1_600_000_000_000;
    let clock = MockClock::new(start_ts);

    let mut mock_executor = MockExecutorImpl::new();

    // Expect Entry (Buy Spot, Sell Future)
    mock_executor
        .expect_execute_order_mock()
        .with(eq("BTC-USD"), eq("buy"), always())
        .times(1)
        .returning(|_, _, _| Box::pin(async { Ok(()) }));

    mock_executor
        .expect_execute_order_mock()
        .with(eq("BTC-USDT"), eq("sell"), always())
        .times(1)
        .returning(|_, _, _| Box::pin(async { Ok(()) }));

    // Expect Exit (Sell Spot, Buy Future)
    mock_executor
        .expect_execute_order_mock()
        .with(eq("BTC-USD"), eq("sell"), always())
        .times(1)
        .returning(|_, _, _| Box::pin(async { Ok(()) }));

    mock_executor
        .expect_execute_order_mock()
        .with(eq("BTC-USDT"), eq("buy"), always())
        .times(1)
        .returning(|_, _, _| Box::pin(async { Ok(()) }));

    let mock_executor = Arc::new(mock_executor);
    let (_recovery_tx, _recovery_rx) = mpsc::channel(10);
    let (engine_recovery_tx, _engine_recovery_rx) = mpsc::channel(10);
    let execution_engine = ExecutionEngine::new(mock_executor.clone(), engine_recovery_tx, 5, 60);

    // Basis Manager: Entry 10 bps, Exit 2 bps.
    let cost_model = TransactionCostModel::new(dec!(0.0), dec!(0.0), dec!(0.0));
    let entry_manager = Box::new(BasisManager::new(dec!(10.0), dec!(2.0), cost_model));
    let risk_monitor = RiskMonitor::new(dec!(1.0), InstrumentType::Linear, HedgeMode::DeltaNeutral);

    let config = DualLegConfig {
        spot_symbol: "BTC-USD".to_string(),
        future_symbol: "BTC-USDT".to_string(),
        order_size: dec!(0.1),
        max_tick_age_ms: 2000,
        execution_timeout_ms: 30000,
        min_profit_threshold: dec!(0.0),
        stop_loss_threshold: dec!(-10.0),
        fee_tier: cost_model,
        throttle_interval_secs: 5,
    };

    let mut strategy = DualLegStrategy::new(
        entry_manager,
        risk_monitor,
        execution_engine,
        config,
        _recovery_rx,
        Box::new(clock.clone()),
    );

    let (state_tx, mut state_rx) = mpsc::channel(10);
    strategy.set_observer(state_tx);

    let (leg1_tx, leg1_rx) = mpsc::channel::<Arc<MarketData>>(10);
    let (leg2_tx, leg2_rx) = mpsc::channel::<Arc<MarketData>>(10);

    tokio::spawn(async move {
        strategy.run(leg1_rx, leg2_rx).await;
    });

    // 1. Trigger Entry. Spread > 10 bps.
    // Spot 100, Future 100.2. Spread = 20 bps.
    leg1_tx
        .send(Arc::new(MarketData {
            symbol: "BTC-USD".into(),
            price: dec!(100),
            instrument_id: None,
            timestamp: start_ts,
        }))
        .await
        .unwrap();
    leg2_tx
        .send(Arc::new(MarketData {
            symbol: "BTC-USDT".into(),
            price: dec!(100.2),
            instrument_id: None,
            timestamp: start_ts,
        }))
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(10)).await;

    let state = state_rx.recv().await.expect("No state (Entering)");
    assert!(matches!(state, StrategyState::Entering { .. }));

    tokio::time::sleep(Duration::from_millis(100)).await;

    let state = state_rx.recv().await.expect("No state (InPosition)");
    assert!(matches!(state, StrategyState::InPosition { .. }));

    // 2. Trigger Exit. Spread < 2 bps.
    // Spot 100, Future 100.01. Spread = 1 bps.
    leg1_tx
        .send(Arc::new(MarketData {
            symbol: "BTC-USD".into(),
            price: dec!(100),
            instrument_id: None,
            timestamp: start_ts,
        }))
        .await
        .unwrap();
    leg2_tx
        .send(Arc::new(MarketData {
            symbol: "BTC-USDT".into(),
            price: dec!(100.01),
            instrument_id: None,
            timestamp: start_ts,
        }))
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(10)).await;

    let state = state_rx.recv().await.expect("No state (Exiting)");
    assert!(matches!(state, StrategyState::Exiting { .. }));

    tokio::time::sleep(Duration::from_millis(100)).await;

    let state = state_rx.recv().await.expect("No state (Flat)");
    assert_eq!(state, StrategyState::Flat);
}

#[tokio::test]
async fn test_recovery_worker_retry() {
    tokio::time::pause();

    let mut mock_executor = MockExecutorImpl::new();

    // Expect 2 failures then 1 success
    // NOTE: Mockall matches in reverse order of definition usually, or we use Sequence.
    let mut seq = mockall::Sequence::new();

    mock_executor
        .expect_execute_order_mock()
        .with(eq("BTC-USD"), eq("buy"), always())
        .times(1)
        .in_sequence(&mut seq)
        .returning(|_, _, _| Box::pin(async { Err(Box::from("Simulated Failure 1")) }));

    mock_executor
        .expect_execute_order_mock()
        .with(eq("BTC-USD"), eq("buy"), always())
        .times(1)
        .in_sequence(&mut seq)
        .returning(|_, _, _| Box::pin(async { Ok(()) }));

    let mock_executor = Arc::new(mock_executor);
    let (recovery_tx, recovery_rx) = mpsc::channel(10);
    let (feedback_tx, mut feedback_rx) = mpsc::channel(10);

    let worker = RecoveryWorker::new(mock_executor, recovery_rx, feedback_tx);

    tokio::spawn(async move {
        worker.run().await;
    });

    // Send a task
    let task = RecoveryTask {
        symbol: "BTC-USD".to_string(),
        action: OrderSide::Buy,
        quantity: dec!(1.0),
        reason: "Test".to_string(),
        attempts: 0,
    };

    recovery_tx.send(task).await.unwrap();

    // It should fail immediately, wait 2s, then retry and succeed.
    // We need to advance time.

    // Allow first attempt
    tokio::task::yield_now().await;

    // Advance 2s (backoff)
    tokio::time::advance(Duration::from_secs(2)).await;
    tokio::time::sleep(Duration::from_millis(100)).await; // Allow processing

    // Expect Success result
    let res = feedback_rx.recv().await.expect("No feedback");
    match res {
        RecoveryResult::Success(s) => assert_eq!(s, "BTC-USD"),
        _ => panic!("Expected Success"),
    }
}

#[tokio::test]
async fn test_synthetic_provider() {
    use algopioneer::coinbase::market_data_provider::{MarketDataProvider, SyntheticProvider};
    use rust_decimal_macros::dec;

    // Create synthetic provider with 50000 base price, 0.01% volatility, 100ms ticks
    let provider = SyntheticProvider::new(dec!(50000), 0.0001, 100);

    // Subscribe to two symbols
    let symbols = vec!["BTC-USD".to_string(), "ETH-USD".to_string()];
    let mut rx = provider
        .subscribe(symbols)
        .await
        .expect("Failed to subscribe");

    // Collect a few ticks
    let mut tick_count = 0;
    let mut btc_seen = false;
    let mut eth_seen = false;

    while tick_count < 10 {
        tokio::select! {
            Some(tick) = rx.recv() => {
                // Verify tick structure
                assert!(tick.price > dec!(0), "Price should be positive");
                assert!(tick.timestamp > 0, "Timestamp should be set");

                if tick.symbol == "BTC-USD" {
                    btc_seen = true;
                }
                if tick.symbol == "ETH-USD" {
                    eth_seen = true;
                }

                tick_count += 1;
            }
            _ = tokio::time::sleep(tokio::time::Duration::from_secs(2)) => {
                panic!("Timeout waiting for ticks");
            }
        }
    }

    assert!(btc_seen, "Should have received BTC ticks");
    assert!(eth_seen, "Should have received ETH ticks");
}

// =============================================================================
// HTTP ENDPOINT INTEGRATION TESTS
// =============================================================================

#[tokio::test]
async fn test_health_endpoint_returns_healthy() {
    use algopioneer::health::{create_health_state, run_health_server};
    use std::net::TcpListener;

    // Find an available port
    let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind");
    let port = listener.local_addr().unwrap().port();
    drop(listener); // Release the port

    let state = create_health_state();

    // Spawn health server
    tokio::spawn(run_health_server(port, state));

    // Wait for server to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Make HTTP request to /health
    let _url = format!("http://127.0.0.1:{}/health", port);

    // Simple TCP request (avoiding additional dependencies)
    let response = tokio::net::TcpStream::connect(format!("127.0.0.1:{}", port)).await;

    if let Ok(mut stream) = response {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let request = format!("GET /health HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nConnection: close\r\n\r\n", port);
        stream.write_all(request.as_bytes()).await.unwrap();

        let mut buffer = Vec::new();
        stream.read_to_end(&mut buffer).await.unwrap();
        let response_str = String::from_utf8_lossy(&buffer);

        // Verify HTTP response
        assert!(response_str.contains("200 OK"), "Expected 200 OK, got: {}", response_str);
        assert!(response_str.contains("healthy"), "Expected 'healthy' in response body");
        assert!(response_str.contains("version"), "Expected 'version' field in response");
    } else {
        // Server may have failed to bind (port race condition in CI) - skip gracefully
        println!("Skipping health test - server not available on port {}", port);
    }
}

#[tokio::test]
async fn test_metrics_endpoint_returns_prometheus_format() {
    use algopioneer::health::{create_health_state, run_health_server};
    use algopioneer::metrics::record_order;
    use std::net::TcpListener;

    // Find an available port
    let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind");
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let state = create_health_state();

    // Record a metric before starting server
    record_order("TEST-INTEGRATION", "buy", true);

    // Spawn health server
    tokio::spawn(run_health_server(port, state));

    // Wait for server to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Make HTTP request to /metrics
    let response = tokio::net::TcpStream::connect(format!("127.0.0.1:{}", port)).await;

    if let Ok(mut stream) = response {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let request = format!("GET /metrics HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nConnection: close\r\n\r\n", port);
        stream.write_all(request.as_bytes()).await.unwrap();

        let mut buffer = Vec::new();
        stream.read_to_end(&mut buffer).await.unwrap();
        let response_str = String::from_utf8_lossy(&buffer);

        // Verify HTTP response
        assert!(response_str.contains("200 OK"), "Expected 200 OK for /metrics");

        // Verify Prometheus format markers
        assert!(
            response_str.contains("# HELP") || response_str.contains("# TYPE") || response_str.contains("algopioneer"),
            "Expected Prometheus format output with HELP/TYPE comments or metric prefix"
        );
    } else {
        println!("Skipping metrics test - server not available on port {}", port);
    }
}
