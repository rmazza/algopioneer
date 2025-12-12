#[tokio::test]
    async fn test_strategy_execution_sizing() {
        // Setup
        let executor = Arc::new(MockExecutor::new());
        let (recovery_tx, _) = mpsc::channel(100);
        let (feedback_tx, feedback_rx) = mpsc::channel(100); // Unused
        
        let engine = ExecutionEngine::new(
            executor.clone(),
            recovery_tx,
            1,
            5,
        );

        let config = DualLegConfigBuilder::new()
            .spot_symbol("BTC-USD")
            .future_symbol("BTC-USDT")
            .order_size(dec!(1000.0)) // $1000 allocation
            .build()
            .unwrap();

        let pair = InstrumentPair {
            spot_symbol: "BTC-USD".to_string(),
            future_symbol: "BTC-USDT".to_string(),
        };

        // Create a dummy manager that always buys
        struct AlwaysBuy;
        #[async_trait]
        impl EntryStrategy for AlwaysBuy {
            async fn analyze(&self, _l1: &MarketData, _l2: &MarketData) -> Signal {
                Signal::Buy
            }
        }

        let risk_monitor = RiskMonitor::new(dec!(1.0), InstrumentType::Linear, HedgeMode::DollarNeutral);
        let clock = Box::new(SystemClock);

        let mut strategy = DualLegStrategy::new(
            Box::new(AlwaysBuy),
            risk_monitor,
            engine,
            config,
            feedback_rx,
            clock,
        );

        // Send Ticks to trigger buy
        let leg1 = MarketData {
            symbol: "BTC-USD".into(),
            price: dec!(50.0), // $50 price
            timestamp: Utc::now().timestamp_millis(),
            instrument_id: None,
        };
        let leg2 = MarketData {
            symbol: "BTC-USDT".into(),
            price: dec!(51.0),
            timestamp: Utc::now().timestamp_millis(),
            instrument_id: None,
        };

        // Manually trigger process_tick (since we can't easily access private method, we must run it via public API or ensure it is accessible)
        // Check visibility of process_tick. It is private. 
        // We must stick to public API or make it pub(crate).
        // `run` is public but loops.
        // `DualLegStrategy` structure:
        // pub fn new(...)
        // pub async fn run(...)
        
        // Since `process_tick` is private, we have to spawn `run` and feed channels.
    }
