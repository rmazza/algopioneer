//! Prometheus Metrics Module
//!
//! Pre-registered metrics for production observability.
//! All metrics use lock-free atomics for minimal hot-path overhead.

use lazy_static::lazy_static;
use prometheus::{
    opts, register_gauge_vec, register_histogram_vec, register_int_counter_vec, Encoder, GaugeVec,
    HistogramVec, IntCounterVec, TextEncoder,
};

lazy_static! {
    // --- Order Metrics ---

    /// Total orders executed (by symbol, side, status)
    pub static ref ORDERS_TOTAL: IntCounterVec = register_int_counter_vec!(
        opts!("algopioneer_orders_total", "Total orders executed"),
        &["symbol", "side", "status"]
    ).expect("FATAL: Failed to register ORDERS_TOTAL metric - check for duplicate registration");

    /// Order execution latency in seconds
    pub static ref ORDER_LATENCY: HistogramVec = register_histogram_vec!(
        "algopioneer_order_latency_seconds",
        "Order execution latency",
        &["symbol"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    ).expect("FATAL: Failed to register ORDER_LATENCY metric - check for duplicate registration");

    // --- Strategy Metrics ---

    /// Current PnL per strategy (gauge for real-time value)
    pub static ref STRATEGY_PNL: GaugeVec = register_gauge_vec!(
        opts!("algopioneer_strategy_pnl", "Current strategy PnL"),
        &["strategy_id", "strategy_type"]
    ).expect("FATAL: Failed to register STRATEGY_PNL metric - check for duplicate registration");

    /// Strategy state transitions
    pub static ref STRATEGY_STATE: IntCounterVec = register_int_counter_vec!(
        opts!("algopioneer_strategy_state_transitions_total", "Strategy state transitions"),
        &["strategy_id", "from_state", "to_state"]
    ).expect("FATAL: Failed to register STRATEGY_STATE metric - check for duplicate registration");

    // --- WebSocket Metrics ---

    /// WebSocket ticks received
    pub static ref WS_TICKS_TOTAL: IntCounterVec = register_int_counter_vec!(
        opts!("algopioneer_websocket_ticks_total", "WebSocket ticks received"),
        &["symbol"]
    ).expect("FATAL: Failed to register WS_TICKS_TOTAL metric - check for duplicate registration");

    /// WebSocket ticks dropped (backpressure)
    pub static ref WS_TICKS_DROPPED: IntCounterVec = register_int_counter_vec!(
        opts!("algopioneer_websocket_ticks_dropped_total", "WebSocket ticks dropped due to backpressure"),
        &["symbol", "reason"]
    ).expect("FATAL: Failed to register WS_TICKS_DROPPED metric - check for duplicate registration");

    /// WebSocket reconnections
    pub static ref WS_RECONNECTIONS: IntCounterVec = register_int_counter_vec!(
        opts!("algopioneer_websocket_reconnections_total", "WebSocket reconnection attempts"),
        &["status"]
    ).expect("FATAL: Failed to register WS_RECONNECTIONS metric - check for duplicate registration");

    // --- Circuit Breaker Metrics ---

    /// Circuit breaker state (0=closed, 1=half_open, 2=open)
    pub static ref CIRCUIT_BREAKER_STATE: GaugeVec = register_gauge_vec!(
        opts!("algopioneer_circuit_breaker_state", "Circuit breaker state (0=closed, 1=half_open, 2=open)"),
        &["name"]
    ).expect("FATAL: Failed to register CIRCUIT_BREAKER_STATE metric - check for duplicate registration");

    /// Circuit breaker trips
    pub static ref CIRCUIT_BREAKER_TRIPS: IntCounterVec = register_int_counter_vec!(
        opts!("algopioneer_circuit_breaker_trips_total", "Circuit breaker trips"),
        &["name"]
    ).expect("FATAL: Failed to register CIRCUIT_BREAKER_TRIPS metric - check for duplicate registration");

    // --- Recovery Metrics ---

    /// Recovery attempts
    pub static ref RECOVERY_ATTEMPTS: IntCounterVec = register_int_counter_vec!(
        opts!("algopioneer_recovery_attempts_total", "Recovery attempts"),
        &["symbol", "status"]
    ).expect("FATAL: Failed to register RECOVERY_ATTEMPTS metric - check for duplicate registration");
}

/// Record an order execution
pub fn record_order(symbol: &str, side: &str, success: bool) {
    let status = if success { "success" } else { "failure" };
    ORDERS_TOTAL
        .with_label_values(&[symbol, side, status])
        .inc();
}

/// Record order latency
pub fn record_order_latency(symbol: &str, latency_secs: f64) {
    ORDER_LATENCY
        .with_label_values(&[symbol])
        .observe(latency_secs);
}

/// Update strategy PnL gauge
pub fn set_strategy_pnl(strategy_id: &str, strategy_type: &str, pnl: f64) {
    STRATEGY_PNL
        .with_label_values(&[strategy_id, strategy_type])
        .set(pnl);
}

/// Record WebSocket tick
pub fn record_ws_tick(symbol: &str) {
    WS_TICKS_TOTAL.with_label_values(&[symbol]).inc();
}

/// Record dropped tick
pub fn record_dropped_tick(symbol: &str, reason: &str) {
    WS_TICKS_DROPPED.with_label_values(&[symbol, reason]).inc();
}

/// Get metrics as text for /metrics endpoint
///
/// CB-3 FIX: Handles encoding errors gracefully instead of panicking
pub fn gather_metrics() -> String {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();

    // CB-3 FIX: Handle encoding errors gracefully
    if let Err(e) = encoder.encode(&metric_families, &mut buffer) {
        tracing::error!("Failed to encode Prometheus metrics: {}", e);
        return String::new();
    }

    // CB-3 FIX: Handle UTF-8 conversion errors gracefully
    match String::from_utf8(buffer) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("Prometheus metrics buffer is not valid UTF-8: {}", e);
            String::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_order() {
        record_order("BTC-USD", "buy", true);
        // Metric should be incremented (we can't easily assert on prometheus counters)
    }

    #[test]
    fn test_gather_metrics() {
        // Trigger lazy initialization of at least one metric
        record_order("TEST-SYM", "buy", true);

        let output = gather_metrics();
        // Now the metrics should be initialized and contain our prefix
        assert!(
            output.contains("algopioneer") || output.contains("orders_total"),
            "Expected metrics output to contain 'algopioneer' or 'orders_total', got: {}",
            &output[..output.len().min(200)]
        );
    }
}
