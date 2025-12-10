//! AS10: Health check HTTP endpoint for monitoring

use crate::metrics;
use crate::resilience::{CircuitBreaker, CircuitState};
use axum::{routing::get, Json, Router};
use chrono::Utc; // Added for `chrono::Utc::now()`
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HealthResponse {
    pub status: String, // "healthy", "degraded", "critical"
    pub version: String,
    // AS6: Enhanced health metrics
    pub circuit_breaker_state: Option<String>, // "closed", "open", "half_open"
    pub active_positions: usize,
    pub recovery_queue_depth: usize,
    pub uptime_seconds: u64,
    pub timestamp: i64,
}

impl Default for HealthResponse {
    fn default() -> Self {
        Self {
            status: "healthy".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            circuit_breaker_state: Some("closed".to_string()),
            active_positions: 0,
            recovery_queue_depth: 0,
            uptime_seconds: 0,
            timestamp: Utc::now().timestamp(),
        }
    }
}

// AS6: Shared health state for dynamic updates
pub type HealthState = Arc<RwLock<HealthResponse>>;

pub fn create_health_state() -> HealthState {
    Arc::new(RwLock::new(HealthResponse::default()))
}

/// AS2: Update health state from CircuitBreaker for operational visibility
pub async fn update_from_circuit_breaker(state: &HealthState, breaker: &CircuitBreaker) {
    let mut health = state.write().await;
    let cb_state = breaker.get_state().await;
    health.circuit_breaker_state = Some(format_circuit_state(cb_state));
    health.timestamp = Utc::now().timestamp();

    // Update overall status based on circuit breaker
    health.status = match cb_state {
        CircuitState::Closed => "healthy".to_string(),
        CircuitState::HalfOpen => "degraded".to_string(),
        CircuitState::Open => "critical".to_string(),
    };
}

/// Format CircuitState as a human-readable string
fn format_circuit_state(state: CircuitState) -> String {
    match state {
        CircuitState::Closed => "closed".to_string(),
        CircuitState::Open => "open".to_string(),
        CircuitState::HalfOpen => "half_open".to_string(),
    }
}

async fn health_check(
    axum::extract::State(state): axum::extract::State<HealthState>,
) -> Json<HealthResponse> {
    let health = state.read().await.clone();
    Json(health)
}

/// AS1: Prometheus metrics endpoint
async fn metrics_endpoint() -> String {
    metrics::gather_metrics()
}

pub async fn run_health_server(port: u16, state: HealthState) {
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/metrics", get(metrics_endpoint))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    tracing::info!("Health check server listening on {}", addr);

    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("Health server failed to bind to {}: {}. System will continue without health endpoint.", addr, e);
            return;
        }
    };

    if let Err(e) = axum::serve(listener, app).await {
        tracing::error!("Health check server failed: {}", e);
    }
}

