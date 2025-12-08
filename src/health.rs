//! AS10: Health check HTTP endpoint for monitoring

use axum::{
    routing::get,
    Router,
    Json,
};
use serde::{Serialize, Deserialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::Utc; // Added for `chrono::Utc::now()`

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

async fn health_check(
    axum::extract::State(state): axum::extract::State<HealthState>
) -> Json<HealthResponse> {
    let health = state.read().await.clone();
    Json(health)
}

pub async fn run_health_server(port: u16, state: HealthState) {
    let app = Router::new()
        .route("/health", get(health_check))
        .with_state(state);
    
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    
    tracing::info!("Health check server listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind health check server");
    
    axum::serve(listener, app)
        .await
        .expect("Health check server failed");
}
