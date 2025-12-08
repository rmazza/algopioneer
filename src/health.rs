//! AS10: Health check HTTP endpoint for monitoring

use axum::{
    routing::get,
    Router,
    Json,
};
use serde::{Serialize, Deserialize};
use std::net::SocketAddr;

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

pub async fn run_health_server(port: u16) {
    let app = Router::new()
        .route("/health", get(health_check));
    
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    
    tracing::info!("Health check server listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind health check server");
    
    axum::serve(listener, app)
        .await
        .expect("Health check server failed");
}
