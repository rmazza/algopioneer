//! AS11: OpenTelemetry observability for production tracing

use opentelemetry::{global, KeyValue};
use opentelemetry_sdk::{trace as sdktrace, Resource};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Initialize OpenTelemetry tracing with Jaeger exporter
pub fn init_telemetry(service_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Configure Jaeger exporter
    global::set_text_map_propagator(opentelemetry_jaeger::Propagator::new());
    
    let tracer = opentelemetry_jaeger::new_agent_pipeline()
        .with_service_name(service_name)
        .with_trace_config(
            sdktrace::config().with_resource(Resource::new(vec![
                KeyValue::new("service.name", service_name.to_string()),
            ]))
        )
        .install_simple()?;

    // Create tracing subscriber with OpenTelemetry layer
    tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(tracer))
        .with(tracing_subscriber::fmt::layer())
        .try_init()?;

    Ok(())
}

/// Shutdown OpenTelemetry gracefully
pub fn shutdown_telemetry() {
    global::shutdown_tracer_provider();
}
