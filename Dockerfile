# ============================================================================
# Multi-stage Dockerfile for algopioneer (Refined)
# ============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Chef (Prepares the recipe)
# ------------------------------------------------------------------------------
FROM rust:1.83-bookworm AS chef
# Install cargo-chef globally (pinned for Rust 1.83 compatibility)
RUN cargo install cargo-chef --locked --version 0.1.68
WORKDIR /app

# ------------------------------------------------------------------------------
# Stage 2: Planner (Computes the lockfile recipe)
# ------------------------------------------------------------------------------
FROM chef AS planner
COPY . .
# Creates a minimal representation of dependencies
RUN cargo chef prepare --recipe-path recipe.json

# ------------------------------------------------------------------------------
# Stage 3: Builder (Compiles the binary)
# ------------------------------------------------------------------------------
FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json

# Install build dependencies (cmake for jemalloc, pkg-config for various crates)
# Note: libssl-dev NOT needed since we use rustls (pure Rust TLS)
RUN apt-get update && apt-get install -y \
    pkg-config \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Build dependencies - this is the cached layer!
RUN cargo chef cook --release --recipe-path recipe.json

# Build application
COPY . .
RUN cargo build --release --bin algopioneer

# Strip binary
RUN strip target/release/algopioneer

# ------------------------------------------------------------------------------
# Stage 4: Runtime
# ------------------------------------------------------------------------------
FROM gcr.io/distroless/cc-debian12:nonroot

WORKDIR /app

# Copy binary
COPY --from=builder /app/target/release/algopioneer /app/algopioneer

# NOTE: Distroless usually includes root CAs, but copying them from builder
# guarantees they match the build environment. 
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Optional: Copy timezone data if you need non-UTC logging
# COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# Run as non-root (ID 65532)
USER nonroot:nonroot

EXPOSE 8080

ENTRYPOINT ["/app/algopioneer"]
CMD ["--help"]