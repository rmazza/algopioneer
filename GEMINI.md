# Project Overview

This is a Rust-based algorithmic trading project named `algopioneer`. It is designed to interact with the Coinbase Advanced Trade API for trading research and execution.

**Key Technologies:**

*   **Language:** Rust
*   **Core Libraries:**
    *   `cbadv`: For interacting with the Coinbase Advanced Trade API.
    *   `polars`: For high-performance data manipulation and analysis.
    *   `ta`: For technical analysis indicators.
    *   `tokio`: Asynchronous runtime for handling API requests.
    *   `dotenv`: For managing environment variables (API keys).

**Architecture:**

The project is a single binary application. The main logic is in `src/main.rs`, which initializes the connection to the Coinbase API and is intended to house the data fetching and trading strategy logic.

# Building and Running

**Prerequisites:**

*   Rust and Cargo installed.
*   A `.env` file in the root directory with the following variables:
    *   `COINbase_API_KEY`
    *   `COINbase_API_SECRET`

**Build:**

```bash
cargo build --release
```

**Run:**

```bash
cargo run --release
```

**Test:**

```bash
cargo test
```

# Development Conventions

*   **Configuration:** API keys and other secrets are managed through a `.env` file.
*   **Asynchronous Operations:** The project uses the `tokio` runtime for asynchronous operations, which is essential for interacting with web APIs.
*   **Error Handling:** The code uses `Result` and `Box<dyn std::error::Error>` for error handling, which is a standard practice in Rust.


