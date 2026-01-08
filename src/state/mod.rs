//! Trade state persistence with atomic file writes.
//!
//! This module provides position tracking and state persistence to prevent
//! "Ghost Position" risk where positions are tracked incorrectly after crashes.
//!
//! # Safety
//! - Uses atomic file writes (write to temp, fsync, rename) for durability
//! - All state changes are persisted before being considered complete

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::Write;

/// Default state file path
const STATE_FILE: &str = "trade_state.json";

/// Detailed position information for reconciliation.
///
/// Tracks all information needed to properly close a position
/// and calculate PnL.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PositionDetail {
    /// Trading symbol (e.g., "BTC-USD")
    pub symbol: String,
    /// Order side ("buy" or "sell")
    pub side: String,
    /// Position quantity in base currency
    pub quantity: Decimal,
    /// Entry price for PnL calculation
    pub entry_price: Decimal,
}

/// State persistence with proper position tracking.
///
/// Fixes Ghost Position risk by maintaining a persistent map of open positions
/// that survives application restarts.
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct TradeState {
    /// Key: Symbol (e.g., "BTC-USD"), Value: Position details
    pub positions: HashMap<String, PositionDetail>,
}

impl TradeState {
    /// Load state from disk, returning default if file doesn't exist or is corrupted.
    pub fn load() -> Self {
        if let Ok(data) = fs::read_to_string(STATE_FILE) {
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Self::default()
        }
    }

    /// Persist state to disk atomically.
    ///
    /// Uses write-to-temp, fsync, rename pattern to ensure durability:
    /// 1. Write to temporary file
    /// 2. Sync to disk (fsync)
    /// 3. Atomic rename (POSIX guarantees atomicity on same filesystem)
    ///
    /// # Errors
    /// Returns error if file operations fail.
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let json = serde_json::to_string_pretty(self)?;
        let temp_path = format!("{}.tmp", STATE_FILE);

        // Write to temporary file first
        let mut file = fs::File::create(&temp_path)?;
        file.write_all(json.as_bytes())?;

        // Sync data to disk before rename (fsync for durability on Linux/Unix)
        // This ensures the write is fully committed before we make it visible
        file.sync_all()?;

        // Atomic rename: POSIX guarantees rename is atomic on the same filesystem
        // If we crash here, either the old file or new file exists - never a partial file
        fs::rename(&temp_path, STATE_FILE)?;

        Ok(())
    }

    /// Check if we have an open position for a symbol.
    pub fn has_position(&self, symbol: &str) -> bool {
        self.positions.contains_key(symbol)
    }

    /// Open a new position, replacing any existing position for the symbol.
    pub fn open_position(&mut self, detail: PositionDetail) {
        self.positions.insert(detail.symbol.clone(), detail);
    }

    /// Close a position and return its details for PnL calculation.
    ///
    /// Returns `None` if no position exists for the symbol.
    pub fn close_position(&mut self, symbol: &str) -> Option<PositionDetail> {
        self.positions.remove(symbol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_position_lifecycle() {
        let mut state = TradeState::default();

        // Initially no position
        assert!(!state.has_position("BTC-USD"));

        // Open position
        let detail = PositionDetail {
            symbol: "BTC-USD".to_string(),
            side: "buy".to_string(),
            quantity: dec!(0.001),
            entry_price: dec!(50000),
        };
        state.open_position(detail);
        assert!(state.has_position("BTC-USD"));

        // Close position
        let closed = state.close_position("BTC-USD");
        assert!(closed.is_some());
        assert!(!state.has_position("BTC-USD"));
    }
}
