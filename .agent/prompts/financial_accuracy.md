# Financial Accuracy Prompt

You are enforcing financial correctness standards. All monetary calculations must be precise, auditable, and deterministic.

## Non-Negotiable Rules

### Rule 1: No Floating Point for Money
```rust
// ❌ FORBIDDEN
let price: f64 = 100.50;
let total = price * quantity;

// ✅ REQUIRED
use rust_decimal::Decimal;
let price = Decimal::from_str("100.50").unwrap();
let total = price * quantity;
```

### Rule 2: Explicit Rounding
```rust
// ❌ Implicit rounding
let fee = total * Decimal::from_str("0.001").unwrap();

// ✅ Explicit rounding strategy
use rust_decimal::RoundingStrategy;
let fee = (total * Decimal::from_str("0.001").unwrap())
    .round_dp_with_strategy(2, RoundingStrategy::MidpointAwayFromZero);
```

### Rule 3: Currency Units Are Part of the Type
```rust
// ❌ Ambiguous
fn calculate_pnl(entry: Decimal, exit: Decimal) -> Decimal

// ✅ Clear units
fn calculate_pnl(entry_usd: Decimal, exit_usd: Decimal) -> Decimal
// Or use newtypes:
struct Usd(Decimal);
fn calculate_pnl(entry: Usd, exit: Usd) -> Usd
```

### Rule 4: Deterministic Calculations
```rust
// ❌ Non-deterministic (relies on system time)
let timestamp = SystemTime::now();

// ✅ Deterministic (injectable clock)
fn calculate_pnl(&self, clock: &impl Clock) -> Decimal {
    let now = clock.now();
    // ...
}
```

### Rule 5: Overflow Protection
```rust
// ❌ Can panic on overflow
let total = quantity * price;

// ✅ Checked arithmetic
let total = quantity
    .checked_mul(price)
    .ok_or(CalculationError::Overflow)?;
```

## Financial Calculation Patterns

### PnL Calculation
```rust
pub fn calculate_realized_pnl(
    entry_price: Decimal,
    exit_price: Decimal,
    quantity: Decimal,
    side: OrderSide,
) -> Decimal {
    let gross_pnl = match side {
        OrderSide::Buy => (exit_price - entry_price) * quantity,
        OrderSide::Sell => (entry_price - exit_price) * quantity,
    };
    gross_pnl.round_dp_with_strategy(2, RoundingStrategy::MidpointAwayFromZero)
}
```

### Fee Calculation
```rust
pub fn calculate_fee(notional: Decimal, fee_rate: Decimal) -> Decimal {
    (notional * fee_rate)
        .round_dp_with_strategy(8, RoundingStrategy::AwayFromZero)
}
```

## Testing Requirements

1. Test with edge cases: `0`, negative values, very large numbers
2. Test rounding at boundaries
3. Verify PnL matches manual calculation
4. Ensure calculations are reversible where applicable
