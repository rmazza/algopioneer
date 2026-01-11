# Strategy Documentation Template

## Strategy Name
<!-- e.g., Mean Reversion, Momentum Breakout -->

## Overview
<!-- 2-3 sentence description of what this strategy does -->

## Theory
<!-- Academic/theoretical basis for the strategy -->

### Hypothesis
<!-- What market behavior does this exploit? -->

### Edge
<!-- Why does this work? What inefficiency does it capture? -->

## Implementation

### Entry Conditions
- Condition 1
- Condition 2

### Exit Conditions
- **Take Profit**: 
- **Stop Loss**: 
- **Time-based**: 

### Position Sizing
<!-- How is position size determined? -->

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | u64 | 20 | Lookback window for calculations |
| `entry_z_score` | Decimal | 2.0 | Z-score threshold for entry |
| `exit_z_score` | Decimal | 0.5 | Z-score threshold for exit |

## Risk Management

### Max Drawdown
<!-- Maximum acceptable drawdown before halting -->

### Position Limits
<!-- Maximum position size/exposure -->

### Correlation Risk
<!-- How does this interact with other strategies? -->

## Backtesting Results

| Metric | Value |
|--------|-------|
| Sharpe Ratio | |
| Max Drawdown | |
| Win Rate | |
| Profit Factor | |
| Total Trades | |
| Date Range | |

## Live Trading Considerations

### Market Conditions
<!-- When does this strategy perform well/poorly? -->

### Execution Risks
<!-- Slippage, liquidity concerns -->

### Monitoring
<!-- What to watch for in production -->

## Code Location
- Strategy: `src/strategy/xxx.rs`
- Tests: `tests/xxx_test.rs`
- Config: `configs/xxx.json`

## Changelog
<!-- Major changes to the strategy -->
| Date | Change |
|------|--------|
| YYYY-MM-DD | Initial implementation |
