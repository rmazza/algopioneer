# Trading Terms Reference

Quick reference for trading concepts specific to algopioneer strategies.

## Pairs Trading Concepts

### Cointegration
Two assets are cointegrated if their linear combination is stationary (mean-reverting). This is different from correlation:
- **Correlation**: Prices move together
- **Cointegration**: Spread between prices reverts to mean

### Z-Score Calculation
```
z_score = (spread - mean(spread)) / std(spread)
```

### Entry/Exit Logic
```
If z_score > entry_threshold:
    Short spread (short A, long B)
If z_score < -entry_threshold:
    Long spread (long A, short B)
If |z_score| < exit_threshold:
    Exit position
```

## Position Sizing

### Dollar-Neutral
Equal dollar amounts on each leg:
```
leg_a_qty = capital / (2 * price_a)
leg_b_qty = capital / (2 * price_b)
```

### Beta-Neutral
Weighted by beta to reduce market exposure:
```
leg_a_qty = capital / (2 * price_a)
leg_b_qty = capital / (2 * price_b) * hedge_ratio
```

## Risk Metrics

### Sharpe Ratio
```
sharpe = (mean_return - risk_free_rate) / std_return * sqrt(252)
```
- > 1.0: Acceptable
- > 2.0: Good
- > 3.0: Excellent

### Maximum Drawdown
```
max_drawdown = (peak - trough) / peak
```
Typical acceptable range: 10-20%

### Profit Factor
```
profit_factor = gross_profit / gross_loss
```
- > 1.0: Profitable
- > 1.5: Good
- > 2.0: Excellent

## Exchange-Specific

### Crypto (Coinbase)
- 24/7 trading
- Real-time WebSocket
- Fees: ~0.5% taker

### Equities (Alpaca)
- Market hours: 9:30 AM - 4:00 PM ET
- 1-minute polling (no real-time WebSocket)
- Fees: $0 commission
- T+2 settlement

## Time Zones
- **UTC**: All timestamps in code
- **ET**: US market hours
- **Crypto**: No market hours (24/7)
