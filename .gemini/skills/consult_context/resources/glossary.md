# Trading Glossary

Domain-specific terms used in algopioneer.

## General Trading

| Term | Definition |
|------|------------|
| **Long** | Buying an asset expecting price to rise |
| **Short** | Selling an asset expecting price to fall |
| **Position** | Current holding of an asset |
| **PnL** | Profit and Loss |
| **Realized PnL** | PnL from closed positions |
| **Unrealized PnL** | PnL from open positions (paper gain/loss) |
| **Slippage** | Difference between expected and actual execution price |
| **Spread** | Difference between bid and ask prices |

## Pairs/Statistical Arbitrage

| Term | Definition |
|------|------------|
| **Cointegration** | Statistical relationship where spread is mean-reverting |
| **Spread** | Difference between normalized prices of two assets |
| **Z-Score** | Standard deviations from the mean spread |
| **Half-life** | Time for spread to revert halfway to mean |
| **Entry Z-Score** | Z-score threshold to open position |
| **Exit Z-Score** | Z-score threshold to close position |
| **Window Size** | Lookback period for calculating statistics |

## Basis Trading

| Term | Definition |
|------|------------|
| **Spot** | Immediate delivery price |
| **Future** | Contract for future delivery |
| **Basis** | Difference between spot and future prices |
| **Contango** | Future price > Spot price |
| **Backwardation** | Spot price > Future price |
| **Roll** | Closing expiring future, opening next |

## Risk Management

| Term | Definition |
|------|------------|
| **Stop Loss** | Automatic exit at loss threshold |
| **Take Profit** | Automatic exit at profit threshold |
| **Drawdown** | Peak-to-trough decline in portfolio value |
| **Sharpe Ratio** | Risk-adjusted return (return / volatility) |
| **Circuit Breaker** | Automatic halt on repeated failures |

## Technical

| Term | Definition |
|------|------------|
| **Tick** | Single price update |
| **OHLCV** | Open, High, Low, Close, Volume candle |
| **WebSocket** | Real-time bidirectional connection |
| **REST** | Request/response API pattern |
| **Rate Limit** | Maximum API calls per time period |
