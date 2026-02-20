# Exchange APIs Reference

Quick reference for exchange integrations in algopioneer.

## Coinbase Advanced Trade

### Authentication
- **Method**: HMAC-SHA256 signature
- **Headers**: `CB-ACCESS-KEY`, `CB-ACCESS-SIGN`, `CB-ACCESS-TIMESTAMP`

### Endpoints
| Endpoint | Purpose |
|----------|---------|
| `GET /api/v3/brokerage/accounts` | List accounts |
| `POST /api/v3/brokerage/orders` | Place order |
| `GET /api/v3/brokerage/orders/{id}` | Get order status |
| `DELETE /api/v3/brokerage/orders/{id}` | Cancel order |

### WebSocket
- **URL**: `wss://advanced-trade-ws.coinbase.com`
- **Channels**: `ticker`, `level2`, `matches`
- **Rate Limit**: 750 requests/second

### Rate Limits
- REST: 10 requests/second per endpoint
- WebSocket: 750 messages/second

---

## Alpaca Markets

### Authentication
- **Headers**: `APCA-API-KEY-ID`, `APCA-API-SECRET-KEY`

### Base URLs
| Environment | URL |
|-------------|-----|
| Paper | `https://paper-api.alpaca.markets` |
| Live | `https://api.alpaca.markets` |

### Endpoints
| Endpoint | Purpose |
|----------|---------|
| `GET /v2/account` | Account info |
| `GET /v2/positions` | All positions |
| `GET /v2/orders` | List orders |
| `POST /v2/orders` | Place order |
| `GET /v2/clock` | Market hours |

### Market Hours
- US Equities: 9:30 AM - 4:00 PM ET (Mon-Fri)
- Pre-market: 4:00 AM - 9:30 AM ET
- After-hours: 4:00 PM - 8:00 PM ET

### Rate Limits
- 200 requests/minute

---

## Kraken

### Authentication
- **Method**: HMAC-SHA512
- **Headers**: `API-Key`, `API-Sign`

### Endpoints
| Endpoint | Purpose |
|----------|---------|
| `POST /0/private/Balance` | Account balance |
| `POST /0/private/AddOrder` | Place order |
| `POST /0/private/CancelOrder` | Cancel order |
| `GET /0/public/Ticker` | Current prices |

### WebSocket
- **URL**: `wss://ws.kraken.com`
- **Channels**: `ticker`, `ohlc`, `trade`

### Rate Limits
- REST: 15 requests/minute (public), varies by tier (private)
- WebSocket: No hard limit, fair use

---

## Common Patterns

### Order Types (All Exchanges)
| Type | Description |
|------|-------------|
| `market` | Execute at best available price |
| `limit` | Execute at specified price or better |
| `stop` | Trigger market order at price |
| `stop_limit` | Trigger limit order at price |

### Order Sides
| Side | Direction |
|------|-----------|
| `buy` | Long / Increase position |
| `sell` | Short / Decrease position |

### Time in Force
| TIF | Meaning |
|-----|---------|
| `gtc` | Good 'til canceled |
| `ioc` | Immediate or cancel |
| `fok` | Fill or kill |
| `day` | Good for day |
