---
description: Quick check-in on Alpaca equity paper trading status
---

# Alpaca Paper Trading Check-In

This workflow provides a quick status check on the running Alpaca equity paper trading strategy.

## Prerequisites

Get the EC2 instance IP dynamically (choose one method):

```bash
# Option A: From Terraform (if in project root)
EC2_IP=$(cd terraform && terraform output -raw public_ip)

# Option B: From AWS CLI
EC2_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=*algopioneer*" "Name=instance-state-name,Values=running" \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

echo "EC2 IP: $EC2_IP"
```

## Steps

### 0. Reference Cheat Sheet
// turbo
View the project cheat sheet for useful commands and deployment details (ignores gitignore):

```bash
cat CHEATSHEET.md || echo "Cheat sheet not found"
```

---

### 1. Check Container Status
// turbo
Verify the Alpaca paper trading container is running on EC2:

```bash
ssh -i ~/.ssh/trading-key.pem ec2-user@$EC2_IP 'docker ps --format "{{.Names}}: {{.Status}}" | grep alpaca'
```

Expected output: `algopioneer-alpaca: Up X minutes/hours`

---

### 2. Check Alpaca Paper Trading Account
// turbo
Query Alpaca's paper trading API for account status:

```bash
# Set your Alpaca paper trading credentials
source .env  # or export ALPACA_API_KEY and ALPACA_API_SECRET

curl -s -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
       -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET" \
       "https://paper-api.alpaca.markets/v2/account" | \
  jq '{cash: .cash, portfolio_value: .portfolio_value, equity: .equity, buying_power: .buying_power}'
```

---

### 3. Check Current Positions
// turbo
View all open positions in the Alpaca paper account:

```bash
curl -s -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
       -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET" \
       "https://paper-api.alpaca.markets/v2/positions" | \
  jq '.[] | {symbol: .symbol, qty: .qty, side: .side, market_value: .market_value, unrealized_pl: .unrealized_pl, unrealized_plpc: .unrealized_plpc}'
```

---

### 4. Check Recent Orders (Last 24h)
// turbo
Review recent orders placed by the bot:

```bash
curl -s -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
       -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET" \
       "https://paper-api.alpaca.markets/v2/orders?status=all&limit=20&direction=desc" | \
  jq '.[] | {symbol: .symbol, side: .side, qty: .qty, status: .status, filled_at: .filled_at, filled_avg_price: .filled_avg_price}'
```

---

### 5. Check Container Logs
// turbo
View recent trading activity from container logs:

```bash
ssh -i ~/.ssh/trading-key.pem ec2-user@$EC2_IP 'docker logs --tail 50 algopioneer-alpaca 2>&1 | grep -E "(PAPER TRADE|Signal|position|Entry|Exit)"'
```

---

### 6. Check Paper Trades CSV
// turbo
View the local paper trades log file:

```bash
ssh -i ~/.ssh/trading-key.pem ec2-user@$EC2_IP 'tail -20 paper_trades_alpaca.csv 2>/dev/null || echo "No paper trades file yet"'
```

---

### 7. Check Market Hours
// turbo
Verify if the US stock market is currently open:

```bash
curl -s -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
       -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET" \
       "https://paper-api.alpaca.markets/v2/clock" | \
  jq '{is_open: .is_open, next_open: .next_open, next_close: .next_close}'
```

> **Note:** US stock market hours are 9:30 AM - 4:00 PM ET, Mon-Fri.

---

## Quick One-Liner Status Check

For a fast summary, run:

```bash
source .env && \
echo "=== Alpaca Account ===" && \
curl -s -H "APCA-API-KEY-ID: $ALPACA_API_KEY" -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET" \
  "https://paper-api.alpaca.markets/v2/account" | jq '{equity: .equity, cash: .cash}' && \
echo "=== Open Positions ===" && \
curl -s -H "APCA-API-KEY-ID: $ALPACA_API_KEY" -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET" \
  "https://paper-api.alpaca.markets/v2/positions" | jq 'length' | xargs -I {} echo "Position count: {}" && \
echo "=== Market Status ===" && \
curl -s -H "APCA-API-KEY-ID: $ALPACA_API_KEY" -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET" \
  "https://paper-api.alpaca.markets/v2/clock" | jq '.is_open'
```

---

## Differences from Crypto Check-In

| Aspect | Crypto (Coinbase) | Equities (Alpaca) |
|--------|-------------------|-------------------|
| Market Hours | 24/7 | 9:30 AM - 4:00 PM ET |
| Polling | Real-time WebSocket | 1-minute polling |
| API | CloudWatch logs | Direct Alpaca API |
| Positions | Container logs | Alpaca Positions API |
