---
name: Check Trading Status
description: Comprehensive health check for the Alpaca Paper Trading strategy.
---

# Check Trading Status

This skill acts as the central dashboard for the trading system's health, focusing specifically on the Alpaca trading strategy.

## Usage

**"Check Alpaca status"** -> Runs Alpaca Paper Trading Check-In.

## Sub-Routines

### 1. Alpaca Paper Trading Check-In
*   **Goal**: Ensure Alpaca paper trading is active and healthy.
*   **Context**: The application uses the Alpaca v2 API. Authentication credentials require `ALPACA_API_KEY` and `ALPACA_API_SECRET` which must be sourced from the `.env` file in the project directory. The container runs on an AWS EC2 instance.
*   **Steps**:
    1.  **Container Status**: Verify the Docker container is running on the EC2 instance by fetching the IP from Terraform and executing `docker ps`. Note that the container only runs during market hours (09:15 AM - 04:15 PM EST).
        ```bash
        export EC2_IP=$(cd terraform && terraform output -raw public_ip)
        ssh -i ~/.ssh/trading-key.pem ec2-user@$EC2_IP 'docker ps | grep algopioneer-alpaca'
        ```
    2.  **Container Logs**: Check the recent logs of the container on the EC2 instance to ensure no errors or halt states:
        ```bash
        export EC2_IP=$(cd terraform && terraform output -raw public_ip)
        ssh -i ~/.ssh/trading-key.pem ec2-user@$EC2_IP 'docker logs --tail 50 algopioneer-alpaca'
        ```
    3.  **API Account**: Query Alpaca Account API for equity & buying power locally (ensure you are using credentials from `.env`):
        ```bash
        source .env && curl -s -H "APCA-API-KEY-ID: $ALPACA_API_KEY" -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET" https://paper-api.alpaca.markets/v2/account | jq '{status, equity, buying_power}'
        ```
    4.  **Positions**: List open positions and show the unrealized PnL:
        ```bash
        source .env && curl -s -H "APCA-API-KEY-ID: $ALPACA_API_KEY" -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET" https://paper-api.alpaca.markets/v2/positions | jq 'map({symbol, qty, market_value, unrealized_pl})'
        ```
    5.  **Orders**: Check last 20 orders for activity tracking execution:
        ```bash
        source .env && curl -s -H "APCA-API-KEY-ID: $ALPACA_API_KEY" -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET" "https://paper-api.alpaca.markets/v2/orders?status=all&limit=20" | jq 'map({symbol, side, qty, status, filled_at})'
        ```
    6.  **Market**: Verify if the US equity market is currently open:
        ```bash
        source .env && curl -s -H "APCA-API-KEY-ID: $ALPACA_API_KEY" -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET" https://paper-api.alpaca.markets/v2/clock | jq '{is_open, next_open, next_close}'
        ```

## Output Template

When reporting the status back to the user, use the following format:

```markdown
# Alpaca Trading Status Check
**Timestamp**: <current timestamp>
**Market Status**: <Open / Closed (Next Open: ...)>

## System Health
- **Container**: `<running / stopped / not found>` (EC2 IP: `<IP>`)
- **Recent Logs**:
  ```
  <snippet of relevant logs or "No errors found">
  ```

## Account Status
- **Equity**: $<value>
- **Buying Power**: $<value>
- **Account Status**: <ACTIVE / ...>

## Positions
| Symbol | Qty | Market Value | Unrealized PnL |
|---|---|---|---|
| <SYM> | <qty> | $<value> | $<value> |

## Recent Activity (Last 5 Orders)
| Symbol | Side | Qty | Status | Time |
|---|---|---|---|---|
| <SYM> | <BUY/SELL> | <qty> | <FILLED/etc> | <time> |
```
