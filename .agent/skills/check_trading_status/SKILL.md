---
name: Check Trading Status
description: Comprehensive health check for Alpaca, Coinbase, Post-Deployment, and Pairs Trading.
---

# Check Trading Status

This skill acts as the central dashboard for the trading system's health. It unifies status checks for all exchanges and lifecycle stages.

## Usage

**"Check Alpaca status"** -> Runs Alpaca Paper Trading Check-In.
**"Verify deployment"** -> Runs Post-Deploy Verification.
**"Check pairs health"** -> Runs Pairs Health Check & Rebalancing.

## Sub-Routines

### 1. Alpaca Paper Trading Check-In
*   **Goal**: Ensure Alpaca paper trading is active and healthy.
*   **Steps**:
    1.  **Container**: `ssh ... 'docker ps | grep algopioneer-alpaca'`
    2.  **API**: Query Alpaca Account API for equity & buying power.
    3.  **Positions**: List open positions.
    4.  **Orders**: Check last 20 orders for activity.
    5.  **Market**: Verify if market is open.

### 2. Coinbase Paper Trading Check-In
*   **Goal**: Monitor Coinbase simulation status.
*   **Steps**:
    1.  **Logs**: Check CloudWatch or local logs for activity.
    2.  **Container**: Verify container uptime.

### 3. Post-Deploy Verification
*   **Goal**: Detect "Ghost Trades" and verify new deployments.
*   **Steps**:
    1.  **Ghost Trade Check**: Compare internal logs vs. Exchange API.
    2.  **Log Inspection**: `docker logs --tail 100 ...` to check for startup errors.

### 4. Pairs Health Check
*   **Goal**: Monthly analysis of pairs trading performance.
*   **Steps**:
    1.  **Analyze**: Review PnL per pair.
    2.  **Rebalance**: Identification of cointegration breakdowns.
    3.  **Action**: Recommend keeping, removing, or adding pairs.
