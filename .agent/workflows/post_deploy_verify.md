---
description: Verify deployment by checking logs align with API reality
---

# Post-Deploy Verification

After deploying, run this to confirm the container is correctly integrated with the exchange API.

// turbo-all

## Prerequisites

```bash
EC2_IP=$(cd terraform && terraform output -raw public_ip)
source .env
echo "EC2 IP: $EC2_IP"
```

---

## Steps

### 1. Check Container Running
```bash
ssh -i ~/.ssh/trading-key.pem ec2-user@$EC2_IP 'docker ps | grep algopioneer-alpaca'
```

---

### 2. Check No Crash Loop
```bash
ssh -i ~/.ssh/trading-key.pem ec2-user@$EC2_IP 'docker inspect algopioneer-alpaca --format "{{.State.Status}} (Restarts: {{.RestartCount}})"'
```

Expected: `running (Restarts: 0)`

> [!WARNING]
> If `RestartCount > 0`, check logs for startup errors (missing env vars, wrong exchange flag).

---

### 3. Verify Correct Exchange in Logs
```bash
ssh -i ~/.ssh/trading-key.pem ec2-user@$EC2_IP 'docker logs algopioneer-alpaca 2>&1 | head -20 | grep -E "(Exchange:|Alpaca|Coinbase)"'
```

Expected: Should show `Alpaca` (not `Coinbase`) if deploying Alpaca strategy.

---

### 4. Wait for First Trade Signal (If Market Open)
```bash
ssh -i ~/.ssh/trading-key.pem ec2-user@$EC2_IP 'docker logs --tail 100 algopioneer-alpaca 2>&1 | grep -E "(order created|Entry Signal|Exit Signal)"'
```

---

### 5. Cross-Check API vs Logs ("Ghost Trade" Detection)

If logs show trades but API shows nothing, you have a version mismatch.

```bash
# Check API for recent orders
curl -s -H "APCA-API-KEY-ID: $ALPACA_API_KEY" -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET" \
  "https://paper-api.alpaca.markets/v2/orders?status=all&limit=5&direction=desc" | grep -o '"symbol":"[^"]*"' | head -5 || echo "No orders"
```

> [!CAUTION]
> If logs show "order created" but API shows no orders, the container is running mock/outdated code. **Redeploy with current code.**

---

## Quick Validation Command

```bash
ssh -i ~/.ssh/trading-key.pem ec2-user@$EC2_IP 'docker inspect algopioneer-alpaca --format "Status: {{.State.Status}}, Restarts: {{.RestartCount}}" && docker logs algopioneer-alpaca 2>&1 | head -5'
```
