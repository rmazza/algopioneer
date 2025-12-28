---
description: Quick check-in on paper trading strategy status
---

# Coinbase Paper Trading Check-In

This workflow provides a quick status check on the running Coinbase paper trading strategy.

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

### 1. Check Container Status
// turbo
Verify the paper trading container is running on EC2:

```bash
ssh -i ~/.ssh/trading-key.pem ec2-user@$EC2_IP 'docker ps --format "{{.Names}}: {{.Status}}"'
```

Expected output: `algopioneer-paper: Up X minutes/hours`

---

### 2. Check for Recent Trades (Exits)
// turbo
Look for any realized trades in the last 10 minutes:

```bash
aws logs tail /algopioneer/paper-trading --since 10m --format short 2>&1 | grep -E "(Exit Signal|Transitioning to Exiting|PAPER TRADE)" | head -10
```

If empty, no trades have been executed recently.

---

### 3. Query Realized PnL (Last Hour)
// turbo
Run a CloudWatch Insights query to get aggregate PnL:

```bash
# Start the query
QUERY_ID=$(aws logs start-query \
  --log-group-name /algopioneer/paper-trading \
  --start-time $(date -d "1 hour ago" +%s) \
  --end-time $(date +%s) \
  --query-string 'fields @timestamp, @message
| filter @message like /Transitioning to Exiting/
| parse @message "Net PnL: * (Gross:" as net
| stats sum(net) as RealizedPnL, count() as TradeCount' \
  --output text --query 'queryId')

# Wait 3 seconds then fetch results
sleep 3
aws logs get-query-results --query-id $QUERY_ID
```

---

### 4. Check Current Holding Positions
// turbo
View the latest holding status and floating PnL:

```bash
aws logs tail /algopioneer/paper-trading --since 2m --format short 2>&1 | grep "Holding" | tail -10
```

This shows positions with their current Net PnL. Positions should be between:
- **Stop Loss**: -$15.00
- **Profit Target**: +$7.00

---

### 5. Verify Active Configuration (Optional)
Confirm the deployed config values:

```bash
ssh -i ~/.ssh/trading-key.pem ec2-user@$EC2_IP 'head -15 pairs_v3.json'
```

Expected values:
- `min_profit_threshold`: "7.0"
- `stop_loss_threshold`: "-15.0"

---

## Quick One-Liner Status Check

For a fast summary, run:

```bash
EC2_IP=$(aws ec2 describe-instances --filters "Name=tag:Name,Values=*algopioneer*" "Name=instance-state-name,Values=running" --query "Reservations[0].Instances[0].PublicIpAddress" --output text) && \
ssh -i ~/.ssh/trading-key.pem ec2-user@$EC2_IP 'docker ps -f name=algopioneer-paper --format "Container: {{.Status}}"' && \
aws logs tail /algopioneer/paper-trading --since 5m --format short 2>&1 | grep -c "Transitioning to Exiting" | xargs -I {} echo "Trades (5m): {}" && \
aws logs tail /algopioneer/paper-trading --since 1m --format short 2>&1 | grep "Holding" | wc -l | xargs -I {} echo "Active Positions: {}"
```
