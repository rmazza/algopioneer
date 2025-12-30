---
description: Monthly analysis of pairs trading health and rebalancing recommendations
---

# Pairs Health Check & Rebalancing Analysis

This workflow analyzes the current pairs trading performance and provides rebalancing recommendations.

## Prerequisites

Ensure you have:
- Active Alpaca paper trading account
- Current `discovered_pairs.json` deployed on EC2
- Access to EC2 instance

## Steps

### 1. Check Current Account Performance
// turbo
Query Alpaca paper trading account for overall performance:

```bash
source .env && \
curl -s -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
       -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET" \
       "https://paper-api.alpaca.markets/v2/account" | \
python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Equity: \${d[\"equity\"]}  |  Starting: \$100,000  |  PnL: \${float(d[\"equity\"])-100000:.2f}')"
```

---

### 2. Check Historical Orders (Last 30 Days)
// turbo
Review all orders to calculate per-pair performance:

```bash
source .env && \
curl -s -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
       -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET" \
       "https://paper-api.alpaca.markets/v2/orders?status=all&limit=100&direction=desc" | \
python3 -c "
import sys, json
from collections import defaultdict

orders = json.load(sys.stdin)
trades_by_symbol = defaultdict(list)

for o in orders:
    if o['status'] == 'filled':
        trades_by_symbol[o['symbol']].append({
            'side': o['side'],
            'qty': float(o['filled_qty']),
            'price': float(o['filled_avg_price']) if o['filled_avg_price'] else 0
        })

print('Symbol | Trades | Avg Price')
print('-' * 35)
for sym, trades in sorted(trades_by_symbol.items()):
    avg_price = sum(t['price'] for t in trades) / len(trades) if trades else 0
    print(f'{sym:6} | {len(trades):6} | \${avg_price:.2f}')
"
```

---

### 3. Re-run Discovery with Current Data
// turbo
Run the discovery pipeline to find current cointegrated pairs:

```bash
cargo run --release -- discover-pairs \
  --exchange alpaca \
  --symbols default \
  --lookback-days 730 \
  --initial-capital 2000 \
  --output discovered_pairs_new.json
```

---

### 4. Compare Old vs New Pairs

After discovery completes, compare the current deployed pairs with newly discovered pairs:

```bash
echo "=== Currently Deployed Pairs ===" && \
cat discovered_pairs.json | python3 -c "
import sys, json
pairs = json.load(sys.stdin)
for p in pairs:
    print(f\"  {p['spot_symbol']}/{p['future_symbol']} (window={p['window_size']}, z={p['entry_z_score'][:4]})\")
" && \
echo "" && \
echo "=== Newly Discovered Pairs ===" && \
cat discovered_pairs_new.json | python3 -c "
import sys, json
pairs = json.load(sys.stdin)
for p in pairs:
    print(f\"  {p['spot_symbol']}/{p['future_symbol']} (window={p['window_size']}, z={p['entry_z_score'][:4]})\")
"
```

---

### 5. Analyze Pair Health

For each current pair, check if it still appears in the new discovery. Pairs that disappear may have lost cointegration.

**Questions to answer:**
1. Which pairs are in both old and new? (Keep these)
2. Which pairs are only in old? (Review for removal)
3. Which pairs are only in new? (Consider adding)
4. Has the validation Sharpe changed significantly?

---

### 6. Decision Matrix

| Scenario | Action |
|----------|--------|
| Pair in both, similar Sharpe | Keep |
| Pair in both, Sharpe dropped >50% | Review window/z-score params |
| Pair only in old | Remove if losing money |
| Pair only in new, high Sharpe | Add if capital available |
| Pair losing 3+ consecutive trades | Remove immediately |

---

### 7. Deploy Updated Config (If Changes Made)

If you decide to update the pairs:

```bash
# Update max_tick_age_ms in new config
sed -i 's/"max_tick_age_ms": 2000/"max_tick_age_ms": 120000/g' discovered_pairs_new.json

# Upload to EC2
scp -i ~/.ssh/trading-key.pem discovered_pairs_new.json ec2-user@$(cd terraform && terraform output -raw public_ip):~/discovered_pairs.json

# Redeploy
ssh -i ~/.ssh/trading-key.pem ec2-user@$(cd terraform && terraform output -raw public_ip) 'bash ~/deploy_alpaca.sh'
```

---

## Health Check Summary Template

After running this workflow, document:

```
Date: YYYY-MM-DD
Account Equity: $XX,XXX
Monthly PnL: $X,XXX
Trades This Month: XX

Pairs Status:
- [KEEP] ABBV/JNJ - Still cointegrated, Sharpe stable
- [KEEP] GS/MS - Traditional pair, consistent
- [REVIEW] COP/KHC - Cross-sector, monitor closely
- [REMOVE] XXX/YYY - Lost cointegration
- [ADD] AAA/BBB - New discovery, high Sharpe

Next Review: YYYY-MM-DD
```
