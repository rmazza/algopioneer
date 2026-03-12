# AlgoPioneer Trading Log

This log tracks pair rotations, performance snapshots, and key decisions.

---

## 2026-03-05 | Operational Stabilization: Daily Restart Cycle

**System Status Update:**
Detected significant execution latency (up to 46s tick age) on Alpaca Paper Trading.
- **Root Cause 1:** 12-second clock drift on EC2 (server was ahead of Alpaca API).
- **Root Cause 2:** WebSocket backpressure on Alpaca Paper feed over long-running sessions.

**Actions Taken:**
- **Manual Restart:** Recycled the `algopioneer-alpaca` container; confirmed latency dropped to <500ms.
- **EC2 Infrastructure:** Installed `cronie` on Amazon Linux 2023 instance.
- **Automation:** Configured daily `cron` jobs (Mon-Fri) to align with US Equity market:
  - `09:15 EST`: `deploy_alpaca.sh` (Fresh start with latest discovery config).
  - `16:15 EST`: `docker stop` (Graceful shutdown after close).

**Account Snapshot:**
| Metric | Value |
|--------|-------|
| Equity | $104,377.02 |
| Account Status | ACTIVE |
| Position State | FLAT |

---

## 2026-01-10 | Pairs Health Check & Rotation

**Account Snapshot:**
| Metric | Value |
|--------|-------|
| Equity | $99,993.10 |
| Starting Capital | $100,000 |
| Cumulative PnL | -$6.90 |
| Total Filled Trades | 27 |

**Pair Rotation:**
| Action | Pair | Reason |
|--------|------|--------|
| ❌ REMOVED | F/JNJ | Lost cointegration |
| ❌ REMOVED | WFC/JPM | Lost cointegration |
| ❌ REMOVED | WMT/WFC | Lost cointegration |
| ✅ ADDED | JNJ/ABBV | Healthcare sector, Train SR 1.64 |
| ✅ ADDED | KHC/GIS | Consumer staples (food) |
| ✅ ADDED | AXP/BAC | Financials (credit/bank) |
| ✅ ADDED | PFE/PEP | Cross-sector, Val SR 4.46 ⭐ |
| ✅ ADDED | AXP/JPM | Financials (credit/bank) |
| ✅ ADDED | MS/GS | Investment banks |

**Notes:**
- All 6 old positions liquidated at market close
- Complete pair rotation due to lost cointegration across all deployed pairs
- PFE/PEP shows exceptional validation Sharpe (4.46) - watch closely

**Next Review:** 2026-02-10

---

## 2026-01-22 | Pairs Health Check & Rotation

**Account Snapshot:**
| Metric | Value |
|--------|-------|
| Equity | $93,438.72 |
| Starting Capital | $100,000 |
| Cumulative PnL | -$6,561.28 |

**Pair Rotation:**
| Action | Pair | Reason |
|--------|------|--------|
| ❌ REMOVED | JNJ/ABBV | Lost cointegration |
| ❌ REMOVED | MS/GS | Lost cointegration |
| ✅ ADDED | COP/PEP | New discovery (Energy/Consumer) |
| ✅ ADDED | IBM/AXP | New discovery (Tech/Finance) |
| ✅ ADDED | IBM/JPM | New discovery (Tech/Finance) |
| ✅ ADDED | WFC/AXP | New discovery (Banking) |
| 🔄 UPDATED | KHC/GIS | Replaced with GIS/KHC (New direction) |

**Notes:**
- Deployed 8 total pairs (4 new, 3 kept, 1 updated)
- Increased `max_tick_age_ms` to 120s for better stability
- **ACTION REQUIRED**: Open positions for removed pairs (GS, ABBV) are now orphaned and must be closed manually via Alpaca dashboard or CLI.

**Next Review:** 2026-02-22

---

## Template for Future Entries

```markdown
## YYYY-MM-DD | [Health Check / Rotation / Incident]

**Account Snapshot:**
| Metric | Value |
|--------|-------|
| Equity | $XX,XXX |
| Monthly PnL | $X,XXX |
| Trades This Month | XX |

**Changes Made:**
- [Description of any pair changes, parameter tweaks, etc.]

**Notes:**
- [Key observations, market conditions, lessons learned]

**Next Review:** YYYY-MM-DD
```
