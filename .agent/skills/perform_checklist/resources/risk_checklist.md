# Risk Assessment Checklist

Use this checklist when evaluating risk for new strategies or changes.

## Market Risk

| Factor | Assessment |
|--------|------------|
| Max drawdown acceptable | $ _______ |
| Daily loss limit | $ _______ |
| Correlation with other strategies | Low / Medium / High |
| Market regime sensitivity | Low / Medium / High |

## Operational Risk

- [ ] What happens if the exchange goes down?
- [ ] What happens if our server crashes mid-position?
- [ ] What happens if we lose internet connectivity?
- [ ] What happens if API credentials are revoked?
- [ ] What happens if the strategy has a bug?

## Execution Risk

- [ ] Slippage estimated: _______ bps
- [ ] Liquidity verified for position sizes
- [ ] Order fill rate in paper trading: _______%
- [ ] Average latency: _______ ms

## Position Limits

| Limit | Value |
|-------|-------|
| Max position size (USD) | $ _______ |
| Max positions per symbol | _______ |
| Max total exposure | $ _______ |
| Max correlated positions | _______ |

## Stress Scenarios

| Scenario | Expected Loss | Action |
|----------|---------------|--------|
| Flash crash (-10% in 5 min) | $ _______ | |
| Exchange outage (1 hour) | $ _______ | |
| Strategy bug (wrong direction) | $ _______ | |
| Network partition | $ _______ | |

## Risk Controls Verification

- [ ] Stop loss triggers verified
- [ ] Circuit breaker tested
- [ ] Position limit enforced
- [ ] Emergency shutdown works
- [ ] Alerts configured

## Approval

- Risk Level: Low / Medium / High / Critical
- Approved by: _______________
- Date: _______________
- Review date: _______________
