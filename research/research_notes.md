# Research Notes — Crypto Quant Strategy Foundation

**Author:** MrTQQQ  
**Date:** 2026-02-22  
**Status:** Active Reference

---

## Papers Read & Key Takeaways

### 1. "Fundamentals of Perpetual Futures" (He, Manela, Ross, von Wachter — arXiv:2212.06888, 2024)
*Washington University in St. Louis / University of Copenhagen*

**Core finding:** Perpetual futures deviate significantly from theoretical no-arbitrage prices. These deviations are exploitable.

**Key empirical results:**
- Mean absolute deviation from theoretical price: **60–90% per year** across major cryptos — far larger than traditional FX markets
- Deviations **co-move across currencies** (systemic, not idiosyncratic) — driven by shared arbitrage capital and sentiment
- Deviations **decline ~11%/year** (markets becoming more efficient — window is closing, but still open)
- **Past return momentum explains >50% of the futures-spot gap** (R² > 0.50) — when recent returns are high, futures trade at premium to spot
- Random-maturity arbitrage (long spot + short perp when spread exceeds theoretical bound, close when it returns to fair value) yields:
  - **Sharpe ratio of 1.8** for retail investors (high fees, Binance tier)
  - **Sharpe ratio of 3.5** for market makers (zero fees)
  - Even better for ETH and altcoins vs BTC

**Implication for our strategy:**
- Don't just harvest funding blindly — use the **futures-spot spread vs theoretical bound** as the entry signal
- The spread is driven by: (1) interest rates, (2) leverage demand from longs, (3) momentum
- When momentum is high → longs push perp above spot → we get paid to short perp + long spot
- This makes our strategy **momentum-conditional**: enter funding arb when the market is in an uptrend (funding is highest and most persistent)
- Exit before momentum reverses (funding flips negative)

**Strategy upgrade from this paper:**
> Original: Enter when funding > X%  
> **Upgraded:** Enter when (1) funding > X% AND (2) futures-spot spread > theoretical bound AND (3) 30d momentum is positive  
> This captures the regime where the paper shows the edge is highest.

---

### 2. "A Time-Varying Network for Cryptocurrencies" (Guo, Härdle, Tao — 2018/2022)
*Cross-predictability of crypto returns via network effects*

**Key finding:** Crypto returns exhibit **cross-predictability** — returns of related coins (by technology/ecosystem) predict each other with a lag.

**Implication:**  
- BTC moves first → ETH follows (BTC leads the market)
- Layer-1 ecosystems cluster (SOL, AVAX, ATOM move together)
- **Use BTC return as a leading signal for altcoin momentum**
- Network topology changes over time — need time-varying weights

**Strategy implication:**
- In our cross-sectional momentum: weight BTC signal more heavily
- When BTC has strong positive momentum → go long the highest-beta altcoins
- When BTC is negative → go flat or reduce exposure (not short altcoins due to liquidation risk)

---

### 3. "Multi-Factor Inception: What to Do with All of These Features?" (Liu, Zohren — arXiv 2023)
*Crypto factor investing with ML*

**Key finding:** In crypto, many traditional equity factors transfer — but with important differences:
- **Momentum** works but is shorter-horizon than equities (days/weeks, not months)
- **Reversal** at 1-week horizon is strong (esp. after large moves)
- **Volume/turnover** is a significant factor
- **Volatility** is a factor (low-vol assets outperform long-term)
- ML methods (gradient boosting, neural nets) beat simple factor combinations

**Implication:**
- Our momentum lookback of 28 days is likely too long — **7–14 days** is the sweet spot in crypto
- Need to add a **short-term reversal filter**: if last 1–3 days return is extreme, fade rather than follow
- Volume surge + momentum is a stronger signal than momentum alone

---

## Synthesized Strategy Insights

### What Works in Crypto (Evidence-Based)

| Factor | Works? | Horizon | Notes |
|--------|--------|---------|-------|
| Momentum | ✅ Strong | 7–14d | Shorter than equities |
| Short-term reversal | ✅ | 1–3d | Fade extreme moves |
| Funding rate arb | ✅ Very Strong | Continuous | Sharpe 1.8–3.5 |
| Futures-spot spread | ✅ | Hours–days | Entry signal for arb |
| Volume-momentum | ✅ | 7d | Better than price alone |
| Low volatility | ✅ Modest | Monthly | Risk-adjusted |
| Cross-asset lead-lag (BTC→alts) | ✅ | Hours–2d | BTC leads market |
| Pairs stat-arb | ⚠️ Mixed | Days–weeks | Pairs break down in crypto more than equities |

### Key Risks in Crypto Not in Equities
1. **Liquidation cascades** — high leverage across the market causes correlated crashes; stop losses are essential
2. **Exchange counterparty risk** — never keep more than needed on any one exchange
3. **Funding rate flips** — when market turns bearish, funding goes negative (longs get paid) — this destroys the arb
4. **Regulatory risk** — sudden exchange shutdowns (e.g., FTX) can freeze capital
5. **Correlation spikes** — in crashes, all correlations → 1, diversification fails

---

## Revised Strategy Priorities

### Priority 1: Enhanced Funding Rate Arb (Highest Sharpe, Most Evidence)
**Original paper Sharpe: 1.8–3.5**

Upgrade from naive funding harvest to:
1. Compute theoretical perpetual price = Spot × (1 + r/κ) where r=risk-free rate, κ=funding intensity
2. Entry: futures-spot spread > theoretical bound AND funding > 0.01%/8h AND 30d BTC momentum > 0
3. Exit: spread returns to fair value OR momentum turns negative OR funding < 0
4. Size: fixed $notional, 1x leverage (delta neutral via spot + perp)

### Priority 2: Momentum with BTC Lead Signal
Upgrade from naive cross-sectional rank to:
1. Use 7–14 day lookback (not 28)
2. Add BTC return as a factor weight (when BTC strong → amplify altcoin longs)
3. Add short-term reversal filter: skip assets with extreme 3-day returns
4. Volume-weighted momentum: momentum × volume_ratio signal

### Priority 3: Funding Rate as Directional Signal
New strategy (not yet implemented):
- High positive funding = crowded longs = potential for squeeze down
- Extreme funding (>0.1%/8h) → contrarian short signal for spot
- Low/negative funding = crowded shorts = potential squeeze up
- This is a **mean-reversion on funding extremes**, different from the arb

---

## Implementation Checklist

- [x] Data pipeline (OHLCV via CCXT)
- [ ] Funding rate data pipeline (CCXT fetch_funding_rate_history)
- [ ] Futures-spot spread computation
- [ ] Theoretical no-arbitrage bound calculator
- [x] Basic funding rate arb backtest
- [ ] Enhanced funding rate arb (momentum-conditional entry)
- [x] Cross-sectional momentum (basic)
- [ ] Upgraded momentum (7–14d, volume-weighted, BTC lead)
- [x] Pairs mean reversion
- [ ] Funding rate as directional signal strategy
- [ ] Walk-forward validation (avoid overfitting)
- [ ] Transaction cost model (tiered by exchange level)
- [ ] Combined portfolio (correlation-weighted allocation)

---

## References

1. He S., Manela A., Ross O., von Wachter V. (2024). "Fundamentals of Perpetual Futures." arXiv:2212.06888
2. Liu Y., Tsyvinski A., Wu X. (2022). "Common Risk Factors in Cryptocurrency." Journal of Finance. (3-factor model)
3. Guo L., Härdle W.K., Tao Y. (2022). "A Time-Varying Network for Cryptocurrencies." arXiv:2108.11921
4. Liu T., Zohren S. (2023). "Multi-Factor Inception: What to Do with All of These Features?" arXiv (crypto ML factors)
5. Cong L., Karolyi A., Tang H., Zhao W. (2022). "Value Premium, Global Internet Adoption, and Cryptocurrency Returns." (5-factor model)
