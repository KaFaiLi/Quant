# ⚡ Quant — Crypto Trading Research & Automation

> Quantitative research, backtesting, and live strategy deployment for crypto spot/futures and prediction markets.

---

## Structure

```
Quant/
├── data/               # Market data storage
│   ├── raw/            # Raw OHLCV, tick data (gitignored)
│   ├── processed/      # Cleaned, feature-engineered datasets
│   └── cache/          # API response cache (gitignored)
├── strategies/         # Strategy implementations
│   ├── spot_futures/   # Trend, momentum, mean-reversion, basis
│   ├── prediction_markets/ # Polymarket/event probability edge
│   └── arb/            # Funding rate arb, CEX-DEX arb
├── backtesting/        # Backtest engine wrappers & results
├── research/           # Research notes, strategy tearsheets
├── notebooks/          # Jupyter exploration notebooks
├── scripts/            # Data pipeline, runner scripts
└── monitoring/         # Live PnL tracking, alerting
```

## Strategy Roadmap

### Phase 1 — Low Risk / Proof of Concept
- [ ] **Funding Rate Arb**: Long spot + Short perp, harvest funding
- [ ] **Basis Trade**: Cash-and-carry on futures premium

### Phase 2 — Alpha Strategies
- [ ] **Momentum**: Cross-sectional crypto momentum (top/bottom decile)
- [ ] **Mean Reversion**: Stat-arb on correlated pairs (BTC/ETH, etc.)
- [ ] **Funding Rate Signal**: Use funding rate as directional signal

### Phase 3 — Prediction Markets
- [ ] **Polymarket Edge**: Mispriced event probabilities via news/sentiment
- [ ] **Calibration Model**: Probability calibration on political/macro events

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # Add API keys
```

## Status

🟡 Research & backtesting phase — no live capital deployed yet.
