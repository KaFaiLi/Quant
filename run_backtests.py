"""
Master Backtest Runner
=======================
Runs all strategies and produces a consolidated tearsheet / comparison.
"""

import sys
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

log = logging.getLogger(__name__)
RESULTS_DIR = Path(__file__).parent / "backtesting"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_all_backtests(fetch_fresh: bool = False):
    results_summary = []

    # ── 1. Funding Rate Arb ─────────────────────────────────────────────────
    print("\n🔄 Running: Funding Rate Arbitrage...")
    try:
        from strategies.arb.funding_rate_arb import run_funding_arb
        r = run_funding_arb("BTC/USDT")
        results_summary.append({
            "strategy": "Funding Rate Arb (BTC/USDT)",
            "total_return_pct": r.get("total_return_pct"),
            "annualized_return_pct": r.get("annualized_return_pct"),
            "sharpe_ratio": r.get("sharpe_ratio"),
            "max_drawdown_pct": r.get("max_drawdown_pct"),
            "n_trades": r.get("n_trades"),
        })
    except Exception as e:
        log.error(f"Funding Rate Arb failed: {e}")

    # ── 2. Pairs Mean Reversion ──────────────────────────────────────────────
    print("\n🔄 Running: Pairs Mean Reversion (BTC/ETH)...")
    try:
        from strategies.spot_futures.pairs_mean_reversion import run_pairs
        r = run_pairs(("BTC/USDT", "ETH/USDT"))
        results_summary.append({
            "strategy": "Pairs MeanRev (BTC/ETH)",
            "total_return_pct": r.get("total_return_pct"),
            "annualized_return_pct": r.get("annualized_return_pct"),
            "sharpe_ratio": r.get("sharpe_ratio"),
            "max_drawdown_pct": r.get("max_drawdown_pct"),
            "n_trades": r.get("n_trades"),
        })
    except Exception as e:
        log.error(f"Pairs MeanRev failed: {e}")

    # ── 3. Cross-Sectional Momentum ──────────────────────────────────────────
    print("\n🔄 Running: Cross-Sectional Momentum...")
    try:
        from strategies.spot_futures.momentum import run_momentum
        r = run_momentum()
        results_summary.append({
            "strategy": "Cross-Sectional Momentum",
            "total_return_pct": r.get("total_return_pct"),
            "annualized_return_pct": r.get("annualized_return_pct"),
            "sharpe_ratio": r.get("sharpe_ratio"),
            "max_drawdown_pct": r.get("max_drawdown_pct"),
            "n_trades": r.get("n_rebalances"),
        })
    except Exception as e:
        log.error(f"Momentum failed: {e}")

    # ── Summary Table ────────────────────────────────────────────────────────
    if results_summary:
        df = pd.DataFrame(results_summary).set_index("strategy")
        summary_path = RESULTS_DIR / "strategy_comparison.csv"
        df.to_csv(summary_path)

        print("\n" + "=" * 70)
        print("  STRATEGY COMPARISON SUMMARY")
        print("=" * 70)
        print(df.to_string())
        print("=" * 70)
        print(f"\n✅ Saved → {summary_path}")

    return results_summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_all_backtests()
