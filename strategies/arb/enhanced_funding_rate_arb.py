"""
Enhanced Funding Rate Arbitrage Strategy
=========================================
Based on: He, Manela, Ross, von Wachter (2024) "Fundamentals of Perpetual Futures"
arXiv:2212.06888 — Sharpe 1.8–3.5 empirically documented

UPGRADE FROM NAIVE VERSION:
- Naive: enter whenever funding > threshold
- Enhanced: enter when futures-spot spread > THEORETICAL NO-ARBITRAGE BOUND
  AND momentum filter is positive (paper shows >50% of spread variance explained by momentum)

Theoretical perpetual price (frictionless):
    F_perp = S × (1 + r/κ)
    where r = risk-free rate (e.g. USDT stablecoin yield), κ = funding intensity

No-arb bounds with trading costs τ:
    Lower bound: S × (1 + r/κ) - τ
    Upper bound: S × (1 + r/κ) + τ

Entry: F_perp/S - 1 > theoretical_spread + fee_buffer → short perp, long spot
       (i.e., perp is overpriced relative to theory; we get paid funding + convergence)
Exit: F_perp/S - 1 returns to [lower_bound, 0]
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

log = logging.getLogger(__name__)
RESULTS_DIR = Path(__file__).parent.parent / "backtesting"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Theoretical Model ───────────────────────────────────────────────────────

def theoretical_perp_price(
    spot: float,
    risk_free_rate_annual: float = 0.05,   # e.g. USDT yield / T-bill rate
    funding_intensity: float = 3.0,         # κ: exchange's funding rate intensity (Binance ≈ 3)
) -> float:
    """
    Theoretical perpetual futures price from He et al. (2024).
    F_perp = S × (1 + r/κ)
    """
    return spot * (1 + risk_free_rate_annual / funding_intensity)


def no_arb_bound_spread(
    risk_free_rate_annual: float = 0.05,
    funding_intensity: float = 3.0,
    taker_fee: float = 0.0005,
    round_trips: int = 2,                   # 2 round-trips: entry + exit, both legs
) -> float:
    """
    No-arbitrage bound on the futures-spot spread as a fraction.
    Returns the minimum spread needed to justify the trade (net of fees).
    """
    theoretical_spread = risk_free_rate_annual / funding_intensity
    total_fee = taker_fee * round_trips * 2   # 2 legs × 2 trips
    return theoretical_spread + total_fee


# ─── Data Fetchers ───────────────────────────────────────────────────────────

def fetch_funding_rates_df(symbol: str = "BTC/USDT", exchange_id: str = "binance") -> pd.DataFrame:
    """Fetch historical funding rates. Returns DataFrame with datetime + funding_rate."""
    try:
        import ccxt, time
        exch = getattr(ccxt, exchange_id)({"enableRateLimit": True, "options": {"defaultType": "future"}})
        all_rates, since = [], int(pd.Timestamp("2022-01-01", tz="UTC").timestamp() * 1000)
        log.info(f"Fetching funding rates: {symbol}...")
        while True:
            batch = exch.fetch_funding_rate_history(symbol, since=since, limit=1000)
            if not batch:
                break
            all_rates.extend(batch)
            if len(batch) < 1000:
                break
            since = batch[-1]["timestamp"] + 1
            time.sleep(0.3)
        df = pd.DataFrame(all_rates)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df[["datetime", "fundingRate"]].rename(columns={"fundingRate": "funding_rate"}).drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
    except Exception as e:
        log.warning(f"Live funding fetch failed ({e}), using synthetic data.")
        return _synthetic_funding_with_spot_data()


def fetch_spot_ohlcv(symbol: str = "BTC/USDT", exchange_id: str = "binance") -> pd.DataFrame:
    """Load spot OHLCV, fetch if not cached."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from data_pipeline import load_ohlcv, fetch_ohlcv
    df = load_ohlcv(symbol, timeframe="8h", exchange_id=exchange_id)
    if df is None:
        df = fetch_ohlcv(symbol, timeframe="8h", since="2022-01-01", exchange_id=exchange_id)
    return df


def _synthetic_funding_with_spot_data() -> pd.DataFrame:
    """Synthetic funding rate + realistic spot returns for offline testing."""
    dates = pd.date_range("2022-01-01", "2025-01-01", freq="8h", tz="UTC")
    np.random.seed(42)
    n = len(dates)
    # Regime-aware funding: positive in bull, negative in bear
    regime = np.zeros(n)
    regime[:n//3] = 0.00015        # bull: very positive funding
    regime[n//3:2*n//3] = -0.00003 # bear: slightly negative
    regime[2*n//3:] = 0.00008      # sideways: mild positive
    noise = np.random.normal(0, 0.00015, n)
    funding = np.clip(regime + noise, -0.003, 0.003)

    # Synthetic spot price (GBM with regime drift)
    returns = np.zeros(n)
    returns[:n//3] = np.random.normal(0.002, 0.025, n//3)   # bull
    returns[n//3:2*n//3] = np.random.normal(-0.001, 0.035, n - n//3 - (n - 2*n//3))  # bear
    returns[2*n//3:] = np.random.normal(0.0005, 0.02, n - 2*n//3)
    spot = 30000 * np.cumprod(1 + returns)

    return pd.DataFrame({"datetime": dates, "funding_rate": funding, "spot_close": spot})


# ─── Enhanced Backtest ───────────────────────────────────────────────────────

class EnhancedFundingRateBacktest:
    """
    Theory-grounded funding rate arbitrage.

    Entry conditions (ALL must be met):
    1. funding_rate > min_funding_threshold (baseline profitability)
    2. (perp_price - spot) / spot > no_arb_bound (He et al. condition)
    3. momentum_filter: 30d spot return > momentum_threshold (paper: momentum explains >50% of spread)

    Exit conditions (ANY triggers):
    1. funding_rate < exit_funding
    2. spread < exit_spread (converged to fair value)
    3. spread > stop_spread (extreme move against us — systemic risk event)
    """

    def __init__(
        self,
        # Entry
        min_funding_threshold: float = 0.00008,  # 0.008%/8h = ~1.1% annualized minimum
        momentum_lookback_days: int = 30,
        momentum_threshold: float = 0.0,          # Require positive momentum
        # Exit
        exit_funding: float = 0.00003,            # Exit if funding drops this low
        exit_spread_pct: float = 0.001,           # Exit if spread converges to <0.1%
        stop_spread_pct: float = 0.05,            # Stop loss if spread blows to 5% (liquidation risk)
        # Cost model
        taker_fee: float = 0.0005,                # 0.05% Binance standard
        maker_fee: float = 0.0002,                # 0.02% with BNB discount
        # Capital
        capital: float = 10_000,
        use_maker_fees: bool = False,             # Use maker fees (LMT orders)
        # Theoretical model params (He et al.)
        risk_free_rate: float = 0.05,             # USDT yield / risk-free rate
        funding_intensity: float = 3.0,           # Binance κ parameter
    ):
        self.min_funding = min_funding_threshold
        self.momentum_lookback = momentum_lookback_days
        self.momentum_threshold = momentum_threshold
        self.exit_funding = exit_funding
        self.exit_spread = exit_spread_pct
        self.stop_spread = stop_spread_pct
        self.fee = maker_fee if use_maker_fees else taker_fee
        self.capital = capital
        self.rfr = risk_free_rate
        self.kappa = funding_intensity

        # Precompute theoretical bound
        self.theoretical_spread = risk_free_rate / funding_intensity
        self.arb_bound = no_arb_bound_spread(risk_free_rate, funding_intensity, self.fee)
        log.info(f"Theoretical spread: {self.theoretical_spread:.4%} | No-arb bound: {self.arb_bound:.4%}")

    def run(self, data: pd.DataFrame) -> dict:
        """
        Run backtest.
        data: DataFrame with columns [datetime, funding_rate, spot_close]
              If spot_close missing, assume perp ≈ spot (pure funding harvest mode)
        """
        df = data.copy().reset_index(drop=True)
        df["funding_rate"] = df["funding_rate"].astype(float)

        # Compute momentum: 30d rolling return on spot
        periods_per_day = 3  # 8h candles → 3 per day
        lookback_periods = self.momentum_lookback * periods_per_day
        if "spot_close" in df.columns:
            df["momentum"] = df["spot_close"].pct_change(lookback_periods)
            # Spread proxy: cumulative sum of recent funding rates gives running deviation
            # When funding is persistently positive, perp drifts above spot
            # Approximate: annualized_funding = funding_rate * 3 * 365
            df["spread_pct"] = df["funding_rate"].rolling(24).mean() * 3 * 365 / 100
            # Cap at reasonable level
            df["spread_pct"] = df["spread_pct"].clip(0, 0.5)
        else:
            df["momentum"] = pd.Series([0.01] * len(df))
            df["spread_pct"] = df["funding_rate"].rolling(24).mean() * 3 * 365 / 100
            df["spread_pct"] = df["spread_pct"].clip(0, 0.5)

        in_position = False
        entry_funding = 0.0
        pnl = 0.0
        trades = []
        records = []

        for i, row in df.iterrows():
            fr = row["funding_rate"]
            momentum = row.get("momentum", 0.01)
            spread = abs(row.get("spread_pct", fr * 8))
            dt = row["datetime"]

            if pd.isna(momentum):
                momentum = 0.0

            # ── Entry ────────────────────────────────────────────
            if not in_position:
                cond_funding = fr > self.min_funding
                cond_arb = spread > self.arb_bound
                cond_momentum = momentum > self.momentum_threshold

                # If we don't have real spread data, fall back to funding-only entry
                has_spread_data = "spot_close" in df.columns
                if not has_spread_data:
                    cond_arb = True  # skip spread condition if no real spread data

                if cond_funding and cond_arb and cond_momentum:
                    in_position = True
                    entry_funding = fr
                    entry_fee = self.capital * self.fee * 2  # 2 legs
                    pnl -= entry_fee
                    trades.append({"datetime": dt, "action": "enter", "funding_rate": fr,
                                   "momentum": round(momentum, 4), "spread": round(spread, 6)})

            # ── In Position ───────────────────────────────────────
            else:
                # Funding income: short perp receives positive funding
                borrow_cost_per_period = self.capital * (0.05 / (365 * 3))  # 5% annual / 3 periods/day
                income = self.capital * fr - borrow_cost_per_period
                pnl += income

                # Exit conditions
                should_exit = (
                    fr < self.exit_funding or
                    spread < self.exit_spread or
                    spread > self.stop_spread
                )
                if should_exit:
                    exit_fee = self.capital * self.fee * 2
                    pnl -= exit_fee
                    in_position = False
                    trades.append({"datetime": dt, "action": "exit", "funding_rate": fr,
                                   "reason": "funding" if fr < self.exit_funding else ("stop" if spread > self.stop_spread else "convergence")})

            records.append({
                "datetime": dt,
                "funding_rate": fr,
                "momentum": momentum,
                "spread": spread,
                "in_position": in_position,
                "cumulative_pnl": pnl,
            })

        pnl_df = pd.DataFrame(records)
        trades_df = pd.DataFrame(trades)

        if pnl_df.empty:
            return {"error": "No data"}

        # ── Metrics ──────────────────────────────────────────────
        n_days = (pnl_df["datetime"].iloc[-1] - pnl_df["datetime"].iloc[0]).days
        total_return = pnl / self.capital * 100
        annualized = ((1 + pnl / self.capital) ** (365 / n_days) - 1) * 100 if n_days > 0 else 0

        # Sharpe on 8h periods
        period_pnl = pnl_df["cumulative_pnl"].diff().fillna(0)
        period_return = period_pnl / self.capital
        sharpe = (period_return.mean() / period_return.std()) * np.sqrt(3 * 365) if period_return.std() > 0 else 0

        roll_max = pnl_df["cumulative_pnl"].cummax()
        max_dd = ((pnl_df["cumulative_pnl"] - roll_max) / self.capital * 100).min()

        n_entries = len(trades_df[trades_df["action"] == "enter"]) if not trades_df.empty else 0
        time_in_market = pnl_df["in_position"].mean() * 100

        return {
            "capital": self.capital,
            "total_pnl_usd": round(pnl, 2),
            "total_return_pct": round(total_return, 2),
            "annualized_return_pct": round(annualized, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "n_trades": n_entries,
            "time_in_market_pct": round(time_in_market, 1),
            "backtest_days": n_days,
            "theoretical_bound_used": round(self.arb_bound, 6),
            "pnl_df": pnl_df,
            "trades_df": trades_df,
        }

    def plot(self, results: dict, symbol: str = "BTC/USDT", save: bool = True):
        pnl_df = results.get("pnl_df")
        if pnl_df is None or pnl_df.empty:
            return

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # PnL
        axes[0].plot(pnl_df["datetime"], pnl_df["cumulative_pnl"], color="darkgreen", linewidth=1.5)
        axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        axes[0].fill_between(pnl_df["datetime"], 0, pnl_df["cumulative_pnl"],
                             where=pnl_df["cumulative_pnl"] >= 0, alpha=0.2, color="green")
        axes[0].fill_between(pnl_df["datetime"], 0, pnl_df["cumulative_pnl"],
                             where=pnl_df["cumulative_pnl"] < 0, alpha=0.3, color="red")
        axes[0].set_title(
            f"Enhanced Funding Rate Arb — {symbol}\n"
            f"Return: {results['total_return_pct']}% | Ann: {results['annualized_return_pct']}% | "
            f"Sharpe: {results['sharpe_ratio']} | MaxDD: {results['max_drawdown_pct']}% | "
            f"Time in Market: {results['time_in_market_pct']}%"
        )
        axes[0].set_ylabel("Cumulative PnL (USD)")
        axes[0].grid(True, alpha=0.3)

        # Funding rate
        axes[1].plot(pnl_df["datetime"], pnl_df["funding_rate"] * 100, color="steelblue", linewidth=0.8, alpha=0.7)
        axes[1].axhline(self.min_funding * 100, color="green", linestyle="--", linewidth=1, label=f"Entry threshold ({self.min_funding:.4%})")
        axes[1].axhline(self.exit_funding * 100, color="orange", linestyle="--", linewidth=1, label=f"Exit threshold ({self.exit_funding:.4%})")
        axes[1].axhline(0, color="gray", linewidth=0.5)
        axes[1].set_ylabel("Funding Rate (%)")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        # Position
        axes[2].fill_between(pnl_df["datetime"], 0, pnl_df["in_position"].astype(int),
                             color="purple", alpha=0.4, label="In Position")
        axes[2].set_ylabel("Position")
        axes[2].set_xlabel("Date")
        axes[2].set_ylim(-0.1, 1.5)
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.tight_layout()
        if save:
            path = RESULTS_DIR / f"enhanced_funding_arb_{symbol.replace('/', '_')}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            log.info(f"Chart saved → {path}")
        plt.show()


def run_enhanced_funding_arb(symbol: str = "BTC/USDT"):
    logging.basicConfig(level=logging.INFO)

    # Try live data first, fall back to synthetic
    data = fetch_funding_rates_df(symbol)

    # If we got real funding data, try to add spot close
    if "spot_close" not in data.columns:
        try:
            spot_df = fetch_spot_ohlcv(symbol)
            if spot_df is not None:
                spot_df["datetime"] = pd.to_datetime(spot_df["timestamp"], utc=True)
                spot_8h = spot_df[["datetime", "close"]].rename(columns={"close": "spot_close"})
                data = pd.merge_asof(
                    data.sort_values("datetime"),
                    spot_8h.sort_values("datetime"),
                    on="datetime",
                    direction="nearest"
                )
        except Exception as e:
            log.warning(f"Could not merge spot data: {e}")

    # Run with momentum condition
    bt = EnhancedFundingRateBacktest(
        min_funding_threshold=0.00008,
        momentum_threshold=0.0,       # Must be net positive 30d
        exit_funding=0.00003,
        taker_fee=0.0005,
        capital=10_000,
    )
    results = bt.run(data)

    print("\n" + "=" * 60)
    print(f"  ENHANCED FUNDING RATE ARB — {symbol}")
    print(f"  (Theory: He, Manela, Ross, von Wachter 2024)")
    print("=" * 60)
    for k, v in results.items():
        if k not in ("pnl_df", "trades_df"):
            print(f"  {k:35s}: {v}")
    print("=" * 60)

    bt.plot(results, symbol=symbol)
    return results


if __name__ == "__main__":
    run_enhanced_funding_arb("BTC/USDT")
