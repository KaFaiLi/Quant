"""
Funding Rate Arbitrage Strategy
================================
Core idea: On perpetual futures, longs pay shorts (or vice versa) every 8h based on the funding rate.
When funding is consistently positive → shorts get paid → we short perp + long spot = delta-neutral, collect funding.

This backtest measures:
- Historical funding rate income
- Entry/exit logic based on funding threshold
- Estimated PnL after fees
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "backtesting"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def fetch_funding_rates(symbol: str = "BTC/USDT", exchange_id: str = "binance") -> pd.DataFrame:
    """
    Fetch historical funding rates via CCXT.
    Falls back to synthetic data if unavailable.
    """
    try:
        import ccxt
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        log.info(f"Fetching funding rates for {symbol} on {exchange_id}...")

        all_rates = []
        since = int(pd.Timestamp("2022-01-01", tz="UTC").timestamp() * 1000)

        while True:
            rates = exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
            if not rates:
                break
            all_rates.extend(rates)
            last_ts = rates[-1]["timestamp"]
            if len(rates) < 1000:
                break
            since = last_ts + 1
            import time; time.sleep(0.2)

        df = pd.DataFrame(all_rates)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df[["datetime", "fundingRate"]].rename(columns={"fundingRate": "funding_rate"})
        df = df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
        log.info(f"  Got {len(df)} funding rate records")
        return df

    except Exception as e:
        log.warning(f"Could not fetch live funding rates: {e}. Using synthetic data.")
        return _generate_synthetic_funding(symbol)


def _generate_synthetic_funding(symbol: str) -> pd.DataFrame:
    """Generate realistic synthetic funding rate data for backtesting."""
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="8h", tz="UTC")
    np.random.seed(42)

    # Funding rates: mean ~0.01% per 8h (0.03% daily), with regime shifts
    n = len(dates)
    base_rate = 0.0001  # 0.01% per 8h

    # Add regime: bull markets have higher positive funding
    regimes = np.zeros(n)
    regimes[: n // 3] = 0.00015   # bull
    regimes[n // 3 : 2 * n // 3] = -0.00005  # bear (negative funding common)
    regimes[2 * n // 3 :] = 0.0001  # sideways

    noise = np.random.normal(0, 0.0002, n)
    funding = regimes + noise
    funding = np.clip(funding, -0.003, 0.003)  # exchange caps

    return pd.DataFrame({"datetime": dates, "funding_rate": funding})


class FundingRateBacktest:
    """
    Delta-neutral funding rate harvesting backtest.

    Strategy:
    - Enter when 8h funding rate > entry_threshold (shorts get paid)
    - Hold until funding drops below exit_threshold or goes negative
    - Assume: spot long + perp short (delta neutral)
    - Costs: maker fee each side on entry/exit
    """

    def __init__(
        self,
        entry_threshold: float = 0.0001,   # Enter if funding > 0.01% per 8h
        exit_threshold: float = 0.00005,   # Exit if funding < 0.005% per 8h
        maker_fee: float = 0.0002,          # 0.02% maker fee (Binance)
        capital: float = 10_000,            # USD notional
        leverage: float = 1.0,              # 1x = no leverage (spot + perp)
        annual_borrow_cost: float = 0.05,   # 5% annual borrow for margin
    ):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.maker_fee = maker_fee
        self.capital = capital
        self.leverage = leverage
        self.annual_borrow_cost = annual_borrow_cost

    def run(self, funding_df: pd.DataFrame) -> dict:
        """Run backtest on funding rate DataFrame."""
        df = funding_df.copy().reset_index(drop=True)
        df["funding_rate"] = df["funding_rate"].astype(float)

        # --- State ---
        in_position = False
        trades = []
        pnl_series = []
        cumulative_pnl = 0.0
        entry_cost = self.capital * self.maker_fee * 2  # 2 legs

        for i, row in df.iterrows():
            fr = row["funding_rate"]
            dt = row["datetime"]

            if not in_position:
                if fr > self.entry_threshold:
                    in_position = True
                    entry_fee = entry_cost
                    cumulative_pnl -= entry_fee
                    trades.append({"datetime": dt, "action": "enter", "funding_rate": fr, "fee": entry_fee})
            else:
                # Collect funding: we're short perp, so we receive funding when positive
                funding_income = self.capital * fr
                # Borrow cost for holding margin (prorated per 8h)
                borrow_cost = self.capital * (self.annual_borrow_cost / (365 * 3))
                net = funding_income - borrow_cost
                cumulative_pnl += net
                pnl_series.append({"datetime": dt, "funding_income": funding_income, "borrow_cost": borrow_cost, "net": net, "cumulative_pnl": cumulative_pnl})

                if fr < self.exit_threshold:
                    in_position = False
                    exit_fee = entry_cost
                    cumulative_pnl -= exit_fee
                    trades.append({"datetime": dt, "action": "exit", "funding_rate": fr, "fee": exit_fee})

        pnl_df = pd.DataFrame(pnl_series) if pnl_series else pd.DataFrame()
        trades_df = pd.DataFrame(trades)

        # --- Metrics ---
        if pnl_df.empty:
            return {"error": "No positions taken"}

        total_pnl = cumulative_pnl
        total_return_pct = (total_pnl / self.capital) * 100
        n_days = (df["datetime"].iloc[-1] - df["datetime"].iloc[0]).days
        annualized_return = ((1 + total_pnl / self.capital) ** (365 / n_days) - 1) * 100

        # Sharpe
        pnl_df["net_return"] = pnl_df["net"] / self.capital
        sharpe = (pnl_df["net_return"].mean() / pnl_df["net_return"].std()) * np.sqrt(3 * 365) if pnl_df["net_return"].std() > 0 else 0

        # Max drawdown
        cum = pnl_df["cumulative_pnl"]
        roll_max = cum.cummax()
        drawdown = (cum - roll_max)
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / self.capital) * 100

        n_entries = len(trades_df[trades_df["action"] == "enter"]) if not trades_df.empty else 0

        results = {
            "capital": self.capital,
            "total_pnl_usd": round(total_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "annualized_return_pct": round(annualized_return, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_usd": round(max_drawdown, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "n_trades": n_entries,
            "backtest_days": n_days,
            "pnl_df": pnl_df,
            "trades_df": trades_df,
        }
        return results

    def plot(self, results: dict, symbol: str = "BTC/USDT", save: bool = True):
        pnl_df = results["pnl_df"]
        if pnl_df.empty:
            print("No data to plot.")
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Cumulative PnL
        axes[0].plot(pnl_df["datetime"], pnl_df["cumulative_pnl"], color="green", linewidth=1.5)
        axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        axes[0].fill_between(pnl_df["datetime"], pnl_df["cumulative_pnl"], 0,
                             where=pnl_df["cumulative_pnl"] >= 0, alpha=0.2, color="green")
        axes[0].fill_between(pnl_df["datetime"], pnl_df["cumulative_pnl"], 0,
                             where=pnl_df["cumulative_pnl"] < 0, alpha=0.2, color="red")
        axes[0].set_ylabel("Cumulative PnL (USD)")
        axes[0].set_title(f"Funding Rate Arb — {symbol} | Return: {results['total_return_pct']}% | Sharpe: {results['sharpe_ratio']} | MaxDD: {results['max_drawdown_pct']}%")
        axes[0].grid(True, alpha=0.3)

        # Net income per period
        axes[1].bar(pnl_df["datetime"], pnl_df["net"], color=["green" if x >= 0 else "red" for x in pnl_df["net"]], alpha=0.6, width=0.25)
        axes[1].set_ylabel("Net Income per 8h (USD)")
        axes[1].set_xlabel("Date")
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.tight_layout()
        if save:
            path = RESULTS_DIR / f"funding_arb_{symbol.replace('/', '_')}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            log.info(f"Chart saved → {path}")
        plt.show()


def run_funding_arb(symbol: str = "BTC/USDT", exchange_id: str = "binance"):
    funding_df = fetch_funding_rates(symbol, exchange_id)

    bt = FundingRateBacktest(
        entry_threshold=0.0001,
        exit_threshold=0.00005,
        maker_fee=0.0002,
        capital=10_000,
    )
    results = bt.run(funding_df)

    print("\n" + "=" * 50)
    print(f"  FUNDING RATE ARB — {symbol}")
    print("=" * 50)
    for k, v in results.items():
        if k not in ("pnl_df", "trades_df"):
            print(f"  {k:30s}: {v}")
    print("=" * 50)

    bt.plot(results, symbol=symbol)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_funding_arb("BTC/USDT")
