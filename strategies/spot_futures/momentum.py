"""
Cross-Sectional Momentum Strategy
===================================
Classic quant: rank assets by recent return, long top decile, short bottom decile.
Rebalance weekly or monthly.

Adapted for crypto: rank by 4-week return, hold 1 week.
Backtest on a universe of major coins.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

log = logging.getLogger(__name__)
RESULTS_DIR = Path(__file__).parent.parent / "backtesting"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_universe(symbols: list[str], timeframe: str = "1d", exchange_id: str = "binance") -> pd.DataFrame:
    """
    Load OHLCV for a universe of coins and return a close price DataFrame.
    Attempts to load from cache; fetches if not available.
    """
    import sys
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path: sys.path.insert(0, project_root)
    from scripts.data_pipeline import load_ohlcv, fetch_ohlcv

    closes = {}
    for symbol in symbols:
        df = load_ohlcv(symbol, timeframe=timeframe, exchange_id=exchange_id)
        if df is None:
            log.info(f"Fetching {symbol}...")
            df = fetch_ohlcv(symbol, timeframe=timeframe, since="2022-01-01", exchange_id=exchange_id)
        if df is not None and len(df) > 100:
            closes[symbol] = df.set_index("timestamp")["close"]
        else:
            log.warning(f"Skipping {symbol}: insufficient data")

    if not closes:
        raise ValueError("No data loaded for universe")

    price_df = pd.DataFrame(closes)
    price_df = price_df.ffill().dropna(how="all")
    return price_df


class MomentumBacktest:
    """
    Cross-sectional momentum strategy.

    At each rebalance:
    1. Rank assets by lookback_days return
    2. Long top N assets
    3. (Optional) Short bottom N assets
    4. Equal weight within each leg
    """

    def __init__(
        self,
        lookback_days: int = 28,        # Return measurement window
        hold_days: int = 7,             # Rebalance frequency
        top_n: int = 3,                 # # of longs
        short_n: int = 0,               # # of shorts (0 = long-only)
        capital: float = 10_000,
        taker_fee: float = 0.0005,      # 0.05% taker fee
        skip_days: int = 1,             # Skip most recent day (avoid reversal)
    ):
        self.lookback_days = lookback_days
        self.hold_days = hold_days
        self.top_n = top_n
        self.short_n = short_n
        self.capital = capital
        self.taker_fee = taker_fee
        self.skip_days = skip_days

    def run(self, price_df: pd.DataFrame) -> dict:
        """Run momentum backtest on price DataFrame (rows=dates, cols=symbols)."""
        price_df = price_df.copy()
        returns = price_df.pct_change()

        portfolio_value = self.capital
        portfolio_series = []
        rebalance_log = []

        # Rebalance dates
        rebalance_dates = price_df.index[self.lookback_days::self.hold_days]
        prev_holdings = {}

        for i, date in enumerate(rebalance_dates):
            # Get price at this rebalance
            if date not in price_df.index:
                continue
            loc = price_df.index.get_loc(date)
            if loc < self.lookback_days:
                continue

            # Compute lookback return (skip last skip_days)
            end_loc = loc - self.skip_days
            start_loc = end_loc - self.lookback_days
            if start_loc < 0:
                continue

            start_prices = price_df.iloc[start_loc]
            end_prices = price_df.iloc[end_loc]
            momentum = (end_prices - start_prices) / start_prices
            momentum = momentum.dropna()

            if len(momentum) < self.top_n:
                continue

            ranked = momentum.sort_values(ascending=False)
            longs = ranked.head(self.top_n).index.tolist()
            shorts = ranked.tail(self.short_n).index.tolist() if self.short_n > 0 else []

            # Transaction costs: fees on positions changed
            n_changes = len(set(longs) - set(prev_holdings.keys())) + len(set(prev_holdings.keys()) - set(longs))
            fee_cost = n_changes * (portfolio_value / max(self.top_n, 1)) * self.taker_fee
            portfolio_value -= fee_cost

            # Hold until next rebalance
            if i + 1 < len(rebalance_dates):
                next_date = rebalance_dates[i + 1]
            else:
                next_date = price_df.index[-1]

            hold_slice = price_df.loc[date:next_date]
            if len(hold_slice) < 2:
                continue

            entry_prices = price_df.loc[date, longs]
            exit_prices = price_df.loc[next_date, longs] if next_date in price_df.index else hold_slice.iloc[-1][longs]

            long_returns = ((exit_prices - entry_prices) / entry_prices).mean()
            period_return = long_returns

            portfolio_value *= (1 + period_return)

            portfolio_series.append({"date": date, "portfolio_value": portfolio_value, "longs": longs, "period_return": period_return})
            prev_holdings = {s: True for s in longs}

            rebalance_log.append({
                "date": date,
                "longs": ", ".join(longs),
                "period_return_pct": round(period_return * 100, 3),
                "portfolio_value": round(portfolio_value, 2),
            })

        if not portfolio_series:
            return {"error": "No rebalance periods generated"}

        port_df = pd.DataFrame(portfolio_series)
        port_df["return"] = port_df["portfolio_value"].pct_change().fillna(0)

        # --- Metrics ---
        total_return = (portfolio_value - self.capital) / self.capital * 100
        n_days = (port_df["date"].iloc[-1] - port_df["date"].iloc[0]).days
        annualized = ((portfolio_value / self.capital) ** (365 / n_days) - 1) * 100 if n_days > 0 else 0
        sharpe = (port_df["return"].mean() / port_df["return"].std()) * np.sqrt(52) if port_df["return"].std() > 0 else 0  # weekly
        roll_max = port_df["portfolio_value"].cummax()
        drawdown = (port_df["portfolio_value"] - roll_max) / roll_max
        max_drawdown = drawdown.min() * 100

        return {
            "capital": self.capital,
            "final_value": round(portfolio_value, 2),
            "total_return_pct": round(total_return, 2),
            "annualized_return_pct": round(annualized, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_drawdown, 2),
            "n_rebalances": len(portfolio_series),
            "backtest_days": n_days,
            "port_df": port_df,
            "rebalance_log": pd.DataFrame(rebalance_log),
        }

    def plot(self, results: dict, save: bool = True):
        port_df = results.get("port_df")
        if port_df is None or port_df.empty:
            return
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(port_df["date"], port_df["portfolio_value"], color="royalblue", linewidth=1.5, label="Portfolio")
        ax.axhline(self.capital, color="gray", linestyle="--", linewidth=0.8, label="Initial Capital")
        ax.fill_between(port_df["date"], self.capital, port_df["portfolio_value"],
                        where=port_df["portfolio_value"] >= self.capital, alpha=0.2, color="green")
        ax.fill_between(port_df["date"], self.capital, port_df["portfolio_value"],
                        where=port_df["portfolio_value"] < self.capital, alpha=0.2, color="red")
        ax.set_title(f"Momentum Strategy | Return: {results['total_return_pct']}% | Ann: {results['annualized_return_pct']}% | Sharpe: {results['sharpe_ratio']} | MaxDD: {results['max_drawdown_pct']}%")
        ax.set_ylabel("Portfolio Value (USD)")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.tight_layout()
        if save:
            path = RESULTS_DIR / "momentum_strategy.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            log.info(f"Chart saved → {path}")
        plt.show()


def run_momentum(symbols: Optional[list[str]] = None, timeframe: str = "1d"):
    if symbols is None:
        symbols = [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
            "ADA/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
            "MATIC/USDT", "ATOM/USDT",
        ]

    price_df = load_universe(symbols, timeframe=timeframe)
    bt = MomentumBacktest(lookback_days=28, hold_days=7, top_n=3, capital=10_000)
    results = bt.run(price_df)

    print("\n" + "=" * 50)
    print("  CROSS-SECTIONAL MOMENTUM STRATEGY")
    print("=" * 50)
    for k, v in results.items():
        if k not in ("port_df", "rebalance_log"):
            print(f"  {k:30s}: {v}")
    print("=" * 50)

    if "rebalance_log" in results and not results["rebalance_log"].empty:
        print("\nLast 5 rebalances:")
        print(results["rebalance_log"].tail())

    bt.plot(results)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_momentum()
