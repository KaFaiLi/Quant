"""
Pairs Mean Reversion (Stat-Arb) Strategy
==========================================
Core idea: Find cointegrated pairs (e.g. BTC/ETH), model the spread,
trade when spread is extreme (Z-score > threshold), exit at mean.

Steps:
1. Test cointegration (Engle-Granger)
2. Fit OLS hedge ratio
3. Compute spread Z-score
4. Long spread when Z < -entry_z, short when Z > entry_z
5. Exit when Z crosses 0 (mean reversion)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

log = logging.getLogger(__name__)
RESULTS_DIR = Path(__file__).parent.parent / "backtesting"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def adf_test(series: pd.Series) -> Tuple[float, float, bool]:
    """Augmented Dickey-Fuller test for stationarity. Returns (stat, pvalue, is_stationary)."""
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(series.dropna(), autolag="AIC")
    return result[0], result[1], result[1] < 0.05


def cointegration_test(y: pd.Series, x: pd.Series) -> Tuple[float, float, bool]:
    """Engle-Granger cointegration test. Returns (stat, pvalue, is_cointegrated)."""
    from statsmodels.tsa.stattools import coint
    stat, pvalue, _ = coint(y, x)
    return stat, pvalue, pvalue < 0.05


def fit_hedge_ratio(y: pd.Series, x: pd.Series) -> Tuple[float, float]:
    """OLS regression: y = beta*x + alpha. Returns (beta, alpha)."""
    slope, intercept, _, _, _ = stats.linregress(x, y)
    return slope, intercept


def compute_spread_zscore(y: pd.Series, x: pd.Series, hedge_ratio: float, alpha: float, window: int = 60) -> pd.Series:
    """Compute rolling Z-score of the spread."""
    spread = y - (hedge_ratio * x + alpha)
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    zscore = (spread - rolling_mean) / rolling_std
    return zscore, spread


class PairsMeanReversionBacktest:
    """
    Stat-arb pairs trading backtest.

    Signal:
    - Z > entry_z  → short the spread (short y, long x)
    - Z < -entry_z → long the spread (long y, short x)
    - |Z| < exit_z → close position
    """

    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 3.5,            # Stop loss if Z goes too extreme
        zscore_window: int = 60,        # Rolling window for Z-score (days)
        lookback_for_hedge: int = 120,  # Window to estimate hedge ratio
        capital: float = 10_000,
        fee: float = 0.0004,            # 0.04% per trade
        refit_interval: int = 30,       # Days between re-fitting hedge ratio
    ):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.zscore_window = zscore_window
        self.lookback_for_hedge = lookback_for_hedge
        self.capital = capital
        self.fee = fee
        self.refit_interval = refit_interval

    def run(self, price_y: pd.Series, price_x: pd.Series, pair_name: str = "Pair") -> dict:
        """
        Run backtest on two aligned price series.
        Returns metrics + PnL DataFrame.
        """
        # Align
        df = pd.DataFrame({"y": price_y, "x": price_x}).dropna()
        if len(df) < self.lookback_for_hedge + self.zscore_window + 50:
            return {"error": f"Insufficient data: {len(df)} rows"}

        # Test cointegration on full sample (for info)
        try:
            coint_stat, coint_pval, is_coint = cointegration_test(df["y"], df["x"])
        except ImportError:
            log.warning("statsmodels not available — skipping cointegration test")
            coint_pval, is_coint = None, True

        log.info(f"{pair_name}: Cointegrated={is_coint}, p={coint_pval:.4f}" if coint_pval else f"{pair_name}: Running without cointegration check")

        # Walk-forward: refit hedge ratio every refit_interval days
        hedge_ratio, alpha_coef = 1.0, 0.0
        records = []
        position = 0    # 0 = flat, 1 = long spread, -1 = short spread
        entry_price_y, entry_price_x = None, None
        entry_hedge = 1.0
        pnl = 0.0
        fee_total = 0.0

        for i in range(self.lookback_for_hedge, len(df)):
            if (i - self.lookback_for_hedge) % self.refit_interval == 0:
                # Refit on lookback window
                window = df.iloc[max(0, i - self.lookback_for_hedge): i]
                hedge_ratio, alpha_coef = fit_hedge_ratio(window["y"], window["x"])

            # Compute Z-score
            hist = df.iloc[max(0, i - self.zscore_window): i + 1]
            spread = hist["y"] - (hedge_ratio * hist["x"] + alpha_coef)
            z = (spread.iloc[-1] - spread.mean()) / spread.std() if spread.std() > 0 else 0

            curr_y = df["y"].iloc[i]
            curr_x = df["x"].iloc[i]
            dt = df.index[i]

            # Position management
            if position == 0:
                if z < -self.entry_z:
                    # Long spread: long y, short x
                    position = 1
                    entry_price_y = curr_y
                    entry_price_x = curr_x
                    entry_hedge = hedge_ratio
                    fee_cost = self.capital * self.fee * 2
                    pnl -= fee_cost
                    fee_total += fee_cost
                elif z > self.entry_z:
                    # Short spread: short y, long x
                    position = -1
                    entry_price_y = curr_y
                    entry_price_x = curr_x
                    entry_hedge = hedge_ratio
                    fee_cost = self.capital * self.fee * 2
                    pnl -= fee_cost
                    fee_total += fee_cost
            else:
                # Mark-to-market P&L
                if position == 1:
                    trade_pnl = (curr_y - entry_price_y) - entry_hedge * (curr_x - entry_price_x)
                else:
                    trade_pnl = -(curr_y - entry_price_y) + entry_hedge * (curr_x - entry_price_x)

                trade_pnl_normalized = trade_pnl / entry_price_y * self.capital * 0.5

                # Exit conditions
                should_exit = abs(z) < self.exit_z or abs(z) > self.stop_z
                if should_exit:
                    pnl += trade_pnl_normalized
                    fee_cost = self.capital * self.fee * 2
                    pnl -= fee_cost
                    fee_total += fee_cost
                    position = 0
                    entry_price_y = entry_price_x = None

            records.append({
                "datetime": dt,
                "z_score": z,
                "position": position,
                "cumulative_pnl": pnl,
                "y_price": curr_y,
                "x_price": curr_x,
            })

        if not records:
            return {"error": "No records generated"}

        pnl_df = pd.DataFrame(records)
        total_return = pnl / self.capital * 100
        n_days = (pnl_df["datetime"].iloc[-1] - pnl_df["datetime"].iloc[0]).days
        annualized = ((1 + pnl / self.capital) ** (365 / n_days) - 1) * 100 if n_days > 0 else 0

        daily_pnl = pnl_df["cumulative_pnl"].diff().fillna(0)
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0

        roll_max = pnl_df["cumulative_pnl"].cummax()
        max_dd = ((pnl_df["cumulative_pnl"] - roll_max) / self.capital * 100).min()

        n_trades = int((pnl_df["position"].diff().abs() > 0).sum() // 2)

        return {
            "pair": pair_name,
            "capital": self.capital,
            "total_pnl_usd": round(pnl, 2),
            "total_return_pct": round(total_return, 2),
            "annualized_return_pct": round(annualized, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "total_fees_usd": round(fee_total, 2),
            "n_trades": n_trades,
            "cointegration_pval": round(coint_pval, 4) if coint_pval else "N/A",
            "is_cointegrated": is_coint,
            "backtest_days": n_days,
            "pnl_df": pnl_df,
        }

    def plot(self, results: dict, save: bool = True):
        pnl_df = results.get("pnl_df")
        if pnl_df is None or pnl_df.empty:
            return

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        pair = results.get("pair", "Pair")

        # PnL
        axes[0].plot(pnl_df["datetime"], pnl_df["cumulative_pnl"], color="navy")
        axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        axes[0].set_ylabel("Cum. PnL (USD)")
        axes[0].set_title(f"Pairs Mean Reversion — {pair} | Return: {results['total_return_pct']}% | Sharpe: {results['sharpe_ratio']} | MaxDD: {results['max_drawdown_pct']}%")
        axes[0].grid(True, alpha=0.3)

        # Z-score
        axes[1].plot(pnl_df["datetime"], pnl_df["z_score"], color="purple", linewidth=0.8)
        axes[1].axhline(self.entry_z, color="red", linestyle="--", linewidth=0.8, label=f"+{self.entry_z}σ")
        axes[1].axhline(-self.entry_z, color="green", linestyle="--", linewidth=0.8, label=f"-{self.entry_z}σ")
        axes[1].axhline(0, color="gray", linestyle="-", linewidth=0.5)
        axes[1].set_ylabel("Z-Score")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        # Positions
        axes[2].fill_between(pnl_df["datetime"], pnl_df["position"],
                             where=pnl_df["position"] > 0, color="green", alpha=0.5, label="Long spread")
        axes[2].fill_between(pnl_df["datetime"], pnl_df["position"],
                             where=pnl_df["position"] < 0, color="red", alpha=0.5, label="Short spread")
        axes[2].set_ylabel("Position")
        axes[2].set_xlabel("Date")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.tight_layout()
        if save:
            safe_pair = pair.replace("/", "_").replace(" ", "_")
            path = RESULTS_DIR / f"pairs_mean_reversion_{safe_pair}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            log.info(f"Chart saved → {path}")
        plt.show()


def run_pairs(pair: Tuple[str, str] = ("BTC/USDT", "ETH/USDT"), timeframe: str = "1d"):
    import sys
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path: sys.path.insert(0, project_root)
    from scripts.data_pipeline import load_ohlcv, fetch_ohlcv

    sym_y, sym_x = pair
    for sym in [sym_y, sym_x]:
        df = load_ohlcv(sym, timeframe=timeframe)
        if df is None:
            fetch_ohlcv(sym, timeframe=timeframe, since="2022-01-01")

    df_y = load_ohlcv(sym_y, timeframe=timeframe)
    df_x = load_ohlcv(sym_x, timeframe=timeframe)

    if df_y is None or df_x is None:
        raise ValueError("Could not load price data")

    df_y = df_y.set_index("timestamp")["close"].rename(sym_y)
    df_x = df_x.set_index("timestamp")["close"].rename(sym_x)

    bt = PairsMeanReversionBacktest(entry_z=2.0, exit_z=0.5, capital=10_000)
    results = bt.run(df_y, df_x, pair_name=f"{sym_y} / {sym_x}")

    print("\n" + "=" * 50)
    print(f"  PAIRS MEAN REVERSION — {sym_y} / {sym_x}")
    print("=" * 50)
    for k, v in results.items():
        if k not in ("pnl_df",):
            print(f"  {k:30s}: {v}")
    print("=" * 50)

    bt.plot(results)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pairs(("BTC/USDT", "ETH/USDT"))
