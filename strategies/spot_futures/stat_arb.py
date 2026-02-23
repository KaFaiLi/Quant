"""
Pairs Stat-Arb — Dollar-Neutral, Mark-to-Market PnL
=====================================================
Fixes over earlier version:
1. Continuous mark-to-market: cumulative_pnl reflects unrealized while in position
2. Capital floor: halt new trades if portfolio < 20% of starting capital
3. Half-life guard: clamp to finite values
4. Dollar-neutral sizing per leg (not price-ratio based)
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


def compute_half_life(spread: pd.Series) -> float:
    try:
        import statsmodels.api as sm
        lag = spread.shift(1).dropna()
        delta = (spread - spread.shift(1)).dropna()
        common = lag.index.intersection(delta.index)
        model = sm.OLS(delta.loc[common], sm.add_constant(lag.loc[common])).fit()
        lam = model.params.iloc[1]
        return float(-np.log(2) / lam) if lam < 0 else 60.0
    except Exception:
        return 40.0


class PairsStatArbBacktest:
    def __init__(
        self,
        entry_z: float = 1.5,
        exit_z: float = 0.3,
        stop_z: float = 3.5,
        zscore_window: int = 40,
        hedge_window: int = 90,
        refit_interval: int = 20,
        capital: float = 10_000,
        fee: float = 0.0004,
        capital_floor: float = 0.20,        # halt trading if equity < 20% of start
        btc_trend_window: int = 30,
        btc_trend_threshold: float = -0.10,
    ):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.zscore_window = zscore_window
        self.hedge_window = hedge_window
        self.refit_interval = refit_interval
        self.capital = capital
        self.fee = fee
        self.capital_floor = capital_floor
        self.btc_trend_window = btc_trend_window
        self.btc_trend_threshold = btc_trend_threshold

    def run(
        self,
        price_y: pd.Series,
        price_x: pd.Series,
        pair_name: str = "Pair",
        btc_prices: Optional[pd.Series] = None,
    ) -> dict:
        df = pd.DataFrame({"y": price_y, "x": price_x}).dropna()
        if btc_prices is not None:
            df = df.join(btc_prices.rename("btc"), how="left")
        else:
            df["btc"] = np.nan

        min_rows = self.hedge_window + self.zscore_window + 20
        if len(df) < min_rows:
            return {"error": f"Insufficient data: {len(df)} rows"}

        # Cointegration check on first half
        try:
            from statsmodels.tsa.stattools import coint
            half = len(df) // 2
            _, coint_pval, _ = coint(df["y"].iloc[:half], df["x"].iloc[:half])
        except Exception:
            coint_pval = None

        hedge_ratio = 1.0
        intercept_val = 0.0
        position = 0          # 0 = flat, 1 = long spread, -1 = short spread
        entry_y = entry_x = None
        equity = self.capital  # running equity (realised only)
        fee_total = 0.0
        records = []
        trades = []

        for i in range(self.hedge_window, len(df)):
            curr_y = df["y"].iloc[i]
            curr_x = df["x"].iloc[i]
            dt = df.index[i]

            # Refit hedge
            if (i - self.hedge_window) % self.refit_interval == 0:
                window = df.iloc[max(0, i - self.hedge_window): i]
                sl, ic, _, _, _ = stats.linregress(window["x"], window["y"])
                hedge_ratio = sl
                intercept_val = ic
                spread_w = window["y"] - (hedge_ratio * window["x"] + intercept_val)
                hl = compute_half_life(spread_w)
                hl = min(max(hl, 10.0), 120.0)   # clamp to [10, 120] days
                self.zscore_window = max(20, min(80, int(hl * 1.5)))

            # Z-score
            hist = df.iloc[max(0, i - self.zscore_window): i + 1]
            spread_hist = hist["y"] - hedge_ratio * hist["x"]
            spread_now = curr_y - hedge_ratio * curr_x
            s_mean = spread_hist.mean()
            s_std = spread_hist.std()
            z = (spread_now - s_mean) / s_std if s_std > 0 else 0.0

            # Mark-to-market unrealized
            unrealized = 0.0
            if position != 0 and entry_y is not None:
                leg = self.capital / 2
                units_y = leg / entry_y
                units_x = leg / entry_x
                if position == 1:
                    unrealized = units_y * (curr_y - entry_y) - units_x * (curr_x - entry_x)
                else:
                    unrealized = -units_y * (curr_y - entry_y) + units_x * (curr_x - entry_x)

            total_equity = equity + unrealized

            # Capital floor: close and halt if equity too low
            if total_equity < self.capital * self.capital_floor:
                if position != 0:
                    fee_exit = (self.capital / 2) * self.fee * 2
                    equity += unrealized - fee_exit
                    fee_total += fee_exit
                    position = 0
                    entry_y = entry_x = None
                records.append({"datetime": dt, "z_score": z, "position": 0,
                                 "cumulative_pnl": equity - self.capital})
                continue

            # BTC regime
            btc_ok = True
            if not np.isnan(df["btc"].iloc[i]) and i >= self.btc_trend_window:
                btc_ret = (df["btc"].iloc[i] / df["btc"].iloc[i - self.btc_trend_window]) - 1
                if btc_ret < self.btc_trend_threshold:
                    btc_ok = False

            # Entry
            if position == 0 and btc_ok:
                leg = self.capital / 2
                fee_entry = leg * self.fee * 2
                if equity - fee_entry > self.capital * self.capital_floor:
                    if z < -self.entry_z:
                        position = 1
                        entry_y, entry_x = curr_y, curr_x
                        equity -= fee_entry
                        fee_total += fee_entry
                        trades.append({"dt": dt, "action": "enter_long", "z": round(z, 3)})
                    elif z > self.entry_z:
                        position = -1
                        entry_y, entry_x = curr_y, curr_x
                        equity -= fee_entry
                        fee_total += fee_entry
                        trades.append({"dt": dt, "action": "enter_short", "z": round(z, 3)})

            # Exit
            elif position != 0:
                should_exit = abs(z) < self.exit_z or abs(z) > self.stop_z or not btc_ok
                if should_exit:
                    fee_exit = (self.capital / 2) * self.fee * 2
                    equity += unrealized - fee_exit
                    fee_total += fee_exit
                    reason = "mean_rev" if abs(z) < self.exit_z else ("stop" if abs(z) > self.stop_z else "regime")
                    trades.append({"dt": dt, "action": "exit", "z": round(z, 3),
                                   "reason": reason, "trade_pnl": round(unrealized - fee_exit, 2)})
                    position = 0
                    entry_y = entry_x = None
                    unrealized = 0.0

            records.append({
                "datetime": dt,
                "z_score": z,
                "position": position,
                "cumulative_pnl": equity - self.capital + unrealized,
            })

        if not records:
            return {"error": "No records"}

        pnl_df = pd.DataFrame(records)
        n_days = (pnl_df["datetime"].iloc[-1] - pnl_df["datetime"].iloc[0]).days
        final_pnl = equity - self.capital
        total_return = final_pnl / self.capital * 100
        annualized = ((1 + final_pnl / self.capital) ** (365 / n_days) - 1) * 100 if n_days > 0 and (1 + final_pnl / self.capital) > 0 else float('nan')
        daily_pnl = pnl_df["cumulative_pnl"].diff().fillna(0)
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0
        roll_max = pnl_df["cumulative_pnl"].cummax()
        max_dd = ((pnl_df["cumulative_pnl"] - roll_max) / self.capital * 100).min()
        n_trades = len([t for t in trades if "enter" in t.get("action", "")])

        return {
            "pair": pair_name,
            "capital": self.capital,
            "total_pnl_usd": round(final_pnl, 2),
            "total_return_pct": round(total_return, 2),
            "annualized_return_pct": round(annualized, 2) if np.isfinite(annualized) else "N/A",
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "total_fees_usd": round(fee_total, 2),
            "n_trades": n_trades,
            "cointegration_pval": round(coint_pval, 4) if coint_pval is not None else "N/A",
            "backtest_days": n_days,
            "pnl_df": pnl_df,
        }

    def plot(self, results: dict, save: bool = True):
        pnl_df = results.get("pnl_df")
        if pnl_df is None or pnl_df.empty:
            return
        pair = results.get("pair", "Pair")
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(pnl_df["datetime"], pnl_df["cumulative_pnl"], color="navy")
        axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        axes[0].fill_between(pnl_df["datetime"], 0, pnl_df["cumulative_pnl"],
                             where=pnl_df["cumulative_pnl"] >= 0, alpha=0.2, color="green")
        axes[0].fill_between(pnl_df["datetime"], 0, pnl_df["cumulative_pnl"],
                             where=pnl_df["cumulative_pnl"] < 0, alpha=0.2, color="red")
        axes[0].set_title(
            f"Stat-Arb (Dollar-Neutral) — {pair} | "
            f"Return: {results['total_return_pct']}% | Sharpe: {results['sharpe_ratio']} | "
            f"MaxDD: {results['max_drawdown_pct']}%"
        )
        axes[0].set_ylabel("Cum. PnL (USD)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(pnl_df["datetime"], pnl_df["z_score"], color="purple", linewidth=0.8)
        axes[1].axhline(self.entry_z, color="red", linestyle="--", linewidth=0.8)
        axes[1].axhline(-self.entry_z, color="green", linestyle="--", linewidth=0.8)
        axes[1].axhline(0, color="gray", linewidth=0.5)
        axes[1].set_ylabel("Z-Score")
        axes[1].set_xlabel("Date")
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.tight_layout()
        if save:
            safe = pair.replace("/", "_").replace(" ", "_")
            path = RESULTS_DIR / f"stat_arb_{safe}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            log.info(f"Chart saved → {path}")
        plt.show()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from scripts.data_pipeline import load_ohlcv

    prices = {}
    for s in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'LINK/USDT']:
        df = load_ohlcv(s, '1d', 'okx')
        if df is not None:
            prices[s] = df.set_index('timestamp')['close']

    btc = prices.get('BTC/USDT')
    for sym_y, sym_x in [('ADA/USDT', 'LINK/USDT'), ('SOL/USDT', 'LINK/USDT'), ('ETH/USDT', 'LINK/USDT')]:
        bt = PairsStatArbBacktest(entry_z=1.5, exit_z=0.3, capital=10_000)
        r = bt.run(prices[sym_y], prices[sym_x], pair_name=f'{sym_y}/{sym_x}', btc_prices=btc)
        bt.plot(r)
        for k, v in r.items():
            if k != 'pnl_df':
                print(f"  {k}: {v}")
