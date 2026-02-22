"""
Data Pipeline — OHLCV fetcher via CCXT
Fetches historical candlestick data from any CCXT-supported exchange (default: Binance).
Saves to data/processed/ as CSV and optionally parquet.
"""

import os
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_exchange(exchange_id: str = "binance") -> ccxt.Exchange:
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    return exchange


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    since: Optional[str] = None,         # e.g. "2023-01-01"
    limit_per_call: int = 1000,
    exchange_id: str = "binance",
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch full OHLCV history for a symbol with pagination.
    Returns a DataFrame with columns: timestamp, open, high, low, close, volume
    """
    exchange = get_exchange(exchange_id)

    if since:
        since_ts = int(datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    else:
        # Default: 2 years back
        since_ts = int((datetime.now(timezone.utc).timestamp() - 2 * 365 * 86400) * 1000)

    log.info(f"Fetching {symbol} {timeframe} from {since or '2y ago'} on {exchange_id}...")

    all_candles = []
    current_since = since_ts

    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=current_since, limit=limit_per_call)
        if not candles:
            break
        all_candles.extend(candles)
        last_ts = candles[-1][0]
        log.info(f"  Fetched {len(candles)} candles, up to {datetime.fromtimestamp(last_ts/1000, tz=timezone.utc).date()}")

        if len(candles) < limit_per_call:
            break

        current_since = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    log.info(f"  Total: {len(df)} candles | {df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}")

    if save:
        safe_symbol = symbol.replace("/", "_")
        path = DATA_DIR / f"{exchange_id}_{safe_symbol}_{timeframe}.csv"
        df.to_csv(path, index=False)
        log.info(f"  Saved → {path}")

    return df


def load_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    exchange_id: str = "binance",
) -> Optional[pd.DataFrame]:
    """Load cached OHLCV data from disk."""
    safe_symbol = symbol.replace("/", "_")
    path = DATA_DIR / f"{exchange_id}_{safe_symbol}_{timeframe}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def fetch_multiple(
    symbols: list[str],
    timeframe: str = "1h",
    since: Optional[str] = None,
    exchange_id: str = "binance",
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple symbols, returns dict of symbol → DataFrame."""
    result = {}
    for symbol in symbols:
        try:
            df = fetch_ohlcv(symbol, timeframe=timeframe, since=since, exchange_id=exchange_id)
            result[symbol] = df
        except Exception as e:
            log.error(f"Failed to fetch {symbol}: {e}")
    return result


if __name__ == "__main__":
    # Quick test — fetch BTC and ETH 4h data
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
    fetch_multiple(symbols, timeframe="4h", since="2022-01-01")
    print("\n✅ Data pipeline test complete.")
