"""
Dukascopy tick-data downloader → OHLC bars, for backtesting the strategies on
LONGER / out-of-sample history than MT5's 99,999-bar cap allows.

Dukascopy serves LZMA (.bi5) tick files per instrument-hour; each 20-byte record
is (ms_offset:>u4, ask*factor:>u4, bid*factor:>u4, askvol:>f4, bidvol:>f4).
We aggregate BID ticks into bars (MT5 bars are bid too, so this matches the feed
the strategy was built on). Threaded download + on-disk pickle cache.

External feed ≠ your broker's feed, so treat this as OUT-OF-SAMPLE validation of
the edge, not a tick-for-tick match to the live account.
"""
import os
import lzma
import time
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

BASE = "https://datafeed.dukascopy.com/datafeed"
CACHE = "ext_data"
os.makedirs(CACHE, exist_ok=True)

# broker_symbol: (dukascopy_code, decimal_factor)
INSTR = {
    "EURUSD":     ("EURUSD",        1e5),
    "GBPUSD":     ("GBPUSD",        1e5),
    "USDJPY":     ("USDJPY",        1e3),
    "USDCHF":     ("USDCHF",        1e5),
    "AUDUSD":     ("AUDUSD",        1e5),
    "US500.cash": ("USA500IDXUSD",  1e3),
    "US30.cash":  ("USA30IDXUSD",   1e3),
    "US100.cash": ("USATECHIDXUSD", 1e3),
    "GER40.cash": ("DEUIDXEUR",     1e3),
    "UK100.cash": ("GBRIDXGBP",     1e3),
}

_REC = np.dtype([("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"),
                 ("av", ">f4"), ("bv", ">f4")])

_session = requests.Session()
_session.mount("https://", HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=0))


def _hour(code, factor, day, h, attempts=5):
    """Returns (times, bids) on success, None for a legit empty hour (holiday/
    weekend/404), or 'FAIL' if every retry errored (so we can count real drops)."""
    url = f"{BASE}/{code}/{day.year}/{day.month-1:02d}/{day.day:02d}/{h:02d}h_ticks.bi5"
    for a in range(attempts):
        try:
            r = _session.get(url, timeout=30)
            if r.status_code == 404:
                return None                       # no file this hour = legit no data
            if r.status_code != 200:
                raise IOError(f"HTTP {r.status_code}")   # 503/429 throttle → retry
            if not r.content:
                return None                       # empty hour
            data = lzma.decompress(r.content, format=lzma.FORMAT_ALONE)
            if len(data) < 20:
                return None
            rec = np.frombuffer(data[: len(data) // 20 * 20], dtype=_REC)
            base = dt.datetime(day.year, day.month, day.day, h)
            t = np.datetime64(base) + rec["ms"].astype("timedelta64[ms]")
            return t, rec["bid"].astype(np.float64) / factor
        except Exception:
            if a == attempts - 1:
                return "FAIL"
            time.sleep(0.4 * (a + 1))             # backoff on throttle/reset
    return "FAIL"


def fetch(symbol, start, end, tf="5min", workers=16):
    """Return a bid-OHLC DataFrame for [start, end) resampled to tf. Cached."""
    code, factor = INSTR[symbol]
    key = os.path.join(CACHE, f"{symbol}_{tf}_{start}_{end}.pkl")
    if os.path.exists(key):
        return pd.read_pickle(key)
    d0 = dt.date.fromisoformat(start); d1 = dt.date.fromisoformat(end)
    tasks = []
    d = d0
    while d < d1:
        if d.weekday() < 5:                         # skip weekends (no ticks)
            tasks += [(d, h) for h in range(24)]
        d += dt.timedelta(days=1)
    t0 = time.time()
    times, bids = [], []
    fails = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(lambda x: _hour(code, factor, x[0], x[1]), tasks):
            if res == "FAIL":
                fails += 1
            elif res is not None:
                times.append(res[0]); bids.append(res[1])
    if fails > len(tasks) * 0.02:
        raise RuntimeError(f"{symbol}: {fails}/{len(tasks)} hourly requests FAILED after retries "
                           f"— feed throttling; NOT caching partial data")
    if not times:
        raise RuntimeError(f"{symbol}: no ticks downloaded for {start}..{end}")
    idx = pd.to_datetime(np.concatenate(times))
    s = pd.Series(np.concatenate(bids), index=idx).sort_index()
    ohlc = s.resample(tf).ohlc().dropna()
    ohlc.columns = ["open", "high", "low", "close"]
    ohlc.to_pickle(key)
    print(f"  {symbol:12} {len(tasks):>6} hrs ({fails} fail) → {len(s):>10,} ticks → "
          f"{len(ohlc):>7,} {tf} bars [{ohlc.close.min():.3f}..{ohlc.close.max():.3f}]  "
          f"{time.time()-t0:.0f}s", flush=True)
    return ohlc


if __name__ == "__main__":
    # 1-week validation: confirm instrument codes + decimal factors + speed
    print("Validation pull · 2023-06-05 → 2023-06-12 · M5")
    for s in ["EURUSD", "USDJPY", "US500.cash", "GER40.cash", "US100.cash", "UK100.cash"]:
        try:
            fetch(s, "2023-06-05", "2023-06-12", tf="5min")
        except Exception as e:
            print(f"  {s:12} FAILED: {e!r}")
