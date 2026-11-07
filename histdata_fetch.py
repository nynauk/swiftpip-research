"""
HistData.com fetcher → OHLC bars, for deep out-of-sample backtesting.
HistData serves free M1 bar data as yearly ZIPs (one download per instrument-
year, back to ~2000 for FX and ~2018 for indices). We resample M1 → M5 / M15.

Covers FX + the index/commodity engines the strategy needs:
  SPXUSD=US500, NSXUSD=US100, GRXEUR=DAX(GER40), UKXGBP=FTSE(UK100), WTIUSD=oil.
(No Dow/US30 on HistData.)  Timezone is EST (no DST) — a session offset vs the
broker feed, irrelevant to the pattern-based strategy. Raw zips + resampled bars
are cached on disk. External feed = OUT-OF-SAMPLE validation, not a broker match.
"""
import os
import zipfile
import pandas as pd
from histdata import download_hist_data as dl
from histdata.api import Platform as P, TimeFrame as TF

CACHE = "ext_data"
ZIPS = os.path.join(CACHE, "histdata_zips")
os.makedirs(ZIPS, exist_ok=True)

PAIR = {  # broker_symbol : histdata_code
    "EURUSD": "eurusd", "GBPUSD": "gbpusd", "USDJPY": "usdjpy", "USDCHF": "usdchf",
    "USDCAD": "usdcad", "AUDUSD": "audusd", "NZDUSD": "nzdusd", "EURGBP": "eurgbp",
    "EURJPY": "eurjpy", "GBPJPY": "gbpjpy",
    "US500.cash": "spxusd", "US100.cash": "nsxusd", "GER40.cash": "grxeur",
    "UK100.cash": "ukxgbp", "USOIL.cash": "wtiusd",
}


def _year_m1(code, year):
    fname = f"DAT_ASCII_{code.upper()}_M1_{year}.zip"
    zfull = os.path.abspath(os.path.join(ZIPS, fname))
    if not os.path.exists(zfull):
        cwd = os.getcwd()
        try:
            os.chdir(ZIPS)
            dl(year=str(year), month=None, pair=code,
               platform=P.GENERIC_ASCII, time_frame=TF.ONE_MINUTE)
        finally:
            os.chdir(cwd)
    zf = zipfile.ZipFile(zfull)
    csv = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    df = pd.read_csv(zf.open(csv), sep=";", header=None,
                     names=["dt", "open", "high", "low", "close", "vol"])
    df["time"] = pd.to_datetime(df["dt"], format="%Y%m%d %H%M%S")
    return df.set_index("time")[["open", "high", "low", "close"]]


def fetch(symbol, years, tf="5min"):
    """Yearly M1 → resampled OHLC over `years` (list), cached."""
    key = os.path.join(CACHE, f"HD_{symbol}_{tf}_{years[0]}-{years[-1]}.pkl")
    if os.path.exists(key):
        return pd.read_pickle(key)
    frames = []
    for y in years:
        try:
            frames.append(_year_m1(PAIR[symbol], y))
        except Exception as e:
            print(f"    {symbol} {y}: skip ({type(e).__name__})")
    if not frames:
        raise RuntimeError(f"{symbol}: no data for {years}")
    m1 = pd.concat(frames).sort_index()
    m1 = m1[~m1.index.duplicated(keep="first")]
    # ── clean bad ticks (HistData index feeds have occasional spurious prints) ──
    raw = len(m1)
    m1 = m1[m1["close"] > 0]
    m1 = m1[(m1["high"] - m1["low"]) / m1["close"] < 0.08]        # impossible 1-min wick
    med = m1["close"].rolling(120, min_periods=30, center=True).median()
    m1 = m1[med.isna() | ((m1["close"] - med).abs() / med < 0.25)]  # close-price spikes
    dropped = raw - len(m1)
    bars = m1.resample(tf).agg({"open": "first", "high": "max",
                                "low": "min", "close": "last"}).dropna()
    bars.to_pickle(key)
    print(f"  {symbol:12} {years[0]}-{years[-1]} → {len(m1):>9,} M1 ({dropped} bad dropped) → "
          f"{len(bars):>7,} {tf} bars [{bars.close.min():.3f}..{bars.close.max():.3f}]", flush=True)
    return bars


if __name__ == "__main__":
    print("Validation: EURUSD + SPXUSD + GRXEUR, 2022-2023, M5")
    for s in ["EURUSD", "US500.cash", "GER40.cash", "US100.cash", "UK100.cash"]:
        try:
            fetch(s, [2022, 2023], tf="5min")
        except Exception as e:
            print(f"  {s:12} FAILED: {e!r}")
