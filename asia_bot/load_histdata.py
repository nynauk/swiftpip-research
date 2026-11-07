"""
Build M5 OHLC frames for any pair from the HistData yearly M1 ASCII zips.

Format inside each zip: YYYYMMDD HHMMSS;open;high;low;close;volume  (semicolon,
no header). Index is EST with NO DST — same convention as the pre-built
HD_EURUSD_*.pkl files, so downstream code uses offset_h = -5.

Each pair is parsed once and cached to ext_data/HD_{PAIR}_5min_hd.pkl.

    python -m asia_bot.load_histdata EURUSD GBPUSD ...
"""
import os, sys, glob, zipfile
import pandas as pd

ZIP_DIR = os.path.join("ext_data", "histdata_zips")
CACHE_FMT = os.path.join("ext_data", "HD_{}_5min_hd.pkl")


def _read_zip(path):
    z = zipfile.ZipFile(path)
    name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    with z.open(name) as f:
        df = pd.read_csv(f, sep=";", header=None,
                         names=["ts", "open", "high", "low", "close", "vol"])
    df["ts"] = pd.to_datetime(df["ts"], format="%Y%m%d %H%M%S")
    return df.set_index("ts")[["open", "high", "low", "close"]]


def _clean(df):
    """Drop physically-impossible bars (bad HistData ticks): non-positive prices,
    malformed OHLC, or absurd candle ranges (>25% of price — impossible on M5)."""
    ok = (df[["open", "high", "low", "close"]] > 0).all(axis=1)
    ok &= (df["high"] >= df["low"])
    ok &= (df["high"] / df["low"] < 1.25)
    return df[ok]


def build_m5(pair, force=False):
    cache = CACHE_FMT.format(pair)
    if os.path.exists(cache) and not force:
        df = pd.read_pickle(cache)
        return _clean(df)          # clean cached frames too (cheap, idempotent)
    zips = sorted(glob.glob(os.path.join(ZIP_DIR, f"DAT_ASCII_{pair}_M1_*.zip")))
    if not zips:
        return None
    parts = []
    for zp in zips:
        try:
            m1 = _read_zip(zp)
        except Exception as e:
            print(f"    ! {os.path.basename(zp)}: {e}"); continue
        # resample M1 -> M5 immediately so we never hold 9M rows per pair
        m5 = m1.resample("5min").agg({"open": "first", "high": "max",
                                      "low": "min", "close": "last"}).dropna()
        parts.append(m5)
    if not parts:
        return None
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.to_pickle(cache)
    return df


def main():
    pairs = sys.argv[1:] or ["EURUSD"]
    for p in pairs:
        df = build_m5(p)
        if df is None:
            print(f"  {p:8} no zips found")
        else:
            print(f"  {p:8} {len(df):>9,} M5 bars  {df.index[0]} -> {df.index[-1]}")


if __name__ == "__main__":
    main()
