"""
Backtest on REAL CME futures (yfinance continuous contracts).
Deep daily history → CRT 1D + Donchian-trend + TSMOM (the daily-testable strategies).
~2yr hourly → CRT 1H. Realistic futures costs (≈2-tick effective spread + commission).
Causal, era-split survivor gate. The intraday ICT setups (5m/1m sweep/CISD/IFVG,
HTF nested, CRT+5m) need deep INTRADAY futures data (paid) — not runnable free here;
the index-CFD proxies (US500=ES, US100=NQ, GER40=FDAX, CL, GC) already covered them.
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf

import frequency_tune as fq
import realistic_ema_dol_rsi as edr
import find_edge as fe
import crt_hunt as crt

CACHE = "ext_data"
# yahoo, our_name, spread(pts), slippage, min_fvg, pip
FUT = [
    ("ES=F", "ES", 0.50, 0.25, 1.0, 1.0),   ("NQ=F", "NQ", 1.0, 0.5, 5.0, 1.0),
    ("YM=F", "YM", 2.0, 1.0, 8.0, 1.0),      ("RTY=F", "RTY", 0.3, 0.1, 1.0, 0.1),
    ("GC=F", "GC", 0.20, 0.1, 0.3, 0.1),     ("SI=F", "SI", 0.02, 0.01, 0.02, 0.01),
    ("HG=F", "HG", 0.002, 0.001, 0.003, 0.001), ("CL=F", "CL", 0.02, 0.01, 0.05, 0.01),
    ("NG=F", "NG", 0.005, 0.002, 0.01, 0.001),
    ("6E=F", "6E", 1e-4, 5e-5, 2e-4, 1e-4),  ("6B=F", "6B", 2e-4, 1e-4, 3e-4, 1e-4),
    ("6J=F", "6J", 5e-6, 2e-6, 1e-5, 1e-6),  ("6A=F", "6A", 1e-4, 5e-5, 2e-4, 1e-4),
    ("ZN=F", "ZN", 0.03, 0.015, 0.05, 0.01), ("ZB=F", "ZB", 0.06, 0.03, 0.1, 0.03),
    ("ZC=F", "ZC", 0.5, 0.25, 1.0, 0.25),    ("ZS=F", "ZS", 0.5, 0.25, 1.0, 0.25),
]
SPLITS = [("2000-2011", "2000-01-01", "2012-01-01"),
          ("2012-2018", "2012-01-01", "2019-01-01"),
          ("2019-2026", "2019-01-01", "2027-01-01")]

# inject futures specs so shared strategy code prices each contract
for y, nm, sp, sl, mf, pip in FUT:
    fq.SPEC[nm] = (sp, sl, mf); edr.PIP[nm] = pip


def flat(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    return df[["open", "high", "low", "close"]].dropna()


def daily(y, nm):
    p = os.path.join(CACHE, f"FUT_{nm}_1d.pkl")
    if os.path.exists(p):
        return pd.read_pickle(p)
    try:
        df = flat(yf.download(y, start="2000-01-01", interval="1d", progress=False, auto_adjust=False))
        if len(df) > 200:
            df.to_pickle(p); return df
    except Exception as e:
        print(f"    {nm} daily FAIL {e}")
    return None


def hourly(y, nm):
    p = os.path.join(CACHE, f"FUT_{nm}_1h.pkl")
    if os.path.exists(p):
        return pd.read_pickle(p)
    try:
        df = flat(yf.download(y, period="730d", interval="1h", progress=False, auto_adjust=False))
        if len(df) > 500:
            df.to_pickle(p); return df
    except Exception:
        pass
    return None


def report(tag, dfs, splits, degfn):
    raw = pd.concat([x for x in dfs if len(x)], ignore_index=True)
    if not len(raw):
        print(f"  {tag}: no trades"); return
    raw["entry_time"] = pd.to_datetime(raw["entry_time"])
    deg = degfn(raw.sort_values("entry_time"))
    st = edr.stats(deg); yrs = max((raw.entry_time.max() - raw.entry_time.min()).days / 365.25, 0.1)
    print(f"\n  ══ {tag} ══  N={st['n']} ({st['n']/yrs:.0f}/yr) · WR {st['wr']:.1f}% · PF {st['pf']:.2f} · "
          f"{st['tot']:+.0f}R · DD {st['dd']:.0f}% · ${st['bal']:,.0f}", flush=True)
    ok = True
    for name, a, b in splits:
        g = deg[(deg.entry_time >= a) & (deg.entry_time < b)]; s2 = edr.stats(g)
        if s2 is None or s2["n"] < 25:
            print(f"      {name}: few"); ok = False; continue
        print(f"      {name}: N={s2['n']:>4} WR {s2['wr']:>4.0f}% PF {s2['pf']:>5.2f} {s2['tot']:>+6.0f}R")
        if s2["pf"] < 1.25: ok = False
    print(f"      VERDICT: {'✅ SURVIVOR' if ok and st['pf'] >= 1.3 else '❌ no edge'}", flush=True)


def main():
    edr.SL_SPREAD_PAD = True
    print("Downloading real CME futures (yfinance)...", flush=True)
    dd, hh = {}, {}
    for y, nm, *_ in FUT:
        d = daily(y, nm)
        if d is not None:
            dd[nm] = d
        h = hourly(y, nm)
        if h is not None:
            hh[nm] = h
    print(f"  daily: {len(dd)} · hourly: {len(hh)}", flush=True)

    print("\n" + "=" * 100)
    print("REAL FUTURES BACKTEST · your strategies (daily-testable) · causal · futures costs")
    print("=" * 100, flush=True)

    print("\n### CRT (Candle Range Theory) on real futures ###", flush=True)
    report("CRT · 1D · futures", [crt.run_crt(nm, dd[nm]) for nm in dd], SPLITS, crt.degrade)
    if hh:
        h_splits = [("2024-2025", "2024-01-01", "2025-07-01"), ("2025-2026", "2025-07-01", "2027-01-01")]
        report("CRT · 1H · futures (2yr)", [crt.run_crt(nm, hh[nm]) for nm in hh], h_splits, crt.degrade)

    fed = lambda d: fe.degrade(d, 1)
    print("\n### Trend benchmark (the one edge that ever worked) on real futures ###", flush=True)
    report("Donchian D1 N=100 k=3 · futures", [fe.run_donch(nm, dd[nm], 100, 3.0) for nm in dd], SPLITS, fed)
    report("Donchian D1 N=55 k=2 · futures", [fe.run_donch(nm, dd[nm], 55, 2.0) for nm in dd], SPLITS, fed)
    report("TSMOM D1 N=120 · futures", [fe.run_tsmom(nm, dd[nm], 120, 3.0) for nm in dd], SPLITS, fed)


if __name__ == "__main__":
    main()
