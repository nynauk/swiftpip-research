"""
ALTERNATIVE-SIGNAL HUNT v1 — fundamentally different inputs, same iron rules
(causal · per-instrument costs · era splits · survivor = works in EVERY era).

A. OVERNIGHT EFFECT (indices): long at cash close, exit next cash open.
   The documented "equity returns accrue overnight" anomaly. Includes spread both
   ways + 1.5bp/night financing haircut. R-unit = 1 daily ATR.
B. CROSS-SECTIONAL MOMENTUM (all 40): weekly rank by 26-week return; long top
   quartile, short bottom quartile; turnover-costed. Relative signal — a genuinely
   different animal from absolute price patterns.
C. CROSS-ASSET RISK GATE: Donchian D1 breakout on indices, longs only when the
   risk-on proxy (USDJPY 20d momentum) agrees, shorts only when risk-off.
D. VOL-REGIME GATE: the one real edge (Donchian D1 N=100 quality-14) traded only
   when volatility is expanding (ATR14 > ATR100).
"""
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import frequency_tune as fq
import realistic_ema_dol_rsi as edr
import all_pairs_2000_2026 as ap
import find_edge as fe

edr.SL_SPREAD_PAD = True
IDX = ["US500.cash", "US100.cash", "GER40.cash", "UK100.cash", "FRA40.cash"]
Q14 = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "EURJPY", "GBPJPY", "AUDJPY",
       "XAUUSD", "US500.cash", "US100.cash", "GER40.cash", "UK100.cash",
       "USOIL.cash", "UKOIL.cash"]
SPLITS_IDX = [("2010-2015", "2010-01-01", "2016-01-01"),
              ("2016-2020", "2016-01-01", "2021-01-01"),
              ("2021-2024", "2021-01-01", "2025-01-01")]
SPLITS_ALL = [("2000-2012", "2000-01-01", "2013-01-01"),
              ("2013-2019", "2013-01-01", "2020-01-01"),
              ("2020-2026", "2020-01-01", "2027-01-01")]
FIN_BP = 1.5e-4          # 1.5bp/night financing haircut on overnight holds


def split_eval(tag, df, splits, minn=50):
    if not len(df):
        print(f"  {tag:38} no trades"); return None
    df = df.sort_values("entry_time")
    cells = []; ok = True
    for name, a, b in splits:
        g = df[(df.entry_time >= a) & (df.entry_time < b)]
        st = edr.stats(g)
        if st is None or st["n"] < minn:
            cells.append(f"{'—':>12}"); ok = False; continue
        cells.append(f"{st['pf']:>5.2f}/{st['n']:>5}")
        if st["pf"] < 1.25:
            ok = False
    st = edr.stats(df)
    flag = " ✅ SURVIVOR" if ok and st["pf"] >= 1.3 else ""
    print(f"  {tag:38} {''.join(cells)}  | all PF {st['pf']:>5.2f} N {st['n']:>5} "
          f"DD {st['dd']:>6.1f}%{flag}", flush=True)
    return st, ok


# ── A. overnight effect ─────────────────────────────────────────────
def overnight(sym, m5, close_hh, open_hh, open_mm=30):
    spread, slip, _ = fq.SPEC[sym]
    d1 = m5.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                "close": "last"}).dropna()
    A = pd.Series(fe.atr(d1.high.values, d1.low.values, d1.close.values),
                  index=d1.index)
    g = m5.copy(); g["day"] = g.index.normalize()
    days = sorted(g.day.unique())
    trades = []
    by_day = dict(tuple(g.groupby("day")))
    for i in range(len(days) - 1):
        d0, d1x = days[i], days[i + 1]
        b0, b1 = by_day[d0], by_day[d1x]
        ent = b0[(b0.index.hour == close_hh) & (b0.index.minute < 10)]
        exi = b1[(b1.index.hour == open_hh) & (b1.index.minute >= open_mm)]
        if not len(ent) or not len(exi):
            continue
        a = A.asof(d0)
        if pd.isna(a) or a <= 0:
            continue
        entry = ent.close.iloc[0] + spread + slip
        exit_ = exi.open.iloc[0] - slip
        pnl = exit_ - entry - FIN_BP * entry
        r = pnl / a
        trades.append(dict(symbol=sym, entry_time=ent.index[0], exit_time=exi.index[0],
                           pnl_r=r, outcome="win" if r > 0.01 else ("loss" if r < -0.01 else "be")))
    return pd.DataFrame(trades)


# ── C/D. gated Donchian ─────────────────────────────────────────────
def donch_gated(sym, df, N, k, allow_long=None, allow_short=None):
    spread, slip, _ = fq.SPEC[sym]; pad = edr.sl_pad(sym, spread)
    h, l, c, idx = df.high.values, df.low.values, df.close.values, df.index
    A = fe.atr(h, l, c); n = len(df)
    trades = []; pos = None
    for i in range(N + 20, n):
        if pos is None:
            if np.isnan(A[i]) or A[i] <= 0:
                continue
            hh = h[i - N:i].max(); ll = l[i - N:i].min()
            if c[i] > hh and (allow_long is None or allow_long[i]):
                entry = c[i] + spread + slip; stop = entry - k * A[i] - pad
                pos = ("long", i, entry, stop, entry - stop)
            elif c[i] < ll and (allow_short is None or allow_short[i]):
                entry = c[i] - slip; stop = entry + k * A[i] + pad
                pos = ("short", i, entry, stop, stop - entry)
        else:
            d, ei, entry, stop, risk = pos
            if d == "long":
                if l[i] <= stop:
                    r = (stop - entry) / risk
                    trades.append(dict(symbol=sym, entry_time=idx[ei], exit_time=idx[i], pnl_r=r,
                                       outcome="win" if r > 0.01 else ("loss" if r < -0.01 else "be")))
                    pos = None; continue
                stop = max(stop, c[i] - k * A[i])
            else:
                if h[i] + spread >= stop:
                    r = (entry - stop) / risk
                    trades.append(dict(symbol=sym, entry_time=idx[ei], exit_time=idx[i], pnl_r=r,
                                       outcome="win" if r > 0.01 else ("loss" if r < -0.01 else "be")))
                    pos = None; continue
                stop = min(stop, c[i] + k * A[i] + spread)
            pos = (d, ei, entry, stop, risk)
    return pd.DataFrame(trades)


def main():
    have = mt5.initialize()
    d1 = {}
    for s in ap.U:
        b = fe.load_tf(s, "1D", mt5.TIMEFRAME_D1)
        if b is not None and len(b) > 400:
            d1[s] = b
    if have:
        mt5.shutdown()

    print("=" * 120)
    print("ALTERNATIVE-SIGNAL HUNT v1 · causal · costed · survivor = PF>=1.25 in every era")
    print("=" * 120, flush=True)

    # ── A. overnight effect ──
    print("\nA · OVERNIGHT EFFECT (long close→open, indices, 1.5bp/night financing)", flush=True)
    print(f"  {'config':38} {'10-15':>12} {'16-20':>12} {'21-24':>12}  | overall", flush=True)
    for close_hh, open_hh, tag in [(16, 9, "US anchors 16:00→09:30"),
                                   (15, 8, "US anchors 15:00→08:30")]:
        dfs = []
        for s in ["US500.cash", "US100.cash"]:
            p = os.path.join("ext_data", f"HD_{s}_5min_2000-2024.pkl")
            if os.path.exists(p):
                m5 = pd.read_pickle(p)
                dfs.append(overnight(s, m5[m5.index >= "2010-01-01"], close_hh, open_hh))
        split_eval(f"OVN US {tag}", pd.concat(dfs, ignore_index=True), SPLITS_IDX, minn=200)
    for close_hh, open_hh, om, tag in [(11, 3, 0, "EU 17:00CET→09:00CET")]:
        dfs = []
        for s in ["GER40.cash", "UK100.cash", "FRA40.cash"]:
            p = os.path.join("ext_data", f"HD_{s}_5min_2000-2024.pkl")
            if os.path.exists(p):
                m5 = pd.read_pickle(p)
                dfs.append(overnight(s, m5[m5.index >= "2010-01-01"], close_hh, open_hh, om))
        split_eval(f"OVN EU {tag}", pd.concat(dfs, ignore_index=True), SPLITS_IDX, minn=200)

    # ── B. cross-sectional momentum ──
    print("\nB · CROSS-SECTIONAL MOMENTUM (weekly, 26w lookback, top/bottom quartile, all 40)", flush=True)
    wk = pd.DataFrame({s: d1[s]["close"].resample("W-FRI").last() for s in d1}).dropna(how="all")
    mom = wk / wk.shift(26) - 1.0
    fwd = wk.shift(-1) / wk - 1.0
    rets = []
    prev_pos = {}
    for t in range(26, len(wk) - 1):
        row = mom.iloc[t].dropna()
        if len(row) < 12:
            continue
        q = len(row) // 4
        longs = row.nlargest(q).index; shorts = row.nsmallest(q).index
        pos = {s: 1 for s in longs} | {s: -1 for s in shorts}
        pnl = 0.0; nleg = 0
        for s, sgn in pos.items():
            f = fwd.iloc[t].get(s, np.nan)
            if np.isnan(f):
                continue
            cost = (fq.SPEC[s][0] / wk.iloc[t][s]) if prev_pos.get(s) != sgn else 0.0
            pnl += sgn * f - cost; nleg += 1
        prev_pos = pos
        if nleg:
            rets.append((wk.index[t], pnl / nleg))
    rr = pd.DataFrame(rets, columns=["entry_time", "ret"])
    rr["pnl_r"] = rr["ret"] / rr["ret"].std()
    rr["outcome"] = np.where(rr.pnl_r > 0.01, "win", np.where(rr.pnl_r < -0.01, "loss", "be"))
    rr["exit_time"] = rr["entry_time"]
    print(f"  {'config':38} {'00-12':>12} {'13-19':>12} {'20-26':>12}  | overall", flush=True)
    st = split_eval("XSMOM 26w quartiles all40", rr, SPLITS_ALL, minn=100)
    ann = rr.set_index("entry_time")["ret"]
    print(f"    weekly mean {ann.mean()*1e4:.1f}bp · ann.Sharpe {ann.mean()/ann.std()*np.sqrt(52):.2f}",
          flush=True)

    # ── C. cross-asset risk gate on index breakout ──
    print("\nC · RISK-ON/OFF GATE (USDJPY 20d momentum) on Donchian D1 indices", flush=True)
    print(f"  {'config':38} {'10-15':>12} {'16-20':>12} {'21-24':>12}  | overall", flush=True)
    uj = d1["USDJPY"]["close"]
    risk_on = (uj / uj.shift(20) - 1.0) > 0
    for N, k in [(55, 2.0), (100, 3.0)]:
        dfs_g, dfs_u = [], []
        for s in IDX:
            if s not in d1:
                continue
            df = d1[s]
            ro = risk_on.reindex(df.index).ffill().fillna(False).values
            dfs_g.append(donch_gated(s, df, N, k, allow_long=ro, allow_short=~ro))
            dfs_u.append(donch_gated(s, df, N, k))
        split_eval(f"DONCH idx N={N} k={k} +riskgate", pd.concat(dfs_g, ignore_index=True), SPLITS_IDX)
        split_eval(f"DONCH idx N={N} k={k} ungated", pd.concat(dfs_u, ignore_index=True), SPLITS_IDX)

    # ── D. vol-regime gate on the one real edge ──
    print("\nD · VOL-REGIME GATE (ATR14>ATR100) on Donchian D1 quality-14", flush=True)
    print(f"  {'config':38} {'00-12':>12} {'13-19':>12} {'20-26':>12}  | overall", flush=True)
    for N, k in [(100, 3.0), (55, 2.0)]:
        dfs_g = []
        for s in Q14:
            if s not in d1:
                continue
            df = d1[s]
            a14 = fe.atr(df.high.values, df.low.values, df.close.values, 14)
            a100 = fe.atr(df.high.values, df.low.values, df.close.values, 100)
            hot = a14 > a100
            dfs_g.append(donch_gated(s, df, N, k, allow_long=hot, allow_short=hot))
        split_eval(f"DONCH q14 N={N} k={k} +volgate", pd.concat(dfs_g, ignore_index=True), SPLITS_ALL)


if __name__ == "__main__":
    main()
