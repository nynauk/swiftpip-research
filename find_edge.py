"""
CAUSAL EDGE HUNT v1 — honest from the first line.
Rules: strictly causal features (no swing/center-window anything), per-instrument
spread+slippage on every fill, stop widened by spread+1pip, execution degradation
(spread widening / stop slip / tail gaps), and THREE time splits — a config must
be profitable in ALL of them after costs to survive.

Families (robust, classic):
  DONCH  N-bar breakout (close through prior N-bar extreme) + k×ATR trailing stop
  MAX    EMA F/S cross + fixed k×ATR stop, exit on opposite cross
  TSMOM  time-series momentum (close vs close[N]) + k×ATR stop, D1
  RSI2   Connors daily pullback fade (SMA200 trend, RSI2 extreme, ATR stop)

TFs: D1 + H4 (resampled from cached HistData M5, 2000-2024, + MT5 2025-26 tail).
"""
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import frequency_tune as fq
import realistic_ema_dol_rsi as edr
import all_pairs_2000_2026 as ap          # SPEC/PIP injections + U universe

SYMS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "EURJPY", "GBPJPY", "AUDJPY",
        "XAUUSD", "US500.cash", "US100.cash", "GER40.cash", "UK100.cash",
        "USOIL.cash", "UKOIL.cash"]
SPLITS = [("2000-2012", "2000-01-01", "2013-01-01"),
          ("2013-2019", "2013-01-01", "2020-01-01"),
          ("2020-2026", "2020-01-01", "2027-01-01")]
OHLC = {"open": "first", "high": "max", "low": "min", "close": "last"}
RNG = np.random.default_rng(42)
OUT = os.path.join("strategies", "edge_hunt")


# ─────────────────────────── data ───────────────────────────
def load_tf(sym, rule, mt5_tf):
    p = os.path.join("ext_data", f"HD_{sym}_5min_2000-2024.pkl")
    if not os.path.exists(p):
        return None
    m5 = pd.read_pickle(p)
    bars = m5.resample(rule).agg(OHLC).dropna()
    code, cands, spec, pip = ap.U[sym]
    tail = ap.mt5_bars(cands, mt5_tf, n=99999)
    if tail is not None:
        tail = tail[tail.index >= "2025-01-01"]
        bars = pd.concat([bars[bars.index < "2025-01-01"], tail])
    return bars


def atr(h, l, c, n=14):
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().values


def rsi(c, n):
    s = pd.Series(c); d = s.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return (100 - 100 / (1 + ag / al.replace(0, np.nan))).values


# ─────────────────────────── strategy loops (all causal) ───────────────────────────
def _rec(trades, sym, idx, ei, xj, entry, risk, px, direction):
    r = (px - entry) / risk if direction == "long" else (entry - px) / risk
    trades.append(dict(symbol=sym, entry_time=idx[ei], exit_time=idx[xj], pnl_r=r,
                       spread_r=np.nan, outcome="win" if r > 0.01 else ("loss" if r < -0.01 else "be")))
    return trades


def run_donch(sym, df, N, k):
    spread, slip, _ = fq.SPEC[sym]; pad = edr.sl_pad(sym, spread)
    h, l, c, idx = df.high.values, df.low.values, df.close.values, df.index
    A = atr(h, l, c); n = len(df)
    trades = []; pos = None
    for i in range(N + 20, n):
        if pos is None:
            hh = h[i - N:i].max(); ll = l[i - N:i].min()
            if np.isnan(A[i]) or A[i] <= 0:
                continue
            if c[i] > hh:
                entry = c[i] + spread + slip; stop = entry - k * A[i] - pad
                pos = ("long", i, entry, stop, entry - stop)
            elif c[i] < ll:
                entry = c[i] - slip; stop = entry + k * A[i] + pad
                pos = ("short", i, entry, stop, stop - entry)
        else:
            d, ei, entry, stop, risk = pos
            if d == "long":
                if l[i] <= stop:
                    _rec(trades, sym, idx, ei, i, entry, risk, stop, d); pos = None; continue
                stop = max(stop, c[i] - k * A[i])
            else:
                if h[i] + spread >= stop:
                    _rec(trades, sym, idx, ei, i, entry, risk, stop, d); pos = None; continue
                stop = min(stop, c[i] + k * A[i] + spread)
            pos = (d, ei, entry, stop, risk)
    for t in trades:
        t["spread_r"] = spread  # placeholder; converted later
    return pd.DataFrame(trades)


def run_max(sym, df, F, S, k):
    spread, slip, _ = fq.SPEC[sym]; pad = edr.sl_pad(sym, spread)
    h, l, c, idx = df.high.values, df.low.values, df.close.values, df.index
    A = atr(h, l, c)
    ef = pd.Series(c).ewm(span=F, adjust=False).mean().values
    es = pd.Series(c).ewm(span=S, adjust=False).mean().values
    above = ef > es; n = len(df)
    trades = []; pos = None
    for i in range(S + 20, n):
        xu = above[i] and not above[i - 1]; xd = (not above[i]) and above[i - 1]
        if pos is not None:
            d, ei, entry, stop, risk = pos
            if d == "long":
                if l[i] <= stop:
                    _rec(trades, sym, idx, ei, i, entry, risk, stop, d); pos = None
                elif xd:
                    _rec(trades, sym, idx, ei, i, entry, risk, c[i], d); pos = None
            else:
                if h[i] + spread >= stop:
                    _rec(trades, sym, idx, ei, i, entry, risk, stop, d); pos = None
                elif xu:
                    _rec(trades, sym, idx, ei, i, entry, risk, c[i] + spread, d); pos = None
        if pos is None and (xu or xd) and not np.isnan(A[i]) and A[i] > 0:
            if xu:
                entry = c[i] + spread + slip; stop = entry - k * A[i] - pad
                pos = ("long", i, entry, stop, entry - stop)
            else:
                entry = c[i] - slip; stop = entry + k * A[i] + pad
                pos = ("short", i, entry, stop, stop - entry)
    for t in trades:
        t["spread_r"] = spread
    return pd.DataFrame(trades)


def run_tsmom(sym, df, N, k):
    spread, slip, _ = fq.SPEC[sym]; pad = edr.sl_pad(sym, spread)
    h, l, c, idx = df.high.values, df.low.values, df.close.values, df.index
    A = atr(h, l, c); n = len(df)
    trades = []; pos = None
    for i in range(N + 20, n):
        sig = 1 if c[i] > c[i - N] else -1
        if pos is not None:
            d, ei, entry, stop, risk = pos
            if d == "long":
                if l[i] <= stop:
                    _rec(trades, sym, idx, ei, i, entry, risk, stop, d); pos = None
                elif sig < 0:
                    _rec(trades, sym, idx, ei, i, entry, risk, c[i], d); pos = None
            else:
                if h[i] + spread >= stop:
                    _rec(trades, sym, idx, ei, i, entry, risk, stop, d); pos = None
                elif sig > 0:
                    _rec(trades, sym, idx, ei, i, entry, risk, c[i] + spread, d); pos = None
        if pos is None and not np.isnan(A[i]) and A[i] > 0:
            if sig > 0:
                entry = c[i] + spread + slip; stop = entry - k * A[i] - pad
                pos = ("long", i, entry, stop, entry - stop)
            else:
                entry = c[i] - slip; stop = entry + k * A[i] + pad
                pos = ("short", i, entry, stop, stop - entry)
    for t in trades:
        t["spread_r"] = spread
    return pd.DataFrame(trades)


def run_rsi2(sym, df, lo, hi, ex, katr):
    spread, slip, _ = fq.SPEC[sym]; pad = edr.sl_pad(sym, spread)
    h, l, c, idx = df.high.values, df.low.values, df.close.values, df.index
    A = atr(h, l, c); r2 = rsi(c, 2)
    sma = pd.Series(c).rolling(200).mean().values
    n = len(df); trades = []; pos = None
    for i in range(220, n):
        if pos is not None:
            d, ei, entry, stop, risk, held = pos
            if d == "long":
                if l[i] <= stop:
                    _rec(trades, sym, idx, ei, i, entry, risk, stop, d); pos = None; continue
                if r2[i] >= ex or held >= 10:
                    _rec(trades, sym, idx, ei, i, entry, risk, c[i], d); pos = None; continue
            else:
                if h[i] + spread >= stop:
                    _rec(trades, sym, idx, ei, i, entry, risk, stop, d); pos = None; continue
                if r2[i] <= 100 - ex or held >= 10:
                    _rec(trades, sym, idx, ei, i, entry, risk, c[i] + spread, d); pos = None; continue
            pos = (d, ei, entry, stop, risk, held + 1)
        if pos is None and not (np.isnan(sma[i]) or np.isnan(r2[i]) or np.isnan(A[i])) and A[i] > 0:
            risk_d = katr * A[i] + pad
            if risk_d < 3 * spread:
                continue
            if c[i] > sma[i] and r2[i] < lo:
                entry = c[i] + spread + slip
                pos = ("long", i, entry, entry - risk_d, risk_d, 0)
            elif c[i] < sma[i] and r2[i] > hi:
                entry = c[i] - slip
                pos = ("short", i, entry, entry + risk_d, risk_d, 0)
    for t in trades:
        t["spread_r"] = spread
    return pd.DataFrame(trades)


# ─────────────────────────── evaluation ───────────────────────────
def degrade(df, seed):
    if not len(df):
        return df
    rng = np.random.default_rng(seed)
    d = df.copy().reset_index(drop=True)
    r = d.pnl_r.to_numpy(float)
    # spread_r column holds raw spread; convert to R via median risk proxy: use per-trade
    # spread/risk unavailable here → approximate with 0 extra (costs already in fills) and
    # apply only WIDENING (1.5x spread on 15%) via R-approx from pf of losses… keep simple:
    wide = rng.random(len(d)) < 0.15
    r = r - np.where(wide, 0.05, 0.0)                 # ≈1.5×spread as ~5% of typical ATR risk
    stopped = r <= -0.90
    r = r - np.where(stopped, 0.02, 0.0)              # stop slippage
    tail = rng.random(len(d)) < 0.002
    r = np.where(tail & (r < 0), -3.0, r)             # gap through stop
    d["pnl_r"] = r
    d["outcome"] = np.where(r > 0.01, "win", np.where(r < -0.01, "loss", "be"))
    return d


def evaluate(tag, dfs):
    df = pd.concat(dfs, ignore_index=True)
    if not len(df):
        return None
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df = degrade(df.sort_values("entry_time"), seed=hash(tag) % 2**31)
    cells = []
    ok = True
    for name, a, b in SPLITS:
        g = df[(df.entry_time >= a) & (df.entry_time < b)]
        st = edr.stats(g)
        if st is None or st["n"] < 30:
            cells.append(f"{'—':>12}"); ok = False; continue
        cells.append(f"{st['pf']:>5.2f}/{st['n']:>5}")
        if st["pf"] < 1.25:
            ok = False
    st_all = edr.stats(df)
    flag = " ✅" if ok and st_all and st_all["pf"] >= 1.3 else ""
    print(f"  {tag:24} {''.join(cells)}  | all PF {st_all['pf']:>5.2f} N {st_all['n']:>5} "
          f"DD {st_all['dd']:>6.1f}%{flag}", flush=True)
    return (tag, df, st_all, ok)


def main():
    os.makedirs(OUT, exist_ok=True)
    edr.SL_SPREAD_PAD = True
    have = mt5.initialize()
    data = {}
    for tf_name, rule, mtf in [("D1", "1D", mt5.TIMEFRAME_D1), ("H4", "4h", mt5.TIMEFRAME_H4)]:
        for s in SYMS:
            b = load_tf(s, rule, mtf)
            if b is not None and len(b) > 500:
                data[(s, tf_name)] = b
    if have:
        mt5.shutdown()
    print(f"data loaded: {len(data)} (symbol, tf) series", flush=True)

    print("\n" + "=" * 118)
    print("CAUSAL EDGE HUNT · 14 instruments · costs+degradation in · splits: "
          + " | ".join(s[0] for s in SPLITS) + "  (survivor = PF≥1.25 in EVERY split)")
    print("=" * 118)
    print(f"  {'config':24} {'PF/N 00-12':>12}{'PF/N 13-19':>12}{'PF/N 20-26':>12}  | overall")
    print("-" * 118, flush=True)

    results = []
    for tf in ["D1", "H4"]:
        for N, k in [(20, 3.0), (55, 3.0), (55, 2.0), (100, 3.0)]:
            dfs = [run_donch(s, data[(s, tf)], N, k) for s in SYMS if (s, tf) in data]
            r = evaluate(f"DONCH {tf} N={N} k={k}", dfs)
            if r: results.append(r)
    for tf in ["D1", "H4"]:
        for F, S, k in [(20, 100, 3.0), (50, 200, 3.0), (10, 50, 2.5)]:
            dfs = [run_max(s, data[(s, tf)], F, S, k) for s in SYMS if (s, tf) in data]
            r = evaluate(f"MAX {tf} {F}/{S} k={k}", dfs)
            if r: results.append(r)
    for N, k in [(60, 3.0), (120, 3.0), (250, 3.0)]:
        dfs = [run_tsmom(s, data[(s, "D1")], N, k) for s in SYMS if (s, "D1") in data]
        r = evaluate(f"TSMOM D1 N={N} k={k}", dfs)
        if r: results.append(r)
    for lo, hi, ex, katr in [(5, 95, 60, 2.5), (10, 90, 60, 2.5), (10, 90, 50, 1.5)]:
        dfs = [run_rsi2(s, data[(s, "D1")], lo, hi, ex, katr) for s in SYMS if (s, "D1") in data]
        r = evaluate(f"RSI2 D1 {lo}/{hi} ex{ex} k{katr}", dfs)
        if r: results.append(r)

    survivors = [(t, df, st) for t, df, st, ok in results if ok and st["pf"] >= 1.3]
    print("\n  SURVIVORS (profitable in every era, after costs+degradation):", flush=True)
    for t, df, st in sorted(survivors, key=lambda x: -x[2]["pf"]):
        cl = df[df.outcome != "open"]
        wk = max((df.entry_time.max() - df.entry_time.min()).days / 7, 1)
        print(f"    {t:26} PF {st['pf']:.2f} · N {st['n']} ({st['n']/wk:.1f}/wk) · WR {st['wr']:.0f}% · "
              f"DD {st['dd']:.1f}% · LS {st['ls']} · ${st['bal']:,.0f}", flush=True)
        df.to_csv(os.path.join(OUT, f"trades_{t.replace(' ', '_').replace('/', '-')}.csv"), index=False)
    if not survivors:
        print("    none — no family/config survives honest costs in all eras.", flush=True)


if __name__ == "__main__":
    main()
