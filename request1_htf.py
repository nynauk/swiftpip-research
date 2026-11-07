"""
REQUEST 1 — HTF sweep → 5m BOS → 1m opposite BOS (pullback) → 1m target BOS (entry).
Strictly causal, event-driven, 4 majors, M1 2010-2024, costs + degradation.

Chain (bearish example; bullish mirrors):
  1H  : a bar sweeps a CONFIRMED 1H swing high (high>swing, close<swing) → bias SHORT
  5m  : within 4h, a 5m close breaks a CONFIRMED 5m swing low (BOS down)
  1m  : within 60m, a 1m close breaks a CONFIRMED 1m swing high (pullback UP)
  1m  : within 30m after that, a 1m close breaks a CONFIRMED 1m swing low → ENTER short
  SL  : pullback swing high (+spread+1pip) · TP: next liquidity (5m swing low) OR fixed 2R
All swings usable only AFTER their confirmation lag. One trade per sweep.
"""
import os
import numpy as np
import pandas as pd

import frequency_tune as fq
import realistic_ema_dol_rsi as edr
import histdata_fetch as hd

edr.SL_SPREAD_PAD = True
SYMS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"]
YEARS = list(range(2010, 2025))
SPLITS = [("2010-2014", "2010-01-01", "2015-01-01"),
          ("2015-2019", "2015-01-01", "2020-01-01"),
          ("2020-2024", "2020-01-01", "2025-01-01")]
N1H, N5, N1 = 3, 3, 5
OHLC = {"open": "first", "high": "max", "low": "min", "close": "last"}


def conf_swings(df, n):
    """Return confirmed-by-bar-k arrays for swing high/low value+index (numpy)."""
    h = df.high.values; l = df.low.values; N = len(df)
    rmax = pd.Series(h).rolling(2 * n + 1, center=True).max().values
    rmin = pd.Series(l).rolling(2 * n + 1, center=True).min().values
    is_sh = h >= rmax; is_sl = l <= rmin
    def build(mask, vals):
        cv = np.full(N, np.nan); ci = np.full(N, np.nan)
        sw = np.where(mask)[0]; cp = sw + n; ok = cp < N
        cv[cp[ok]] = vals[sw[ok]]; ci[cp[ok]] = sw[ok]
        return pd.Series(cv).ffill().values, pd.Series(ci).ffill().values
    shv, shi = build(is_sh, h); slv, sli = build(is_sl, l)
    return shv, shi, slv, sli


def run(sym, m1):
    spread, slip, _ = fq.SPEC[sym]; pad = edr.sl_pad(sym, spread)
    m5 = m1.resample("5min").agg(OHLC).dropna()
    h1 = m1.resample("1h").agg(OHLC).dropna()
    o = m1.open.values; h = m1.high.values; l = m1.low.values; c = m1.close.values
    t1 = m1.index.values
    h5, l5, c5 = m5.high.values, m5.low.values, m5.close.values
    t5 = m5.index.values
    hh1, cl1 = h1.high.values, h1.close.values
    th1 = h1.index.values
    _, _, csl5v, _ = conf_swings(m5, N5)
    csh5v, _, _, _ = conf_swings(m5, N5)
    csh1v, _, csl1v, _ = conf_swings(m1, N1)
    csh1hv, _, csl1hv, _ = conf_swings(h1, N1H)
    HR = np.timedelta64(1, "h"); M5 = np.timedelta64(5, "m"); M1 = np.timedelta64(1, "m")

    dol, r2 = [], []
    last_swept = None
    for b in range(N1H + 2, len(h1)):
        bias = None
        if not np.isnan(csh1hv[b]) and hh1[b] > csh1hv[b] and cl1[b] < csh1hv[b]:
            bias = "short"; lvl = csh1hv[b]
        elif not np.isnan(csl1hv[b]) and h1.low.values[b] < csl1hv[b] and cl1[b] > csl1hv[b]:
            bias = "long"; lvl = csl1hv[b]
        if bias is None:
            continue
        if last_swept is not None and abs(lvl - last_swept) < spread:
            continue
        last_swept = lvl
        t_sweep = th1[b] + HR

        # ── 5m BOS within 4h ──
        e0 = np.searchsorted(t5, t_sweep); e1 = np.searchsorted(t5, t_sweep + 4 * HR)
        e = None
        for ee in range(e0, min(e1, len(m5))):
            if bias == "short" and not np.isnan(csl5v[ee]) and c5[ee] < csl5v[ee]:
                e = ee; break
            if bias == "long" and not np.isnan(csh5v[ee]) and c5[ee] > csh5v[ee]:
                e = ee; break
        if e is None:
            continue
        t_5bos = t5[e] + M5

        # ── 1m pullback (opposite BOS) within 60m ──
        u0 = np.searchsorted(t1, t_5bos); u1 = np.searchsorted(t1, t_5bos + 60 * M1)
        u = None
        for uu in range(u0, min(u1, len(m1))):
            if bias == "short" and not np.isnan(csh1v[uu]) and c[uu] > csh1v[uu]:
                u = uu; break
            if bias == "long" and not np.isnan(csl1v[uu]) and c[uu] < csl1v[uu]:
                u = uu; break
        if u is None:
            continue

        # ── 1m trigger (target BOS) within 30m ──
        g1 = np.searchsorted(t1, t1[u] + 30 * M1)
        g = None
        for gg in range(u + 1, min(g1, len(m1))):
            if bias == "short" and not np.isnan(csl1v[gg]) and c[gg] < csl1v[gg]:
                g = gg; break
            if bias == "long" and not np.isnan(csh1v[gg]) and c[gg] > csh1v[gg]:
                g = gg; break
        if g is None:
            continue

        if bias == "short":
            entry = c[g] - slip
            sl = h[u:g + 1].max() + pad; risk = sl - entry
            if risk <= 0: continue
            liq = csl5v[np.searchsorted(t5, t1[g]) - 1]
            tp_dol = liq if (not np.isnan(liq) and liq < entry - risk) else entry - 2 * risk
            for tp, bucket in [(entry - 2 * risk, r2), (tp_dol, dol)]:
                res = None
                for x in range(g + 1, min(g + 480, len(m1))):
                    if h[x] + spread >= sl: res = (x, -1.0); break
                    if l[x] + spread <= tp: res = (x, (entry - tp) / risk); break
                if res is None:
                    x = min(g + 480, len(m1)) - 1; res = (x, (entry - (c[x] + spread)) / risk)
                bucket.append(dict(symbol=sym, entry_time=m1.index[g], exit_time=m1.index[res[0]],
                                   pnl_r=res[1], spread_r=spread / risk,
                                   outcome="win" if res[1] > 0.01 else ("loss" if res[1] < -0.01 else "be")))
        else:
            entry = c[g] + spread + slip
            sl = l[u:g + 1].min() - pad; risk = entry - sl
            if risk <= 0: continue
            liq = csh5v[np.searchsorted(t5, t1[g]) - 1]
            tp_dol = liq if (not np.isnan(liq) and liq > entry + risk) else entry + 2 * risk
            for tp, bucket in [(entry + 2 * risk, r2), (tp_dol, dol)]:
                res = None
                for x in range(g + 1, min(g + 480, len(m1))):
                    if l[x] <= sl: res = (x, -1.0); break
                    if h[x] >= tp: res = (x, (tp - entry) / risk); break
                if res is None:
                    x = min(g + 480, len(m1)) - 1; res = (x, (c[x] - entry) / risk)
                bucket.append(dict(symbol=sym, entry_time=m1.index[g], exit_time=m1.index[res[0]],
                                   pnl_r=res[1], spread_r=spread / risk,
                                   outcome="win" if res[1] > 0.01 else ("loss" if res[1] < -0.01 else "be")))
    return pd.DataFrame(r2), pd.DataFrame(dol)


def degrade(df, seed=5):
    if not len(df): return df
    rng = np.random.default_rng(seed); d = df.reset_index(drop=True).copy()
    d = d[rng.random(len(d)) >= 0.10].reset_index(drop=True)
    r = d.pnl_r.to_numpy(float); sr = d.spread_r.to_numpy(float)
    r = r - 0.3 * sr
    r = r - np.where(rng.random(len(d)) < 0.15, 1.5 * sr, 0.0)
    r = r - np.where(r <= -0.90, 0.5 * sr, 0.0)
    r = np.where(rng.random(len(d)) < 0.002, -3.0, r)
    d["pnl_r"] = r
    d["outcome"] = np.where(r > 0.01, "win", np.where(r < -0.01, "loss", "be"))
    return d


def main():
    r2s, dols = [], []
    for s in SYMS:
        m1 = hd.fetch(s, YEARS, tf="1min")
        a, b = run(s, m1)
        print(f"  {s}: {len(a)} setups", flush=True)
        r2s.append(a); dols.append(b)
    r2 = pd.concat(r2s, ignore_index=True); dl = pd.concat(dols, ignore_index=True)
    for df in (r2, dl):
        df["entry_time"] = pd.to_datetime(df["entry_time"])

    print("\n" + "=" * 108)
    print("REQUEST 1 · HTF sweep→5m BOS→1m pullback→1m trigger · 4 majors · 2010-2024 · causal+costs")
    print("=" * 108)
    for tpname, raw in [("TP = 2R fixed", r2), ("TP = next liquidity", dl)]:
        deg = degrade(raw.sort_values("entry_time"))
        st = edr.stats(deg); wk = max((raw.entry_time.max() - raw.entry_time.min()).days / 7, 1)
        print(f"\n  ── {tpname} ──")
        print(f"  realistic: N={st['n']} ({st['n']/wk:.2f}/wk) · WR {st['wr']:.1f}% · PF {st['pf']:.2f} · "
              f"{st['tot']:+.0f}R · DD {st['dd']:.0f}% · end ${st['bal']:,.0f}", flush=True)
        ok = True
        for name, x, y in SPLITS:
            g = deg[(deg.entry_time >= x) & (deg.entry_time < y)]; s2 = edr.stats(g)
            if s2 is None or s2["n"] < 30:
                print(f"    {name}: few"); ok = False; continue
            print(f"    {name}: N={s2['n']:>4} WR {s2['wr']:>4.0f}% PF {s2['pf']:>5.2f} {s2['tot']:>+6.0f}R")
            if s2["pf"] < 1.25: ok = False
        v = "✅ SURVIVOR" if ok and st["pf"] >= 1.3 else "❌ no edge"
        print(f"    VERDICT: {v}", flush=True)


if __name__ == "__main__":
    main()
