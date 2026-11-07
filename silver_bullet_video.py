"""
ICT "Silver Bullet" for indices (the video's exact rules) — faithful causal backtest.
Windows (NY/EST, HistData clock): AM 09:30-11:30, PM 13:30-15:30.
  1. LIQUIDITY SWEEP: a 5m bar takes out a confirmed swing low (→long) or high
     (→short) and CLOSES back inside (rejection = the HTF-PDA rejection proxy).
  2. MSS: after the sweep, a 5m close breaks a confirmed swing in the reversal
     direction (displacement / market-structure shift).
  3. ENTRY at the 50% EQUILIBRIUM of the displacement leg (presenter's optimal
     entry — buy discount / sell premium), filled on the retrace.
  4. SL beyond the swing that caused the MSS (+spread+1pip). TP = 2.5R.
One trade per window. Index proxies, 5m, causal, costs + degradation, era splits.
"""
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import frequency_tune as fq
import realistic_ema_dol_rsi as edr
import all_pairs_2000_2026 as ap
from request1_htf import conf_swings

edr.SL_SPREAD_PAD = True
SYMS = ["US500.cash", "US100.cash", "GER40.cash"]
SPLITS = [("2010-2014", "2010-01-01", "2015-01-01"),
          ("2015-2019", "2015-01-01", "2020-01-01"),
          ("2020-2024", "2020-01-01", "2025-01-01")]
RR = 2.5
MSS_WIN = 12; RETR_WIN = 12; MAX_HOLD = 96
AM = (9 * 60 + 30, 11 * 60 + 30); PM = (13 * 60 + 30, 15 * 60 + 30)


def load5(sym):
    p = os.path.join("ext_data", f"HD_{sym}_5min_2000-2024.pkl")
    m5 = pd.read_pickle(p)
    tail = ap.mt5_bars(ap.U[sym][1], mt5.TIMEFRAME_M5, n=99999)
    if tail is not None:
        m5 = pd.concat([m5[m5.index < "2025-01-01"], tail[tail.index >= "2025-01-01"]])
    return m5


def run(sym, m5):
    spread, slip, _ = fq.SPEC[sym]; pad = edr.sl_pad(sym, spread)
    o, h, l, c = m5.open.values, m5.high.values, m5.low.values, m5.close.values
    idx = m5.index; M = len(m5)
    mod = idx.hour.values * 60 + idx.minute.values
    day = idx.normalize().values
    csh_v, _, csl_v, _ = conf_swings(m5, 3)
    trades = []
    for w0, w1 in (AM, PM):
        wb = np.where((mod >= w0) & (mod < w1))[0]
        if not len(wb):
            continue
        splits = np.where(np.concatenate([[True], day[wb][1:] != day[wb][:-1]]))[0]
        for si in range(len(splits)):
            a = wb[splits[si]]; b = wb[splits[si + 1] - 1] if si + 1 < len(splits) else wb[-1]
            took = False
            for k in range(a, b + 1):
                if took:
                    break
                bias = None
                if not np.isnan(csl_v[k]) and l[k] < csl_v[k] and c[k] > csl_v[k]:
                    bias = "long"
                elif not np.isnan(csh_v[k]) and h[k] > csh_v[k] and c[k] < csh_v[k]:
                    bias = "short"
                if bias is None:
                    continue
                mss = None
                for mb in range(k + 1, min(k + MSS_WIN, M)):
                    if bias == "long" and not np.isnan(csh_v[mb]) and c[mb] > csh_v[mb]:
                        mss = mb; break
                    if bias == "short" and not np.isnan(csl_v[mb]) and c[mb] < csl_v[mb]:
                        mss = mb; break
                if mss is None:
                    continue
                leg_lo = l[k:mss + 1].min(); leg_hi = h[k:mss + 1].max()
                eq = (leg_lo + leg_hi) / 2.0
                if bias == "long":
                    r = next((rr for rr in range(mss + 1, min(mss + RETR_WIN, M)) if l[rr] <= eq), None)
                    if r is None:
                        continue
                    entry = eq + spread + slip; sl = leg_lo - pad; risk = entry - sl
                    if risk <= 0:
                        continue
                    tp = entry + RR * risk; res = None
                    for x in range(r + 1, min(r + MAX_HOLD, M)):
                        if l[x] <= sl: res = (x, -1.0); break
                        if h[x] >= tp: res = (x, RR); break
                    if res is None:
                        x = min(r + MAX_HOLD, M) - 1; res = (x, (c[x] - entry) / risk)
                else:
                    r = next((rr for rr in range(mss + 1, min(mss + RETR_WIN, M)) if h[rr] >= eq), None)
                    if r is None:
                        continue
                    entry = eq - slip; sl = leg_hi + pad; risk = sl - entry
                    if risk <= 0:
                        continue
                    tp = entry - RR * risk; res = None
                    for x in range(r + 1, min(r + MAX_HOLD, M)):
                        if h[x] + spread >= sl: res = (x, -1.0); break
                        if l[x] + spread <= tp: res = (x, RR); break
                    if res is None:
                        x = min(r + MAX_HOLD, M) - 1; res = (x, (entry - (c[x] + spread)) / risk)
                trades.append(dict(symbol=sym, entry_time=idx[r], exit_time=idx[res[0]], direction=bias,
                                   pnl_r=res[1], spread_r=spread / risk,
                                   outcome="win" if res[1] > 0.01 else ("loss" if res[1] < -0.01 else "be")))
                took = True
    return pd.DataFrame(trades)


def degrade(df, seed=23):
    if not len(df):
        return df
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
    have = mt5.initialize()
    allt = [run(s, load5(s)) for s in SYMS]
    if have:
        mt5.shutdown()
    raw = pd.concat(allt, ignore_index=True); raw["entry_time"] = pd.to_datetime(raw["entry_time"])
    deg = degrade(raw.sort_values("entry_time"))
    st = edr.stats(deg); yrs = (raw.entry_time.max() - raw.entry_time.min()).days / 365.25
    print("=" * 96)
    print(f"SILVER BULLET (video rules · indices) · sweep+MSS+50%-equilibrium · {RR}R · causal+costs")
    print("=" * 96)
    print(f"  N={st['n']} ({st['n']/yrs:.0f}/yr) · WR {st['wr']:.1f}% (BE {100/(1+RR):.0f}%) · "
          f"PF {st['pf']:.2f} · {st['tot']:+.0f}R · DD {st['dd']:.0f}% · ${st['bal']:,.0f}")
    ok = True
    for name, a, b in SPLITS:
        g = deg[(deg.entry_time >= a) & (deg.entry_time < b)]; s2 = edr.stats(g)
        if s2 is None or s2["n"] < 25:
            print(f"    {name}: few"); ok = False; continue
        print(f"    {name}: N={s2['n']:>4} WR {s2['wr']:>4.0f}% PF {s2['pf']:>5.2f} {s2['tot']:>+6.0f}R")
        if s2["pf"] < 1.25:
            ok = False
    for s in SYMS:
        g = deg[deg.symbol == s]; s2 = edr.stats(g)
        if s2:
            print(f"    {s}: N={s2['n']:>4} WR {s2['wr']:>4.0f}% PF {s2['pf']:>5.2f}")
    # also test entering at market on the MSS (no 50% wait) as a robustness check
    print(f"\n  VERDICT: {'SURVIVOR — develop it' if ok and st['pf'] >= 1.3 else 'no edge after costs'}")


if __name__ == "__main__":
    main()
