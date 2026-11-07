"""
CISD head-to-head: 3-minute vs 5-minute. Same causal engine, same costs.
A confirmed swing leg ends → CISD line = open of the leg's origin candle → when
price closes back through it, enter the reversal → SL at swing, TP = 2R.
FX majors + indices, 2010-2024, era-split survivor gate. (1m version: UK100 0.85.)
"""
import os
import numpy as np
import pandas as pd

import frequency_tune as fq
import realistic_ema_dol_rsi as edr
import histdata_fetch as hd
from request1_htf import conf_swings

edr.SL_SPREAD_PAD = True
SYMS = ["EURUSD", "GBPUSD", "USDJPY", "US500.cash", "US100.cash", "GER40.cash", "UK100.cash"]
YEARS = list(range(2010, 2025))
SPLITS = [("2010-2014", "2010-01-01", "2015-01-01"),
          ("2015-2019", "2015-01-01", "2020-01-01"),
          ("2020-2024", "2020-01-01", "2025-01-01")]
RR = 2.0


def run_cisd(sym, bars, trig_win, max_hold):
    spread, slip, _ = fq.SPEC[sym]; pad = edr.sl_pad(sym, spread)
    o, h, l, c = bars.open.values, bars.high.values, bars.low.values, bars.close.values
    idx = bars.index; M = len(bars)
    shv, shi, slv, sli = conf_swings(bars, 3)
    trades = []; busy = -1; last_sl = -1; last_sh = -1
    for i in range(5, M):
        if i <= busy:
            continue
        slj = int(sli[i]) if sli[i] == sli[i] else -1
        shj = int(shi[i]) if shi[i] == shi[i] else -1
        if slj > last_sl and slj >= 0:
            last_sl = slj
            m0 = slj
            while m0 > 0 and c[m0 - 1] < o[m0 - 1]:
                m0 -= 1
            cisd = o[m0]; swlow = l[slj]
            g = next((gg for gg in range(max(slj + 3, i), min(slj + trig_win, M)) if c[gg] > cisd), None)
            if g is not None:
                entry = c[g] + spread + slip; sl = swlow - pad; risk = entry - sl
                if risk > 0:
                    tp = entry + RR * risk; res = None
                    for x in range(g + 1, min(g + max_hold, M)):
                        if l[x] <= sl: res = (x, -1.0); break
                        if h[x] >= tp: res = (x, RR); break
                    if res is None:
                        x = min(g + max_hold, M) - 1; res = (x, (c[x] - entry) / risk)
                    busy = res[0]
                    trades.append(dict(symbol=sym, entry_time=idx[g], exit_time=idx[res[0]],
                                       pnl_r=res[1], spread_r=spread / risk,
                                       outcome="win" if res[1] > 0.01 else ("loss" if res[1] < -0.01 else "be")))
                    continue
        if shj > last_sh and shj >= 0:
            last_sh = shj
            m0 = shj
            while m0 > 0 and c[m0 - 1] > o[m0 - 1]:
                m0 -= 1
            cisd = o[m0]; swhigh = h[shj]
            g = next((gg for gg in range(max(shj + 3, i), min(shj + trig_win, M)) if c[gg] < cisd), None)
            if g is not None:
                entry = c[g] - slip; sl = swhigh + pad; risk = sl - entry
                if risk > 0:
                    tp = entry - RR * risk; res = None
                    for x in range(g + 1, min(g + max_hold, M)):
                        if h[x] + spread >= sl: res = (x, -1.0); break
                        if l[x] + spread <= tp: res = (x, RR); break
                    if res is None:
                        x = min(g + max_hold, M) - 1; res = (x, (entry - (c[x] + spread)) / risk)
                    busy = res[0]
                    trades.append(dict(symbol=sym, entry_time=idx[g], exit_time=idx[res[0]],
                                       pnl_r=res[1], spread_r=spread / risk,
                                       outcome="win" if res[1] > 0.01 else ("loss" if res[1] < -0.01 else "be")))
    return pd.DataFrame(trades)


def degrade(df, seed=31):
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


def report(tf, raw):
    raw["entry_time"] = pd.to_datetime(raw["entry_time"])
    deg = degrade(raw.sort_values("entry_time"))
    st = edr.stats(deg); yrs = (raw.entry_time.max() - raw.entry_time.min()).days / 365.25
    print(f"\n  ══ {tf} CISD ══  N={st['n']} ({st['n']/yrs:.0f}/yr) · WR {st['wr']:.1f}% "
          f"(BE {100/(1+RR):.0f}%) · PF {st['pf']:.2f} · {st['tot']:+.0f}R · DD {st['dd']:.0f}%", flush=True)
    ok = True
    for name, a, b in SPLITS:
        g = deg[(deg.entry_time >= a) & (deg.entry_time < b)]; s2 = edr.stats(g)
        if s2 is None or s2["n"] < 30:
            print(f"      {name}: few"); ok = False; continue
        print(f"      {name}: N={s2['n']:>5} WR {s2['wr']:>4.0f}% PF {s2['pf']:>5.2f} {s2['tot']:>+7.0f}R")
        if s2["pf"] < 1.25:
            ok = False
    for s in SYMS:
        g = deg[deg.symbol == s]; s2 = edr.stats(g)
        if s2 and s2["n"] > 30:
            print(f"      {s:12} PF {s2['pf']:.2f} (n{s2['n']})")
    print(f"      VERDICT: {'SURVIVOR' if ok and st['pf'] >= 1.3 else 'no edge'}", flush=True)


def main():
    print("=" * 92)
    print("CISD · 3-MINUTE vs 5-MINUTE · causal · costs+degradation · 2R · FX majors + indices")
    print("=" * 92)
    for tf, rule, trig, hold in [("3m", "3min", 100, 160), ("5m", "5min", 60, 96)]:
        allt = []
        for s in SYMS:
            try:
                bars = hd.fetch(s, YEARS, tf=rule)
            except Exception:
                continue
            if len(bars) > 5000:
                allt.append(run_cisd(s, bars, trig, hold))
        report(tf, pd.concat([x for x in allt if len(x)], ignore_index=True))


if __name__ == "__main__":
    main()
