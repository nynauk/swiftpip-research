"""
Standalone 1-minute CISD on UK100 (FTSE) — strictly causal.
  bullish: a confirmed swing LOW ends a down-leg; CISD line = open of the FIRST
           candle of the consecutive down-run into that low. When a bar (at/after
           the low's confirmation) CLOSES above the CISD line → enter LONG.
  bearish: mirror off a confirmed swing HIGH.
  SL = the swing extreme (+spread+1pip) · TP = fixed R (2 and 3 tested).
One trade at a time. Per-instrument spread on fills + execution degradation +
era-split survivor gate. M1 2010-2024 (HistData) + MT5 tail.
"""
import os
import numpy as np
import pandas as pd

import frequency_tune as fq
import realistic_ema_dol_rsi as edr
import histdata_fetch as hd
from request1_htf import conf_swings

edr.SL_SPREAD_PAD = True
SYM = "UK100.cash"
YEARS = list(range(2010, 2025))
SPLITS = [("2010-2014", "2010-01-01", "2015-01-01"),
          ("2015-2019", "2015-01-01", "2020-01-01"),
          ("2020-2024", "2020-01-01", "2025-01-01")]
N = 3               # swing confirmation lag
TRIG_WIN = 60       # bars after confirmation to get the CISD close-through
MAX_HOLD = 480      # bars to resolve


def run(m1, rr):
    spread, slip, _ = fq.SPEC[SYM]; pad = edr.sl_pad(SYM, spread)
    o, h, l, c = m1.open.values, m1.high.values, m1.low.values, m1.close.values
    idx = m1.index; M = len(m1)
    shv, shi, slv, sli = conf_swings(m1, N)
    trades = []; busy = -1; last_sl = -1; last_sh = -1
    for i in range(N + 2, M):
        if i <= busy:
            continue
        slj = int(sli[i]) if sli[i] == sli[i] else -1
        shj = int(shi[i]) if shi[i] == shi[i] else -1
        # ── bullish CISD off a newly-confirmed swing low ──
        if slj > last_sl and slj >= 0:
            last_sl = slj
            m0 = slj
            while m0 > 0 and c[m0 - 1] < o[m0 - 1]:
                m0 -= 1
            cisd = o[m0]; swlow = l[slj]
            g = next((gg for gg in range(max(slj + N, i), min(slj + TRIG_WIN, M))
                      if c[gg] > cisd), None)
            if g is not None:
                entry = c[g] + spread + slip; sl = swlow - pad; risk = entry - sl
                if risk > 0:
                    tp = entry + rr * risk; res = None
                    for x in range(g + 1, min(g + MAX_HOLD, M)):
                        if l[x] <= sl: res = (x, -1.0); break
                        if h[x] >= tp: res = (x, rr); break
                    if res is None:
                        x = min(g + MAX_HOLD, M) - 1; res = (x, (c[x] - entry) / risk)
                    busy = res[0]
                    trades.append(dict(symbol=SYM, entry_time=idx[g], exit_time=idx[res[0]],
                                       direction="long", pnl_r=res[1], spread_r=spread / risk,
                                       outcome="win" if res[1] > 0.01 else ("loss" if res[1] < -0.01 else "be")))
                    continue
        # ── bearish CISD off a newly-confirmed swing high ──
        if shj > last_sh and shj >= 0:
            last_sh = shj
            m0 = shj
            while m0 > 0 and c[m0 - 1] > o[m0 - 1]:
                m0 -= 1
            cisd = o[m0]; swhigh = h[shj]
            g = next((gg for gg in range(max(shj + N, i), min(shj + TRIG_WIN, M))
                      if c[gg] < cisd), None)
            if g is not None:
                entry = c[g] - slip; sl = swhigh + pad; risk = sl - entry
                if risk > 0:
                    tp = entry - rr * risk; res = None
                    for x in range(g + 1, min(g + MAX_HOLD, M)):
                        if h[x] + spread >= sl: res = (x, -1.0); break
                        if l[x] + spread <= tp: res = (x, rr); break
                    if res is None:
                        x = min(g + MAX_HOLD, M) - 1; res = (x, (entry - (c[x] + spread)) / risk)
                    busy = res[0]
                    trades.append(dict(symbol=SYM, entry_time=idx[g], exit_time=idx[res[0]],
                                       direction="short", pnl_r=res[1], spread_r=spread / risk,
                                       outcome="win" if res[1] > 0.01 else ("loss" if res[1] < -0.01 else "be")))
    return pd.DataFrame(trades)


def degrade(df, seed=13):
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
    print("Loading UK100 M1 2010-2024...", flush=True)
    m1 = hd.fetch(SYM, YEARS, tf="1min")
    print(f"  {len(m1):,} bars", flush=True)
    print("\n" + "=" * 96)
    print("1-MINUTE CISD · UK100 · causal · costs+degradation · survivor gate PF>=1.25 every era")
    print("=" * 96)
    for rr in (2.0, 3.0):
        raw = run(m1, rr)
        raw["entry_time"] = pd.to_datetime(raw["entry_time"])
        deg = degrade(raw.sort_values("entry_time"))
        st = edr.stats(deg); yrs = (raw.entry_time.max() - raw.entry_time.min()).days / 365.25
        print(f"\n  ══ {rr:.0f}R ══  N={st['n']} ({st['n']/yrs:.0f}/yr) · WR {st['wr']:.1f}% "
              f"(BE {100/(1+rr):.0f}%) · PF {st['pf']:.2f} · {st['tot']:+.0f}R · DD {st['dd']:.0f}% · "
              f"${st['bal']:,.0f}", flush=True)
        ok = True
        for name, a, b in SPLITS:
            g = deg[(deg.entry_time >= a) & (deg.entry_time < b)]; s2 = edr.stats(g)
            if s2 is None or s2["n"] < 30:
                print(f"      {name}: few"); ok = False; continue
            print(f"      {name}: N={s2['n']:>5} WR {s2['wr']:>4.0f}% PF {s2['pf']:>5.2f} {s2['tot']:>+7.0f}R")
            if s2["pf"] < 1.25:
                ok = False
        print(f"      VERDICT: {'✅ SURVIVOR' if ok and st['pf'] >= 1.3 else '❌ no edge'}", flush=True)


if __name__ == "__main__":
    main()
