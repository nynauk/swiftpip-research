"""
CISD harvester — robustness + regime + HTF POI.
  1. STRESS: re-run the FTMO harvester on the recent base CISD under 1.0-2.0x
     execution costs → find the break-even cost multiple.
  2. FULL-HISTORY: harvester on the 2010-2024 distribution (was PF 0.97) + per-era
     → quantify how much the +EV depends on the recent regime.
  3. HTF POI: require the 5m CISD swing to have SWEPT a confirmed 1H swing (took
     higher-timeframe liquidity) — the classic ICT point-of-interest confluence.
"""
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import frequency_tune as fq
import realistic_ema_dol_rsi as edr
import all_pairs_2000_2026 as ap
from request1_htf import conf_swings
from cisd_dev import run as run_base, harvest_ev, SYMS, TRIG, HOLD, RR

edr.SL_SPREAD_PAD = True


def load5(sym, since):
    m5 = pd.read_pickle(os.path.join("ext_data", f"HD_{sym}_5min_2000-2024.pkl"))
    if since >= "2019-01-01":
        tail = ap.mt5_bars(ap.U[sym][1], mt5.TIMEFRAME_M5, n=99999)
        if tail is not None:
            m5 = pd.concat([m5[m5.index < "2025-01-01"], tail[tail.index >= "2025-01-01"]])
    return m5[(m5.index >= since) & (m5.index < "2025-01-01") if since < "2019-01-01" else (m5.index >= since)]


def stress_degrade(df, mult, seed=41):
    if not len(df):
        return df
    rng = np.random.default_rng(seed); d = df.reset_index(drop=True).copy()
    miss = min(0.10 + 0.05 * (mult - 1), 0.30)
    d = d[rng.random(len(d)) >= miss].reset_index(drop=True)
    r = d.pnl_r.to_numpy(float); sr = d.spread_r.to_numpy(float)
    r = r - 0.3 * mult * sr
    r = r - np.where(rng.random(len(d)) < 0.15, 1.5 * mult * sr, 0.0)
    r = r - np.where(r <= -0.90, 0.5 * mult * sr, 0.0)
    r = np.where(rng.random(len(d)) < 0.002 * mult, -3.0, r)
    return pd.Series(r)


def run_poi(sym, bars):
    """base CISD but the 5m swing must have swept a confirmed 1H swing (HTF POI)."""
    spread, slip, _ = fq.SPEC[sym]; pad = edr.sl_pad(sym, spread)
    o, h, l, c = bars.open.values, bars.high.values, bars.low.values, bars.close.values
    idx = bars.index; M = len(bars)
    shv, shi, slv, sli = conf_swings(bars, 3)
    htf = bars.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    csh1, _, csl1, _ = conf_swings(htf, 3)
    t1 = htf.index.values
    trades = []; busy = -1; last_sl = -1; last_sh = -1
    for i in range(5, M):
        if i <= busy:
            continue
        slj = int(sli[i]) if sli[i] == sli[i] else -1
        shj = int(shi[i]) if shi[i] == shi[i] else -1
        if slj > last_sl and slj >= 0:
            last_sl = slj
            hi_ix = np.searchsorted(t1, idx.values[slj]) - 1
            poi = hi_ix >= 0 and not np.isnan(csl1[hi_ix]) and l[slj] < csl1[hi_ix]
            if not poi:
                continue
            m0 = slj
            while m0 > 0 and c[m0 - 1] < o[m0 - 1]:
                m0 -= 1
            cisd = o[m0]; sw = l[slj]
            g = next((gg for gg in range(max(slj + 3, i), min(slj + TRIG, M)) if c[gg] > cisd), None)
            if g is None:
                continue
            entry = c[g] + spread + slip; sl = sw - pad; risk = entry - sl
            if risk <= 0:
                continue
            tp = entry + RR * risk; res = None
            for x in range(g + 1, min(g + HOLD, M)):
                if l[x] <= sl: res = (x, -1.0); break
                if h[x] >= tp: res = (x, RR); break
            if res is None:
                x = min(g + HOLD, M) - 1; res = (x, (c[x] - entry) / risk)
            busy = res[0]
            trades.append(dict(symbol=sym, entry_time=idx[g], exit_time=idx[res[0]], pnl_r=res[1],
                               spread_r=spread / risk,
                               outcome="win" if res[1] > 0.01 else ("loss" if res[1] < -0.01 else "be")))
            continue
        if shj > last_sh and shj >= 0:
            last_sh = shj
            hi_ix = np.searchsorted(t1, idx.values[shj]) - 1
            poi = hi_ix >= 0 and not np.isnan(csh1[hi_ix]) and h[shj] > csh1[hi_ix]
            if not poi:
                continue
            m0 = shj
            while m0 > 0 and c[m0 - 1] > o[m0 - 1]:
                m0 -= 1
            cisd = o[m0]; sw = h[shj]
            g = next((gg for gg in range(max(shj + 3, i), min(shj + TRIG, M)) if c[gg] < cisd), None)
            if g is None:
                continue
            entry = c[g] - slip; sl = sw + pad; risk = sl - entry
            if risk <= 0:
                continue
            tp = entry - RR * risk; res = None
            for x in range(g + 1, min(g + HOLD, M)):
                if h[x] + spread >= sl: res = (x, -1.0); break
                if l[x] + spread <= tp: res = (x, RR); break
            if res is None:
                x = min(g + HOLD, M) - 1; res = (x, (entry - (c[x] + spread)) / risk)
            busy = res[0]
            trades.append(dict(symbol=sym, entry_time=idx[g], exit_time=idx[res[0]], pnl_r=res[1],
                               spread_r=spread / risk,
                               outcome="win" if res[1] > 0.01 else ("loss" if res[1] < -0.01 else "be")))
    return pd.DataFrame(trades)


def main():
    have = mt5.initialize()
    recent = {s: load5(s, "2019-01-01") for s in SYMS}
    full = {s: load5(s, "2010-01-01") for s in SYMS}
    if have:
        mt5.shutdown()

    # ── 1. STRESS TEST (recent base) ──
    raw = pd.concat([run_base(s, recent[s]) for s in SYMS], ignore_index=True)
    raw["entry_time"] = pd.to_datetime(raw["entry_time"]); raw = raw.sort_values("entry_time")
    print("=" * 88)
    print("1 · ROBUSTNESS STRESS TEST · recent 2019-2026 base 5m CISD · harvester @1% risk")
    print("=" * 88)
    print(f"  {'cost×':>6} {'expectancy':>11} {'harvester EV/challenge':>24} {'funded%':>8}")
    for mult in (1.0, 1.25, 1.5, 1.75, 2.0):
        r = stress_degrade(raw, mult).values
        ev, fr, pp = harvest_ev(r, 0.01, n=8000)
        print(f"  {mult:>5.2f}x {r.mean():>+10.4f}R  ${ev:>+13.0f}  ({'+EV' if ev>0 else '-EV'}) {fr:>7.1%}", flush=True)

    # ── 2. FULL-HISTORY HARVESTER ──
    rawf = pd.concat([run_base(s, full[s]) for s in SYMS], ignore_index=True)
    rawf["entry_time"] = pd.to_datetime(rawf["entry_time"]); rawf = rawf.sort_values("entry_time")
    from cisd_dev import degrade
    degf = degrade(rawf)
    print("\n" + "=" * 88)
    print("2 · FULL-HISTORY HARVESTER · 2010-2024 base 5m CISD")
    print("=" * 88)
    st = edr.stats(degf)
    print(f"  full: N {st['n']} · PF {st['pf']:.2f} · expectancy {degf.pnl_r.mean():+.4f}R")
    ev, fr, pp = harvest_ev(degf.pnl_r.values, 0.01, n=8000)
    print(f"  harvester @1%: EV ${ev:+.0f} ({'+EV' if ev>0 else '-EV'}) · funded {fr:.1%}")
    degf["yr"] = pd.to_datetime(degf.entry_time).dt.year
    for lbl, a, b in [("2010-2014", 2010, 2015), ("2015-2019", 2015, 2020), ("2020-2024", 2020, 2025)]:
        g = degf[(degf.yr >= a) & (degf.yr < b)]
        e2, f2, _ = harvest_ev(g.pnl_r.values, 0.01, n=6000)
        print(f"    {lbl}: PF {edr.stats(g)['pf']:.2f} · exp {g.pnl_r.mean():+.4f}R · harvester EV ${e2:+.0f}", flush=True)

    # ── 3. HTF POI variant ──
    rawp = pd.concat([run_poi(s, recent[s]) for s in SYMS], ignore_index=True)
    rawp["entry_time"] = pd.to_datetime(rawp["entry_time"]); rawp = rawp.sort_values("entry_time")
    degp = degrade(rawp)
    print("\n" + "=" * 88)
    print("3 · HTF POI variant · CISD swing must sweep a confirmed 1H swing · recent")
    print("=" * 88)
    stp = edr.stats(degp)
    print(f"  N {stp['n']} ({stp['n']/((rawp.entry_time.max()-rawp.entry_time.min()).days/365.25):.0f}/yr) · "
          f"WR {stp['wr']:.1f}% · PF {stp['pf']:.2f} · expectancy {degp.pnl_r.mean():+.4f}R · DD {stp['dd']:.0f}%")
    print("  per-instrument PF: " + " · ".join(
        f"{s} {edr.stats(degp[degp.symbol==s])['pf']:.2f}" for s in SYMS if len(degp[degp.symbol==s]) > 20))
    for risk in (0.005, 0.01):
        ev, fr, pp = harvest_ev(degp.pnl_r.values, risk, n=8000)
        print(f"  harvester @{risk:.1%}: EV ${ev:+.0f} ({'+EV' if ev>0 else '-EV'}) · funded {fr:.1%} · P(profit) {pp:.1%}", flush=True)


if __name__ == "__main__":
    main()
