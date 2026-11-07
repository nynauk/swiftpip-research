"""
Optimal base CISD (indices) with TP = OPPOSING LIQUIDITY instead of fixed 2R.
  long  → TP = nearest confirmed swing HIGH above entry (buy-side draw on liquidity)
  short → TP = nearest confirmed swing LOW below entry
SL = swing (+spread+1pip). Variable RR (require 0.5-15R, else skip). Causal.
Compared head-to-head with fixed-2R; full stats + harvester + OOS.
"""
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import frequency_tune as fq
import realistic_ema_dol_rsi as edr
from request1_htf import conf_swings
from cisd_dev import degrade
from cisd_stats import streaks
from cisd_indices import load5, harvest_seq, harvest_iid

edr.SL_SPREAD_PAD = True
IDX = ["US100.cash", "US500.cash", "GER40.cash"]
TRIG, HOLD, N = 60, 96, 3
MIN_RR, MAX_RR, LIQ_LB = 0.5, 15.0, 300


def swing_vals(bars, n):
    h, l = bars.high.values, bars.low.values
    rmax = pd.Series(h).rolling(2 * n + 1, center=True).max().values
    rmin = pd.Series(l).rolling(2 * n + 1, center=True).min().values
    return np.where(h >= rmax, h, np.nan), np.where(l <= rmin, l, np.nan)


def run(sym, bars, mode="liq"):
    spread, slip, _ = fq.SPEC[sym]; pad = edr.sl_pad(sym, spread)
    o, h, l, c = bars.open.values, bars.high.values, bars.low.values, bars.close.values
    idx = bars.index; M = len(bars)
    shv, shi, slv, sli = conf_swings(bars, N)
    sh_arr, sl_arr = swing_vals(bars, N)
    trades = []; busy = -1; ls = lsh = -1
    for i in range(5, M):
        if i <= busy:
            continue
        for side in ("L", "S"):
            if side == "L":
                sj = int(sli[i]) if sli[i] == sli[i] else -1
                if not (sj > ls and sj >= 0):
                    continue
                ls = sj; m0 = sj
                while m0 > 0 and c[m0 - 1] < o[m0 - 1]:
                    m0 -= 1
                cisd = o[m0]; sw = l[sj]
                g = next((gg for gg in range(max(sj + 3, i), min(sj + TRIG, M)) if c[gg] > cisd), None)
                if g is None:
                    continue
                entry = c[g] + spread + slip; sl = sw - pad; risk = entry - sl
                if risk <= 0:
                    continue
                if mode == "liq":
                    best = None
                    for j in range(g - N, max(g - N - LIQ_LB, -1), -1):
                        if not np.isnan(sh_arr[j]) and sh_arr[j] > entry:
                            best = sh_arr[j] if best is None else min(best, sh_arr[j])
                    if best is None:
                        continue
                    tp = best; rr = (tp - entry) / risk
                    if not (MIN_RR <= rr <= MAX_RR):
                        continue
                else:
                    rr = 2.0; tp = entry + rr * risk
                res = None
                for x in range(g + 1, min(g + HOLD, M)):
                    if l[x] <= sl: res = (x, -1.0); break
                    if h[x] >= tp: res = (x, rr); break
                if res is None:
                    x = min(g + HOLD, M) - 1; res = (x, (c[x] - entry) / risk)
                busy = res[0]
                trades.append(dict(symbol=sym, entry_time=idx[g], pnl_r=res[1], rr=rr, spread_r=spread / risk,
                                   outcome="win" if res[1] > 0.01 else ("loss" if res[1] < -0.01 else "be")))
                break
            else:
                sj = int(shi[i]) if shi[i] == shi[i] else -1
                if not (sj > lsh and sj >= 0):
                    continue
                lsh = sj; m0 = sj
                while m0 > 0 and c[m0 - 1] > o[m0 - 1]:
                    m0 -= 1
                cisd = o[m0]; sw = h[sj]
                g = next((gg for gg in range(max(sj + 3, i), min(sj + TRIG, M)) if c[gg] < cisd), None)
                if g is None:
                    continue
                entry = c[g] - slip; sl = sw + pad; risk = sl - entry
                if risk <= 0:
                    continue
                if mode == "liq":
                    best = None
                    for j in range(g - N, max(g - N - LIQ_LB, -1), -1):
                        if not np.isnan(sl_arr[j]) and sl_arr[j] < entry:
                            best = sl_arr[j] if best is None else max(best, sl_arr[j])
                    if best is None:
                        continue
                    tp = best; rr = (entry - tp) / risk
                    if not (MIN_RR <= rr <= MAX_RR):
                        continue
                else:
                    rr = 2.0; tp = entry - rr * risk
                res = None
                for x in range(g + 1, min(g + HOLD, M)):
                    if h[x] + spread >= sl: res = (x, -1.0); break
                    if l[x] + spread <= tp: res = (x, rr); break
                if res is None:
                    x = min(g + HOLD, M) - 1; res = (x, (entry - (c[x] + spread)) / risk)
                busy = res[0]
                trades.append(dict(symbol=sym, entry_time=idx[g], pnl_r=res[1], rr=rr, spread_r=spread / risk,
                                   outcome="win" if res[1] > 0.01 else ("loss" if res[1] < -0.01 else "be")))
                break
    return pd.DataFrame(trades)


def stats(df):
    r = df.pnl_r.astype(float).values; st = edr.stats(df); lst, _ = streaks(r)
    d = df.copy(); d["day"] = pd.to_datetime(d.entry_time).dt.normalize()
    dd = d.groupby("day").pnl_r.sum(); sh = dd.mean() / dd.std() * np.sqrt(252) if dd.std() > 0 else 0
    return st, r.mean(), max(lst) if lst else 0, sh


def main():
    have = mt5.initialize()
    rec = {s: load5(s, "2019-01-01") for s in IDX}
    full = {s: load5(s, "2010-01-01") for s in IDX}
    if have:
        mt5.shutdown()
    print("=" * 92)
    print("CISD exit: OPPOSING LIQUIDITY vs FIXED 2R · indices · recent 2019-2026")
    print("=" * 92)
    print(f"  {'exit':16} {'N':>6} {'WR':>6} {'medRR':>6} {'PF':>6} {'exp/tr':>9} {'Sharpe':>7} {'maxDD':>7} {'maxLS':>6}")
    keep = {}
    for lbl, mode in [("fixed 2R", "fixed"), ("opposing liquidity", "liq")]:
        raw = pd.concat([run(s, rec[s], mode) for s in IDX], ignore_index=True)
        raw["entry_time"] = pd.to_datetime(raw["entry_time"])
        df = degrade(raw.sort_values("entry_time")); keep[lbl] = (df, raw)
        st, exp, mls, sh = stats(df)
        print(f"  {lbl:16} {st['n']:>6} {st['wr']:>5.1f}% {raw.rr.median():>6.2f} {st['pf']:>6.3f} "
              f"{exp:>+8.4f}R {sh:>7.2f} {st['dd']:>6.0f}R {mls:>6}", flush=True)

    df = keep["opposing liquidity"][0]
    print("\n  ── opposing-liquidity: per-instrument & harvester ──")
    for s in IDX:
        g = df[df.symbol == s]; s2 = edr.stats(g)
        print(f"    {s:12} N={s2['n']:>5} WR {s2['wr']:>4.0f}% PF {s2['pf']:>5.2f}")
    for risk in (0.005, 0.01):
        ei, fi, _ = harvest_iid(df.pnl_r.values, risk, n=8000)
        es, fs, ps = harvest_seq(df.pnl_r.values, risk, n=8000)
        print(f"    @{risk:.1%}: i.i.d. ${ei:+.0f} · SEQ ${es:+.0f} ({fs:.0%} funded, {ps:.0%} profit)", flush=True)

    rawf = pd.concat([run(s, full[s], "liq") for s in IDX], ignore_index=True)
    rawf["entry_time"] = pd.to_datetime(rawf["entry_time"]); dff = degrade(rawf.sort_values("entry_time"))
    stf = edr.stats(dff); es, fs, _ = harvest_seq(dff.pnl_r.values, 0.01, n=8000)
    print(f"\n  ── OOS full-history 2010-2024 (liq): PF {stf['pf']:.3f} · exp {dff.pnl_r.mean():+.4f}R · "
          f"harvester SEQ @1% ${es:+.0f}", flush=True)


if __name__ == "__main__":
    main()
