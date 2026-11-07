"""
CISD variant search on indices (US100/US500/GER40) to find the OPTIMAL config.
Base = confirmed-swing CISD, close-through entry, SL swing, 2R. Variants add:
  entry ∈ {close, fvg (displacement demand-FVG retrace), ifvg (inverted opposing FVG)}
  +MSS  = CISD trigger must also break a confirmed swing (structure shift)
  +HTF  = the CISD swing must tap a 1H FVG (higher-timeframe PD array / POI)
All causal, min-stop guarded, per-instrument spread + degradation. Best → harvester + OOS.
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
OHLC = {"open": "first", "high": "max", "low": "min", "close": "last"}
TRIG, HOLD, RETR, RR = 60, 96, 24, 2.0


def htf_fvgs(bars):
    htf = bars.resample("1h").agg(OHLC).dropna()
    H, L = htf.high.values, htf.low.values; t = htf.index.values
    bt, bb, btop, rt, rb, rtop = [], [], [], [], [], []
    for k in range(2, len(htf)):
        if L[k] > H[k - 2]:
            bt.append(t[k]); bb.append(H[k - 2]); btop.append(L[k])
        if H[k] < L[k - 2]:
            rt.append(t[k]); rb.append(H[k]); rtop.append(L[k - 2])
    return (np.array(bt), np.array(bb), np.array(btop)), (np.array(rt), np.array(rb), np.array(rtop))


def in_fvg(fvg, swing_t, price):
    tt, bot, top = fvg
    if len(tt) == 0:
        return False
    j = np.searchsorted(tt, swing_t)
    for i in range(j - 1, max(j - 60, -1), -1):
        if (swing_t - tt[i]) > np.timedelta64(48, "h"):
            break
        if bot[i] <= price <= top[i]:
            return True
    return False


def run(sym, bars, entry="close", mss=False, htf=False):
    spread, slip, _ = fq.SPEC[sym]; pad = edr.sl_pad(sym, spread)
    o, h, l, c = bars.open.values, bars.high.values, bars.low.values, bars.close.values
    idx = bars.index; tv = idx.values; M = len(bars)
    shv, shi, slv, sli = conf_swings(bars, 3)
    bull, bear = htf_fvgs(bars) if htf else (None, None)
    trades = []; busy = -1; ls = lsh = -1
    for i in range(5, M):
        if i <= busy:
            continue
        for side in ("L", "S"):
            if side == "L":
                sj = int(sli[i]) if sli[i] == sli[i] else -1
                if not (sj > ls and sj >= 0):
                    continue
                ls = sj
                if htf and not in_fvg(bull, tv[sj], l[sj]):
                    continue
                m0 = sj
                while m0 > 0 and c[m0 - 1] < o[m0 - 1]:
                    m0 -= 1
                cisd = o[m0]; sw = l[sj]
                g = next((gg for gg in range(max(sj + 3, i), min(sj + TRIG, M)) if c[gg] > cisd), None)
                if g is None:
                    continue
                if mss and not (not np.isnan(shv[g]) and c[g] > shv[g]):
                    continue
                leg = h[m0:g + 1].max() - l[m0:g + 1].min()
                if entry in ("fvg", "ifvg"):
                    fk = None
                    if entry == "fvg":
                        for k in range(m0 + 2, g + 1):
                            if l[k] > h[k - 2]:
                                fk = ("dem", l[k])
                    else:
                        for k in range(m0 + 2, sj + 1):
                            if h[k] < l[k - 2]:
                                fk = ("inv", l[k - 2])
                    if fk is None:
                        continue
                    lvl = fk[1]
                    r = next((rr for rr in range(g + 1, min(g + RETR, M)) if l[rr] <= lvl), None)
                    if r is None:
                        continue
                    ei = r; ep = lvl + spread + slip
                else:
                    ei = g; ep = c[g] + spread + slip
                sl = sw - pad; risk = ep - sl
                if risk <= 0 or risk < 0.20 * leg:
                    continue
                tp = ep + RR * risk; res = None
                for x in range(ei + 1, min(ei + HOLD, M)):
                    if l[x] <= sl: res = (x, -1.0); break
                    if h[x] >= tp: res = (x, RR); break
                if res is None:
                    x = min(ei + HOLD, M) - 1; res = (x, (c[x] - ep) / risk)
                busy = res[0]
                trades.append(dict(symbol=sym, entry_time=idx[ei], pnl_r=res[1], spread_r=spread / risk,
                                   outcome="win" if res[1] > 0.01 else ("loss" if res[1] < -0.01 else "be")))
                break
            else:
                sj = int(shi[i]) if shi[i] == shi[i] else -1
                if not (sj > lsh and sj >= 0):
                    continue
                lsh = sj
                if htf and not in_fvg(bear, tv[sj], h[sj]):
                    continue
                m0 = sj
                while m0 > 0 and c[m0 - 1] > o[m0 - 1]:
                    m0 -= 1
                cisd = o[m0]; sw = h[sj]
                g = next((gg for gg in range(max(sj + 3, i), min(sj + TRIG, M)) if c[gg] < cisd), None)
                if g is None:
                    continue
                if mss and not (not np.isnan(slv[g]) and c[g] < slv[g]):
                    continue
                leg = h[m0:g + 1].max() - l[m0:g + 1].min()
                if entry in ("fvg", "ifvg"):
                    fk = None
                    if entry == "fvg":
                        for k in range(m0 + 2, g + 1):
                            if h[k] < l[k - 2]:
                                fk = h[k]
                    else:
                        for k in range(m0 + 2, sj + 1):
                            if l[k] > h[k - 2]:
                                fk = h[k - 2]
                    if fk is None:
                        continue
                    lvl = fk
                    r = next((rr for rr in range(g + 1, min(g + RETR, M)) if h[rr] >= lvl), None)
                    if r is None:
                        continue
                    ei = r; ep = lvl - slip
                else:
                    ei = g; ep = c[g] - slip
                sl = sw + pad; risk = sl - ep
                if risk <= 0 or risk < 0.20 * leg:
                    continue
                tp = ep - RR * risk; res = None
                for x in range(ei + 1, min(ei + HOLD, M)):
                    if h[x] + spread >= sl: res = (x, -1.0); break
                    if l[x] + spread <= tp: res = (x, RR); break
                if res is None:
                    x = min(ei + HOLD, M) - 1; res = (x, (ep - (c[x] + spread)) / risk)
                busy = res[0]
                trades.append(dict(symbol=sym, entry_time=idx[ei], pnl_r=res[1], spread_r=spread / risk,
                                   outcome="win" if res[1] > 0.01 else ("loss" if res[1] < -0.01 else "be")))
                break
    return pd.DataFrame(trades)


def evaluate(data, entry, mss, htf):
    raw = pd.concat([run(s, data[s], entry, mss, htf) for s in IDX], ignore_index=True)
    if not len(raw):
        return None
    raw["entry_time"] = pd.to_datetime(raw["entry_time"])
    df = degrade(raw.sort_values("entry_time"))
    r = df.pnl_r.astype(float).values
    st = edr.stats(df); ls, _ = streaks(r)
    d = df.copy(); d["day"] = pd.to_datetime(d.entry_time).dt.normalize()
    daily = d.groupby("day").pnl_r.sum(); sh = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    return dict(n=st["n"], wr=st["wr"], pf=st["pf"], exp=r.mean(), tot=r.sum(), dd=st["dd"],
                maxls=max(ls) if ls else 0, sharpe=sh, df=df)


def main():
    have = mt5.initialize()
    rec = {s: load5(s, "2019-01-01") for s in IDX}
    full = {s: load5(s, "2010-01-01") for s in IDX}
    if have:
        mt5.shutdown()
    variants = [
        ("base (close)", "close", False, False),
        ("+MSS", "close", True, False),
        ("+HTF PDA", "close", False, True),
        ("+MSS+HTF", "close", True, True),
        ("FVG entry", "fvg", False, False),
        ("FVG+MSS", "fvg", True, False),
        ("FVG+HTF", "fvg", False, True),
        ("IFVG entry", "ifvg", False, False),
        ("IFVG+MSS", "ifvg", True, False),
        ("IFVG+HTF", "ifvg", False, True),
    ]
    print("=" * 98)
    print("CISD VARIANT SEARCH · indices (US100+US500+GER40) · recent 2019-2026 · ranked by Sharpe")
    print("=" * 98)
    print(f"  {'variant':16} {'N':>6} {'WR':>6} {'PF':>6} {'exp/tr':>9} {'Sharpe':>7} {'maxDD':>7} {'maxLS':>6} {'totR':>7}")
    rows = []
    for lbl, e, m, ht in variants:
        r = evaluate(rec, e, m, ht)
        if r is None:
            print(f"  {lbl:16} no trades"); continue
        rows.append((lbl, e, m, ht, r))
    for lbl, e, m, ht, r in sorted(rows, key=lambda x: -x[4]["sharpe"]):
        print(f"  {lbl:16} {r['n']:>6} {r['wr']:>5.1f}% {r['pf']:>6.3f} {r['exp']:>+8.4f}R "
              f"{r['sharpe']:>7.2f} {r['dd']:>6.0f}R {r['maxls']:>6} {r['tot']:>+6.0f}R", flush=True)

    best = max(rows, key=lambda x: x[4]["sharpe"])
    lbl, e, m, ht, r = best
    print(f"\n  >> OPTIMAL: {lbl} (Sharpe {r['sharpe']:.2f}, PF {r['pf']:.3f})", flush=True)
    print("  ── harvester on OPTIMAL (recent) ──")
    for risk in (0.005, 0.01):
        ei, fi, _ = harvest_iid(r["df"].pnl_r.values, risk, n=8000)
        es, fs, ps = harvest_seq(r["df"].pnl_r.values, risk, n=8000)
        print(f"    @{risk:.1%}: i.i.d. ${ei:+.0f} · SEQ(clustered) ${es:+.0f} ({fs:.0%} funded, {ps:.0%} profit)", flush=True)
    # OOS
    rawf = pd.concat([run(s, full[s], e, m, ht) for s in IDX], ignore_index=True)
    rawf["entry_time"] = pd.to_datetime(rawf["entry_time"]); dff = degrade(rawf.sort_values("entry_time"))
    stf = edr.stats(dff)
    es, fs, _ = harvest_seq(dff.pnl_r.values, 0.01, n=8000)
    print(f"  ── OOS full-history 2010-2024: PF {stf['pf']:.3f} · exp {dff.pnl_r.mean():+.4f}R · "
          f"harvester SEQ @1% ${es:+.0f}", flush=True)


if __name__ == "__main__":
    main()
