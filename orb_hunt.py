"""
SESSION EDGE HUNT — opening-range breakout (ORB) on index CFDs, strictly causal.
At the cash-open anchor, take the first W minutes' range; after it completes,
enter on an M5 CLOSE breaking out; stop = opposite side of the range (+spread+1pip
pad); exit at +2R target or the session cutoff close. One trade per day per
instrument. Costs on every fill. Split filter: PF>=1.25 in EVERY era or dead.

Data: HistData M5 2010-2024 (EST timestamps, no DST → anchors tested at both
winter/summer clock positions; a real edge must survive the smear).
"""
import os
import numpy as np
import pandas as pd

import frequency_tune as fq
import realistic_ema_dol_rsi as edr
import all_pairs_2000_2026 as ap
import find_edge as fe

edr.SL_SPREAD_PAD = True
SPLITS = [("2010-2015", "2010-01-01", "2016-01-01"),
          ("2016-2020", "2016-01-01", "2021-01-01"),
          ("2021-2024", "2021-01-01", "2025-01-01")]

US = ["US500.cash", "US100.cash"]
EU = ["GER40.cash", "UK100.cash", "FRA40.cash"]


def run_orb(sym, df, anchor_hhmm, width_min=30, tgt_r=2.0, sess_hours=6):
    spread, slip, _ = fq.SPEC[sym]
    pad = edr.sl_pad(sym, spread)
    hh, mm = anchor_hhmm
    d = df.copy()
    d["day"] = d.index.normalize()
    trades = []
    nbars = width_min // 5
    for day, g in d.groupby("day"):
        idx = g.index
        anchor = day + pd.Timedelta(hours=hh, minutes=mm)
        rng = g[(idx >= anchor) & (idx < anchor + pd.Timedelta(minutes=width_min))]
        if len(rng) < nbars:
            continue
        rhi, rlo = rng.high.max(), rng.low.min()
        if rhi - rlo <= 0:
            continue
        after = g[(idx >= anchor + pd.Timedelta(minutes=width_min)) &
                  (idx < anchor + pd.Timedelta(hours=sess_hours))]
        if len(after) < 3:
            continue
        h = after.high.values; l = after.low.values; c = after.close.values
        ei = None
        for j in range(len(after)):
            if c[j] > rhi:
                direction = "long"; entry = c[j] + spread + slip
                stop = rlo - pad; risk = entry - stop; ei = j; break
            if c[j] < rlo:
                direction = "short"; entry = c[j] - slip
                stop = rhi + pad; risk = stop - entry; ei = j; break
        if ei is None or risk <= 0:
            continue
        tgt = entry + tgt_r * risk if direction == "long" else entry - tgt_r * risk
        oc_px = None
        for j in range(ei + 1, len(after)):
            if direction == "long":
                if l[j] <= stop: oc_px = stop; xj = j; break
                if h[j] >= tgt: oc_px = tgt; xj = j; break
            else:
                if h[j] + spread >= stop: oc_px = stop; xj = j; break
                if l[j] + spread <= tgt: oc_px = tgt; xj = j; break
        if oc_px is None:
            xj = len(after) - 1
            oc_px = c[xj] if direction == "long" else c[xj] + spread
        r = (oc_px - entry) / risk if direction == "long" else (entry - oc_px) / risk
        trades.append(dict(symbol=sym, entry_time=after.index[ei], exit_time=after.index[xj],
                           pnl_r=r, spread_r=spread / risk,
                           outcome="win" if r > 0.01 else ("loss" if r < -0.01 else "be")))
    return pd.DataFrame(trades)


def evaluate(tag, dfs):
    df = pd.concat([x for x in dfs if len(x)], ignore_index=True)
    if not len(df):
        print(f"  {tag:34} no trades"); return None
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df = fe.degrade(df.sort_values("entry_time"), seed=hash(tag) % 2**31)
    cells = []; ok = True
    for name, a, b in SPLITS:
        g = df[(df.entry_time >= a) & (df.entry_time < b)]
        st = edr.stats(g)
        if st is None or st["n"] < 50:
            cells.append(f"{'—':>12}"); ok = False; continue
        cells.append(f"{st['pf']:>5.2f}/{st['n']:>5}")
        if st["pf"] < 1.25:
            ok = False
    st = edr.stats(df)
    flag = " ✅" if ok and st["pf"] >= 1.3 else ""
    print(f"  {tag:34} {''.join(cells)}  | all PF {st['pf']:>5.2f} N {st['n']:>5} "
          f"DD {st['dd']:>6.1f}%{flag}", flush=True)
    return tag, df, st, ok


def main():
    data = {}
    for s in US + EU:
        p = os.path.join("ext_data", f"HD_{s}_5min_2000-2024.pkl")
        if os.path.exists(p):
            m5 = pd.read_pickle(p)
            data[s] = m5[m5.index >= "2010-01-01"]
    print(f"loaded {len(data)} index M5 series", flush=True)
    print("=" * 118)
    print("ORB HUNT · index cash-open breakout · causal · costed · survivor = PF>=1.25 every era")
    print("=" * 118)
    print(f"  {'config':34} {'10-15':>12} {'16-20':>12} {'21-24':>12}  | overall", flush=True)
    print("-" * 118, flush=True)
    results = []
    # US cash open: 09:30 ET = 09:30 EST (winter) / 08:30 EST (summer) in HistData clock
    for anchor in [(9, 30), (8, 30)]:
        for tgt in [2.0, 3.0]:
            dfs = [run_orb(s, data[s], anchor, 30, tgt) for s in US if s in data]
            r = evaluate(f"ORB US open {anchor[0]:02d}:{anchor[1]:02d} W30 tgt{tgt}", dfs)
            if r: results.append(r)
    # EU cash open 09:00 CET = 03:00 EST (winter) / 02:00 (summer)
    for anchor in [(3, 0), (2, 0)]:
        for tgt in [2.0, 3.0]:
            dfs = [run_orb(s, data[s], anchor, 30, tgt) for s in EU if s in data]
            r = evaluate(f"ORB EU open {anchor[0]:02d}:{anchor[1]:02d} W30 tgt{tgt}", dfs)
            if r: results.append(r)
    # wider range window variant on the better anchor
    for anchor, basket, nm in [((9, 30), US, "US"), ((3, 0), EU, "EU")]:
        dfs = [run_orb(s, data[s], anchor, 60, 2.0) for s in basket if s in data]
        r = evaluate(f"ORB {nm} open W60 tgt2.0", dfs)
        if r: results.append(r)

    surv = [(t, st) for t, df, st, ok in results if ok and st["pf"] >= 1.3]
    print("\n  SURVIVORS:", flush=True)
    if surv:
        for t, st in surv:
            print(f"    {t}: PF {st['pf']:.2f} N {st['n']} DD {st['dd']:.1f}%", flush=True)
    else:
        print("    none.", flush=True)


if __name__ == "__main__":
    main()
