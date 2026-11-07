"""
Make 1% base risk viable for FTMO by DRAWDOWN THROTTLING.

Flat 1% blows the -10% max-loss limit. Fix: scale position size down as the
account draws down, so you can never breach. Base 1% at the equity peak, cut
to a fraction as drawdown deepens:

    drawdown from peak     risk multiplier
        0 - 3%                 x1.00  (full 1%)
        3 - 5%                 x0.50  (0.5%)
        5 - 7%                 x0.25  (0.25%)
        7%+                    x0.10  (0.1%)  ← near the floor, tiny bets

Management = partial@1R (the survivor). Daily self-stop at -4%.
"""
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import main as bt
import eurusd_basic as eb

ACCOUNT = 10_000; FLOOR = 9_000; SPLIT = 0.80
DAILY_STOP = 4.0

mt5.initialize()
df_h4 = bt.fetch_data("EURUSD", mt5.TIMEFRAME_H4,  bt.START_DATE, bt.END_DATE, tf_name="H4")
df_15 = bt.fetch_data("EURUSD", mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
mt5.shutdown()

def prep(d):
    d = d[d["outcome"] != "open"].sort_values("entry_time").reset_index(drop=True)
    d["date"] = pd.to_datetime(d["entry_time"]).dt.date
    d["month"] = pd.to_datetime(d["entry_time"]).dt.to_period("M")
    return d

PART = prep(eb.run(df_h4, df_15, rr=3.0, partial_at=1.0))
pnl_all = PART["pnl_r"].values
dates_all = PART["date"].values


def throttle(dd_pct):
    if dd_pct < 3:  return 1.00
    if dd_pct < 5:  return 0.50
    if dd_pct < 7:  return 0.25
    return 0.10


def challenge(start, base_risk, use_throttle, target=10.0, daily=5.0, overall=10.0, mind=4):
    eq = 0.0; peak = 0.0; dpl = 0.0; cur = None; td = set()
    for i in range(start, min(start + 500, len(pnl_all))):
        dt = dates_all[i]
        if dt != cur: cur = dt; dpl = 0.0
        if dpl <= -DAILY_STOP: continue
        dd = peak - eq
        r = base_risk * (throttle(dd) if use_throttle else 1.0)
        pl = r * pnl_all[i]; eq += pl; dpl += pl; td.add(dt)
        peak = max(peak, eq)
        if dpl <= -daily or eq <= -overall: return "FAIL", len(td)
        if eq >= target and len(td) >= mind: return "PASS", len(td)
    return "INC", len(td)


def pass_stats(base_risk, use_throttle):
    starts = [i for i in range(len(PART)) if i == 0 or dates_all[i] != dates_all[i-1]]
    starts = [s for s in starts if len(PART) - s >= 120]
    p = f = 0; days = []
    for s in starts:
        r, nd = challenge(s, base_risk, use_throttle)
        if r == "PASS": p += 1; days.append(nd)
        elif r == "FAIL": f += 1
    tot = p + f
    return (p/tot*100 if tot else 0), (int(np.median(days)) if days else 0)


def funded(base_risk, use_throttle):
    months = sorted(PART["month"].unique())
    payouts = []; bal = float(ACCOUNT); peak = float(ACCOUNT); blown = False; bm = None
    maxdd = 0.0
    for m in months:
        if blown: break
        md = PART[PART["month"] == m]
        dpl = 0.0; cur = None
        for _, t in md.iterrows():
            if t["date"] != cur: cur = t["date"]; dpl = 0.0
            if dpl <= -DAILY_STOP/100*peak: continue
            dd_pct = (peak - bal) / ACCOUNT * 100
            r = base_risk * (throttle(dd_pct) if use_throttle else 1.0)
            pl = bal * (r/100) * t["pnl_r"]; bal += pl; dpl += pl
            peak = max(peak, bal); maxdd = min(maxdd, (bal - peak)/ACCOUNT*100)
            if bal <= FLOOR: blown = True; bm = str(m); break
        if blown: break
        if bal > ACCOUNT:
            payouts.append((bal - ACCOUNT) * SPLIT); bal = ACCOUNT; peak = ACCOUNT
        else:
            payouts.append(0.0)
    return np.array(payouts), blown, bm, maxdd, len(months)


print("=" * 92)
print("MAKING 1% RISK WORK · partial@1R · EURUSD · flat vs drawdown-throttled")
print("=" * 92)

print("\n── FTMO PHASE-1 pass (+10% on $10k) ──")
print(f"  {'config':28} {'pass%':>7} {'med days':>9} {'~calendar':>11}")
for label, br, thr in [("flat 1.0%", 1.0, False),
                       ("throttled 1.0%", 1.0, True),
                       ("flat 0.75%", 0.75, False),
                       ("throttled 0.75%", 0.75, True),
                       ("reference flat 0.33%", 0.33, False)]:
    pr, md = pass_stats(br, thr)
    print(f"  {label:28} {pr:>6.1f}% {md:>9} {md/5*7/7:>8.1f} wk")

print("\n── FUNDED $10k · monthly payouts + survival ──")
print(f"  {'config':28} {'avg/mo':>9} {'paid mos':>9} {'maxDD':>8} {'total':>10} {'blown?':>12}")
for label, br, thr in [("flat 1.0%", 1.0, False),
                       ("throttled 1.0%", 1.0, True),
                       ("throttled 0.75%", 0.75, True),
                       ("reference flat 0.33%", 0.33, False)]:
    po, blown, bm, mdd, nm = funded(br, thr)
    paid = (po > 0).sum()
    print(f"  {label:28} ${np.mean(po):>7,.0f} {paid:>3}/{len(po):<3} {mdd:>7.1f}% "
          f"${po.sum():>8,.0f} {('BLOWN '+bm if blown else 'survives'):>12}")
