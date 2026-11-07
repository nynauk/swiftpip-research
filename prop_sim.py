"""
Prop-firm (FTMO-style) optimizer for the 3RR EURUSD sweep+IFVG strategy.

FTMO Challenge (Phase 1) rules modelled:
   • Profit target : +10%
   • Max daily loss: -5%  (relative to start-of-day balance)
   • Max overall   : -10% (relative to initial balance)
   • Min trading days: 4
Phase 2 (Verification): +5% target, same DD rules.

We don't optimise the entry edge (fixed). We optimise:
   • risk % per trade   (sizing → controls how DD/target scale)
   • daily self-stop %  (stop trading for the day after this intraday loss →
                          protects the hard -5% daily limit)
   • trade management   (plain / BE@1R / partial@1R)

Method: rolling Monte-Carlo. Start a fresh challenge at many points in the
trade history, run forward until PASS or FAIL, and report the pass rate.
"""
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import main as bt
import eurusd_basic as eb

mt5.initialize()
df_h4 = bt.fetch_data("EURUSD", mt5.TIMEFRAME_H4,  bt.START_DATE, bt.END_DATE, tf_name="H4")
df_15 = bt.fetch_data("EURUSD", mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
mt5.shutdown()

def _prep(d):
    d = d[d["outcome"] != "open"].sort_values("entry_time").reset_index(drop=True)
    d["date"] = pd.to_datetime(d["entry_time"]).dt.date
    return d

# (RR, management) → prepared trade frame
VARIANTS = {}
for rr in [2.0, 3.0]:
    VARIANTS[(rr, "plain")]   = _prep(eb.run(df_h4, df_15, rr=rr))
    VARIANTS[(rr, "BE@1R")]   = _prep(eb.run(df_h4, df_15, rr=rr, be_at=1.0))
    VARIANTS[(rr, "partial")] = _prep(eb.run(df_h4, df_15, rr=rr, partial_at=1.0))

MGMTS = {k: VARIANTS[(3.0, k)] for k in ["plain", "BE@1R", "partial"]}  # legacy 3RR refs


def run_challenge(pnl, dates, start, risk, daily_stop,
                  target, daily_max=5.0, overall_max=10.0, min_days=4,
                  max_trades=400):
    """One challenge from trade index `start`. Returns (result, n_trades, days)."""
    equity = 0.0          # % from initial
    day_pl = 0.0
    cur_day = None
    tdays = set()
    for i in range(start, min(start + max_trades, len(pnl))):
        d = dates[i]
        if d != cur_day:
            cur_day = d; day_pl = 0.0
        # daily self-stop: once today's loss hits the self-stop, skip remaining
        if day_pl <= -daily_stop:
            continue
        pl = risk * pnl[i]          # this trade's account % move (loss = -risk)
        equity += pl; day_pl += pl
        tdays.add(d)
        if day_pl <= -daily_max:
            return "FAIL_daily", i - start + 1, len(tdays)
        if equity <= -overall_max:
            return "FAIL_maxdd", i - start + 1, len(tdays)
        if equity >= target and len(tdays) >= min_days:
            return "PASS", i - start + 1, len(tdays)
    return "INCOMPLETE", min(max_trades, len(pnl) - start), len(tdays)


def pass_rate(df, risk, daily_stop, target):
    pnl = df["pnl_r"].values; dates = df["date"].values
    # start a challenge at the first trade of each distinct day (realistic)
    starts = [i for i in range(len(df)) if i == 0 or dates[i] != dates[i-1]]
    # only starts with enough runway
    starts = [s for s in starts if len(df) - s >= 120]
    res = {"PASS": 0, "FAIL_daily": 0, "FAIL_maxdd": 0, "INCOMPLETE": 0}
    days_to_pass = []
    for s in starts:
        r, nt, nd = run_challenge(pnl, dates, s, risk, daily_stop, target)
        res[r] += 1
        if r == "PASS":
            days_to_pass.append(nd)
    tot = sum(res.values()) - res["INCOMPLETE"]
    pr = res["PASS"] / tot * 100 if tot else 0
    med_days = int(np.median(days_to_pass)) if days_to_pass else 0
    return pr, res, med_days, len(starts)


print("=" * 104)
print("FTMO PHASE-1 CHALLENGE  ·  +10% target · -5% daily · -10% max · min 4 days · EURUSD")
print("2RR vs 3RR · rolling Monte-Carlo pass rate")
print("=" * 104)
print(f"  {'RR':>4} {'management':10} {'risk%':>6} {'PASS%':>7} {'medDays':>8} "
      f"{'failMaxDD':>10}")

best = None
for rr in [2.0, 3.0]:
    for mgmt in ["plain", "BE@1R", "partial"]:
        df = VARIANTS[(rr, mgmt)]
        for risk in [0.33, 0.5, 0.75, 1.0]:
            pr, res, md, nstart = pass_rate(df, risk, 3.0, target=10.0)
            if best is None or pr > best[0]:
                best = (pr, mgmt, risk, 3.0, md, rr)
            print(f"  {rr:>4.1f} {mgmt:10} {risk:>6.2f} {pr:>6.1f}% {md:>8} {res['FAIL_maxdd']:>10}")
        print()

pr, mgmt, risk, dstop, md, rrb = best
print(f">>> BEST PHASE-1 CONFIG: {rrb}RR · {mgmt} · risk {risk}%  →  "
      f"{pr:.1f}% pass, median {md} days")
# ── Funded-account full-history sim: 2RR vs 3RR, partial@1R, across risk ──
def funded(df, risk, dstop=3.0):
    pnl = df["pnl_r"].values; dates = df["date"].values
    eq = pk = dpl = 0.0; cur = None; mdd = wd = 0.0; bd = bm = 0
    for i in range(len(df)):
        d = dates[i]
        if d != cur:
            if cur is not None: wd = min(wd, dpl)
            cur = d; dpl = 0.0
        if dpl <= -dstop: continue
        pl = risk * pnl[i]; eq += pl; dpl += pl
        pk = max(pk, eq); mdd = min(mdd, eq - pk)
        if dpl <= -5.0: bd += 1
        if eq - pk <= -10.0: bm += 1
    wd = min(wd, dpl)
    yrs = (pd.to_datetime(df["entry_time"]).max() - pd.to_datetime(df["entry_time"]).min()).days / 365.25
    return eq, eq / yrs, mdd, wd, bd, bm

print("\n" + "=" * 104)
print("FUNDED-ACCOUNT · partial@1R · 2RR vs 3RR · risk sweep · full history")
print("=" * 104)
print(f"  {'RR':>4} {'risk%':>6} {'net%':>9} {'/yr':>8} {'maxDD%':>8} {'worstDay%':>10} {'ddBreaches':>11}  {'FTMO-safe?':>10}")
for rr in [2.0, 3.0]:
    df = VARIANTS[(rr, "partial")]
    for rk in [0.33, 0.5, 0.75, 1.0]:
        net, py, mdd, wd, bd, bm = funded(df, rk)
        safe = "YES" if (mdd > -10 and wd > -5) else "NO (breach)"
        print(f"  {rr:>4.1f} {rk:>6.2f} {net:>+8.1f}% {py:>+7.1f}% {mdd:>7.1f}% {wd:>9.1f}% "
              f"{bm:>11}  {safe:>10}")
    print()
