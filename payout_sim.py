"""
$10k FTMO account: time-to-pass + monthly payout simulation.
Compares plain (full 3RR) / BE@1R / partial@1R management.

Funded-account payout model:
  • Start each month from $10,000 baseline (carry forward if in drawdown).
  • Month-end: if balance > $10k → withdraw (balance-10k)*SPLIT, reset to $10k.
                if balance <= $10k → NO payout, carry the balance forward.
  • If balance ever hits $9,000 (-10%) → account BLOWN.
"""
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import main as bt
import eurusd_basic as eb

ACCOUNT = 10_000
FLOOR   = 9_000          # -10% hard blow-up
SPLIT   = 0.80           # trader keeps 80%
DAILY_STOP = 3.0         # % intraday self-stop

mt5.initialize()
df_h4 = bt.fetch_data("EURUSD", mt5.TIMEFRAME_H4,  bt.START_DATE, bt.END_DATE, tf_name="H4")
df_15 = bt.fetch_data("EURUSD", mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
mt5.shutdown()

def prep(d):
    d = d[d["outcome"] != "open"].sort_values("entry_time").reset_index(drop=True)
    d["date"]  = pd.to_datetime(d["entry_time"]).dt.date
    d["month"] = pd.to_datetime(d["entry_time"]).dt.to_period("M")
    return d

MG = {
    "plain (full 3R)": prep(eb.run(df_h4, df_15, rr=3.0)),
    "BE@1R":           prep(eb.run(df_h4, df_15, rr=3.0, be_at=1.0)),
    "partial@1R":      prep(eb.run(df_h4, df_15, rr=3.0, partial_at=1.0)),
}

# ── 1. Average winning-trade R (proves partial < 3R) ──
print("=" * 84)
print("Q: does PARTIAL mean you get less than 3R?  →  average R on WINNING trades")
print("=" * 84)
for name, d in MG.items():
    w = d[d["pnl_r"] > 0]
    dist = d["pnl_r"].round(2).value_counts().sort_index()
    dd = "  ".join(f"{r:+.1f}R×{n}" for r, n in dist.items() if r > 0)
    print(f"  {name:16}: avg winner = {w['pnl_r'].mean():+.2f}R   [{dd}]")

# ── 2. Time to pass Phase-1 (+10%) on $10k ──
def median_days_to_pass(d, risk, target=10.0, daily_max=5.0, overall=10.0, mind=4):
    pnl = d["pnl_r"].values; dates = d["date"].values
    starts = [i for i in range(len(d)) if i == 0 or dates[i] != dates[i-1]]
    starts = [s for s in starts if len(d) - s >= 120]
    days = []; passes = 0; tot = 0
    for s in starts:
        eq = dpl = 0.0; cur = None; td = set(); res = None
        for i in range(s, min(s+400, len(d))):
            dt = dates[i]
            if dt != cur: cur = dt; dpl = 0.0
            if dpl <= -DAILY_STOP: continue
            pl = risk*pnl[i]; eq += pl; dpl += pl; td.add(dt)
            if dpl <= -daily_max or eq <= -overall: res = "F"; break
            if eq >= target and len(td) >= mind: res = "P"; days.append(len(td)); break
        if res: tot += 1
        if res == "P": passes += 1
    return (passes/tot*100 if tot else 0), (int(np.median(days)) if days else 0)

print("\n" + "=" * 84)
print("TIME TO PASS PHASE-1 (+10% = +$1,000 on $10k)")
print("=" * 84)
print(f"  {'management':16} {'risk':>6} {'pass%':>7} {'trad.days':>10} {'~calendar':>12}")
for name, d in MG.items():
    for risk in [0.33, 0.75]:
        pr, md = median_days_to_pass(d, risk)
        cal = md / 5 * 7   # trading days → calendar days (≈5 trad. days/week)
        print(f"  {name:16} {risk:>5.2f}% {pr:>6.1f}% {md:>10} {cal/7:>9.1f} wk")

# ── 3. Monthly payout simulation on funded $10k ──
def payout_sim(d, risk):
    months = sorted(d["month"].unique())
    payouts = []; bal = float(ACCOUNT); blown = False; blow_month = None
    for m in months:
        if blown: break
        md = d[d["month"] == m]
        dpl = 0.0; cur = None
        for _, t in md.iterrows():
            if t["date"] != cur: cur = t["date"]; dpl = 0.0
            if dpl <= -DAILY_STOP/100*ACCOUNT: continue
            pl = bal * (risk/100) * t["pnl_r"]
            bal += pl; dpl += pl
            if bal <= FLOOR:
                blown = True; blow_month = str(m); break
        if blown: break
        if bal > ACCOUNT:
            payouts.append((bal - ACCOUNT) * SPLIT)   # trader's cut
            bal = ACCOUNT                              # reset baseline
        else:
            payouts.append(0.0)                        # no payout, carry forward
    return payouts, blown, blow_month, len(months)

print("\n" + "=" * 84)
print(f"FUNDED $10k · MONTHLY PAYOUTS (reset to $10k after payout · 80% split · no pay if red)")
print("=" * 84)
for risk in [0.33, 0.5]:
    print(f"\n  --- risk {risk}% per trade ---")
    print(f"  {'management':16} {'avg/mo':>9} {'median':>8} {'paid mos':>9} {'best':>8} "
          f"{'total':>10} {'blown?':>7}")
    for name, d in MG.items():
        po, blown, bm, nm = payout_sim(d, risk)
        po = np.array(po)
        paid = (po > 0).sum()
        print(f"  {name:16} ${np.mean(po):>7,.0f} ${np.median(po):>6,.0f} "
              f"{paid:>3}/{len(po):<3} ${po.max():>6,.0f} ${po.sum():>8,.0f} "
              f"{('BLOWN '+bm if blown else 'no'):>7}")
