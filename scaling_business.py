"""
Prop-firm SCALING BUSINESS model.

You treat challenges as inventory: buy them, pass them, earn payouts, and
reinvest payouts into buying MORE accounts. Blow-ups are just a cost of goods
(re-buy). FTMO refunds the challenge fee with your first payout.

We (a) compute unit economics per $10k account, then (b) Monte-Carlo the
reinvestment business over 24 months.
"""
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import main as bt
import eurusd_basic as eb

rng = np.random.default_rng(42)

ACCOUNT = 10_000; FLOOR = 9_000; SPLIT = 0.80
FEE_10K = 100                    # ~FTMO $10k challenge fee (refunded on 1st payout)
DAILY_STOP = 4.0
TRADES_PER_MONTH = 52            # 2516 trades / ~48.5 months
BASE_RISK = 0.75                 # % (with throttle)

mt5.initialize()
df_h4 = bt.fetch_data("EURUSD", mt5.TIMEFRAME_H4,  bt.START_DATE, bt.END_DATE, tf_name="H4")
df_15 = bt.fetch_data("EURUSD", mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
mt5.shutdown()

part = eb.run(df_h4, df_15, rr=3.0, partial_at=1.0)
R = part[part["outcome"] != "open"]["pnl_r"].values     # pool of per-trade R


def throttle(dd):
    if dd < 3:  return 1.00
    if dd < 5:  return 0.50
    if dd < 7:  return 0.25
    return 0.10

# ── Bootstrap a funded month: sample trades, apply throttle, return (payout, blown) ──
def sim_month(bal, peak):
    seq = R[rng.integers(0, len(R), TRADES_PER_MONTH)]
    for r in seq:
        dd = (peak - bal) / ACCOUNT * 100
        risk = BASE_RISK * throttle(dd)
        bal += bal * (risk / 100) * r
        peak = max(peak, bal)
        if bal <= FLOOR:
            return 0.0, True, ACCOUNT, ACCOUNT   # blown → account gone
    if bal > ACCOUNT:
        payout = (bal - ACCOUNT) * SPLIT
        return payout, False, ACCOUNT, ACCOUNT   # reset baseline+peak on payout
    return 0.0, False, bal, peak                 # carry forward

# ── Bootstrap a challenge attempt: reach +target% before -10% (throttled) ──
def sim_challenge(target):
    eq = 0.0; peak = 0.0
    for _ in range(600):
        r = R[rng.integers(0, len(R))]
        dd = peak - eq
        risk = BASE_RISK * throttle(dd)
        eq += risk * r; peak = max(peak, eq)
        if eq <= -10.0: return False
        if eq >= target: return True
    return False

# ── Unit economics (Monte-Carlo) ──
N = 1500
p1 = np.mean([sim_challenge(10.0) for _ in range(N)])
p2 = np.mean([sim_challenge(5.0)  for _ in range(N)])
# expected lifetime payouts of a funded account until blow-up
lifetimes = []; life_payouts = []
for _ in range(500):
    bal = ACCOUNT; peak = ACCOUNT; tot = 0.0; months = 0
    for m in range(120):
        po, blown, bal, peak = sim_month(bal, peak)
        tot += po; months += 1
        if blown: break
    lifetimes.append(months); life_payouts.append(tot)
E_life_payout = np.mean(life_payouts); E_life_months = np.mean(lifetimes)
month_payouts = []
for _ in range(2000):
    po, _, _, _ = sim_month(ACCOUNT, ACCOUNT)
    month_payouts.append(po)
E_month = np.mean(month_payouts)

print("=" * 84)
print(f"UNIT ECONOMICS · $10k account · throttled {BASE_RISK}% · partial@1R")
print("=" * 84)
print(f"  Phase-1 pass (+10%)         : {p1*100:.1f}%")
print(f"  Phase-2 pass (+5%)          : {p2*100:.1f}%")
print(f"  P(pass both → funded)       : {p1*p2*100:.1f}%")
print(f"  Avg payout / funded month   : ${E_month:,.0f}")
print(f"  Avg funded lifetime         : {E_life_months:.0f} months (until blow-up)")
print(f"  Avg lifetime payouts/acct   : ${E_life_payout:,.0f}")
print(f"  Challenge fee (refunded 1st payout): ${FEE_10K}")
ev = -FEE_10K + p1*p2*(FEE_10K + E_life_payout)
print(f"  >>> EV per $100 challenge   : ${ev:,.0f}   ({ev/FEE_10K*100:.0f}% ROI per attempt)")

# ── Reinvestment business Monte-Carlo over 24 months ──
def run_business(start_cash, months=24, reinvest=True):
    cash = start_cash
    funded = []            # list of [bal, peak, got_first_payout]
    pipeline = []          # challenges maturing: list of months-remaining until funded
    income_curve = []; equity_curve = []
    for m in range(months):
        # 1. advance pipeline
        newf = []
        for r in pipeline:
            r -= 1
            if r <= 0: funded.append([ACCOUNT, ACCOUNT, False])
            else: newf.append(r)
        pipeline = newf
        # 2. funded accounts trade this month
        income = 0.0; survivors = []
        for acc in funded:
            po, blown, acc[0], acc[1] = sim_month(acc[0], acc[1])
            if po > 0 and not acc[2]:
                po += FEE_10K; acc[2] = True     # fee refunded on first payout
            income += po
            if not blown: survivors.append(acc)
        funded = survivors
        cash += income
        # 3. reinvest: buy as many $10k challenges as cash allows
        if reinvest:
            buy = int(cash // FEE_10K)
            if buy > 0:
                cash -= buy * FEE_10K
                for _ in range(buy):
                    if rng.random() < p1 * p2:   # passes both phases
                        pipeline.append(2)       # funded in 2 months
                    # else: fee lost (fail)
        income_curve.append(income)
        equity_curve.append(cash + len(funded) * 0)   # cash only (accounts are FTMO's)
    return income_curve, cash, len(funded), len(pipeline)

print("\n" + "=" * 84)
print("REINVESTMENT BUSINESS · start $300 (3 challenges) · buy $10k challenges · 24 months")
print("=" * 84)
sims = [run_business(300, 24) for _ in range(400)]
# median monthly income + funded count trajectory
inc = np.array([s[0] for s in sims])
med_inc = np.median(inc, axis=0)
fund_final = np.array([s[2] for s in sims])
cash_final = np.array([s[1] for s in sims])
for mo in [3, 6, 12, 18, 24]:
    print(f"  Month {mo:2d}: median income ${np.median(inc[:,mo-1]):>7,.0f}/mo   "
          f"(25th ${np.percentile(inc[:,mo-1],25):>6,.0f} · 75th ${np.percentile(inc[:,mo-1],75):>7,.0f})")
print(f"\n  After 24 months: median {int(np.median(fund_final))} funded accts, "
      f"${np.median(cash_final):,.0f} cash, ~${np.median(inc[:,-1]):,.0f}/mo income")
print(f"  (top 25%: {int(np.percentile(fund_final,75))} accts, "
      f"${np.percentile(inc[:,-1],75):,.0f}/mo)")
