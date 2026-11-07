"""
Worst-period analysis for the M3+M5+M15 combined portfolio (2000-2026):
worst DAY / WEEK / MONTH net P&L (realized, exit-time), expressed in R and in %
of a $100k FTMO account at 0.5% / 1% / 2% per-trade risk — i.e. would the worst
cluster ever have busted the account (-5% daily / -10% overall)?
Also: worst rolling 24h window (FTMO's daily rule is rolling, not calendar) and
the deepest peak-to-trough equity drawdown in R.
"""
import numpy as np
import pandas as pd

df = pd.read_csv("strategies/full_2000_2026/full_trades.csv")
df["exit_time"] = pd.to_datetime(df["exit_time"])
cl = df[df.outcome != "open"].sort_values("exit_time").reset_index(drop=True)

# realized P&L series in R, at exit time
t = cl["exit_time"]; r = cl["pnl_r"].astype(float)

day = r.groupby(t.dt.normalize()).sum()
week = r.groupby(t.dt.to_period("W").dt.start_time).sum()
month = r.groupby(t.dt.to_period("M").dt.start_time).sum()

# worst rolling 24h (FTMO daily is a rolling window from the first loss)
s = pd.Series(r.values, index=t)
roll24 = s.rolling("24h").sum()

# deepest peak-to-trough in R (cumulative, exit-ordered)
cum = r.cumsum(); dd_r = (cum - cum.cummax()).min()

def line(label, val_r, when):
    out = f"  {label:22} {val_r:>+7.2f}R  on {when}"
    for pct in (0.5, 1.0, 2.0):
        out += f"   {pct}%: {val_r*pct:+.1f}%"
    return out

print("=" * 100)
print("WORST-PERIOD ANALYSIS · M3+M5+M15 combined · 2000-2026 · realized at exit")
print("  (right columns = % hit to a $100k account at each per-trade risk)")
print("=" * 100)
print(line("Worst DAY", day.min(), day.idxmin().date()))
print(line("Worst rolling 24h", roll24.min(), roll24.idxmin().date()))
print(line("Worst WEEK", week.min(), week.idxmin().date()))
print(line("Worst MONTH", month.min(), str(month.idxmin())[:7]))
print(line("Deepest equity DD", dd_r, "(peak-to-trough)"))
print("-" * 100)
print("  FTMO limits: -5% in a day · -10% overall.  Bust thresholds in R:")
for pct in (0.5, 1.0, 2.0):
    print(f"    at {pct}%/trade: daily bust = {-5/pct:.0f}R in 24h · overall bust = {-10/pct:.0f}R cumulative")

# margin to bust at 1%
print("-" * 100)
print(f"  At the LIVE 1%: worst 24h {roll24.min():+.2f}R vs -5R daily limit "
      f"(margin {abs(-5 - roll24.min()):.1f}R) · deepest DD {dd_r:+.2f}R vs -10R overall "
      f"(margin {abs(-10 - dd_r):.1f}R)")

# distribution of bad days: how many days worse than thresholds
for thr in (-2, -3, -4, -5):
    n = (day <= thr).sum()
    print(f"  Days ≤ {thr}R: {n:>3}  ({n/len(day)*100:.2f}% of {len(day)} active days)")

# worst 5 days / weeks / months tables
print("\n  worst 5 DAYS:            worst 5 WEEKS:            worst 5 MONTHS:")
wd = day.nsmallest(5); ww = week.nsmallest(5); wm = month.nsmallest(5)
for i in range(5):
    print(f"    {wd.index[i].date()} {wd.iloc[i]:+6.2f}R     "
          f"{ww.index[i].date()} {ww.iloc[i]:+6.2f}R      "
          f"{str(wm.index[i])[:7]} {wm.iloc[i]:+6.2f}R")
