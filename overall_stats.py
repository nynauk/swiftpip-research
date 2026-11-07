"""
Overall stats for the M3+M5+M15 three-engine portfolio run simultaneously from
the start of the backtest (2000-2026), flat $1,000 risk/trade PER ENGINE on one
$100k account (so up to 3% concurrent when all three fire). Buffered, spread-
aware. Reads the combined trade stream saved by full_2000_2026.py.
"""
import numpy as np
import pandas as pd

RISK, START = 1000.0, 100000.0
df = pd.read_csv("strategies/full_2000_2026/full_trades.csv")
df["entry_time"] = pd.to_datetime(df["entry_time"])
df["exit_time"] = pd.to_datetime(df["exit_time"])
cl = df[df["outcome"] != "open"].sort_values("entry_time").reset_index(drop=True)
r = cl["pnl_r"].astype(float)

N = len(cl)
wins = r[r > 0.01]; losses = r[r < -0.01]; be = r[(r >= -0.01) & (r <= 0.01)]
wr = len(wins) / N * 100
gw = wins.sum(); gl = -losses.sum(); pf = gw / gl
totR = r.sum(); end = START + RISK * totR

# streaks
cw = clx = bw = bl = 0
for x in r:
    if x > 0.01:
        cw += 1; clx = 0; bw = max(bw, cw)
    elif x < -0.01:
        clx += 1; cw = 0; bl = max(bl, clx)

# entry-ordered equity & DD
eq = START + (RISK * r).cumsum()
peak = eq.cummax(); dd_pct = ((eq - peak) / peak * 100).min(); dd_usd = (eq - peak).min()

# realized daily (exit-ordered) equity, DD, Sharpe
d = cl.copy(); d["day"] = d["exit_time"].dt.normalize()
daily = d.groupby("day")["pnl_r"].sum() * RISK
span = pd.date_range(cl["entry_time"].min().normalize(), cl["exit_time"].max().normalize())
dser = daily.reindex(span, fill_value=0.0)
ceq = START + dser.cumsum(); rpeak = ceq.cummax()
real_dd = ((ceq - rpeak) / rpeak * 100).min()
ret = dser / START
sharpe = ret.mean() / ret.std() * np.sqrt(252)
# monthly
mret = dser.resample("ME").sum()
mwin = (mret > 0).sum() / len(mret) * 100
# yearly
yret = dser.resample("YE").sum()
ywin = (yret > 0).sum() / len(yret) * 100

years = (cl["entry_time"].max() - cl["entry_time"].min()).days / 365.25
tdays = (cl["entry_time"].max() - cl["entry_time"].min()).days * 5 / 7

print("=" * 74)
print("OVERALL STATS · M3 + M5 + M15 running together · 2000-2026 · $1k/$100k each engine")
print("=" * 74)
print(f"  Span                : {cl['entry_time'].min().date()} → {cl['exit_time'].max().date()}  ({years:.1f} yrs)")
print("  --- activity ---")
print(f"  Total trades        : {N:,}")
print(f"  Trades per day/wk/mo: {N/tdays:.2f} / {N/(tdays/5):.1f} / {N/(years*12):.1f}")
print(f"  Avg hold            : {cl['hold_h'].mean():.1f}h")
print("  --- outcomes ---")
print(f"  Win / Loss / BE     : {len(wins):,} / {len(losses):,} / {len(be):,}")
print(f"  Win rate            : {wr:.1f}%")
print(f"  Profit factor       : {pf:.2f}")
print("  --- returns ---")
print(f"  Total R             : {totR:+.1f}R")
print(f"  Expectancy (avg R)  : {r.mean():+.3f}R/trade   (median {r.median():+.3f}R)")
print(f"  Avg win / avg loss  : +{wins.mean():.2f}R / {losses.mean():.2f}R")
print(f"  Median win / loss   : +{wins.median():.2f}R / {losses.median():.2f}R")
print(f"  Largest win / loss  : +{r.max():.1f}R / {r.min():.1f}R")
print(f"  End balance         : ${end:,.0f}   (+{end/START-1:.0%} on $100k, non-compounding)")
print(f"  Avg profit / year   : ${RISK*totR/years:,.0f}")
print("  --- risk ---")
print(f"  Max drawdown        : {dd_pct:.1f}%  (${dd_usd:,.0f})  [entry-ordered]")
print(f"  Max drawdown        : {real_dd:.1f}%  [realized, exit-ordered daily]")
print(f"  Max loss streak     : {bl}    Max win streak: {bw}")
print(f"  Daily Sharpe (ann.) : {sharpe:.2f}")
print("  --- consistency ---")
print(f"  Green days          : {(daily>0).sum()} / {(daily!=0).sum()} active ({(daily>0).sum()/(daily!=0).sum()*100:.0f}%)")
print(f"  Best / worst day    : +${daily.max():,.0f} / -${abs(daily.min()):,.0f}")
print(f"  Profitable months   : {mwin:.0f}%   Profitable years: {ywin:.0f}% ({(yret>0).sum()}/{len(yret)})")
print("=" * 74)
