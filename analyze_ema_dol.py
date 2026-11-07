"""Deep stats on the EMA+DOL strategy from saved trades.csv."""
import os
import numpy as np
import pandas as pd

CSV = os.path.join("strategies", "ema_dol", "trades.csv")
df = pd.read_csv(CSV, parse_dates=["entry_time", "exit_time", "cross_time", "fvg_time"])
df = df.sort_values("entry_time").reset_index(drop=True)
df["pnl_r"] = pd.to_numeric(df["pnl_r"])
closed = df[df["outcome"] != "open"].copy()
opens = df[df["outcome"] == "open"]

# Basic
n_total = len(df)
n_closed = len(closed)
n_open = len(opens)
n_win = len(closed[closed["outcome"] == "win"])
n_loss = len(closed[closed["outcome"] == "loss"])
wr = n_win / n_closed * 100 if n_closed else 0

total_r = closed["pnl_r"].sum()
gross_win_r = closed.loc[closed["outcome"] == "win", "pnl_r"].sum()
gross_loss_r = -closed.loc[closed["outcome"] == "loss", "pnl_r"].sum()  # positive
avg_win = closed.loc[closed["outcome"] == "win", "pnl_r"].mean() if n_win else 0
avg_loss = closed.loc[closed["outcome"] == "loss", "pnl_r"].mean() if n_loss else 0
biggest_win = closed["pnl_r"].max()
biggest_loss = closed["pnl_r"].min()
expectancy = closed["pnl_r"].mean()
profit_factor = gross_win_r / gross_loss_r if gross_loss_r else float("inf")
avg_rr = closed["achieved_rr"].mean()
median_rr = closed["achieved_rr"].median()

# Equity, DD, streaks
def equity_dd_streaks(df_, initial=10_000, risk_pct=1.0):
    bal = initial
    eq = [bal]
    cur_w = cur_l = 0
    best_w = best_l = 0
    win_dates = loss_dates = (None, None)
    run_start = None
    for _, row in df_.iterrows():
        bal += bal * (risk_pct / 100) * float(row["pnl_r"])
        eq.append(bal)
        if row["outcome"] == "win":
            if cur_w == 0:
                run_start = row["entry_time"]
            cur_w += 1
            if cur_w > best_w:
                best_w = cur_w
                win_dates = (run_start, row["entry_time"])
            cur_l = 0
        elif row["outcome"] == "loss":
            if cur_l == 0:
                run_start = row["entry_time"]
            cur_l += 1
            if cur_l > best_l:
                best_l = cur_l
                loss_dates = (run_start, row["entry_time"])
            cur_w = 0
    eq_arr = np.array(eq)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak * 100
    return bal, dd.min(), best_w, best_l, win_dates, loss_dates, eq_arr

bal, max_dd, best_w, best_l, win_span, loss_span, eq = equity_dd_streaks(closed)

# Time in trade
closed["duration"] = (pd.to_datetime(closed["exit_time"]) - pd.to_datetime(closed["entry_time"])).dt.total_seconds() / 3600
avg_dur = closed["duration"].mean()
median_dur = closed["duration"].median()
max_dur = closed["duration"].max()
avg_win_dur = closed.loc[closed["outcome"] == "win", "duration"].mean()
avg_loss_dur = closed.loc[closed["outcome"] == "loss", "duration"].mean()

# Recovery factor
recovery_factor = total_r / abs(max_dd / 100 * 10000 / 100) if max_dd else float("inf")  # R per % DD

# Per-month
closed["month"] = pd.to_datetime(closed["entry_time"]).dt.to_period("M")
monthly = closed.groupby("month").agg(
    n=("pnl_r", "size"),
    wins=("outcome", lambda s: (s == "win").sum()),
    r=("pnl_r", "sum"),
)
monthly["wr"] = (monthly["wins"] / monthly["n"] * 100).round(1)
n_pos_months = (monthly["r"] > 0).sum()
n_neg_months = (monthly["r"] < 0).sum()
worst_month_r = monthly["r"].min()
best_month_r = monthly["r"].max()

# Per-DOW
closed["dow"] = pd.to_datetime(closed["entry_time"]).dt.day_name()
dow_stats = closed.groupby("dow").agg(
    n=("pnl_r", "size"),
    wins=("outcome", lambda s: (s == "win").sum()),
    r=("pnl_r", "sum"),
)
dow_stats["wr"] = (dow_stats["wins"] / dow_stats["n"] * 100).round(1)
dow_stats = dow_stats.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]).fillna(0)

# Per-hour (entry hour, broker time)
closed["hour"] = pd.to_datetime(closed["entry_time"]).dt.hour
hour_stats = closed.groupby("hour").agg(
    n=("pnl_r", "size"),
    wins=("outcome", lambda s: (s == "win").sum()),
    r=("pnl_r", "sum"),
)
hour_stats["wr"] = (hour_stats["wins"] / hour_stats["n"] * 100).round(1)

# Long vs Short
ln = closed[closed["direction"] == "long"]
sh = closed[closed["direction"] == "short"]

# RR distribution
rr_buckets = pd.cut(closed["achieved_rr"], bins=[2, 2.5, 3, 4, 5, 7, 10, 100],
                     labels=["2-2.5", "2.5-3", "3-4", "4-5", "5-7", "7-10", "10+"])
rr_dist = rr_buckets.value_counts().sort_index()

# Print
print("=" * 80)
print(f"EMA9/21 + ICT DOL · M15 · 7 majors · min 2RR · news-blocked")
print(f"Backtest period: {closed['entry_time'].min()} → {closed['entry_time'].max()}")
print("=" * 80)

print("\n── HEADLINE STATS ──")
print(f"  Total signals               : {n_total}")
print(f"  Closed trades               : {n_closed}   (Open: {n_open})")
print(f"  Wins / Losses               : {n_win} / {n_loss}")
print(f"  Win rate                    : {wr:.2f}%")
print(f"  Expectancy per trade        : {expectancy:+.3f}R")
print(f"  Total R                     : {total_r:+.1f}R")
print(f"  Profit factor (gross W/L)   : {profit_factor:.2f}")

print("\n── RR & TRADE QUALITY ──")
print(f"  Avg win   (R)               : +{avg_win:.2f}R")
print(f"  Avg loss  (R)               : {avg_loss:.2f}R")
print(f"  Avg achieved RR             : {avg_rr:.2f}")
print(f"  Median achieved RR          : {median_rr:.2f}")
print(f"  Biggest win                 : +{biggest_win:.2f}R")
print(f"  Biggest loss                : {biggest_loss:.2f}R")
print(f"\n  RR distribution (winning trades):")
for b, c in rr_dist.items():
    pct = c / n_win * 100 if n_win else 0
    print(f"    {b:>6}R : {c:3d}  ({pct:4.1f}%)")

print("\n── DRAWDOWN & STREAKS ──")
print(f"  Max drawdown (1% risk)      : {max_dd:.2f}%")
print(f"  Longest winning streak      : {best_w}")
if win_span[0] is not None:
    print(f"    span                      : {win_span[0]} → {win_span[1]}")
print(f"  Longest losing streak       : {best_l}")
if loss_span[0] is not None:
    print(f"    span                      : {loss_span[0]} → {loss_span[1]}")
print(f"  Recovery factor (R per %DD) : {abs(total_r / max_dd):.2f}")

print("\n── COMPOUNDED EQUITY (1% risk per trade) ──")
print(f"  Starting balance            : $10,000")
print(f"  Ending balance              : ${bal:,.0f}")
print(f"  Net return                  : {(bal/10000 - 1)*100:+,.1f}%")
years = (closed["entry_time"].max() - closed["entry_time"].min()).days / 365.25
cagr = (bal / 10000) ** (1 / years) - 1
print(f"  Backtest length             : {years:.2f} years")
print(f"  CAGR (compounded)           : {cagr*100:+.1f}%")

print("\n── TIME IN TRADE ──")
print(f"  Average duration            : {avg_dur:.1f} hours  ({avg_dur/24:.2f} days)")
print(f"  Median duration             : {median_dur:.1f} hours")
print(f"  Longest trade               : {max_dur:.1f} hours  ({max_dur/24:.1f} days)")
print(f"  Avg WIN duration            : {avg_win_dur:.1f} hours")
print(f"  Avg LOSS duration           : {avg_loss_dur:.1f} hours")

print("\n── LONG vs SHORT ──")
for side, sub in [("LONG", ln), ("SHORT", sh)]:
    if not len(sub):
        continue
    swr = (sub["outcome"] == "win").sum() / len(sub) * 100
    sR = sub["pnl_r"].sum()
    s_exp = sub["pnl_r"].mean()
    s_avg_rr = sub["achieved_rr"].mean()
    print(f"  {side:6}: N={len(sub):3d}  WR={swr:5.1f}%  R={sR:+7.1f}  "
          f"Expectancy={s_exp:+.3f}R  AvgRR={s_avg_rr:.2f}")

print("\n── MONTHLY CONSISTENCY ──")
print(f"  Total months                : {len(monthly)}")
print(f"  Profitable months           : {n_pos_months}  ({n_pos_months/len(monthly)*100:.1f}%)")
print(f"  Losing months               : {n_neg_months}  ({n_neg_months/len(monthly)*100:.1f}%)")
print(f"  Best  month                 : +{best_month_r:.1f}R")
print(f"  Worst month                 : {worst_month_r:+.1f}R")
print(f"  Avg trades / month          : {monthly['n'].mean():.1f}")
print(f"  Avg R / month               : {monthly['r'].mean():+.2f}R")

print("\n── DAY-OF-WEEK ──")
for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
    if d in dow_stats.index:
        row = dow_stats.loc[d]
        print(f"  {d:10}: N={int(row['n']):3d}  WR={row['wr']:5.1f}%  R={row['r']:+6.1f}")

print("\n── ENTRY HOUR (broker time = your local +2h) ──")
for hr in sorted(hour_stats.index):
    row = hour_stats.loc[hr]
    bar = "█" * int(abs(row["r"]))
    sign = "+" if row["r"] >= 0 else "-"
    print(f"  {hr:02d}:00 : N={int(row['n']):3d}  WR={row['wr']:5.1f}%  R={row['r']:+6.1f}  {sign}{bar}")
