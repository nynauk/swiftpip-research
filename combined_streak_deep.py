"""
Combined-streak deep-dive for Option A (top-3 traded simultaneously).
Trades are interleaved by entry_time across all 3 symbols.

Questions answered:
  1. Longest consecutive-loss streak in the combined feed (already reported: 28)
  2. Rolling window: worst N-trade window (loss count + R lost)
  3. Calendar window: worst 1d / 3d / 7d / 14d / 30d R drawdown
  4. Peak concurrent losses (were all 3 open positions losers at once?)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "combined_top3_3R_trades.csv"
df = pd.read_csv(CSV, parse_dates=["sweep_time", "entry_time", "exit_time"])
df = df[df["outcome"] != "open"].sort_values("entry_time").reset_index(drop=True)
N = len(df)
print(f"COMBINED FEED — top 3 symbols interleaved chronologically")
print(f"  {N} trades over {(df['entry_time'].max() - df['entry_time'].min()).days} days\n")

# ── 1. Combined chronological streak (re-confirm) ──
outcomes = df["outcome"].values
best = cur = 0
best_start = best_end = cur_start = None
for i, o in enumerate(outcomes):
    if o == "loss":
        if cur == 0: cur_start = i
        cur += 1
        if cur > best:
            best = cur; best_start = cur_start; best_end = i
    else:
        cur = 0
print("── 1. LONGEST CONSECUTIVE-LOSS STREAK (combined feed) ──")
print(f"  {best} losses in a row")
print(f"    Trade # {best_start} → {best_end}")
print(f"    Dates:  {df.iloc[best_start]['entry_time']} → {df.iloc[best_end]['entry_time']}")
symcount = df.iloc[best_start:best_end + 1]["symbol"].value_counts()
print(f"    Symbols involved: {dict(symcount)}")
print(f"    R lost:  {df.iloc[best_start:best_end + 1]['pnl_r'].sum():.0f}R")
print()

# ── 2. Rolling window analysis ──
print("── 2. WORST ROLLING TRADE WINDOWS ──")
r = df["pnl_r"].values
print(f"  {'window':>8} {'worst R':>10} {'losses in it':>14} {'wins in it':>12} {'when':>25}")
for W in [10, 20, 30, 50, 100, 200]:
    if W > N: continue
    # Rolling sum of R
    csum = np.cumsum(np.concatenate([[0], r]))
    rolling = csum[W:] - csum[:-W]
    worst_idx = int(np.argmin(rolling))
    worst_R = rolling[worst_idx]
    slice_ = df.iloc[worst_idx:worst_idx + W]
    losses = (slice_["outcome"] == "loss").sum()
    wins = (slice_["outcome"] == "win").sum()
    when = f"{slice_['entry_time'].iloc[0].strftime('%Y-%m-%d')} → {slice_['entry_time'].iloc[-1].strftime('%Y-%m-%d')}"
    print(f"  {W:>8} {worst_R:>+9.0f}R {losses:>14} {wins:>12}    {when}")
print()

# ── 3. Calendar-time worst-window R drawdown ──
print("── 3. WORST CALENDAR-WINDOW R DRAWDOWNS ──")
print(f"  {'window':>8} {'worst R':>10} {'trades':>8} {'when':>25}")
df_r = df.set_index("entry_time")["pnl_r"].sort_index()
for label, days in [("1 day", 1), ("3 days", 3), ("7 days", 7),
                     ("14 days", 14), ("30 days", 30)]:
    roll = df_r.rolling(f"{days}D").sum()
    idx = roll.idxmin(); val = roll.min()
    end_time = idx
    start_time = end_time - pd.Timedelta(days=days)
    n_trades = int(df_r.loc[start_time:end_time].count())
    when = f"{start_time.strftime('%Y-%m-%d')} → {end_time.strftime('%Y-%m-%d')}"
    print(f"  {label:>8} {val:>+9.0f}R {n_trades:>8}    {when}")
print()

# ── 4. Peak-concurrent losses (were open positions all red at same time?) ──
print("── 4. CONCURRENT OPEN LOSSES ──")
events = []
for _, row in df.iterrows():
    events.append((row["entry_time"], "open", row.name, row["outcome"]))
    events.append((row["exit_time"], "close", row.name, row["outcome"]))
events.sort(key=lambda x: x[0])

open_positions = {}  # id → outcome
peak_all_losing = 0
peak_time = None
for t, kind, tid, oc in events:
    if kind == "open":
        open_positions[tid] = oc
    else:
        open_positions.pop(tid, None)
    if open_positions:
        all_losing = all(v == "loss" for v in open_positions.values())
        if all_losing and len(open_positions) > peak_all_losing:
            peak_all_losing = len(open_positions)
            peak_time = t
print(f"  Peak simultaneous open positions that ALL lost: {peak_all_losing}")
if peak_time is not None:
    print(f"    At time: {peak_time}")
print()

# ── PLOT the worst 28-loss streak in detail ──
BG, FG = "#0d1117", "#e6edf3"
GRN, RED, YLW = "#3fb950", "#f85149", "#d29922"

# Highlight the 28-loss streak on the cumulative R chart
cum = df["pnl_r"].cumsum().values
fig, ax = plt.subplots(figsize=(15, 6), facecolor=BG)
ax.set_facecolor(BG)
ax.plot(cum, color=GRN, linewidth=1.0)
ax.axhline(0, color="#444c56", linewidth=0.8, linestyle="--")

# Shade the two 28-loss streaks
markers = []
tmp_best = tmp_cur = 0; tmp_cs = None; ranges = []
for i, o in enumerate(outcomes):
    if o == "loss":
        if tmp_cur == 0: tmp_cs = i
        tmp_cur += 1
    else:
        if tmp_cur >= 15:
            ranges.append((tmp_cs, i - 1, tmp_cur))
        tmp_cur = 0
if tmp_cur >= 15:
    ranges.append((tmp_cs, len(outcomes) - 1, tmp_cur))

for s, e, length in ranges:
    ax.axvspan(s, e, color=RED, alpha=0.35)
    ax.annotate(f"{length} losses",
                xy=((s + e) / 2, cum[s]),
                xytext=((s + e) / 2, cum[s] + 100),
                color=YLW, ha="center", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=YLW, alpha=0.5))

ax.set_title(f"Combined feed cumulative R  ·  worst streaks (≥15 consecutive losses) highlighted",
             color=FG, fontsize=12, fontweight="bold", pad=10)
ax.set_ylabel("Cumulative R", color=FG)
ax.set_xlabel("Trade # (chronological across all 3 symbols)", color=FG)
ax.tick_params(colors=FG); ax.grid(True, alpha=0.10, color=FG)
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
plt.tight_layout()
plt.savefig("strategies/top3_optionA/worst_streaks_highlighted.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(" → chart saved: strategies/top3_optionA/worst_streaks_highlighted.png")
