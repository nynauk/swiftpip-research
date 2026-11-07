"""
Full stats + equity curves for Powell's PO3 + Fib OTE strategy.
Analyses the ALL-21 result set (3,336 trades) — the strongest configuration.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "powell_po3_fib_all-21_trades.csv"
OUT_DIR = "strategies/powell_po3_fib"
os.makedirs(OUT_DIR, exist_ok=True)

RR = 5.0

# Realistic cost model per symbol (matches powell_po3_fib.py)
COSTS = {
    "EURUSD":     (0.00015, 0.00003),
    "GBPUSD":     (0.00020, 0.00003),
    "USDJPY":     (0.020,   0.003),
    "USDCHF":     (0.00020, 0.00003),
    "EURJPY":     (0.025,   0.005),
    "GBPJPY":     (0.021,   0.005),
    "AUDJPY":     (0.025,   0.005),
    "CADJPY":     (0.025,   0.005),
    "NZDJPY":     (0.030,   0.005),
    "EURAUD":     (0.00030, 0.00005),
    "GBPAUD":     (0.00040, 0.00005),
    "XAUUSD":     (0.30,    0.05),
    "US500.cash": (0.55,    0.05),
    "US100.cash": (2.00,    0.20),
    "GER40.cash": (1.50,    0.20),
    "UK100.cash": (1.00,    0.15),
    "FRA40.cash": (1.50,    0.20),
    "JP225.cash": (10.0,    2.0),
    "HK50.cash":  (10.0,    2.0),
    "USOIL.cash": (0.05,    0.01),
    "UKOIL.cash": (0.063,   0.01),
}


df = pd.read_csv(CSV, parse_dates=["signal_time", "entry_time", "exit_time"])
df = df[df["outcome"] != "open"].sort_values("entry_time").reset_index(drop=True)
N = len(df)
print(f"Loaded {N} trades from {df['entry_time'].min()} → {df['entry_time'].max()}\n")


# ── Compute per-trade R (naive + realistic) ──
def compute_r_after(row):
    sym = row["symbol"]
    if sym not in COSTS:
        return row["pnl_r"]
    spread, slip = COSTS[sym]
    rp = float(row["risk_price"])
    if rp <= 0:
        return 0.0
    if row["outcome"] == "win":
        return (RR * rp - spread) / rp
    else:
        return -(rp + spread + slip) / rp


df["pnl_r_real"] = df.apply(compute_r_after, axis=1)
df["is_win"]     = df["outcome"] == "win"
df["week"]       = df["entry_time"].dt.to_period("W")
df["month"]      = df["entry_time"].dt.to_period("M")
df["year"]       = df["entry_time"].dt.year

wins   = int(df["is_win"].sum())
losses = N - wins
WR = wins / N * 100
PF_n = (wins * RR) / losses if losses else float("inf")
tot_naive = df["pnl_r"].sum()
tot_real  = df["pnl_r_real"].sum()

# realistic PF
gw_r = df.loc[df["pnl_r_real"] > 0, "pnl_r_real"].sum()
gl_r = -df.loc[df["pnl_r_real"] < 0, "pnl_r_real"].sum()
PF_r = gw_r / gl_r if gl_r else float("inf")

print("=" * 90)
print("HEADLINE STATS")
print("=" * 90)
print(f"  Trades           : {N}")
print(f"  Wins / Losses    : {wins} / {losses}")
print(f"  Win rate         : {WR:.2f}%")
print(f"  Profit factor    : {PF_n:.2f} naive  ·  {PF_r:.2f} realistic")
print(f"  Total R (naive)  : {tot_naive:+.0f}R")
print(f"  Total R (realistic): {tot_real:+.0f}R")
print(f"  Avg R / trade    : {tot_naive/N:+.3f}R naive  ·  {tot_real/N:+.3f}R realistic")
print(f"  Duration         : {(df['entry_time'].max() - df['entry_time'].min()).days} days"
      f"  ({(df['entry_time'].max() - df['entry_time'].min()).days/7:.0f} weeks)")


# ── Trades per week ──
wk_counts = df.groupby("week").size()
wk_r_naive = df.groupby("week")["pnl_r"].sum()
wk_r_real  = df.groupby("week")["pnl_r_real"].sum()

print("\n" + "=" * 90)
print("TRADES PER WEEK DISTRIBUTION")
print("=" * 90)
print(f"  Total weeks with trades   : {len(wk_counts)}")
print(f"  Mean trades / week        : {wk_counts.mean():.1f}")
print(f"  Median trades / week      : {wk_counts.median():.0f}")
print(f"  Std                       : {wk_counts.std():.1f}")
print(f"  Min / Max                 : {wk_counts.min()} / {wk_counts.max()}")
print(f"  Weeks with 0 trades       : (spans of no signal — the 4.5y contains "
      f"{int((df['entry_time'].max() - df['entry_time'].min()).days/7) - len(wk_counts)})")
print(f"\n  Distribution buckets:")
bins = [(1,2), (3,5), (6,10), (11,15), (16,25), (26,40), (41,100)]
for lo, hi in bins:
    n = ((wk_counts >= lo) & (wk_counts <= hi)).sum()
    print(f"    {lo:>3}-{hi:<3} trades/wk : {n:>3} weeks  ({n/len(wk_counts)*100:>5.1f}%)")

print(f"\n  Weekly R (naive)    · mean {wk_r_naive.mean():+.2f}R · std {wk_r_naive.std():.2f} · "
      f"min {wk_r_naive.min():+.1f} · max {wk_r_naive.max():+.1f}")
print(f"  Weekly R (realistic)· mean {wk_r_real.mean():+.2f}R · std {wk_r_real.std():.2f} · "
      f"min {wk_r_real.min():+.1f} · max {wk_r_real.max():+.1f}")
print(f"  Positive weeks     : {(wk_r_real > 0).sum()} / {len(wk_r_real)}  "
      f"({(wk_r_real > 0).mean()*100:.1f}%)")


# ── Losing streaks (chronological, combined feed) ──
def compute_streaks(outcomes, dates):
    runs = []
    cur_type = None; cur_len = 0; cur_start = None
    for i, o in enumerate(outcomes):
        t = "win" if o == "win" else "loss"
        if t == cur_type:
            cur_len += 1
        else:
            if cur_type is not None:
                runs.append({"type": cur_type, "length": cur_len,
                             "start": cur_start, "end": dates[i - 1]})
            cur_type = t; cur_len = 1; cur_start = dates[i]
    runs.append({"type": cur_type, "length": cur_len,
                 "start": cur_start, "end": dates[-1]})
    return runs


runs = compute_streaks(df["outcome"].values, df["entry_time"].values)
loss_runs = [r for r in runs if r["type"] == "loss"]
win_runs  = [r for r in runs if r["type"] == "win"]
loss_lens = [r["length"] for r in loss_runs]
win_lens  = [r["length"] for r in win_runs]

print("\n" + "=" * 90)
print("LOSING STREAKS (chronological combined feed)")
print("=" * 90)
print(f"  Total losing runs         : {len(loss_runs)}")
print(f"  Mean length               : {np.mean(loss_lens):.2f}")
print(f"  Median                    : {int(np.median(loss_lens))}")
print(f"  Longest                   : {max(loss_lens)}")
print(f"\n  Distribution:")
buckets = {"1":0, "2-3":0, "4-6":0, "7-10":0, "11-15":0, "16+":0}
for L in loss_lens:
    if L == 1: buckets["1"] += 1
    elif L <= 3: buckets["2-3"] += 1
    elif L <= 6: buckets["4-6"] += 1
    elif L <= 10: buckets["7-10"] += 1
    elif L <= 15: buckets["11-15"] += 1
    else: buckets["16+"] += 1
for k, v in buckets.items():
    print(f"    {k:>6}: {v:>4}  ({v/len(loss_runs)*100:>5.1f}%)")

print("\n  Top 5 longest losing streaks:")
for r in sorted(loss_runs, key=lambda x: x["length"], reverse=True)[:5]:
    s = pd.Timestamp(r["start"]).strftime("%Y-%m-%d %H:%M")
    e = pd.Timestamp(r["end"]).strftime("%Y-%m-%d %H:%M")
    days = (pd.Timestamp(r["end"]) - pd.Timestamp(r["start"])).days
    print(f"    {r['length']:>3} losses  ·  {s}  →  {e}  ({days} days)")

print(f"\n  Longest WINNING streak    : {max(win_lens)}")
print(f"  Mean win streak           : {np.mean(win_lens):.2f}")


# ── Per-year breakdown ──
print("\n" + "=" * 90)
print("PER-YEAR BREAKDOWN")
print("=" * 90)
print(f"  {'year':>6}  {'N':>5}  {'WR':>6}  {'PF_r':>6}  {'R_naive':>8}  {'R_real':>8}")
for yr, grp in df.groupby("year"):
    n = len(grp); w = grp["is_win"].sum(); wr = w/n*100
    gw = grp.loc[grp["pnl_r_real"] > 0, "pnl_r_real"].sum()
    gl = -grp.loc[grp["pnl_r_real"] < 0, "pnl_r_real"].sum()
    pf = gw/gl if gl else float("inf")
    print(f"  {yr:>6}  {n:>5}  {wr:>5.1f}%  {pf:>5.2f}  "
          f"{grp['pnl_r'].sum():>+7.0f}R  {grp['pnl_r_real'].sum():>+7.0f}R")


# ── Equity curves ──
df_sorted = df.sort_values("entry_time").reset_index(drop=True)
cum_naive = df_sorted["pnl_r"].cumsum().values
cum_real  = df_sorted["pnl_r_real"].cumsum().values


def compound(pnl, risk_pct):
    bal = 10000.0; eq = [bal]
    for p in pnl:
        bal += bal * risk_pct/100 * p; eq.append(bal)
    return np.array(eq)


eq_050 = compound(df_sorted["pnl_r_real"].values, 0.5)
eq_100 = compound(df_sorted["pnl_r_real"].values, 1.0)
eq_200 = compound(df_sorted["pnl_r_real"].values, 2.0)

peak_100 = np.maximum.accumulate(eq_100)
dd_100 = (eq_100 - peak_100) / peak_100 * 100

# ── Plot 1: Equity curve + wins/losses scatter ──
BG, FG = "#0d1117", "#e6edf3"
GRN, RED, YLW, BLU = "#3fb950", "#f85149", "#d29922", "#58a6ff"

fig, axes = plt.subplots(3, 1, figsize=(16, 15), facecolor=BG,
                          gridspec_kw={"height_ratios": [3, 2, 2]})

# 1. Cumulative R with win/loss scatter overlay
ax = axes[0]; ax.set_facecolor(BG)
ax.plot(cum_naive, color=BLU, linewidth=1.2, alpha=0.65, label=f"Naive R  →  +{cum_naive[-1]:.0f}R")
ax.plot(cum_real,  color=GRN, linewidth=1.4, label=f"Realistic R  →  +{cum_real[-1]:.0f}R")

# Scatter markers under the curve
win_idx  = df_sorted.index[df_sorted["is_win"]].tolist()
loss_idx = df_sorted.index[~df_sorted["is_win"]].tolist()
y_win_marker  = cum_real.max() * 0.03
y_loss_marker = -cum_real.max() * 0.03
ax.scatter(win_idx,  [y_win_marker]*len(win_idx),  s=3, c=GRN, alpha=0.35, marker="|")
ax.scatter(loss_idx, [y_loss_marker]*len(loss_idx), s=3, c=RED, alpha=0.25, marker="|")

ax.axhline(0, color="#444c56", linewidth=0.8, linestyle="--")
ax.set_title(f"Powell PO3 + Fib OTE  ·  ALL-21 symbols  ·  {N} trades  ·  "
             f"WR {WR:.1f}%  ·  PF {PF_r:.2f} realistic",
             color=FG, fontsize=13, fontweight="bold", pad=10)
ax.set_ylabel("Cumulative R", color=FG)
ax.set_xlabel("Trade #", color=FG)
ax.legend(facecolor="#161b22", edgecolor="#444c56", labelcolor=FG, loc="upper left")
ax.tick_params(colors=FG); ax.grid(True, alpha=0.10, color=FG)
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# 2. Compounded equity
ax = axes[1]; ax.set_facecolor(BG)
ax.plot(eq_050, color=GRN, linewidth=1.4, label=f"0.5% risk  →  ${eq_050[-1]:,.0f}")
ax.plot(eq_100, color=YLW, linewidth=1.4, label=f"1.0% risk  →  ${eq_100[-1]:,.0f}")
ax.plot(eq_200, color=BLU, linewidth=1.4, label=f"2.0% risk  →  ${eq_200[-1]:,.0f}")
ax.axhline(10000, color="#444c56", linewidth=0.8, linestyle="--")
ax.set_yscale("log")
ax.set_title(f"Compounded equity from $10,000  ·  realistic-R  ·  log scale",
             color=FG, fontsize=11, fontweight="bold", pad=8)
ax.set_ylabel("Balance ($)", color=FG); ax.set_xlabel("Trade #", color=FG)
ax.legend(facecolor="#161b22", edgecolor="#444c56", labelcolor=FG, loc="lower right")
ax.tick_params(colors=FG); ax.grid(True, alpha=0.10, color=FG, which="both")
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# 3. Drawdown at 1%
ax = axes[2]; ax.set_facecolor(BG)
ax.fill_between(range(len(dd_100)), 0, dd_100, color=RED, alpha=0.35)
ax.plot(dd_100, color=RED, linewidth=0.8)
ax.axhline(-10, color=YLW, linewidth=1.0, linestyle="--", label="-10% FTMO max DD")
ax.set_title(f"Drawdown @ 1% risk  ·  max DD {dd_100.min():.1f}%",
             color=FG, fontsize=11, fontweight="bold", pad=8)
ax.set_ylabel("Drawdown (%)", color=FG); ax.set_xlabel("Trade #", color=FG)
ax.legend(facecolor="#161b22", edgecolor="#444c56", labelcolor=FG, loc="lower right")
ax.tick_params(colors=FG); ax.grid(True, alpha=0.10, color=FG)
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

plt.tight_layout()
p1 = os.path.join(OUT_DIR, "equity_and_drawdown.png")
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig)
print(f"\n → chart saved: {p1}")


# ── Plot 2: Trades-per-week histogram + weekly R distribution ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)

ax = axes[0]; ax.set_facecolor(BG)
ax.hist(wk_counts.values, bins=range(1, wk_counts.max() + 2),
        color=BLU, alpha=0.85, edgecolor="#161b22")
ax.axvline(wk_counts.median(), color=YLW, linestyle="--", linewidth=1.3,
           label=f"median {int(wk_counts.median())}")
ax.axvline(wk_counts.mean(), color=GRN, linestyle="--", linewidth=1.3,
           label=f"mean {wk_counts.mean():.1f}")
ax.set_title(f"Trades per week distribution  ·  {len(wk_counts)} active weeks",
             color=FG, fontsize=12, fontweight="bold", pad=8)
ax.set_xlabel("Trades in the week", color=FG); ax.set_ylabel("Number of weeks", color=FG)
ax.legend(facecolor="#161b22", edgecolor="#444c56", labelcolor=FG)
ax.tick_params(colors=FG); ax.grid(True, alpha=0.10, color=FG)
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

ax = axes[1]; ax.set_facecolor(BG)
colors = [GRN if x > 0 else RED for x in wk_r_real.values]
ax.bar(range(len(wk_r_real)), wk_r_real.values, color=colors, alpha=0.75, width=1.0)
ax.axhline(0, color="#444c56", linewidth=0.8, linestyle="--")
pos = (wk_r_real > 0).sum(); neg = (wk_r_real < 0).sum()
ax.set_title(f"Weekly realistic R  ·  {pos} positive, {neg} negative "
             f"({pos/len(wk_r_real)*100:.0f}% profitable weeks)",
             color=FG, fontsize=12, fontweight="bold", pad=8)
ax.set_xlabel("Week #", color=FG); ax.set_ylabel("Realistic R for that week", color=FG)
ax.tick_params(colors=FG); ax.grid(True, alpha=0.10, color=FG)
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

plt.tight_layout()
p2 = os.path.join(OUT_DIR, "weekly_distribution.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig)
print(f" → chart saved: {p2}")


# ── Plot 3: Loss streak histogram ──
fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
ax.set_facecolor(BG)
bins = list(range(1, max(loss_lens) + 2))
ax.hist(loss_lens, bins=bins, color=RED, alpha=0.85, edgecolor="#161b22")
ax.axvline(np.median(loss_lens), color=YLW, linestyle="--", linewidth=1.3,
           label=f"median {int(np.median(loss_lens))}")
ax.axvline(np.mean(loss_lens), color=BLU, linestyle="--", linewidth=1.3,
           label=f"mean {np.mean(loss_lens):.1f}")
ax.set_title(f"Losing streak distribution  ·  {len(loss_runs)} runs  ·  longest {max(loss_lens)}",
             color=FG, fontsize=12, fontweight="bold", pad=10)
ax.set_xlabel("Consecutive losses in a row", color=FG); ax.set_ylabel("Frequency", color=FG)
ax.legend(facecolor="#161b22", edgecolor="#444c56", labelcolor=FG)
ax.tick_params(colors=FG); ax.grid(True, alpha=0.10, color=FG)
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
plt.tight_layout()
p3 = os.path.join(OUT_DIR, "loss_streak_histogram.png")
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=BG); plt.close(fig)
print(f" → chart saved: {p3}")


# ── Compounded @ multiple risk levels ──
print("\n" + "=" * 90)
print("COMPOUNDED $10k @ VARIOUS RISK LEVELS (realistic R)")
print("=" * 90)
print(f"  {'risk%':>6}  {'end balance':>15}  {'max DD':>8}  {'FTMO -10% safe?':>18}")
for r in [0.25, 0.33, 0.5, 0.75, 1.0, 2.0]:
    eq = compound(df_sorted["pnl_r_real"].values, r)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    mdd = dd.min(); endb = eq[-1]
    safe = "YES" if mdd > -10 else "NO (breach)"
    print(f"  {r:>5.2f}% ${endb:>13,.0f}  {mdd:>7.1f}%  {safe:>18}")
