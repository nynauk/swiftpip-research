"""
Deep streak analysis + equity graph for Option A (top-3 combined, 3RR).
Reads combined_top3_3R_trades.csv and produces:
  · loss/win streak distributions
  · per-symbol streaks
  · longest streak date ranges
  · cumulative R chart
  · compounded equity chart @ multiple risk levels
  · drawdown chart
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "combined_top3_3R_trades.csv"
OUT_DIR = "strategies/top3_optionA"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV, parse_dates=["sweep_time", "entry_time", "exit_time"])
df = df[df["outcome"] != "open"].sort_values("entry_time").reset_index(drop=True)
N = len(df)
print(f"Loaded {N} trades from {df['entry_time'].min()} → {df['entry_time'].max()}\n")


def streaks(outcomes, dates):
    """Return list of dicts: {type: 'win'|'loss', length, start, end}."""
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


all_runs = streaks(df["outcome"].values, df["entry_time"].values)
losses = [r for r in all_runs if r["type"] == "loss"]
wins   = [r for r in all_runs if r["type"] == "win"]

# ── Loss streak stats ──
loss_lens = [r["length"] for r in losses]
print("── LOSS STREAK DISTRIBUTION (all runs combined) ──")
print(f"  Total loss runs      : {len(losses)}")
print(f"  Mean length          : {np.mean(loss_lens):.2f}")
print(f"  Median length        : {int(np.median(loss_lens))}")
print(f"  Longest              : {max(loss_lens)}")
print(f"  Loss runs by length:")
buckets = {"1":0, "2-3":0, "4-6":0, "7-10":0, "11-15":0, "16-20":0, "21+":0}
for L in loss_lens:
    if L == 1: buckets["1"] += 1
    elif L <= 3: buckets["2-3"] += 1
    elif L <= 6: buckets["4-6"] += 1
    elif L <= 10: buckets["7-10"] += 1
    elif L <= 15: buckets["11-15"] += 1
    elif L <= 20: buckets["16-20"] += 1
    else: buckets["21+"] += 1
for k, v in buckets.items():
    pct = v/len(losses)*100
    print(f"    {k:>6}: {v:>4}  ({pct:>5.1f}%)")

# ── Top 5 longest loss streaks with dates ──
print("\n── TOP 5 LONGEST LOSS STREAKS ──")
for r in sorted(losses, key=lambda x: x["length"], reverse=True)[:5]:
    start = pd.Timestamp(r["start"]).strftime("%Y-%m-%d %H:%M")
    end   = pd.Timestamp(r["end"]).strftime("%Y-%m-%d %H:%M")
    days  = (pd.Timestamp(r["end"]) - pd.Timestamp(r["start"])).days
    print(f"  {r['length']:>3} losses in a row  ·  {start}  →  {end}  ({days} days)")

# ── Per-symbol loss streaks ──
print("\n── PER-SYMBOL LOSS STREAKS ──")
print(f"  {'symbol':13} {'total losses':>12} {'longest':>8} {'mean':>7} {'median':>7}")
for sym in df["symbol"].unique():
    sub = df[df["symbol"] == sym].sort_values("entry_time")
    sub_runs = streaks(sub["outcome"].values, sub["entry_time"].values)
    slos = [r["length"] for r in sub_runs if r["type"] == "loss"]
    if slos:
        print(f"  {sym:13} {sum(slos):>12} {max(slos):>8} "
              f"{np.mean(slos):>6.2f}  {int(np.median(slos)):>6}")

# ── WIN STREAKS ──
win_lens = [r["length"] for r in wins]
print("\n── WIN STREAK DISTRIBUTION ──")
print(f"  Total win runs      : {len(wins)}")
print(f"  Mean length         : {np.mean(win_lens):.2f}")
print(f"  Longest             : {max(win_lens)}")
print(f"  Win runs by length:")
wbuckets = {"1":0, "2-3":0, "4-6":0, "7+":0}
for L in win_lens:
    if L == 1: wbuckets["1"] += 1
    elif L <= 3: wbuckets["2-3"] += 1
    elif L <= 6: wbuckets["4-6"] += 1
    else: wbuckets["7+"] += 1
for k, v in wbuckets.items():
    print(f"    {k:>6}: {v:>4}  ({v/len(wins)*100:>5.1f}%)")

# ── EQUITY CURVES ──
# Cumulative R (additive) + compounded @ 3 risk levels
cum_r = df["pnl_r"].cumsum().values

def compound(pnl, risk_pct):
    bal = 10000.0
    eq = [bal]
    for p in pnl:
        bal += bal * risk_pct/100 * p
        eq.append(bal)
    return np.array(eq)

eq_033 = compound(df["pnl_r"].values, 0.33)
eq_050 = compound(df["pnl_r"].values, 0.50)
eq_100 = compound(df["pnl_r"].values, 1.00)

# Drawdown at 0.33%
peak = np.maximum.accumulate(eq_033)
dd_033 = (eq_033 - peak) / peak * 100

# ── PLOT ──
BG, FG = "#0d1117", "#e6edf3"
GRN, RED, YLW, BLU = "#3fb950", "#f85149", "#d29922", "#58a6ff"

fig, axes = plt.subplots(3, 1, figsize=(15, 14), facecolor=BG,
                        gridspec_kw={"height_ratios": [3, 2, 2]})

# 1. Cumulative R (additive, the true edge)
ax = axes[0]; ax.set_facecolor(BG)
ax.plot(cum_r, color=GRN, linewidth=1.2)
ax.axhline(0, color="#444c56", linewidth=0.8, linestyle="--")
ax.fill_between(range(len(cum_r)), 0, cum_r, where=cum_r >= 0,
                alpha=0.15, color=GRN)
ax.fill_between(range(len(cum_r)), 0, cum_r, where=cum_r < 0,
                alpha=0.20, color=RED)
ax.set_title(f"Option A · Top-3 Combined · 3RR · SL disrespect · news-blocked  ·  "
             f"{N} trades  ·  +{int(cum_r[-1])}R total",
             color=FG, fontsize=13, fontweight="bold", pad=10)
ax.set_ylabel("Cumulative R", color=FG)
ax.set_xlabel("Trade #", color=FG)
ax.tick_params(colors=FG); ax.grid(True, alpha=0.10, color=FG)
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# 2. Compounded equity at 3 risk levels
ax = axes[1]; ax.set_facecolor(BG)
ax.plot(eq_033, color=GRN, linewidth=1.4, label=f"0.33% risk  →  ${eq_033[-1]:,.0f}")
ax.plot(eq_050, color=YLW, linewidth=1.4, label=f"0.50% risk  →  ${eq_050[-1]:,.0f}")
ax.plot(eq_100, color=BLU, linewidth=1.4, label=f"1.00% risk  →  ${eq_100[-1]:,.0f}")
ax.axhline(10000, color="#444c56", linewidth=0.8, linestyle="--")
ax.set_yscale("log")
ax.set_title("Compounded equity from $10,000 · 3 risk levels (log scale)",
             color=FG, fontsize=11, fontweight="bold", pad=8)
ax.set_ylabel("Balance ($)", color=FG)
ax.set_xlabel("Trade #", color=FG)
ax.legend(facecolor="#161b22", edgecolor="#444c56", labelcolor=FG, loc="upper left")
ax.tick_params(colors=FG); ax.grid(True, alpha=0.10, color=FG, which="both")
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

# 3. Drawdown chart @ 0.33% risk
ax = axes[2]; ax.set_facecolor(BG)
ax.fill_between(range(len(dd_033)), 0, dd_033, color=RED, alpha=0.35)
ax.plot(dd_033, color=RED, linewidth=0.8)
ax.axhline(-10, color=YLW, linewidth=1.0, linestyle="--", label="-10% FTMO max DD limit")
ax.set_title(f"Drawdown (%) @ 0.33% risk · max DD {dd_033.min():.1f}%",
             color=FG, fontsize=11, fontweight="bold", pad=8)
ax.set_ylabel("Drawdown (%)", color=FG)
ax.set_xlabel("Trade #", color=FG)
ax.legend(facecolor="#161b22", edgecolor="#444c56", labelcolor=FG, loc="lower right")
ax.tick_params(colors=FG); ax.grid(True, alpha=0.10, color=FG)
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "equity_and_drawdown.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"\n → chart saved: {out_path}")

# ── Loss streak histogram (separate chart) ──
fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
ax.set_facecolor(BG)
bins = list(range(1, max(loss_lens) + 2))
ax.hist(loss_lens, bins=bins, color=RED, alpha=0.85, edgecolor="#161b22")
ax.axvline(np.median(loss_lens), color=YLW, linestyle="--", linewidth=1.3,
           label=f"median {int(np.median(loss_lens))}")
ax.axvline(np.mean(loss_lens), color=BLU, linestyle="--", linewidth=1.3,
           label=f"mean {np.mean(loss_lens):.1f}")
ax.set_title(f"Loss-streak distribution · {len(losses)} losing runs · "
             f"longest {max(loss_lens)}",
             color=FG, fontsize=12, fontweight="bold", pad=10)
ax.set_xlabel("Consecutive losses in a row", color=FG)
ax.set_ylabel("Frequency", color=FG)
ax.legend(facecolor="#161b22", edgecolor="#444c56", labelcolor=FG)
ax.tick_params(colors=FG); ax.grid(True, alpha=0.10, color=FG)
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
plt.tight_layout()
out2 = os.path.join(OUT_DIR, "loss_streak_histogram.png")
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f" → chart saved: {out2}")

# ── DD stats table at different risk % ──
def dd_at(risk_pct):
    eq = compound(df["pnl_r"].values, risk_pct)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    return dd.min(), eq[-1]

print("\n── COMPOUNDED $10k @ VARIOUS RISK LEVELS ──")
print(f"  {'risk%':>6} {'end balance':>15} {'max DD':>8}  {'FTMO -10% safe?':>18}")
for r in [0.25, 0.33, 0.5, 0.75, 1.0]:
    mdd, endb = dd_at(r)
    safe = "YES" if mdd > -10 else "NO (breach)"
    print(f"  {r:>5.2f}% ${endb:>13,.0f}  {mdd:>7.1f}%  {safe:>18}")
