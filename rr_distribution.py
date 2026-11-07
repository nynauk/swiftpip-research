"""
R-multiple distribution of the deployed EMA+DOL gradient bot.
Reads the trades from backtest_bot.py and reports median/mean R (all trades,
wins, losses) + plots the outcome distribution.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join("strategies", "bot_strategy")
BG, FG = "#0d1117", "#e6edf3"
GREEN, RED, BLUE, AMBER = "#3fb950", "#f85149", "#58a6ff", "#d29922"

df = pd.read_csv(os.path.join(OUT, "bot_trades.csv"))
r = df.loc[df.outcome != "open", "pnl_r"].astype(float)
wins = r[r > 0.01]; losses = r[r < -0.01]

print("=" * 60)
print(f"EMA+DOL gradient bot · R-multiple distribution ({len(r)} trades)")
print("=" * 60)
print(f"  Median R (ALL trades) : {r.median():+.3f}R")
print(f"  Mean   R (ALL trades) : {r.mean():+.3f}R   (this is the edge/expectancy)")
print(f"  Median WIN            : +{wins.median():.3f}R    (mean +{wins.mean():.3f}R)")
print(f"  Median LOSS           : {losses.median():+.3f}R   (mean {losses.mean():+.3f}R)")
print(f"  Win rate              : {len(wins)/len(r)*100:.1f}%   "
      f"(full -1R stops: {(r <= -0.99).sum()}, scratches |R|<0.1: {(r.abs() < 0.1).sum()})")
print(f"  Percentiles  10/25/50/75/90 : "
      f"{np.percentile(r,10):+.2f} / {np.percentile(r,25):+.2f} / "
      f"{np.percentile(r,50):+.2f} / {np.percentile(r,75):+.2f} / {np.percentile(r,90):+.2f}")
print(f"  Best / worst          : {r.max():+.2f}R / {r.min():+.2f}R")

# ── distribution plot ──
fig, ax = plt.subplots(figsize=(13, 6.2), facecolor=BG); ax.set_facecolor(BG)
bins = np.arange(np.floor(r.min() * 2) / 2, np.ceil(r.max() * 2) / 2 + 0.5, 0.5)
counts, edges, patches = ax.hist(r, bins=bins, edgecolor="#0d1117", linewidth=0.6)
for c, patch, e in zip(counts, patches, edges):
    patch.set_facecolor(RED if e < -0.01 else (GREEN if e >= 0.01 else "#7d8590"))
ax.axvline(r.median(), color=BLUE, linestyle="-", linewidth=1.8,
           label=f"median (all) {r.median():+.2f}R")
ax.axvline(r.mean(), color=AMBER, linestyle="--", linewidth=1.8,
           label=f"mean / expectancy {r.mean():+.2f}R")
ax.axvline(0, color="#444c56", linewidth=0.8)
ax.set_title(f"EMA+DOL gradient bot · R-multiple distribution · {len(r)} trades · "
             f"WR {len(wins)/len(r)*100:.0f}% · PF 5.50",
             color=FG, fontsize=13, fontweight="bold")
ax.set_xlabel("Trade outcome (R multiples)", color=FG); ax.set_ylabel("Number of trades", color=FG)
ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG, axis="y")
ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=10, loc="upper right")
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
# annotate the two structural clusters
ax.annotate("full -1R stops", xy=(-1, counts[np.searchsorted(edges, -1, "right") - 1]),
            xytext=(-1, max(counts) * 0.8), color=RED, fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color=RED))
ax.annotate("gradient scratches\n(exit near breakeven)", xy=(0.25, max(counts)),
            xytext=(2.5, max(counts) * 0.9), color=FG, fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color="#7d8590"))
plt.tight_layout()
plt.savefig(os.path.join(OUT, "rr_distribution.png"), dpi=140, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"\nSaved -> {OUT}/rr_distribution.png")
