"""
1. FTMO prop-firm simulation on the M3+M5+M15 combined portfolio (2000-2026):
   rolling $100k account, +10% -> cash out & reset (a payout), -5% daily (24h) or
   -10% overall -> blown ($550 re-buy fee). Run at the live 1%/engine sizing.
2. Distribution of ALL winning trades' R multiples — count per bucket AND total-R
   contributed per bucket, to show where the reward actually lives.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import realistic_ema_dol_rsi as edr

OUT = os.path.join("strategies", "full_2000_2026")
BG, FG = "#0d1117", "#e6edf3"
GREEN, RED, BLUE, AMBER = "#3fb950", "#f85149", "#58a6ff", "#d29922"

df = pd.read_csv(os.path.join(OUT, "full_trades.csv"))
df["exit_time"] = pd.to_datetime(df["exit_time"])
df["entry_time"] = pd.to_datetime(df["entry_time"])

# ─────────────────── 1) FTMO SIMULATION ───────────────────
print("=" * 78)
print("FTMO PROP-FIRM SIM · M3+M5+M15 combined · 2000-2026 · $100k acct · "
      "+10% payout/reset · -5% daily · -10% overall · $550/blowup")
print("=" * 78)
print(f"  {'risk/trade':>11} {'Payouts':>8} {'Banked $':>13} {'Blown':>6} "
      f"{'(daily/overall)':>16} {'Fees $':>9} {'NET $':>14}")
res_1pct = None
for pct in (0.5, 1.0, 2.0):
    ps = edr.prop_sim(df, risk_usd=100000 * pct / 100)
    if pct == 1.0:
        res_1pct = ps
    print(f"  {pct:>9.1f}% {ps['passes']:>8} ${ps['banked']:>11,.0f} {ps['blown']:>6} "
          f"{'('+str(ps['blown_daily'])+'/'+str(ps['blown_overall'])+')':>16} "
          f"${ps['fees']:>7,.0f} ${ps['net']:>12,.0f}")

ps = res_1pct
print(f"\n  >> At your live 1%/engine: {ps['passes']} payouts, ${ps['banked']:,.0f} banked, "
      f"{ps['blown']} blown, NET ${ps['net']:,.0f} per account cycle over 26 yrs")

# sawtooth chart of the account at 1%
fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG); ax.set_facecolor(BG)
ax.plot(ps["trace_t"], ps["trace_b"], color=GREEN, linewidth=0.7)
ax.axhline(100000, color="#7d8590", linestyle="--", linewidth=0.7)
ax.axhline(110000, color=GREEN, linestyle=":", linewidth=0.9)
ax.axhline(90000, color=RED, linestyle=":", linewidth=0.9)
if ps["passes_t"]:
    ax.scatter([t for t, _ in ps["passes_t"]], [110000] * len(ps["passes_t"]),
               marker="^", s=22, color=GREEN, zorder=5, label=f"payout ({ps['passes']})")
if ps["blows_t"]:
    ax.scatter([t for t, _ in ps["blows_t"]], [90000] * len(ps["blows_t"]),
               marker="x", s=40, color=RED, zorder=5, label=f"blown ({ps['blown']})")
ax.set_title(f"FTMO account cycle · M3+M5+M15 @ 1%/engine · banked ${ps['banked']:,.0f} · "
             f"{ps['passes']} payouts · {ps['blown']} blown", color=FG, fontsize=12, fontweight="bold")
ax.set_xlabel("Date", color=FG); ax.set_ylabel("Account balance ($)", color=FG)
ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "ftmo_sawtooth.png"), dpi=140, facecolor=BG); plt.close(fig)

# ─────────────────── 2) WINNING-TRADE R DISTRIBUTION ───────────────────
w = df.loc[df["pnl_r"] > 0.01, "pnl_r"].astype(float).values
edges = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 15, 25]
labels = ["0-.25", ".25-.5", ".5-.75", ".75-1", "1-1.5", "1.5-2", "2-3", "3-4",
          "4-5", "5-7.5", "7.5-10", "10-15", "15-25"]
cnt, _ = np.histogram(w, bins=edges)
rsum = np.array([w[(w >= edges[i]) & (w < edges[i + 1])].sum() for i in range(len(edges) - 1)])
totw = w.sum()

print("\n" + "=" * 78)
print(f"WINNING-TRADE R DISTRIBUTION · {len(w):,} winners · total +{totw:.0f}R · "
      f"median +{np.median(w):.2f}R · mean +{w.mean():.2f}R")
print("=" * 78)
print(f"  {'bucket(R)':>9} {'#trades':>8} {'%wins':>7} {'sumR':>8} {'%ofR':>7}")
for i in range(len(labels)):
    print(f"  {labels[i]:>9} {cnt[i]:>8} {cnt[i]/len(w)*100:>6.1f}% {rsum[i]:>+7.0f} {rsum[i]/totw*100:>6.1f}%")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
x = np.arange(len(labels))
a1.bar(x, cnt, color=GREEN, edgecolor=BG)
a1.axvline(np.searchsorted(edges, np.median(w)) - 1.0, color=BLUE, linewidth=1.8,
           label=f"median +{np.median(w):.2f}R")
a1.axvline(np.searchsorted(edges, w.mean()) - 1.0, color=AMBER, linewidth=1.8, linestyle="--",
           label=f"mean +{w.mean():.2f}R")
a1.set_title(f"Where the WINNERS are (count) · {len(w):,} winning trades",
             color=FG, fontsize=12, fontweight="bold")
a1.set_ylabel("Number of winning trades", color=FG)
a1.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=10)

a2.bar(x, rsum, color=AMBER, edgecolor=BG)
a2.set_title("Where the REWARD is (total R contributed per bucket)",
             color=FG, fontsize=12, fontweight="bold")
a2.set_ylabel("Total R from bucket", color=FG)

for a in (a1, a2):
    a.set_xticks(x); a.set_xticklabels(labels, rotation=45, ha="right", color=FG, fontsize=8)
    a.set_xlabel("Winning-trade size (R multiple)", color=FG)
    a.set_facecolor(BG); a.tick_params(colors=FG); a.grid(True, alpha=0.1, color=FG, axis="y")
    for sp in a.spines.values(): sp.set_edgecolor("#30363d")
fig.suptitle("Winning-trade R distribution · M3+M5+M15 · 2000-2026",
             color=FG, fontsize=13, fontweight="bold")
fig.patch.set_facecolor(BG)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT, "winners_distribution.png"), dpi=140, facecolor=BG); plt.close(fig)
print(f"\nSaved → {OUT}/  (ftmo_sawtooth.png, winners_distribution.png)")
