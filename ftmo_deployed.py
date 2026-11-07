"""
FTMO simulation of the DEPLOYED bot exactly as it runs live:
  curated 21 instruments · M3+M5+M15 · 1%/trade · 4-trade/day hard cap ·
  $10k account (+10% payout&reset · -5% daily rolling · -10% overall ·
  €89 re-buy · 80% profit split on payouts).
Also: uncapped comparison, and the same config on a $100k account (scaling view).
Trade stream: the 26-year curated backtest (2000-2026, spread-buffered).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import realistic_ema_dol_rsi as edr

OUT = os.path.join("strategies", "all_pairs")
BG, FG = "#0d1117", "#e6edf3"
GREEN, RED = "#3fb950", "#f85149"
SPLIT = 0.80

df = pd.read_csv(os.path.join(OUT, "curated_trades.csv"))
df["entry_time"] = pd.to_datetime(df["entry_time"])
df["exit_time"] = pd.to_datetime(df["exit_time"])
df = df.sort_values("entry_time").reset_index(drop=True)

# ── apply the live bot's 4-trade/day hard cap (first 4 entries per broker day) ──
df["eday"] = df["entry_time"].dt.normalize()
capped = df.groupby("eday", group_keys=False).head(4).reset_index(drop=True)
yrs = (df.entry_time.max() - df.entry_time.min()).days / 365.25
print("=" * 96)
print("FTMO SIM · DEPLOYED CONFIG · curated 21 · M3+M5+M15 · 1%/trade · 2000-2026")
print("=" * 96)
print(f"  trade stream: {len(df):,} uncapped → {len(capped):,} after 4/day cap "
      f"({len(df)-len(capped):,} clipped, {(len(df)-len(capped))/len(df)*100:.0f}%)")

def run(name, trades, start, fee):
    ps = edr.prop_sim(trades, start=start, risk_usd=start * 0.01)
    gross = ps["banked"]                      # +10% cashed per payout
    trader_net = gross * SPLIT - ps["blown"] * fee
    per_yr = ps["passes"] / yrs
    # payout cadence
    if len(ps["passes_t"]) > 1:
        ts = pd.Series([t for t, _ in ps["passes_t"]])
        gaps = ts.diff().dt.days.dropna()
        med_gap = gaps.median()
    else:
        med_gap = float("nan")
    print(f"\n  {name}")
    print(f"    Payouts   : {ps['passes']}  ({per_yr:.1f}/yr · median {med_gap:.0f} days between)")
    print(f"    Banked    : ${gross:,.0f} gross (+10% cycles)  →  ${gross*SPLIT:,.0f} at 80% split")
    print(f"    Blown     : {ps['blown']} ({ps['blown_daily']} daily / {ps['blown_overall']} overall) "
          f"· fees ${ps['blown']*fee:,.0f}")
    print(f"    NET to you: ${trader_net:,.0f}   (${trader_net/yrs:,.0f}/yr per account)")
    return ps

ps10 = run("DEPLOYED · $10k acct · 4/day cap · €89 fee", capped, 10_000.0, 89.0)
run("comparison · $10k · UNCAPPED", df, 10_000.0, 89.0)
run("scaling view · $100k acct · 4/day cap · $550 fee", capped, 100_000.0, 550.0)

# ── sawtooth chart of the deployed $10k sim ──
fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG); ax.set_facecolor(BG)
ax.plot(ps10["trace_t"], ps10["trace_b"], color=GREEN, linewidth=0.6)
ax.axhline(10_000, color="#7d8590", linestyle="--", linewidth=0.7)
ax.axhline(11_000, color=GREEN, linestyle=":", linewidth=0.9)
ax.axhline(9_000, color=RED, linestyle=":", linewidth=0.9, label="blow floor $9,000")
if ps10["passes_t"]:
    ax.scatter([t for t, _ in ps10["passes_t"]], [11_000] * len(ps10["passes_t"]),
               marker="^", s=18, color=GREEN, zorder=5, label=f"payout ({ps10['passes']})")
if ps10["blows_t"]:
    ax.scatter([t for t, _ in ps10["blows_t"]], [9_000] * len(ps10["blows_t"]),
               marker="x", s=40, color=RED, zorder=5, label=f"blown ({ps10['blown']})")
ax.set_title(f"DEPLOYED bot on a $10k FTMO account · 4/day cap · 1%/trade · "
             f"{ps10['passes']} payouts · {ps10['blown']} blown · 2000-2026",
             color=FG, fontsize=12, fontweight="bold")
ax.set_xlabel("Date", color=FG); ax.set_ylabel("Account balance ($)", color=FG)
ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="lower right")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "ftmo_deployed_10k.png"), dpi=140, facecolor=BG)
plt.close(fig)
print(f"\nSaved → {OUT}/ftmo_deployed_10k.png")
