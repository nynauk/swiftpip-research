"""
CURATED universe: filter the all-pairs (40-instrument) trade stream down to the
instruments with PF >= 2.0 over the full 26 years — the economically-trending
set (majors ex-comm-dollars, JPY crosses, gold, strong indices, oils). Recompute
the full stat pack, FTMO sim @1%, worst periods, and equity vs the full universe.
NOTE: this is in-sample selection (coarse: PF>=2 over 26yr, huge samples), so
treat the uplift as an upper bound — but the cut list is economically coherent
(mean-reverting antipodean crosses + silver), not noise-picking.
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

KEEP = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",                      # majors (PF>=2)
    "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "NZDJPY",            # JPY crosses
    "EURAUD", "GBPAUD",                                          # trending EUR/GBP crosses
    "XAUUSD",                                                    # gold
    "US500.cash", "US100.cash", "GER40.cash", "UK100.cash",      # core indices
    "FRA40.cash", "JP225.cash", "HK50.cash",                     # extra indices
    "USOIL.cash", "UKOIL.cash",                                  # energy
]

full = pd.read_csv(os.path.join(OUT, "all_pairs_trades.csv"))
full["entry_time"] = pd.to_datetime(full["entry_time"])
full["exit_time"] = pd.to_datetime(full["exit_time"])
cur = full[full.symbol.isin(KEEP)].sort_values("entry_time").reset_index(drop=True)
days = (cur.entry_time.max() - cur.entry_time.min()).days * 5 / 7


def pack(name, df):
    st = edr.stats(df)
    cl = df[df.outcome != "open"]; w = cl.loc[cl.pnl_r > 0.01, "pnl_r"]
    print(f"  {name:22} {st['n']:>6} {st['n']/days:>6.2f} {st['wr']:>6.1f}% {st['pf']:>6.2f} "
          f"{st['tot']:>+9.1f} {st['dd']:>6.1f}% {st['ls']:>4} +{w.mean():.2f}/{w.median():.2f} "
          f"${st['bal']:>11,.0f}")
    return st


print("=" * 112)
print(f"CURATED UNIVERSE · {len(KEEP)} instruments (PF>=2 over 26yr) · M3+M5+M15 · 2000-2026 · $1k/$100k")
print("=" * 112)
print(f"  {'stream':22} {'N':>6} {'/day':>6} {'WR':>7} {'PF':>6} {'totR':>9} {'DD':>7} {'LS':>4} "
      f"{'avgW/med':>10} {'End$':>12}")
print("-" * 112)
sc = pack("CURATED combined", cur)
sf = pack("full 40 (reference)", full)
print("\n  per-TF (curated):")
for tf_n, tf_syms in [("M3", None)]:
    pass
# per-TF split: infer TF from... trades don't carry tf column; approximate via hold?
# Not stored — skip per-TF split; the combined is what matters for deployment.

# FTMO sim at several risks
print("\n  ── FTMO sim ($100k · +10% payout/reset · -5% daily · -10% overall · $550/blowup) ──")
print(f"  {'universe':10} {'risk':>6} {'Payouts':>8} {'Banked $':>13} {'Blown':>6} {'NET $':>13}")
for name, df in [("CURATED", cur), ("full 40", full)]:
    for pct in (1.0, 0.5):
        ps = edr.prop_sim(df, risk_usd=1000.0 * pct)
        print(f"  {name:10} {pct:>5.1f}% {ps['passes']:>8} ${ps['banked']:>11,.0f} "
              f"{ps['blown']:>6} ${ps['net']:>11,.0f}")

# worst periods (curated)
cl = cur[cur.outcome != "open"].sort_values("exit_time")
r = cl["pnl_r"].astype(float); t = cl["exit_time"]
day = r.groupby(t.dt.normalize()).sum()
roll24 = pd.Series(r.values, index=t).rolling("24h").sum()
week = r.groupby(t.dt.to_period("W").dt.start_time).sum()
month = r.groupby(t.dt.to_period("M").dt.start_time).sum()
cum = r.cumsum(); ddr = (cum - cum.cummax()).min()
print("\n  ── CURATED worst periods (×1% = % of $100k account) ──")
print(f"  worst day {day.min():+.2f}R ({day.idxmin().date()}) · worst 24h {roll24.min():+.2f}R · "
      f"worst week {week.min():+.2f}R · worst month {month.min():+.2f}R")
print(f"  deepest equity DD {ddr:+.2f}R · days<=-3R: {(day<=-3).sum()} · days<=-5R: {(day<=-5).sum()}")

# year-by-year (curated)
print("\n  ── CURATED year-by-year ──")
cy = cur.copy(); cy["yr"] = cy.entry_time.dt.year
for yr in sorted(cy.yr.unique()):
    st = edr.stats(cy[cy.yr == yr])
    if st:
        print(f"    {yr}: N={st['n']:>4} /day {st['n']/252:>4.2f}  WR {st['wr']:>4.0f}%  "
              f"PF {st['pf']:>5.2f}  {st['tot']:>+7.1f}R  DD {st['dd']:>5.1f}%  LS{st['ls']}")

cur.to_csv(os.path.join(OUT, "curated_trades.csv"), index=False)

# equity chart: curated vs full
fig, ax = plt.subplots(figsize=(14, 6.5), facecolor=BG); ax.set_facecolor(BG)
for name, df, col, lw in [("full 40 instruments", full, "#7d8590", 1.2),
                          (f"CURATED {len(KEEP)}", cur, "#3fb950", 1.8)]:
    st = edr.stats(df)
    ax.plot(st["times"], st["eqa"], color=col, linewidth=lw, drawstyle="steps-post",
            label=f"{name}: N={st['n']}, {st['n']/days:.2f}/day, PF {st['pf']:.2f}, "
                  f"DD {st['dd']:.1f}%, LS {st['ls']}, ${st['bal']:,.0f}")
ax.axhline(100000, color="#444c56", linestyle="--", linewidth=0.8)
ax.set_title("Curated vs full universe · M3+M5+M15 · 2000-2026 · spread-aware · $1k/$100k",
             color=FG, fontsize=12, fontweight="bold")
ax.set_xlabel("Date", color=FG); ax.set_ylabel("Balance ($)", color=FG)
ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "curated_vs_full.png"), dpi=140, facecolor=BG)
plt.close(fig)
print(f"\nSaved → {OUT}/  (curated_vs_full.png, curated_trades.csv)")
