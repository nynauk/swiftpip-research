"""
Full stat pack for the base 5m CISD (recent 2019-2026, realistic costs):
win rate, equity curves (combined + per-instrument), drawdown, loss/win-streak
distributions, per-year & per-instrument tables, expectancy, Sharpe.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import MetaTrader5 as mt5

import realistic_ema_dol_rsi as edr
from cisd_dev import run, degrade, load5, SYMS

OUT = os.path.join("strategies", "cisd_dev")
BG, FG, GREEN, RED, BLUE, AMBER = "#0d1117", "#e6edf3", "#3fb950", "#f85149", "#58a6ff", "#d29922"


def streaks(r):
    """Return (loss_streak_lengths, win_streak_lengths)."""
    ls, ws = [], []; cl = cw = 0
    for x in r:
        if x < -0.01:
            cl += 1;
            if cw: ws.append(cw); cw = 0
        elif x > 0.01:
            cw += 1
            if cl: ls.append(cl); cl = 0
        else:
            if cl: ls.append(cl); cl = 0
            if cw: ws.append(cw); cw = 0
    if cl: ls.append(cl)
    if cw: ws.append(cw)
    return ls, ws


def main():
    os.makedirs(OUT, exist_ok=True)
    have = mt5.initialize()
    data = {s: load5(s) for s in SYMS}
    if have:
        mt5.shutdown()
    raw = pd.concat([run(s, data[s]) for s in SYMS], ignore_index=True)
    raw["entry_time"] = pd.to_datetime(raw["entry_time"])
    df = degrade(raw.sort_values("entry_time")).reset_index(drop=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    r = df.pnl_r.astype(float).values
    w = r[r > 0.01]; los = r[r < -0.01]
    yrs = (df.entry_time.max() - df.entry_time.min()).days / 365.25

    cum = np.cumsum(r); peak = np.maximum.accumulate(cum); dd = cum - peak
    ls, ws = streaks(r)
    d = df.copy(); d["day"] = d.entry_time.dt.normalize()
    daily = d.groupby("day").pnl_r.sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252)

    print("=" * 78)
    print("5m CISD (base · recent 2019-2026 · realistic) — FULL STATS")
    print("=" * 78)
    print(f"  Trades              : {len(r):,}  ({len(r)/yrs:.0f}/yr · {len(r)/(yrs*52):.0f}/wk)")
    print(f"  Win / Loss / BE     : {len(w):,} / {len(los):,} / {len(r)-len(w)-len(los):,}")
    print(f"  WIN RATE            : {len(w)/len(r)*100:.1f}%   (breakeven at 2R = 33.3%)")
    print(f"  Profit factor       : {w.sum()/-los.sum():.3f}")
    print(f"  Expectancy          : {r.mean():+.4f}R / trade")
    print(f"  Total               : {cum[-1]:+.0f}R")
    print(f"  Avg win / avg loss  : +{w.mean():.2f}R / {los.mean():.2f}R")
    print(f"  Best / worst trade  : +{r.max():.1f}R / {r.min():.1f}R")
    print(f"  Daily Sharpe (ann)  : {sharpe:.2f}")
    print("  --- streaks ---")
    print(f"  MAX LOSING STREAK   : {max(ls)}    (mean {np.mean(ls):.1f}, 95th pct {np.percentile(ls,95):.0f})")
    print(f"  Max winning streak  : {max(ws)}    (mean {np.mean(ws):.1f})")
    print(f"  Losing streaks >=10 : {sum(1 for x in ls if x>=10)}   >=15: {sum(1 for x in ls if x>=15)}   "
          f">=20: {sum(1 for x in ls if x>=20)}")
    print(f"  --- drawdown (in R) ---")
    print(f"  Max drawdown        : {dd.min():.1f}R")
    # time in worst DD
    trough = int(np.argmin(dd)); pk = int(np.argmax(cum[:trough+1])) if trough > 0 else 0
    print(f"  Worst DD span       : {df.entry_time.iloc[pk].date()} → {df.entry_time.iloc[trough].date()} "
          f"({(df.entry_time.iloc[trough]-df.entry_time.iloc[pk]).days} days)")

    print("\n  per instrument:")
    print(f"  {'sym':12} {'N':>6} {'WR':>6} {'PF':>6} {'totR':>8}")
    for s in SYMS:
        g = df[df.symbol == s]
        if len(g):
            st = edr.stats(g)
            print(f"  {s:12} {st['n']:>6} {st['wr']:>5.1f}% {st['pf']:>6.2f} {st['tot']:>+7.0f}R")
    print("\n  per year:")
    df["yr"] = df.entry_time.dt.year
    for y in sorted(df.yr.unique()):
        g = df[df.yr == y]; st = edr.stats(g)
        gl, _ = streaks(g.pnl_r.values)
        print(f"    {y}: N={st['n']:>5} WR {st['wr']:>4.0f}% PF {st['pf']:>5.2f} {st['tot']:>+6.0f}R "
              f"maxLS {max(gl) if gl else 0}")

    # ── figure ──
    fig, ax = plt.subplots(2, 2, figsize=(15, 9.5), facecolor=BG)
    t = df.entry_time.values
    ax[0, 0].plot(t, cum, color=GREEN, lw=1.2)
    ax[0, 0].axhline(0, color="#444c56", ls="--", lw=0.7)
    ax[0, 0].set_title(f"Equity (cumulative R) · {cum[-1]:+.0f}R · PF {w.sum()/-los.sum():.2f} · "
                       f"Sharpe {sharpe:.2f}", color=FG, fontsize=11, fontweight="bold")
    ax[0, 0].set_ylabel("Cumulative R", color=FG)
    ax[0, 1].fill_between(t, dd, 0, color=RED, alpha=0.5)
    ax[0, 1].set_title(f"Underwater (drawdown in R) · max {dd.min():.0f}R", color=FG, fontsize=11, fontweight="bold")
    ax[0, 1].set_ylabel("Drawdown (R)", color=FG)
    ax[1, 0].hist(ls, bins=range(1, max(ls) + 2), color=RED, edgecolor=BG)
    ax[1, 0].axvline(max(ls), color=AMBER, lw=1.8, label=f"max {max(ls)}")
    ax[1, 0].set_title("Losing-streak length distribution", color=FG, fontsize=11, fontweight="bold")
    ax[1, 0].set_xlabel("Consecutive losses", color=FG); ax[1, 0].set_ylabel("Count", color=FG)
    ax[1, 0].legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9)
    yr_r = df.groupby("yr").pnl_r.sum()
    ax[1, 1].bar(yr_r.index.astype(str), yr_r.values,
                 color=[GREEN if v > 0 else RED for v in yr_r.values], edgecolor=BG)
    ax[1, 1].axhline(0, color="#444c56", lw=0.7)
    ax[1, 1].set_title("R per year", color=FG, fontsize=11, fontweight="bold")
    ax[1, 1].set_ylabel("Total R", color=FG)
    for a in ax.flat:
        a.set_facecolor(BG); a.tick_params(colors=FG); a.grid(True, alpha=0.1, color=FG)
        for sp in a.spines.values(): sp.set_edgecolor("#30363d")
    for a in (ax[0, 0], ax[0, 1]):
        a.xaxis.set_major_locator(mdates.YearLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("5m CISD (base, recent, realistic) · full stats", color=FG, fontsize=13, fontweight="bold")
    fig.patch.set_facecolor(BG)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(OUT, "cisd_full_stats.png"), dpi=140, facecolor=BG)
    plt.close(fig)

    # per-instrument equity curves
    fig, axx = plt.subplots(figsize=(13, 6.5), facecolor=BG); axx.set_facecolor(BG)
    cols = plt.cm.tab10(np.linspace(0, 1, len(SYMS)))
    for s, col in zip(SYMS, cols):
        g = df[df.symbol == s].sort_values("entry_time")
        axx.plot(g.entry_time.values, np.cumsum(g.pnl_r.values), lw=1.2, color=col,
                 label=f"{s} {np.sum(g.pnl_r.values):+.0f}R")
    axx.axhline(0, color="#444c56", ls="--", lw=0.7)
    axx.set_title("Per-instrument equity (cumulative R)", color=FG, fontsize=12, fontweight="bold")
    axx.set_ylabel("Cumulative R", color=FG); axx.tick_params(colors=FG); axx.grid(True, alpha=0.1, color=FG)
    axx.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
    axx.xaxis.set_major_locator(mdates.YearLocator()); axx.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in axx.spines.values(): sp.set_edgecolor("#30363d")
    fig.patch.set_facecolor(BG); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "cisd_per_instrument.png"), dpi=140, facecolor=BG)
    plt.close(fig)
    print(f"\nSaved → {OUT}/  (cisd_full_stats.png, cisd_per_instrument.png)")


if __name__ == "__main__":
    main()
