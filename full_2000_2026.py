"""
FULL 2000 → 2026 backtest of the gradient strategy, using BOTH data sources:
  • HistData (website)  2000-2024  — deep history (FX to 2000, indices to 2010)
  • MT5 (broker feed)   2025-2026  — the recent, broker-accurate tail
Trade streams are stitched at 2025-01-01 (HD entries ≤ 2024-12-31, MT5 entries ≥
2025-01-01) so there's no overlap/double-count. Spread-aware throughout (per-
instrument SPEC + spread+1pip stop buffer), flat $1,000 risk / $100k.

Timeframes: M5, M15, and the COMBINED M5+M15 portfolio. Full breakdown:
trades/day, loss streaks, per-year table, equity curves + underwater (drawdown
relative to time). READ-ONLY; the live bot is a separate process.
"""
import os
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import MetaTrader5 as mt5

import frequency_tune as fq
import test_cross_exit as tc
import test_gradient_exit as tg
import realistic_ema_dol_rsi as edr
import histdata_fetch as hd

EMA = (7, 18)
SYMS = ["EURUSD", "USDJPY", "GBPUSD", "US500.cash", "US100.cash", "GER40.cash", "UK100.cash"]
YEARS_HD = list(range(2000, 2025))          # HistData 2000-2024
CUT = pd.Timestamp("2025-01-01")
HD_END = pd.Timestamp("2024-12-31 23:59:59")
TFS = [("M3", "3min", mt5.TIMEFRAME_M3), ("M5", "5min", mt5.TIMEFRAME_M5),
       ("M15", "15min", mt5.TIMEFRAME_M15)]
OUT = os.path.join("strategies", "full_2000_2026")
RISK = 1000.0
BG, FG = "#0d1117", "#e6edf3"
COL = {"M3": "#d29922", "M5": "#58a6ff", "M15": "#3fb950", "COMBINED": "#ff7b72"}


def mt5_df(sym, tfc, n=99999):
    mt5.symbol_select(sym, True)
    b = mt5.copy_rates_from_pos(sym, tfc, 0, n)
    if b is None or not len(b):
        return None
    df = pd.DataFrame(b); df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.set_index("time")[["open", "high", "low", "close"]]


def daily(df):
    d = df[df.outcome != "open"].copy()
    d["day"] = pd.to_datetime(d["exit_time"]).dt.normalize()
    return d.groupby("day")["pnl_r"].sum() * RISK


def main():
    os.makedirs(OUT, exist_ok=True)
    edr.SL_SPREAD_PAD = True
    print("=" * 118, flush=True)
    print("FULL 2000-2026 · gradient · buffered (spread+1pip) · HistData 2000-2024 + MT5 2025-2026 · "
          f"{len(SYMS)} instruments", flush=True)
    print("=" * 118, flush=True)

    tf_parts = {tf: [] for tf, _, _ in TFS}

    # ── HistData 2000-2024 ──
    print("HistData 2000-2024:", flush=True)
    for s in SYMS:
        for tf, rule, _ in TFS:
            try:
                bars = hd.fetch(s, YEARS_HD, tf=rule)
            except Exception as e:
                print(f"    {s} {tf}: skip ({type(e).__name__})", flush=True); continue
            if len(bars) < 2000:
                continue
            base = fq.precompute_base(s, bars)
            t = tg.run_gradient(s, base, EMA, fq.BASEP, 1, True)
            t = t[pd.to_datetime(t.entry_time) <= HD_END]
            if len(t):
                tf_parts[tf].append(t)
            del bars, base, t; gc.collect()
        print(f"    {s} done", flush=True)

    # ── MT5 2025-2026 ──
    print("MT5 2025-2026:", flush=True)
    if not mt5.initialize():
        print("  MT5 init FAILED — 2025-2026 tail skipped", flush=True)
    else:
        for s in SYMS:
            for tf, _, tfc in TFS:
                df = mt5_df(s, tfc)
                if df is None or len(df) < 2000:
                    continue
                base = fq.precompute_base(s, df)
                t = tg.run_gradient(s, base, EMA, fq.BASEP, 1, True)
                t = t[pd.to_datetime(t.entry_time) >= CUT]
                if len(t):
                    tf_parts[tf].append(t)
                del df, base, t; gc.collect()
            print(f"    {s} done", flush=True)
        mt5.shutdown()

    # ── assemble ──
    d3 = tc.combine(tf_parts["M3"]) if tf_parts["M3"] else pd.DataFrame()
    d5 = tc.combine(tf_parts["M5"]) if tf_parts["M5"] else pd.DataFrame()
    d15 = tc.combine(tf_parts["M15"]) if tf_parts["M15"] else pd.DataFrame()
    comb = pd.concat([d3, d5, d15]).sort_values("entry_time").reset_index(drop=True)
    comb2 = pd.concat([d5, d15]).sort_values("entry_time").reset_index(drop=True)  # no-M3 baseline
    span0 = pd.to_datetime(comb.entry_time).min(); span1 = pd.to_datetime(comb.entry_time).max()
    days = (span1 - span0).days * 5 / 7

    print("\n" + "=" * 118, flush=True)
    print(f"  span {span0.date()} → {span1.date()}  (~{days:.0f} trading days · {days/252:.1f} yrs)", flush=True)
    print(f"  {'stream':12} {'N':>5} {'/day':>6} {'/wk':>6} {'WR':>7} {'PF':>6} {'totR':>9} "
          f"{'DD':>7} {'maxLS':>6} {'avgW/med':>10} {'End$':>11}", flush=True)
    print("-" * 118, flush=True)
    for name, df in [("M3", d3), ("M5", d5), ("M15", d15),
                     ("M5+M15", comb2), ("M3+M5+M15", comb)]:
        st = edr.stats(df)
        cl = df[df.outcome != "open"]; w = cl.loc[cl.pnl_r > 0.01, "pnl_r"]
        print(f"  {name:12} {st['n']:>5} {st['n']/days:>6.2f} {st['n']/(days/5):>6.1f} "
              f"{st['wr']:>6.1f}% {st['pf']:>6.2f} {st['tot']:>+9.1f} {st['dd']:>6.1f}% {st['ls']:>6} "
              f"+{w.mean():>3.2f}/{w.median():>3.2f} ${st['bal']:>10,.0f}", flush=True)

    # ── per-year breakdown (COMBINED): trades/day, WR, PF, DD, loss streak ──
    print("\n  ── COMBINED per-year (trades/day · WR · PF · totR · maxDD · lossStreak) ──", flush=True)
    print(f"  {'yr':4} {'N':>5} {'/day':>6} {'WR':>6} {'PF':>7} {'totR':>8} {'DD':>7} {'LS':>4} {'src':>8}", flush=True)
    cc = comb.copy(); cc["yr"] = pd.to_datetime(cc.entry_time).dt.year
    for yr in sorted(cc.yr.unique()):
        g = cc[cc.yr == yr]; st = edr.stats(g)
        src = "HD" if yr <= 2024 else "MT5"
        yd = 252
        print(f"  {yr:4} {st['n']:>5} {st['n']/yd:>6.2f} {st['wr']:>5.0f}% {st['pf']:>7.2f} "
              f"{st['tot']:>+8.1f} {st['dd']:>6.1f}% {st['ls']:>4} {src:>8}", flush=True)

    comb.to_csv(os.path.join(OUT, "full_trades.csv"), index=False)

    # ── charts: (1) equity vs time, (2) combined underwater, (3) trades/yr ──
    fig, (axE, axU, axT) = plt.subplots(3, 1, figsize=(15, 13), facecolor=BG,
                                        gridspec_kw={"height_ratios": [2.0, 1.0, 1.0]})
    for name, df in [("M3", d3), ("M5", d5), ("M15", d15), ("COMBINED", comb)]:
        st = edr.stats(df)
        axE.plot(st["times"], st["eqa"], color=COL[name],
                 linewidth=2.0 if name == "COMBINED" else 1.2, drawstyle="steps-post",
                 label=f"{name}: N={st['n']}, {st['n']/days:.2f}/day, PF {st['pf']:.2f}, "
                       f"DD {st['dd']:.1f}%, LS {st['ls']}, ${st['bal']:,.0f}")
    axE.axvline(CUT, color="#8b949e", linestyle=":", linewidth=1.0)
    axE.text(CUT, axE.get_ylim()[1]*0.5, " MT5 feed →", color="#8b949e", fontsize=8, rotation=90, va="center")
    axE.axhline(100000, color="#444c56", linestyle="--", linewidth=0.8)
    axE.set_title("Gradient · 2000-2026 · HistData + MT5 · buffered · $1k/$100k each engine",
                  color=FG, fontsize=13, fontweight="bold")
    axE.set_ylabel("Balance ($)", color=FG)
    axE.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
    axE.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # underwater (combined, realized daily, exit-ordered)
    cd = daily(comb)
    ceq = 100000 + cd.cumsum(); peak = ceq.cummax(); uw = (ceq - peak) / peak * 100
    axU.fill_between(uw.index, uw.values, 0, color="#f85149", alpha=0.5)
    axU.plot(uw.index, uw.values, color="#f85149", linewidth=0.8)
    axU.set_title(f"COMBINED underwater (drawdown vs time) · max {uw.min():.1f}%",
                  color=FG, fontsize=11, fontweight="bold")
    axU.set_ylabel("Drawdown (%)", color=FG)

    # trades per year (stacked M3 / M5 / M15)
    y3 = pd.to_datetime(d3.entry_time).dt.year.value_counts().sort_index()
    y5 = pd.to_datetime(d5.entry_time).dt.year.value_counts().sort_index()
    y15 = pd.to_datetime(d15.entry_time).dt.year.value_counts().sort_index()
    yrs = sorted(set(y3.index) | set(y5.index) | set(y15.index))
    b3 = np.array([y3.get(y, 0) for y in yrs]); b5 = np.array([y5.get(y, 0) for y in yrs])
    b15 = np.array([y15.get(y, 0) for y in yrs])
    axT.bar(yrs, b3, color=COL["M3"], label="M3")
    axT.bar(yrs, b5, bottom=b3, color=COL["M5"], label="M5")
    axT.bar(yrs, b15, bottom=b3 + b5, color=COL["M15"], label="M15")
    axT.set_title("Trades per year (M3 + M5 + M15)", color=FG, fontsize=11, fontweight="bold")
    axT.set_ylabel("Trades", color=FG); axT.set_xlabel("Year", color=FG)
    axT.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")

    for ax in (axE, axU, axT):
        ax.set_facecolor(BG); ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    for ax in (axE, axU):
        ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "full_2000_2026.png"), dpi=135, facecolor=BG)
    plt.close(fig)
    print(f"\nSaved → {OUT}/  (full_2000_2026.png, full_trades.csv)", flush=True)


if __name__ == "__main__":
    main()
