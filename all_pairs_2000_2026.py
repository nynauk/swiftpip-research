"""
ALL-PAIRS test · M3+M5+M15 · 2000-2026 · HistData (2000-2024) + MT5 (2025-2026).
Universe = every instrument HistData offers that the MT5 broker also quotes:
7 FX majors + 20 crosses + 2 metals + 9 indices + 2 oils (~38 instruments... as
available). EVERY trade is spread-aware: per-instrument spread on entry/exit
(long fills at ask, short SL/TP vs ask) AND the stop widened by spread+1pip.
Flat $1,000 (1%) per trade per engine on one $100k account.
"""
import os
import gc
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import frequency_tune as fq
import test_cross_exit as tc
import test_gradient_exit as tg
import realistic_ema_dol_rsi as edr
import histdata_fetch as hd

EMA = (7, 18)
YEARS = list(range(2000, 2025))
CUT = pd.Timestamp("2025-01-01"); HD_END = pd.Timestamp("2024-12-31 23:59:59")
TFS = [("M3", "3min", mt5.TIMEFRAME_M3), ("M5", "5min", mt5.TIMEFRAME_M5),
       ("M15", "15min", mt5.TIMEFRAME_M15)]
OUT = os.path.join("strategies", "all_pairs")
BG, FG = "#0d1117", "#e6edf3"

# symbol: (histdata_code, [mt5 candidate names], (spread, slippage, min_fvg), pip)
U = {
    # ── FX majors ──
    "EURUSD": ("eurusd", ["EURUSD"], (1.0e-4, 0.3e-4, 2.0e-4), 1e-4),
    "GBPUSD": ("gbpusd", ["GBPUSD"], (1.5e-4, 0.3e-4, 2.0e-4), 1e-4),
    "USDJPY": ("usdjpy", ["USDJPY"], (1.0e-2, 0.3e-2, 2.0e-2), 1e-2),
    "USDCHF": ("usdchf", ["USDCHF"], (1.5e-4, 0.3e-4, 2.0e-4), 1e-4),
    "USDCAD": ("usdcad", ["USDCAD"], (1.5e-4, 0.3e-4, 2.0e-4), 1e-4),
    "AUDUSD": ("audusd", ["AUDUSD"], (1.5e-4, 0.3e-4, 2.0e-4), 1e-4),
    "NZDUSD": ("nzdusd", ["NZDUSD"], (2.0e-4, 0.3e-4, 2.0e-4), 1e-4),
    # ── crosses ──
    "EURGBP": ("eurgbp", ["EURGBP"], (1.5e-4, 0.3e-4, 2.0e-4), 1e-4),
    "EURJPY": ("eurjpy", ["EURJPY"], (1.8e-2, 0.3e-2, 2.0e-2), 1e-2),
    "GBPJPY": ("gbpjpy", ["GBPJPY"], (2.5e-2, 0.4e-2, 3.0e-2), 1e-2),
    "AUDJPY": ("audjpy", ["AUDJPY"], (1.8e-2, 0.3e-2, 2.0e-2), 1e-2),
    "CADJPY": ("cadjpy", ["CADJPY"], (2.0e-2, 0.3e-2, 2.0e-2), 1e-2),
    "CHFJPY": ("chfjpy", ["CHFJPY"], (2.5e-2, 0.4e-2, 3.0e-2), 1e-2),
    "NZDJPY": ("nzdjpy", ["NZDJPY"], (2.2e-2, 0.4e-2, 2.5e-2), 1e-2),
    "EURAUD": ("euraud", ["EURAUD"], (2.5e-4, 0.4e-4, 3.0e-4), 1e-4),
    "EURCAD": ("eurcad", ["EURCAD"], (2.5e-4, 0.4e-4, 3.0e-4), 1e-4),
    "EURCHF": ("eurchf", ["EURCHF"], (2.0e-4, 0.3e-4, 2.0e-4), 1e-4),
    "EURNZD": ("eurnzd", ["EURNZD"], (3.5e-4, 0.5e-4, 4.0e-4), 1e-4),
    "GBPAUD": ("gbpaud", ["GBPAUD"], (3.5e-4, 0.5e-4, 4.0e-4), 1e-4),
    "GBPCAD": ("gbpcad", ["GBPCAD"], (3.5e-4, 0.5e-4, 4.0e-4), 1e-4),
    "GBPCHF": ("gbpchf", ["GBPCHF"], (3.0e-4, 0.5e-4, 3.5e-4), 1e-4),
    "GBPNZD": ("gbpnzd", ["GBPNZD"], (4.5e-4, 0.6e-4, 5.0e-4), 1e-4),
    "AUDCAD": ("audcad", ["AUDCAD"], (2.0e-4, 0.3e-4, 2.5e-4), 1e-4),
    "AUDCHF": ("audchf", ["AUDCHF"], (2.0e-4, 0.3e-4, 2.5e-4), 1e-4),
    "AUDNZD": ("audnzd", ["AUDNZD"], (2.5e-4, 0.4e-4, 3.0e-4), 1e-4),
    "NZDCAD": ("nzdcad", ["NZDCAD"], (2.5e-4, 0.4e-4, 3.0e-4), 1e-4),
    "NZDCHF": ("nzdchf", ["NZDCHF"], (2.5e-4, 0.4e-4, 3.0e-4), 1e-4),
    # ── metals ──
    "XAUUSD": ("xauusd", ["XAUUSD"], (0.25, 0.05, 0.30), 0.1),
    "XAGUSD": ("xagusd", ["XAGUSD"], (0.025, 0.005, 0.015), 0.01),
    # ── indices ──
    "US500.cash":  ("spxusd", ["US500.cash"],  (0.6, 0.1, 1.5), 1.0),
    "US100.cash":  ("nsxusd", ["US100.cash"],  (2.0, 0.4, 5.0), 1.0),
    "GER40.cash":  ("grxeur", ["GER40.cash"],  (1.5, 0.3, 3.0), 1.0),
    "UK100.cash":  ("ukxgbp", ["UK100.cash"],  (1.2, 0.2, 2.0), 1.0),
    "EU50.cash":   ("etxeur", ["EU50.cash"],   (1.5, 0.3, 2.0), 1.0),
    "FRA40.cash":  ("frxeur", ["FRA40.cash", "FR40.cash"], (1.5, 0.3, 3.0), 1.0),
    "JP225.cash":  ("jpxjpy", ["JP225.cash"],  (8.0, 1.5, 15.0), 1.0),
    "AUS200.cash": ("auxaud", ["AUS200.cash"], (2.0, 0.4, 3.0), 1.0),
    "HK50.cash":   ("hkxhkd", ["HK50.cash"],   (8.0, 1.5, 15.0), 1.0),
    # ── energy ──
    "USOIL.cash": ("wtiusd", ["USOIL.cash"], (0.04, 0.01, 0.08), 0.01),
    "UKOIL.cash": ("bcousd", ["UKOIL.cash"], (0.04, 0.01, 0.08), 0.01),
}
GROUPS = [("FX majors", [s for s in U if len(s) == 6 and "USD" in s and not s.startswith("XA")
                         and (s[:3] == "USD" or s[3:] == "USD")]),
          ("FX crosses", [s for s in U if len(s) == 6 and not s.startswith("XA")
                          and s[:3] != "USD" and s[3:] != "USD"]),
          ("Metals", ["XAUUSD", "XAGUSD"]),
          ("Indices", [s for s in U if ".cash" in s and "OIL" not in s]),
          ("Energy", ["USOIL.cash", "UKOIL.cash"])]

# inject specs so the shared strategy code prices every trade with ITS OWN spread
for sym, (code, cands, spec, pip) in U.items():
    fq.SPEC[sym] = spec
    edr.PIP[sym] = pip
    hd.PAIR[sym] = code


def mt5_bars(cands, tfc, n=99999):
    for name in cands:
        if not mt5.symbol_select(name, True):
            continue
        b = mt5.copy_rates_from_pos(name, tfc, 0, n)
        if b is not None and len(b) > 2000:
            df = pd.DataFrame(b); df["time"] = pd.to_datetime(df["time"], unit="s")
            return df.set_index("time")[["open", "high", "low", "close"]]
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    edr.SL_SPREAD_PAD = True
    print("=" * 118, flush=True)
    print(f"ALL-PAIRS · {len(U)} instruments · M3+M5+M15 · 2000-2026 · HistData+MT5 · "
          "per-instrument spread + spread+1pip stop buffer · $1k/$100k", flush=True)
    print("=" * 118, flush=True)

    parts = {tf: [] for tf, _, _ in TFS}
    have_mt5 = mt5.initialize()
    for sym, (code, cands, spec, pip) in U.items():
        got = []
        for tf, rule, tfc in TFS:
            # HistData 2000-2024
            try:
                bars = hd.fetch(sym, YEARS, tf=rule)
            except Exception:
                bars = None
            n_hd = 0
            if bars is not None and len(bars) > 2000:
                base = fq.precompute_base(sym, bars)
                t = tg.run_gradient(sym, base, EMA, fq.BASEP, 1, True)
                if len(t):
                    t = t[pd.to_datetime(t.entry_time) <= HD_END]
                if len(t):
                    parts[tf].append(t); n_hd = len(t)
                del base
            del bars; gc.collect()
            # MT5 2025-2026
            n_mt = 0
            if have_mt5:
                df = mt5_bars(cands, tfc)
                if df is not None:
                    base = fq.precompute_base(sym, df)
                    t = tg.run_gradient(sym, base, EMA, fq.BASEP, 1, True)
                    if len(t):
                        t = t[pd.to_datetime(t.entry_time) >= CUT]
                    if len(t):
                        parts[tf].append(t); n_mt = len(t)
                    del base
                del df; gc.collect()
            got.append(f"{tf}:{n_hd}+{n_mt}")
        print(f"  {sym:12} " + "  ".join(got), flush=True)
    if have_mt5:
        mt5.shutdown()

    d = {tf: (tc.combine(parts[tf]) if parts[tf] else pd.DataFrame()) for tf, _, _ in TFS}
    comb = pd.concat([d["M3"], d["M5"], d["M15"]]).sort_values("entry_time").reset_index(drop=True)
    span0 = pd.to_datetime(comb.entry_time).min(); span1 = pd.to_datetime(comb.entry_time).max()
    days = (span1 - span0).days * 5 / 7

    print("\n" + "=" * 118, flush=True)
    print(f"  span {span0.date()} → {span1.date()} (~{days:.0f} tdays · {days/252:.1f} yrs) · "
          f"{comb.symbol.nunique()} instruments traded", flush=True)
    print(f"  {'stream':10} {'N':>6} {'/day':>6} {'/wk':>5} {'WR':>7} {'PF':>6} {'totR':>9} "
          f"{'DD':>7} {'LS':>4} {'avgW/med':>11} {'End$':>12}", flush=True)
    print("-" * 118, flush=True)
    for name in ["M3", "M5", "M15"]:
        st = edr.stats(d[name])
        cl = d[name][d[name].outcome != "open"]; w = cl.loc[cl.pnl_r > 0.01, "pnl_r"]
        print(f"  {name:10} {st['n']:>6} {st['n']/days:>6.2f} {st['n']/(days/5):>5.1f} "
              f"{st['wr']:>6.1f}% {st['pf']:>6.2f} {st['tot']:>+9.1f} {st['dd']:>6.1f}% {st['ls']:>4} "
              f"+{w.mean():>4.2f}/{w.median():>4.2f} ${st['bal']:>11,.0f}", flush=True)
    st = edr.stats(comb)
    cl = comb[comb.outcome != "open"]; w = cl.loc[cl.pnl_r > 0.01, "pnl_r"]
    print(f"  {'COMBINED':10} {st['n']:>6} {st['n']/days:>6.2f} {st['n']/(days/5):>5.1f} "
          f"{st['wr']:>6.1f}% {st['pf']:>6.2f} {st['tot']:>+9.1f} {st['dd']:>6.1f}% {st['ls']:>4} "
          f"+{w.mean():>4.2f}/{w.median():>4.2f} ${st['bal']:>11,.0f}", flush=True)

    # group + per-instrument breakdown (combined across TFs)
    print("\n  ── by group (combined M3+M5+M15) ──", flush=True)
    for gname, syms in GROUPS:
        g = comb[comb.symbol.isin(syms)]
        if len(g):
            s2 = edr.stats(g)
            print(f"    {gname:11} N={s2['n']:>5}  WR {s2['wr']:>4.0f}%  PF {s2['pf']:>5.2f}  "
                  f"{s2['tot']:>+8.1f}R  DD {s2['dd']:>5.1f}%", flush=True)
    print("\n  ── per instrument (combined, sorted by totR) ──", flush=True)
    rows = []
    for sym in U:
        g = comb[comb.symbol == sym]
        if len(g):
            s2 = edr.stats(g)
            rows.append((sym, s2))
    for sym, s2 in sorted(rows, key=lambda x: -x[1]["tot"]):
        print(f"    {sym:12} N={s2['n']:>5} WR {s2['wr']:>4.0f}% PF {s2['pf']:>6.2f} "
              f"{s2['tot']:>+8.1f}R DD {s2['dd']:>5.1f}%", flush=True)

    # FTMO sim + worst periods at 1%
    ps = edr.prop_sim(comb.assign(exit_time=pd.to_datetime(comb.exit_time)), risk_usd=1000.0)
    print(f"\n  FTMO @1%: {ps['passes']} payouts · banked ${ps['banked']:,.0f} · "
          f"blown {ps['blown']} ({ps['blown_daily']}d/{ps['blown_overall']}o) · "
          f"NET ${ps['net']:,.0f}", flush=True)
    clx = comb[comb.outcome != "open"].copy()
    clx["exit_time"] = pd.to_datetime(clx["exit_time"])
    clx = clx.sort_values("exit_time")
    r = clx["pnl_r"].astype(float); t = clx["exit_time"]
    day = r.groupby(t.dt.normalize()).sum()
    roll24 = pd.Series(r.values, index=t).rolling("24h").sum()
    week = r.groupby(t.dt.to_period("W").dt.start_time).sum()
    month = r.groupby(t.dt.to_period("M").dt.start_time).sum()
    cum = r.cumsum(); ddr = (cum - cum.cummax()).min()
    print(f"  worst day {day.min():+.2f}R · worst 24h {roll24.min():+.2f}R · "
          f"worst week {week.min():+.2f}R · worst month {month.min():+.2f}R · "
          f"deepest DD {ddr:+.2f}R   (at 1%: ×1 = % of account)", flush=True)

    comb.to_csv(os.path.join(OUT, "all_pairs_trades.csv"), index=False)

    # equity chart
    fig, ax = plt.subplots(figsize=(14, 6.5), facecolor=BG); ax.set_facecolor(BG)
    for name, col in [("M3", "#d29922"), ("M5", "#58a6ff"), ("M15", "#3fb950")]:
        s2 = edr.stats(d[name])
        ax.plot(s2["times"], s2["eqa"], color=col, linewidth=1.1, drawstyle="steps-post",
                label=f"{name}: N={s2['n']}, PF {s2['pf']:.2f}, DD {s2['dd']:.1f}%, ${s2['bal']:,.0f}")
    ax.plot(st["times"], st["eqa"], color="#ff7b72", linewidth=2.0, drawstyle="steps-post",
            label=f"COMBINED: N={st['n']}, PF {st['pf']:.2f}, DD {st['dd']:.1f}%, ${st['bal']:,.0f}")
    ax.axhline(100000, color="#444c56", linestyle="--", linewidth=0.8)
    ax.set_title(f"ALL PAIRS ({comb.symbol.nunique()} instruments) · M3+M5+M15 · 2000-2026 · "
                 "spread-aware", color=FG, fontsize=12, fontweight="bold")
    ax.set_xlabel("Date", color=FG); ax.set_ylabel("Balance ($)", color=FG)
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "all_pairs_equity.png"), dpi=140, facecolor=BG)
    plt.close(fig)
    print(f"\nSaved → {OUT}/  (all_pairs_equity.png, all_pairs_trades.csv)", flush=True)


if __name__ == "__main__":
    main()
