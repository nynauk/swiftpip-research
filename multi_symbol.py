"""
Multi-symbol backtest harness.
Fetches H4 + M15 for each symbol (cached), runs the J2 config on each,
aggregates results. Produces a per-symbol table and a combined trade log.
"""
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

import main as bt
import news_filter as nf
import plot_utils as pu


SYMBOLS = [
    "EURUSD", "GBPUSD", "USDCAD", "USDCHF", "USDJPY", "XAUUSD",
    "GER40.cash", "US30.cash", "US500.cash", "UK100.cash",
]

# Cherry-picked winners from the first multi-symbol run (>= 53% WR @ 1RR).
WINNERS = ["EURUSD", "GBPUSD", "USDCAD", "US30.cash", "US500.cash", "UK100.cash"]


# Per-symbol min FVG width (price units). Forex majors ~2 pips = 0.0002.
# USDJPY/XAU/Indices have different point sizes — scale by typical bar range.
MIN_W = {
    "EURUSD":     0.00020,
    "GBPUSD":     0.00020,
    "USDCAD":     0.00020,
    "USDCHF":     0.00020,
    "USDJPY":     0.020,     # JPY pair: ~2 pips at 100+ scale
    "XAUUSD":     0.20,      # gold ~ $0.20
    "GER40.cash": 2.0,       # ~2 index points
    "US30.cash":  3.0,
    "US500.cash": 0.3,
    "UK100.cash": 1.5,
}


def fetch(symbol: str):
    bt.SYMBOL = symbol
    df_h4  = bt.fetch_data(symbol, mt5.TIMEFRAME_H4,  bt.START_DATE, bt.END_DATE, tf_name="H4")
    df_15m = bt.fetch_data(symbol, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
    return df_h4, df_15m


def run_symbol(symbol: str, rr=1.0, sess_start=12, sess_end=16, news_block=False,
               close_loc=0.3, min_w_mult=1.0, htf_bias="OFF", sweep_mode="H4_SWING",
               sl_mode="IFVG_EDGE"):
    df_h4, df_15m = fetch(symbol)
    df_4h_sw = bt.find_swings(df_h4, bt.SWING_LOOKBACK)
    if sweep_mode == "PDH_PDL":
        sweeps = bt.find_pdh_pdl_sweeps(df_15m)
    elif sweep_mode == "ASIAN_RANGE":
        sweeps = bt.find_session_range_sweeps(df_15m, 0, 7, 23, "Asian")
    elif sweep_mode == "LONDON_RANGE":
        sweeps = bt.find_session_range_sweeps(df_15m, 8, 12, 23, "London")
    else:
        sweeps = bt.find_sweeps(df_4h_sw)
    fvgs_15m = bt.find_all_fvgs(df_15m)
    fvgs_4h  = bt.find_all_fvgs(df_4h_sw)

    bt.DISRESPECT_MULT    = 1e9
    bt.MIN_FVG_WIDTH      = MIN_W.get(symbol, 0.0002) * min_w_mult
    bt.REQUIRE_4H_TOUCH   = False
    bt.MAX_CLOSE_LOC      = close_loc
    bt.RR                 = rr
    bt.SESSION_START_HOUR = sess_start
    bt.SESSION_END_HOUR   = sess_end
    bt.HTF_BIAS_MODE      = htf_bias
    bt.SL_MODE            = sl_mode
    bt.NEWS_BLOCK_ENABLED = news_block
    if news_block and not bt.NEWS_BLOCK_SET:
        bt.NEWS_BLOCK_SET = nf.build_news_block_set(
            df_15m.index[0].date(), df_15m.index[-1].date())

    daily_bias = bt.build_daily_bias(df_15m) if htf_bias == "DAILY" else {}
    trades, funnel = bt.run_backtest(df_4h_sw, df_15m, sweeps, fvgs_15m, fvgs_4h, daily_bias)
    df = pd.DataFrame(trades) if trades else pd.DataFrame()

    if not len(df):
        return {"symbol": symbol, "n": 0, "wr": 0, "total_r": 0,
                "df": df, "funnel": funnel,
                "start": df_15m.index[0], "end": df_15m.index[-1]}

    closed = df[df["outcome"] != "open"]
    wr = len(closed[closed["outcome"] == "win"]) / len(closed) * 100 if len(closed) else 0
    df["symbol"] = symbol
    return {
        "symbol":  symbol,
        "n":       len(df),
        "n_closed": len(closed),
        "wr":      wr,
        "total_r": closed["pnl_r"].sum(),
        "df":      df,
        "funnel":  funnel,
        "start":   df_15m.index[0],
        "end":     df_15m.index[-1],
    }


def sweep_table(symbols, rr, sess_start, sess_end, label, news_block=False,
                close_loc=0.3, min_w_mult=1.0, htf_bias="OFF", sweep_mode="H4_SWING",
                sl_mode="IFVG_EDGE"):
    results = []
    for s in symbols:
        try:
            r = run_symbol(s, rr=rr, sess_start=sess_start, sess_end=sess_end,
                           news_block=news_block, close_loc=close_loc,
                           min_w_mult=min_w_mult, htf_bias=htf_bias,
                           sweep_mode=sweep_mode, sl_mode=sl_mode)
            results.append(r)
        except Exception as e:
            print(f"  [!] {s} failed: {e}")

    all_df = pd.concat([r["df"] for r in results if len(r["df"])], ignore_index=True)
    if not len(all_df):
        return None
    closed = all_df[all_df["outcome"] != "open"]
    wr = len(closed[closed["outcome"] == "win"]) / len(closed) * 100 if len(closed) else 0
    r_total = closed["pnl_r"].sum()

    print(f"\n  ── {label} ──")
    print(f"    {'Symbol':14}  {'N':>4}  {'WR':>6}  {'R':>7}")
    for r in results:
        if not r["n"]:
            continue
        print(f"    {r['symbol']:14}  {r['n']:4d}  {r['wr']:5.1f}%  {r['total_r']:+6.1f}R")
    print(f"    {'COMBINED':14}  {len(all_df):4d}  {wr:5.1f}%  {r_total:+6.1f}R")

    # Annual breakdown
    all_df = all_df.copy()
    all_df["year"] = pd.to_datetime(all_df["entry_time"]).dt.year
    print("    per-year:")
    for yr in sorted(all_df["year"].unique()):
        sub = all_df[all_df["year"] == yr]
        cl = sub[sub["outcome"] != "open"]
        ywr = len(cl[cl["outcome"] == "win"]) / len(cl) * 100 if len(cl) else 0
        print(f"      {yr}: {len(sub):3d} trades, WR {ywr:.1f}%, R {cl['pnl_r'].sum():+.1f}")

    return {"results": results, "all_df": all_df, "wr": wr, "r_total": r_total}


def equity_and_streaks(df, initial=10_000, risk_pct=1.0):
    """Compounded equity curve at risk_pct per trade. Returns (equity, max_dd, longest_loss_streak)."""
    bal = initial
    equity = [bal]
    cur_loss = 0
    best_loss = 0
    for _, row in df.iterrows():
        bal += bal * (risk_pct / 100) * row["pnl_r"]
        equity.append(bal)
        if row["outcome"] == "loss":
            cur_loss += 1
            if cur_loss > best_loss:
                best_loss = cur_loss
        elif row["outcome"] == "win":
            cur_loss = 0
    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    return eq, dd.min(), best_loss


def plot_equity(df, label, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eq, max_dd, streak = equity_and_streaks(df)

    BG, FG, GRN, RED = "#0d1117", "#e6edf3", "#3fb950", "#f85149"
    fig, ax = plt.subplots(figsize=(15, 7), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(eq, color=GRN, linewidth=1.3)
    ax.axhline(10_000, color="#444c56", linestyle="--", linewidth=0.8)
    ax.fill_between(range(len(eq)), 10_000, eq,
                    where=[v >= 10_000 for v in eq], alpha=0.15, color=GRN)
    ax.fill_between(range(len(eq)), 10_000, eq,
                    where=[v < 10_000 for v in eq], alpha=0.18, color=RED)
    ax.set_title(f"{label}  |  {len(df)} trades  |  end ${eq[-1]:,.0f}  |  DD {max_dd:.1f}%  |  longest loss {streak}",
                 color=FG, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Trade #", color=FG)
    ax.set_ylabel("Balance ($)", color=FG)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.tick_params(colors=FG)
    ax.grid(True, alpha=0.08, color=FG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  equity chart → {out_path}")


def main():
    print("► Connecting to MT5...")
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    # COMBINATION SWEEP at RR=2.0, news-blocked. New: SL_MODE compared.
    # Columns: (label, syms, rr, ss, se, news, close_loc, min_w_mult, htf_bias, sweep_mode, sl_mode)
    EXPERIMENTS = [
        # 4H swing — IFVG_EDGE vs SWEEP_EXTREME
        ("T1 4H-swing  · 12-18 · IFVG_EDGE",  WINNERS, 2.0, 12, 18, True, 0.3, 1.0, "OFF", "H4_SWING",     "IFVG_EDGE"),
        ("T2 4H-swing  · 12-18 · SWEEP_EXTR", WINNERS, 2.0, 12, 18, True, 0.3, 1.0, "OFF", "H4_SWING",     "SWEEP_EXTREME"),
        ("T3 4H-swing  · 24h   · IFVG_EDGE",  WINNERS, 2.0,  0, 24, True, 0.3, 1.0, "OFF", "H4_SWING",     "IFVG_EDGE"),
        # PDH/PDL
        ("T4 PDH/PDL   · 12-18 · IFVG_EDGE",  WINNERS, 2.0, 12, 18, True, 0.3, 1.0, "OFF", "PDH_PDL",      "IFVG_EDGE"),
        ("T5 PDH/PDL   · 24h   · IFVG_EDGE",  WINNERS, 2.0,  0, 24, True, 0.3, 1.0, "OFF", "PDH_PDL",      "IFVG_EDGE"),
        # Asian range
        ("T6 Asian rng · 8-12  · IFVG_EDGE",  WINNERS, 2.0,  8, 12, True, 0.3, 1.0, "OFF", "ASIAN_RANGE",  "IFVG_EDGE"),
        ("T7 Asian rng · 7-23  · IFVG_EDGE",  WINNERS, 2.0,  7, 23, True, 0.3, 1.0, "OFF", "ASIAN_RANGE",  "IFVG_EDGE"),
        # London range
        ("T8 London rng· 13-23 · IFVG_EDGE",  WINNERS, 2.0, 13, 23, True, 0.3, 1.0, "OFF", "LONDON_RANGE", "IFVG_EDGE"),
        # Tightest quality combos
        ("T9 4H-swing  · 12-18 · close=0.2",  WINNERS, 2.0, 12, 18, True, 0.2, 1.0, "OFF", "H4_SWING",     "IFVG_EDGE"),
    ]

    summaries = []
    for label, syms, rr, ss, se, news, cl, mw, htf, sm, slm in EXPERIMENTS:
        print("=" * 80)
        s = sweep_table(syms, rr, ss, se, label, news_block=news,
                        close_loc=cl, min_w_mult=mw, htf_bias=htf,
                        sweep_mode=sm, sl_mode=slm)
        if s:
            summaries.append((label, s))

    print("\n" + "=" * 72)
    print("FINAL COMPARISON")
    print("=" * 72)
    print(f"{'Config':40}  {'N':>5}  {'WR':>6}  {'R':>8}  {'DD':>7}  {'Streak':>7}")
    print("-" * 72)
    for label, s in summaries:
        eq, max_dd, streak = equity_and_streaks(s["all_df"])
        print(f"{label:40}  {len(s['all_df']):5d}  {s['wr']:5.1f}%  {s['r_total']:+7.1f}R  {max_dd:6.1f}%  {streak:>7}")

    # Save best by R AND best by WR
    best_r  = max(summaries, key=lambda x: x[1]["r_total"])
    best_wr = max(summaries, key=lambda x: x[1]["wr"])
    print(f"\n→ Best by total R : {best_r[0]}")
    print(f"→ Best by WR      : {best_wr[0]}")

    OUT = os.path.join("strategies", "sweep_ifvg")
    os.makedirs(OUT, exist_ok=True)
    best_r[1]["all_df"].to_csv(os.path.join(OUT, "trades.csv"), index=False)
    print(f"\nSaved {OUT}/trades.csv (best-by-R config)")
    plot_equity(best_r[1]["all_df"], best_r[0],   os.path.join(OUT, "equity_best_r.png"))
    plot_equity(best_wr[1]["all_df"], best_wr[0], os.path.join(OUT, "equity_best_wr.png"))

    # Recent-trade chart for the best-by-R config
    best_df = best_r[1]["all_df"]
    closed = best_df[best_df["outcome"] != "open"].sort_values("entry_time")
    if len(closed):
        recent = closed.iloc[-1]
        sym = recent["symbol"]
        df_15m = bt.fetch_data(sym, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
        pu.plot_recent_trade(df_15m, recent.to_dict(),
                              os.path.join(OUT, "recent_trade.png"),
                              title_prefix="SWEEP-IFVG MOST RECENT", symbol=sym)
        print(f"  recent trade  -> {OUT}/recent_trade.png  ({sym}  {recent['entry_time']})")

    mt5.shutdown()
    return

    # Combine all trade logs
    all_df = pd.concat([r["df"] for r in results if len(r["df"])], ignore_index=True)
    if not len(all_df):
        print("\nNo trades across any symbol.")
        return
    all_df = all_df.sort_values("entry_time").reset_index(drop=True)

    print("\n" + "=" * 72)
    print("MULTI-SYMBOL J2 SUMMARY (RR=1.0, 12-16, close=0.3, min_w=2p (per symbol))")
    print("=" * 72)
    print(f"{'Symbol':14}  {'N':>4}  {'WR':>6}  {'R':>7}  {'LongWR':>7}  {'ShortWR':>8}")
    print("-" * 72)
    for r in results:
        if not r["n"]:
            print(f"{r['symbol']:14}  {0:4d}  {'--':>6}  {'--':>7}")
            continue
        df = r["df"]
        cl = df[df["outcome"] != "open"]
        ln = cl[cl["direction"] == "long"]
        sh = cl[cl["direction"] == "short"]
        lwr = len(ln[ln["outcome"] == "win"]) / len(ln) * 100 if len(ln) else 0
        swr = len(sh[sh["outcome"] == "win"]) / len(sh) * 100 if len(sh) else 0
        print(f"{r['symbol']:14}  {r['n']:4d}  {r['wr']:5.1f}%  {r['total_r']:+6.1f}R  {lwr:6.1f}%  {swr:7.1f}%")

    print("-" * 72)
    closed = all_df[all_df["outcome"] != "open"]
    wr = len(closed[closed["outcome"] == "win"]) / len(closed) * 100
    print(f"{'COMBINED':14}  {len(all_df):4d}  {wr:5.1f}%  {closed['pnl_r'].sum():+6.1f}R")
    print()

    # Per-year breakdown of combined
    all_df["year"] = pd.to_datetime(all_df["entry_time"]).dt.year
    for yr in sorted(all_df["year"].unique()):
        sub = all_df[all_df["year"] == yr]
        cl = sub[sub["outcome"] != "open"]
        ywr = len(cl[cl["outcome"] == "win"]) / len(cl) * 100 if len(cl) else 0
        print(f"  {yr}: {len(sub):3d} trades, WR {ywr:.1f}%, R {cl['pnl_r'].sum():+.1f}")

    out = "multi_symbol_trades.csv"
    all_df.to_csv(out, index=False)
    print(f"\nCombined log saved -> {out}")


if __name__ == "__main__":
    main()
