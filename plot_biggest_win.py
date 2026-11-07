"""
Render the biggest-win EMA+DOL trade with full ICT context:
  - EMA9 (fast) and EMA21 (slow) overlays + cross marker
  - FVG zone (the entry imbalance)
  - Entry / SL / TP with annotations
  - Swing high/low context
  - Position boxes (green TP, red SL)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MetaTrader5 as mt5

import main as bt


CSV = os.path.join("strategies", "ema_dol", "trades.csv")
OUT = os.path.join("strategies", "ema_dol", "biggest_win.png")

EMA_FAST = 9
EMA_SLOW = 21


def main():
    df = pd.read_csv(CSV, parse_dates=["entry_time", "exit_time", "cross_time", "fvg_time"])
    closed_wins = df[df["outcome"] == "win"].copy()
    closed_wins = closed_wins.sort_values("pnl_r", ascending=False)
    trade = closed_wins.iloc[0]
    print("Biggest win:")
    print(f"  Symbol         : {trade['symbol']}")
    print(f"  Direction      : {trade['direction']}")
    print(f"  Entry time     : {trade['entry_time']}")
    print(f"  Exit time      : {trade['exit_time']}")
    print(f"  Entry          : {trade['entry']}")
    print(f"  SL             : {trade['sl']}")
    print(f"  TP             : {trade['tp']}")
    print(f"  Achieved RR    : {trade['achieved_rr']}")
    print(f"  PnL            : +{trade['pnl_r']:.2f}R")

    # Fetch M15 data
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    df_15m = bt.fetch_data(trade["symbol"], mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
    mt5.shutdown()

    # Compute EMAs
    ema_f = df_15m["close"].ewm(span=EMA_FAST, adjust=False).mean()
    ema_s = df_15m["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    # Locate trade in the dataframe
    entry_idx = df_15m.index.get_indexer([trade["entry_time"]], method="nearest")[0]
    exit_idx  = df_15m.index.get_indexer([trade["exit_time"]],  method="nearest")[0]
    cross_idx = df_15m.index.get_indexer([trade["cross_time"]], method="nearest")[0]
    fvg_idx   = df_15m.index.get_indexer([trade["fvg_time"]],   method="nearest")[0]

    pad_left = max(40, entry_idx - cross_idx + 10)
    pad_right = 12
    lo = max(0, cross_idx - 10)
    hi = min(len(df_15m), exit_idx + pad_right)
    win = df_15m.iloc[lo:hi].copy().reset_index()
    win.rename(columns={win.columns[0]: "time"}, inplace=True)
    ema_f_w = ema_f.iloc[lo:hi].values
    ema_s_w = ema_s.iloc[lo:hi].values

    cx = cross_idx - lo
    fx = fvg_idx - lo
    ex = entry_idx - lo
    xx = exit_idx - lo

    is_short = trade["direction"] == "short"
    entry, sl, tp = float(trade["entry"]), float(trade["sl"]), float(trade["tp"])

    # ── Plot ──
    BG, FG = "#0d1117", "#e6edf3"
    GRN, RED, YLW = "#3fb950", "#f85149", "#d29922"
    PURP, BLU = "#bc8cff", "#58a6ff"
    UP, DN = "#26a69a", "#ef5350"
    EMA_F_COL = "#ff9f43"   # orange for fast EMA
    EMA_S_COL = "#5b9eff"   # blue   for slow EMA

    fig, ax = plt.subplots(figsize=(18, 10), facecolor=BG)
    ax.set_facecolor(BG)

    # Candles
    for k, row in win.iterrows():
        col = UP if row["close"] >= row["open"] else DN
        ax.plot([k, k], [row["low"], row["high"]], color=col, linewidth=0.9, zorder=2)
        body_lo = min(row["open"], row["close"])
        h = abs(row["close"] - row["open"])
        if h == 0:
            h = (win["high"].max() - win["low"].min()) * 0.0005
        ax.add_patch(plt.Rectangle((k - 0.35, body_lo), 0.7, h,
                                   facecolor=col, edgecolor=col, zorder=3))

    # EMAs
    xs = np.arange(len(win))
    ax.plot(xs, ema_f_w, color=EMA_F_COL, linewidth=1.6, label=f"EMA {EMA_FAST}", zorder=4)
    ax.plot(xs, ema_s_w, color=EMA_S_COL, linewidth=1.6, label=f"EMA {EMA_SLOW}", zorder=4)

    # Cross marker
    if 0 <= cx < len(win):
        ax.axvline(cx, color=YLW, linestyle="--", alpha=0.5, linewidth=1.0, zorder=1)
        ax.scatter([cx], [ema_f_w[cx]], marker="X", s=180, color=YLW,
                   edgecolor="black", zorder=7, label="EMA cross")
        ax.text(cx, win["high"].max(), "  EMA cross",
                color=YLW, fontsize=10, va="top", fontweight="bold", rotation=90)

    # FVG zone
    ft, fb = float(trade["fvg_top"]), float(trade["fvg_bottom"])
    ax.axhspan(fb, ft, color=PURP, alpha=0.20, zorder=1)
    ax.axhline(ft, color=PURP, linewidth=0.8, linestyle=":", alpha=0.7)
    ax.axhline(fb, color=PURP, linewidth=0.8, linestyle=":", alpha=0.7)
    if 0 <= fx < len(win):
        ax.axvline(fx, color=PURP, linestyle=":", alpha=0.4, linewidth=0.8)
        ax.text(fx, ft, "  FVG forms",
                color=PURP, fontsize=9, va="bottom", fontweight="bold")

    # Position boxes
    box_w = max(xx - ex, 8)
    tgt_lo, tgt_hi = sorted([entry, tp])
    ax.add_patch(plt.Rectangle((ex, tgt_lo), box_w, tgt_hi - tgt_lo,
                               facecolor=GRN, alpha=0.15, edgecolor=GRN, linewidth=1.0, zorder=4))
    stp_lo, stp_hi = sorted([entry, sl])
    ax.add_patch(plt.Rectangle((ex, stp_lo), box_w, stp_hi - stp_lo,
                               facecolor=RED, alpha=0.15, edgecolor=RED, linewidth=1.0, zorder=4))

    # Entry / SL / TP lines + labels (staggered)
    span = max(win["high"].max() - win["low"].min(), 1e-6)
    min_gap = span * 0.025
    lvls = sorted([
        (entry, FG,  f"Entry  {entry}"),
        (tp,    GRN, f"TP (DOL = +{trade['achieved_rr']:.2f}R)  {tp}"),
        (sl,    RED, f"SL  {sl}"),
    ], key=lambda x: -x[0])
    last_y = None
    for price, col, lbl in lvls:
        ax.axhline(price, color=col, linewidth=1.0, alpha=0.85, zorder=5)
        y = price
        if last_y is not None and (last_y - y) < min_gap:
            y = last_y - min_gap
        last_y = y
        ax.text(ex + box_w + 0.5, y, f"  {lbl}", color=col,
                fontsize=10, va="center", fontweight="bold")

    # Entry / exit markers
    ax.scatter([ex], [entry], marker="v" if is_short else "^",
               s=220, color=YLW, edgecolor="black", zorder=7, label="Entry")
    ax.scatter([xx], [float(trade["exit"])], marker="o", s=180, color=GRN,
               edgecolor="black", zorder=7, label="Exit (WIN)")

    # X-axis ticks
    n_ticks = 10
    step = max(1, len(win) // n_ticks)
    ticks = list(range(0, len(win), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([win["time"].iloc[t].strftime("%m-%d %H:%M") for t in ticks],
                       rotation=30, ha="right", fontsize=9, color=FG)

    sym = trade["symbol"]
    dur_hr = (pd.to_datetime(trade["exit_time"]) - pd.to_datetime(trade["entry_time"])).total_seconds() / 3600
    ax.set_title(
        f"BIGGEST WIN  |  {sym} M15  |  {trade['direction'].upper()}  |  "
        f"+{trade['pnl_r']:.2f}R  (achieved {trade['achieved_rr']:.2f}R)  |  "
        f"Entry {pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d %H:%M')}  "
        f"|  Duration {dur_hr:.1f}h",
        color=FG, fontsize=12, fontweight="bold", pad=14
    )
    ax.set_ylabel("Price", color=FG)
    ax.tick_params(colors=FG)
    ax.grid(True, alpha=0.10, color=FG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.legend(facecolor="#161b22", edgecolor="#444c56", labelcolor=FG, loc="upper left", fontsize=10)

    # Annotation panel (right side, embedded text box)
    annot = (
        f"Symbol      : {sym}\n"
        f"Direction   : {trade['direction'].upper()}\n"
        f"Cross time  : {pd.to_datetime(trade['cross_time']).strftime('%Y-%m-%d %H:%M')}\n"
        f"FVG time    : {pd.to_datetime(trade['fvg_time']).strftime('%Y-%m-%d %H:%M')}\n"
        f"Entry time  : {pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d %H:%M')}\n"
        f"Exit time   : {pd.to_datetime(trade['exit_time']).strftime('%Y-%m-%d %H:%M')}\n"
        f"\n"
        f"Entry       : {entry}\n"
        f"Stop loss   : {sl}\n"
        f"Take profit : {tp}\n"
        f"Risk dist   : {abs(entry - sl):.5f}\n"
        f"Reward dist : {abs(tp - entry):.5f}\n"
        f"Achieved RR : {trade['achieved_rr']:.2f}\n"
        f"P/L         : +{trade['pnl_r']:.2f}R\n"
        f"Duration    : {dur_hr:.1f} hours\n"
        f"\n"
        f"FVG top     : {trade['fvg_top']}\n"
        f"FVG bottom  : {trade['fvg_bottom']}\n"
        f"FVG width   : {trade['fvg_width']}"
    )
    ax.text(0.985, 0.02, annot, transform=ax.transAxes,
            fontsize=9, color=FG, va="bottom", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#161b22",
                      edgecolor="#444c56", alpha=0.95))

    plt.tight_layout()
    plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
