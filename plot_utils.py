"""
Shared plotting utilities — recent-trade chart and equity curve.

All strategies use these so charts look consistent.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BG, FG = "#0d1117", "#e6edf3"
GRN, RED, YLW = "#3fb950", "#f85149", "#d29922"
BLU, PURP = "#58a6ff", "#bc8cff"
UP, DN = "#26a69a", "#ef5350"


def plot_recent_trade(df_15m: pd.DataFrame, trade: dict, out_path: str,
                       title_prefix: str = "RECENT TRADE", symbol: str = ""):
    """
    Render a single trade as a 15M candlestick chart. The `trade` dict needs
    these keys (any not provided will be skipped gracefully):
       required: entry_time, entry_idx, direction, entry, sl, tp, outcome,
                  exit, exit_time, pnl_r, fvg_top, fvg_bottom
       optional: swept_level, sweep_extreme, sweep_time
    """
    entry_idx = int(trade["entry_idx"])
    exit_idx = int(np.searchsorted(df_15m.index, trade["exit_time"], side="left"))
    exit_idx = min(exit_idx, len(df_15m) - 1)
    pad_l, pad_r = 35, 8
    lo = max(0, entry_idx - pad_l)
    hi = min(len(df_15m), exit_idx + pad_r)
    win = df_15m.iloc[lo:hi].copy().reset_index()
    win.rename(columns={win.columns[0]: "time"}, inplace=True)
    ex = entry_idx - lo
    xx = exit_idx - lo

    is_short = trade["direction"] == "short"
    entry, sl, tp = float(trade["entry"]), float(trade["sl"]), float(trade["tp"])

    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
    ax.set_facecolor(BG)

    # candlesticks
    for k, row in win.iterrows():
        col = UP if row["close"] >= row["open"] else DN
        ax.plot([k, k], [row["low"], row["high"]], color=col, linewidth=0.9, zorder=2)
        body_lo = min(row["open"], row["close"])
        h = abs(row["close"] - row["open"])
        if h == 0:
            h = (win["high"].max() - win["low"].min()) * 0.0005
        ax.add_patch(plt.Rectangle((k - 0.3, body_lo), 0.6, h,
                                   facecolor=col, edgecolor=col, zorder=3))

    # IFVG zone
    if "fvg_top" in trade and "fvg_bottom" in trade:
        ft, fb = float(trade["fvg_top"]), float(trade["fvg_bottom"])
        ax.axhspan(fb, ft, color=PURP, alpha=0.14, zorder=1)
        ax.axhline(ft, color=PURP, linewidth=0.8, linestyle=":", alpha=0.7)
        ax.axhline(fb, color=PURP, linewidth=0.8, linestyle=":", alpha=0.7)
        ax.text(0.6, ft, "  FVG / IFVG zone", color=PURP, fontsize=9, va="bottom", fontweight="bold")

    # Swept level (if provided — sweep-IFVG strategies)
    if trade.get("swept_level") is not None and not pd.isna(trade.get("swept_level")):
        lvl = float(trade["swept_level"])
        lbl = "Swept high" if is_short else "Swept low"
        ax.axhline(lvl, color=BLU, linewidth=1.2, linestyle="--", alpha=0.9)
        ax.text(0.6, lvl, f"  {lbl} ({lvl})", color=BLU, fontsize=9, va="bottom", fontweight="bold")
        if trade.get("sweep_extreme") is not None and not pd.isna(trade.get("sweep_extreme")):
            ax.axhline(float(trade["sweep_extreme"]), color=BLU, linewidth=0.8,
                       linestyle=":", alpha=0.6)

    # position boxes — entry→TP (green), entry→SL (red)
    box_x0 = ex
    box_w = max(xx - ex, 6)
    tgt_lo, tgt_hi = sorted([entry, tp])
    ax.add_patch(plt.Rectangle((box_x0, tgt_lo), box_w, tgt_hi - tgt_lo,
                               facecolor=GRN, alpha=0.18, edgecolor=GRN, linewidth=1.0, zorder=4))
    stp_lo, stp_hi = sorted([entry, sl])
    ax.add_patch(plt.Rectangle((box_x0, stp_lo), box_w, stp_hi - stp_lo,
                               facecolor=RED, alpha=0.18, edgecolor=RED, linewidth=1.0, zorder=4))

    # price labels (staggered)
    lvls = sorted([
        (entry, FG,  f"Entry {entry}"),
        (tp,    GRN, f"TP (2R) {tp}"),
        (sl,    RED, f"SL {sl}"),
    ], key=lambda x: -x[0])
    span = max(win["high"].max() - win["low"].min(), 1e-6)
    min_gap = span * 0.025
    last_y = None
    for price, col, lbl in lvls:
        ax.axhline(price, color=col, linewidth=1.0, alpha=0.85, zorder=5)
        y = price
        if last_y is not None and (last_y - y) < min_gap:
            y = last_y - min_gap
        last_y = y
        ax.text(box_x0 + box_w, y, f"  {lbl}", color=col, fontsize=9, va="center", fontweight="bold")

    # markers
    ax.scatter([ex], [entry], marker="v" if is_short else "^",
               s=160, color=YLW, edgecolor="black", zorder=6, label="Entry")
    exit_col = GRN if trade["outcome"] == "win" else (RED if trade["outcome"] == "loss" else YLW)
    ax.scatter([xx], [float(trade["exit"])], marker="o", s=140, color=exit_col,
               edgecolor="black", zorder=6, label=f"Exit ({trade['outcome']})")

    # x ticks
    n_ticks = 8
    step = max(1, len(win) // n_ticks)
    ticks = list(range(0, len(win), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([win["time"].iloc[t].strftime("%m-%d %H:%M") for t in ticks],
                       rotation=30, ha="right", fontsize=8)

    sym_label = f"  |  {symbol}" if symbol else ""
    ax.set_title(
        f"{title_prefix}{sym_label}  |  {trade['direction'].upper()}  |  "
        f"{trade['outcome'].upper()}  ({trade['pnl_r']:+.1f}R)  |  "
        f"{pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d %H:%M')}",
        color=FG, fontsize=12, fontweight="bold", pad=12
    )
    ax.set_ylabel("Price", color=FG)
    ax.tick_params(colors=FG)
    ax.grid(True, alpha=0.08, color=FG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.legend(facecolor="#161b22", edgecolor="#444c56", labelcolor=FG, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_equity(df: pd.DataFrame, out_path: str, label: str, initial=10_000, risk_pct=1.0):
    """Compounded equity curve with DD shading."""
    bal = initial
    eq = [bal]
    cl = 0
    best_cl = 0
    for _, row in df.iterrows():
        bal += bal * (risk_pct / 100) * float(row["pnl_r"])
        eq.append(bal)
        if row["outcome"] == "loss":
            cl += 1
            best_cl = max(best_cl, cl)
        elif row["outcome"] == "win":
            cl = 0
    eq_arr = np.array(eq)
    peak = np.maximum.accumulate(eq_arr)
    dd = ((eq_arr - peak) / peak * 100).min()

    fig, ax = plt.subplots(figsize=(15, 7), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(eq_arr, color=GRN, linewidth=1.2)
    ax.axhline(initial, color="#444c56", linestyle="--", linewidth=0.8)
    ax.fill_between(range(len(eq_arr)), initial, eq_arr,
                    where=eq_arr >= initial, alpha=0.15, color=GRN)
    ax.fill_between(range(len(eq_arr)), initial, eq_arr,
                    where=eq_arr < initial, alpha=0.18, color=RED)
    ax.set_title(
        f"{label}  |  {len(df)} trades  |  end ${eq_arr[-1]:,.0f}  |  "
        f"DD {dd:.1f}%  |  longest loss {best_cl}",
        color=FG, fontsize=12, fontweight="bold", pad=10
    )
    ax.set_xlabel("Trade #", color=FG)
    ax.set_ylabel("Balance ($)", color=FG)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.tick_params(colors=FG)
    ax.grid(True, alpha=0.08, color=FG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
