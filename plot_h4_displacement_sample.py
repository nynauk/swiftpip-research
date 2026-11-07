"""Plot biggest win + most recent trade from the H4-displacement strategy."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MetaTrader5 as mt5

import main as bt


def plot_trade(df_15m, trade, out_path, title_prefix):
    BG, FG = "#0d1117", "#e6edf3"
    GRN, RED, YLW = "#3fb950", "#f85149", "#d29922"
    BLU, PURP, ORG = "#58a6ff", "#bc8cff", "#ff9f43"
    UP, DN = "#26a69a", "#ef5350"

    entry_idx = df_15m.index.get_indexer([pd.Timestamp(trade["entry_time"])], method="nearest")[0]
    exit_idx  = df_15m.index.get_indexer([pd.Timestamp(trade["exit_time"])],  method="nearest")[0]
    disp_idx  = int(trade["displacement_idx"])

    ctx = min(entry_idx - 25, disp_idx - 5)
    lo = max(0, ctx)
    hi = min(len(df_15m), exit_idx + 10)
    win = df_15m.iloc[lo:hi].copy().reset_index()
    win.rename(columns={win.columns[0]: "time"}, inplace=True)

    ex = entry_idx - lo
    xx = exit_idx - lo
    dx = disp_idx - lo

    is_short = trade["direction"] == "short"
    entry, sl, tp = float(trade["entry"]), float(trade["sl"]), float(trade["tp"])

    fig, ax = plt.subplots(figsize=(18, 10), facecolor=BG)
    ax.set_facecolor(BG)

    # Candles — highlight displacement candle with orange border
    for k, row in win.iterrows():
        col = UP if row["close"] >= row["open"] else DN
        ax.plot([k, k], [row["low"], row["high"]], color=col, linewidth=0.9, zorder=2)
        body_lo = min(row["open"], row["close"])
        h = abs(row["close"] - row["open"])
        if h == 0:
            h = (win["high"].max() - win["low"].min()) * 0.0005
        edge = ORG if k == dx else col
        edgewidth = 2.0 if k == dx else 1.0
        ax.add_patch(plt.Rectangle((k - 0.35, body_lo), 0.7, h,
                                   facecolor=col, edgecolor=edge,
                                   linewidth=edgewidth, zorder=3))

    # FVG zone
    ft, fb = float(trade["fvg_top"]), float(trade["fvg_bottom"])
    ax.axhspan(fb, ft, color=PURP, alpha=0.20, zorder=1)
    ax.axhline(ft, color=PURP, linewidth=0.8, linestyle=":", alpha=0.7)
    ax.axhline(fb, color=PURP, linewidth=0.8, linestyle=":", alpha=0.7)
    ax.text(0.5, ft, "  FVG / IFVG zone", color=PURP, fontsize=9, va="bottom", fontweight="bold")

    # Swept level
    if "swept_level" in trade and pd.notna(trade["swept_level"]):
        ax.axhline(float(trade["swept_level"]), color=BLU, linewidth=1.2,
                   linestyle="--", alpha=0.85)
        lbl = "Swept high (4H)" if is_short else "Swept low (4H)"
        ax.text(0.5, float(trade["swept_level"]), f"  {lbl}",
                color=BLU, fontsize=9, va="bottom", fontweight="bold")

    # Displacement candle marker
    if 0 <= dx < len(win):
        disp_extreme = float(trade["displacement_low"]) if not is_short else float(trade["displacement_high"])
        ax.scatter([dx], [disp_extreme], marker="*", s=220, color=ORG,
                   edgecolor="black", zorder=7, label="Displacement candle extreme")
        ax.annotate("FVG-creating\ndisplacement\ncandle",
                    xy=(dx, disp_extreme),
                    xytext=(dx, disp_extreme - (win["high"].max() - win["low"].min()) * 0.10 *
                            (1 if not is_short else -1)),
                    color=ORG, fontsize=9, fontweight="bold", ha="center",
                    arrowprops=dict(arrowstyle="->", color=ORG, lw=1.2))

    # Position boxes
    box_w = max(xx - ex, 8)
    tgt_lo, tgt_hi = sorted([entry, tp])
    ax.add_patch(plt.Rectangle((ex, tgt_lo), box_w, tgt_hi - tgt_lo,
                               facecolor=GRN, alpha=0.15, edgecolor=GRN, linewidth=1.0, zorder=4))
    stp_lo, stp_hi = sorted([entry, sl])
    ax.add_patch(plt.Rectangle((ex, stp_lo), box_w, stp_hi - stp_lo,
                               facecolor=RED, alpha=0.15, edgecolor=RED, linewidth=1.0, zorder=4))

    # Levels + labels
    span = max(win["high"].max() - win["low"].min(), 1e-6)
    min_gap = span * 0.025
    rr_str = f"+{trade['rr_target']:.2f}R" if trade["outcome"] == "win" else f"({trade['pnl_r']:+.2f}R)"
    lvls = sorted([
        (entry, FG,  f"Entry  {entry}"),
        (tp,    GRN, f"TP {rr_str}  {tp}"),
        (sl,    RED, f"SL  {sl}  (displacement {('low' if not is_short else 'high')})"),
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

    # Entry/exit markers
    ax.scatter([ex], [entry], marker="v" if is_short else "^",
               s=220, color=YLW, edgecolor="black", zorder=7, label="Entry")
    ec = GRN if trade["outcome"] == "win" else RED
    ax.scatter([xx], [float(trade["exit"])], marker="o", s=180, color=ec,
               edgecolor="black", zorder=7, label=f"Exit ({trade['outcome'].upper()})")

    # X axis
    n_ticks = 10
    step = max(1, len(win) // n_ticks)
    ticks = list(range(0, len(win), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([win["time"].iloc[t].strftime("%m-%d %H:%M") for t in ticks],
                       rotation=30, ha="right", fontsize=9, color=FG)

    sym = trade["symbol"]
    dur_hr = (pd.to_datetime(trade["exit_time"]) - pd.to_datetime(trade["entry_time"])).total_seconds() / 3600
    ax.set_title(
        f"{title_prefix}  |  {sym} M15  |  {trade['direction'].upper()}  |  "
        f"{trade['outcome'].upper()}  ({trade['pnl_r']:+.2f}R)  |  "
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

    # Annotation panel
    lines = [
        f"Symbol            : {sym}",
        f"Direction         : {trade['direction'].upper()}",
        f"Sweep time (4H)   : {pd.to_datetime(trade['sweep_time']).strftime('%Y-%m-%d %H:%M')}",
        f"Swept level       : {trade['swept_level']}",
        f"FVG form time     : {pd.to_datetime(trade['fvg_form_time']).strftime('%Y-%m-%d %H:%M')}",
        f"Entry time        : {pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d %H:%M')}",
        f"Exit time         : {pd.to_datetime(trade['exit_time']).strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Displacement low  : {trade['displacement_low']}",
        f"Displacement high : {trade['displacement_high']}",
        f"FVG top           : {trade['fvg_top']}",
        f"FVG bottom        : {trade['fvg_bottom']}",
        f"FVG width         : {trade['fvg_width']}",
        "",
        f"Entry             : {entry}",
        f"Stop loss         : {sl}",
        f"Take profit       : {tp}",
        f"Risk distance     : {abs(entry - sl):.5f}",
        f"Reward distance   : {abs(tp - entry):.5f}",
        f"Achieved RR       : {trade['rr_target']:.2f}",
        f"P/L               : {trade['pnl_r']:+.2f}R",
    ]
    annot = "\n".join(lines)
    ax.text(0.985, 0.02, annot, transform=ax.transAxes,
            fontsize=9, color=FG, va="bottom", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#161b22",
                      edgecolor="#444c56", alpha=0.95))

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    OUT = os.path.join("strategies", "h4_displacement")
    for path_in, tag in [
        (os.path.join(OUT, "trades_Fixed_2R___news-blocked.csv"), "Fixed_2R"),
        (os.path.join(OUT, "trades_Fixed_3R___news-blocked.csv"), "Fixed_3R"),
        (os.path.join(OUT, "trades_DOL_TP_ge2R___news-blocked.csv"), "DOL"),
    ]:
        if not os.path.exists(path_in):
            continue
        df = pd.read_csv(path_in, parse_dates=["sweep_time", "fvg_form_time", "entry_time", "exit_time"])
        closed = df[df["outcome"] != "open"].sort_values("entry_time")
        if not len(closed):
            continue
        biggest = closed[closed["outcome"] == "win"].sort_values("pnl_r", ascending=False).iloc[0]
        recent = closed.iloc[-1]
        for tname, t in [("biggest_win", biggest), ("most_recent", recent)]:
            sym = t["symbol"]
            df_15m = bt.fetch_data(sym, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
            out = os.path.join(OUT, f"{tag}_{tname}.png")
            title = f"H4-DISP · {tag.replace('_', ' ')} · {tname.replace('_', ' ').upper()}"
            plot_trade(df_15m, t.to_dict(), out, title)
            print(f"Saved {out}  ({sym} {t['entry_time']}  {t['pnl_r']:+.2f}R)")

    mt5.shutdown()


if __name__ == "__main__":
    main()
