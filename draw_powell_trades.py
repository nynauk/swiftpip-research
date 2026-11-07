"""
Draw one winning and one losing Powell PO3 trade as candlestick charts
with signal candle, impulse peak, fib OTE zone, SL, TP, entry & exit.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import timedelta
import MetaTrader5 as mt5

OUT_DIR = "strategies/powell_po3_fib"
os.makedirs(OUT_DIR, exist_ok=True)

RR         = 5.0
OTE        = 0.79
LOOKBACK   = 40         # M5 bars to show BEFORE the signal
FORWARD    = 80         # M5 bars to show AFTER the signal


df = pd.read_csv("powell_po3_fib_all-21_trades.csv",
                 parse_dates=["signal_time", "entry_time", "exit_time"])
df = df[df["outcome"] != "open"].reset_index(drop=True)

# ── pick trades ──
# Nice winner: XAUUSD long, big obvious 5R move
win_candidates = df[(df["symbol"] == "XAUUSD")
                    & (df["outcome"] == "win")
                    & (df["direction"] == "long")].copy()
win_candidates["dur"] = (win_candidates["exit_time"] - win_candidates["entry_time"]).dt.total_seconds() / 60
# want a moderate-duration one (not too fast, not too slow)
win_candidates = win_candidates[(win_candidates["dur"] > 60) & (win_candidates["dur"] < 400)]
winner = win_candidates.iloc[len(win_candidates) // 2]

# Nice loser: US500 short, clean stopout
loss_candidates = df[(df["symbol"] == "US500.cash")
                     & (df["outcome"] == "loss")
                     & (df["direction"] == "short")].copy()
loss_candidates["dur"] = (loss_candidates["exit_time"] - loss_candidates["entry_time"]).dt.total_seconds() / 60
loss_candidates = loss_candidates[(loss_candidates["dur"] > 30) & (loss_candidates["dur"] < 300)]
loser = loss_candidates.iloc[len(loss_candidates) // 2]


def fetch_range(sym, start_ts, end_ts):
    """Fetch M5 bars between start_ts and end_ts."""
    bars = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M5,
                                 start_ts.to_pydatetime(),
                                 end_ts.to_pydatetime())
    if bars is None or len(bars) == 0:
        return pd.DataFrame()
    d = pd.DataFrame(bars)
    d["time"] = pd.to_datetime(d["time"], unit="s")
    return d.set_index("time")[["open", "high", "low", "close"]]


def draw_trade(ax, tr, bars):
    BG, FG = "#0d1117", "#e6edf3"
    GRN, RED, YLW, BLU, ORNG = "#3fb950", "#f85149", "#d29922", "#58a6ff", "#f78166"
    ax.set_facecolor(BG)
    o = bars["open"].values; h = bars["high"].values
    l = bars["low"].values;  c = bars["close"].values
    n = len(bars)
    xs = np.arange(n)

    # Candlesticks
    body_w = 0.65
    wick_w = 0.15
    for i in range(n):
        color = GRN if c[i] >= o[i] else RED
        # wick
        ax.plot([i, i], [l[i], h[i]], color=color, linewidth=1.0, zorder=2)
        # body
        top = max(o[i], c[i]); bot = min(o[i], c[i])
        ax.add_patch(Rectangle((i - body_w/2, bot), body_w, max(top - bot, 1e-9),
                               facecolor=color, edgecolor=color, alpha=0.9, zorder=3))

    # locate signal and entry bars
    sig_idx = int(bars.index.get_indexer([tr["signal_time"]], method="nearest")[0])
    ent_idx = int(bars.index.get_indexer([tr["entry_time"]], method="nearest")[0])
    ext_idx = int(bars.index.get_indexer([tr["exit_time"]],  method="nearest")[0])

    entry = tr["entry"]; sl = tr["sl"]; tp = tr["tp"]
    risk = abs(entry - sl)

    # recompute the impulse H_max / L_min from signal to entry bar
    if tr["direction"] == "long":
        # L = SL, H_max = highest high between signal and entry (inclusive)
        L = sl
        H_max = bars["high"].iloc[sig_idx:ent_idx+1].max()
        fib_low  = OTE * L + (1 - OTE) * H_max   # 79% retrace
        fib_high = 0.62 * L + 0.38 * H_max       # 62% retrace (top of OTE zone)
    else:
        H = sl
        L_min = bars["low"].iloc[sig_idx:ent_idx+1].min()
        fib_low  = 0.62 * H + 0.38 * L_min
        fib_high = OTE * H + (1 - OTE) * L_min

    x_left  = min(sig_idx, ent_idx) - 2
    x_right = n - 1

    # Fib OTE zone (62-79%)
    ax.axhspan(min(fib_low, fib_high), max(fib_low, fib_high),
               xmin=(x_left + 0.5) / n, xmax=(ent_idx + 2) / n,
               facecolor=YLW, alpha=0.12, zorder=1)

    # SL, TP, entry lines
    ax.axhline(sl,    color=RED, linewidth=1.2, linestyle="--", alpha=0.85, zorder=4)
    ax.axhline(tp,    color=GRN, linewidth=1.2, linestyle="--", alpha=0.85, zorder=4)
    ax.axhline(entry, color=BLU, linewidth=1.0, linestyle=":", alpha=0.75, zorder=4)

    # Signal candle callout — draw a bold outline around it
    ax.add_patch(Rectangle((sig_idx - body_w/2 - 0.15, min(o[sig_idx], c[sig_idx]) - risk*0.05),
                           body_w + 0.3, abs(c[sig_idx] - o[sig_idx]) + risk*0.10,
                           facecolor="none", edgecolor=ORNG, linewidth=2.0,
                           linestyle="-", zorder=5))

    # Manipulation extreme (rejection wick low/high) marker
    if tr["direction"] == "long":
        ax.plot(sig_idx, l[sig_idx], marker="v", markersize=12,
                markerfacecolor=RED, markeredgecolor="white", zorder=6)
    else:
        ax.plot(sig_idx, h[sig_idx], marker="^", markersize=12,
                markerfacecolor=RED, markeredgecolor="white", zorder=6)

    # Impulse peak marker (H_max or L_min)
    if tr["direction"] == "long":
        peak_idx = sig_idx + int(bars["high"].iloc[sig_idx:ent_idx+1].values.argmax())
        ax.plot(peak_idx, H_max, marker="^", markersize=12,
                markerfacecolor=YLW, markeredgecolor="white", zorder=6)
    else:
        peak_idx = sig_idx + int(bars["low"].iloc[sig_idx:ent_idx+1].values.argmin())
        ax.plot(peak_idx, L_min, marker="v", markersize=12,
                markerfacecolor=YLW, markeredgecolor="white", zorder=6)

    # Entry & exit markers
    ax.plot(ent_idx, entry, marker="o", markersize=12,
            markerfacecolor=BLU, markeredgecolor="white", linewidth=2, zorder=7)
    exit_color = GRN if tr["outcome"] == "win" else RED
    exit_price = tr["exit"] if not pd.isna(tr["exit"]) else (tp if tr["outcome"] == "win" else sl)
    ax.plot(ext_idx, exit_price, marker="X", markersize=14,
            markerfacecolor=exit_color, markeredgecolor="white", linewidth=2, zorder=7)

    # Entry-to-exit connecting line
    ax.plot([ent_idx, ext_idx], [entry, exit_price],
            color=exit_color, linewidth=1.8, alpha=0.7, zorder=6)

    # Right-side labels
    ax.text(n + 1, sl,    f"  SL {sl:.2f}",    color=RED, fontsize=9, va="center")
    ax.text(n + 1, tp,    f"  TP {tp:.2f}  (+5R)", color=GRN, fontsize=9, va="center")
    ax.text(n + 1, entry, f"  Entry {entry:.2f}", color=BLU, fontsize=9, va="center")

    ax.set_xlim(x_left, n + 8)

    # Sample the time axis
    step = max(n // 8, 1)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([bars.index[i].strftime("%b %d\n%H:%M") for i in ticks],
                       color=FG, fontsize=8)
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.08, color=FG)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

    dur_min = (tr["exit_time"] - tr["entry_time"]).total_seconds() / 60
    real_r = (RR - 0.5) if tr["outcome"] == "win" else -1.05  # approx
    title = (f"{tr['symbol']}  ·  {tr['direction'].upper()}  ·  "
             f"{'WIN +' + str(int(RR)) + 'R' if tr['outcome'] == 'win' else 'LOSS -1R'}"
             f"  ·  entered {tr['entry_time'].strftime('%Y-%m-%d %H:%M')}  ·  "
             f"held {int(dur_min)} min")
    ax.set_title(title, color=FG, fontsize=12, fontweight="bold", pad=10)


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    BG, FG = "#0d1117", "#e6edf3"
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), facecolor=BG)

    for ax, tr in zip(axes, [winner, loser]):
        start = tr["signal_time"] - timedelta(minutes=LOOKBACK * 5)
        end   = tr["exit_time"]   + timedelta(minutes=FORWARD * 5)
        bars = fetch_range(tr["symbol"], start, end)
        if bars.empty:
            ax.text(0.5, 0.5, f"no bars for {tr['symbol']}",
                    color=FG, ha="center", va="center", transform=ax.transAxes)
            continue
        draw_trade(ax, tr, bars)

    # Add legend explaining the markers
    legend_lines = [
        "▼ orange marker  =  manipulation extreme (rejection wick)",
        "▲ yellow marker  =  impulse peak (H_max / L_min)",
        "● blue marker    =  entry at 79% OTE",
        "✕ green/red X    =  exit (TP win / SL loss)",
        "Yellow shading   =  fib 62-79% OTE zone",
        "Orange outline   =  signal candle (5M rejection block)",
    ]
    fig.text(0.02, 0.02, "  ".join(legend_lines), color="#8b949e",
             fontsize=8, ha="left", va="bottom")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(OUT_DIR, "trade_examples.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f" → chart saved: {out}")

    print(f"\nWinner: {winner['symbol']} {winner['direction'].upper()} @ "
          f"{winner['entry_time']}  entry {winner['entry']}  SL {winner['sl']}  TP {winner['tp']}")
    print(f"Loser : {loser['symbol']}  {loser['direction'].upper()} @ "
          f"{loser['entry_time']}  entry {loser['entry']}  SL {loser['sl']}  TP {loser['tp']}")

    mt5.shutdown()


if __name__ == "__main__":
    main()
