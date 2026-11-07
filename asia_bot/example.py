"""
Draw how the bot would trade, from real MT5 M1 data.

Runs the EXACT detector the bot uses (asia_bot.detector) on today's session
(UTC+1); if today produced no confirmed CISD on either pair, walks back day by
day to the most recent one, and simulates the limit fill + SL/TP outcome with
bid/ask spread realism.

    python -m asia_bot.example            # today (falls back to recent days)
    python -m asia_bot.example 2026-07-04 # a specific session date
"""
import os, sys
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MetaTrader5 as mt5

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asia_bot import config as cfg
from asia_bot import detector as det
from live import broker

SPREAD_PIPS = {"EURUSD": 1.0, "USDCHF": 1.5}


def out_png_for(symbol):
    return os.path.join(os.path.dirname(__file__), f"example_trade_{symbol}.png")

# chart ink/surface tokens (project chart system)
BG, PANEL = "#0d1117", "#161b22"
INK, INK2 = "#e6edf3", "#8b949e"
UP, DN    = "#26a69a", "#ef5350"
BLU, YLW  = "#58a6ff", "#d29922"
GRN, RED  = "#3fb950", "#f85149"


def simulate_fill_and_exit(dfc, t1, setup, prices, spread, london_end):
    """Fill the resting limit, then walk to SL/TP (bid bars, ask-adjusted)."""
    h = dfc["high"].values; l = dfc["low"].values
    entry, sl, tp = prices["entry"], prices["sl"], prices["tp"]
    n = len(dfc)
    fill_i = None
    # The limit taps when the bid (chart) touches the entry level — symmetric on
    # both sides. buy: bid falls to it (l <= entry); sell: bid rises (h >= entry).
    for j in range(setup.confirm_i + 1, n):
        if t1[j] >= london_end:
            break
        if setup.direction == "sell" and h[j] >= entry:
            fill_i = j; break
        if setup.direction == "buy" and l[j] <= entry:
            fill_i = j; break
    if fill_i is None:
        return None, None, "no_fill", 0.0
    for j in range(fill_i, min(fill_i + 2000, n)):
        if setup.direction == "sell":
            if h[j] + spread >= sl: return fill_i, j, "loss", -1.0
            if l[j] + spread <= tp: return fill_i, j, "win", cfg.RR
        else:
            if l[j] <= sl: return fill_i, j, "loss", -1.0
            if h[j] >= tp: return fill_i, j, "win", cfg.RR
    return fill_i, None, "open", 0.0


def plot_example(dfc, t1, setup, prices, fill_i, exit_i, outcome, pnl_r, spread):
    sb = det.session_bounds_utc1(setup.session_date)
    entry, sl, tp = prices["entry"], prices["sl"], prices["tp"]

    start_t = sb["london_start"] - pd.Timedelta(minutes=90)
    if exit_i is not None:
        end_t = t1[exit_i] + pd.Timedelta(minutes=30)
    else:
        end_t = sb["london_end"] + pd.Timedelta(minutes=60)
    m = (t1 >= start_t) & (t1 <= end_t)
    w = dfc.loc[m].reset_index(drop=True)
    wt = t1[m]
    gidx = np.where(m)[0]
    g2l = {g: k for k, g in enumerate(gidx)}          # global idx → local x

    fig, ax = plt.subplots(figsize=(20, 10), facecolor=BG)
    ax.set_facecolor(BG)

    # London window band
    lx = [k for k, ts in enumerate(wt) if sb["london_start"] <= ts < sb["london_end"]]
    if lx:
        ax.axvspan(lx[0], lx[-1], color=BLU, alpha=0.045, zorder=0)
        ax.text(lx[0] + 1, w["low"].min(), " LONDON 07:00–10:00 UTC+1",
                color=BLU, fontsize=9, va="bottom", alpha=0.8)

    # candles
    for k, row in w.iterrows():
        colr = UP if row["close"] >= row["open"] else DN
        ax.plot([k, k], [row["low"], row["high"]], color=colr, linewidth=0.7, zorder=2)
        blo = min(row["open"], row["close"])
        bh = abs(row["close"] - row["open"]) or (w["high"].max() - w["low"].min()) * 4e-4
        ax.add_patch(plt.Rectangle((k - 0.36, blo), 0.72, bh, facecolor=colr,
                                   edgecolor=colr, linewidth=0, zorder=3))

    span = max(w["high"].max() - w["low"].min(), 1e-6)

    # Asia levels
    for lvl, name in ((setup.asia_high, "Asia HIGH"), (setup.asia_low, "Asia LOW")):
        ax.axhline(lvl, color=BLU, linewidth=1.1, linestyle="--", alpha=0.8, zorder=4)
        ax.text(0.4, lvl + span * 0.004, f" {name} {lvl:.5f}", color=BLU,
                fontsize=10, fontweight="bold", va="bottom")

    # sweep marker
    if setup.sweep_i in g2l:
        sx = g2l[setup.sweep_i]
        ax.axvline(sx, color=BLU, linestyle=":", alpha=0.55, linewidth=1.1)
        ax.annotate("SWEEP", (sx, setup.sweep_extreme),
                    textcoords="offset points", xytext=(6, 10),
                    color=BLU, fontsize=10, fontweight="bold")

    # delivery series shading + CISD level
    if setup.series_start_i in g2l and setup.confirm_i in g2l:
        s0 = g2l[setup.series_start_i]
        ax.axvspan(s0 - 0.4, g2l[setup.confirm_i] - 0.5, color=YLW, alpha=0.06, zorder=1)
    ax.axhline(setup.cisd_level, color=YLW, linewidth=1.3, zorder=5)
    ax.text(0.4, setup.cisd_level - span * 0.012,
            f" CISD level {setup.cisd_level:.5f} (open of delivery series)",
            color=YLW, fontsize=10, fontweight="bold", va="top")
    if setup.confirm_i in g2l:
        cx = g2l[setup.confirm_i]
        ax.annotate("CISD CONFIRMED\n(close through level)",
                    (cx, setup.confirm_close), textcoords="offset points",
                    xytext=(10, -28 if setup.direction == "sell" else 20),
                    color=YLW, fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=YLW, alpha=0.8))

    # displacement FVG (if the CISD produced one → entry rests on its near edge)
    if setup.entry_source == "fvg" and setup.fvg_i in g2l:
        fx = g2l[setup.fvg_i]
        fx0 = fx - 2                                   # 3-candle FVG pattern
        fx_end = g2l.get(fill_i, fx + 6) if fill_i is not None else fx + 6
        ax.add_patch(plt.Rectangle((fx0 - 0.4, setup.fvg_bottom),
                                   max(fx_end - fx0, 4) + 0.8,
                                   setup.fvg_top - setup.fvg_bottom,
                                   facecolor=BLU, alpha=0.10, edgecolor=BLU,
                                   linewidth=1.0, linestyle="--", zorder=4))
        edge_name = "FVG HIGH" if setup.direction == "buy" else "FVG LOW"
        ax.text(fx0, setup.fvg_top + span * 0.004,
                f" FVG (entry on {edge_name})", color=BLU, fontsize=9,
                fontweight="bold", va="bottom", alpha=0.9)

    # trade boxes from fill (or confirm if no fill)
    bx0 = g2l.get(fill_i if fill_i is not None else setup.confirm_i,
                  len(w) // 2)
    bx1 = g2l.get(exit_i, len(w) - 1) if exit_i is not None else len(w) - 1
    bw = max(bx1 - bx0, 8)
    tlo, thi = sorted([entry, tp]); slo, shi = sorted([entry, sl])
    ax.add_patch(plt.Rectangle((bx0, tlo), bw, thi - tlo, facecolor=GRN,
                               alpha=0.13, edgecolor=GRN, linewidth=1, zorder=4))
    ax.add_patch(plt.Rectangle((bx0, slo), bw, shi - slo, facecolor=RED,
                               alpha=0.13, edgecolor=RED, linewidth=1, zorder=4))
    _entry_lbl = ("FVG " + ("high" if setup.direction == "buy" else "low")) \
        if setup.entry_source == "fvg" else "CISD level"
    for price, colr, lbl, dy in ((entry, INK, f"ENTRY (limit @ {_entry_lbl}) {entry:.5f}", 0.004),
                                 (sl, RED, f"SL (swing {'high' if setup.direction=='sell' else 'low'} + pad) {sl:.5f}", 0.004),
                                 (tp, GRN, f"TP (2R) {tp:.5f}", 0.004)):
        ax.axhline(price, color=colr, linewidth=1.0, alpha=0.9, zorder=5)
        ax.text(bx0 + bw + 0.6, price, f" {lbl}", color=colr, fontsize=10,
                va="center", fontweight="bold")

    if fill_i is not None and fill_i in g2l:
        ax.scatter([g2l[fill_i]], [entry],
                   marker="v" if setup.direction == "sell" else "^",
                   s=220, color=YLW, edgecolor="black", zorder=7,
                   label="Entry fill (limit)")
    if exit_i is not None and exit_i in g2l:
        px = tp if outcome == "win" else sl
        ax.scatter([g2l[exit_i]], [px], marker="o", s=180,
                   color=GRN if outcome == "win" else RED, edgecolor="black",
                   zorder=7, label=f"Exit — {outcome.upper()} ({pnl_r:+.0f}R)")

    step = max(1, len(w) // 12)
    ticks = list(range(0, len(w), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([wt[t].strftime("%H:%M") for t in ticks],
                       rotation=0, fontsize=9, color=INK2)
    ax.set_ylabel("Price (bid)", color=INK2)
    ax.tick_params(colors=INK2)
    ax.grid(True, alpha=0.08, color=INK)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")

    res = {"win": f"WIN +{cfg.RR:.0f}R", "loss": "LOSS -1R",
           "open": "STILL OPEN", "no_fill": "LIMIT NOT FILLED"}[outcome]
    ax.set_title(f"ASIA SWEEP → LONDON 1M CISD · {setup.symbol} · "
                 f"{setup.session_date} (times UTC+1) · "
                 f"{'SHORT' if setup.direction == 'sell' else 'LONG'} · {res}",
                 color=INK, fontsize=13, fontweight="bold", pad=12)
    if fill_i is not None:
        ax.legend(facecolor=PANEL, edgecolor="#444c56", labelcolor=INK,
                  loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png_for(setup.symbol), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    # Args in any order: a session date (YYYY-MM-DD) and/or a symbol to target.
    want = None
    want_symbol = None
    for arg in sys.argv[1:]:
        up = arg.upper()
        if up in cfg.SYMBOLS:
            want_symbol = up
        else:
            want = arg
    search_symbols = [want_symbol] if want_symbol else list(cfg.SYMBOLS)

    t = mt5.symbol_info_tick(cfg.SYMBOLS[0])
    offset_h = det.broker_offset_hours(t.time) if t else 3
    print(f"broker offset: UTC+{offset_h}")

    frames = {}
    for s in search_symbols:
        mt5.symbol_select(s, True)
        bars = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M1, 0, 40_000)
        df = pd.DataFrame(bars)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        frames[s] = df.set_index("time")[["open", "high", "low", "close"]]
    mt5.shutdown()

    today = pd.Timestamp(want).date() if want else \
        (frames[search_symbols[0]].index[-1] - pd.Timedelta(hours=offset_h - 1)).date()

    chosen = None
    for back in range(0, 15):
        day = today - timedelta(days=back)
        if day.weekday() >= 5:
            continue
        for s in search_symbols:
            setup = det.analyze_day(frames[s], s, offset_h, session_date=day)
            if setup is None:
                continue
            tag = setup.stage if setup.stage != "confirmed" else \
                f"confirmed {setup.direction.upper()} @ {setup.cisd_level:.5f}"
            print(f"  {day} {s:8} → {tag}")
            if setup.stage == "confirmed" and chosen is None:
                chosen = (day, s, setup)
        if chosen:
            break

    if not chosen:
        print("No confirmed setup found in the last 15 sessions.")
        return

    day, sym, setup = chosen
    pip = cfg.PIP[sym]
    spread = SPREAD_PIPS.get(sym, 1.0) * pip
    prices = det.build_order_prices(setup, sym, spread)
    dfc = frames[sym].iloc[:-1]
    t1 = det.utc1_index(dfc, offset_h)
    sb = det.session_bounds_utc1(day)
    fill_i, exit_i, outcome, pnl_r = simulate_fill_and_exit(
        dfc, t1, setup, prices, spread, sb["london_end"])

    print(f"\nEXAMPLE: {sym} {day} {'SHORT' if setup.direction=='sell' else 'LONG'}")
    print(f"  Asia range   : {setup.asia_low:.5f} – {setup.asia_high:.5f}")
    print(f"  Sweep        : {setup.sweep_time} UTC+1 (extreme {setup.sweep_extreme:.5f})")
    print(f"  CISD confirm : {setup.confirm_time} UTC+1 · level {setup.cisd_level:.5f}")
    if setup.entry_source == "fvg":
        edge = "high" if setup.direction == "buy" else "low"
        print(f"  FVG produced : {setup.fvg_bottom:.5f} – {setup.fvg_top:.5f} "
              f"→ entry on FVG {edge}")
    else:
        print(f"  FVG produced : none → entry at CISD level")
    print(f"  Entry {prices['entry']:.5f} ({setup.entry_source}) · SL {prices['sl']:.5f} · "
          f"TP {prices['tp']:.5f} ({(prices['risk']/pip):.1f}p stop, {cfg.RR:.0f}R)")
    print(f"  Outcome      : {outcome.upper()} ({pnl_r:+.1f}R)")

    plot_example(dfc, t1, setup, prices, fill_i, exit_i, outcome, pnl_r, spread)
    print(f"\nSaved → {out_png_for(sym)}")


if __name__ == "__main__":
    main()
