"""
Visualise EMA(9/21)+DOL + RSI-momentum(45/55) trades with every ICT component
annotated. Renders the BIGGEST WIN and a representative LOSS across the 7 majors,
so the range is honest (not a hand-picked clean winner).

FVG selection is mechanical: after the EMA cross we take the FIRST aligned FVG
within FVG_LOOKUP_WINDOW bars that meets the min width. No optimisation.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import main as bt
import news_filter as nf
import ema_dol_strategy as ed
import realistic_ema_dol_rsi as edr
import MetaTrader5 as mt5

SYMS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]
RSI_BAND = (45, 100, 0, 55)
OUT = os.path.join("strategies", "realistic_ema_dol_rsi")


def collect_trades(df, symbol):
    spread, slippage, MIN_W = edr.SPEC[symbol]
    h = df["high"].values; l = df["low"].values; c = df["close"].values
    idx = df.index
    ema_f = pd.Series(c, index=idx).ewm(span=ed.EMA_FAST, adjust=False).mean().values
    ema_s = pd.Series(c, index=idx).ewm(span=ed.EMA_SLOW, adjust=False).mean().values
    above = ema_f > ema_s
    cu = np.zeros(len(df), bool); cd = np.zeros(len(df), bool)
    cu[1:] = above[1:] & ~above[:-1]; cd[1:] = ~above[1:] & above[:-1]
    N = len(df)
    next_bull = np.full(N + 1, N, dtype=int); next_bear = np.full(N + 1, N, dtype=int)
    for k in range(N - 1, -1, -1):
        next_bull[k] = k if cu[k] else next_bull[k + 1]
        next_bear[k] = k if cd[k] else next_bear[k + 1]
    sh, sl_arr = ed.find_swings_arr(h, l, n=ed.SWING_LOOKBACK)
    pdh_map, pdl_map = ed.build_pdh_pdl(df)
    rsi_arr = edr.rsi(c)
    news_set = nf.build_news_block_set(idx[0].date(), idx[-1].date())

    trades = []; open_until = []
    for i in range(N):
        if not (cu[i] or cd[i]): continue
        bias = "long" if cu[i] else "short"
        tdir = "bullish" if bias == "long" else "bearish"
        # most-recent-cross rule: setup dies at the next opposite cross
        opp = next_bear[i + 1] if bias == "long" else next_bull[i + 1]
        end_idx = min(i + ed.FVG_LOOKUP_WINDOW, opp - 1, N - 1)
        if end_idx < i + 2: continue
        fvg = ed.find_aligned_fvg(h, l, i, end_idx, tdir, MIN_W)
        if fvg is None: continue
        rt_end = min(fvg["form_idx"] + ed.RETEST_WINDOW, opp - 1, N - 1)
        ei = None
        for k in range(fvg["form_idx"] + 1, rt_end + 1):
            if bias == "long" and l[k] <= fvg["top"]:    ei = k; break
            if bias == "short" and h[k] >= fvg["bottom"]: ei = k; break
        if ei is None: continue
        open_until = [t for t in open_until if t > idx[ei]]
        if len(open_until) >= 1: continue
        if idx[ei].date() in news_set: continue
        rv = rsi_arr[ei]
        if np.isnan(rv): continue
        llo, lhi, slo, shi = RSI_BAND
        if bias == "long" and not (llo <= rv <= lhi): continue
        if bias == "short" and not (slo <= rv <= shi): continue

        d_today = pd.Timestamp(idx[ei]).normalize()
        pdh_t = pdh_map.get(d_today); pdl_t = pdl_map.get(d_today)
        buf = fvg["width"] * ed.SL_BUFFER_FRAC
        if bias == "long":
            entry = fvg["top"] + spread + slippage
            swlow = ed.find_recent_swing_low(sl_arr, ei, lookback=80)
            anchor = swlow if swlow is not None and swlow < fvg["bottom"] else fvg["bottom"]
            sl = anchor - buf; risk = entry - sl
            if risk <= 0: continue
            dol = ed.next_liquidity_above(entry, sh, ei, pdh_t)
            if dol is None: continue
            tp = dol; rr = (tp - entry) / risk; direction = "long"
        else:
            entry = fvg["bottom"] - slippage
            swhigh = ed.find_recent_swing_high(sh, ei, lookback=80)
            anchor = swhigh if swhigh is not None and swhigh > fvg["top"] else fvg["top"]
            sl = anchor + buf; risk = sl - entry
            if risk <= 0: continue
            dol = ed.next_liquidity_below(entry, sl_arr, ei, pdl_t)
            if dol is None: continue
            tp = dol; rr = (entry - tp) / risk; direction = "short"
        if rr < ed.MIN_RR_TO_TAKE or rr > ed.MAX_RR_CAP: continue

        outcome, exit_i = "open", min(ei + 400, len(df) - 1)
        for j in range(ei + 1, min(ei + 400, len(df))):
            if direction == "long":
                if l[j] <= sl: outcome, exit_i = "loss", j; break
                if h[j] >= tp: outcome, exit_i = "win", j; break
            else:
                if h[j] + spread >= sl: outcome, exit_i = "loss", j; break
                if l[j] + spread <= tp: outcome, exit_i = "win", j; break
        open_until.append(idx[exit_i])
        trades.append(dict(symbol=symbol, cross_i=i, fvg_i=fvg["form_idx"], fvg_top=fvg["top"],
                           fvg_bottom=fvg["bottom"], entry_i=ei, entry=entry, sl=sl, tp=tp,
                           rr=rr, direction=direction, rsi=float(rv), exit_i=exit_i,
                           outcome=outcome))
    return trades, ema_f, ema_s, rsi_arr


def plot_trade(t, df, ema_f, ema_s, rsi_arr, out_path, fast=9, slow=21):
    symbol = t["symbol"]
    h = df["high"].values; l = df["low"].values
    o = df["open"].values; c = df["close"].values; idx = df.index
    dig = 3 if "JPY" in symbol else (2 if "." in symbol else 5)
    win = t["outcome"] == "win"
    is_long = t["direction"] == "long"
    lo = max(0, t["cross_i"] - 20)
    hi = min(len(df) - 1, t["exit_i"] + 10)
    # keep candles readable if the winner ran a long way
    if hi - lo > 150:
        lo = max(0, t["entry_i"] - 40)
    xs = np.arange(lo, hi + 1)

    BG, FG = "#0d1117", "#e6edf3"
    UP, DN = "#26a69a", "#ef5350"
    GRN, RED, YLW, PURP, BLU, ORG = "#3fb950", "#f85149", "#f2cc60", "#bc8cff", "#58a6ff", "#ff9f43"

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(16, 10), facecolor=BG,
                                  gridspec_kw={"height_ratios": [3.2, 1]}, sharex=True)
    ax.set_facecolor(BG); axr.set_facecolor(BG)
    for xi in xs:
        col = UP if c[xi] >= o[xi] else DN
        ax.plot([xi, xi], [l[xi], h[xi]], color=col, linewidth=0.7, zorder=2)
        b0 = min(o[xi], c[xi]); bh = abs(c[xi] - o[xi]) or (h[xi] - l[xi]) * 0.02
        ax.add_patch(plt.Rectangle((xi - 0.3, b0), 0.6, bh, facecolor=col, edgecolor=col, zorder=3))
    ax.plot(xs, ema_f[lo:hi + 1], color=YLW, linewidth=1.3, label=f"EMA {fast}", zorder=4)
    ax.plot(xs, ema_s[lo:hi + 1], color=BLU, linewidth=1.3, label=f"EMA {slow}", zorder=4)

    if lo <= t["cross_i"] <= hi:
        ax.axvline(t["cross_i"], color=FG, linestyle=":", alpha=0.5, zorder=1)
        ax.annotate(f"① EMA {fast}/{slow} cross", xy=(t["cross_i"], ema_f[t["cross_i"]]),
                    xytext=(t["cross_i"] + 1, h[lo:hi + 1].max()),
                    color=FG, fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=FG, alpha=0.7))
    ax.add_patch(plt.Rectangle((t["fvg_i"] - 0.5, t["fvg_bottom"]),
                               t["entry_i"] - t["fvg_i"] + 1.5, t["fvg_top"] - t["fvg_bottom"],
                               facecolor=PURP, alpha=0.22, edgecolor=PURP, zorder=1))
    ax.text(t["fvg_i"], t["fvg_top"], "② aligned FVG (first after cross)", color=PURP,
            fontsize=9, va="bottom", fontweight="bold")

    x_e, x_x = t["entry_i"], t["exit_i"]
    ax.hlines(t["entry"], x_e, x_x, color=FG, linewidth=1.2, zorder=5)
    ax.hlines(t["sl"], x_e, x_x, color=RED, linewidth=1.2, zorder=5)
    ax.hlines(t["tp"], x_e, x_x, color=GRN, linewidth=1.2, zorder=5)
    ax.add_patch(plt.Rectangle((x_e, min(t["entry"], t["sl"])), x_x - x_e,
                               abs(t["entry"] - t["sl"]), facecolor=RED, alpha=0.10, zorder=1))
    ax.add_patch(plt.Rectangle((x_e, min(t["entry"], t["tp"])), x_x - x_e,
                               abs(t["entry"] - t["tp"]), facecolor=GRN, alpha=0.10, zorder=1))
    ax.text(x_x + 0.4, t["entry"], f"  ③ Entry {t['entry']:.{dig}f}", color=FG, fontsize=9,
            va="center", fontweight="bold")
    ax.text(x_x + 0.4, t["sl"], f"  SL {t['sl']:.{dig}f} {'(HIT)' if not win else ''}",
            color=RED, fontsize=9, va="center", fontweight="bold")
    ax.text(x_x + 0.4, t["tp"], f"  ④ DOL target {t['tp']:.{dig}f} = {t['rr']:.1f}R "
            f"{'(HIT)' if win else '(not reached)'}", color=GRN, fontsize=9, va="center", fontweight="bold")

    ax.scatter([x_e], [t["entry"]], marker="^" if is_long else "v", s=160,
               color=YLW, edgecolor="black", zorder=7)
    if win:
        ax.scatter([x_x], [t["tp"]], marker="*", s=320, color=GRN, edgecolor="black", zorder=8)
    else:
        ax.scatter([x_x], [t["sl"]], marker="X", s=200, color=RED, edgecolor="black", zorder=8)

    ax.set_ylabel("Price", color=FG)
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.08, color=FG)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor=FG, loc="lower left", fontsize=10)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    dur_h = (idx[x_x] - idx[x_e]).total_seconds() / 3600
    res = f"WIN +{t['rr']:.1f}R" if win else "LOSS -1.0R"
    rescol = GRN if win else RED
    ax.set_title(f"EMA({fast}/{slow})+DOL + RSI momentum · {symbol} M15 · {t['direction'].upper()} · "
                 f"{res} · {idx[x_e].strftime('%Y-%m-%d %H:%M')} · held {dur_h:.0f}h",
                 color=rescol, fontsize=13, fontweight="bold", pad=12)

    axr.plot(xs, rsi_arr[lo:hi + 1], color=ORG, linewidth=1.2)
    axr.axhline(50, color=FG, linestyle="--", alpha=0.4)
    thr = 45 if is_long else 55
    axr.axhline(thr, color=GRN, linestyle="-", alpha=0.5, linewidth=1.0)
    if is_long: axr.axhspan(10, 45, color=RED, alpha=0.07)
    else:       axr.axhspan(55, 90, color=RED, alpha=0.07)
    if lo <= x_e <= hi:
        axr.scatter([x_e], [t["rsi"]], marker="o", s=85, color=GRN, edgecolor="black", zorder=6)
    side = ">= 45 floor (long ✓)" if is_long else "<= 55 cap (short ✓)"
    axr.annotate(f"⑤ RSI {t['rsi']:.0f}  {side}", xy=(x_e, t["rsi"]),
                 xytext=(lo + 2, 78 if is_long else 18), color=FG, fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=FG, alpha=0.7))
    axr.set_ylim(10, 90); axr.set_ylabel("RSI(14)", color=FG)
    axr.tick_params(colors=FG); axr.grid(True, alpha=0.08, color=FG)
    for sp in axr.spines.values(): sp.set_edgecolor("#30363d")
    step = max(1, len(xs) // 10); ticks = list(xs[::step])
    axr.set_xticks(ticks)
    axr.set_xticklabels([idx[tk].strftime("%m-%d %H:%M") for tk in ticks],
                        rotation=25, ha="right", fontsize=8, color=FG)
    axr.set_xlabel("M15 bar time", color=FG)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    os.makedirs(OUT, exist_ok=True)
    data = {}; alltr = []
    for s in SYMS:
        df = bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
        tr, ef, es, rs = collect_trades(df, s)
        data[s] = (df, ef, es, rs)
        alltr += tr

    wins = [t for t in alltr if t["outcome"] == "win"]
    losses = [t for t in alltr if t["outcome"] == "loss"]
    biggest = max(wins, key=lambda t: t["rr"])
    # representative loss: clean context, normal intended RR, readable duration
    losc = [t for t in losses if t["cross_i"] > 60 and 8 <= (t["exit_i"] - t["cross_i"]) <= 120]
    losc.sort(key=lambda t: abs(t["rr"] - 3.0))
    a_loss = losc[len(losc) // 2] if losc else losses[0]

    for t, name in [(biggest, "sample_trade_biggest_win.png"), (a_loss, "sample_trade_loss.png")]:
        df, ef, es, rs = data[t["symbol"]]
        plot_trade(t, df, ef, es, rs, os.path.join(OUT, name))
        print(f"{name}: {t['symbol']} {t['direction']} intended {t['rr']:.1f}R -> {t['outcome']} "
              f"RSI {t['rsi']:.0f}  entry {df.index[t['entry_i']]}")
    print(f"Totals: {len(wins)} wins, {len(losses)} losses (biggest win {biggest['rr']:.1f}R on {biggest['symbol']})")
    mt5.shutdown()


if __name__ == "__main__":
    main()
