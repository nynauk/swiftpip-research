"""
Gradient-exit sample trades (largest win / normal / loss) + a trades &
loss-streaks overview. EMA 7/18, DOL>=2R filter, RSI 45/55, gradient exit,
17 instruments, realistic, $1k/$100k.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import MetaTrader5 as mt5

import main as bt
import ema_dol_strategy as ed
import realistic_ema_dol_rsi as edr
import frequency_tune as fq

EMA = (7, 18)
RISK_USD = 1000.0
OUT = os.path.join("strategies", "ema718_final")


def load(s):
    pth = os.path.join("data_cache", f"{s}_M15.pkl")
    try:
        return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
    except Exception:
        if os.path.exists(pth): os.remove(pth)
        return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")


def collect(symbol, df):
    spread, slippage, min_w = fq.SPEC[symbol]
    h = df["high"].values; l = df["low"].values; c = df["close"].values; o = df["open"].values
    idx = df.index
    ef = pd.Series(c, index=idx).ewm(span=EMA[0], adjust=False).mean().values
    es = pd.Series(c, index=idx).ewm(span=EMA[1], adjust=False).mean().values
    cu, cd, nb, nbr = fq.cross_arrays(c, idx, EMA[0], EMA[1])
    sh, sl_arr = ed.find_swings_arr(h, l, n=ed.SWING_LOOKBACK)
    pdh_map, pdl_map = ed.build_pdh_pdl(df)
    rsi_arr = edr.rsi(c)
    import news_filter as nf
    news_set = nf.build_news_block_set(idx[0].date(), idx[-1].date())
    dig = 3 if "JPY" in symbol else (2 if "." in symbol else 5)
    N = len(df); p = fq.BASEP
    out = []; open_until = []
    for i in range(N):
        if not (cu[i] or cd[i]): continue
        bias = "long" if cu[i] else "short"
        tdir = "bullish" if bias == "long" else "bearish"
        opp = nbr[i + 1] if bias == "long" else nb[i + 1]
        end_idx = min(i + p["fvg_window"], opp - 1, N - 1)
        if end_idx < i + 2: continue
        fvg = ed.find_aligned_fvg(h, l, i, end_idx, tdir, min_w)
        if fvg is None: continue
        rt_end = min(fvg["form_idx"] + p["retest_window"], opp - 1, N - 1)
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
        if bias == "long" and not (45 <= rv <= 100): continue
        if bias == "short" and not (0 <= rv <= 55): continue
        d_today = pd.Timestamp(idx[ei]).normalize()
        pdh_t = pdh_map.get(d_today); pdl_t = pdl_map.get(d_today)
        buf = fvg["width"] * p["sl_buf"]
        if bias == "long":
            entry = fvg["top"] + spread + slippage
            swlow = ed.find_recent_swing_low(sl_arr, ei, lookback=80)
            anchor = swlow if swlow is not None and swlow < fvg["bottom"] else fvg["bottom"]
            sl = anchor - buf; risk = entry - sl; direction = "long"
            if risk <= 0: continue
            dol = ed.next_liquidity_above(entry, sh, ei, pdh_t)
            if dol is None or not (2.0 <= (dol - entry) / risk <= 10.0): continue
        else:
            entry = fvg["bottom"] - slippage
            swhigh = ed.find_recent_swing_high(sh, ei, lookback=80)
            anchor = swhigh if swhigh is not None and swhigh > fvg["top"] else fvg["top"]
            sl = anchor + buf; risk = sl - entry; direction = "short"
            if risk <= 0: continue
            dol = ed.next_liquidity_below(entry, sl_arr, ei, pdl_t)
            if dol is None or not (2.0 <= (entry - dol) / risk <= 10.0): continue
        # gradient exit
        reason = "end"; xj = min(opp, N - 1); px = c[xj]; pr = 0
        for j in range(ei + 1, min(opp, ei + 2000, N - 1) + 1):
            if direction == "long":
                if l[j] <= sl: reason, xj, px, pr = "stop", j, sl, (sl - entry) / risk; break
            else:
                if h[j] + spread >= sl: reason, xj, px, pr = "stop", j, sl, (entry - sl) / risk; break
            grad = ef[j] - ef[j - 1]
            inp = (c[j] > entry) if direction == "long" else (c[j] + spread < entry)
            if inp and ((direction == "long" and grad < 0) or (direction == "short" and grad > 0)):
                pxx = c[j] if direction == "long" else c[j] + spread
                reason, xj, px, pr = "gradient", j, pxx, ((pxx - entry) if direction == "long" else (entry - pxx)) / risk; break
            if j == opp:
                pxx = c[j] if direction == "long" else c[j] + spread
                reason, xj, px, pr = "cross", j, pxx, ((pxx - entry) if direction == "long" else (entry - pxx)) / risk; break
        open_until.append(idx[xj])
        w0 = max(0, i - 12); w1 = min(N - 1, xj + 8)
        bars = [[round(float(o[k]), dig), round(float(h[k]), dig), round(float(l[k]), dig),
                 round(float(c[k]), dig), round(float(ef[k]), dig), round(float(es[k]), dig),
                 round(float(rsi_arr[k]), 1) if not np.isnan(rsi_arr[k]) else 50.0]
                for k in range(w0, w1 + 1)]
        out.append(dict(symbol=symbol, dir=direction, rr=round(float(pr), 2), reason=reason,
                        outcome=("win" if pr > 0.01 else ("loss" if pr < -0.01 else "be")),
                        rsi=int(round(float(rv))), entry=round(entry, dig), sl=round(sl, dig),
                        exit=round(float(px), dig), entry_time=idx[ei], hold=round((idx[xj] - idx[ei]).total_seconds() / 3600, 1),
                        ft=round(fvg["top"], dig), fb=round(fvg["bottom"], dig), dig=dig,
                        ci=i - w0, fi=fvg["form_idx"] - w0, ei=ei - w0, xi=xj - w0, bars=bars,
                        times=[idx[k] for k in range(w0, w1 + 1)]))
    return out


def plot_trade(t, path):
    b = np.array([r[:6] for r in t["bars"]], float); rsi = np.array([r[6] for r in t["bars"]], float)
    n = len(b); win = t["outcome"] == "win"; islong = t["dir"] == "long"
    BG, FG = "#0d1117", "#e6edf3"; UP, DN = "#26a69a", "#ef5350"
    GRN, RED, YLW, PURP, BLU = "#3fb950", "#f85149", "#f2cc60", "#bc8cff", "#58a6ff"
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(15, 9), facecolor=BG,
                                  gridspec_kw={"height_ratios": [3.2, 1]}, sharex=True)
    ax.set_facecolor(BG); axr.set_facecolor(BG)
    for k in range(n):
        col = UP if b[k, 3] >= b[k, 0] else DN
        ax.plot([k, k], [b[k, 2], b[k, 1]], color=col, lw=0.8, zorder=2)
        y0 = min(b[k, 0], b[k, 3]); hh = abs(b[k, 3] - b[k, 0]) or (b[k, 1] - b[k, 2]) * .02
        ax.add_patch(plt.Rectangle((k - .32, y0), .64, hh, facecolor=col, edgecolor=col, zorder=3))
    ax.plot(range(n), b[:, 4], color=YLW, lw=1.3, label="EMA 7", zorder=4)
    ax.plot(range(n), b[:, 5], color=BLU, lw=1.3, label="EMA 18", zorder=4)
    ax.axvline(t["ci"], color=FG, ls=":", alpha=.5)
    ax.annotate("① EMA 7/18 cross", (t["ci"], b[t["ci"], 4]), (t["ci"] + 1, b[:, 1].max()),
                color=FG, fontsize=10, fontweight="bold", arrowprops=dict(arrowstyle="->", color=FG, alpha=.7))
    ax.add_patch(plt.Rectangle((t["fi"] - .5, t["fb"]), t["ei"] - t["fi"] + 1.5, t["ft"] - t["fb"],
                               facecolor=PURP, alpha=.22, edgecolor=PURP, zorder=1))
    ax.text(t["fi"], t["ft"], "② FVG", color=PURP, fontsize=9, va="bottom", fontweight="bold")
    ax.hlines(t["entry"], t["ei"], t["xi"], color=FG, lw=1.2)
    ax.hlines(t["sl"], t["ei"], t["xi"], color=RED, lw=1.2)
    ax.hlines(t["exit"], t["ei"], t["xi"], color=GRN if win else RED, lw=1.2)
    ax.add_patch(plt.Rectangle((t["ei"], min(t["entry"], t["sl"])), t["xi"] - t["ei"], abs(t["entry"] - t["sl"]),
                               facecolor=RED, alpha=.08, zorder=1))
    ax.add_patch(plt.Rectangle((t["ei"], min(t["entry"], t["exit"])), t["xi"] - t["ei"], abs(t["entry"] - t["exit"]),
                               facecolor=(GRN if win else RED), alpha=.10, zorder=1))
    ax.scatter([t["ei"]], [t["entry"]], marker="^" if islong else "v", s=150, color=YLW, edgecolor="k", zorder=7)
    exlabel = {"gradient": "③ gradient exit (EMA slope flipped)", "stop": "③ stop hit",
               "cross": "③ cross exit", "end": "③ exit"}[t["reason"]]
    ax.scatter([t["xi"]], [t["exit"]], marker="*" if win else "X", s=260, color=GRN if win else RED, edgecolor="k", zorder=8)
    ax.text(t["ei"] + .3, t["entry"], f"  entry {t['entry']}", color=FG, fontsize=9, va="bottom", fontweight="bold")
    ax.text(t["xi"] + .3, t["exit"], f"  {exlabel}  {t['exit']}  ({'+' if t['rr']>=0 else ''}{t['rr']}R)",
            color=GRN if win else RED, fontsize=9, va="center", fontweight="bold")
    ax.text(t["ei"] + .3, t["sl"], f"  SL {t['sl']}", color=RED, fontsize=9, va="top", fontweight="bold")
    res = f"WIN +{t['rr']}R" if win else f"LOSS {t['rr']}R"
    ax.set_title(f"EMA 7/18 + gradient exit · {t['symbol']} M15 · {t['dir'].upper()} · {res} · "
                 f"{t['entry_time'].strftime('%Y-%m-%d %H:%M')} · held {t['hold']}h",
                 color=GRN if win else RED, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Price", color=FG); ax.tick_params(colors=FG); ax.grid(True, alpha=.08, color=FG)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor=FG, loc="lower left", fontsize=10)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    axr.plot(range(n), rsi, color="#ff9f43", lw=1.2)
    axr.axhline(50, color=FG, ls="--", alpha=.4); axr.axhline(45 if islong else 55, color=GRN, alpha=.5)
    axr.scatter([t["ei"]], [rsi[t["ei"]]], color=GRN, s=80, edgecolor="k", zorder=6)
    axr.text(t["ei"] + .3, rsi[t["ei"]], f" ④ RSI {t['rsi']}", color=FG, fontsize=9, fontweight="bold")
    axr.set_ylim(15, 85); axr.set_ylabel("RSI(14)", color=FG); axr.tick_params(colors=FG); axr.grid(True, alpha=.08, color=FG)
    for sp in axr.spines.values(): sp.set_edgecolor("#30363d")
    step = max(1, n // 10); tk = list(range(0, n, step))
    axr.set_xticks(tk); axr.set_xticklabels([t["times"][x].strftime("%m-%d %H:%M") for x in tk],
                                            rotation=25, ha="right", fontsize=8, color=FG)
    plt.tight_layout(); plt.savefig(path, dpi=140, bbox_inches="tight", facecolor=BG); plt.close(fig)


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    allt = []
    for s in fq.UNIVERSE:
        allt += collect(s, load(s))
    allt.sort(key=lambda t: t["entry_time"])
    r = np.array([t["rr"] for t in allt])

    largest = max(allt, key=lambda t: t["rr"])
    # normal: a typical gradient-close winner (~median win), readable duration
    normw = sorted([t for t in allt if t["reason"] == "gradient" and 0.5 <= t["rr"] <= 1.3
                    and 8 <= (t["xi"] - t["ci"]) <= 70], key=lambda t: abs(t["rr"] - 0.8))
    normal = normw[len(normw) // 4] if normw else allt[0]
    # loss: a representative stop-out with readable context
    losses = sorted([t for t in allt if t["reason"] == "stop" and 10 <= (t["xi"] - t["ci"]) <= 90],
                    key=lambda t: t["entry_time"])
    loss = losses[len(losses) // 2] if losses else [t for t in allt if t["outcome"] == "loss"][0]

    for t, name in [(largest, "grad_largest_win.png"), (normal, "grad_normal_trade.png"), (loss, "grad_loss.png")]:
        plot_trade(t, os.path.join(OUT, name))
        print(f"  {name}: {t['symbol']} {t['dir']} {t['rr']:+.1f}R ({t['reason']}, held {t['hold']}h, {t['entry_time'].date()})")

    # ── trades + loss streaks overview ──
    bal = 100000.0; eq = [bal]
    for x in r: bal += x * RISK_USD; eq.append(bal)
    eq = np.array(eq)
    # loss streaks
    streaks = []; run = 0; run_start = 0
    for k, x in enumerate(r):
        if x < -0.01:
            if run == 0: run_start = k
            run += 1
        else:
            if run: streaks.append((run_start, run)); run = 0
    if run: streaks.append((run_start, run))
    from collections import Counter
    slen = Counter(s[1] for s in streaks)
    maxstreak = max(slen) if slen else 0

    BG, FG = "#0d1117", "#e6edf3"
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8), facecolor=BG, gridspec_kw={"height_ratios": [1.5, 1]})
    a1.set_facecolor(BG); a2.set_facecolor(BG)
    a1.plot(range(len(eq)), eq, color="#3fb950", lw=1.3, zorder=3)
    # highlight loss-streak spans
    for st, ln in streaks:
        a1.axvspan(st, st + ln, color="#f85149", alpha=.12, zorder=1)
    # mark the worst streak
    worst = max(streaks, key=lambda s: s[1]) if streaks else None
    if worst:
        a1.axvspan(worst[0], worst[0] + worst[1], color="#f85149", alpha=.30, zorder=2)
        a1.annotate(f"worst streak: {worst[1]} losses", (worst[0] + worst[1] / 2, eq[worst[0]]),
                    (worst[0], eq.max() * .8), color="#f85149", fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#f85149"))
    a1.set_title(f"All {len(r)} trades · equity ($1k risk / $100k) · loss-streak spans in red · "
                 f"max streak {maxstreak}", color=FG, fontsize=12, fontweight="bold")
    a1.set_ylabel("Balance ($)", color=FG); a1.set_xlabel("Trade #", color=FG)
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    a1.tick_params(colors=FG); a1.grid(True, alpha=.1, color=FG)
    for sp in a1.spines.values(): sp.set_edgecolor("#30363d")

    xs = sorted(slen)
    a2.bar([str(x) for x in xs], [slen[x] for x in xs], color="#f85149")
    for x in xs:
        a2.text(str(x), slen[x], str(slen[x]), ha="center", va="bottom", color=FG, fontsize=10)
    a2.set_title(f"Loss-streak length distribution · {len(streaks)} streaks total · "
                 f"{int((r<-0.01).sum())} losing trades of {len(r)}", color=FG, fontsize=12, fontweight="bold")
    a2.set_xlabel("consecutive losses", color=FG); a2.set_ylabel("# occurrences", color=FG)
    a2.tick_params(colors=FG); a2.grid(True, alpha=.1, color=FG, axis="y")
    for sp in a2.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "grad_trades_streaks.png"), dpi=140,
                                    bbox_inches="tight", facecolor=BG); plt.close(fig)
    print(f"  grad_trades_streaks.png: {len(r)} trades, {int((r<-0.01).sum())} losses, "
          f"{len(streaks)} streaks, max {maxstreak}")
    print(f"  streak counts: " + ", ".join(f"{k}x{v}" for k, v in sorted(slen.items())))
    mt5.shutdown()


if __name__ == "__main__":
    main()
