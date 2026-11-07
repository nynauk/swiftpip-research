"""
Full broker-realism analysis + trade visualisation.

1. Re-runs the champion gradient config (EMA 7/18, grad lb=1, profit-only) on:
     - M15  (the DEPLOYED bot's strategy)
     - M5   (the latest / higher-frequency strategy)
   both WITHOUT and WITH the new spread+1pip stop buffer, so we can read the
   exact cost of the extra realism.
2. Per-instrument + year-by-year breakdown (buffer ON = the honest numbers).
3. Plots several representative M5 trades (candles + EMA 7/18 + entry/SL/DOL +
   gradient exit) so the entry/exit logic can be eyeballed.
4. Before/after equity curves.

All realistic: per-instrument spread, LONG fills at ask / SHORT at bid, stop
widened by (spread + 1 pip), flat $1,000 risk on a $100k account.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import MetaTrader5 as mt5

import main as bt
import ema_dol_strategy as ed
import frequency_tune as fq
import test_cross_exit as tc
import test_gradient_exit as tg
import realistic_ema_dol_rsi as edr

EMA = (7, 18)
OUT = os.path.join("strategies", "spread_analysis")
BG, FG = "#0d1117", "#e6edf3"
GREEN, RED, BLUE, AMBER = "#3fb950", "#f85149", "#58a6ff", "#d29922"


# ─────────────────────────── data ───────────────────────────
def load_m15(s):
    p = os.path.join("data_cache", f"{s}_M15.pkl")
    try:
        return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
    except Exception:
        if os.path.exists(p): os.remove(p)
        return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")


def fetch_m5(symbol):
    mt5.symbol_select(symbol, True)
    bars = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 99999)
    if bars is None or not len(bars):
        return None
    df = pd.DataFrame(bars)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.set_index("time")[["open", "high", "low", "close"]]


# ────────────────── verbose trade collector (for plots) ──────────────────
def collect(symbol, df, base, ema, p, grad_lb=1, profit_only=True):
    """Mirror tg.run_gradient but record full geometry (entry_i, xj, entry, sl,
    dol) so individual trades can be drawn. Buffer state follows edr.SL_SPREAD_PAD."""
    long_lo, long_hi, short_lo, short_hi = p["rsi_band"]
    spread, slippage, min_w = fq.SPEC[symbol]
    h, l, c, idx = base["h"], base["l"], base["c"], base["idx"]
    sh, sl_arr = base["sh"], base["sl_arr"]
    pdh_map, pdl_map, rsi_arr = base["pdh"], base["pdl"], base["rsi"]
    news_set = base["news"] if p["news_on"] else set()
    ef = pd.Series(c, index=idx).ewm(span=ema[0], adjust=False).mean().values
    cu, cd, nb, nbr = fq.cross_arrays(c, idx, ema[0], ema[1])
    N = len(idx)
    FVGW, RTW, SLB = p["fvg_window"], p["retest_window"], p["sl_buf"]

    trades = []; open_until = []
    for i in range(N):
        if not (cu[i] or cd[i]):
            continue
        bias = "long" if cu[i] else "short"
        tdir = "bullish" if bias == "long" else "bearish"
        opp = nbr[i + 1] if bias == "long" else nb[i + 1]
        end_idx = min(i + FVGW, opp - 1, N - 1)
        if end_idx < i + 2:
            continue
        fvg = ed.find_aligned_fvg(h, l, i, end_idx, tdir, min_w)
        if fvg is None:
            continue
        rt_end = min(fvg["form_idx"] + RTW, opp - 1, N - 1)
        ei = None
        for k in range(fvg["form_idx"] + 1, rt_end + 1):
            if bias == "long" and l[k] <= fvg["top"]:    ei = k; break
            if bias == "short" and h[k] >= fvg["bottom"]: ei = k; break
        if ei is None:
            continue
        open_until = [t for t in open_until if t > idx[ei]]
        if len(open_until) >= 1:
            continue
        if idx[ei].date() in news_set:
            continue
        rv = rsi_arr[ei]
        if np.isnan(rv):
            continue
        if bias == "long" and not (long_lo <= rv <= long_hi):
            continue
        if bias == "short" and not (short_lo <= rv <= short_hi):
            continue
        d_today = pd.Timestamp(idx[ei]).normalize()
        pdh_t = pdh_map.get(d_today); pdl_t = pdl_map.get(d_today)
        buf = fvg["width"] * SLB
        pad = edr.sl_pad(symbol, spread)
        if bias == "long":
            entry = fvg["top"] + spread + slippage
            swlow = ed.find_recent_swing_low(sl_arr, ei, lookback=80)
            anchor = swlow if swlow is not None and swlow < fvg["bottom"] else fvg["bottom"]
            sl = anchor - buf - pad; risk = entry - sl; direction = "long"
            if risk <= 0: continue
            dol = ed.next_liquidity_above(entry, sh, ei, pdh_t)
            if dol is None or not (2.0 <= (dol - entry) / risk <= 10.0): continue
        else:
            entry = fvg["bottom"] - slippage
            swhigh = ed.find_recent_swing_high(sh, ei, lookback=80)
            anchor = swhigh if swhigh is not None and swhigh > fvg["top"] else fvg["top"]
            sl = anchor + buf + pad; risk = sl - entry; direction = "short"
            if risk <= 0: continue
            dol = ed.next_liquidity_below(entry, sl_arr, ei, pdl_t)
            if dol is None or not (2.0 <= (entry - dol) / risk <= 10.0): continue
        oc, xj, pr = tg.sim_gradient(h, l, c, ef, ei, direction, entry, sl, risk, opp,
                                     spread, grad_lb, profit_only)
        open_until.append(pd.Timestamp(idx[xj]))
        trades.append(dict(symbol=symbol, entry_i=ei, xj=xj, direction=direction,
                           entry=entry, sl=sl, dol=dol, risk=risk, outcome=oc, pnl_r=pr,
                           entry_time=idx[ei], exit_time=idx[xj],
                           hold_h=(idx[xj] - idx[ei]).total_seconds() / 3600))
    return trades


# ─────────────────────────── candlestick trade plot ───────────────────────────
def plot_trade(ax, df, ef, es, tr, pad_bars=26, tail=10):
    a = max(tr["entry_i"] - pad_bars, 0)
    b = min(tr["xj"] + tail, len(df) - 1)
    o = df["open"].values[a:b + 1]; hi = df["high"].values[a:b + 1]
    lo = df["low"].values[a:b + 1]; cl = df["close"].values[a:b + 1]
    x = np.arange(len(o))
    for k in range(len(o)):
        col = GREEN if cl[k] >= o[k] else RED
        ax.plot([k, k], [lo[k], hi[k]], color=col, linewidth=0.7, zorder=2)
        lo_body, hi_body = sorted((o[k], cl[k]))
        ax.add_patch(Rectangle((k - 0.3, lo_body), 0.6, max(hi_body - lo_body, 1e-9),
                               facecolor=col, edgecolor=col, linewidth=0.5, zorder=3))
    ax.plot(x, ef[a:b + 1], color=BLUE, linewidth=1.2, label=f"EMA{EMA[0]}", zorder=4)
    ax.plot(x, es[a:b + 1], color=AMBER, linewidth=1.2, label=f"EMA{EMA[1]}", zorder=4)
    ei_r = tr["entry_i"] - a; xj_r = tr["xj"] - a
    # entry / stop / target
    ax.axhline(tr["entry"], color=FG, linestyle="-", linewidth=0.8, alpha=0.7)
    ax.axhline(tr["sl"], color=RED, linestyle="--", linewidth=0.9, alpha=0.9)
    ax.axhline(tr["dol"], color=GREEN, linestyle=":", linewidth=0.9, alpha=0.9)
    up = tr["direction"] == "long"
    ax.scatter([ei_r], [tr["entry"]], marker="^" if up else "v", s=90,
               color=GREEN if up else RED, edgecolor=FG, zorder=6)
    exit_px = df["close"].values[tr["xj"]]
    ax.scatter([xj_r], [exit_px], marker="o", s=55, color=FG, edgecolor="#000", zorder=6)
    ax.axvspan(ei_r, xj_r, color=FG, alpha=0.04, zorder=1)
    res = "WIN" if tr["pnl_r"] > 0.01 else ("LOSS" if tr["pnl_r"] < -0.01 else "BE")
    rescol = GREEN if tr["pnl_r"] > 0.01 else (RED if tr["pnl_r"] < -0.01 else FG)
    ax.set_title(f"{tr['symbol']}  {tr['direction'].upper()}  {res} {tr['pnl_r']:+.2f}R  "
                 f"hold {tr['hold_h']:.1f}h  ({tr['entry_time']:%Y-%m-%d %H:%M})",
                 color=rescol, fontsize=9.5, fontweight="bold")
    ax.set_facecolor(BG); ax.tick_params(colors=FG, labelsize=7)
    ax.grid(True, alpha=0.08, color=FG)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=7, loc="upper left")
    ax.margins(x=0.01)


# ─────────────────────────── reporting ───────────────────────────
def summarize(df, tdays):
    st = edr.stats(df)
    cl = df[df["outcome"] != "open"]
    w = cl.loc[cl["pnl_r"] > 0.01, "pnl_r"]
    return dict(st=st, n=st["n"], perday=st["n"] / tdays, perwk=st["n"] / (tdays / 5),
                wr=st["wr"], pf=st["pf"], tot=st["tot"], dd=st["dd"], ls=st["ls"],
                bal=st["bal"], avgw=w.mean() if len(w) else 0.0,
                medw=w.median() if len(w) else 0.0, hold=cl["hold_h"].mean())


def run_config(bases, syms, ema, pad_on, tdays):
    edr.SL_SPREAD_PAD = pad_on
    df = tc.combine([tg.run_gradient(s, bases[s], ema, fq.BASEP, grad_lb=1, profit_only=True)
                     for s in syms])
    return df, summarize(df, tdays)


def line(tag, r):
    return (f"  {tag:16} {r['n']:>5} {r['perday']:>6.2f} {r['perwk']:>6.2f} {r['wr']:>6.1f}% "
            f"{r['pf']:>5.2f} {r['tot']:>+8.1f} {r['dd']:>7.1f}% {r['ls']:>4} "
            f"+{r['avgw']:>4.2f}/{r['medw']:>4.2f} ${r['bal']:>10,.0f}")


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    os.makedirs(OUT, exist_ok=True)
    syms = fq.UNIVERSE

    # ---- M15 (deployed bot) ----
    m15 = {s: load_m15(s) for s in syms}
    b15 = {s: fq.precompute_base(s, m15[s]) for s in syms}
    span15 = (max(v.index[-1] for v in m15.values()) - min(v.index[0] for v in m15.values())).days
    td15 = span15 * 5 / 7

    # ---- M5 (latest) ----
    m5 = {}
    for s in syms:
        d = fetch_m5(s)
        if d is not None and len(d) >= 1000: m5[s] = d
    b5 = {s: fq.precompute_base(s, m5[s]) for s in m5}
    span5 = (max(v.index[-1] for v in m5.values()) - min(v.index[0] for v in m5.values())).days
    td5 = span5 * 5 / 7

    hdr = (f"  {'config':16} {'N':>5} {'/day':>6} {'/wk':>6} {'WR':>7} {'PF':>5} "
           f"{'totR':>8} {'DD':>8} {'LS':>4} {'avgW/med':>10} {'End$':>11}")

    print("=" * 116)
    print("BROKER-REALISM ANALYSIS · champion gradient (EMA 7/18, lb1, profit-only) · "
          "flat $1k/$100k · 17 instruments")
    print("  buffer = widen every stop by (broker spread + 1 pip)")
    print("=" * 116)

    # M15 before/after
    print(f"\n  ── M15  (the DEPLOYED bot) · ~{td15:.0f} trading days ──")
    print(hdr)
    d15_off, r15_off = run_config(b15, syms, EMA, False, td15)
    d15_on,  r15_on  = run_config(b15, syms, EMA, True,  td15)
    print(line("no buffer", r15_off))
    print(line("spread+1pip", r15_on))

    # M5 before/after
    print(f"\n  ── M5  (the latest strategy) · {len(m5)} instruments · ~{td5:.0f} trading days ──")
    print(hdr)
    d5_off, r5_off = run_config(b5, list(m5), EMA, False, td5)
    d5_on,  r5_on  = run_config(b5, list(m5), EMA, True,  td5)
    print(line("no buffer", r5_off))
    print(line("spread+1pip", r5_on))

    # keep buffer ON for the rest (honest numbers)
    edr.SL_SPREAD_PAD = True

    # per-instrument (M5, buffer ON)
    print(f"\n  ── M5 per-instrument (buffer ON) ──")
    print(f"  {'sym':12} {'N':>4} {'WR':>6} {'PF':>6} {'totR':>8} {'DD':>7}")
    for s in syms:
        g = d5_on[d5_on["symbol"] == s]
        if len(g):
            st = edr.stats(g)
            print(f"  {s:12} {st['n']:>4} {st['wr']:>5.1f}% {st['pf']:>5.2f} "
                  f"{st['tot']:>+7.1f}R {st['dd']:>6.1f}%")

    # year-by-year (M15, buffer ON)
    print(f"\n  ── M15 year-by-year (buffer ON) ──")
    d15_on["yr"] = pd.to_datetime(d15_on["entry_time"]).dt.year
    for yr in sorted(d15_on["yr"].unique()):
        st = edr.stats(d15_on[d15_on["yr"] == yr])
        if st:
            print(f"    {yr}: N={st['n']:>4} PF {st['pf']:>4.2f} {st['tot']:>+6.1f}R "
                  f"DD {st['dd']:>5.1f}% LS{st['ls']}")

    d5_on.to_csv(os.path.join(OUT, "m5_trades_buffered.csv"), index=False)

    # ─────────────── sample M5 trades (buffer ON) ───────────────
    edr.SL_SPREAD_PAD = True
    plot_syms = ["US30.cash", "GER40.cash", "US100.cash", "UK100.cash", "US500.cash", "EURUSD"]
    plot_syms = [s for s in plot_syms if s in m5]
    alltr = []
    ef_cache = {}; es_cache = {}
    for s in plot_syms:
        c = m5[s]["close"]
        ef_cache[s] = c.ewm(span=EMA[0], adjust=False).mean().values
        es_cache[s] = c.ewm(span=EMA[1], adjust=False).mean().values
        for tr in collect(s, m5[s], b5[s], EMA, fq.BASEP, 1, True):
            alltr.append(tr)
    wins = sorted([t for t in alltr if t["pnl_r"] > 0.01], key=lambda t: t["pnl_r"], reverse=True)
    losses = sorted([t for t in alltr if t["pnl_r"] < -0.01], key=lambda t: t["pnl_r"])
    # pick: biggest win, a mid win, an early gradient-scratch win, a loss, another loss, a typical win
    picks = []
    if wins: picks.append(wins[0])                       # biggest win
    if len(wins) > 6: picks.append(wins[len(wins) // 4]) # upper-mid win
    if len(wins) > 2: picks.append(wins[len(wins) // 2]) # median-ish win
    if losses: picks.append(losses[0])                   # worst loss
    if len(losses) > 1: picks.append(losses[len(losses) // 2])
    if len(wins) > 3: picks.append(wins[-1])             # smallest win (gradient scratch)
    picks = picks[:6]

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), facecolor=BG)
    for ax, tr in zip(axes.flat, picks):
        plot_trade(ax, m5[tr["symbol"]], ef_cache[tr["symbol"]], es_cache[tr["symbol"]], tr)
    for ax in axes.flat[len(picks):]:
        ax.set_visible(False)
    fig.suptitle("Sample M5 gradient trades  ·  ^entry  ·  dashed red = stop  ·  "
                 "dotted green = DOL target  ·  o = gradient/cross exit",
                 color=FG, fontsize=12, fontweight="bold")
    fig.patch.set_facecolor(BG)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(OUT, "sample_m5_trades.png"), dpi=135, facecolor=BG)
    plt.close(fig)

    # ─────────────── before/after equity (M15 & M5) ───────────────
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
    for ax, title, doff, don in [(axA, "M15 (deployed bot)", d15_off, d15_on),
                                 (axB, "M5 (latest strategy)", d5_off, d5_on)]:
        for df, col, lab in [(doff, "#7d8590", "no buffer"), (don, GREEN, "spread+1pip")]:
            st = edr.stats(df)
            ax.plot(st["times"], st["eqa"], color=col, linewidth=1.5, drawstyle="steps-post",
                    label=f"{lab}: N={st['n']}, PF {st['pf']:.2f}, DD {st['dd']:.1f}%, ${st['bal']:,.0f}")
        ax.axhline(edr.START_BAL, color="#444c56", linestyle="--", linewidth=0.8)
        ax.set_title(title, color=FG, fontsize=12, fontweight="bold")
        ax.set_facecolor(BG); ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
        ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=8.5, loc="upper left")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "equity_before_after.png"), dpi=140, facecolor=BG)
    plt.close(fig)

    print(f"\nSaved -> {OUT}/  (sample_m5_trades.png, equity_before_after.png, m5_trades_buffered.csv)")
    mt5.shutdown()


if __name__ == "__main__":
    main()
