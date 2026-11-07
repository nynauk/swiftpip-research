"""
Gradient exit: close the trade the moment the fast EMA's slope turns against the
position (negative gradient for a long, positive for a short) — i.e. BEFORE the
full EMA cross-back. Idea: lock in the move near its peak instead of giving back
to the cross. Same entry as the cross-back config (EMA 7/18 -> FVG retest ->
RSI 45/55 -> DOL>=2R filter -> SL at swing). Compared head-to-head vs cross-back.

profit_only=True only arms the gradient exit once the trade is in profit (so a
shallow pullback right after entry can't scratch it).
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
import test_cross_exit as tc

EMA = (7, 18)
OUT = os.path.join("strategies", "ema718_final")


def sim_gradient(h, l, c, ef, ei, direction, entry, sl, risk, opp, spread,
                 grad_lb, profit_only, max_bars=2000):
    end = min(opp, ei + max_bars, len(h) - 1)
    for j in range(ei + 1, end + 1):
        # 1) stop first
        if direction == "long":
            if l[j] <= sl:
                return "loss", j, (sl - entry) / risk
        else:
            if h[j] + spread >= sl:
                return "loss", j, (entry - sl) / risk
        # 2) gradient exit (fast EMA slope turned against us)
        if j - grad_lb >= 0:
            grad = ef[j] - ef[j - grad_lb]
            inprofit = (c[j] > entry) if direction == "long" else (c[j] + spread < entry)
            arm = (not profit_only) or inprofit
            if arm and ((direction == "long" and grad < 0) or (direction == "short" and grad > 0)):
                px = c[j] if direction == "long" else c[j] + spread
                r = (px - entry) / risk if direction == "long" else (entry - px) / risk
                return ("win" if r > 0.01 else ("loss" if r < -0.01 else "be")), j, r
        # 3) fallback: the actual cross
        if j == opp:
            px = c[j] if direction == "long" else c[j] + spread
            r = (px - entry) / risk if direction == "long" else (entry - px) / risk
            return ("win" if r > 0.01 else ("loss" if r < -0.01 else "be")), j, r
    j = end
    px = c[j] if direction == "long" else c[j] + spread
    r = (px - entry) / risk if direction == "long" else (entry - px) / risk
    return ("win" if r > 0.01 else ("loss" if r < -0.01 else "be")), j, r


def run_gradient(symbol, base, ema, p, grad_lb=2, profit_only=True):
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
        pad = edr.sl_pad(symbol, spread)                  # spread+1pip stop buffer
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
        oc, xj, pr = sim_gradient(h, l, c, ef, ei, direction, entry, sl, risk, opp,
                                  spread, grad_lb, profit_only)
        open_until.append(pd.Timestamp(idx[xj]))
        trades.append(dict(symbol=symbol, entry_time=idx[ei], direction=direction,
                           outcome=oc, exit_time=idx[xj], pnl_r=pr,
                           hold_h=(idx[xj] - idx[ei]).total_seconds() / 3600))
    return pd.DataFrame(trades)


def row(label, df):
    st = edr.stats(df)
    cl = df[df["outcome"] != "open"] if len(df) else df
    avgr = cl["pnl_r"].mean(); mx = cl["pnl_r"].max()
    avgh = df["hold_h"].mean() if "hold_h" in df and len(df) else float("nan")
    return (f"  {label:34} {st['n']:>5} {st['wr']:>5.1f}% {st['pf']:>4.2f} "
            f"{st['tot']:>+7.1f}R {avgr:>+5.2f}R {avgh:>5.0f}h {mx:>6.1f}R "
            f"{st['dd']:>6.1f}% {st['ls']:>4} ${st['bal']:>9,.0f}"), st


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    def load(s):
        p = os.path.join("data_cache", f"{s}_M15.pkl")
        try:
            return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
        except Exception:
            if os.path.exists(p): os.remove(p)
            return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
    bases = {s: fq.precompute_base(s, load(s)) for s in fq.UNIVERSE}

    xback = tc.combine([tc.run_cross(s, bases[s], EMA, fq.BASEP, require_dol=True) for s in fq.UNIVERSE])

    print("=" * 108)
    print(f"EXIT: cross-back vs EMA-gradient · EMA {EMA[0]}/{EMA[1]} · 17 instruments · realistic · $1k/$100k")
    print("=" * 108)
    print(f"  {'Exit':34} {'N':>5} {'WR':>6} {'PF':>5} {'R':>8} {'avgR':>6} {'hold':>6} "
          f"{'maxWin':>7} {'DD':>7} {'LS':>4} {'End$':>10}")
    line, _ = row("cross-back (baseline)", xback); print(line)

    keep = {}
    for lb in [1, 2, 3]:
        for po in [True, False]:
            df = tc.combine([run_gradient(s, bases[s], EMA, fq.BASEP, grad_lb=lb, profit_only=po)
                             for s in fq.UNIVERSE])
            tag = f"gradient lb={lb} {'profit-only' if po else 'always'}"
            line, st = row(tag, df); print(line)
            keep[(lb, po)] = (df, st)

    # equity: baseline vs best gradient
    best = max(keep, key=lambda k: keep[k][1]["bal"])
    bdf, bst = keep[best]
    BG, FG = "#0d1117", "#e6edf3"
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG); ax.set_facecolor(BG)
    for label, df, col in [("cross-back", xback, "#58a6ff"),
                           (f"gradient lb={best[0]} {'profit-only' if best[1] else 'always'}", bdf, "#3fb950")]:
        st = edr.stats(df)
        ax.plot(st["times"], st["eqa"], color=col, linewidth=1.6, drawstyle="steps-post",
                label=f"{label}: N={st['n']}, PF {st['pf']:.2f}, DD {st['dd']:.1f}%, ${st['bal']:,.0f}")
    ax.axhline(edr.START_BAL, color="#444c56", linestyle="--", linewidth=0.8)
    ax.set_title(f"Cross-back vs EMA-gradient exit · EMA {EMA[0]}/{EMA[1]} · realistic",
                 color=FG, fontsize=12, fontweight="bold")
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
    ax.set_xlabel("Date", color=FG); ax.set_ylabel("Balance ($)", color=FG)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "exit_gradient_vs_cross.png"), dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\n  Best gradient variant: lb={best[0]} {'profit-only' if best[1] else 'always'} "
          f"-> ${bst['bal']:,.0f} (cross-back ${edr.stats(xback)['bal']:,.0f})")
    print(f"Saved -> {OUT}/exit_gradient_vs_cross.png")
    mt5.shutdown()


if __name__ == "__main__":
    main()
