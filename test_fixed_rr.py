"""
Fixed R:R exit vs the gradient / cross-back exits. Same entry (EMA 7/18 -> FVG
retest -> RSI 45/55 -> DOL>=2R filter -> SL at swing). Fixed R:R takes profit at
exactly rr*risk (win = +rr, loss = -1). Compared to gradient (lb=1 profit-only)
and cross-back on the 17-instrument universe, realistic, $1k/$100k.
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
import test_gradient_exit as tg

EMA = (7, 18)
OUT = os.path.join("strategies", "ema718_final")


def sim_fixed(h, l, ei, direction, entry, sl, tp, rr, spread, max_bars=2000):
    for j in range(ei + 1, min(ei + max_bars, len(h))):
        if direction == "long":
            if l[j] <= sl: return "loss", j, -1.0
            if h[j] >= tp: return "win", j, rr
        else:
            if h[j] + spread >= sl: return "loss", j, -1.0
            if l[j] + spread <= tp: return "win", j, rr
    return "open", min(ei + max_bars - 1, len(h) - 1), 0.0


def run_fixed(symbol, base, ema, p, rr):
    long_lo, long_hi, short_lo, short_hi = p["rsi_band"]
    spread, slippage, min_w = fq.SPEC[symbol]
    h, l, c, idx = base["h"], base["l"], base["c"], base["idx"]
    sh, sl_arr = base["sh"], base["sl_arr"]
    pdh_map, pdl_map, rsi_arr = base["pdh"], base["pdl"], base["rsi"]
    news_set = base["news"]
    cu, cd, nb, nbr = fq.cross_arrays(c, idx, ema[0], ema[1])
    N = len(idx); FVGW, RTW, SLB = p["fvg_window"], p["retest_window"], p["sl_buf"]
    trades = []; open_until = []
    for i in range(N):
        if not (cu[i] or cd[i]): continue
        bias = "long" if cu[i] else "short"
        tdir = "bullish" if bias == "long" else "bearish"
        opp = nbr[i + 1] if bias == "long" else nb[i + 1]
        end_idx = min(i + FVGW, opp - 1, N - 1)
        if end_idx < i + 2: continue
        fvg = ed.find_aligned_fvg(h, l, i, end_idx, tdir, min_w)
        if fvg is None: continue
        rt_end = min(fvg["form_idx"] + RTW, opp - 1, N - 1)
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
        if bias == "long" and not (long_lo <= rv <= long_hi): continue
        if bias == "short" and not (short_lo <= rv <= short_hi): continue
        d_today = pd.Timestamp(idx[ei]).normalize()
        pdh_t = pdh_map.get(d_today); pdl_t = pdl_map.get(d_today)
        buf = fvg["width"] * SLB
        if bias == "long":
            entry = fvg["top"] + spread + slippage
            swlow = ed.find_recent_swing_low(sl_arr, ei, lookback=80)
            anchor = swlow if swlow is not None and swlow < fvg["bottom"] else fvg["bottom"]
            sl = anchor - buf; risk = entry - sl; direction = "long"
            if risk <= 0: continue
            dol = ed.next_liquidity_above(entry, sh, ei, pdh_t)
            if dol is None or not (2.0 <= (dol - entry) / risk <= 10.0): continue
            tp = entry + rr * risk
        else:
            entry = fvg["bottom"] - slippage
            swhigh = ed.find_recent_swing_high(sh, ei, lookback=80)
            anchor = swhigh if swhigh is not None and swhigh > fvg["top"] else fvg["top"]
            sl = anchor + buf; risk = sl - entry; direction = "short"
            if risk <= 0: continue
            dol = ed.next_liquidity_below(entry, sl_arr, ei, pdl_t)
            if dol is None or not (2.0 <= (entry - dol) / risk <= 10.0): continue
            tp = entry - rr * risk
        oc, xj, pr = sim_fixed(h, l, ei, direction, entry, sl, tp, rr, spread)
        open_until.append(pd.Timestamp(idx[xj]))
        trades.append(dict(symbol=symbol, entry_time=idx[ei], direction=direction,
                           outcome=oc, exit_time=idx[xj], pnl_r=pr))
    return pd.DataFrame(trades)


def line(label, df):
    st = edr.stats(df)
    cl = df[df["outcome"] != "open"]
    w = cl.loc[cl.pnl_r > 0.01, "pnl_r"]
    mw = w.median() if len(w) else 0
    aw = w.mean() if len(w) else 0
    print(f"  {label:26} {st['n']:>5} {st['wr']:>5.1f}% {st['pf']:>5.2f} {st['tot']:>+8.1f}R "
          f"{aw:>+5.2f}R {mw:>+6.2f}R {st['dd']:>6.1f}% {st['ls']:>4} ${st['bal']:>9,.0f}")
    return st


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    def load(s):
        p = os.path.join("data_cache", f"{s}_M15.pkl")
        try: return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
        except Exception:
            if os.path.exists(p): os.remove(p)
            return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
    bases = {s: fq.precompute_base(s, load(s)) for s in fq.UNIVERSE}

    grad = tc.combine([tg.run_gradient(s, bases[s], EMA, fq.BASEP, 1, True) for s in fq.UNIVERSE])
    xb = tc.combine([tc.run_cross(s, bases[s], EMA, fq.BASEP, require_dol=True) for s in fq.UNIVERSE])

    print("=" * 104)
    print(f"EXIT COMPARISON · EMA {EMA[0]}/{EMA[1]} · 17 instruments · realistic · $1k/$100k")
    print("=" * 104)
    print(f"  {'Exit':26} {'N':>5} {'WR':>6} {'PF':>5} {'totR':>9} {'avgWin':>7} {'medWin':>7} "
          f"{'DD':>7} {'LS':>4} {'End$':>10}")
    line("gradient (current)", grad)
    line("cross-back", xb)
    results = {}
    for rr in [2.0, 2.5, 3.0]:
        df = tc.combine([run_fixed(s, bases[s], EMA, fq.BASEP, rr) for s in fq.UNIVERSE])
        results[rr] = df
        line(f"fixed {rr:.1f}R", df)

    # equity comparison
    BG, FG = "#0d1117", "#e6edf3"
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG); ax.set_facecolor(BG)
    series = [("gradient", grad, "#3fb950"), ("cross-back", xb, "#58a6ff"), ("fixed 2R", results[2.0], "#f2cc60")]
    for lbl, df, col in series:
        st = edr.stats(df)
        ax.plot(st["times"], st["eqa"], color=col, linewidth=1.6, drawstyle="steps-post",
                label=f"{lbl}: PF {st['pf']:.2f}, WR {st['wr']:.0f}%, ${st['bal']:,.0f}")
    ax.axhline(edr.START_BAL, color="#444c56", linestyle="--", linewidth=0.8)
    ax.set_title(f"Gradient vs cross-back vs fixed 2R exit · EMA {EMA[0]}/{EMA[1]} · realistic",
                 color=FG, fontsize=12, fontweight="bold")
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
    ax.set_xlabel("Date", color=FG); ax.set_ylabel("Balance ($)", color=FG)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "exit_fixed_vs_gradient.png"), dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nSaved -> {OUT}/exit_fixed_vs_gradient.png")
    mt5.shutdown()


if __name__ == "__main__":
    main()
