"""
Alternative exit: hold until the EMA crosses back over (trend-ride), instead of
the DOL target. Same entry (cross -> FVG retest -> RSI 45/55 momentum, SL at
swing+0.25xFVG). No target / no min-RR filter — the trade runs until either the
SL or the opposite EMA cross. Compared head-to-head with the DOL exit at 7/18.

Exit fill: long closes at bid (=close of cross bar), short at ask (close+spread).
Trade-by-trade bar sim; realistic per-instrument spread; flat $1,000 / $100k.
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
import frequency_tune as fq   # SPEC, UNIVERSE, precompute_base, cross_arrays, BASEP

EMA = (7, 18)
OUT = os.path.join("strategies", "ema718_final")


def sim_cross(h, l, c, ei, direction, entry, sl, risk, opp, spread, max_bars=2000):
    """SL first each bar; else exit at the opposite EMA cross (bar `opp`, close)."""
    end = min(opp, ei + max_bars, len(h) - 1)
    for j in range(ei + 1, end + 1):
        if direction == "long":
            if l[j] <= sl:
                return "loss", j, (sl - entry) / risk
        else:
            if h[j] + spread >= sl:
                return "loss", j, (entry - sl) / risk
        if j == opp:                                   # opposite cross confirmed
            if direction == "long":
                return _lbl((c[j] - entry) / risk), j, (c[j] - entry) / risk
            px = c[j] + spread                          # short exits at ask
            return _lbl((entry - px) / risk), j, (entry - px) / risk
    j = end
    if direction == "long":
        r = (c[j] - entry) / risk
    else:
        r = (entry - (c[j] + spread)) / risk
    return _lbl(r), j, r


def _lbl(r):
    return "win" if r > 0.01 else ("loss" if r < -0.01 else "be")


def run_cross(symbol, base, ema, p, require_dol=False):
    long_lo, long_hi, short_lo, short_hi = p["rsi_band"]
    spread, slippage, min_w = fq.SPEC[symbol]
    h, l, c, idx = base["h"], base["l"], base["c"], base["idx"]
    sh, sl_arr = base["sh"], base["sl_arr"]
    pdh_map, pdl_map = base["pdh"], base["pdl"]
    rsi_arr = base["rsi"]
    news_set = base["news"] if p["news_on"] else set()
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
        buf = fvg["width"] * SLB
        pad = edr.sl_pad(symbol, spread)               # spread+1pip stop buffer
        d_today = pd.Timestamp(idx[ei]).normalize()
        pdh_t = pdh_map.get(d_today); pdl_t = pdl_map.get(d_today)
        if bias == "long":
            entry = fvg["top"] + spread + slippage
            swlow = ed.find_recent_swing_low(sl_arr, ei, lookback=80)
            anchor = swlow if swlow is not None and swlow < fvg["bottom"] else fvg["bottom"]
            sl = anchor - buf - pad; risk = entry - sl; direction = "long"
            if risk <= 0:
                continue
            if require_dol:                            # keep the DOL>=2R selection filter
                dol = ed.next_liquidity_above(entry, sh, ei, pdh_t)
                if dol is None or not (2.0 <= (dol - entry) / risk <= 10.0):
                    continue
        else:
            entry = fvg["bottom"] - slippage
            swhigh = ed.find_recent_swing_high(sh, ei, lookback=80)
            anchor = swhigh if swhigh is not None and swhigh > fvg["top"] else fvg["top"]
            sl = anchor + buf + pad; risk = sl - entry; direction = "short"
            if risk <= 0:
                continue
            if require_dol:
                dol = ed.next_liquidity_below(entry, sl_arr, ei, pdl_t)
                if dol is None or not (2.0 <= (entry - dol) / risk <= 10.0):
                    continue
        oc, xj, pr = sim_cross(h, l, c, ei, direction, entry, sl, risk, opp, spread)
        open_until.append(pd.Timestamp(idx[xj]))
        trades.append(dict(symbol=symbol, entry_time=idx[ei], direction=direction,
                           outcome=oc, exit_time=idx[xj], pnl_r=pr,
                           hold_h=(idx[xj] - idx[ei]).total_seconds() / 3600))
    return pd.DataFrame(trades)


def combine(lst):
    return pd.concat(lst, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    os.makedirs(OUT, exist_ok=True)
    bases = {}
    for s in fq.UNIVERSE:
        df = bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
        bases[s] = fq.precompute_base(s, df)

    dol_df = combine([fq.run_freq(s, bases[s], EMA, fq.BASEP) for s in fq.UNIVERSE])
    x_all = combine([run_cross(s, bases[s], EMA, fq.BASEP, require_dol=False) for s in fq.UNIVERSE])
    x_sel = combine([run_cross(s, bases[s], EMA, fq.BASEP, require_dol=True) for s in fq.UNIVERSE])

    print("=" * 104)
    print(f"EXIT COMPARISON at EMA {EMA[0]}/{EMA[1]} · 17 instruments · realistic · flat $1k/$100k")
    print("=" * 104)
    print(f"  {'Exit / selection':30} {'N':>6} {'WR':>6} {'PF':>5} {'R':>8} {'avgR':>6} "
          f"{'avgHold':>7} {'maxWin':>7} {'DD':>8} {'LS':>4} {'End$':>10}")
    for label, df in [("DOL target · DOL>=2R filter", dol_df),
                      ("cross-back · DOL>=2R filter", x_sel),
                      ("cross-back · NO filter", x_all)]:
        st = edr.stats(df)
        cl = df[df["outcome"] != "open"] if len(df) else df
        avgr = cl["pnl_r"].mean(); mx = cl["pnl_r"].max()
        avgh = df["hold_h"].mean() if "hold_h" in df else float("nan")
        print(f"  {label:30} {st['n']:>6} {st['wr']:>5.1f}% {st['pf']:>4.2f} "
              f"{st['tot']:>+7.1f}R {avgr:>+5.2f}R {avgh:>5.0f}h {mx:>6.1f}R "
              f"{st['dd']:>7.1f}% {st['ls']:>4} ${st['bal']:>9,.0f}")
    x_df = x_sel   # year-by-year on the fair (same-selection) variant

    # year-by-year for cross-back on the same high-quality setups
    print(f"\n  EMA cross-back (same DOL>=2R setups) — year by year:")
    x_df["yr"] = pd.to_datetime(x_df["entry_time"]).dt.year
    for yr in sorted(x_df["yr"].unique()):
        st = edr.stats(x_df[x_df["yr"] == yr])
        if st:
            print(f"    {yr}: N={st['n']:>4} PF {st['pf']:>4.2f} {st['tot']:>+6.1f}R "
                  f"DD {st['dd']:>5.1f}% LS{st['ls']}")

    # equity comparison
    BG, FG = "#0d1117", "#e6edf3"
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG); ax.set_facecolor(BG)
    for label, df, col in [("DOL target", dol_df, "#3fb950"),
                           ("cross-back (same setups)", x_sel, "#58a6ff"),
                           ("cross-back (no filter)", x_all, "#f85149")]:
        st = edr.stats(df)
        ax.plot(st["times"], st["eqa"], color=col, linewidth=1.6, drawstyle="steps-post",
                label=f"{label}: N={st['n']}, PF {st['pf']:.2f}, DD {st['dd']:.1f}%, ${st['bal']:,.0f}")
    ax.axhline(edr.START_BAL, color="#444c56", linestyle="--", linewidth=0.8)
    ax.set_title(f"Exit test at EMA {EMA[0]}/{EMA[1]} · DOL target vs EMA cross-back · realistic",
                 color=FG, fontsize=12, fontweight="bold")
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
    ax.set_xlabel("Date", color=FG); ax.set_ylabel("Balance ($)", color=FG)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "exit_dol_vs_cross.png"), dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nSaved -> {OUT}/exit_dol_vs_cross.png")
    mt5.shutdown()


if __name__ == "__main__":
    main()
