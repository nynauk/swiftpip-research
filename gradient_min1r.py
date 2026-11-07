"""
Variant: gradient exit only ARMS at >= +1R open profit ("ignore anything below
1RR"). Below +1R the bot holds — SL and the opposite-cross fallback still apply.
Former sub-1R scratches must now either grow to >=1R, exit at the cross, or die
at -1R. Run on the SAME full 2000-2026 dual-feed dataset (HistData 2000-2024 +
MT5 2025-2026, buffered) for M3/M5/M15 + combined, vs the baseline.
"""
import os
import gc
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import frequency_tune as fq
import test_cross_exit as tc
import test_gradient_exit as tg
import realistic_ema_dol_rsi as edr
import histdata_fetch as hd
import ema_dol_strategy as ed
from full_2000_2026 import mt5_df, SYMS, YEARS_HD, CUT, HD_END, TFS

EMA = (7, 18)
MIN_ARM_R = 1.0
OUT = os.path.join("strategies", "full_2000_2026")
BG, FG = "#0d1117", "#e6edf3"


def sim_gradient_min(h, l, c, ef, ei, direction, entry, sl, risk, opp, spread,
                     grad_lb, min_arm_r, max_bars=2000):
    """tg.sim_gradient but the gradient exit only arms at >= min_arm_r open profit
    (measured on bar close, spread-adjusted for shorts) — matches the live manager's
    closed-candle rule."""
    end = min(opp, ei + max_bars, len(h) - 1)
    for j in range(ei + 1, end + 1):
        if direction == "long":
            if l[j] <= sl:
                return "loss", j, (sl - entry) / risk
        else:
            if h[j] + spread >= sl:
                return "loss", j, (entry - sl) / risk
        if j - grad_lb >= 0:
            grad = ef[j] - ef[j - grad_lb]
            openr = ((c[j] - entry) if direction == "long"
                     else (entry - (c[j] + spread))) / risk
            if openr >= min_arm_r and ((direction == "long" and grad < 0) or
                                       (direction == "short" and grad > 0)):
                px = c[j] if direction == "long" else c[j] + spread
                r = (px - entry) / risk if direction == "long" else (entry - px) / risk
                return ("win" if r > 0.01 else ("loss" if r < -0.01 else "be")), j, r
        if j == opp:
            px = c[j] if direction == "long" else c[j] + spread
            r = (px - entry) / risk if direction == "long" else (entry - px) / risk
            return ("win" if r > 0.01 else ("loss" if r < -0.01 else "be")), j, r
    j = end
    px = c[j] if direction == "long" else c[j] + spread
    r = (px - entry) / risk if direction == "long" else (entry - px) / risk
    return ("win" if r > 0.01 else ("loss" if r < -0.01 else "be")), j, r


def run_min1r(symbol, base, ema, p, grad_lb=1):
    """tg.run_gradient with the min-1R-armed gradient exit."""
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
        oc, xj, pr = sim_gradient_min(h, l, c, ef, ei, direction, entry, sl, risk, opp,
                                      spread, grad_lb, MIN_ARM_R)
        open_until.append(pd.Timestamp(idx[xj]))
        trades.append(dict(symbol=symbol, entry_time=idx[ei], direction=direction,
                           outcome=oc, exit_time=idx[xj], pnl_r=pr,
                           hold_h=(idx[xj] - idx[ei]).total_seconds() / 3600))
    return pd.DataFrame(trades)


def main():
    edr.SL_SPREAD_PAD = True
    print("=" * 116, flush=True)
    print(f"MIN-1R GRADIENT EXIT · 2000-2026 dual-feed · gradient only arms at >= +{MIN_ARM_R}R "
          "(else hold: SL / cross-fallback)", flush=True)
    print("=" * 116, flush=True)

    tf_parts = {tf: [] for tf, _, _ in TFS}
    for s in SYMS:                                  # HistData 2000-2024 (cached bars)
        for tf, rule, _ in TFS:
            try:
                bars = hd.fetch(s, YEARS_HD, tf=rule)
            except Exception:
                continue
            if len(bars) < 2000:
                continue
            base = fq.precompute_base(s, bars)
            t = run_min1r(s, base, EMA, fq.BASEP, 1)
            t = t[pd.to_datetime(t.entry_time) <= HD_END]
            if len(t):
                tf_parts[tf].append(t)
            del bars, base, t; gc.collect()
        print(f"    {s} HD done", flush=True)
    if mt5.initialize():                            # MT5 2025-2026 tail
        for s in SYMS:
            for tf, _, tfc in TFS:
                df = mt5_df(s, tfc)
                if df is None or len(df) < 2000:
                    continue
                base = fq.precompute_base(s, df)
                t = run_min1r(s, base, EMA, fq.BASEP, 1)
                t = t[pd.to_datetime(t.entry_time) >= CUT]
                if len(t):
                    tf_parts[tf].append(t)
                del df, base, t; gc.collect()
            print(f"    {s} MT5 done", flush=True)
        mt5.shutdown()

    d3 = tc.combine(tf_parts["M3"]); d5 = tc.combine(tf_parts["M5"]); d15 = tc.combine(tf_parts["M15"])
    comb = pd.concat([d3, d5, d15]).sort_values("entry_time").reset_index(drop=True)

    # baseline for comparison
    base_df = pd.read_csv(os.path.join(OUT, "full_trades.csv"))
    span0 = pd.to_datetime(comb.entry_time).min(); span1 = pd.to_datetime(comb.entry_time).max()
    days = (span1 - span0).days * 5 / 7

    print("\n" + "=" * 116, flush=True)
    print(f"  {'stream':16} {'N':>5} {'/day':>6} {'WR':>7} {'PF':>6} {'totR':>9} "
          f"{'DD':>7} {'maxLS':>6} {'avgW/med':>11} {'End$':>11}", flush=True)
    print("-" * 116, flush=True)
    for name, df in [("M3 min1R", d3), ("M5 min1R", d5), ("M15 min1R", d15),
                     ("COMBINED min1R", comb), ("COMBINED baseline", base_df)]:
        st = edr.stats(df)
        cl = df[df.outcome != "open"]; w = cl.loc[cl.pnl_r > 0.01, "pnl_r"]
        print(f"  {name:16} {st['n']:>5} {st['n']/days:>6.2f} {st['wr']:>6.1f}% {st['pf']:>6.2f} "
              f"{st['tot']:>+9.1f} {st['dd']:>6.1f}% {st['ls']:>6} "
              f"+{w.mean():>4.2f}/{w.median():>4.2f} ${st['bal']:>10,.0f}", flush=True)

    # FTMO sim comparison at 1%
    for name, df in [("min1R", comb), ("baseline", base_df)]:
        d = df.copy(); d["exit_time"] = pd.to_datetime(d["exit_time"])
        ps = edr.prop_sim(d, risk_usd=1000.0)
        print(f"\n  FTMO @1% ({name:8}): {ps['passes']} payouts · banked ${ps['banked']:,.0f} · "
              f"blown {ps['blown']} · NET ${ps['net']:,.0f}", flush=True)

    comb.to_csv(os.path.join(OUT, "min1r_trades.csv"), index=False)

    # equity comparison chart
    fig, ax = plt.subplots(figsize=(13.5, 6.5), facecolor=BG); ax.set_facecolor(BG)
    for name, df, col in [("baseline (arm >0R)", base_df, "#7d8590"),
                          (f"min-1R armed", comb, "#3fb950")]:
        st = edr.stats(df)
        ax.plot(st["times"], st["eqa"], color=col, linewidth=1.4, drawstyle="steps-post",
                label=f"{name}: N={st['n']}, WR {st['wr']:.0f}%, PF {st['pf']:.2f}, "
                      f"DD {st['dd']:.1f}%, LS {st['ls']}, ${st['bal']:,.0f}")
    ax.axhline(100000, color="#444c56", linestyle="--", linewidth=0.8)
    ax.set_title("Gradient exit armed at +1R vs baseline · M3+M5+M15 · 2000-2026",
                 color=FG, fontsize=12, fontweight="bold")
    ax.set_xlabel("Date", color=FG); ax.set_ylabel("Balance ($)", color=FG)
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "min1r_vs_baseline.png"), dpi=140, facecolor=BG)
    plt.close(fig)
    print(f"\nSaved → {OUT}/  (min1r_vs_baseline.png, min1r_trades.csv)", flush=True)


if __name__ == "__main__":
    main()
