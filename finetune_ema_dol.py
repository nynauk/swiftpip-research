"""
Fine-tuning pass on the (fixed, most-recent-cross) EMA(9/21)+DOL + RSI strategy.

Sweeps each parameter ONE at a time around the base config, holding the rest
fixed — looking for a robust plateau, not an overfit peak. Includes an explicit
news-filter on/off test, a year-by-year robustness check, and a per-group table.
All P&L uses trade-by-trade bar simulation (SL checked before TP each bar).

Realistic per-instrument spread + slippage, 16 instruments, flat $1k / $100k.
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

SPEC, GROUPS, SYMBOLS = edr.SPEC, edr.GROUPS, edr.SYMBOLS
OUT = os.path.join("strategies", "finetune_ema_dol")

BASE = dict(rsi_band=(45, 100, 0, 55), fvg_window=24, retest_window=48,
            sl_buf=0.25, min_rr=2.0, max_rr=10.0, news_on=True)


def run_ft(symbol, pc, p, realistic=True):
    long_lo, long_hi, short_lo, short_hi = p["rsi_band"]
    spread_px, slip_px, min_w = SPEC[symbol]
    spread = spread_px if realistic else 0.0
    slippage = slip_px if realistic else 0.0
    h, l, c, idx = pc["h"], pc["l"], pc["c"], pc["idx"]
    cu, cd = pc["cu"], pc["cd"]
    nb, nbr = pc["next_bull"], pc["next_bear"]
    sh, sl_arr = pc["sh"], pc["sl_arr"]
    pdh_map, pdl_map, rsi_arr = pc["pdh"], pc["pdl"], pc["rsi"]
    news_set = pc["news"] if p["news_on"] else set()
    N = len(idx)
    FVGW, RTW, SLB = p["fvg_window"], p["retest_window"], p["sl_buf"]
    MINRR, MAXRR = p["min_rr"], p["max_rr"]

    trades = []; open_until = []
    for i in range(N):
        if not (cu[i] or cd[i]):
            continue
        bias = "long" if cu[i] else "short"
        tdir = "bullish" if bias == "long" else "bearish"
        opp = nbr[i + 1] if bias == "long" else nb[i + 1]         # next opposite cross
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
        if rr < MINRR or rr > MAXRR:
            continue
        oc, xt, pr = edr.sim_realistic(h, l, idx, ei, direction, entry, sl, tp, rr, spread)
        open_until.append(pd.Timestamp(xt))
        trades.append(dict(symbol=symbol, entry_time=idx[ei], direction=direction,
                           entry=entry, sl=sl, tp=tp, achieved_rr=round(rr, 2),
                           outcome=oc, exit_time=xt, pnl_r=pr))
    return pd.DataFrame(trades)


def combine(lst):
    return pd.concat(lst, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def evalp(pcs, p):
    return combine([run_ft(s, pcs[s], p) for s in SYMBOLS])


def line(label, st):
    if st is None:
        return f"  {label:26} no trades"
    return (f"  {label:26} {st['n']:>5} {st['wr']:>5.1f}% {st['pf']:>4.2f} "
            f"{st['tot']:>+7.1f}R {st['dd']:>6.1f}% {st['ls']:>4} ${st['bal']:>9,.0f}")


def sweep(pcs, name, key, values, base):
    print(f"\n  === sweep: {name} ===")
    print(f"  {'value':26} {'N':>5} {'WR':>6} {'PF':>5} {'R':>8} {'DD':>7} {'LS':>4} {'End$':>10}")
    for v in values:
        p = dict(base); p[key] = v
        st = edr.stats(evalp(pcs, p))
        tag = f"{key}={v}" + ("  <base>" if v == base[key] else "")
        print(line(tag, st))


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    os.makedirs(OUT, exist_ok=True)
    pcs = {}
    for s in SYMBOLS:
        df = bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
        pcs[s] = edr.precompute(s, df)

    print("=" * 92)
    print("FINE-TUNING · EMA(9/21)+DOL + RSI · fixed most-recent-cross · realistic · 16 instruments")
    print("trade-by-trade sim (SL checked before TP each bar) · flat $1,000 / $100k")
    print("=" * 92)
    print("  Base:", BASE)

    # 1) News filter on/off — the explicit test the user asked for
    sweep(pcs, "news filter", "news_on", [True, False], BASE)
    # 2) RSI momentum band
    sweep(pcs, "RSI band", "rsi_band",
          [(50, 100, 0, 50), (47, 100, 0, 53), (45, 100, 0, 55),
           (42, 100, 0, 58), (40, 100, 0, 60)], BASE)
    # 3) DOL floor (min RR)
    sweep(pcs, "min RR (DOL floor)", "min_rr", [2.0, 2.5, 3.0], BASE)
    # 4) SL buffer (fraction of FVG width beyond swing)
    sweep(pcs, "SL buffer frac", "sl_buf", [0.0, 0.25, 0.5], BASE)
    # 5) FVG lookup window
    sweep(pcs, "FVG window", "fvg_window", [16, 24, 32], BASE)
    # 6) Retest window
    sweep(pcs, "retest window", "retest_window", [36, 48, 60], BASE)

    # Year-by-year robustness of the base config
    base_df = evalp(pcs, BASE)
    print("\n  === year-by-year robustness (base config) ===")
    print(f"  {'Year':6} {'N':>5} {'WR':>6} {'PF':>5} {'R':>8} {'DD':>7} {'LS':>4}")
    base_df["yr"] = pd.to_datetime(base_df["entry_time"]).dt.year
    for yr in sorted(base_df["yr"].unique()):
        st = edr.stats(base_df[base_df["yr"] == yr])
        if st:
            print(f"  {yr:6} {st['n']:>5} {st['wr']:>5.1f}% {st['pf']:>4.2f} "
                  f"{st['tot']:>+7.1f}R {st['dd']:>6.1f}% {st['ls']:>4}")

    # Per-group
    print("\n  === per group (base config) ===")
    print(f"  {'Group':11} {'N':>5} {'WR':>6} {'PF':>5} {'R':>8} {'DD':>7} {'LS':>4}")
    for gname, syms in GROUPS:
        st = edr.stats(base_df[base_df["symbol"].isin(syms)])
        if st:
            print(f"  {gname:11} {st['n']:>5} {st['wr']:>5.1f}% {st['pf']:>4.2f} "
                  f"{st['tot']:>+7.1f}R {st['dd']:>6.1f}% {st['ls']:>4}")

    st = edr.stats(base_df)
    print("\n  BASE combined:", line("", st).strip())
    base_df.to_csv(os.path.join(OUT, "trades_base.csv"), index=False)

    # Equity of the tuned base
    BG, FG = "#0d1117", "#e6edf3"
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(st["times"], st["eqa"], color="#3fb950", linewidth=1.6, drawstyle="steps-post")
    ax.axhline(edr.START_BAL, color="#444c56", linestyle="--", linewidth=0.8)
    ax.set_title(f"Tuned EMA+DOL momentum (news ON, RSI 45/55, most-recent-cross) · "
                 f"PF {st['pf']:.2f} · DD {st['dd']:.1f}% · end ${st['bal']:,.0f}",
                 color=FG, fontsize=12, fontweight="bold")
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.set_xlabel("Date", color=FG); ax.set_ylabel("Balance ($)", color=FG)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "equity_tuned.png"), dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nSaved -> {OUT}/")
    mt5.shutdown()


if __name__ == "__main__":
    main()
