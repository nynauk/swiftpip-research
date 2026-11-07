"""
Realistic-conditions test on the EMA(9)/EMA(21) + DOL strategy.

Same mechanical logic as ema_dol_strategy.py, but:
  • Entry filled at ask (long) / bid (short) with adverse slippage
  • Bar-by-bar walk models bid OHLC → SHORT SL/TP triggers offset by spread
  • Per-symbol spread in price units matches typical FTMO/ICMarkets retail
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MetaTrader5 as mt5

import main as bt
import news_filter as nf
import ema_dol_strategy as ed


SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]

SPREAD_PRICE = {
    "EURUSD": 1.0e-4, "GBPUSD": 1.5e-4, "USDJPY": 1.0e-2, "USDCHF": 1.5e-4,
    "USDCAD": 1.5e-4, "AUDUSD": 1.5e-4, "NZDUSD": 2.0e-4,
}
SLIPPAGE_PRICE = {k: 3e-5 if "JPY" not in k else 3e-3 for k in SPREAD_PRICE}


def sim_realistic(h, l, idx, ei, direction, real_entry, sl, tp, achieved_rr,
                  spread, max_bars=400):
    """Bid-side OHLC. LONG tests as-is. SHORT SL/TP shifted by spread (adverse)."""
    for j in range(ei + 1, min(ei + max_bars, len(h))):
        if direction == "long":
            if l[j] <= sl: return "loss", idx[j], -1.0
            if h[j] >= tp: return "win",  idx[j],  achieved_rr
        else:
            if h[j] + spread >= sl: return "loss", idx[j], -1.0
            if l[j] + spread <= tp: return "win",  idx[j],  achieved_rr
    return "open", idx[min(ei + max_bars - 1, len(h) - 1)], 0.0


def run_realistic(symbol, realistic=True, df=None, causal_swings=False):
    """Same logic as ed.run_ema_dol but with realistic fills + spread-adjusted triggers.

    causal_swings=True removes the backtest's swing-confirmation lookahead:
    a swing at bar j needs SWING_LOOKBACK bars after it to be confirmed, so at
    entry bar i only swings with j <= i - SWING_LOOKBACK are visible. This is
    what a live bot actually sees.
    """
    if df is None:
        df = bt.fetch_data(symbol, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
    lag = ed.SWING_LOOKBACK if causal_swings else 0
    h = df["high"].values; l = df["low"].values; c = df["close"].values
    idx = df.index

    ema_f = pd.Series(c, index=idx).ewm(span=ed.EMA_FAST, adjust=False).mean().values
    ema_s = pd.Series(c, index=idx).ewm(span=ed.EMA_SLOW, adjust=False).mean().values
    above = ema_f > ema_s
    cross_up   = np.zeros(len(df), dtype=bool)
    cross_down = np.zeros(len(df), dtype=bool)
    cross_up[1:]   = above[1:]  & ~above[:-1]
    cross_down[1:] = ~above[1:] &  above[:-1]

    sh, sl_arr = ed.find_swings_arr(h, l, n=ed.SWING_LOOKBACK)
    pdh_map, pdl_map = ed.build_pdh_pdl(df)
    news_set = nf.build_news_block_set(idx[0].date(), idx[-1].date())

    spread   = SPREAD_PRICE[symbol]   if realistic else 0.0
    slippage = SLIPPAGE_PRICE[symbol] if realistic else 0.0
    min_w    = ed.MIN_W.get(symbol, 2e-4)

    trades = []; open_until = []
    for i in range(len(df)):
        if not (cross_up[i] or cross_down[i]): continue
        bias = "long" if cross_up[i] else "short"
        target_dir = "bullish" if bias == "long" else "bearish"

        end_idx = min(i + ed.FVG_LOOKUP_WINDOW, len(df) - 1)
        fvg = ed.find_aligned_fvg(h, l, i, end_idx, target_dir, min_w)
        if fvg is None: continue

        retest_end = min(fvg["form_idx"] + ed.RETEST_WINDOW, len(df) - 1)
        entry_i = None
        for k in range(fvg["form_idx"] + 1, retest_end + 1):
            if bias == "long"  and l[k] <= fvg["top"]:    entry_i = k; break
            if bias == "short" and h[k] >= fvg["bottom"]: entry_i = k; break
        if entry_i is None: continue

        open_until = [t for t in open_until if t > idx[entry_i]]
        if len(open_until) >= 1: continue
        if idx[entry_i].date() in news_set: continue

        d_today = pd.Timestamp(idx[entry_i]).normalize()
        pdh_t = pdh_map.get(d_today); pdl_t = pdl_map.get(d_today)
        buf = fvg["width"] * ed.SL_BUFFER_FRAC

        if bias == "long":
            entry_signal = fvg["top"]
            # LONG fills at ask + slippage
            real_entry = entry_signal + spread + slippage
            swing_low = ed.find_recent_swing_low(sl_arr, entry_i - lag, lookback=80)
            anchor = swing_low if swing_low is not None and swing_low < fvg["bottom"] else fvg["bottom"]
            sl = anchor - buf
            risk = real_entry - sl
            if risk <= 0: continue
            dol = ed.next_liquidity_above(real_entry, sh, entry_i - lag + (1 if lag else 0), pdh_t)
            if dol is None: continue
            tp = dol
            achieved_rr = (tp - real_entry) / risk
            direction = "long"
        else:
            entry_signal = fvg["bottom"]
            # SHORT fills at bid, adverse slippage moves entry down
            real_entry = entry_signal - slippage
            swing_high = ed.find_recent_swing_high(sh, entry_i - lag, lookback=80)
            anchor = swing_high if swing_high is not None and swing_high > fvg["top"] else fvg["top"]
            sl = anchor + buf
            risk = sl - real_entry
            if risk <= 0: continue
            dol = ed.next_liquidity_below(real_entry, sl_arr, entry_i - lag + (1 if lag else 0), pdl_t)
            if dol is None: continue
            tp = dol
            achieved_rr = (real_entry - tp) / risk
            direction = "short"

        if achieved_rr < ed.MIN_RR_TO_TAKE or achieved_rr > ed.MAX_RR_CAP: continue

        oc, xt, pr = sim_realistic(h, l, idx, entry_i, direction, real_entry, sl, tp,
                                    achieved_rr, spread)
        open_until.append(pd.Timestamp(xt))
        trades.append({
            "symbol": symbol, "entry_time": idx[entry_i], "direction": direction,
            "entry": round(real_entry, 6), "sl": round(sl, 6), "tp": round(tp, 6),
            "achieved_rr": round(achieved_rr, 2),
            "outcome": oc, "exit_time": xt, "pnl_r": pr,
        })
    return pd.DataFrame(trades)


def stats(df):
    cl = df[df["outcome"] != "open"]
    if not len(cl): return None
    n = len(cl)
    wins = (cl["outcome"] == "win").sum()
    wr = wins / n * 100
    tot = cl["pnl_r"].sum()
    gw = cl.loc[cl["pnl_r"] > 0, "pnl_r"].sum()
    gl = -cl.loc[cl["pnl_r"] < 0, "pnl_r"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    avg_win = cl.loc[cl["pnl_r"] > 0, "pnl_r"].mean() if wins else 0
    bal = 10000.0; eq = [bal]; cl_ = cw = bcl = bcw = 0
    for _, r in cl.iterrows():
        bal += bal * 0.01 * float(r["pnl_r"]); eq.append(bal)
        if r["pnl_r"] < -0.01: cl_ += 1; cw = 0; bcl = max(bcl, cl_)
        elif r["pnl_r"] > 0.01: cw += 1; cl_ = 0; bcw = max(bcw, cw)
    eqa = np.array(eq)
    dd = ((eqa - np.maximum.accumulate(eqa)) / np.maximum.accumulate(eqa) * 100).min()
    return dict(n=n, wr=wr, pf=pf, tot=tot, dd=dd, ls=bcl, ws=bcw, bal=bal,
                avg_win=avg_win, eqa=eqa)


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    OUT = os.path.join("strategies", "realistic_ema_dol")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 108)
    print("EMA(9)/EMA(21) + DOL STRATEGY - realistic conditions test")
    print("Signal: EMA cross -> aligned FVG -> retest entry -> SL at ICT swing -> TP at DOL (>=2R)")
    print("Symbols: 7 forex majors, 4yr M15, news filter ON")
    print("=" * 108)

    all_naive = []
    all_real = []
    per_sym = []
    for s in SYMBOLS:
        df_n = run_realistic(s, realistic=False)
        df_r = run_realistic(s, realistic=True)
        all_naive.append(df_n)
        all_real.append(df_r)
        sn = stats(df_n); sr = stats(df_r)
        if sn and sr:
            per_sym.append((s, sn, sr))

    print(f"\n  Per-symbol naive vs realistic:")
    print(f"  {'Symbol':8} {'N_n':>4} {'WR_n':>5} {'PF_n':>4} {'R_n':>7}  "
          f"|  {'N_r':>4} {'WR_r':>5} {'PF_r':>4} {'R_r':>7} {'DD_r':>6}")
    for s, sn, sr in per_sym:
        print(f"  {s:8} {sn['n']:>4} {sn['wr']:>4.1f}% {sn['pf']:>4.2f} {sn['tot']:>+6.1f}R  "
              f"|  {sr['n']:>4} {sr['wr']:>4.1f}% {sr['pf']:>4.2f} {sr['tot']:>+6.1f}R {sr['dd']:>5.1f}%")

    combined_naive = pd.concat(all_naive, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    combined_real  = pd.concat(all_real,  ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    sn = stats(combined_naive); sr = stats(combined_real)

    print(f"\n  COMBINED (all 7 pairs interleaved chronologically):")
    print(f"  NAIVE     : N={sn['n']:>4}  WR={sn['wr']:.1f}%  PF={sn['pf']:.2f}  "
          f"R={sn['tot']:+.1f}  avg_win={sn['avg_win']:.2f}R  DD={sn['dd']:.1f}%  end ${sn['bal']:,.0f}")
    print(f"  REALISTIC : N={sr['n']:>4}  WR={sr['wr']:.1f}%  PF={sr['pf']:.2f}  "
          f"R={sr['tot']:+.1f}  avg_win={sr['avg_win']:.2f}R  DD={sr['dd']:.1f}%  end ${sr['bal']:,.0f}")

    combined_real.to_csv(os.path.join(OUT, "trades_realistic.csv"), index=False)

    # Equity plot
    BG, FG, GRN, RED = "#0d1117", "#e6edf3", "#3fb950", "#f85149"
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(sn["eqa"], color="#7d8590", linewidth=1.3, label=f"Naive (PF {sn['pf']:.2f})")
    ax.plot(sr["eqa"], color=GRN,       linewidth=1.5, label=f"Realistic (PF {sr['pf']:.2f})")
    ax.axhline(10000, color="#444c56", linestyle="--", linewidth=0.8)
    ax.set_title(f"EMA(9/21) + DOL | 7 majors | Naive vs Realistic (spread + slippage)",
                 color=FG, fontsize=12, fontweight="bold")
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG)
    ax.set_xlabel("Trade #", color=FG); ax.set_ylabel("Balance ($)", color=FG)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "equity.png"), dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    print(f"\n  Saved -> {OUT}/")
    mt5.shutdown()


if __name__ == "__main__":
    main()
