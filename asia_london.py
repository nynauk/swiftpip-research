"""
ICT Asia → HTF PD-array → London Reversal strategy.

Sequence (all checked mechanically):
  1. Asian range = high/low of 00:00 - 07:00 broker time.
  2. After 07:00, the FIRST 15M candle that wicks past Asian H or L and
     CLOSES BACK INSIDE counts as the Asia sweep (manipulation phase).
  3. The sweep extreme (the wick high for Asia-high sweep, wick low for
     Asia-low sweep) MUST fall inside an HTF (4H) PD array of matching
     direction — i.e. into a 4H FVG that price is reversing from:
        Asia high swept (bearish bias) → sweep_extreme ∈ a bearish 4H FVG
        Asia low  swept (bullish bias) → sweep_extreme ∈ a bullish 4H FVG
     (FVG must have formed BEFORE the sweep.)
  4. After confluence is confirmed, watch the 15M chart during the LONDON
     SESSION (08:00 - 12:00 broker time) for a reversal IFVG:
        - long  bias → a bearish 15M FVG formed after the sweep
                        + a candle that closes ABOVE it (inverts it)
        - short bias → a bullish 15M FVG  + close BELOW (inverts)
  5. Entry: close of the inversion candle.
  6. Stop: opposite far side of the IFVG + buffer.
        long  → SL = fvg.bottom - SL_BUFFER * fvg.width
        short → SL = fvg.top    + SL_BUFFER * fvg.width
  7. Take profit: 2R (minimum 2RR per spec).
  8. News filter: skip the entire day if a red-folder event is scheduled.
"""
import os
import numpy as np
import pandas as pd

import main as bt
import news_filter as nf
import plot_utils as pu


ASIA_START   = 0   # broker hour: Asian range start
ASIA_END     = 7   # broker hour: Asian range end
LONDON_START = 8   # broker hour: London session start (entries only)
LONDON_END   = 12  # broker hour: London session end


MIN_W = {
    "EURUSD":     0.00020, "GBPUSD":     0.00020, "USDCAD":     0.00020,
    "USDCHF":     0.00020, "USDJPY":     0.020,   "XAUUSD":     0.20,
    "GER40.cash": 2.0,     "US30.cash":  3.0,     "US500.cash": 0.3,
    "UK100.cash": 1.5,
}


def htf_pd_confluence(sweep_extreme: float, bias: str, sweep_time, fvgs_4h, df_4h_index) -> bool:
    """
    True if the sweep_extreme falls inside an HTF (4H) FVG of matching
    direction that formed BEFORE the sweep.
    """
    for f in fvgs_4h:
        if f["direction"] == "bullish" and bias == "bullish":
            # Asia low swept: want a bullish 4H FVG containing the sweep low
            if f["bottom"] <= sweep_extreme <= f["top"]:
                form_time = df_4h_index[f["form_idx"]]
                if form_time <= sweep_time:
                    return True
        elif f["direction"] == "bearish" and bias == "bearish":
            # Asia high swept: want a bearish 4H FVG containing the sweep high
            if f["bottom"] <= sweep_extreme <= f["top"]:
                form_time = df_4h_index[f["form_idx"]]
                if form_time <= sweep_time:
                    return True
    return False


def simulate(df_15m, entry_idx, direction, entry, sl, tp, rr=2.0, max_bars=300):
    h = df_15m["high"].values
    l = df_15m["low"].values
    idx = df_15m.index
    for j in range(entry_idx + 1, min(entry_idx + max_bars, len(df_15m))):
        if direction == "long":
            if l[j] <= sl:  return {"outcome": "loss", "exit": sl, "exit_time": idx[j], "pnl_r": -1.0}
            if h[j] >= tp:  return {"outcome": "win",  "exit": tp, "exit_time": idx[j], "pnl_r":  rr}
        else:
            if h[j] >= sl:  return {"outcome": "loss", "exit": sl, "exit_time": idx[j], "pnl_r": -1.0}
            if l[j] <= tp:  return {"outcome": "win",  "exit": tp, "exit_time": idx[j], "pnl_r":  rr}
    return {"outcome": "open", "exit": df_15m["close"].iloc[-1], "exit_time": idx[-1], "pnl_r": 0.0}


def run_asia_london(symbol: str, *,
                    rr: float = 2.0,
                    min_fvg_width: float = None,
                    sl_buffer: float = 0.25,
                    max_disrespect_delay: int = 12,
                    news_block: bool = True,
                    require_htf_pd: bool = True) -> dict:
    import MetaTrader5 as mt5

    if min_fvg_width is None:
        min_fvg_width = MIN_W.get(symbol, 0.0002)

    df_4h  = bt.fetch_data(symbol, mt5.TIMEFRAME_H4,  bt.START_DATE, bt.END_DATE, tf_name="H4")
    df_15m = bt.fetch_data(symbol, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")

    sweeps  = bt.find_session_range_sweeps(df_15m, ASIA_START, ASIA_END, 23, "Asian")
    fvgs_15 = bt.find_all_fvgs(df_15m)
    fvgs_4h = bt.find_all_fvgs(df_4h)

    news_set = nf.build_news_block_set(df_15m.index[0].date(), df_15m.index[-1].date()) if news_block else set()

    h = df_15m["high"].values
    l = df_15m["low"].values
    c = df_15m["close"].values
    idx = df_15m.index
    hours = idx.hour

    fvgs_sorted  = sorted(fvgs_15, key=lambda f: f["form_idx"])
    form_indices = np.array([f["form_idx"] for f in fvgs_sorted])

    trades = []
    taken_today = set()  # (date, direction)

    for sweep in sweeps:
        bias = sweep["direction"]
        sweep_time = sweep["time"]
        d = pd.Timestamp(sweep_time).date()

        if (d, bias) in taken_today:
            continue
        if news_block and d in news_set:
            continue

        sweep_i = int(np.searchsorted(idx, sweep_time, side="left"))
        if sweep_i >= len(idx):
            continue

        # Compute the actual sweep wick extreme
        # (we look at the sweep candle's high for high-sweep, low for low-sweep)
        sweep_extreme = h[sweep_i] if bias == "bearish" else l[sweep_i]

        # HTF PD array confluence
        if require_htf_pd:
            if not htf_pd_confluence(sweep_extreme, bias, sweep_time, fvgs_4h, df_4h.index):
                continue

        # Reversal FVG must form post-sweep, of OPPOSITE direction (gets inverted)
        target_dir = "bearish" if bias == "bullish" else "bullish"
        rev_end = min(sweep_i + 24, len(idx) - 1)
        lo_pos = int(np.searchsorted(form_indices, sweep_i, side="left"))
        hi_pos = int(np.searchsorted(form_indices, rev_end, side="right"))

        for p in range(hi_pos - 1, lo_pos - 1, -1):
            fvg = fvgs_sorted[p]
            if fvg["direction"] != target_dir:
                continue
            if fvg["width"] < min_fvg_width:
                continue

            f_idx = fvg["form_idx"]
            dis_lo = f_idx + 1
            dis_hi = min(f_idx + max_disrespect_delay, len(idx) - 1)

            for di in range(dis_lo, dis_hi + 1):
                # entry must be in London session
                if not (LONDON_START <= hours[di] < LONDON_END):
                    continue
                if news_block and idx[di].date() in news_set:
                    continue

                # Inversion: close through FVG
                if fvg["direction"] == "bearish":
                    inverted = c[di] > fvg["top"]
                else:
                    inverted = c[di] < fvg["bottom"]
                if not inverted:
                    continue

                entry = c[di]
                buf = fvg["width"] * sl_buffer
                if bias == "bullish":
                    direction = "long"
                    sl = fvg["bottom"] - buf
                    risk = entry - sl
                    if risk <= 0:
                        break
                    tp = entry + rr * risk
                else:
                    direction = "short"
                    sl = fvg["top"] + buf
                    risk = sl - entry
                    if risk <= 0:
                        break
                    tp = entry - rr * risk

                result = simulate(df_15m, di, direction, entry, sl, tp, rr=rr)
                trades.append({
                    "symbol":      symbol,
                    "sweep_time":  sweep_time,
                    "sweep_level": float(sweep["swept_level"]),
                    "sweep_extreme": float(sweep_extreme),
                    "entry_time":  idx[di],
                    "entry_idx":   di,
                    "direction":   direction,
                    "entry":       round(entry, 6),
                    "sl":          round(sl, 6),
                    "tp":          round(tp, 6),
                    "fvg_top":     round(fvg["top"], 6),
                    "fvg_bottom":  round(fvg["bottom"], 6),
                    "fvg_width":   round(fvg["width"], 6),
                    "outcome":     result["outcome"],
                    "exit":        round(result["exit"], 6),
                    "exit_time":   result["exit_time"],
                    "pnl_r":       float(result["pnl_r"]),
                })
                taken_today.add((d, bias))
                break
            if (d, bias) in taken_today:
                break

    df = pd.DataFrame(trades)
    if not len(df):
        return {"symbol": symbol, "n": 0, "wr": 0, "total_r": 0, "df": df}
    closed = df[df["outcome"] != "open"]
    wr = len(closed[closed["outcome"] == "win"]) / len(closed) * 100 if len(closed) else 0
    return {"symbol": symbol, "n": len(df), "wr": wr,
            "total_r": closed["pnl_r"].sum(), "df": df}


def equity_dd_streak(df, initial=10_000, risk_pct=1.0):
    bal = initial
    eq = [bal]
    cl = 0
    best = 0
    for _, row in df.iterrows():
        bal += bal * (risk_pct / 100) * row["pnl_r"]
        eq.append(bal)
        if row["outcome"] == "loss":
            cl += 1
            best = max(best, cl)
        elif row["outcome"] == "win":
            cl = 0
    eq_arr = np.array(eq)
    dd = ((eq_arr - np.maximum.accumulate(eq_arr)) / np.maximum.accumulate(eq_arr) * 100).min()
    return bal, dd, best


def run_basket(symbols, **kwargs):
    results = []
    for s in symbols:
        r = run_asia_london(s, **kwargs)
        results.append(r)
        if r["n"]:
            print(f"  {s:14}: {r['n']:4d} | WR {r['wr']:5.1f}% | R {r['total_r']:+7.1f}")
    return results


def main():
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    SYMBOLS = ["EURUSD", "GBPUSD", "USDCAD", "US30.cash", "US500.cash", "UK100.cash"]

    print("\n=== Asia → HTF PD → London  ·  RR=2.0  ·  news-blocked ===")
    results = run_basket(SYMBOLS)

    all_df = pd.concat([r["df"] for r in results if len(r["df"])], ignore_index=True)
    if not len(all_df):
        print("No setups.")
        return
    all_df = all_df.sort_values("entry_time").reset_index(drop=True)
    closed = all_df[all_df["outcome"] != "open"]
    wr = len(closed[closed["outcome"] == "win"]) / len(closed) * 100
    bal, dd, streak = equity_dd_streak(all_df)
    print(f"\n  COMBINED       : {len(all_df):4d} | WR {wr:5.1f}% | "
          f"R {closed['pnl_r'].sum():+7.1f} | DD {dd:.1f}% | streak {streak}")
    print(f"  End balance (1% risk on $10k): ${bal:,.0f}")

    print("\n  Per-year:")
    all_df["year"] = pd.to_datetime(all_df["entry_time"]).dt.year
    for yr in sorted(all_df["year"].unique()):
        sub = all_df[all_df["year"] == yr]
        cl = sub[sub["outcome"] != "open"]
        if not len(cl):
            continue
        wr_y = len(cl[cl["outcome"] == "win"]) / len(cl) * 100
        print(f"    {yr}: {len(sub):4d} trades, WR {wr_y:.1f}%, R {cl['pnl_r'].sum():+.1f}")

    # Comparison: with vs without HTF PD confluence
    print("\n=== Without HTF PD confluence (sanity check) ===")
    results2 = run_basket(SYMBOLS, require_htf_pd=False)
    all2 = pd.concat([r["df"] for r in results2 if len(r["df"])], ignore_index=True)
    if len(all2):
        all2 = all2.sort_values("entry_time").reset_index(drop=True)
        cl = all2[all2["outcome"] != "open"]
        wr2 = len(cl[cl["outcome"] == "win"]) / len(cl) * 100
        bal2, dd2, streak2 = equity_dd_streak(all2)
        print(f"  COMBINED       : {len(all2):4d} | WR {wr2:5.1f}% | "
              f"R {cl['pnl_r'].sum():+7.1f} | DD {dd2:.1f}% | streak {streak2}")

    OUT = os.path.join("strategies", "asia_london")
    os.makedirs(OUT, exist_ok=True)
    all_df.to_csv(os.path.join(OUT, "trades.csv"), index=False)
    print(f"\nTrade log saved -> {OUT}/trades.csv")

    # Equity chart
    pu.plot_equity(all_df, os.path.join(OUT, "equity.png"),
                   f"Asia → HTF PD → London  ·  RR=2.0  ·  news-blocked")
    print(f"  equity chart -> {OUT}/equity.png")

    # Recent trade
    closed = all_df[all_df["outcome"] != "open"].sort_values("entry_time")
    if len(closed):
        recent = closed.iloc[-1]
        sym = recent["symbol"]
        df_15m = bt.fetch_data(sym, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
        pu.plot_recent_trade(df_15m, recent.to_dict(),
                              os.path.join(OUT, "recent_trade.png"),
                              title_prefix="ASIA→LONDON MOST RECENT", symbol=sym)
        print(f"  recent trade  -> {OUT}/recent_trade.png  ({sym}  {recent['entry_time']})")
    mt5.shutdown()


if __name__ == "__main__":
    main()
