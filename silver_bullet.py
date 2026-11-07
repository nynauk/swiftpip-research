"""
ICT Silver Bullet — proper mechanical implementation per
ictrading.com and ICT 2023 mentorship rules.

The flow (executed automatically):
  1. Pre-mark BSL (buyside liquidity) = Previous Day High
              SSL (sellside liquidity) = Previous Day Low
     (Primary liquidity references. Session H/L are secondary.)
  2. Wait for the Silver Bullet window. Three windows per day (NY local):
        London SB  03:00-04:00 NY  =  10:00-11:00 broker (GMT+3/EEST)
        NY AM  SB  10:00-11:00 NY  =  17:00-18:00 broker
        NY PM  SB  14:00-15:00 NY  =  21:00-22:00 broker
     NY AM is the strongest of the three.
  3. Inside the window, look for a SWEEP of either BSL or SSL:
        BSL swept (price wicked above PDH, closed back below)  →  SHORT bias
        SSL swept (price wicked below PDL, closed back above)  →  LONG  bias
  4. After the sweep, find the first 15M FVG in the OPPOSITE direction —
     this is the reversal play, not continuation:
        SHORT bias (BSL swept) → look for a BEARISH FVG forming after sweep
        LONG  bias (SSL swept) → look for a BULLISH FVG forming after sweep
     FVG must be at least MIN_FVG_WIDTH wide.
  5. Wait for price to RETRACE back into the FVG (NOT enter at formation).
        long  → trigger when 15m low touches FVG top (price falls into the gap)
        short → trigger when 15m high touches FVG bottom (price rallies into gap)
  6. Enter at the FVG edge.
        long  → entry = fvg.top   (limit on retrace down)
        short → entry = fvg.bottom (limit on retrace up)
  7. Stop loss beyond the opposite far side of the FVG + small buffer:
        long  → SL = fvg.bottom - SL_BUFFER * fvg.width
        short → SL = fvg.top    + SL_BUFFER * fvg.width
  8. Take profit at the user's minimum 2RR (or any RR you set). The article
     suggests 1:3 to next liquidity pool; we make this a parameter.
  9. News filter blocks red-folder days entirely.

Risk caps:
  • Max 1 concurrent position per symbol (a new sweep is ignored if a previous
    trade on the same symbol is still open).
"""
import os
import numpy as np
import pandas as pd

import main as bt
import news_filter as nf
import plot_utils as pu


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
SB_WINDOWS = {
    # broker hour ranges (MT5 server time, typically GMT+3 with DST)
    "London":  (10, 11),    # NY 03:00 - 04:00 ET — London SB
    "NYAM":    (17, 18),    # NY 10:00 - 11:00 ET — NYAM SB (best)
    "NYPM":    (21, 22),    # NY 14:00 - 15:00 ET — NYPM SB
}


MIN_W = {
    "EURUSD":     0.00020, "GBPUSD":     0.00020, "USDCAD":     0.00020,
    "USDCHF":     0.00020, "USDJPY":     0.020,   "AUDUSD":     0.00020,
    "NZDUSD":     0.00020,
    "XAUUSD":     0.20,    "GER40.cash": 2.0,     "US30.cash":  3.0,
    "US500.cash": 0.3,     "UK100.cash": 1.5,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_pdh_pdl(df_15m: pd.DataFrame):
    """Return two dicts keyed by date: {date: pdh}, {date: pdl}."""
    daily = df_15m.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    pdh = daily["high"].shift(1).dropna().to_dict()
    pdl = daily["low"].shift(1).dropna().to_dict()
    return pdh, pdl


def streak_and_dd(df, initial=10_000, risk_pct=1.0):
    """Return (final_balance, max_dd_pct, longest_loss_streak)."""
    bal = initial
    eq = [bal]
    cl = 0
    best_cl = 0
    for _, row in df.sort_values("entry_time").iterrows():
        bal += bal * (risk_pct / 100) * float(row["pnl_r"])
        eq.append(bal)
        if row["outcome"] == "loss":
            cl += 1
            best_cl = max(best_cl, cl)
        elif row["outcome"] == "win":
            cl = 0
    eq_arr = np.array(eq)
    dd = ((eq_arr - np.maximum.accumulate(eq_arr)) / np.maximum.accumulate(eq_arr) * 100).min()
    return bal, dd, best_cl


# ─────────────────────────────────────────────────────────────────────────────
# Core: detect sweep within window, then look for opposite-direction FVG +
# retest entry within the same trading day.
# ─────────────────────────────────────────────────────────────────────────────
def simulate(df_15m, entry_idx, direction, entry, sl, tp, rr, max_bars=300):
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


def run_silver_bullet(symbol: str, *,
                      windows: list = None,
                      rr: float = 2.0,
                      min_fvg_width: float = None,
                      sl_buffer: float = 0.25,
                      retrace_window_bars: int = 16,
                      post_sweep_fvg_window: int = 8,
                      news_block: bool = True,
                      max_open_per_symbol: int = 1,
                      tf: str = "M15") -> dict:
    """
    Returns {symbol, n, wr, total_r, df} for one symbol.
    Default windows = ["NYAM"] (the best per ICT).
    tf = "M15" or "M5" — execution timeframe (sweep + FVG + retest all on this TF).
    Note: 5M data is capped to ~1yr by MT5 broker (vs ~4yr for 15M).
    """
    import MetaTrader5 as mt5
    if min_fvg_width is None:
        min_fvg_width = MIN_W.get(symbol, 0.0002)
    if windows is None:
        windows = ["NYAM"]

    tf_const = mt5.TIMEFRAME_M5 if tf == "M5" else mt5.TIMEFRAME_M15
    df_15m = bt.fetch_data(symbol, tf_const, bt.START_DATE, bt.END_DATE, tf_name=tf)

    # Scale windows for M5 — 3x more bars per same time interval
    if tf == "M5":
        post_sweep_fvg_window *= 3
        retrace_window_bars *= 3

    h = df_15m["high"].values
    l = df_15m["low"].values
    c = df_15m["close"].values
    idx = df_15m.index
    hours = idx.hour
    dates = idx.normalize()

    pdh_map, pdl_map = build_pdh_pdl(df_15m)
    news_set = nf.build_news_block_set(idx[0].date(), idx[-1].date()) if news_block else set()

    trades = []
    open_until_per_sym = []  # list of exit timestamps for currently-open positions

    for win_name in windows:
        ws, we = SB_WINDOWS[win_name]

        # Iterate per-day: find first sweep within window, then setup
        unique_days, day_starts = np.unique(dates, return_index=True)
        day_ends = np.append(day_starts[1:], len(df_15m))

        for di_day, day in enumerate(unique_days):
            d_date = pd.Timestamp(day).date()
            if news_block and d_date in news_set:
                continue

            pdh = pdh_map.get(day)
            pdl = pdl_map.get(day)
            if pdh is None or pdl is None or pd.isna(pdh) or pd.isna(pdl):
                continue

            a, b = day_starts[di_day], day_ends[di_day]
            day_hours = hours[a:b]
            in_win = np.where((day_hours >= ws) & (day_hours < we))[0]
            if not len(in_win):
                continue

            # ── STEP 1: find first sweep of PDH (BSL) or PDL (SSL) in window ──
            sweep_dir = None  # "short" (bsl swept) or "long" (ssl swept)
            sweep_i = None    # global M15 index of the sweep candle
            for j in in_win:
                i = a + j
                if h[i] > pdh and c[i] < pdh:
                    sweep_dir = "short"   # BSL swept → look for SHORT (bearish reversal)
                    sweep_i = i
                    break
                if l[i] < pdl and c[i] > pdl:
                    sweep_dir = "long"    # SSL swept → look for LONG (bullish reversal)
                    sweep_i = i
                    break
            if sweep_dir is None:
                continue

            # Concurrent-position check
            open_until_per_sym = [t for t in open_until_per_sym if t > idx[sweep_i]]
            if len(open_until_per_sym) >= max_open_per_symbol:
                continue

            # ── STEP 2: find first OPPOSITE-direction FVG after sweep ──
            # FVG must form within post_sweep_fvg_window candles of the sweep
            target_fvg_dir = "bearish" if sweep_dir == "short" else "bullish"
            fvg_search_end = min(sweep_i + post_sweep_fvg_window, len(df_15m) - 3)
            fvg = None
            for k in range(sweep_i + 1, fvg_search_end + 1):
                # candle pattern uses k-2, k-1, k for 3-candle FVG completion
                if k < 2:
                    continue
                if target_fvg_dir == "bullish":
                    # bullish FVG: c[k].low > c[k-2].high (gap up between c[k-2] and c[k])
                    if l[k] > h[k - 2]:
                        fvg = {"direction": "bullish", "top": float(l[k]),
                               "bottom": float(h[k - 2]), "width": float(l[k] - h[k - 2]),
                               "form_idx": k, "time": idx[k]}
                        break
                else:
                    if h[k] < l[k - 2]:
                        fvg = {"direction": "bearish", "top": float(l[k - 2]),
                               "bottom": float(h[k]), "width": float(l[k - 2] - h[k]),
                               "form_idx": k, "time": idx[k]}
                        break
            if fvg is None or fvg["width"] < min_fvg_width:
                continue

            # ── STEP 3: wait for retest into the FVG ──
            retest_end = min(fvg["form_idx"] + retrace_window_bars, len(df_15m) - 1)
            entry_i = None
            for k in range(fvg["form_idx"] + 1, retest_end + 1):
                if sweep_dir == "long":
                    # long: price falls back into the bullish FVG → tap of fvg.top
                    if l[k] <= fvg["top"]:
                        entry_i = k
                        break
                else:
                    # short: price rallies back into the bearish FVG → tap of fvg.bottom
                    if h[k] >= fvg["bottom"]:
                        entry_i = k
                        break
            if entry_i is None:
                continue

            # ── STEP 4: place trade ──
            buf = fvg["width"] * sl_buffer
            if sweep_dir == "long":
                entry = fvg["top"]
                sl = fvg["bottom"] - buf
                risk = entry - sl
                if risk <= 0:
                    continue
                tp = entry + rr * risk
                direction = "long"
            else:
                entry = fvg["bottom"]
                sl = fvg["top"] + buf
                risk = sl - entry
                if risk <= 0:
                    continue
                tp = entry - rr * risk
                direction = "short"

            # News filter on entry day too
            if news_block and idx[entry_i].date() in news_set:
                continue

            result = simulate(df_15m, entry_i, direction, entry, sl, tp, rr=rr)
            open_until_per_sym.append(pd.Timestamp(result["exit_time"]))
            trades.append({
                "symbol":       symbol,
                "window":       win_name,
                "sweep_time":   idx[sweep_i],
                "swept_level":  float(pdh) if sweep_dir == "short" else float(pdl),
                "sweep_dir":    sweep_dir,
                "fvg_form_time": fvg["time"],
                "entry_time":   idx[entry_i],
                "entry_idx":    entry_i,
                "direction":    direction,
                "entry":        round(entry, 6),
                "sl":           round(sl, 6),
                "tp":           round(tp, 6),
                "fvg_top":      round(fvg["top"], 6),
                "fvg_bottom":   round(fvg["bottom"], 6),
                "fvg_width":    round(fvg["width"], 6),
                "outcome":      result["outcome"],
                "exit":         round(result["exit"], 6),
                "exit_time":    result["exit_time"],
                "pnl_r":        float(result["pnl_r"]),
            })

    df = pd.DataFrame(trades)
    if not len(df):
        return {"symbol": symbol, "n": 0, "wr": 0, "total_r": 0, "df": df}
    closed = df[df["outcome"] != "open"]
    wr = len(closed[closed["outcome"] == "win"]) / len(closed) * 100 if len(closed) else 0
    return {"symbol": symbol, "n": len(df), "wr": wr,
            "total_r": closed["pnl_r"].sum(), "df": df}


# ─────────────────────────────────────────────────────────────────────────────
# Main: run on 7 forex majors, full per-pair + combined stats
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]

    # Quick variant scan to find the best mechanical setup
    print("=" * 96)
    print("ICT SILVER BULLET — variant scan on 7 majors")
    print("=" * 96)
    variants_summary = []
    for label, kwargs in [
        ("M15 · NYAM · RR=2",      dict(windows=["NYAM"], rr=2.0, tf="M15")),
        ("M15 · NYAM · RR=3",      dict(windows=["NYAM"], rr=3.0, tf="M15")),
        ("M5  · NYAM · RR=2",      dict(windows=["NYAM"], rr=2.0, tf="M5")),
        ("M5  · NYAM · RR=3",      dict(windows=["NYAM"], rr=3.0, tf="M5")),
        ("M5  · All 3 · RR=2",     dict(windows=["London", "NYAM", "NYPM"], rr=2.0, tf="M5")),
    ]:
        combined_dfs = []
        for s in MAJORS:
            r = run_silver_bullet(s, **kwargs)
            if r["n"]:
                combined_dfs.append(r["df"])
        if not combined_dfs:
            continue
        all_df = pd.concat(combined_dfs, ignore_index=True).sort_values("entry_time")
        cl = all_df[all_df["outcome"] != "open"]
        wr = len(cl[cl["outcome"] == "win"]) / len(cl) * 100 if len(cl) else 0
        _, dd, streak = streak_and_dd(all_df)
        print(f"  {label:30}  N={len(all_df):4d}  WR={wr:5.1f}%  "
              f"R={cl['pnl_r'].sum():+7.1f}  DD={dd:5.1f}%  streak={streak}")
        variants_summary.append((label, len(all_df), wr, cl["pnl_r"].sum(), dd, streak))

    # Per-pair detail on M5 · NYAM · RR=2 (user's new ask: 5M timeframe)
    RR = 2.0
    TF = "M5"
    print("\n" + "=" * 96)
    print(f"PER-PAIR DETAIL · {TF} · NYAM window · RR={RR} · news-blocked")
    print("=" * 96)

    per_symbol = []
    for s in MAJORS:
        print(f"--- {s} ---")
        r = run_silver_bullet(s, windows=["NYAM"], rr=RR, tf=TF, news_block=True)
        if r["n"]:
            bal, dd, streak = streak_and_dd(r["df"])
            long_df = r["df"][r["df"]["direction"] == "long"]
            short_df = r["df"][r["df"]["direction"] == "short"]
            l_cl = long_df[long_df["outcome"] != "open"]
            s_cl = short_df[short_df["outcome"] != "open"]
            lwr = len(l_cl[l_cl["outcome"] == "win"]) / len(l_cl) * 100 if len(l_cl) else 0
            swr = len(s_cl[s_cl["outcome"] == "win"]) / len(s_cl) * 100 if len(s_cl) else 0
            per_symbol.append({
                "symbol": s, "n": r["n"], "wr": r["wr"], "r": r["total_r"],
                "dd": dd, "streak": streak,
                "long_n": len(long_df), "long_wr": lwr,
                "short_n": len(short_df), "short_wr": swr,
                "df": r["df"],
            })
            print(f"  {r['n']:4d} trades | WR {r['wr']:5.1f}% | R {r['total_r']:+7.1f} "
                  f"| DD {dd:.1f}% | streak {streak}")
        else:
            print("  no setups")

    # ── Per-pair table ──
    print("\n" + "=" * 100)
    print("PER-PAIR BREAKDOWN")
    print("=" * 100)
    print(f"{'Symbol':8}  {'N':>5}  {'WR':>6}  {'R':>8}  {'DD':>7}  {'Streak':>7}  "
          f"{'Long N':>6}  {'L-WR':>6}  {'Short N':>7}  {'S-WR':>6}")
    print("-" * 100)
    for p in per_symbol:
        print(f"{p['symbol']:8}  {p['n']:5d}  {p['wr']:5.1f}%  {p['r']:+7.1f}R  "
              f"{p['dd']:6.1f}%  {p['streak']:>7}  "
              f"{p['long_n']:6d}  {p['long_wr']:5.1f}%  {p['short_n']:7d}  {p['short_wr']:5.1f}%")

    # ── Combined ──
    if not per_symbol:
        print("\nNo trades at all.")
        return
    all_df = pd.concat([p["df"] for p in per_symbol], ignore_index=True)
    all_df = all_df.sort_values("entry_time").reset_index(drop=True)
    closed = all_df[all_df["outcome"] != "open"]
    wr = len(closed[closed["outcome"] == "win"]) / len(closed) * 100 if len(closed) else 0
    bal_c, dd_c, streak_c = streak_and_dd(all_df)
    print("-" * 100)
    print(f"{'TOTAL':8}  {len(all_df):5d}  {wr:5.1f}%  {closed['pnl_r'].sum():+7.1f}R  "
          f"{dd_c:6.1f}%  {streak_c:>7}")

    print(f"\nStarting balance       : $10,000")
    print(f"End balance (1% risk)  : ${bal_c:,.0f}")
    print(f"Net return             : {(bal_c/10000 - 1)*100:+.1f}%")
    print(f"Max drawdown           : {dd_c:.1f}%")
    print(f"Longest loss streak    : {streak_c} (across all majors mixed chronologically)")

    # Per-year
    all_df["year"] = pd.to_datetime(all_df["entry_time"]).dt.year
    print("\nPer-year (combined):")
    for yr in sorted(all_df["year"].unique()):
        sub = all_df[all_df["year"] == yr]
        cl = sub[sub["outcome"] != "open"]
        ywr = len(cl[cl["outcome"] == "win"]) / len(cl) * 100 if len(cl) else 0
        _, ydd, ystreak = streak_and_dd(sub)
        print(f"  {yr}: {len(sub):4d} trades, WR {ywr:.1f}%, R {cl['pnl_r'].sum():+.1f}, "
              f"DD {ydd:.1f}%, streak {ystreak}")

    # Save to folder
    OUT = os.path.join("strategies", "silver_bullet")
    os.makedirs(OUT, exist_ok=True)
    all_df.to_csv(os.path.join(OUT, "trades.csv"), index=False)
    pu.plot_equity(all_df, os.path.join(OUT, "equity.png"),
                   f"Silver Bullet (proper ICT) · NY AM · RR={RR} · 7 majors")
    print(f"\nSaved {OUT}/trades.csv")
    print(f"Saved {OUT}/equity.png")

    closed_sorted = all_df[all_df["outcome"] != "open"].sort_values("entry_time")
    if len(closed_sorted):
        recent = closed_sorted.iloc[-1]
        sym = recent["symbol"]
        tf_const = mt5.TIMEFRAME_M5 if TF == "M5" else mt5.TIMEFRAME_M15
        df_tf = bt.fetch_data(sym, tf_const, bt.START_DATE, bt.END_DATE, tf_name=TF)
        pu.plot_recent_trade(df_tf, recent.to_dict(),
                              os.path.join(OUT, "recent_trade.png"),
                              title_prefix=f"SB MOST RECENT ({TF})", symbol=sym)
        print(f"Saved {OUT}/recent_trade.png  ({sym}  {recent['entry_time']})")

    mt5.shutdown()


if __name__ == "__main__":
    main()
