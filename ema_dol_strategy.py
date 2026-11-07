"""
EMA Crossover + ICT Draw-on-Liquidity (DOL) strategy.

Logic (all mechanical):
  1. Compute EMA(FAST) and EMA(SLOW) on M15.
  2. A CROSSOVER is the trade trigger:
        FAST crosses ABOVE SLOW  → LONG bias
        FAST crosses BELOW SLOW  → SHORT bias
  3. After the cross, find the FIRST aligned FVG within FVG_LOOKUP_WINDOW bars:
        LONG bias  → bullish FVG (continuation)
        SHORT bias → bearish FVG (continuation)
  4. Wait for price to RETEST the FVG (price tags the FVG zone).
  5. Entry at the FVG edge:
        long  → entry = fvg.top    (limit on pull-back down)
        short → entry = fvg.bottom (limit on pull-back up)
  6. STOP LOSS via ICT liquidity — nearest opposing swing in the lookback,
     with a small buffer:
        long  → SL = recent swing low  - SL_BUFFER * fvg.width
        short → SL = recent swing high + SL_BUFFER * fvg.width
  7. TAKE PROFIT = Draw on Liquidity — next opposing liquidity pool:
        long  → next swing high above entry (or PDH if closer)
        short → next swing low  below entry (or PDL if closer)
     Trade only if DOL is at least MIN_RR away (= 2R by default).
     Final TP = the actual DOL price (not a fixed RR).
  8. News filter blocks red-folder days entirely.
  9. Max 1 concurrent position per symbol.
"""
import os
import numpy as np
import pandas as pd

import main as bt
import news_filter as nf
import plot_utils as pu


# ─────────────────────────────────────────────────────────────────────────────
EMA_FAST = 9
EMA_SLOW = 21
SWING_LOOKBACK = 5                # candles each side for swing detection
FVG_LOOKUP_WINDOW = 24            # M15 bars after EMA cross to find an FVG
RETEST_WINDOW    = 48             # M15 bars after FVG to wait for retest
SL_BUFFER_FRAC   = 0.25           # SL buffer as fraction of FVG width
MIN_RR_TO_TAKE   = 2.0            # user mandate: minimum 2R to take the trade
MAX_RR_CAP       = 10.0           # cap at 10R (avoid unrealistic TP)

MIN_W = {
    "EURUSD":     0.00020, "GBPUSD":     0.00020, "USDCAD":     0.00020,
    "USDCHF":     0.00020, "USDJPY":     0.020,   "AUDUSD":     0.00020,
    "NZDUSD":     0.00020,
}


def streak_and_dd(df, initial=10_000, risk_pct=1.0):
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


def simulate(df, entry_idx, direction, entry, sl, tp, achieved_rr, max_bars=400):
    """Walk forward, return outcome. pnl_r computed from actual TP distance."""
    h = df["high"].values
    l = df["low"].values
    idx = df.index
    for j in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
        if direction == "long":
            if l[j] <= sl:  return {"outcome": "loss", "exit": sl, "exit_time": idx[j], "pnl_r": -1.0}
            if h[j] >= tp:  return {"outcome": "win",  "exit": tp, "exit_time": idx[j], "pnl_r":  achieved_rr}
        else:
            if h[j] >= sl:  return {"outcome": "loss", "exit": sl, "exit_time": idx[j], "pnl_r": -1.0}
            if l[j] <= tp:  return {"outcome": "win",  "exit": tp, "exit_time": idx[j], "pnl_r":  achieved_rr}
    return {"outcome": "open", "exit": df["close"].iloc[-1], "exit_time": idx[-1], "pnl_r": 0.0}


def build_pdh_pdl(df_15m):
    daily = df_15m.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    return daily["high"].shift(1).to_dict(), daily["low"].shift(1).to_dict()


def find_swings_arr(h, l, n=5):
    """Return (swing_highs, swing_lows) as arrays of NaN/price."""
    sh = np.full(len(h), np.nan)
    sl = np.full(len(l), np.nan)
    for i in range(n, len(h) - n):
        if h[i] >= h[i - n:i + n + 1].max():
            sh[i] = h[i]
        if l[i] <= l[i - n:i + n + 1].min():
            sl[i] = l[i]
    return sh, sl


def find_aligned_fvg(h, l, start_idx, end_idx, direction, min_width):
    """Find first FVG of given direction in window. Return dict or None."""
    for k in range(start_idx + 2, end_idx + 1):
        if direction == "bullish":
            if l[k] > h[k - 2]:
                width = float(l[k] - h[k - 2])
                if width >= min_width:
                    return {"direction": "bullish", "top": float(l[k]),
                            "bottom": float(h[k - 2]), "width": width, "form_idx": k}
        else:
            if h[k] < l[k - 2]:
                width = float(l[k - 2] - h[k])
                if width >= min_width:
                    return {"direction": "bearish", "top": float(l[k - 2]),
                            "bottom": float(h[k]), "width": width, "form_idx": k}
    return None


def next_liquidity_above(price: float, swing_highs: np.ndarray, until_idx: int, pdh_today: float):
    """Find next swing high above price within [0, until_idx], or PDH if higher and still above price."""
    # Use most recent swing highs (last N) above price
    candidates = []
    for i in range(until_idx - 1, max(-1, until_idx - 200), -1):
        if not np.isnan(swing_highs[i]) and swing_highs[i] > price:
            candidates.append(swing_highs[i])
    target = min(candidates) if candidates else None
    if pdh_today is not None and not pd.isna(pdh_today) and pdh_today > price:
        if target is None or pdh_today < target:
            target = pdh_today
    return target


def next_liquidity_below(price: float, swing_lows: np.ndarray, until_idx: int, pdl_today: float):
    candidates = []
    for i in range(until_idx - 1, max(-1, until_idx - 200), -1):
        if not np.isnan(swing_lows[i]) and swing_lows[i] < price:
            candidates.append(swing_lows[i])
    target = max(candidates) if candidates else None
    if pdl_today is not None and not pd.isna(pdl_today) and pdl_today < price:
        if target is None or pdl_today > target:
            target = pdl_today
    return target


def find_recent_swing_low(swing_lows: np.ndarray, until_idx: int, lookback: int = 50):
    """Most recent swing low value at or before until_idx, within lookback bars."""
    start = max(0, until_idx - lookback)
    for i in range(until_idx, start - 1, -1):
        if not np.isnan(swing_lows[i]):
            return float(swing_lows[i])
    return None


def find_recent_swing_high(swing_highs: np.ndarray, until_idx: int, lookback: int = 50):
    start = max(0, until_idx - lookback)
    for i in range(until_idx, start - 1, -1):
        if not np.isnan(swing_highs[i]):
            return float(swing_highs[i])
    return None


def run_ema_dol(symbol: str, *,
                fast: int = EMA_FAST, slow: int = EMA_SLOW,
                min_rr: float = MIN_RR_TO_TAKE,
                min_fvg_width: float = None,
                news_block: bool = True,
                max_open_per_symbol: int = 1,
                tp_mode: str = "DOL") -> dict:
    """
    tp_mode:
      "DOL"     — TP at next opposing liquidity (variable RR, ≥ min_rr)
      "FIXED"   — TP at fixed min_rr * risk
    """
    import MetaTrader5 as mt5
    if min_fvg_width is None:
        min_fvg_width = MIN_W.get(symbol, 0.0002)

    df = bt.fetch_data(symbol, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")

    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    idx = df.index

    # EMAs and crosses
    ema_f = pd.Series(c, index=idx).ewm(span=fast, adjust=False).mean().values
    ema_s = pd.Series(c, index=idx).ewm(span=slow, adjust=False).mean().values
    above = ema_f > ema_s
    cross_up   = np.zeros(len(df), dtype=bool)
    cross_down = np.zeros(len(df), dtype=bool)
    cross_up[1:]   = above[1:]  & ~above[:-1]
    cross_down[1:] = ~above[1:] &  above[:-1]

    # Swings + PDH/PDL
    sh, sl_arr = find_swings_arr(h, l, n=SWING_LOOKBACK)
    pdh_map, pdl_map = build_pdh_pdl(df)
    dates = idx.normalize()
    news_set = nf.build_news_block_set(idx[0].date(), idx[-1].date()) if news_block else set()

    trades = []
    open_until = []

    for i in range(len(df)):
        if not (cross_up[i] or cross_down[i]):
            continue
        bias = "long" if cross_up[i] else "short"
        target_dir = "bullish" if bias == "long" else "bearish"

        # Find FVG after the cross
        end_idx = min(i + FVG_LOOKUP_WINDOW, len(df) - 1)
        fvg = find_aligned_fvg(h, l, i, end_idx, target_dir, min_fvg_width)
        if fvg is None:
            continue

        # Wait for retest of FVG
        retest_end = min(fvg["form_idx"] + RETEST_WINDOW, len(df) - 1)
        entry_i = None
        for k in range(fvg["form_idx"] + 1, retest_end + 1):
            if bias == "long":
                if l[k] <= fvg["top"]:
                    entry_i = k
                    break
            else:
                if h[k] >= fvg["bottom"]:
                    entry_i = k
                    break
        if entry_i is None:
            continue

        # Concurrent-position cap
        open_until = [t for t in open_until if t > idx[entry_i]]
        if len(open_until) >= max_open_per_symbol:
            continue

        # News block on entry day
        if news_block and idx[entry_i].date() in news_set:
            continue

        # Entry + SL via ICT swing
        d_today = pd.Timestamp(idx[entry_i]).normalize()
        pdh_t = pdh_map.get(d_today)
        pdl_t = pdl_map.get(d_today)

        buf = fvg["width"] * SL_BUFFER_FRAC
        if bias == "long":
            entry = fvg["top"]
            # SL at recent swing low (or fvg.bottom if no swing nearby, whichever lower)
            swing_low = find_recent_swing_low(sl_arr, entry_i, lookback=80)
            anchor = swing_low if swing_low is not None and swing_low < fvg["bottom"] else fvg["bottom"]
            sl = anchor - buf
            risk = entry - sl
            if risk <= 0:
                continue
            # DOL TP = next swing high above entry (or PDH if closer)
            if tp_mode == "DOL":
                dol = next_liquidity_above(entry, sh, entry_i, pdh_t)
                if dol is None:
                    continue
                tp = dol
                achieved_rr = (tp - entry) / risk
            else:
                achieved_rr = min_rr
                tp = entry + min_rr * risk
            if achieved_rr < min_rr or achieved_rr > MAX_RR_CAP:
                continue
            direction = "long"
        else:
            entry = fvg["bottom"]
            swing_high = find_recent_swing_high(sh, entry_i, lookback=80)
            anchor = swing_high if swing_high is not None and swing_high > fvg["top"] else fvg["top"]
            sl = anchor + buf
            risk = sl - entry
            if risk <= 0:
                continue
            if tp_mode == "DOL":
                dol = next_liquidity_below(entry, sl_arr, entry_i, pdl_t)
                if dol is None:
                    continue
                tp = dol
                achieved_rr = (entry - tp) / risk
            else:
                achieved_rr = min_rr
                tp = entry - min_rr * risk
            if achieved_rr < min_rr or achieved_rr > MAX_RR_CAP:
                continue
            direction = "short"

        result = simulate(df, entry_i, direction, entry, sl, tp, achieved_rr=achieved_rr)
        open_until.append(pd.Timestamp(result["exit_time"]))
        trades.append({
            "symbol":      symbol,
            "cross_time":  idx[i],
            "cross_dir":   bias,
            "fvg_time":    idx[fvg["form_idx"]],
            "entry_time":  idx[entry_i],
            "entry_idx":   entry_i,
            "direction":   direction,
            "entry":       round(entry, 6),
            "sl":          round(sl, 6),
            "tp":          round(tp, 6),
            "achieved_rr": round(achieved_rr, 2),
            "fvg_top":     round(fvg["top"], 6),
            "fvg_bottom":  round(fvg["bottom"], 6),
            "fvg_width":   round(fvg["width"], 6),
            "outcome":     result["outcome"],
            "exit":        round(result["exit"], 6),
            "exit_time":   result["exit_time"],
            "pnl_r":       float(result["pnl_r"]),
        })

    df_t = pd.DataFrame(trades)
    if not len(df_t):
        return {"symbol": symbol, "n": 0, "wr": 0, "total_r": 0, "df": df_t}
    closed = df_t[df_t["outcome"] != "open"]
    wr = len(closed[closed["outcome"] == "win"]) / len(closed) * 100 if len(closed) else 0
    return {"symbol": symbol, "n": len(df_t), "wr": wr,
            "total_r": closed["pnl_r"].sum(), "df": df_t}


def main():
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]

    print("=" * 100)
    print("EMA CROSSOVER + ICT DOL STRATEGY — variant scan on 7 majors (M15)")
    print("=" * 100)
    for label, kwargs in [
        ("EMA9/21 · DOL TP (variable RR, ≥2)", dict(fast=9,  slow=21, tp_mode="DOL",   min_rr=2.0)),
        ("EMA9/21 · Fixed 2R TP",              dict(fast=9,  slow=21, tp_mode="FIXED", min_rr=2.0)),
        ("EMA9/21 · Fixed 3R TP",              dict(fast=9,  slow=21, tp_mode="FIXED", min_rr=3.0)),
        ("EMA12/26 · DOL TP",                  dict(fast=12, slow=26, tp_mode="DOL",   min_rr=2.0)),
        ("EMA20/50 · DOL TP",                  dict(fast=20, slow=50, tp_mode="DOL",   min_rr=2.0)),
        ("EMA50/200 · DOL TP",                 dict(fast=50, slow=200, tp_mode="DOL",  min_rr=2.0)),
    ]:
        dfs = []
        for s in MAJORS:
            r = run_ema_dol(s, news_block=True, **kwargs)
            if r["n"]:
                dfs.append(r["df"])
        if not dfs:
            print(f"  {label:38}  (no trades)")
            continue
        all_df = pd.concat(dfs, ignore_index=True).sort_values("entry_time")
        cl = all_df[all_df["outcome"] != "open"]
        wr = len(cl[cl["outcome"] == "win"]) / len(cl) * 100 if len(cl) else 0
        _, dd, streak = streak_and_dd(all_df)
        print(f"  {label:38}  N={len(all_df):5d}  WR={wr:5.1f}%  "
              f"R={cl['pnl_r'].sum():+7.1f}  DD={dd:6.1f}%  streak={streak}")

    # Per-pair on EMA9/21 + DOL (the user-spec config)
    print("\n" + "=" * 100)
    print("PER-PAIR DETAIL · EMA9/21 · DOL TP · 2R minimum · news-blocked")
    print("=" * 100)

    per_symbol = []
    for s in MAJORS:
        print(f"--- {s} ---")
        r = run_ema_dol(s, fast=9, slow=21, tp_mode="DOL", min_rr=2.0, news_block=True)
        if r["n"]:
            bal, dd, streak = streak_and_dd(r["df"])
            ln = r["df"][r["df"]["direction"] == "long"]
            sh = r["df"][r["df"]["direction"] == "short"]
            l_cl = ln[ln["outcome"] != "open"]
            s_cl = sh[sh["outcome"] != "open"]
            lwr = len(l_cl[l_cl["outcome"] == "win"]) / len(l_cl) * 100 if len(l_cl) else 0
            swr = len(s_cl[s_cl["outcome"] == "win"]) / len(s_cl) * 100 if len(s_cl) else 0
            avg_rr = r["df"]["achieved_rr"].mean()
            per_symbol.append({
                "symbol": s, "n": r["n"], "wr": r["wr"], "r": r["total_r"],
                "dd": dd, "streak": streak,
                "long_n": len(ln), "long_wr": lwr,
                "short_n": len(sh), "short_wr": swr,
                "avg_rr": avg_rr, "df": r["df"],
            })
            print(f"  {r['n']:4d} trades | WR {r['wr']:5.1f}% | R {r['total_r']:+7.1f} "
                  f"| DD {dd:.1f}% | streak {streak} | avg RR {avg_rr:.2f}")

    print("\n" + "=" * 100)
    print("PER-PAIR BREAKDOWN")
    print("=" * 100)
    print(f"{'Symbol':8}  {'N':>5}  {'WR':>6}  {'R':>8}  {'DD':>7}  {'Streak':>7}  "
          f"{'AvgRR':>6}  {'Long N':>6}  {'L-WR':>6}  {'Short N':>7}  {'S-WR':>6}")
    print("-" * 105)
    for p in per_symbol:
        print(f"{p['symbol']:8}  {p['n']:5d}  {p['wr']:5.1f}%  {p['r']:+7.1f}R  "
              f"{p['dd']:6.1f}%  {p['streak']:>7}  {p['avg_rr']:5.2f}  "
              f"{p['long_n']:6d}  {p['long_wr']:5.1f}%  {p['short_n']:7d}  {p['short_wr']:5.1f}%")

    if not per_symbol:
        print("\nNo trades at all.")
        return
    all_df = pd.concat([p["df"] for p in per_symbol], ignore_index=True)
    all_df = all_df.sort_values("entry_time").reset_index(drop=True)
    closed = all_df[all_df["outcome"] != "open"]
    wr = len(closed[closed["outcome"] == "win"]) / len(closed) * 100 if len(closed) else 0
    bal_c, dd_c, streak_c = streak_and_dd(all_df)
    avg_rr_all = all_df["achieved_rr"].mean()
    print("-" * 105)
    print(f"{'TOTAL':8}  {len(all_df):5d}  {wr:5.1f}%  {closed['pnl_r'].sum():+7.1f}R  "
          f"{dd_c:6.1f}%  {streak_c:>7}  {avg_rr_all:5.2f}")

    print(f"\nStarting balance       : $10,000")
    print(f"End balance (1% risk)  : ${bal_c:,.0f}")
    print(f"Net return             : {(bal_c/10000 - 1)*100:+.1f}%")
    print(f"Max drawdown           : {dd_c:.1f}%")
    print(f"Longest loss streak    : {streak_c}")
    print(f"Average RR achieved    : {avg_rr_all:.2f}")

    all_df["year"] = pd.to_datetime(all_df["entry_time"]).dt.year
    print("\nPer-year (combined):")
    for yr in sorted(all_df["year"].unique()):
        sub = all_df[all_df["year"] == yr]
        cl = sub[sub["outcome"] != "open"]
        ywr = len(cl[cl["outcome"] == "win"]) / len(cl) * 100 if len(cl) else 0
        _, ydd, ystreak = streak_and_dd(sub)
        print(f"  {yr}: {len(sub):4d} trades, WR {ywr:.1f}%, R {cl['pnl_r'].sum():+.1f}, "
              f"DD {ydd:.1f}%, streak {ystreak}")

    OUT = os.path.join("strategies", "ema_dol")
    os.makedirs(OUT, exist_ok=True)
    all_df.to_csv(os.path.join(OUT, "trades.csv"), index=False)
    pu.plot_equity(all_df, os.path.join(OUT, "equity.png"),
                   "EMA9/21 + ICT DOL · M15 · 7 majors · min 2R")
    print(f"\nSaved {OUT}/trades.csv")
    print(f"Saved {OUT}/equity.png")

    closed_sorted = all_df[all_df["outcome"] != "open"].sort_values("entry_time")
    if len(closed_sorted):
        recent = closed_sorted.iloc[-1]
        sym = recent["symbol"]
        df_15m = bt.fetch_data(sym, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
        pu.plot_recent_trade(df_15m, recent.to_dict(),
                              os.path.join(OUT, "recent_trade.png"),
                              title_prefix="EMA-DOL MOST RECENT", symbol=sym)
        print(f"Saved {OUT}/recent_trade.png  ({sym}  {recent['entry_time']})")

    mt5.shutdown()


if __name__ == "__main__":
    main()
