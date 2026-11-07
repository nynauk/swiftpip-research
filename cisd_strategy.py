"""
ICT Sweep + 5M IFVG + CISD entry strategy.

Sequence (mechanical):
  1. Wait for PDH or PDL to be SWEPT (any time during the trading day):
        PDH swept (wick > PDH, close < PDH) → SHORT bias
        PDL swept (wick < PDL, close > PDL) → LONG bias
     Detected on 15M (the parent timeframe for liquidity).

  2. After the sweep, drop to 5M. Find the first FVG in the OPPOSITE
     direction of the sweep (the reversal-leg FVG):
        SHORT bias → bearish FVG
        LONG  bias → bullish FVG
     FVG must form within POST_SWEEP_FVG_WINDOW 5M candles of the sweep.

  3. Wait for the FVG to be INVERTED (IFVG) — a 5M candle that closes
     THROUGH the FVG in the bias direction:
        SHORT bias / bearish FVG → candle CLOSES BELOW fvg.bottom
        LONG  bias / bullish FVG → candle CLOSES ABOVE fvg.top
     Must happen within MAX_DISRESPECT_DELAY 5M candles of FVG forming.

  4. After IFVG inversion, wait for CISD (Change in State of Delivery).
     CISD = first 5M candle that closes through the most recent opposing
     swing point on 5M, confirming structural break in the bias direction:
        LONG  bias → first candle close > max(prev N candles' highs)
        SHORT bias → first candle close < min(prev N candles' lows)
     Must happen within CISD_WINDOW 5M candles of IFVG.

  5. ENTRY at the close of the CISD candle.

  6. STOP LOSS just beyond the swept structural extreme (the wick that
     swept the PDH/PDL), plus a small buffer.

  7. TAKE PROFIT at fixed RR (default 2R per user spec).

  8. News filter: skip the day if a red-folder event is scheduled.

  9. Max 1 concurrent position per symbol.
"""
import os
import numpy as np
import pandas as pd

import main as bt
import news_filter as nf
import plot_utils as pu


# ─────────────────────────────────────────────────────────────────────────────
POST_SWEEP_FVG_WINDOW = 24    # 5M candles to look for the FVG after sweep (2 hours)
MAX_DISRESPECT_DELAY  = 12    # 5M candles for IFVG inversion (1 hour)
CISD_WINDOW           = 12    # 5M candles after IFVG to find CISD (1 hour)
CISD_LOOKBACK         = 5     # number of candles for "recent swing" reference

MIN_W = {
    "EURUSD":     0.00020, "GBPUSD":     0.00020, "USDCAD":     0.00020,
    "USDCHF":     0.00020, "USDJPY":     0.020,   "AUDUSD":     0.00020,
    "NZDUSD":     0.00020,
}


def build_pdh_pdl_from(df_any):
    """Resample any TF to daily H/L, then shift for prev-day reference."""
    daily = df_any.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    pdh = daily["high"].shift(1).dropna().to_dict()
    pdl = daily["low"].shift(1).dropna().to_dict()
    return pdh, pdl


def simulate(df, entry_idx, direction, entry, sl, tp, rr, max_bars=300):
    h = df["high"].values
    l = df["low"].values
    idx = df.index
    for j in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
        if direction == "long":
            if l[j] <= sl:  return {"outcome": "loss", "exit": sl, "exit_time": idx[j], "pnl_r": -1.0}
            if h[j] >= tp:  return {"outcome": "win",  "exit": tp, "exit_time": idx[j], "pnl_r":  rr}
        else:
            if h[j] >= sl:  return {"outcome": "loss", "exit": sl, "exit_time": idx[j], "pnl_r": -1.0}
            if l[j] <= tp:  return {"outcome": "win",  "exit": tp, "exit_time": idx[j], "pnl_r":  rr}
    return {"outcome": "open", "exit": df["close"].iloc[-1], "exit_time": idx[-1], "pnl_r": 0.0}


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


def run_cisd(symbol: str, *,
             rr: float = 2.0,
             min_fvg_width: float = None,
             sl_buffer_frac: float = 0.10,   # fraction of (entry-sweep_extreme) distance
             sl_mode: str = "SWEEP_EXTREME", # or "IFVG_EDGE" — opposite far side of IFVG
             news_block: bool = True,
             max_open_per_symbol: int = 1) -> dict:
    """
    PDH/PDL sweep + 5M IFVG + CISD entry, 2R target. Returns same dict shape
    as silver_bullet.run_silver_bullet.
    sl_mode:
      "SWEEP_EXTREME" — SL just past the swept wick (wider, more breathing room)
      "IFVG_EDGE"     — SL just past the IFVG far side (tighter, 2R more reachable)
    """
    import MetaTrader5 as mt5
    if min_fvg_width is None:
        min_fvg_width = MIN_W.get(symbol, 0.0002)

    df_5m = bt.fetch_data(symbol, mt5.TIMEFRAME_M5,  bt.START_DATE, bt.END_DATE, tf_name="M5")

    h = df_5m["high"].values
    l = df_5m["low"].values
    c = df_5m["close"].values
    idx = df_5m.index
    dates = idx.normalize()

    pdh_map, pdl_map = build_pdh_pdl_from(df_5m)
    news_set = nf.build_news_block_set(idx[0].date(), idx[-1].date()) if news_block else set()

    trades = []
    open_until = []
    pdh_today_swept = set()
    pdl_today_swept = set()

    for i in range(len(df_5m)):
        d = dates[i]
        d_date = pd.Timestamp(d).date()
        if news_block and d_date in news_set:
            continue
        pdh = pdh_map.get(d)
        pdl = pdl_map.get(d)
        if pdh is None or pdl is None or pd.isna(pdh) or pd.isna(pdl):
            continue

        # detect sweep at bar i
        sweep_dir = None
        if d not in pdh_today_swept and h[i] > pdh and c[i] < pdh:
            sweep_dir = "short"
            pdh_today_swept.add(d)
            sweep_extreme = h[i]
        elif d not in pdl_today_swept and l[i] < pdl and c[i] > pdl:
            sweep_dir = "long"
            pdl_today_swept.add(d)
            sweep_extreme = l[i]
        else:
            continue

        # concurrent-position check
        open_until = [t for t in open_until if t > idx[i]]
        if len(open_until) >= max_open_per_symbol:
            continue

        sweep_i = i
        sweep_time = idx[i]
        # Target FVG = OPPOSITE of bias (formed during the move INTO the sweep,
        # gets inverted by the reversal):
        #   sweep_dir=long  (PDL swept, bias LONG)  → look for BEARISH FVG that
        #                   was created on the drop into the PDL sweep, then
        #                   gets inverted when price reverses up.
        #   sweep_dir=short (PDH swept, bias SHORT) → BULLISH FVG from the rally
        #                   into PDH, inverted on reversal down.
        target_fvg_dir = "bullish" if sweep_dir == "short" else "bearish"

        # ── Find first opposite-direction FVG within POST_SWEEP_FVG_WINDOW ──
        fvg_end = min(sweep_i + POST_SWEEP_FVG_WINDOW, len(df_5m) - 3)
        fvg = None
        for k in range(sweep_i + 1, fvg_end + 1):
            if k < 2:
                continue
            if target_fvg_dir == "bullish":
                if l[k] > h[k - 2]:
                    fvg = {"direction": "bullish", "top": float(l[k]),
                           "bottom": float(h[k - 2]), "width": float(l[k] - h[k - 2]),
                           "form_idx": k}
                    break
            else:
                if h[k] < l[k - 2]:
                    fvg = {"direction": "bearish", "top": float(l[k - 2]),
                           "bottom": float(h[k]), "width": float(l[k - 2] - h[k]),
                           "form_idx": k}
                    break
        if fvg is None or fvg["width"] < min_fvg_width:
            continue

        # ── Wait for FVG inversion (IFVG) in the BIAS direction ──
        # Bullish FVG inverted = candle CLOSES BELOW its bottom (bearish IFVG / short)
        # Bearish FVG inverted = candle CLOSES ABOVE its top    (bullish IFVG / long)
        inv_end = min(fvg["form_idx"] + MAX_DISRESPECT_DELAY, len(df_5m) - 1)
        ifvg_idx = None
        for k in range(fvg["form_idx"] + 1, inv_end + 1):
            if fvg["direction"] == "bullish" and sweep_dir == "short":
                if c[k] < fvg["bottom"]:
                    ifvg_idx = k
                    break
            elif fvg["direction"] == "bearish" and sweep_dir == "long":
                if c[k] > fvg["top"]:
                    ifvg_idx = k
                    break
        if ifvg_idx is None:
            continue

        # ── CISD: first 5M candle that closes through the recent opposing swing ──
        # For long bias: candle close > max(highs of CISD_LOOKBACK candles before it)
        # For short bias: candle close < min(lows of CISD_LOOKBACK candles before it)
        cisd_end = min(ifvg_idx + CISD_WINDOW, len(df_5m) - 1)
        cisd_idx = None
        for k in range(ifvg_idx + 1, cisd_end + 1):
            if k - CISD_LOOKBACK < 0:
                continue
            if sweep_dir == "long":
                ref = h[k - CISD_LOOKBACK:k].max()
                if c[k] > ref:
                    cisd_idx = k
                    break
            else:
                ref = l[k - CISD_LOOKBACK:k].min()
                if c[k] < ref:
                    cisd_idx = k
                    break
        if cisd_idx is None:
            continue

        # ── Entry at CISD close, SL, TP = RR ──
        entry = c[cisd_idx]
        if sweep_dir == "long":
            if sl_mode == "IFVG_EDGE":
                # SL just below the IFVG support (= former bearish FVG bottom)
                sl = fvg["bottom"] - fvg["width"] * 0.25
            else:
                sl_dist_raw = entry - sweep_extreme
                sl = sweep_extreme - abs(sl_dist_raw) * sl_buffer_frac
            risk = entry - sl
            if risk <= 0:
                continue
            tp = entry + rr * risk
            direction = "long"
        else:
            if sl_mode == "IFVG_EDGE":
                # SL just above the IFVG resistance (= former bullish FVG top)
                sl = fvg["top"] + fvg["width"] * 0.25
            else:
                sl_dist_raw = sweep_extreme - entry
                sl = sweep_extreme + abs(sl_dist_raw) * sl_buffer_frac
            risk = sl - entry
            if risk <= 0:
                continue
            tp = entry - rr * risk
            direction = "short"

        if news_block and idx[cisd_idx].date() in news_set:
            continue

        result = simulate(df_5m, cisd_idx, direction, entry, sl, tp, rr=rr)
        open_until.append(pd.Timestamp(result["exit_time"]))
        trades.append({
            "symbol":        symbol,
            "sweep_time":    sweep_time,
            "swept_level":   float(pdh) if sweep_dir == "short" else float(pdl),
            "sweep_extreme": float(sweep_extreme),
            "sweep_dir":     sweep_dir,
            "fvg_time":      idx[fvg["form_idx"]],
            "ifvg_time":     idx[ifvg_idx],
            "entry_time":    idx[cisd_idx],
            "entry_idx":     cisd_idx,
            "direction":     direction,
            "entry":         round(entry, 6),
            "sl":            round(sl, 6),
            "tp":            round(tp, 6),
            "fvg_top":       round(fvg["top"], 6),
            "fvg_bottom":    round(fvg["bottom"], 6),
            "fvg_width":     round(fvg["width"], 6),
            "outcome":       result["outcome"],
            "exit":          round(result["exit"], 6),
            "exit_time":     result["exit_time"],
            "pnl_r":         float(result["pnl_r"]),
        })

    df = pd.DataFrame(trades)
    if not len(df):
        return {"symbol": symbol, "n": 0, "wr": 0, "total_r": 0, "df": df}
    closed = df[df["outcome"] != "open"]
    wr = len(closed[closed["outcome"] == "win"]) / len(closed) * 100 if len(closed) else 0
    return {"symbol": symbol, "n": len(df), "wr": wr,
            "total_r": closed["pnl_r"].sum(), "df": df}


def main():
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]

    # ── Variant scan (SL mode × RR) ──
    print("=" * 100)
    print("CISD VARIANT SCAN — find the best SL/RR combo")
    print("=" * 100)
    for label, kwargs in [
        ("SL=SWEEP_EXTREME · RR=2", dict(sl_mode="SWEEP_EXTREME", rr=2.0)),
        ("SL=SWEEP_EXTREME · RR=3", dict(sl_mode="SWEEP_EXTREME", rr=3.0)),
        ("SL=IFVG_EDGE    · RR=2", dict(sl_mode="IFVG_EDGE", rr=2.0)),
        ("SL=IFVG_EDGE    · RR=3", dict(sl_mode="IFVG_EDGE", rr=3.0)),
    ]:
        dfs = []
        for s in MAJORS:
            r = run_cisd(s, news_block=True, **kwargs)
            if r["n"]:
                dfs.append(r["df"])
        if not dfs:
            continue
        all_df = pd.concat(dfs, ignore_index=True).sort_values("entry_time")
        cl = all_df[all_df["outcome"] != "open"]
        wr = len(cl[cl["outcome"] == "win"]) / len(cl) * 100 if len(cl) else 0
        _, dd, streak = streak_and_dd(all_df)
        print(f"  {label:32}  N={len(all_df):4d}  WR={wr:5.1f}%  "
              f"R={cl['pnl_r'].sum():+7.1f}  DD={dd:5.1f}%  streak={streak}")

    RR = 2.0
    SL = "IFVG_EDGE"
    print("\n" + "=" * 100)
    print(f"PER-PAIR DETAIL · SL={SL} · RR={RR} · news-blocked (best 2RR config)")
    print("=" * 100)

    per_symbol = []
    for s in MAJORS:
        print(f"--- {s} ---")
        r = run_cisd(s, rr=RR, sl_mode=SL, news_block=True)
        if r["n"]:
            bal, dd, streak = streak_and_dd(r["df"])
            ln = r["df"][r["df"]["direction"] == "long"]
            sh = r["df"][r["df"]["direction"] == "short"]
            l_cl = ln[ln["outcome"] != "open"]
            s_cl = sh[sh["outcome"] != "open"]
            lwr = len(l_cl[l_cl["outcome"] == "win"]) / len(l_cl) * 100 if len(l_cl) else 0
            swr = len(s_cl[s_cl["outcome"] == "win"]) / len(s_cl) * 100 if len(s_cl) else 0
            per_symbol.append({
                "symbol": s, "n": r["n"], "wr": r["wr"], "r": r["total_r"],
                "dd": dd, "streak": streak,
                "long_n": len(ln), "long_wr": lwr,
                "short_n": len(sh), "short_wr": swr,
                "df": r["df"],
            })
            print(f"  {r['n']:4d} trades | WR {r['wr']:5.1f}% | R {r['total_r']:+7.1f} "
                  f"| DD {dd:.1f}% | streak {streak}")
        else:
            print("  no setups")

    # ── Per-pair table ──
    print("\n" + "=" * 100)
    print("PER-PAIR BREAKDOWN — CISD strategy")
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

    days = (idx_first(all_df), idx_last(all_df))
    print(f"\nData period            : {days[0]} → {days[1]}")
    print(f"Starting balance       : $10,000")
    print(f"End balance (1% risk)  : ${bal_c:,.0f}")
    print(f"Net return             : {(bal_c/10000 - 1)*100:+.1f}%")
    print(f"Max drawdown           : {dd_c:.1f}%")
    print(f"Longest loss streak    : {streak_c}")

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

    OUT = os.path.join("strategies", "cisd")
    os.makedirs(OUT, exist_ok=True)
    all_df.to_csv(os.path.join(OUT, "trades.csv"), index=False)
    pu.plot_equity(all_df, os.path.join(OUT, "equity.png"),
                   f"CISD · sweep + 5M IFVG + CISD entry · RR={RR} · 7 majors")
    print(f"\nSaved {OUT}/trades.csv")
    print(f"Saved {OUT}/equity.png")

    closed_sorted = all_df[all_df["outcome"] != "open"].sort_values("entry_time")
    if len(closed_sorted):
        recent = closed_sorted.iloc[-1]
        sym = recent["symbol"]
        df_5m = bt.fetch_data(sym, mt5.TIMEFRAME_M5, bt.START_DATE, bt.END_DATE, tf_name="M5")
        pu.plot_recent_trade(df_5m, recent.to_dict(),
                              os.path.join(OUT, "recent_trade.png"),
                              title_prefix="CISD MOST RECENT (M5)", symbol=sym)
        print(f"Saved {OUT}/recent_trade.png  ({sym}  {recent['entry_time']})")

    mt5.shutdown()


def idx_first(df):
    return pd.to_datetime(df["entry_time"]).min()


def idx_last(df):
    return pd.to_datetime(df["entry_time"]).max()


if __name__ == "__main__":
    main()
