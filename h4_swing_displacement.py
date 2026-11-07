"""
H4 Swing Sweep + 15M IFVG with DISPLACEMENT-CANDLE stop loss.

This is a clean rewrite of the H4-swing + IFVG model with the new SL rule
the user specified. Every trade is simulated CANDLE BY CANDLE from the
inversion forward — no look-ahead.

Mechanics (mechanical, no discretion):

  1. H4 sweep — a 4H candle wicks past the most recent 4H swing high/low and
     closes back inside.
        wick > swing high, close < swing high  →  SHORT bias
        wick < swing low,  close > swing low   →  LONG  bias

  2. After the sweep candle closes, scan forward on the 15M chart for the
     first OPPOSITE-direction FVG within POST_SWEEP_FVG_WINDOW M15 bars:
        LONG  bias → BEARISH 15M FVG (formed on the drop INTO the sweep)
        SHORT bias → BULLISH 15M FVG (formed on the pop INTO the sweep)
     FVG width must be at least MIN_FVG_WIDTH (filters out micro-imbalances).

  3. Wait for that FVG to be INVERTED (IFVG):
        LONG  bias → a candle CLOSES ABOVE the FVG top    (bearish FVG inverts)
        SHORT bias → a candle CLOSES BELOW the FVG bottom (bullish FVG inverts)
     Must happen within MAX_DISRESPECT_DELAY bars of FVG forming.

  4. ENTRY = close of the inversion candle.

  5. STOP LOSS = the DISPLACEMENT CANDLE's extreme + buffer:
     The displacement candle is the middle (2nd) candle of the 3-candle FVG
     pattern — the one whose strong move created the gap.
        LONG  → SL = c[i+1].low  - SL_BUFFER * fvg.width
        SHORT → SL = c[i+1].high + SL_BUFFER * fvg.width

  6. TAKE PROFIT — one of:
        FIXED_2R  : TP = entry ± 2 × risk
        FIXED_3R  : TP = entry ± 3 × risk
        DOL       : TP = next opposing liquidity pool on the 15M chart
                    (next swing high above entry for longs / swing low below for shorts).
                    Skip if DOL implies RR < MIN_RR.

  7. Risk per trade: 1% of account. Max 1 open position per symbol.
  8. News filter blocks red-folder calendar days (NFP / FOMC / CPI / ECB / Retail).
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
import plot_utils as pu


MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD"]   # dropped USDCHF, NZDUSD
MIN_W = {
    "EURUSD": 0.0002, "GBPUSD": 0.0002, "USDCAD": 0.0002, "USDCHF": 0.0002,
    "USDJPY": 0.02,   "AUDUSD": 0.0002, "NZDUSD": 0.0002,
}
POST_SWEEP_FVG_WINDOW = 24    # M15 bars after sweep to find FVG
MAX_DISRESPECT_DELAY  = 7     # M15 bars for FVG inversion
SL_BUFFER             = 0.25  # multiple of FVG width
SWING_LB              = 5     # bars each side for swing detection (M15)
MIN_RR_DOL            = 2.0


def find_swings_arr(h, l, n=5):
    sh = np.full(len(h), np.nan)
    sl = np.full(len(l), np.nan)
    for i in range(n, len(h) - n):
        if h[i] >= h[i - n:i + n + 1].max():
            sh[i] = h[i]
        if l[i] <= l[i - n:i + n + 1].min():
            sl[i] = l[i]
    return sh, sl


def next_swing_above(price, swing_highs, until_idx, lookforward=200):
    """Find next FUTURE swing high above price, within next lookforward bars."""
    # We can't use future swings — DOL is the next UN-RAIDED level visible NOW.
    # Use the nearest swing high above price from RECENT history (lookback).
    candidates = []
    start = max(0, until_idx - 200)
    for i in range(until_idx - 1, start - 1, -1):
        if not np.isnan(swing_highs[i]) and swing_highs[i] > price:
            candidates.append(swing_highs[i])
    return min(candidates) if candidates else None


def next_swing_below(price, swing_lows, until_idx, lookforward=200):
    candidates = []
    start = max(0, until_idx - 200)
    for i in range(until_idx - 1, start - 1, -1):
        if not np.isnan(swing_lows[i]) and swing_lows[i] < price:
            candidates.append(swing_lows[i])
    return max(candidates) if candidates else None


def simulate(h, l, idx, entry_i, direction, entry, sl, tp, rr_value, max_bars=400):
    """Candle-by-candle forward sim — checks SL FIRST then TP each bar."""
    for j in range(entry_i + 1, min(entry_i + max_bars, len(h))):
        if direction == "long":
            if l[j] <= sl:
                return {"outcome": "loss", "exit": sl, "exit_time": idx[j], "pnl_r": -1.0}
            if h[j] >= tp:
                return {"outcome": "win",  "exit": tp, "exit_time": idx[j], "pnl_r":  rr_value}
        else:
            if h[j] >= sl:
                return {"outcome": "loss", "exit": sl, "exit_time": idx[j], "pnl_r": -1.0}
            if l[j] <= tp:
                return {"outcome": "win",  "exit": tp, "exit_time": idx[j], "pnl_r":  rr_value}
    return {"outcome": "open", "exit": float(np.nan), "exit_time": idx[-1], "pnl_r": 0.0}


def run_one(symbol: str, *, tp_mode: str, rr_fixed: float = 2.0,
            news_block: bool = True, min_rr_dol: float = MIN_RR_DOL,
            max_open: int = 1):
    """Single-symbol backtest. Returns dict with df."""
    df_h4  = bt.fetch_data(symbol, mt5.TIMEFRAME_H4,  bt.START_DATE, bt.END_DATE, tf_name="H4")
    df_15m = bt.fetch_data(symbol, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")

    df_4h_sw = bt.find_swings(df_h4, bt.SWING_LOOKBACK)
    sweeps   = bt.find_sweeps(df_4h_sw)
    fvgs_15m = bt.find_all_fvgs(df_15m)

    h = df_15m["high"].values
    l = df_15m["low"].values
    c = df_15m["close"].values
    idx = df_15m.index

    sh_15, sl_15 = find_swings_arr(h, l, n=SWING_LB)
    news_set = nf.build_news_block_set(idx[0].date(), idx[-1].date()) if news_block else set()

    fvgs_sorted  = sorted(fvgs_15m, key=lambda f: f["form_idx"])
    form_indices = np.array([f["form_idx"] for f in fvgs_sorted])

    min_w = MIN_W.get(symbol, 0.0002)
    trades = []
    open_until = []
    used = set()

    for sweep in sweeps:
        bias = sweep["direction"]  # 'bullish' (long) or 'bearish' (short)
        sig = (sweep["time"], bias)
        if sig in used:
            continue

        sweep_i = int(np.searchsorted(idx, sweep["time"], side="left"))
        if sweep_i >= len(idx):
            continue
        if sweep_i == 0 and sweep["time"] < idx[0]:
            continue

        target_dir = "bearish" if bias == "bullish" else "bullish"
        rev_end = min(sweep_i + POST_SWEEP_FVG_WINDOW, len(idx) - 1)
        lo_pos = int(np.searchsorted(form_indices, sweep_i, side="left"))
        hi_pos = int(np.searchsorted(form_indices, rev_end, side="right"))

        trade_taken = False
        for p in range(hi_pos - 1, lo_pos - 1, -1):
            fvg = fvgs_sorted[p]
            if fvg["direction"] != target_dir:
                continue
            if fvg["width"] < min_w:
                continue

            f_idx  = fvg["form_idx"]
            displacement_idx = f_idx - 1   # the middle candle of the 3-candle FVG
            dis_lo = f_idx + 1
            dis_hi = min(f_idx + MAX_DISRESPECT_DELAY, len(idx) - 1)

            for di in range(dis_lo, dis_hi + 1):
                # IFVG inversion check
                if fvg["direction"] == "bearish":
                    inverted = c[di] > fvg["top"]      # long
                else:
                    inverted = c[di] < fvg["bottom"]   # short
                if not inverted:
                    continue

                entry = c[di]
                buf = fvg["width"] * SL_BUFFER

                # ── SL at DISPLACEMENT CANDLE's extreme ──
                if bias == "bullish":
                    sl_px = l[displacement_idx] - buf
                    direction = "long"
                    risk = entry - sl_px
                else:
                    sl_px = h[displacement_idx] + buf
                    direction = "short"
                    risk = sl_px - entry

                if risk <= 0:
                    break

                # News block on entry day
                if news_block and idx[di].date() in news_set:
                    break

                # Concurrent-position cap
                open_until = [t for t in open_until if t > idx[di]]
                if len(open_until) >= max_open:
                    break

                # ── TP ──
                if tp_mode == "DOL":
                    if direction == "long":
                        dol = next_swing_above(entry, sh_15, di)
                        if dol is None:
                            break
                        tp_px = dol
                        rr_val = (tp_px - entry) / risk
                    else:
                        dol = next_swing_below(entry, sl_15, di)
                        if dol is None:
                            break
                        tp_px = dol
                        rr_val = (entry - tp_px) / risk
                    if rr_val < min_rr_dol or rr_val > 15:
                        break
                else:
                    rr_val = rr_fixed
                    if direction == "long":
                        tp_px = entry + rr_val * risk
                    else:
                        tp_px = entry - rr_val * risk

                result = simulate(h, l, idx, di, direction, entry, sl_px, tp_px, rr_val)
                open_until.append(pd.Timestamp(result["exit_time"]))
                trades.append({
                    "symbol":            symbol,
                    "sweep_time":        sweep["time"],
                    "sweep_bias":        sweep["direction"],
                    "swept_level":       round(sweep["swept_level"], 6),
                    "displacement_idx":  displacement_idx,
                    "displacement_low":  round(l[displacement_idx], 6),
                    "displacement_high": round(h[displacement_idx], 6),
                    "fvg_form_time":     idx[f_idx],
                    "entry_time":        idx[di],
                    "entry_idx":         di,
                    "direction":         direction,
                    "entry":             round(entry, 6),
                    "sl":                round(sl_px, 6),
                    "tp":                round(tp_px, 6),
                    "rr_target":         round(rr_val, 2),
                    "fvg_top":           round(fvg["top"], 6),
                    "fvg_bottom":        round(fvg["bottom"], 6),
                    "fvg_width":         round(fvg["width"], 6),
                    "outcome":           result["outcome"],
                    "exit":              round(result["exit"], 6) if not np.isnan(result["exit"]) else np.nan,
                    "exit_time":         result["exit_time"],
                    "pnl_r":             float(result["pnl_r"]),
                })
                used.add(sig)
                trade_taken = True
                break
            if trade_taken:
                break

    df = pd.DataFrame(trades)
    return df


def streak_dd(df, initial=10_000, risk_pct=1.0):
    bal = initial
    eq = [bal]
    cl = 0
    best_cl = 0
    cw = 0
    best_cw = 0
    for _, row in df.sort_values("entry_time").iterrows():
        bal += bal * (risk_pct / 100) * float(row["pnl_r"])
        eq.append(bal)
        if row["outcome"] == "loss":
            cl += 1
            best_cl = max(best_cl, cl)
            cw = 0
        elif row["outcome"] == "win":
            cw += 1
            best_cw = max(best_cw, cw)
            cl = 0
    eq_arr = np.array(eq)
    dd = ((eq_arr - np.maximum.accumulate(eq_arr)) / np.maximum.accumulate(eq_arr) * 100).min()
    return bal, dd, best_cl, best_cw, eq_arr


def report_variant(label, tp_mode, rr_fixed=2.0, news=True, max_open=1):
    print(f"\n── {label} ──")
    all_dfs = []
    per_pair = []
    for s in MAJORS:
        df = run_one(s, tp_mode=tp_mode, rr_fixed=rr_fixed, news_block=news, max_open=max_open)
        if not len(df):
            continue
        all_dfs.append(df)
        closed = df[df["outcome"] != "open"]
        if not len(closed):
            continue
        wr_s = (closed["outcome"] == "win").sum() / len(closed) * 100
        bal, dd, ls, ws, _ = streak_dd(df)
        per_pair.append({
            "sym": s, "n": len(df), "wr": wr_s, "r": closed["pnl_r"].sum(),
            "dd": dd, "ls": ls, "ws": ws,
        })

    print(f"  {'Symbol':8}  {'N':>4}  {'WR':>6}  {'R':>8}  {'DD':>7}  {'LossStrk':>8}  {'WinStrk':>7}")
    for p in per_pair:
        print(f"  {p['sym']:8}  {p['n']:4d}  {p['wr']:5.1f}%  {p['r']:+7.1f}R  "
              f"{p['dd']:6.1f}%  {p['ls']:>8}  {p['ws']:>7}")

    all_df = pd.concat(all_dfs, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    closed = all_df[all_df["outcome"] != "open"]
    wr = (closed["outcome"] == "win").sum() / len(closed) * 100 if len(closed) else 0
    bal, dd, ls, ws, _ = streak_dd(all_df)
    total_r = closed["pnl_r"].sum()
    gw = closed.loc[closed["outcome"] == "win", "pnl_r"].sum()
    gl = -closed.loc[closed["outcome"] == "loss", "pnl_r"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    years = (pd.to_datetime(all_df["entry_time"]).max() - pd.to_datetime(all_df["entry_time"]).min()).days / 365.25
    cagr = (bal / 10_000) ** (1 / years) - 1 if years > 0 else 0

    print(f"  {'TOTAL':8}  {len(all_df):4d}  {wr:5.1f}%  {total_r:+7.1f}R  "
          f"{dd:6.1f}%  {ls:>8}  {ws:>7}")
    print(f"  Profit factor: {pf:.2f}   Trades/year: {len(all_df)/years:.1f}   "
          f"Trades/week: {len(all_df)/years/52:.1f}")
    print(f"  $10k → ${bal:,.0f}  ({(bal/10000-1)*100:+.0f}%, CAGR {cagr*100:+.1f}%)")

    return {"label": label, "all_df": all_df, "wr": wr, "r": total_r, "pf": pf,
            "dd": dd, "ls": ls, "bal": bal}


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    print("=" * 90)
    print("H4-SWING SWEEP + 15M IFVG  ·  SL at DISPLACEMENT CANDLE  ·  candle-by-candle")
    print("=" * 90)

    runs = []
    # Realistic execution (1 concurrent position cap)
    runs.append(report_variant("Fixed 2R · cap=1",        "FIXED", rr_fixed=2.0, max_open=1))
    runs.append(report_variant("Fixed 3R · cap=1",        "FIXED", rr_fixed=3.0, max_open=1))
    runs.append(report_variant("DOL TP (≥2R) · cap=1",    "DOL",   max_open=1))
    # Show true signal count by removing the cap (overlapping trades allowed)
    runs.append(report_variant("Fixed 2R · NO CAP",       "FIXED", rr_fixed=2.0, max_open=99))
    runs.append(report_variant("Fixed 3R · NO CAP",       "FIXED", rr_fixed=3.0, max_open=99))
    runs.append(report_variant("DOL TP (≥2R) · NO CAP",   "DOL",   max_open=99))

    # Save best by R
    OUT = os.path.join("strategies", "h4_displacement")
    os.makedirs(OUT, exist_ok=True)
    best = max(runs, key=lambda r: r["r"])
    safe = best["label"].replace(" ", "_").replace("·", "_").replace("(", "").replace(")", "").replace("≥", "ge")
    best["all_df"].to_csv(os.path.join(OUT, f"trades_BEST_{safe}.csv"), index=False)
    pu.plot_equity(best["all_df"], os.path.join(OUT, "equity_best.png"),
                   f"H4-swing + displacement SL · {best['label']}")
    print(f"\nBest variant: {best['label']}  R={best['r']:+.1f}  saved to {OUT}/")

    # Save all
    for r in runs:
        safe = r["label"].replace(" ", "_").replace("·", "_").replace("(", "").replace(")", "").replace("≥", "ge")
        r["all_df"].to_csv(os.path.join(OUT, f"trades_{safe}.csv"), index=False)

    # Side-by-side summary
    print("\n" + "=" * 90)
    print("SIDE-BY-SIDE SUMMARY")
    print("=" * 90)
    print(f"  {'Variant':38}  {'N':>4}  {'WR':>6}  {'R':>8}  {'PF':>5}  {'DD':>7}  {'LossStrk':>8}  {'End$':>10}")
    for r in runs:
        print(f"  {r['label']:38}  {len(r['all_df']):4d}  {r['wr']:5.1f}%  "
              f"{r['r']:+7.1f}R  {r['pf']:5.2f}  {r['dd']:6.1f}%  {r['ls']:>8}  ${r['bal']:>9,.0f}")

    mt5.shutdown()


if __name__ == "__main__":
    main()
