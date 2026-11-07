"""
4H Sweep + Confluence-Level Sweep  →  IFVG + CISD  (5M / 15M execution).

Idea (mechanical):

  A. A 4H swing high/low is swept (4H candle wicks past it, closes back inside).
     wick below swing low  → bullish bias (LONG)
     wick above swing high → bearish bias (SHORT)

  B. CONFLUENCE — the sweep must coincide with resting HTF liquidity. At least
     one of these must be true:
       (1) a prior key level sits within CONFL_TOL of the swept 4H level, on the
           matching side. Prior levels considered:
               • Previous Day  High / Low   (PDH / PDL)
               • Previous Week High / Low   (PWH / PWL)
               • Previous Month High / Low  (PMH / PML)
               • Previous Session High / Low (Asia / London / NY ranges)
       (2) OR a 4H Fair Value Gap sits OUTSIDE the sweep candle range in the
           bias direction (bullish 4H FVG below a swept low / bearish above a
           swept high) that price is reacting from.

  C. CLUSTERING / "wait for the furthest" — gather every prior level within
     CLUSTER_TOL of the swept 4H level on the correct side. If several cluster
     together (e.g. PDL and PWL both just below), the DEEPEST one is the true
     draw on liquidity. We do NOT act until price trades through that deepest
     level. The deepest extreme then becomes the stop reference.

  D. After the deep sweep, on the execution TF (5M or 15M):
       • find the reversal FVG (opposite the bias, formed on the move into the
         sweep),
       • wait for it to INVERT (IFVG) in the bias direction,
       • wait for a CISD (close through the recent opposing swing) — the entry.

  E. ENTRY = close of the CISD candle.
     STOP  = beyond the deepest swept extreme + buffer.
     TP    = 2R (FIXED) or next opposing liquidity (DOL), configurable.

  F. News-blocked, 1 concurrent position per symbol, candle-by-candle sim.
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


MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]

# Per-symbol pip scale (1 pip in price units)
PIP = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "USDCAD": 0.0001, "USDCHF": 0.0001,
    "USDJPY": 0.01,   "AUDUSD": 0.0001, "NZDUSD": 0.0001,
}
MIN_FVG_PIPS          = 2      # min FVG width in pips (exec TF)
CONFL_TOL_PIPS        = 15     # prior level must be within this of the 4H sweep level
CLUSTER_TOL_PIPS      = 20     # levels within this of the sweep cluster together
POST_SWEEP_WINDOW     = 48     # exec bars after 4H sweep to find deep sweep + FVG
IFVG_DELAY            = 12     # exec bars for FVG inversion
CISD_WINDOW           = 12     # exec bars for CISD after IFVG
CISD_LOOKBACK         = 5
SL_BUFFER             = 0.25   # * FVG width beyond the deepest extreme
SESSIONS = [("Asia", 0, 8), ("London", 8, 13), ("NY", 13, 22)]  # broker hours


# ─────────────────────────────────────────────────────────────────────────────
def build_prior_levels(df_exec: pd.DataFrame):
    """
    Returns a function level_lookup(ts) -> dict of prior levels available at ts:
      {'PDH','PDL','PWH','PWL','PMH','PML','PSH','PSL'}
    All are levels that were fully formed BEFORE the day/week/month/session of ts.
    """
    # Daily
    d = df_exec.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    pdh = d["high"].shift(1)
    pdl = d["low"].shift(1)
    # Weekly (ISO)
    w = df_exec.resample("W").agg({"high": "max", "low": "min"}).dropna()
    pwh = w["high"].shift(1)
    pwl = w["low"].shift(1)
    # Monthly
    m = df_exec.resample("ME").agg({"high": "max", "low": "min"}).dropna()
    pmh = m["high"].shift(1)
    pml = m["low"].shift(1)

    pdh_d = pdh.to_dict(); pdl_d = pdl.to_dict()
    # week/month keyed by period
    pwh_d = {k.to_period("W"): v for k, v in pwh.items()}
    pwl_d = {k.to_period("W"): v for k, v in pwl.items()}
    pmh_d = {k.to_period("M"): v for k, v in pmh.items()}
    pml_d = {k.to_period("M"): v for k, v in pml.items()}

    # Previous session H/L: precompute per (date, session) then find most recent
    # completed session before a given timestamp.
    idx = df_exec.index
    h = df_exec["high"].values
    l = df_exec["low"].values
    hours = idx.hour
    dates = idx.normalize()
    sess_records = []  # (end_time, high, low)
    uniq_days, starts = np.unique(dates, return_index=True)
    ends = np.append(starts[1:], len(df_exec))
    for di, day in enumerate(uniq_days):
        a, b = starts[di], ends[di]
        dh = hours[a:b]
        for sname, s0, s1 in SESSIONS:
            mask = (dh >= s0) & (dh < s1)
            if not mask.any():
                continue
            sh = h[a:b][mask].max()
            sl = l[a:b][mask].min()
            # session end timestamp ≈ day + s1 hours
            end_t = pd.Timestamp(day) + pd.Timedelta(hours=s1)
            sess_records.append((end_t, sh, sl))
    sess_records.sort()
    sess_end_times = np.array([r[0] for r in sess_records])
    sess_highs = np.array([r[1] for r in sess_records])
    sess_lows = np.array([r[2] for r in sess_records])

    def lookup(ts):
        ts = pd.Timestamp(ts)
        day = ts.normalize()
        res = {
            "PDH": pdh_d.get(day), "PDL": pdl_d.get(day),
            "PWH": pwh_d.get(ts.to_period("W")), "PWL": pwl_d.get(ts.to_period("W")),
            "PMH": pmh_d.get(ts.to_period("M")), "PML": pml_d.get(ts.to_period("M")),
            "PSH": None, "PSL": None,
        }
        # most recent completed session strictly before ts
        pos = np.searchsorted(sess_end_times, ts, side="left") - 1
        if pos >= 0:
            res["PSH"] = sess_highs[pos]
            res["PSL"] = sess_lows[pos]
        return res

    return lookup


def simulate(h, l, idx, entry_i, direction, entry, sl, tp, rr_value, max_bars=600):
    for j in range(entry_i + 1, min(entry_i + max_bars, len(h))):
        if direction == "long":
            if l[j] <= sl:  return {"outcome": "loss", "exit": sl, "exit_time": idx[j], "pnl_r": -1.0}
            if h[j] >= tp:  return {"outcome": "win",  "exit": tp, "exit_time": idx[j], "pnl_r":  rr_value}
        else:
            if h[j] >= sl:  return {"outcome": "loss", "exit": sl, "exit_time": idx[j], "pnl_r": -1.0}
            if l[j] <= tp:  return {"outcome": "win",  "exit": tp, "exit_time": idx[j], "pnl_r":  rr_value}
    return {"outcome": "open", "exit": float("nan"), "exit_time": idx[-1], "pnl_r": 0.0}


def _swings(h, l, n=5):
    sh = np.full(len(h), np.nan); slo = np.full(len(l), np.nan)
    for i in range(n, len(h) - n):
        if h[i] >= h[i - n:i + n + 1].max(): sh[i] = h[i]
        if l[i] <= l[i - n:i + n + 1].min(): slo[i] = l[i]
    return sh, slo


def run(symbol: str, *, tf: str = "M15", tp_mode: str = "FIXED", rr: float = 2.0,
        min_rr: float = 2.0, require_confluence: bool = True,
        allow_4h_fvg_alt: bool = True, news_block: bool = True) -> pd.DataFrame:
    pip = PIP.get(symbol, 0.0001)
    min_fvg = MIN_FVG_PIPS * pip
    confl_tol = CONFL_TOL_PIPS * pip
    cluster_tol = CLUSTER_TOL_PIPS * pip

    tf_const = mt5.TIMEFRAME_M5 if tf == "M5" else mt5.TIMEFRAME_M15
    df_h4  = bt.fetch_data(symbol, mt5.TIMEFRAME_H4, bt.START_DATE, bt.END_DATE, tf_name="H4")
    df_ex  = bt.fetch_data(symbol, tf_const, bt.START_DATE, bt.END_DATE, tf_name=tf)

    df_4h_sw = bt.find_swings(df_h4, bt.SWING_LOOKBACK)
    sweeps   = bt.find_sweeps(df_4h_sw)
    fvgs_4h  = bt.find_all_fvgs(df_4h_sw)
    fvgs_ex  = bt.find_all_fvgs(df_ex)

    h = df_ex["high"].values
    l = df_ex["low"].values
    c = df_ex["close"].values
    idx = df_ex.index

    level_lookup = build_prior_levels(df_ex)
    sh_arr, sl_arr = _swings(h, l, n=5) if tp_mode == "DOL" else (None, None)
    news_set = nf.build_news_block_set(idx[0].date(), idx[-1].date()) if news_block else set()

    fvgs_sorted = sorted(fvgs_ex, key=lambda f: f["form_idx"])
    form_indices = np.array([f["form_idx"] for f in fvgs_sorted])

    trades = []
    open_until = []
    used = set()

    for sweep in sweeps:
        bias = sweep["direction"]              # bullish→LONG, bearish→SHORT
        sig = (sweep["time"], bias)
        if sig in used:
            continue
        swept_level = sweep["swept_level"]

        sweep_i = int(np.searchsorted(idx, sweep["time"], side="left"))
        if sweep_i >= len(idx) or (sweep_i == 0 and sweep["time"] < idx[0]):
            continue

        # ── Confluence: prior levels near the swept 4H level, correct side ──
        lv = level_lookup(sweep["time"])
        if bias == "bullish":   # swept a low → gather nearby LOW levels
            cand = [lv[k] for k in ("PDL", "PWL", "PML", "PSL")
                    if lv[k] is not None and abs(lv[k] - swept_level) <= cluster_tol]
        else:                   # swept a high → gather nearby HIGH levels
            cand = [lv[k] for k in ("PDH", "PWH", "PMH", "PSH")
                    if lv[k] is not None and abs(lv[k] - swept_level) <= cluster_tol]

        has_level_confluence = any(abs(x - swept_level) <= confl_tol for x in cand)

        # ── Alternative confluence: 4H FVG outside the sweep, bias direction ──
        has_4h_fvg = False
        if allow_4h_fvg_alt:
            sweep_4h_i = int(np.searchsorted(df_h4.index, sweep["time"], side="left"))
            if sweep_4h_i < len(df_h4):
                s_hi = float(df_h4["high"].iloc[sweep_4h_i])
                s_lo = float(df_h4["low"].iloc[sweep_4h_i])
                for f4 in fvgs_4h:
                    if df_h4.index[f4["form_idx"]] > sweep["time"]:
                        continue
                    if bias == "bullish" and f4["direction"] == "bullish" and f4["top"] <= s_lo:
                        has_4h_fvg = True; break
                    if bias == "bearish" and f4["direction"] == "bearish" and f4["bottom"] >= s_hi:
                        has_4h_fvg = True; break

        if require_confluence and not (has_level_confluence or has_4h_fvg):
            continue

        # ── Deepest clustered level = the "furthest" draw ──
        # Include the swept 4H level itself as a baseline.
        if bias == "bullish":
            deepest = min([swept_level] + cand)   # lowest low
        else:
            deepest = max([swept_level] + cand)   # highest high

        # ── Wait for the furthest level to be taken ──
        win_end = min(sweep_i + POST_SWEEP_WINDOW, len(idx) - 1)
        deep_i = None
        for k in range(sweep_i, win_end + 1):
            if bias == "bullish":
                if l[k] <= deepest:
                    deep_i = k
                    break
            else:
                if h[k] >= deepest:
                    deep_i = k
                    break
        if deep_i is None:
            continue

        # deepest extreme = actual price extreme reached in the leg
        if bias == "bullish":
            deep_extreme = l[sweep_i:deep_i + 1].min()
        else:
            deep_extreme = h[sweep_i:deep_i + 1].max()

        # ── Reversal FVG (opposite bias) after the deep sweep ──
        target_dir = "bearish" if bias == "bullish" else "bullish"
        fvg_end = min(deep_i + POST_SWEEP_WINDOW, len(idx) - 1)
        lo_pos = int(np.searchsorted(form_indices, deep_i, side="left"))
        hi_pos = int(np.searchsorted(form_indices, fvg_end, side="right"))

        trade_done = False
        for p in range(lo_pos, hi_pos):
            fvg = fvgs_sorted[p]
            if fvg["direction"] != target_dir or fvg["width"] < min_fvg:
                continue
            f_idx = fvg["form_idx"]

            # IFVG inversion
            inv_end = min(f_idx + IFVG_DELAY, len(idx) - 1)
            ifvg_idx = None
            for di in range(f_idx + 1, inv_end + 1):
                if fvg["direction"] == "bearish":
                    if c[di] > fvg["top"]:
                        ifvg_idx = di; break
                else:
                    if c[di] < fvg["bottom"]:
                        ifvg_idx = di; break
            if ifvg_idx is None:
                continue

            # CISD
            cisd_end = min(ifvg_idx + CISD_WINDOW, len(idx) - 1)
            cisd_idx = None
            for k in range(ifvg_idx + 1, cisd_end + 1):
                if k - CISD_LOOKBACK < 0:
                    continue
                if bias == "bullish":
                    if c[k] > h[k - CISD_LOOKBACK:k].max():
                        cisd_idx = k; break
                else:
                    if c[k] < l[k - CISD_LOOKBACK:k].min():
                        cisd_idx = k; break
            if cisd_idx is None:
                continue

            # News + concurrency
            if news_block and idx[cisd_idx].date() in news_set:
                continue
            open_until = [t for t in open_until if t > idx[cisd_idx]]
            if len(open_until) >= 1:
                continue

            entry = c[cisd_idx]
            buf = fvg["width"] * SL_BUFFER
            if bias == "bullish":
                direction = "long"
                sl = deep_extreme - buf
                risk = entry - sl
            else:
                direction = "short"
                sl = deep_extreme + buf
                risk = sl - entry
            if risk <= 0:
                continue

            if tp_mode == "DOL":
                if direction == "long":
                    cands = [sh_arr[k] for k in range(max(0, cisd_idx - 400), cisd_idx)
                             if not np.isnan(sh_arr[k]) and sh_arr[k] > entry]
                    dol = min(cands) if cands else None
                    if dol is None:
                        continue
                    tp = dol; rr_val = (tp - entry) / risk
                else:
                    cands = [sl_arr[k] for k in range(max(0, cisd_idx - 400), cisd_idx)
                             if not np.isnan(sl_arr[k]) and sl_arr[k] < entry]
                    dol = max(cands) if cands else None
                    if dol is None:
                        continue
                    tp = dol; rr_val = (entry - tp) / risk
                if rr_val < min_rr or rr_val > 15:
                    continue

            elif tp_mode == "OPP_LIQ":
                # Target the NEAREST opposing liquidity that is >= min_rr away:
                #   opposing prior levels (PDH/PWH/PMH/PSH for longs, lows for shorts)
                #   + near edge of unfilled 4H FVGs in the target direction
                elv = level_lookup(idx[cisd_idx])
                targets = []
                if direction == "long":
                    for key in ("PDH", "PWH", "PMH", "PSH"):
                        v = elv.get(key)
                        if v is not None and v > entry:
                            targets.append(float(v))
                    # 4H FVG near-edges above entry (unfilled imbalance = draw)
                    for f4 in fvgs_4h:
                        if df_h4.index[f4["form_idx"]] > idx[cisd_idx]:
                            continue
                        if f4["bottom"] > entry:      # gap sits above → first touch = bottom
                            targets.append(float(f4["bottom"]))
                    targets = [t for t in targets if (t - entry) >= min_rr * risk and (t - entry) <= 15 * risk]
                    if not targets:
                        continue
                    tp = min(targets)               # nearest opposing liquidity >= 2R
                    rr_val = (tp - entry) / risk
                else:
                    for key in ("PDL", "PWL", "PML", "PSL"):
                        v = elv.get(key)
                        if v is not None and v < entry:
                            targets.append(float(v))
                    for f4 in fvgs_4h:
                        if df_h4.index[f4["form_idx"]] > idx[cisd_idx]:
                            continue
                        if f4["top"] < entry:         # gap sits below → first touch = top
                            targets.append(float(f4["top"]))
                    targets = [t for t in targets if (entry - t) >= min_rr * risk and (entry - t) <= 15 * risk]
                    if not targets:
                        continue
                    tp = max(targets)
                    rr_val = (entry - tp) / risk

            else:
                rr_val = rr
                tp = entry + rr * risk if direction == "long" else entry - rr * risk

            result = simulate(h, l, idx, cisd_idx, direction, entry, sl, tp, rr_val)
            open_until.append(pd.Timestamp(result["exit_time"]))
            trades.append({
                "symbol": symbol, "tf": tf,
                "sweep_time": sweep["time"], "sweep_bias": bias,
                "swept_level": round(swept_level, 6),
                "n_confluent_levels": len(cand),
                "confluence_via": "level" if has_level_confluence else "4h_fvg",
                "deepest_level": round(float(deepest), 6),
                "deep_extreme": round(float(deep_extreme), 6),
                "deep_time": idx[deep_i],
                "fvg_time": idx[f_idx], "fvg_top": round(fvg["top"], 6),
                "fvg_bottom": round(fvg["bottom"], 6), "fvg_width": round(fvg["width"], 6),
                "ifvg_time": idx[ifvg_idx], "cisd_time": idx[cisd_idx],
                "entry_time": idx[cisd_idx], "entry_idx": cisd_idx,
                "direction": direction, "entry": round(entry, 6),
                "sl": round(sl, 6), "tp": round(tp, 6), "rr_target": round(rr_val, 2),
                "outcome": result["outcome"],
                "exit": round(result["exit"], 6) if not np.isnan(result["exit"]) else np.nan,
                "exit_time": result["exit_time"], "pnl_r": float(result["pnl_r"]),
            })
            used.add(sig)
            trade_done = True
            break
        # next sweep

    return pd.DataFrame(trades)


def streak_dd(df, initial=10_000, risk_pct=1.0):
    bal = initial; eq = [bal]; cl = cw = bcl = bcw = 0
    for _, row in df.sort_values("entry_time").iterrows():
        bal += bal * (risk_pct / 100) * float(row["pnl_r"]); eq.append(bal)
        if row["outcome"] == "loss": cl += 1; cw = 0; bcl = max(bcl, cl)
        elif row["outcome"] == "win": cw += 1; cl = 0; bcw = max(bcw, cw)
    eq = np.array(eq)
    dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq) * 100).min()
    return bal, dd, bcl, bcw


def summarize(all_df, label):
    closed = all_df[all_df["outcome"] != "open"]
    if not len(closed):
        print(f"  {label:44}  no trades"); return None
    n = len(closed)
    wr = (closed["outcome"] == "win").sum() / n * 100
    r = closed["pnl_r"].sum()
    gw = closed.loc[closed["outcome"] == "win", "pnl_r"].sum()
    gl = -closed.loc[closed["outcome"] == "loss", "pnl_r"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    bal, dd, ls, ws = streak_dd(closed)
    avg_rr = closed["rr_target"].mean()
    print(f"  {label:44}  N={n:4d}  WR={wr:5.1f}%  R={r:+7.1f}  PF={pf:4.2f}  "
          f"AvgRR={avg_rr:4.2f}  DD={dd:5.1f}%  LStrk={ls:2d}  ${bal:>8,.0f}")
    return {"label": label, "df": all_df, "n": n, "wr": wr, "r": r, "pf": pf,
            "dd": dd, "ls": ls, "bal": bal}


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    OUT = os.path.join("strategies", "confluence_sweep")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 116)
    print("4H SWEEP + CONFLUENCE LEVEL  →  IFVG + CISD   ·  7 FX majors  ·  news-blocked  ·  candle-by-candle")
    print("=" * 116)

    configs = [
        ("M15 · Fixed 2R · confluence ON",     dict(tf="M15", tp_mode="FIXED",   rr=2.0, require_confluence=True)),
        ("M15 · DOL(swings) · confluence ON",  dict(tf="M15", tp_mode="DOL",     min_rr=2.0, require_confluence=True)),
        ("M15 · OPP-LIQ · confluence ON",      dict(tf="M15", tp_mode="OPP_LIQ", min_rr=2.0, require_confluence=True)),
        ("M5  · Fixed 2R · confluence ON",     dict(tf="M5",  tp_mode="FIXED",   rr=2.0, require_confluence=True)),
        ("M5  · OPP-LIQ · confluence ON",      dict(tf="M5",  tp_mode="OPP_LIQ", min_rr=2.0, require_confluence=True)),
        ("M15 · OPP-LIQ · confluence OFF",     dict(tf="M15", tp_mode="OPP_LIQ", min_rr=2.0, require_confluence=False)),
    ]

    all_results = []
    for label, kw in configs:
        combined = []
        for s in MAJORS:
            try:
                df = run(s, **kw)
                if len(df):
                    combined.append(df)
            except Exception as e:
                print(f"    [!] {s} failed: {e}")
        if not combined:
            print(f"  {label:44}  no trades")
            continue
        all_df = pd.concat(combined, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
        res = summarize(all_df, label)
        if res:
            all_results.append(res)
            safe = label.replace(" ", "_").replace("·", "_").replace("__", "_")
            all_df.to_csv(os.path.join(OUT, f"trades_{safe}.csv"), index=False)

    if not all_results:
        print("\nNo trades in any config.")
        mt5.shutdown()
        return

    best = max(all_results, key=lambda r: r["r"])
    print(f"\n>>> BEST BY R: {best['label']}")

    # Per-pair on best
    bdf = best["df"]
    print("\n  Per-pair (best config):")
    for sym, g in bdf[bdf["outcome"] != "open"].groupby("symbol"):
        wr = (g["outcome"] == "win").sum() / len(g) * 100
        print(f"    {sym:8}: N={len(g):3d}  WR={wr:5.1f}%  R={g['pnl_r'].sum():+6.1f}")

    # Per-year
    bdf2 = bdf.copy()
    bdf2["year"] = pd.to_datetime(bdf2["entry_time"]).dt.year
    print("\n  Per-year (best config):")
    for yr, g in bdf2[bdf2["outcome"] != "open"].groupby("year"):
        wr = (g["outcome"] == "win").sum() / len(g) * 100
        print(f"    {yr}: N={len(g):3d}  WR={wr:5.1f}%  R={g['pnl_r'].sum():+6.1f}")

    pu.plot_equity(bdf[bdf["outcome"] != "open"], os.path.join(OUT, "equity_best.png"),
                   f"4H + Confluence · {best['label']}")

    # Sample plot: biggest win
    closed = bdf[bdf["outcome"] != "open"]
    wins = closed[closed["outcome"] == "win"].sort_values("pnl_r", ascending=False)
    if len(wins):
        t = wins.iloc[0]
        tf_const = mt5.TIMEFRAME_M5 if t["tf"] == "M5" else mt5.TIMEFRAME_M15
        dfx = bt.fetch_data(t["symbol"], tf_const, bt.START_DATE, bt.END_DATE, tf_name=t["tf"])
        pu.plot_recent_trade(dfx, t.to_dict(), os.path.join(OUT, "biggest_win.png"),
                             title_prefix="CONFLUENCE BIGGEST WIN", symbol=t["symbol"])
        print(f"\n  Biggest win: {t['symbol']} {t['direction']} {t['pnl_r']:+.1f}R  → {OUT}/biggest_win.png")

    print(f"\nSaved outputs to {OUT}/")
    mt5.shutdown()


if __name__ == "__main__":
    main()
