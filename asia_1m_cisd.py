"""
Asia sweep + 1M IFVG + CISD entry — London-session only, EURUSD.

Rules (strictly mechanical, candle-by-candle simulation):

  1. ASIA RANGE = high/low of the 00:00 - 07:00 broker-time candles per day.

  2. LONDON WINDOW = 07:00 - 10:00 broker time. All sweep detection and
     entry activity is confined to this window. No pre-positioning.

  3. On the 1-MINUTE chart, look for the FIRST sweep of Asia High or Low
     within the London window:
        Asia HIGH swept (wick > AsiaH, close < AsiaH)  →  SHORT bias
        Asia LOW  swept (wick < AsiaL, close > AsiaL)  →  LONG  bias
     Record the sweep candle index and the sweep extreme (the wick high/low).

  4. IFVG on 1M — after the sweep, find the first FVG that INVERTS in the
     bias direction:
        SHORT bias  →  a BULLISH FVG (formed on the rally into the sweep)
                       that gets CLOSED THROUGH BEARISH — a 1M candle
                       closes BELOW the FVG bottom. That inverts the bullish
                       FVG into a bearish IFVG (overhead resistance).
        LONG  bias  →  a BEARISH FVG (formed on the drop into the sweep low)
                       that gets CLOSED THROUGH BULLISH — a 1M candle
                       closes ABOVE the FVG top. That inverts the bearish
                       FVG into a bullish IFVG (support).
     IFVG must occur within IFVG_WINDOW_MIN 1M candles of the sweep.

  5. CISD on 1M — after the IFVG inversion, look for a Change In State of
     Delivery: the first 1M candle whose CLOSE breaks the previous swing
     structure in the bias direction:
        SHORT bias → close BELOW the min low of the previous CISD_LOOKBACK candles
        LONG  bias → close ABOVE the max high of the previous CISD_LOOKBACK candles
     Must fire within CISD_WINDOW candles of the IFVG. Also must still be in
     the London window (before 10:00).

  6. ENTRY = the CLOSE of the CISD candle.

  7. STOP LOSS = just beyond the swept extreme (the Asia H/L wick) + buffer.
        SHORT → SL = sweep_extreme + SL_BUFFER * risk
        LONG  → SL = sweep_extreme - SL_BUFFER * risk

  8. TAKE PROFIT = 2R.

  9. Only one setup per day maximum. Simulation is candle-by-candle from the
     CISD onward (SL is checked first each bar, then TP).

Backtest: EURUSD M1 (last ~99 days available from MT5).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MetaTrader5 as mt5

import main as bt


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
SYMBOL              = "EURUSD"
ASIA_START_HOUR     = 0    # broker time
ASIA_END_HOUR       = 7
LONDON_START_HOUR   = 7
LONDON_END_HOUR     = 10   # entries must fire before 10:00 broker
MIN_FVG_WIDTH       = 0.00005   # 0.5 pip minimum (M1 gaps are tiny)
IFVG_WINDOW_MIN     = 60        # 60 x 1M = 1 hour to find IFVG after sweep
CISD_WINDOW         = 30        # 30 x 1M = 30 minutes for CISD after IFVG
CISD_LOOKBACK       = 5         # bars to reference for the structural break
SL_BUFFER           = 0.10      # 10% buffer beyond the sweep extreme
MIN_SL_PIPS         = 0.0       # minimum absolute SL distance (in price units, e.g. 0.0003 = 3 pips)
RR                  = 2.0


# External CSV path (UTC timestamps). If present, used instead of MT5.
# The CSV is UTC/GMT+0 → London killzone 07:00-10:00 GMT lines up directly
# with the raw timestamps (textbook ICT session alignment).
EXTERNAL_CSV = os.path.join("data", "EURUSD_M1.csv")


def fetch_m1(symbol: str, use_cache: bool = True) -> pd.DataFrame:
    """Load M1 EURUSD. Prefers the external UTC CSV; falls back to MT5."""
    if os.path.exists(EXTERNAL_CSV):
        print(f"  [csv] loading external {EXTERNAL_CSV} (UTC)")
        df = pd.read_csv(EXTERNAL_CSV, header=None,
                         names=["time", "open", "high", "low", "close", "vol"])
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")[["open", "high", "low", "close"]].sort_index()
        # drop dupes if any
        df = df[~df.index.duplicated(keep="first")]
        return df

    cache_path = os.path.join("data_cache", f"{symbol}_M1.pkl")
    os.makedirs("data_cache", exist_ok=True)
    if use_cache and os.path.exists(cache_path):
        print(f"  [cache] loading {cache_path}")
        return pd.read_pickle(cache_path)
    mt5.symbol_select(symbol, True)
    bars = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 99_999)
    if bars is None or not len(bars):
        raise RuntimeError(f"No M1 data for {symbol}")
    df = pd.DataFrame(bars)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")[["open", "high", "low", "close"]]
    df.to_pickle(cache_path)
    print(f"  [cache] saved {cache_path} ({len(df):,} rows)")
    return df


def find_fvg_backward(h, l, up_to_idx, direction, min_width):
    """
    Scan BACKWARD from up_to_idx for the most recent completed FVG of
    the given direction. Returns dict or None.
    """
    # FVG completes at index k with candles [k-2, k-1, k]
    for k in range(up_to_idx, 2, -1):
        if direction == "bullish":
            if l[k] > h[k - 2]:
                w = float(l[k] - h[k - 2])
                if w >= min_width:
                    return {"direction": "bullish", "top": float(l[k]),
                            "bottom": float(h[k - 2]), "width": w, "form_idx": k}
        else:
            if h[k] < l[k - 2]:
                w = float(l[k - 2] - h[k])
                if w >= min_width:
                    return {"direction": "bearish", "top": float(l[k - 2]),
                            "bottom": float(h[k]), "width": w, "form_idx": k}
    return None


def simulate(h, l, idx, entry_i, direction, entry, sl, tp, rr_value, max_bars=1200):
    """SL first then TP each bar."""
    for j in range(entry_i + 1, min(entry_i + max_bars, len(h))):
        if direction == "long":
            if l[j] <= sl:  return {"outcome": "loss", "exit": sl, "exit_time": idx[j], "pnl_r": -1.0}
            if h[j] >= tp:  return {"outcome": "win",  "exit": tp, "exit_time": idx[j], "pnl_r":  rr_value}
        else:
            if h[j] >= sl:  return {"outcome": "loss", "exit": sl, "exit_time": idx[j], "pnl_r": -1.0}
            if l[j] <= tp:  return {"outcome": "win",  "exit": tp, "exit_time": idx[j], "pnl_r":  rr_value}
    return {"outcome": "open", "exit": float("nan"), "exit_time": idx[-1], "pnl_r": 0.0}


def _swings(h, l, n=5):
    sh = np.full(len(h), np.nan)
    slo = np.full(len(l), np.nan)
    for i in range(n, len(h) - n):
        if h[i] >= h[i - n:i + n + 1].max():
            sh[i] = h[i]
        if l[i] <= l[i - n:i + n + 1].min():
            slo[i] = l[i]
    return sh, slo


def run(df_1m: pd.DataFrame,
        sl_buffer: float = SL_BUFFER,
        min_sl_pips: float = MIN_SL_PIPS,
        rr: float = RR,
        mode: str = "IFVG_CISD",
        tp_mode: str = "FIXED",   # "FIXED" or "DOL"
        min_rr: float = 2.0,
        dol_swing_lb: int = 10) -> pd.DataFrame:
    h = df_1m["high"].values
    l = df_1m["low"].values
    c = df_1m["close"].values
    o = df_1m["open"].values
    idx = df_1m.index
    hours = idx.hour
    dates = idx.normalize()

    # Precompute 1M swing highs/lows for DOL targets
    if tp_mode == "DOL":
        sh_arr, sl_arr = _swings(h, l, n=dol_swing_lb)
    else:
        sh_arr = sl_arr = None

    trades = []
    unique_days, day_starts = np.unique(dates, return_index=True)
    day_ends = np.append(day_starts[1:], len(df_1m))

    for di_day, day in enumerate(unique_days):
        a, b = day_starts[di_day], day_ends[di_day]
        day_h = hours[a:b]

        # ── Asia range (00:00 - 07:00) ──
        asia_mask = (day_h >= ASIA_START_HOUR) & (day_h < ASIA_END_HOUR)
        if not asia_mask.any():
            continue
        asia_hi = h[a:b][asia_mask].max()
        asia_lo = l[a:b][asia_mask].min()

        # ── London window (07:00 - 10:00) ──
        lon_mask = (day_h >= LONDON_START_HOUR) & (day_h < LONDON_END_HOUR)
        lon_local = np.where(lon_mask)[0]
        if not len(lon_local):
            continue

        # ── Find FIRST sweep within the London window ──
        sweep_dir = None
        sweep_i = None
        sweep_extreme = None
        for j in lon_local:
            i = a + j
            if h[i] > asia_hi and c[i] < asia_hi:
                sweep_dir = "short"
                sweep_i = i
                sweep_extreme = h[i]
                break
            if l[i] < asia_lo and c[i] > asia_lo:
                sweep_dir = "long"
                sweep_i = i
                sweep_extreme = l[i]
                break
        if sweep_dir is None:
            continue

        # ── Confirmation phase — depends on `mode` ──
        # mode "IFVG_CISD" : sweep -> IFVG inversion -> CISD -> enter at CISD
        # mode "CISD_ONLY" : sweep -> CISD -> enter at CISD  (no FVG requirement)
        # mode "IFVG_ONLY" : sweep -> IFVG inversion -> enter at inversion candle
        fvg = None
        ifvg_idx = None
        cisd_idx = None
        scan_end = min(sweep_i + IFVG_WINDOW_MIN, len(df_1m) - 1)

        if mode in ("IFVG_CISD", "IFVG_ONLY"):
            # Find the FVG that formed on the move INTO the sweep
            target_fvg_dir = "bullish" if sweep_dir == "short" else "bearish"
            fvg_scan_lo = max(2, sweep_i - 60)
            for k in range(sweep_i, fvg_scan_lo - 1, -1):
                if target_fvg_dir == "bullish":
                    if l[k] > h[k - 2]:
                        w = float(l[k] - h[k - 2])
                        if w >= MIN_FVG_WIDTH:
                            fvg = {"direction": "bullish", "top": float(l[k]),
                                   "bottom": float(h[k - 2]), "width": w, "form_idx": k}
                            break
                else:
                    if h[k] < l[k - 2]:
                        w = float(l[k - 2] - h[k])
                        if w >= MIN_FVG_WIDTH:
                            fvg = {"direction": "bearish", "top": float(l[k - 2]),
                                   "bottom": float(h[k]), "width": w, "form_idx": k}
                            break
            if fvg is None:
                continue
            # Forward scan for the inversion candle
            for k in range(sweep_i + 1, scan_end + 1):
                if sweep_dir == "short":
                    if c[k] < fvg["bottom"]:
                        ifvg_idx = k
                        break
                else:
                    if c[k] > fvg["top"]:
                        ifvg_idx = k
                        break
            if ifvg_idx is None:
                continue

        if mode == "IFVG_ONLY":
            # Entry directly on the inversion candle
            entry_i = ifvg_idx
        elif mode == "CISD_ONLY":
            # CISD scanned from sweep forward (no FVG gating)
            cisd_start = sweep_i + 1
            cisd_end = min(sweep_i + IFVG_WINDOW_MIN, len(df_1m) - 1)
            for k in range(cisd_start, cisd_end + 1):
                if k - CISD_LOOKBACK < 0:
                    continue
                if sweep_dir == "short":
                    ref = l[k - CISD_LOOKBACK:k].min()
                    if c[k] < ref:
                        cisd_idx = k
                        break
                else:
                    ref = h[k - CISD_LOOKBACK:k].max()
                    if c[k] > ref:
                        cisd_idx = k
                        break
            if cisd_idx is None:
                continue
            entry_i = cisd_idx
        else:  # IFVG_CISD
            cisd_end = min(ifvg_idx + CISD_WINDOW, len(df_1m) - 1)
            for k in range(ifvg_idx + 1, cisd_end + 1):
                if k - CISD_LOOKBACK < 0:
                    continue
                if sweep_dir == "short":
                    ref = l[k - CISD_LOOKBACK:k].min()
                    if c[k] < ref:
                        cisd_idx = k
                        break
                else:
                    ref = h[k - CISD_LOOKBACK:k].max()
                    if c[k] > ref:
                        cisd_idx = k
                        break
            if cisd_idx is None:
                continue
            entry_i = cisd_idx

        # ── Enforce London-window constraint on entry ──
        if hours[entry_i] < LONDON_START_HOUR or hours[entry_i] >= LONDON_END_HOUR:
            continue

        # ── Entry + SL ──
        entry = c[entry_i]
        if sweep_dir == "short":
            direction = "short"
            base_risk = sweep_extreme - entry
            sl = sweep_extreme + abs(base_risk) * sl_buffer
            if sl - entry < min_sl_pips:
                sl = entry + min_sl_pips
            risk = sl - entry
        else:
            direction = "long"
            base_risk = entry - sweep_extreme
            sl = sweep_extreme - abs(base_risk) * sl_buffer
            if entry - sl < min_sl_pips:
                sl = entry - min_sl_pips
            risk = entry - sl
        if risk <= 0:
            continue

        # ── TP ──
        if tp_mode == "DOL":
            # Next opposing liquidity pool from recent 1M swings.
            # long  → nearest swing HIGH above entry ; short → swing LOW below.
            if direction == "long":
                cands = [sh_arr[k] for k in range(max(0, entry_i - 400), entry_i)
                         if not np.isnan(sh_arr[k]) and sh_arr[k] > entry]
                dol = min(cands) if cands else None
                if dol is None:
                    continue
                tp = dol
                rr_val = (tp - entry) / risk
            else:
                cands = [sl_arr[k] for k in range(max(0, entry_i - 400), entry_i)
                         if not np.isnan(sl_arr[k]) and sl_arr[k] < entry]
                dol = max(cands) if cands else None
                if dol is None:
                    continue
                tp = dol
                rr_val = (entry - tp) / risk
            if rr_val < min_rr or rr_val > 15:
                continue
        else:
            rr_val = rr
            tp = entry - rr * risk if direction == "short" else entry + rr * risk

        result = simulate(h, l, idx, entry_i, direction, entry, sl, tp, rr_val)
        trades.append({
            "date":         pd.Timestamp(day).date(),
            "mode":         mode,
            "rr_target":    round(rr_val, 2),
            "asia_high":    round(float(asia_hi), 6),
            "asia_low":     round(float(asia_lo), 6),
            "sweep_time":   idx[sweep_i],
            "sweep_dir":    sweep_dir,
            "sweep_extreme": round(float(sweep_extreme), 6),
            "fvg_time":     idx[fvg["form_idx"]] if fvg else pd.NaT,
            "fvg_top":      round(fvg["top"], 6) if fvg else np.nan,
            "fvg_bottom":   round(fvg["bottom"], 6) if fvg else np.nan,
            "fvg_width":    round(fvg["width"], 6) if fvg else np.nan,
            "ifvg_time":    idx[ifvg_idx] if ifvg_idx is not None else pd.NaT,
            "cisd_time":    idx[cisd_idx] if cisd_idx is not None else pd.NaT,
            "entry_idx":    entry_i,
            "direction":    direction,
            "entry":        round(entry, 6),
            "sl":           round(sl, 6),
            "tp":           round(tp, 6),
            "outcome":      result["outcome"],
            "exit":         round(result["exit"], 6) if not np.isnan(result["exit"]) else np.nan,
            "exit_time":    result["exit_time"],
            "pnl_r":        float(result["pnl_r"]),
            "entry_time":   idx[entry_i],
            "symbol":       SYMBOL,
        })

    return pd.DataFrame(trades)


def streak_dd(df, initial=10_000, risk_pct=1.0):
    bal = initial
    eq = [bal]
    cl = cw = best_cl = best_cw = 0
    for _, row in df.sort_values("entry_time").iterrows():
        bal += bal * (risk_pct / 100) * float(row["pnl_r"])
        eq.append(bal)
        if row["outcome"] == "loss":
            cl += 1
            cw = 0
            best_cl = max(best_cl, cl)
        elif row["outcome"] == "win":
            cw += 1
            cl = 0
            best_cw = max(best_cw, cw)
    eq_arr = np.array(eq)
    dd = ((eq_arr - np.maximum.accumulate(eq_arr)) / np.maximum.accumulate(eq_arr) * 100).min()
    return bal, dd, best_cl, best_cw, eq_arr


def report(df: pd.DataFrame):
    OUT = os.path.join("strategies", "asia_1m_cisd")
    os.makedirs(OUT, exist_ok=True)
    if not len(df):
        print("No trades.")
        return
    df.to_csv(os.path.join(OUT, "trades.csv"), index=False)

    closed = df[df["outcome"] != "open"].copy()
    open_n = len(df[df["outcome"] == "open"])
    n = len(closed)
    n_win = (closed["outcome"] == "win").sum()
    n_loss = (closed["outcome"] == "loss").sum()
    wr = n_win / n * 100 if n else 0
    total_r = closed["pnl_r"].sum()
    gw = closed.loc[closed["outcome"] == "win", "pnl_r"].sum()
    gl = -closed.loc[closed["outcome"] == "loss", "pnl_r"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    bal, dd, ls, ws, eq = streak_dd(closed)
    expectancy = closed["pnl_r"].mean()
    period_start = pd.to_datetime(closed["entry_time"]).min()
    period_end   = pd.to_datetime(closed["entry_time"]).max()
    days = (period_end - period_start).days + 1
    years = days / 365.25
    long_df = closed[closed["direction"] == "long"]
    short_df = closed[closed["direction"] == "short"]
    lwr = (long_df["outcome"] == "win").sum() / len(long_df) * 100 if len(long_df) else 0
    swr = (short_df["outcome"] == "win").sum() / len(short_df) * 100 if len(short_df) else 0

    # Time in trade
    closed["dur_min"] = (pd.to_datetime(closed["exit_time"]) - pd.to_datetime(closed["entry_time"])).dt.total_seconds() / 60

    print("\n" + "=" * 92)
    print(f"ASIA SWEEP + 1M IFVG + CISD  ·  EURUSD  ·  London window {LONDON_START_HOUR:02d}:00-{LONDON_END_HOUR:02d}:00 broker")
    print("=" * 92)
    print(f"  Period                : {period_start} → {period_end}  ({days} days, {years:.2f} yr)")
    print()
    print(f"  Total trades          : {n}   (open at end: {open_n})")
    print(f"  Wins / Losses         : {n_win} / {n_loss}")
    print(f"  Win rate              : {wr:.2f}%")
    print(f"  Expectancy per trade  : {expectancy:+.3f}R")
    print(f"  Total R               : {total_r:+.1f}R")
    print(f"  Profit factor         : {pf:.2f}")
    print()
    print(f"  Longest WIN streak    : {ws}")
    print(f"  Longest LOSS streak   : {ls}")
    print(f"  Max drawdown (1% risk): {dd:.2f}%")
    print(f"  End balance ($10k)    : ${bal:,.0f}  ({(bal/10000-1)*100:+.1f}%)")
    print()
    print(f"  Trades per day        : {n/days:.2f}")
    print(f"  Trades per week       : {n/days*7:.1f}")
    print(f"  Trades per month      : {n/days*30:.1f}")
    print(f"  Avg trade duration    : {closed['dur_min'].mean():.1f} min")
    print(f"  Median trade duration : {closed['dur_min'].median():.1f} min")
    print(f"  Longest trade         : {closed['dur_min'].max():.1f} min")
    print()
    print(f"  LONG  : N={len(long_df):3d}  WR={lwr:5.1f}%  R={long_df['pnl_r'].sum():+6.1f}")
    print(f"  SHORT : N={len(short_df):3d}  WR={swr:5.1f}%  R={short_df['pnl_r'].sum():+6.1f}")

    # Save equity chart
    from plot_utils import plot_equity
    plot_equity(closed, os.path.join(OUT, "equity.png"),
                f"Asia sweep + 1M IFVG + CISD · EURUSD · London window · RR={RR}")
    print(f"\nTrade log : {OUT}/trades.csv")
    print(f"Equity    : {OUT}/equity.png")


def summarize(trades: pd.DataFrame, label: str):
    if not len(trades):
        print(f"  {label:38}  no trades")
        return
    closed = trades[trades["outcome"] != "open"]
    if not len(closed):
        print(f"  {label:38}  no closed trades")
        return
    n = len(closed)
    wins = (closed["outcome"] == "win").sum()
    wr = wins / n * 100
    r = closed["pnl_r"].sum()
    gw = closed.loc[closed["outcome"] == "win", "pnl_r"].sum()
    gl = -closed.loc[closed["outcome"] == "loss", "pnl_r"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    bal, dd, ls, ws, _ = streak_dd(closed)
    lng = closed[closed["direction"] == "long"]
    sht = closed[closed["direction"] == "short"]
    lwr = (lng["outcome"] == "win").sum() / len(lng) * 100 if len(lng) else 0
    swr = (sht["outcome"] == "win").sum() / len(sht) * 100 if len(sht) else 0
    print(f"  {label:38}  N={n:3d}  WR={wr:5.1f}%  R={r:+6.1f}  "
          f"PF={pf:4.2f}  DD={dd:5.1f}%  LStreak={ls:2d}  "
          f"L-WR={lwr:4.1f}%  S-WR={swr:4.1f}%")


def plot_sample(df_1m, trade, out_path, mode):
    """Plot one trade with sweep / FVG / IFVG / CISD annotated per mode."""
    BG, FG = "#0d1117", "#e6edf3"
    GRN, RED, YLW = "#3fb950", "#f85149", "#d29922"
    BLU, PURP, ORG = "#58a6ff", "#bc8cff", "#ff9f43"
    UP, DN = "#26a69a", "#ef5350"

    entry_idx = int(trade["entry_idx"])
    exit_idx  = df_1m.index.get_indexer([pd.Timestamp(trade["exit_time"])], method="nearest")[0]
    sweep_idx = df_1m.index.get_indexer([pd.Timestamp(trade["sweep_time"])], method="nearest")[0]

    lo = max(0, sweep_idx - 20)
    hi = min(len(df_1m), exit_idx + 15)
    win = df_1m.iloc[lo:hi].copy().reset_index()
    win.rename(columns={win.columns[0]: "time"}, inplace=True)

    ex = entry_idx - lo
    xx = exit_idx - lo
    sx = sweep_idx - lo
    is_short = trade["direction"] == "short"
    entry, sl, tp = float(trade["entry"]), float(trade["sl"]), float(trade["tp"])

    fig, ax = plt.subplots(figsize=(18, 10), facecolor=BG)
    ax.set_facecolor(BG)

    for k, row in win.iterrows():
        col = UP if row["close"] >= row["open"] else DN
        ax.plot([k, k], [row["low"], row["high"]], color=col, linewidth=0.8, zorder=2)
        body_lo = min(row["open"], row["close"])
        hh = abs(row["close"] - row["open"])
        if hh == 0:
            hh = (win["high"].max() - win["low"].min()) * 0.0004
        ax.add_patch(plt.Rectangle((k - 0.35, body_lo), 0.7, hh,
                                   facecolor=col, edgecolor=col, zorder=3))

    # Asia level swept
    ax.axhline(float(trade["sweep_extreme"]), color=BLU, linewidth=1.2, linestyle="--", alpha=0.8)
    swept_lbl = "Asia HIGH swept" if is_short else "Asia LOW swept"
    ax.text(0.5, float(trade["sweep_extreme"]), f"  {swept_lbl}",
            color=BLU, fontsize=9, va="bottom", fontweight="bold")
    if 0 <= sx < len(win):
        ax.axvline(sx, color=BLU, linestyle=":", alpha=0.4)
        ax.text(sx, win["high"].max(), "  Sweep", color=BLU, fontsize=9,
                rotation=90, va="top", fontweight="bold")

    # FVG zone (if present)
    if pd.notna(trade.get("fvg_top")):
        ft, fb = float(trade["fvg_top"]), float(trade["fvg_bottom"])
        ax.axhspan(fb, ft, color=PURP, alpha=0.20, zorder=1)
        ax.axhline(ft, color=PURP, linewidth=0.7, linestyle=":", alpha=0.7)
        ax.axhline(fb, color=PURP, linewidth=0.7, linestyle=":", alpha=0.7)
        ax.text(0.5, ft, "  FVG → IFVG zone", color=PURP, fontsize=9, va="bottom", fontweight="bold")

    # IFVG marker
    if pd.notna(trade.get("ifvg_time")):
        ii = df_1m.index.get_indexer([pd.Timestamp(trade["ifvg_time"])], method="nearest")[0] - lo
        if 0 <= ii < len(win):
            ax.axvline(ii, color=PURP, linestyle="--", alpha=0.5)
            ax.text(ii, win["low"].min(), " IFVG", color=PURP, fontsize=9,
                    rotation=90, va="bottom", fontweight="bold")

    # CISD marker
    if pd.notna(trade.get("cisd_time")):
        ci = df_1m.index.get_indexer([pd.Timestamp(trade["cisd_time"])], method="nearest")[0] - lo
        if 0 <= ci < len(win):
            ax.axvline(ci, color=ORG, linestyle="--", alpha=0.6)
            ax.text(ci, win["high"].max(), " CISD", color=ORG, fontsize=9,
                    rotation=90, va="top", fontweight="bold")

    # Position boxes
    box_w = max(xx - ex, 6)
    tl, th = sorted([entry, tp])
    ax.add_patch(plt.Rectangle((ex, tl), box_w, th - tl, facecolor=GRN, alpha=0.14,
                               edgecolor=GRN, zorder=4))
    stl, sth = sorted([entry, sl])
    ax.add_patch(plt.Rectangle((ex, stl), box_w, sth - stl, facecolor=RED, alpha=0.14,
                               edgecolor=RED, zorder=4))

    span = max(win["high"].max() - win["low"].min(), 1e-6)
    min_gap = span * 0.03
    tp_lbl = f"TP ({trade.get('rr_target', 2):.1f}R) {tp}"
    lvls = sorted([(entry, FG, f"Entry {entry}"),
                   (tp, GRN, tp_lbl),
                   (sl, RED, f"SL {sl}")], key=lambda x: -x[0])
    last_y = None
    for price, col, lbl in lvls:
        ax.axhline(price, color=col, linewidth=1.0, alpha=0.85, zorder=5)
        y = price
        if last_y is not None and (last_y - y) < min_gap:
            y = last_y - min_gap
        last_y = y
        ax.text(ex + box_w + 0.5, y, f"  {lbl}", color=col, fontsize=10, va="center", fontweight="bold")

    ax.scatter([ex], [entry], marker="v" if is_short else "^", s=200, color=YLW,
               edgecolor="black", zorder=7, label="Entry")
    ec = GRN if trade["outcome"] == "win" else RED
    ax.scatter([xx], [float(trade["exit"])], marker="o", s=170, color=ec,
               edgecolor="black", zorder=7, label=f"Exit ({trade['outcome'].upper()})")

    step = max(1, len(win) // 10)
    ticks = list(range(0, len(win), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([win["time"].iloc[t].strftime("%m-%d %H:%M") for t in ticks],
                       rotation=30, ha="right", fontsize=9, color=FG)

    dur = (pd.to_datetime(trade["exit_time"]) - pd.to_datetime(trade["entry_time"])).total_seconds() / 60
    ax.set_title(
        f"ASIA SWEEP · {mode}  |  EURUSD 1M  |  {trade['direction'].upper()}  |  "
        f"{trade['outcome'].upper()} ({trade['pnl_r']:+.1f}R)  |  "
        f"{pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d %H:%M')}  |  {dur:.0f} min",
        color=FG, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Price", color=FG)
    ax.tick_params(colors=FG)
    ax.grid(True, alpha=0.10, color=FG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.legend(facecolor="#161b22", edgecolor="#444c56", labelcolor=FG, loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main():
    if not os.path.exists(EXTERNAL_CSV):
        if not mt5.initialize():
            raise RuntimeError("MT5 init failed")
    df_1m = fetch_m1(SYMBOL)
    print(f"  M1 range: {df_1m.index[0]} → {df_1m.index[-1]}  ({len(df_1m):,} bars)")
    OUT = os.path.join("strategies", "asia_1m_cisd")
    os.makedirs(OUT, exist_ok=True)

    SL_KW = dict(sl_buffer=0.10, min_sl_pips=0.0003)
    dstart, dend = df_1m.index[0], df_1m.index[-1]
    days = (dend - dstart).days + 1

    modes = [
        ("IFVG_CISD",  "Sweep + IFVG + CISD"),
        ("CISD_ONLY",  "Sweep + CISD only"),
        ("IFVG_ONLY",  "Sweep + IFVG only"),
    ]
    tp_variants = [
        ("FIXED", "Fixed 2R"),
        ("DOL",   "DOL target (≥2R)"),
    ]

    print()
    print("=" * 110)
    print(f"ASIA SWEEP · EURUSD 1M (UTC) · London 07-10 · {dstart.date()} → {dend.date()} ({days} days)")
    print("Fixed-2R  vs  DOL-target  across all 3 confirmation modes")
    print("=" * 110)
    print(f"  {'Mode':22}  {'TP':16}  {'N':>3}  {'WR':>6}  {'R':>7}  {'PF':>5}  "
          f"{'AvgRR':>6}  {'DD':>6}  {'LStrk':>5}  {'$10k→':>9}")

    results = {}
    for mkey, mlabel in modes:
        for tkey, tlabel in tp_variants:
            trades = run(df_1m, mode=mkey, tp_mode=tkey, min_rr=2.0, **SL_KW)
            results[(mkey, tkey)] = trades
            closed = trades[trades["outcome"] != "open"] if len(trades) else pd.DataFrame()
            if not len(closed):
                print(f"  {mlabel:22}  {tlabel:16}  no trades")
                continue
            n = len(closed)
            wr = (closed["outcome"] == "win").sum() / n * 100
            r = closed["pnl_r"].sum()
            gw = closed.loc[closed["outcome"] == "win", "pnl_r"].sum()
            gl = -closed.loc[closed["outcome"] == "loss", "pnl_r"].sum()
            pf = gw / gl if gl > 0 else float("inf")
            avg_rr = closed["rr_target"].mean()
            bal, dd, ls, ws, _ = streak_dd(closed)
            print(f"  {mlabel:22}  {tlabel:16}  {n:3d}  {wr:5.1f}%  {r:+6.1f}  {pf:5.2f}  "
                  f"{avg_rr:6.2f}  {dd:5.1f}%  {ls:>5}  ${bal:>8,.0f}")

            trades.to_csv(os.path.join(OUT, f"trades_{mkey}_{tkey}.csv"), index=False)

    # Plot DOL sample for each mode (the new thing the user asked for)
    print()
    for mkey, mlabel in modes:
        tr = results.get((mkey, "DOL"))
        if tr is None or not len(tr):
            continue
        closed = tr[tr["outcome"] != "open"]
        if not len(closed):
            continue
        wins = closed[closed["outcome"] == "win"].sort_values("pnl_r", ascending=False)
        sample = wins.iloc[0] if len(wins) else closed.iloc[-1]
        plot_sample(df_1m, sample.to_dict(),
                    os.path.join(OUT, f"sample_{mkey}_DOL.png"), f"{mlabel} · DOL")
        print(f"  → sample_{mkey}_DOL.png  ({sample['date']} {sample['direction']} "
              f"{sample['pnl_r']:+.1f}R, target {sample['rr_target']:.1f}R)")

    try:
        mt5.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
