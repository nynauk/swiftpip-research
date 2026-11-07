"""
Realistic-conditions test on the Asia-sweep 1M CISD family (asia_1m_cisd.py).

Adds real trading costs to the same mechanical logic:
  • EURUSD spread 1.0 pip, slippage 0.3 pip
  • MT5/CSV OHLC = BID. LONG fills at ASK = close + spread + slippage.
    SHORT fills at BID with adverse slippage = close - slippage.
  • SHORT SL/TP triggers tested against ASK = bid + spread (adverse).

Prints naive vs realistic side-by-side for all 3 confirmation modes at
Fixed-2R (the DOL variants have too few trades to be meaningful).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import asia_1m_cisd as a1c


SPREAD   = 1.0 * 1e-4      # 1.0 pip
SLIPPAGE = 0.3 * 1e-4      # 0.3 pip


def run_costed(df_1m, mode, rr=2.0, sl_buffer=0.10, min_sl_pips=0.0003,
               realistic=True):
    """Asia sweep + (mode) + Fixed-RR, with optional realistic costs."""
    h = df_1m["high"].values
    l = df_1m["low"].values
    c = df_1m["close"].values
    idx = df_1m.index
    hours = idx.hour
    dates = idx.normalize()

    spread   = SPREAD   if realistic else 0.0
    slippage = SLIPPAGE if realistic else 0.0

    trades = []
    unique_days, day_starts = np.unique(dates, return_index=True)
    day_ends = np.append(day_starts[1:], len(df_1m))

    for di_day, day in enumerate(unique_days):
        a, b = day_starts[di_day], day_ends[di_day]
        day_h = hours[a:b]

        asia_mask = (day_h >= a1c.ASIA_START_HOUR) & (day_h < a1c.ASIA_END_HOUR)
        if not asia_mask.any():
            continue
        asia_hi = h[a:b][asia_mask].max()
        asia_lo = l[a:b][asia_mask].min()

        lon_mask = (day_h >= a1c.LONDON_START_HOUR) & (day_h < a1c.LONDON_END_HOUR)
        lon_local = np.where(lon_mask)[0]
        if not len(lon_local):
            continue

        # First sweep in London window
        sweep_dir = sweep_i = sweep_extreme = None
        for j in lon_local:
            i = a + j
            if h[i] > asia_hi and c[i] < asia_hi:
                sweep_dir, sweep_i, sweep_extreme = "short", i, h[i]; break
            if l[i] < asia_lo and c[i] > asia_lo:
                sweep_dir, sweep_i, sweep_extreme = "long", i, l[i]; break
        if sweep_dir is None:
            continue

        fvg = None
        ifvg_idx = None
        cisd_idx = None
        scan_end = min(sweep_i + a1c.IFVG_WINDOW_MIN, len(df_1m) - 1)

        if mode in ("IFVG_CISD", "IFVG_ONLY"):
            target_fvg_dir = "bullish" if sweep_dir == "short" else "bearish"
            fvg_scan_lo = max(2, sweep_i - 60)
            for k in range(sweep_i, fvg_scan_lo - 1, -1):
                if target_fvg_dir == "bullish":
                    if l[k] > h[k - 2]:
                        w = float(l[k] - h[k - 2])
                        if w >= a1c.MIN_FVG_WIDTH:
                            fvg = {"direction": "bullish", "top": float(l[k]),
                                   "bottom": float(h[k - 2]), "width": w, "form_idx": k}; break
                else:
                    if h[k] < l[k - 2]:
                        w = float(l[k - 2] - h[k])
                        if w >= a1c.MIN_FVG_WIDTH:
                            fvg = {"direction": "bearish", "top": float(l[k - 2]),
                                   "bottom": float(h[k]), "width": w, "form_idx": k}; break
            if fvg is None:
                continue
            for k in range(sweep_i + 1, scan_end + 1):
                if sweep_dir == "short" and c[k] < fvg["bottom"]:
                    ifvg_idx = k; break
                if sweep_dir == "long"  and c[k] > fvg["top"]:
                    ifvg_idx = k; break
            if ifvg_idx is None:
                continue

        if mode == "IFVG_ONLY":
            entry_i = ifvg_idx
        elif mode == "CISD_ONLY":
            cisd_end = min(sweep_i + a1c.IFVG_WINDOW_MIN, len(df_1m) - 1)
            for k in range(sweep_i + 1, cisd_end + 1):
                if k - a1c.CISD_LOOKBACK < 0:
                    continue
                if sweep_dir == "short" and c[k] < l[k - a1c.CISD_LOOKBACK:k].min():
                    cisd_idx = k; break
                if sweep_dir == "long"  and c[k] > h[k - a1c.CISD_LOOKBACK:k].max():
                    cisd_idx = k; break
            if cisd_idx is None:
                continue
            entry_i = cisd_idx
        else:  # IFVG_CISD
            cisd_end = min(ifvg_idx + a1c.CISD_WINDOW, len(df_1m) - 1)
            for k in range(ifvg_idx + 1, cisd_end + 1):
                if k - a1c.CISD_LOOKBACK < 0:
                    continue
                if sweep_dir == "short" and c[k] < l[k - a1c.CISD_LOOKBACK:k].min():
                    cisd_idx = k; break
                if sweep_dir == "long"  and c[k] > h[k - a1c.CISD_LOOKBACK:k].max():
                    cisd_idx = k; break
            if cisd_idx is None:
                continue
            entry_i = cisd_idx

        if hours[entry_i] < a1c.LONDON_START_HOUR or hours[entry_i] >= a1c.LONDON_END_HOUR:
            continue

        signal_close = c[entry_i]
        # Realistic fills: LONG at ask, SHORT at bid (adverse slippage)
        if sweep_dir == "short":
            direction = "short"
            real_entry = signal_close - slippage
            base_risk = sweep_extreme - real_entry
            sl = sweep_extreme + abs(base_risk) * sl_buffer
            if sl - real_entry < min_sl_pips:
                sl = real_entry + min_sl_pips
            risk = sl - real_entry
            if risk <= 0:
                continue
            tp = real_entry - rr * risk
        else:
            direction = "long"
            real_entry = signal_close + spread + slippage
            base_risk = real_entry - sweep_extreme
            sl = sweep_extreme - abs(base_risk) * sl_buffer
            if real_entry - sl < min_sl_pips:
                sl = real_entry - min_sl_pips
            risk = real_entry - sl
            if risk <= 0:
                continue
            tp = real_entry + rr * risk

        # Simulate with spread-adjusted SHORT triggers (bid OHLC)
        oc, xt, pr = _sim(h, l, idx, entry_i, direction, real_entry, sl, tp, rr, spread)
        trades.append({
            "date": pd.Timestamp(day).date(), "mode": mode,
            "sweep_time": idx[sweep_i], "sweep_dir": sweep_dir,
            "entry_time": idx[entry_i], "entry_idx": entry_i,
            "direction": direction, "entry": round(real_entry, 6),
            "sl": round(sl, 6), "tp": round(tp, 6),
            "risk_pips": round(risk / 1e-4, 2),
            "outcome": oc, "exit_time": xt, "pnl_r": pr,
        })
    return pd.DataFrame(trades)


def _sim(h, l, idx, ei, direction, entry, sl, tp, rr, spread, max_bars=1200):
    for j in range(ei + 1, min(ei + max_bars, len(h))):
        if direction == "long":
            if l[j] <= sl: return "loss", idx[j], -1.0
            if h[j] >= tp: return "win",  idx[j],  rr
        else:
            if h[j] + spread >= sl: return "loss", idx[j], -1.0
            if l[j] + spread <= tp: return "win",  idx[j],  rr
    return "open", idx[min(ei + max_bars - 1, len(h) - 1)], 0.0


def stat_line(df):
    cl = df[df["outcome"] != "open"] if len(df) else df
    if not len(cl):
        return None
    n = len(cl)
    wins = (cl["outcome"] == "win").sum()
    wr = wins / n * 100
    r = cl["pnl_r"].sum()
    gw = cl.loc[cl["pnl_r"] > 0, "pnl_r"].sum()
    gl = -cl.loc[cl["pnl_r"] < 0, "pnl_r"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    bal, dd, ls, ws, eq = a1c.streak_dd(cl)
    avg_risk = cl["risk_pips"].mean()
    return dict(n=n, wr=wr, r=r, pf=pf, dd=dd, ls=ls, bal=bal,
                avg_risk=avg_risk, eq=eq)


def main():
    if not os.path.exists(a1c.EXTERNAL_CSV):
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise RuntimeError("MT5 init failed")
    df_1m = a1c.fetch_m1(a1c.SYMBOL)
    days = (df_1m.index[-1] - df_1m.index[0]).days + 1
    OUT = os.path.join("strategies", "realistic_1m_cisd")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 104)
    print(f"1M CISD REALISTIC TEST · EURUSD · London 07-10 UTC · {df_1m.index[0].date()} "
          f"-> {df_1m.index[-1].date()} ({days} days)")
    print(f"Costs: spread {SPREAD/1e-4:.1f} pip + slippage {SLIPPAGE/1e-4:.1f} pip · "
          f"min SL 3 pip · Fixed 2R")
    print("=" * 104)
    print(f"  {'Mode':20}  {'':9} {'N':>4} {'WR':>6} {'PF':>5} {'R':>8} {'DD':>7} "
          f"{'LStrk':>5} {'avgSL':>6}  {'$10k->':>9}")

    modes = [("CISD_ONLY", "Sweep+CISD"),
             ("IFVG_ONLY", "Sweep+IFVG"),
             ("IFVG_CISD", "Sweep+IFVG+CISD")]

    keep = {}
    for mkey, mlabel in modes:
        for realistic, tag in [(False, "naive"), (True, "REALISTIC")]:
            df = run_costed(df_1m, mode=mkey, realistic=realistic)
            s = stat_line(df)
            keep[(mkey, realistic)] = (df, s)
            if s is None:
                print(f"  {mlabel:20}  {tag:9} no trades")
                continue
            print(f"  {mlabel:20}  {tag:9} {s['n']:>4} {s['wr']:>5.1f}% {s['pf']:>4.2f} "
                  f"{s['r']:>+7.1f}R {s['dd']:>6.1f}% {s['ls']:>5} {s['avg_risk']:>5.1f}p "
                  f"${s['bal']:>8,.0f}")
        print()

    # Equity comparison for the best naive mode (CISD_ONLY)
    BG, FG, GRN, GRY = "#0d1117", "#e6edf3", "#3fb950", "#7d8590"
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG); ax.set_facecolor(BG)
    for mkey, color, style in [("CISD_ONLY", GRN, "-"), ("IFVG_CISD", "#58a6ff", "-")]:
        for realistic, lw, alpha, lbl_tag in [(False, 1.1, 0.5, "naive"), (True, 1.8, 1.0, "REALISTIC")]:
            _, s = keep[(mkey, realistic)]
            if s is None:
                continue
            c = color if realistic else GRY
            ax.plot(s["eq"], color=c, linewidth=lw, alpha=alpha,
                    label=f"{mkey} {lbl_tag} (PF {s['pf']:.2f})")
    ax.axhline(10000, color="#444c56", linestyle="--", linewidth=0.8)
    ax.set_title("1M CISD · EURUSD · Naive vs Realistic (spread + slippage)",
                 color=FG, fontsize=12, fontweight="bold")
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9)
    ax.set_xlabel("Trade #", color=FG); ax.set_ylabel("Balance ($)", color=FG)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "equity_naive_vs_realistic.png"),
                dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    # Save the best-mode realistic trade log
    keep[("CISD_ONLY", True)][0].to_csv(
        os.path.join(OUT, "trades_CISD_ONLY_realistic.csv"), index=False)
    print(f"Saved equity + trades -> {OUT}/")


if __name__ == "__main__":
    main()
