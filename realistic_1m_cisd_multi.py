"""
Realistic 1M CISD across EURUSD, USDJPY, USDCHF.

Fetches fresh M1 from MT5 (the on-disk M1 pkl caches are corrupt — pickled
under an incompatible pandas). MT5 M1 is BROKER TIME (GMT+3 / EEST in summer),
so the London killzone 07:00-10:00 GMT maps to 10:00-13:00 broker, and the
Asia range 00:00-07:00 GMT maps to 03:00-10:00 broker.

Same mechanical strategy as asia_1m_cisd.py, two confirmation modes:
  CISD_ONLY  — sweep -> CISD -> enter        (best naive mode on EURUSD)
  IFVG_CISD  — sweep -> IFVG -> CISD -> enter (the full chain)

Costs (realistic): per-pair spread + 0.3 pip slippage, 3-pip min stop.
LONG fills at ask (close+spread+slip); SHORT at bid (close-slip) with SL/TP
triggered against ask (bid+spread). Fixed 2R target.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MetaTrader5 as mt5


SYMBOLS = ["EURUSD", "USDJPY", "USDCHF"]

# Broker-time session windows (GMT+3). London KZ 07-10 GMT = 10-13 broker.
ASIA_START, ASIA_END     = 3, 10
LONDON_START, LONDON_END = 10, 13

# Per-pair spec: pip size, spread (pips), min-stop (pips)
SPEC = {
    "EURUSD": dict(pip=1e-4, spread_p=1.0, min_sl_p=3.0),
    "USDJPY": dict(pip=1e-2, spread_p=1.0, min_sl_p=3.0),
    "USDCHF": dict(pip=1e-4, spread_p=1.5, min_sl_p=3.0),
}
SLIPPAGE_P   = 0.3
SL_BUFFER    = 0.10
IFVG_WINDOW  = 60
CISD_WINDOW  = 30
CISD_LOOKBACK = 5
RR           = 2.0


def fetch_m1(symbol):
    mt5.symbol_select(symbol, True)
    bars = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 99999)
    if bars is None or not len(bars):
        raise RuntimeError(f"No M1 for {symbol}")
    df = pd.DataFrame(bars)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.set_index("time")[["open", "high", "low", "close"]]


def run(df_1m, symbol, mode, realistic):
    sp = SPEC[symbol]
    pip = sp["pip"]
    spread   = sp["spread_p"] * pip if realistic else 0.0
    slippage = SLIPPAGE_P     * pip if realistic else 0.0
    min_sl   = sp["min_sl_p"] * pip
    min_fvg  = 0.5 * pip                     # 0.5-pip minimum FVG

    h = df_1m["high"].values; l = df_1m["low"].values; c = df_1m["close"].values
    idx = df_1m.index; hours = idx.hour; dates = idx.normalize()

    trades = []
    unique_days, day_starts = np.unique(dates, return_index=True)
    day_ends = np.append(day_starts[1:], len(df_1m))

    for di, day in enumerate(unique_days):
        a, b = day_starts[di], day_ends[di]
        dh = hours[a:b]
        asia_mask = (dh >= ASIA_START) & (dh < ASIA_END)
        if not asia_mask.any():
            continue
        asia_hi = h[a:b][asia_mask].max()
        asia_lo = l[a:b][asia_mask].min()
        lon_local = np.where((dh >= LONDON_START) & (dh < LONDON_END))[0]
        if not len(lon_local):
            continue

        sweep_dir = sweep_i = sweep_ext = None
        for j in lon_local:
            i = a + j
            if h[i] > asia_hi and c[i] < asia_hi:
                sweep_dir, sweep_i, sweep_ext = "short", i, h[i]; break
            if l[i] < asia_lo and c[i] > asia_lo:
                sweep_dir, sweep_i, sweep_ext = "long", i, l[i]; break
        if sweep_dir is None:
            continue

        fvg = None; ifvg_idx = None; cisd_idx = None
        scan_end = min(sweep_i + IFVG_WINDOW, len(df_1m) - 1)

        if mode == "IFVG_CISD":
            tdir = "bullish" if sweep_dir == "short" else "bearish"
            for k in range(sweep_i, max(2, sweep_i - 60) - 1, -1):
                if tdir == "bullish" and l[k] > h[k - 2]:
                    w = float(l[k] - h[k - 2])
                    if w >= min_fvg:
                        fvg = {"top": float(l[k]), "bottom": float(h[k - 2])}; break
                elif tdir == "bearish" and h[k] < l[k - 2]:
                    w = float(l[k - 2] - h[k])
                    if w >= min_fvg:
                        fvg = {"top": float(l[k - 2]), "bottom": float(h[k])}; break
            if fvg is None:
                continue
            for k in range(sweep_i + 1, scan_end + 1):
                if sweep_dir == "short" and c[k] < fvg["bottom"]:
                    ifvg_idx = k; break
                if sweep_dir == "long" and c[k] > fvg["top"]:
                    ifvg_idx = k; break
            if ifvg_idx is None:
                continue
            cisd_end = min(ifvg_idx + CISD_WINDOW, len(df_1m) - 1)
            scan_from = ifvg_idx + 1
        else:  # CISD_ONLY
            cisd_end = min(sweep_i + IFVG_WINDOW, len(df_1m) - 1)
            scan_from = sweep_i + 1

        for k in range(scan_from, cisd_end + 1):
            if k - CISD_LOOKBACK < 0:
                continue
            if sweep_dir == "short" and c[k] < l[k - CISD_LOOKBACK:k].min():
                cisd_idx = k; break
            if sweep_dir == "long" and c[k] > h[k - CISD_LOOKBACK:k].max():
                cisd_idx = k; break
        if cisd_idx is None:
            continue
        entry_i = cisd_idx
        if hours[entry_i] < LONDON_START or hours[entry_i] >= LONDON_END:
            continue

        sig = c[entry_i]
        if sweep_dir == "short":
            direction = "short"
            real_entry = sig - slippage
            base = sweep_ext - real_entry
            sl = sweep_ext + abs(base) * SL_BUFFER
            if sl - real_entry < min_sl:
                sl = real_entry + min_sl
            risk = sl - real_entry
            if risk <= 0:
                continue
            tp = real_entry - RR * risk
        else:
            direction = "long"
            real_entry = sig + spread + slippage
            base = real_entry - sweep_ext
            sl = sweep_ext - abs(base) * SL_BUFFER
            if real_entry - sl < min_sl:
                sl = real_entry - min_sl
            risk = real_entry - sl
            if risk <= 0:
                continue
            tp = real_entry + RR * risk

        oc, xt, pr = _sim(h, l, idx, entry_i, direction, real_entry, sl, tp, spread)
        trades.append({
            "symbol": symbol, "date": pd.Timestamp(day).date(), "mode": mode,
            "entry_time": idx[entry_i], "direction": direction,
            "entry": real_entry, "sl": sl, "tp": tp,
            "risk_pips": risk / pip, "outcome": oc, "exit_time": xt, "pnl_r": pr,
        })
    return pd.DataFrame(trades)


def _sim(h, l, idx, ei, direction, entry, sl, tp, spread, max_bars=1200):
    for j in range(ei + 1, min(ei + max_bars, len(h))):
        if direction == "long":
            if l[j] <= sl: return "loss", idx[j], -1.0
            if h[j] >= tp: return "win",  idx[j],  RR
        else:
            if h[j] + spread >= sl: return "loss", idx[j], -1.0
            if l[j] + spread <= tp: return "win",  idx[j],  RR
    return "open", idx[min(ei + max_bars - 1, len(h) - 1)], 0.0


def stats(df):
    cl = df[df["outcome"] != "open"] if len(df) else df
    if not len(cl):
        return None
    n = len(cl); wins = (cl["outcome"] == "win").sum()
    wr = wins / n * 100; r = cl["pnl_r"].sum()
    gw = cl.loc[cl["pnl_r"] > 0, "pnl_r"].sum()
    gl = -cl.loc[cl["pnl_r"] < 0, "pnl_r"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    bal = 10000.0; eq = [bal]; cl_ = bcl = 0
    for _, row in cl.sort_values("entry_time").iterrows():
        bal += bal * 0.01 * float(row["pnl_r"]); eq.append(bal)
        if row["pnl_r"] < -0.01: cl_ += 1; bcl = max(bcl, cl_)
        elif row["pnl_r"] > 0.01: cl_ = 0
    eqa = np.array(eq)
    dd = ((eqa - np.maximum.accumulate(eqa)) / np.maximum.accumulate(eqa) * 100).min()
    return dict(n=n, wr=wr, r=r, pf=pf, dd=dd, ls=bcl, bal=bal,
                avg_risk=cl["risk_pips"].mean(), eq=eqa)


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    OUT = os.path.join("strategies", "realistic_1m_cisd_multi")
    os.makedirs(OUT, exist_ok=True)

    data = {s: fetch_m1(s) for s in SYMBOLS}
    d0 = min(df.index[0] for df in data.values())
    d1 = max(df.index[-1] for df in data.values())
    days = (d1 - d0).days + 1

    print("=" * 104)
    print(f"1M CISD REALISTIC · EURUSD/USDJPY/USDCHF · broker-time London KZ 10-13 "
          f"(=07-10 GMT) · {d0.date()} -> {d1.date()} ({days} days)")
    print(f"Costs: per-pair spread + {SLIPPAGE_P} pip slip · 3-pip min stop · Fixed {RR:.0f}R")
    print("=" * 104)

    keep = {}
    for mode in ["CISD_ONLY", "IFVG_CISD"]:
        print(f"\n  === {mode} ===")
        print(f"  {'Symbol':8} {'':9} {'N':>4} {'WR':>6} {'PF':>5} {'R':>8} {'DD':>7} "
              f"{'LStrk':>5} {'avgSL':>6}  {'$10k->':>9}")
        combos = {False: [], True: []}
        for s in SYMBOLS:
            for realistic in (False, True):
                df = run(data[s], s, mode, realistic)
                combos[realistic].append(df)
                st = stats(df)
                tag = "REALISTIC" if realistic else "naive"
                if st is None:
                    print(f"  {s:8} {tag:9} no trades"); continue
                print(f"  {s:8} {tag:9} {st['n']:>4} {st['wr']:>5.1f}% {st['pf']:>4.2f} "
                      f"{st['r']:>+7.1f}R {st['dd']:>6.1f}% {st['ls']:>5} "
                      f"{st['avg_risk']:>5.1f}p ${st['bal']:>8,.0f}")
        # combined
        for realistic in (False, True):
            allc = pd.concat(combos[realistic], ignore_index=True).sort_values("entry_time")
            st = stats(allc)
            keep[(mode, realistic)] = (allc, st)
            tag = "REALISTIC" if realistic else "naive"
            if st:
                print(f"  {'COMBINED':8} {tag:9} {st['n']:>4} {st['wr']:>5.1f}% {st['pf']:>4.2f} "
                      f"{st['r']:>+7.1f}R {st['dd']:>6.1f}% {st['ls']:>5} "
                      f"{st['avg_risk']:>5.1f}p ${st['bal']:>8,.0f}")

    # Equity: combined naive vs realistic for both modes
    BG, FG = "#0d1117", "#e6edf3"
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG); ax.set_facecolor(BG)
    styles = {("CISD_ONLY", False): ("#7d8590", 1.1, 0.6),
              ("CISD_ONLY", True):  ("#3fb950", 1.9, 1.0),
              ("IFVG_CISD", False): ("#9d8590", 1.1, 0.6),
              ("IFVG_CISD", True):  ("#58a6ff", 1.9, 1.0)}
    for (mode, realistic), (color, lw, alpha) in styles.items():
        st = keep[(mode, realistic)][1]
        if st is None:
            continue
        tag = "REALISTIC" if realistic else "naive"
        ax.plot(st["eq"], color=color, linewidth=lw, alpha=alpha,
                label=f"{mode} {tag} (PF {st['pf']:.2f})")
    ax.axhline(10000, color="#444c56", linestyle="--", linewidth=0.8)
    ax.set_title("1M CISD · EURUSD+USDJPY+USDCHF · Naive vs Realistic",
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

    keep[("CISD_ONLY", True)][0].to_csv(
        os.path.join(OUT, "trades_CISD_ONLY_realistic.csv"), index=False)
    print(f"\nSaved -> {OUT}/")
    mt5.shutdown()


if __name__ == "__main__":
    main()
