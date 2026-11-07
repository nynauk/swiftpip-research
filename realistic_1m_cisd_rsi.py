"""
1M CISD + RSI confluence — EURUSD, USDJPY, USDCHF.

Adds an RSI(14) veto filter to the Sweep+CISD setup. Rationale: a genuine
liquidity-grab reversal should sweep into an exhausted momentum condition.
  LONG  (Asia low swept, reversing up)   → require RSI at the sweep <= OS
  SHORT (Asia high swept, reversing down)→ require RSI at the sweep >= OB
If the sweep's RSI is not at the extreme, the trade is vetoed (no entry that day).

Compares baseline (no RSI) vs two RSI thresholds, naive vs realistic, so we can
see whether RSI confluence rescues the edge or just thins the sample.

Data + costs identical to realistic_1m_cisd_multi.py (fresh MT5 M1, broker time,
per-pair spread + 0.3 pip slip, 3-pip min stop, Fixed 2R).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MetaTrader5 as mt5


SYMBOLS = ["EURUSD", "USDJPY", "USDCHF"]
ASIA_START, ASIA_END     = 3, 10
LONDON_START, LONDON_END = 10, 13
SPEC = {
    "EURUSD": dict(pip=1e-4, spread_p=1.0, min_sl_p=3.0),
    "USDJPY": dict(pip=1e-2, spread_p=1.0, min_sl_p=3.0),
    "USDCHF": dict(pip=1e-4, spread_p=1.5, min_sl_p=3.0),
}
SLIPPAGE_P    = 0.3
SL_BUFFER     = 0.10
IFVG_WINDOW   = 60
CISD_LOOKBACK = 5
RR            = 2.0
RSI_PERIOD    = 14


def fetch_m1(symbol):
    mt5.symbol_select(symbol, True)
    bars = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 99999)
    df = pd.DataFrame(bars)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.set_index("time")[["open", "high", "low", "close"]]


def rsi(close, period=RSI_PERIOD):
    """Wilder's RSI via EWM (alpha = 1/period)."""
    s = pd.Series(close)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).values


def run(df_1m, symbol, realistic, rsi_arr, rsi_os=None, rsi_ob=None):
    """Sweep + CISD, optional RSI veto at the sweep candle."""
    sp = SPEC[symbol]; pip = sp["pip"]
    spread   = sp["spread_p"] * pip if realistic else 0.0
    slippage = SLIPPAGE_P     * pip if realistic else 0.0
    min_sl   = sp["min_sl_p"] * pip

    h = df_1m["high"].values; l = df_1m["low"].values; c = df_1m["close"].values
    idx = df_1m.index; hours = idx.hour; dates = idx.normalize()

    trades = []; vetoed = 0
    unique_days, day_starts = np.unique(dates, return_index=True)
    day_ends = np.append(day_starts[1:], len(df_1m))

    for di, day in enumerate(unique_days):
        a, b = day_starts[di], day_ends[di]
        dh = hours[a:b]
        asia_mask = (dh >= ASIA_START) & (dh < ASIA_END)
        if not asia_mask.any():
            continue
        asia_hi = h[a:b][asia_mask].max(); asia_lo = l[a:b][asia_mask].min()
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

        # ── RSI veto at the sweep candle ──
        if rsi_os is not None:
            rv = rsi_arr[sweep_i]
            if np.isnan(rv):
                continue
            if sweep_dir == "long" and rv > rsi_os:
                vetoed += 1; continue
            if sweep_dir == "short" and rv < rsi_ob:
                vetoed += 1; continue

        cisd_end = min(sweep_i + IFVG_WINDOW, len(df_1m) - 1)
        cisd_idx = None
        for k in range(sweep_i + 1, cisd_end + 1):
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
            direction = "short"; real_entry = sig - slippage
            sl = sweep_ext + abs(sweep_ext - real_entry) * SL_BUFFER
            if sl - real_entry < min_sl: sl = real_entry + min_sl
            risk = sl - real_entry
            if risk <= 0: continue
            tp = real_entry - RR * risk
        else:
            direction = "long"; real_entry = sig + spread + slippage
            sl = sweep_ext - abs(real_entry - sweep_ext) * SL_BUFFER
            if real_entry - sl < min_sl: sl = real_entry - min_sl
            risk = real_entry - sl
            if risk <= 0: continue
            tp = real_entry + RR * risk

        oc, xt, pr = _sim(h, l, idx, entry_i, direction, real_entry, sl, tp, spread)
        trades.append({
            "symbol": symbol, "date": pd.Timestamp(day).date(),
            "entry_time": idx[entry_i], "direction": direction,
            "rsi_at_sweep": round(float(rsi_arr[sweep_i]), 1) if not np.isnan(rsi_arr[sweep_i]) else np.nan,
            "entry": real_entry, "sl": sl, "tp": tp,
            "risk_pips": risk / pip, "outcome": oc, "exit_time": xt, "pnl_r": pr,
        })
    df = pd.DataFrame(trades)
    df.attrs["vetoed"] = vetoed
    return df


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
    return dict(n=n, wr=wr, r=r, pf=pf, dd=dd, ls=bcl, bal=bal, eq=eqa)


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    OUT = os.path.join("strategies", "realistic_1m_cisd_rsi")
    os.makedirs(OUT, exist_ok=True)

    data = {}; rsis = {}
    for s in SYMBOLS:
        df = fetch_m1(s); data[s] = df; rsis[s] = rsi(df["close"].values)
    d0 = min(df.index[0] for df in data.values())
    d1 = max(df.index[-1] for df in data.values())
    days = (d1 - d0).days + 1

    variants = [
        ("no RSI (baseline)",  None, None),
        ("RSI 30/70",          30,   70),
        ("RSI 40/60",          40,   60),
    ]

    print("=" * 104)
    print(f"1M CISD + RSI CONFLUENCE · EURUSD/USDJPY/USDCHF · London KZ 10-13 broker "
          f"· {d0.date()} -> {d1.date()} ({days} days)")
    print(f"RSI({RSI_PERIOD}) veto at sweep: LONG needs RSI<=OS, SHORT needs RSI>=OB "
          f"· per-pair spread + {SLIPPAGE_P}p slip · 3p min stop · Fixed {RR:.0f}R")
    print("=" * 104)

    keep = {}
    for vlabel, os_, ob_ in variants:
        print(f"\n  === {vlabel} ===")
        print(f"  {'Symbol':8} {'':9} {'N':>4} {'WR':>6} {'PF':>5} {'R':>8} {'DD':>7} "
              f"{'LStrk':>5}  {'$10k->':>9}  {'vetoed':>6}")
        combos = {False: [], True: []}
        for s in SYMBOLS:
            for realistic in (False, True):
                df = run(data[s], s, realistic, rsis[s], os_, ob_)
                combos[realistic].append(df)
                st = stats(df)
                tag = "REALISTIC" if realistic else "naive"
                vet = df.attrs.get("vetoed", 0)
                if st is None:
                    print(f"  {s:8} {tag:9} no trades  (vetoed {vet})"); continue
                print(f"  {s:8} {tag:9} {st['n']:>4} {st['wr']:>5.1f}% {st['pf']:>4.2f} "
                      f"{st['r']:>+7.1f}R {st['dd']:>6.1f}% {st['ls']:>5}  "
                      f"${st['bal']:>8,.0f}  {vet:>6}")
        for realistic in (False, True):
            allc = pd.concat(combos[realistic], ignore_index=True).sort_values("entry_time")
            st = stats(allc)
            keep[(vlabel, realistic)] = (allc, st)
            tag = "REALISTIC" if realistic else "naive"
            if st:
                print(f"  {'COMBINED':8} {tag:9} {st['n']:>4} {st['wr']:>5.1f}% {st['pf']:>4.2f} "
                      f"{st['r']:>+7.1f}R {st['dd']:>6.1f}% {st['ls']:>5}  ${st['bal']:>8,.0f}")

    # Equity: realistic combined for each RSI variant
    BG, FG = "#0d1117", "#e6edf3"
    colors = {"no RSI (baseline)": "#7d8590", "RSI 30/70": "#3fb950", "RSI 40/60": "#58a6ff"}
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG); ax.set_facecolor(BG)
    for vlabel in colors:
        st = keep[(vlabel, True)][1]
        if st is None: continue
        ax.plot(st["eq"], color=colors[vlabel], linewidth=1.7,
                label=f"{vlabel} REALISTIC (PF {st['pf']:.2f}, N={st['n']})")
    ax.axhline(10000, color="#444c56", linestyle="--", linewidth=0.8)
    ax.set_title("1M CISD + RSI · EURUSD+USDJPY+USDCHF · REALISTIC conditions",
                 color=FG, fontsize=12, fontweight="bold")
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9)
    ax.set_xlabel("Trade #", color=FG); ax.set_ylabel("Balance ($)", color=FG)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "equity_rsi_variants.png"),
                dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nSaved -> {OUT}/")
    mt5.shutdown()


if __name__ == "__main__":
    main()
