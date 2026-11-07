"""
EURUSD · 4H sweep + 15M IFVG · minimum 2RR · loosened filters (~3+ trades/week).

Fundamentals fixed: 4H sweep + 15M IFVG, entry = close or retrace (1.5x rule).
Loosened everything else to raise frequency. Frequency is raised by using a
smaller 4H swing lookback (more swing points → more sweeps) and/or the
previous-4H-candle sweep. RR fixed to 2.0.

Outputs a frequency scan, full stats on the chosen config, a sample-trade
chart, and an equity curve.
"""
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import main as bt
import plot_utils as pu

SYMBOL = "EURUSD"
PIP = 0.0001
REV_WINDOW  = 24
DIS_DELAY   = 7
RETR_WINDOW = 48
DIS_MULT    = 1.5
SL_BUF      = 0.25
RR          = 2.0     # minimum 2RR (fixed)


def find_swings(df, n):
    return bt.find_swings(df, n)


def sweeps_swing(df_h4, n):
    dsw = find_swings(df_h4, n)
    return bt.find_sweeps(dsw)


def sweeps_prev4h(df_15, df_h4):
    prevH = df_h4["high"].shift(1); prevL = df_h4["low"].shift(1)
    floor4h = df_15.index.floor("4h")
    ph = prevH.reindex(floor4h).values; pl = prevL.reindex(floor4h).values
    h = df_15["high"].values; l = df_15["low"].values; c = df_15["close"].values
    idx = df_15.index
    sweeps = []; cur = None; hd = ld = False
    for i in range(len(df_15)):
        p = floor4h[i]
        if p != cur: cur = p; hd = ld = False
        if np.isnan(ph[i]): continue
        if not hd and h[i] > ph[i] and c[i] < ph[i]:
            sweeps.append({"time": idx[i], "direction": "bearish", "swept_level": float(ph[i])}); hd = True
        if not ld and l[i] < pl[i] and c[i] > pl[i]:
            sweeps.append({"time": idx[i], "direction": "bullish", "swept_level": float(pl[i])}); ld = True
    return sweeps


def simulate(h, l, idx, ei, direction, entry, sl, tp, rr, max_bars=400):
    last = ei
    for j in range(ei + 1, min(ei + max_bars, len(h))):
        last = j
        if direction == "long":
            if l[j] <= sl: return "loss", idx[j], sl, -1.0
            if h[j] >= tp: return "win",  idx[j], tp, rr
        else:
            if h[j] >= sl: return "loss", idx[j], sl, -1.0
            if l[j] <= tp: return "win",  idx[j], tp, rr
    return "open", idx[last], float("nan"), 0.0


def run(df_h4, df_15, sweeps, *, sl_mode="IFVG_EDGE", min_fvg_pips=1.0):
    min_fvg = min_fvg_pips * PIP
    fvgs = sorted(bt.find_all_fvgs(df_15), key=lambda f: f["form_idx"])
    fia = np.array([f["form_idx"] for f in fvgs])
    h = df_15["high"].values; l = df_15["low"].values; c = df_15["close"].values
    idx = df_15.index
    trades = []; used = set(); open_until = []

    for sw in sweeps:
        bias = sw["direction"]; sig = (sw["time"], bias)
        if sig in used: continue
        si = int(np.searchsorted(idx, sw["time"], "left"))
        if si >= len(idx) or (si == 0 and sw["time"] < idx[0]): continue
        tdir = "bearish" if bias == "bullish" else "bullish"
        rev_end = min(si + REV_WINDOW, len(idx) - 1)
        lo = int(np.searchsorted(fia, si, "left")); hi = int(np.searchsorted(fia, rev_end, "right"))
        done = False
        for p in range(hi - 1, lo - 1, -1):
            fvg = fvgs[p]
            if fvg["direction"] != tdir or fvg["width"] < min_fvg: continue
            fi = fvg["form_idx"]
            for di in range(fi + 1, min(fi + DIS_DELAY, len(idx) - 1) + 1):
                inv = c[di] > fvg["top"] if fvg["direction"] == "bearish" else c[di] < fvg["bottom"]
                if not inv: continue
                rng = h[di] - l[di]
                too_big = rng > DIS_MULT * fvg["width"]
                buf = fvg["width"] * SL_BUF
                sweep_ext = l[si:di + 1].min() if bias == "bullish" else h[si:di + 1].max()
                if not too_big:
                    ei = di; entry = c[di]; etype = "close"
                else:
                    rend = min(di + RETR_WINDOW, len(idx) - 1); ei = None
                    for rj in range(di + 1, rend + 1):
                        if bias == "bullish" and l[rj] <= fvg["top"]: ei = rj; entry = fvg["top"]; break
                        if bias == "bearish" and h[rj] >= fvg["bottom"]: ei = rj; entry = fvg["bottom"]; break
                    if ei is None: break
                    etype = "retrace"
                open_until = [t for t in open_until if t > idx[ei]]
                if len(open_until) >= 1: done = True; break
                if sl_mode == "IFVG_EDGE":
                    sl = (fvg["bottom"] - buf) if bias == "bullish" else (fvg["top"] + buf)
                else:
                    sl = (sweep_ext - buf) if bias == "bullish" else (sweep_ext + buf)
                if bias == "bullish":
                    direction = "long"; risk = entry - sl; tp = entry + RR * risk
                else:
                    direction = "short"; risk = sl - entry; tp = entry - RR * risk
                if risk <= 0: break
                oc, xt, xp, pr = simulate(h, l, idx, ei, direction, entry, sl, tp, RR)
                open_until.append(pd.Timestamp(xt))
                trades.append({
                    "symbol": SYMBOL, "sweep_time": sw["time"], "bias": bias,
                    "swept_level": round(sw["swept_level"], 6),
                    "entry_time": idx[ei], "entry_idx": ei, "direction": direction,
                    "entry_type": etype, "entry": round(entry, 6), "sl": round(sl, 6),
                    "tp": round(tp, 6), "fvg_top": round(fvg["top"], 6),
                    "fvg_bottom": round(fvg["bottom"], 6), "outcome": oc,
                    "exit": round(xp, 6) if not np.isnan(xp) else np.nan,
                    "exit_time": xt, "pnl_r": pr,
                })
                used.add(sig); done = True; break
            if done: break
    return pd.DataFrame(trades)


def stats(df, weeks):
    cl = df[df["outcome"] != "open"]
    if not len(cl): return None
    n = len(cl); wins = (cl["outcome"] == "win").sum(); wr = wins / n * 100
    gw = wins * RR; glo = (n - wins); pf = gw / glo if glo else float("inf")
    bal = 10000.0; eq = [bal]; cls = cw = bcl = bcw = 0
    for _, r in cl.iterrows():
        bal += bal * 0.01 * r["pnl_r"]; eq.append(bal)
        if r["outcome"] == "loss": cls += 1; cw = 0; bcl = max(bcl, cls)
        elif r["outcome"] == "win": cw += 1; cls = 0; bcw = max(bcw, cw)
    eqa = np.array(eq); dd = ((eqa - np.maximum.accumulate(eqa)) / np.maximum.accumulate(eqa) * 100).min()
    return dict(n=n, wr=wr, pf=pf, totalR=gw - glo, dd=dd, ls=bcl, ws=bcw,
                bal=bal, per_week=n / weeks)


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    OUT = os.path.join("strategies", "eurusd_2r")
    os.makedirs(OUT, exist_ok=True)

    df_h4 = bt.fetch_data(SYMBOL, mt5.TIMEFRAME_H4, bt.START_DATE, bt.END_DATE, tf_name="H4")
    df_15 = bt.fetch_data(SYMBOL, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
    weeks = (df_15.index[-1] - df_15.index[0]).days / 7

    print("=" * 96)
    print(f"EURUSD · 4H sweep + 15M IFVG · RR={RR} · loosened · frequency scan  ({weeks:.0f} weeks)")
    print("=" * 96)
    print(f"  {'Config':40}  {'N':>4}  {'/wk':>4}  {'WR':>6}  {'PF':>4}  {'totR':>6}  {'DD':>6}  {'LStrk':>5}")

    configs = []
    for n_sw in [5, 3, 2]:
        sweeps = sweeps_swing(df_h4, n_sw)
        df = run(df_h4, df_15, sweeps, sl_mode="IFVG_EDGE", min_fvg_pips=1.0)
        s = stats(df, weeks)
        if s:
            configs.append((f"SWING lb={n_sw} · IFVG_EDGE · minFVG1", df, s))
    # prev-4h candle — scan min-FVG to dial frequency toward ~3/week + lift quality
    sweeps_p = sweeps_prev4h(df_15, df_h4)
    for mf in [2, 4, 6, 8, 10]:
        df = run(df_h4, df_15, sweeps_p, sl_mode="IFVG_EDGE", min_fvg_pips=mf)
        s = stats(df, weeks)
        if s:
            configs.append((f"PREV_4H · IFVG_EDGE · minFVG{mf}", df, s))

    for label, df, s in configs:
        print(f"  {label:40}  {s['n']:4d}  {s['per_week']:4.1f}  {s['wr']:5.1f}%  "
              f"{s['pf']:4.2f}  {s['totalR']:+5.0f}  {s['dd']:5.1f}%  {s['ls']:>5}")

    # Pick the config whose frequency is closest to the ~3/week target
    TARGET = 3.0
    chosen = min(configs, key=lambda x: abs(x[2]["per_week"] - TARGET))
    lab, df, s = chosen

    print("\n" + "=" * 96)
    print(f"CHOSEN (>=3/wk, best WR): {lab}")
    print("=" * 96)
    print(f"  Trades {s['n']}  ({s['per_week']:.1f}/week)  |  WR {s['wr']:.1f}%  |  PF {s['pf']:.2f}  |  "
          f"Total {s['totalR']:+.0f}R  |  DD {s['dd']:.1f}%  |  WinStreak {s['ws']}  LossStreak {s['ls']}")
    print(f"  $10k → ${s['bal']:,.0f}  ({(s['bal']/10000-1)*100:+.0f}%)")
    cl = df[df["outcome"] != "open"].copy()
    cl["year"] = pd.to_datetime(cl["entry_time"]).dt.year
    print("  Per-year:")
    for yr, g in cl.groupby("year"):
        wr = (g["outcome"] == "win").sum() / len(g) * 100
        print(f"    {yr}: N={len(g):3d}  WR={wr:5.1f}%  R={g['pnl_r'].sum():+.0f}")
    print("  Entry type:")
    for et, g in cl.groupby("entry_type"):
        wr = (g["outcome"] == "win").sum() / len(g) * 100
        print(f"    {et:8}: N={len(g):3d}  WR={wr:5.1f}%")

    df.to_csv(os.path.join(OUT, "trades.csv"), index=False)
    pu.plot_equity(cl, os.path.join(OUT, "equity.png"),
                   f"EURUSD · {lab} · RR={RR}")

    # Plot a representative trade — the most recent win
    wins = cl[cl["outcome"] == "win"].sort_values("entry_time")
    sample = wins.iloc[-1] if len(wins) else cl.iloc[-1]
    pu.plot_recent_trade(df_15, sample.to_dict(), os.path.join(OUT, "sample_trade.png"),
                         title_prefix="EURUSD 4H-sweep + 15M IFVG", symbol=SYMBOL)
    print(f"\n  Sample trade: {sample['entry_time']} {sample['direction']} "
          f"{sample['entry_type']} {sample['pnl_r']:+.0f}R  → {OUT}/sample_trade.png")
    print(f"  Saved trades.csv + equity.png to {OUT}/")
    mt5.shutdown()


if __name__ == "__main__":
    main()
