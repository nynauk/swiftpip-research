"""
Clean 4H sweep + 15M IFVG — the raw setup, no extra filters.

Sequence (candle-by-candle, no look-ahead):
  1. 4H sweep of a liquidity level. Two definitions tested:
       PREV_4H : the PREVIOUS completed 4H candle's high / low is swept.
       SWING   : a 4H swing high / low (5-bar) is swept.
     A sweep = price wicks past the level and a candle closes back inside.
       swept HIGH → SHORT bias ,  swept LOW → LONG bias.
  2. 15M reversal FVG (opposite the bias) forms after the sweep and then
     INVERTS (IFVG) — a 15M candle closes through it in the bias direction.
     That inversion candle is the "disrespect" candle.
  3. ENTRY:
       • disrespect candle range <= 1.5 x FVG width  →  enter at its CLOSE.
       • disrespect candle range >  1.5 x FVG width  →  the candle is "too big";
         wait for a RETRACE back to the IFVG near edge and enter there
         (long → fvg top, short → fvg bottom).
  4. STOP: beyond the swept structural extreme + 0.25 x FVG width  (SWEEP_EXT)
           — an IFVG-edge variant is also tested.
  5. TARGET: 2R.
  No news filter, no session filter, no min-width filter. Pure setup.
"""
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import main as bt
import plot_utils as pu


MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]
PIP = {"EURUSD":0.0001,"GBPUSD":0.0001,"USDCAD":0.0001,"USDCHF":0.0001,
       "USDJPY":0.01,"AUDUSD":0.0001,"NZDUSD":0.0001}
SWEEP_REVERSAL_WINDOW = 24    # 15M bars after sweep for the reversal FVG
MAX_DISRESPECT_DELAY  = 7     # 15M bars for the FVG to invert
RETRACE_WINDOW        = 48    # 15M bars to wait for the retrace entry
DISRESPECT_MULT       = 1.5   # candle "too big" threshold
SL_BUFFER             = 0.25
RR                    = 2.0


def prev4h_sweeps(df_15m, df_h4):
    """Sweeps of the PREVIOUS completed 4H candle's high/low, detected on 15M."""
    prevH = df_h4["high"].shift(1)
    prevL = df_h4["low"].shift(1)
    h4_open = df_h4.index
    # For each 15M bar, find its current 4H candle open (floor to 4h) → prev candle H/L
    floor4h = df_15m.index.floor("4h")
    ph = prevH.reindex(floor4h).values
    pl = prevL.reindex(floor4h).values

    h = df_15m["high"].values; l = df_15m["low"].values; c = df_15m["close"].values
    idx = df_15m.index
    sweeps = []
    cur_period = None
    hi_done = lo_done = False
    for i in range(len(df_15m)):
        p = floor4h[i]
        if p != cur_period:
            cur_period = p; hi_done = lo_done = False
        if np.isnan(ph[i]):
            continue
        if not hi_done and h[i] > ph[i] and c[i] < ph[i]:
            sweeps.append({"time": idx[i], "direction": "bearish", "swept_level": float(ph[i])})
            hi_done = True
        if not lo_done and l[i] < pl[i] and c[i] > pl[i]:
            sweeps.append({"time": idx[i], "direction": "bullish", "swept_level": float(pl[i])})
            lo_done = True
    return sweeps


def simulate(h, l, idx, entry_i, direction, entry, sl, tp, rr_value, max_bars=400):
    last_j = entry_i
    for j in range(entry_i + 1, min(entry_i + max_bars, len(h))):
        last_j = j
        if direction == "long":
            if l[j] <= sl: return {"outcome":"loss","exit":sl,"exit_time":idx[j],"pnl_r":-1.0}
            if h[j] >= tp: return {"outcome":"win","exit":tp,"exit_time":idx[j],"pnl_r":rr_value}
        else:
            if h[j] >= sl: return {"outcome":"loss","exit":sl,"exit_time":idx[j],"pnl_r":-1.0}
            if l[j] <= tp: return {"outcome":"win","exit":tp,"exit_time":idx[j],"pnl_r":rr_value}
    return {"outcome":"open","exit":float("nan"),"exit_time":idx[last_j],"pnl_r":0.0}


def run(symbol, *, sweep_mode="PREV_4H", sl_mode="SWEEP_EXT", min_fvg_pips=1.0):
    pip = PIP.get(symbol, 0.0001)
    min_fvg = min_fvg_pips * pip
    df_h4 = bt.fetch_data(symbol, mt5.TIMEFRAME_H4, bt.START_DATE, bt.END_DATE, tf_name="H4")
    df_15 = bt.fetch_data(symbol, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")

    if sweep_mode == "PREV_4H":
        sweeps = prev4h_sweeps(df_15, df_h4)
    else:
        df_sw = bt.find_swings(df_h4, bt.SWING_LOOKBACK)
        sweeps = bt.find_sweeps(df_sw)

    fvgs = sorted(bt.find_all_fvgs(df_15), key=lambda f: f["form_idx"])
    fia = np.array([f["form_idx"] for f in fvgs])
    h = df_15["high"].values; l = df_15["low"].values; c = df_15["close"].values
    idx = df_15.index

    trades = []
    used = set()
    open_until = []

    for sweep in sweeps:
        bias = sweep["direction"]
        sig = (sweep["time"], bias)
        if sig in used:
            continue
        si = int(np.searchsorted(idx, sweep["time"], side="left"))
        if si >= len(idx) or (si == 0 and sweep["time"] < idx[0]):
            continue

        target_dir = "bearish" if bias == "bullish" else "bullish"
        rev_end = min(si + SWEEP_REVERSAL_WINDOW, len(idx) - 1)
        lo_p = int(np.searchsorted(fia, si, side="left"))
        hi_p = int(np.searchsorted(fia, rev_end, side="right"))

        done = False
        for p in range(hi_p - 1, lo_p - 1, -1):   # nearest the extreme first
            fvg = fvgs[p]
            if fvg["direction"] != target_dir or fvg["width"] < min_fvg:
                continue
            f_i = fvg["form_idx"]
            dis_lo, dis_hi = f_i + 1, min(f_i + MAX_DISRESPECT_DELAY, len(idx) - 1)
            for di in range(dis_lo, dis_hi + 1):
                inverted = c[di] > fvg["top"] if fvg["direction"] == "bearish" else c[di] < fvg["bottom"]
                if not inverted:
                    continue

                dis_range = h[di] - l[di]
                use_retrace = dis_range > DISRESPECT_MULT * fvg["width"]
                buf = fvg["width"] * SL_BUFFER
                if bias == "bullish":
                    sweep_ext = l[si:di + 1].min()
                else:
                    sweep_ext = h[si:di + 1].max()

                # entry index + price
                if not use_retrace:
                    entry_i = di
                    entry = c[di]
                else:
                    retr_end = min(di + RETRACE_WINDOW, len(idx) - 1)
                    entry_i = None
                    for rj in range(di + 1, retr_end + 1):
                        if bias == "bullish" and l[rj] <= fvg["top"]:
                            entry_i = rj; entry = fvg["top"]; break
                        if bias == "bearish" and h[rj] >= fvg["bottom"]:
                            entry_i = rj; entry = fvg["bottom"]; break
                    if entry_i is None:
                        break   # retrace never came → skip this setup

                # concurrency (1 position) — fixed exit_time avoids poisoning
                open_until = [t for t in open_until if t > idx[entry_i]]
                if len(open_until) >= 1:
                    done = True; break

                # SL
                if sl_mode == "IFVG_EDGE":
                    sl = (fvg["bottom"] - buf) if bias == "bullish" else (fvg["top"] + buf)
                else:  # SWEEP_EXT
                    sl = (sweep_ext - buf) if bias == "bullish" else (sweep_ext + buf)

                if bias == "bullish":
                    direction = "long"; risk = entry - sl; tp = entry + RR * risk
                else:
                    direction = "short"; risk = sl - entry; tp = entry - RR * risk
                if risk <= 0:
                    break

                res = simulate(h, l, idx, entry_i, direction, entry, sl, tp, RR)
                open_until.append(pd.Timestamp(res["exit_time"]))
                trades.append({
                    "symbol": symbol, "sweep_time": sweep["time"], "bias": bias,
                    "swept_level": round(sweep["swept_level"], 6),
                    "entry_time": idx[entry_i], "entry_idx": entry_i, "direction": direction,
                    "entry_type": "retrace" if use_retrace else "close",
                    "entry": round(entry, 6), "sl": round(sl, 6), "tp": round(tp, 6),
                    "fvg_top": round(fvg["top"], 6), "fvg_bottom": round(fvg["bottom"], 6),
                    "fvg_width": round(fvg["width"], 6), "dis_mult": round(dis_range / fvg["width"], 2),
                    "outcome": res["outcome"],
                    "exit": round(res["exit"], 6) if not np.isnan(res["exit"]) else np.nan,
                    "exit_time": res["exit_time"], "pnl_r": float(res["pnl_r"]),
                })
                used.add(sig); done = True; break
            if done:
                break
    return pd.DataFrame(trades)


def streak_dd(df, initial=10_000, risk_pct=1.0):
    bal = initial; eq=[bal]; cl=cw=bcl=bcw=0
    for _, r in df.sort_values("entry_time").iterrows():
        bal += bal*(risk_pct/100)*float(r["pnl_r"]); eq.append(bal)
        if r["outcome"]=="loss": cl+=1; cw=0; bcl=max(bcl,cl)
        elif r["outcome"]=="win": cw+=1; cl=0; bcw=max(bcw,cw)
    eq=np.array(eq); dd=((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq)*100).min()
    return bal, dd, bcl, bcw


def summ(all_df, label):
    cl = all_df[all_df["outcome"] != "open"]
    if not len(cl):
        print(f"  {label:38}  no trades"); return None
    n=len(cl); wr=(cl["outcome"]=="win").sum()/n*100; r=cl["pnl_r"].sum()
    gw=cl.loc[cl["outcome"]=="win","pnl_r"].sum(); glo=-cl.loc[cl["outcome"]=="loss","pnl_r"].sum()
    pf=gw/glo if glo>0 else float("inf")
    bal,dd,ls,ws=streak_dd(cl)
    nret=(cl["entry_type"]=="retrace").sum()
    print(f"  {label:38}  N={n:4d}  WR={wr:5.1f}%  R={r:+7.1f}  PF={pf:4.2f}  "
          f"DD={dd:5.1f}%  LStrk={ls:2d}  retrace={nret:3d}  ${bal:>8,.0f}")
    return {"label":label,"df":all_df,"n":n,"wr":wr,"r":r,"pf":pf,"dd":dd,"ls":ls,"bal":bal}


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    OUT = os.path.join("strategies", "sweep_15m_ifvg")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 108)
    print("CLEAN 4H SWEEP + 15M IFVG  ·  entry = close OR retrace (1.5x rule)  ·  RR=2  ·  7 majors")
    print("=" * 108)

    configs = [
        ("PREV_4H candle · SL=sweep ext", dict(sweep_mode="PREV_4H", sl_mode="SWEEP_EXT")),
        ("PREV_4H candle · SL=IFVG edge", dict(sweep_mode="PREV_4H", sl_mode="IFVG_EDGE")),
        ("4H SWING · SL=sweep ext",       dict(sweep_mode="SWING",   sl_mode="SWEEP_EXT")),
        ("4H SWING · SL=IFVG edge",       dict(sweep_mode="SWING",   sl_mode="IFVG_EDGE")),
    ]
    results = []
    for label, kw in configs:
        parts = []
        for s in MAJORS:
            try:
                df = run(s, **kw)
                if len(df): parts.append(df)
            except Exception as e:
                print(f"    [!] {s}: {e}")
        if not parts:
            print(f"  {label:38}  no trades"); continue
        all_df = pd.concat(parts, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
        res = summ(all_df, label)
        if res:
            results.append(res)
            safe = label.replace(" ","_").replace("·","_").replace("=","")
            all_df.to_csv(os.path.join(OUT, f"trades_{safe}.csv"), index=False)

    if not results:
        mt5.shutdown(); return
    best = max(results, key=lambda r: r["r"])
    print(f"\n>>> BEST BY R: {best['label']}")

    b = best["df"]; cl = b[b["outcome"] != "open"]
    print("\n  Per-pair (best):")
    for sym, g in cl.groupby("symbol"):
        wr=(g["outcome"]=="win").sum()/len(g)*100
        print(f"    {sym:8}: N={len(g):4d}  WR={wr:5.1f}%  R={g['pnl_r'].sum():+6.1f}")
    print("\n  Close vs Retrace entries (best):")
    for et, g in cl.groupby("entry_type"):
        wr=(g["outcome"]=="win").sum()/len(g)*100
        print(f"    {et:8}: N={len(g):4d}  WR={wr:5.1f}%  R={g['pnl_r'].sum():+6.1f}")
    b2 = b.copy(); b2["year"]=pd.to_datetime(b2["entry_time"]).dt.year
    print("\n  Per-year (best):")
    for yr, g in b2[b2["outcome"]!="open"].groupby("year"):
        wr=(g["outcome"]=="win").sum()/len(g)*100
        print(f"    {yr}: N={len(g):4d}  WR={wr:5.1f}%  R={g['pnl_r'].sum():+6.1f}")

    pu.plot_equity(cl, os.path.join(OUT, "equity_best.png"), f"Clean 4H sweep + 15M IFVG · {best['label']}")
    print(f"\nSaved to {OUT}/")
    mt5.shutdown()


if __name__ == "__main__":
    main()
