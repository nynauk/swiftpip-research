"""
EURUSD · dry basic setup — no filters, no tuning.

Rules (exactly as specified):
  1. Sweep = the PREVIOUS 4H candle's high/low gets swept (detected on 15M:
     price wicks past it and a 15M candle closes back inside).
        swept HIGH → SELL bias ,  swept LOW → BUY bias.
  2. After the sweep, find a 15M FVG (opposite the bias) that gets INVERTED
     (IFVG) — a 15M candle closes through it in the bias direction. That
     candle is the "disrespect" candle.
  3. ENTRY:
        • disrespect candle range <= 1.5 x FVG width → enter at its CLOSE.
        • disrespect candle range >  1.5 x FVG width → enter on the RETEST of
          the FVG edge (buy → FVG high, sell → FVG low).
  4. STOP LOSS = the disrespect candle's LOW (buy) / HIGH (sell).
  5. TARGET = 2R.
  No min-width filter, no session filter, no news filter. Pure setup.
"""
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import main as bt
import plot_utils as pu

SYMBOL      = "EURUSD"
REV_WINDOW  = 24     # 15M bars after sweep to find the reversal FVG
DIS_DELAY   = 7      # 15M bars for the FVG to invert
RETR_WINDOW = 48     # 15M bars to wait for the retest entry
DIS_MULT    = 1.5    # "too big" threshold
RR          = 2.0


def sweeps_prev4h(df_15, df_h4):
    """First sweep of the previous 4H candle's high/low, per 4H period, on 15M."""
    prevH = df_h4["high"].shift(1); prevL = df_h4["low"].shift(1)
    floor4h = df_15.index.floor("4h")
    ph = prevH.reindex(floor4h).values
    pl = prevL.reindex(floor4h).values
    h = df_15["high"].values; l = df_15["low"].values; c = df_15["close"].values
    idx = df_15.index
    sweeps = []; cur = None; hd = ld = False
    for i in range(len(df_15)):
        p = floor4h[i]
        if p != cur:
            cur = p; hd = ld = False
        if np.isnan(ph[i]):
            continue
        if not hd and h[i] > ph[i] and c[i] < ph[i]:
            sweeps.append({"time": idx[i], "direction": "bearish", "swept_level": float(ph[i])}); hd = True
        if not ld and l[i] < pl[i] and c[i] > pl[i]:
            sweeps.append({"time": idx[i], "direction": "bullish", "swept_level": float(pl[i])}); ld = True
    return sweeps


def simulate(h, l, idx, ei, direction, entry, sl, tp, rr,
             be_at=None, partial_at=None, partial_frac=0.5, max_bars=400):
    """
    Candle-by-candle. Optional trade management:
      be_at      : when profit reaches be_at * risk, move SL to break-even.
      partial_at : take partial_frac of the position at partial_at * risk,
                   and move SL to break-even on the remainder.
    Returns (outcome_label, exit_time, exit_price, pnl_r).
    Outcome: 'win' (R>0), 'be' (R≈0), 'loss' (R<0), 'open'.
    """
    risk = abs(entry - sl)
    sl_cur = sl
    armed = False
    realized = 0.0
    remaining = 1.0
    trig = be_at if be_at is not None else partial_at
    last = ei
    for j in range(ei + 1, min(ei + max_bars, len(h))):
        last = j
        if direction == "long":
            if l[j] <= sl_cur:
                r = realized + remaining * ((sl_cur - entry) / risk)
                return _lbl(r), idx[j], sl_cur, r
            if h[j] >= tp:
                r = realized + remaining * rr
                return "win", idx[j], tp, r
            if trig is not None and not armed and h[j] >= entry + trig * risk:
                if partial_at is not None:
                    realized += partial_frac * partial_at
                    remaining -= partial_frac
                sl_cur = entry; armed = True
        else:
            if h[j] >= sl_cur:
                r = realized + remaining * ((entry - sl_cur) / risk)
                return _lbl(r), idx[j], sl_cur, r
            if l[j] <= tp:
                r = realized + remaining * rr
                return "win", idx[j], tp, r
            if trig is not None and not armed and l[j] <= entry - trig * risk:
                if partial_at is not None:
                    realized += partial_frac * partial_at
                    remaining -= partial_frac
                sl_cur = entry; armed = True
    return "open", idx[last], float("nan"), realized


def _lbl(r):
    if r > 0.01:  return "win"
    if r < -0.01: return "loss"
    return "be"


def run(df_h4, df_15, rr=RR, require_4h_fvg=False, fvg_tol_pips=3.0,
        be_at=None, partial_at=None):
    sweeps = sweeps_prev4h(df_15, df_h4)
    fvgs = sorted(bt.find_all_fvgs(df_15), key=lambda f: f["form_idx"])
    fia = np.array([f["form_idx"] for f in fvgs])
    h = df_15["high"].values; l = df_15["low"].values; c = df_15["close"].values
    idx = df_15.index

    # 4H FVGs for the HTF confluence filter
    fvgs_4h = bt.find_all_fvgs(df_h4)
    tol = fvg_tol_pips * 0.0001
    h4h = df_h4["high"].values; h4l = df_h4["low"].values; h4c = df_h4["close"].values
    RECENT_4H = 30   # only FVGs formed within last 30 4H candles (~5 trading days)

    def has_4h_confluence(bias, si_bar, sweep_time):
        # sweep reach = the extreme the sweep candle poked to
        reach = h[si_bar] if bias == "bearish" else l[si_bar]
        want = "bearish" if bias == "bearish" else "bullish"   # high swept→bearish 4H FVG
        s4 = int(np.searchsorted(df_h4.index, sweep_time, "left"))
        for f4 in fvgs_4h:
            ff = f4["form_idx"]
            if ff >= s4:                       # formed at/after sweep → ignore
                continue
            if s4 - ff > RECENT_4H:            # too old → ignore (recency)
                continue
            if f4["direction"] != want:
                continue
            # sweep reach must enter the FVG zone (strict, no tolerance)
            if not (f4["bottom"] <= reach <= f4["top"]):
                continue
            # FVG must still be UNMITIGATED at the sweep: for a bearish FVG
            # (overhead resistance), price must not have already closed above its
            # top between formation and the sweep. Mirror for bullish.
            seg = h4c[ff + 1:s4]
            if want == "bearish":
                if len(seg) and (seg > f4["top"]).any():
                    continue
            else:
                if len(seg) and (seg < f4["bottom"]).any():
                    continue
            return True
        return False

    trades = []; used = set(); open_until = []

    for sw in sweeps:
        bias = sw["direction"]; sig = (sw["time"], bias)
        if sig in used:
            continue
        si = int(np.searchsorted(idx, sw["time"], "left"))
        if si >= len(idx) or (si == 0 and sw["time"] < idx[0]):
            continue
        if require_4h_fvg and not has_4h_confluence(bias, si, sw["time"]):
            continue
        tdir = "bearish" if bias == "bullish" else "bullish"
        rev_end = min(si + REV_WINDOW, len(idx) - 1)
        lo = int(np.searchsorted(fia, si, "left")); hi = int(np.searchsorted(fia, rev_end, "right"))

        done = False
        for p in range(hi - 1, lo - 1, -1):     # nearest the sweep first
            fvg = fvgs[p]
            if fvg["direction"] != tdir or fvg["width"] <= 0:
                continue
            fi = fvg["form_idx"]
            for di in range(fi + 1, min(fi + DIS_DELAY, len(idx) - 1) + 1):
                inv = c[di] > fvg["top"] if fvg["direction"] == "bearish" else c[di] < fvg["bottom"]
                if not inv:
                    continue

                dis_low, dis_high = l[di], h[di]
                dis_range = dis_high - dis_low
                too_big = dis_range > DIS_MULT * fvg["width"]

                if not too_big:
                    ei = di; entry = c[di]; etype = "close"
                else:
                    rend = min(di + RETR_WINDOW, len(idx) - 1); ei = None
                    for rj in range(di + 1, rend + 1):
                        if bias == "bullish" and l[rj] <= fvg["top"]:   ei = rj; entry = fvg["top"];    break
                        if bias == "bearish" and h[rj] >= fvg["bottom"]: ei = rj; entry = fvg["bottom"]; break
                    if ei is None:
                        break
                    etype = "retest"

                # SL = disrespect candle low (buy) / high (sell)
                if bias == "bullish":
                    direction = "long"; sl = dis_low; risk = entry - sl
                    tp = entry + rr * risk
                else:
                    direction = "short"; sl = dis_high; risk = sl - entry
                    tp = entry - rr * risk
                if risk <= 0:
                    break

                open_until = [t for t in open_until if t > idx[ei]]
                if len(open_until) >= 1:
                    done = True; break

                oc, xt, xp, pr = simulate(h, l, idx, ei, direction, entry, sl, tp, rr,
                                          be_at=be_at, partial_at=partial_at)
                open_until.append(pd.Timestamp(xt))
                trades.append({
                    "symbol": SYMBOL, "sweep_time": sw["time"], "bias": bias,
                    "swept_level": round(sw["swept_level"], 6),
                    "entry_time": idx[ei], "entry_idx": ei, "direction": direction,
                    "entry_type": etype, "entry": round(entry, 6), "sl": round(sl, 6),
                    "tp": round(tp, 6), "fvg_top": round(fvg["top"], 6),
                    "fvg_bottom": round(fvg["bottom"], 6),
                    "dis_low": round(dis_low, 6), "dis_high": round(dis_high, 6),
                    "outcome": oc, "exit": round(xp, 6) if not np.isnan(xp) else np.nan,
                    "exit_time": xt, "pnl_r": pr,
                })
                used.add(sig); done = True; break
            if done:
                break
    return pd.DataFrame(trades)


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    OUT = os.path.join("strategies", "eurusd_basic")
    os.makedirs(OUT, exist_ok=True)

    df_h4 = bt.fetch_data(SYMBOL, mt5.TIMEFRAME_H4,  bt.START_DATE, bt.END_DATE, tf_name="H4")
    df_15 = bt.fetch_data(SYMBOL, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
    weeks = (df_15.index[-1] - df_15.index[0]).days / 7

    def report(be_at=None, partial_at=None):
        df = run(df_h4, df_15, rr=3.0, be_at=be_at, partial_at=partial_at)
        cl = df[df["outcome"] != "open"].copy()
        n = len(cl)
        wins   = (cl["outcome"] == "win").sum()
        losses = (cl["outcome"] == "loss").sum()
        bes    = (cl["outcome"] == "be").sum()
        wr = wins / n * 100 if n else 0
        gw = cl.loc[cl["pnl_r"] > 0, "pnl_r"].sum()
        gl = -cl.loc[cl["pnl_r"] < 0, "pnl_r"].sum()
        pf = gw / gl if gl else float("inf")
        totR = cl["pnl_r"].sum()
        # compounded equity + streaks (loss streak = consecutive R<0 only)
        bal = 10000.0; eq = [bal]; clc = bcl = cw = bcw = 0
        for _, r in cl.iterrows():
            bal += bal * 0.01 * r["pnl_r"]; eq.append(bal)
            if r["pnl_r"] < -0.01:
                clc += 1; cw = 0; bcl = max(bcl, clc)
            else:                      # win OR break-even resets the loss run
                clc = 0
                if r["pnl_r"] > 0.01:
                    cw += 1; bcw = max(bcw, cw)
        eqa = np.array(eq); dd = ((eqa - np.maximum.accumulate(eqa)) / np.maximum.accumulate(eqa) * 100).min()
        return dict(df=df, cl=cl, n=n, wins=wins, losses=losses, bes=bes, wr=wr, pf=pf,
                    totR=totR, dd=dd, ws=bcw, ls=bcl, bal=bal)

    base = report()                    # plain 3RR
    be   = report(be_at=1.0)           # move SL to BE at +1R
    par  = report(partial_at=1.0)      # take 50% at +1R, rest runs to 3R (SL→BE)

    def col(r): return [r["n"], f"{r['wr']:.1f}%", f"{r['pf']:.2f}", f"{r['totR']:+.0f}R",
                        f"{r['totR']/r['n']:+.3f}R", f"{r['dd']:.1f}%", r["ls"], r["ws"],
                        f"{r['wins']}/{r['bes']}/{r['losses']}", f"${r['bal']:,.0f}"]
    rows = ["trades", "win rate", "profit factor", "total R", "expectancy/trd",
            "max drawdown", "LOSS STREAK", "win streak", "W/BE/L", "$10k →"]
    b, e, p = col(base), col(be), col(par)
    print("=" * 92)
    print(f"EURUSD · prev-4H sweep + 15M IFVG · RR=3 · break-even / partial at +1R · {weeks:.0f} weeks")
    print("=" * 92)
    print(f"  {'metric':16} {'plain 3RR':>16} {'3RR + BE@1R':>16} {'3RR + partial@1R':>18}")
    for i, name in enumerate(rows):
        print(f"  {name:16} {str(b[i]):>16} {str(e[i]):>16} {str(p[i]):>18}")

    OUT = os.path.join("strategies", "eurusd_3r_be")
    os.makedirs(OUT, exist_ok=True)
    be["df"].to_csv(os.path.join(OUT, "trades_be.csv"), index=False)
    par["df"].to_csv(os.path.join(OUT, "trades_partial.csv"), index=False)
    pu.plot_equity(be["cl"],  os.path.join(OUT, "equity_be.png"),
                   "EURUSD · 3RR + break-even @ +1R")
    pu.plot_equity(par["cl"], os.path.join(OUT, "equity_partial.png"),
                   "EURUSD · 3RR + partial 50% @ +1R")
    print(f"\n  Saved → {OUT}/  (trades_be.csv, trades_partial.csv, equity_be.png, equity_partial.png)")
    mt5.shutdown()


if __name__ == "__main__":
    main()
