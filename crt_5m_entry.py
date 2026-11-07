"""
CRT bias (1H/4H/1D) → 5-minute CISD + IFVG entry → FIXED RR. Strictly causal.

  1. HTF CRT: candle B sweeps candle A's extreme & closes back inside → bias.
  2. During candle C's window, drop to 5m and find a DISPLACEMENT that:
       - leaves an FVG in the bias direction (bearish: high[k] < low[k-2]),
       - is a CISD (close breaks the prior 5m structure low/high),
     then ENTER on the IFVG retrace (price trades back into the gap).
  3. SL = local swing (origin of the displacement, +spread+1pip). TP = fixed R.
Tested per HTF, R = 2 and 3, costs + degradation, era-split survivor gate.
"""
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import frequency_tune as fq
import realistic_ema_dol_rsi as edr
import all_pairs_2000_2026 as ap

edr.SL_SPREAD_PAD = True
SYMS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "EURJPY", "GBPJPY", "AUDJPY",
        "XAUUSD", "US500.cash", "US100.cash", "GER40.cash", "UK100.cash",
        "USOIL.cash", "UKOIL.cash"]
SPLITS = [("2000-2011", "2000-01-01", "2012-01-01"),
          ("2012-2018", "2012-01-01", "2019-01-01"),
          ("2019-2026", "2019-01-01", "2027-01-01")]
HTFS = [("1H", "1h", np.timedelta64(1, "h"), mt5.TIMEFRAME_H1),
        ("4H", "4h", np.timedelta64(4, "h"), mt5.TIMEFRAME_H4),
        ("1D", "1D", np.timedelta64(1, "D"), mt5.TIMEFRAME_D1)]
OHLC = {"open": "first", "high": "max", "low": "min", "close": "last"}
FVG_WAIT = 60         # 5m bars after CISD to fill the IFVG retrace
MAX_HOLD = 288        # 5m bars to resolve (24h)


def load5(sym, mtf):
    p = os.path.join("ext_data", f"HD_{sym}_5min_2000-2024.pkl")
    if not os.path.exists(p):
        return None
    m5 = pd.read_pickle(p)
    tail = ap.mt5_bars(ap.U[sym][1], mtf, n=99999) if mtf else None
    if tail is not None:
        m5 = pd.concat([m5[m5.index < "2025-01-01"], tail[tail.index >= "2025-01-01"]])
    return m5


def run(sym, m5, rule, dur, rr):
    spread, slip, _ = fq.SPEC[sym]; pad = edr.sl_pad(sym, spread)
    o5, h5, l5, c5 = m5.open.values, m5.high.values, m5.low.values, m5.close.values
    t5 = m5.index.values
    htf = m5.resample(rule).agg(OHLC).dropna()
    Ho, Hh, Hl, Hc = htf.open.values, htf.high.values, htf.low.values, htf.close.values
    Ht = htf.index.values
    trades = []; last_exit_t = np.datetime64("2000-01-01")
    for i in range(len(htf) - 3):
        Ah, Al = Hh[i], Hl[i]; Bh, Bl, Bc = Hh[i + 1], Hl[i + 1], Hc[i + 1]
        rng = Ah - Al
        if rng <= 0:
            continue
        bias = None
        if Bh > Ah and Al < Bc < Ah:
            bias = "short"
        elif Bl < Al and Al < Bc < Ah:
            bias = "long"
        if bias is None:
            continue
        c_open_t = Ht[i + 2]
        if c_open_t < last_exit_t:
            continue
        w0 = np.searchsorted(t5, c_open_t)
        w1 = np.searchsorted(t5, c_open_t + 2 * dur)      # window = 2 HTF candles
        if w1 - w0 < 5:
            continue
        got = None
        for k in range(w0 + 3, min(w1, len(m5))):
            if bias == "short":
                # bearish FVG + displacement + CISD (broke recent 5m low)
                if h5[k] < l5[k - 2] and c5[k] < o5[k] and c5[k] < l5[max(k - 4, 0):k].min():
                    for r in range(k + 1, min(k + FVG_WAIT, len(m5))):
                        if h5[r] >= h5[k]:                # IFVG retrace into the gap
                            entry = h5[k] - slip
                            sl = h5[max(k - 4, 0):r + 1].max() + pad
                            risk = sl - entry
                            if risk > 0:
                                got = ("short", r, entry, sl, entry - rr * risk, risk)
                            break
                    if got:
                        break
            else:
                if l5[k] > h5[k - 2] and c5[k] > o5[k] and c5[k] > h5[max(k - 4, 0):k].max():
                    for r in range(k + 1, min(k + FVG_WAIT, len(m5))):
                        if l5[r] <= l5[k]:
                            entry = l5[k] + spread + slip
                            sl = l5[max(k - 4, 0):r + 1].min() - pad
                            risk = entry - sl
                            if risk > 0:
                                got = ("long", r, entry, sl, entry + rr * risk, risk)
                            break
                    if got:
                        break
        if not got:
            continue
        d, r, entry, sl, tp, risk = got
        res = None
        for x in range(r + 1, min(r + MAX_HOLD, len(m5))):
            if d == "short":
                if h5[x] + spread >= sl: res = (x, -1.0); break
                if l5[x] + spread <= tp: res = (x, rr); break
            else:
                if l5[x] <= sl: res = (x, -1.0); break
                if h5[x] >= tp: res = (x, rr); break
        if res is None:
            x = min(r + MAX_HOLD, len(m5)) - 1
            res = (x, ((entry - (c5[x] + spread)) if d == "short" else (c5[x] - entry)) / risk)
        last_exit_t = t5[res[0]]
        trades.append(dict(symbol=sym, entry_time=m5.index[r], exit_time=m5.index[res[0]],
                           direction=d, pnl_r=res[1], spread_r=spread / risk,
                           outcome="win" if res[1] > 0.01 else ("loss" if res[1] < -0.01 else "be")))
    return pd.DataFrame(trades)


def degrade(df, seed=9):
    if not len(df):
        return df
    rng = np.random.default_rng(seed); d = df.reset_index(drop=True).copy()
    d = d[rng.random(len(d)) >= 0.10].reset_index(drop=True)
    r = d.pnl_r.to_numpy(float); sr = d.spread_r.to_numpy(float)
    r = r - 0.3 * sr
    r = r - np.where(rng.random(len(d)) < 0.15, 1.5 * sr, 0.0)
    r = r - np.where(r <= -0.90, 0.5 * sr, 0.0)
    r = np.where(rng.random(len(d)) < 0.002, -3.0, r)
    d["pnl_r"] = r
    d["outcome"] = np.where(r > 0.01, "win", np.where(r < -0.01, "loss", "be"))
    return d


def main():
    have = mt5.initialize()
    m5data = {s: load5(s, dict(HTFS)[0] if False else None) for s in SYMS}   # placeholder
    m5data = {}
    for s in SYMS:
        b = load5(s, mt5.TIMEFRAME_M5)
        if b is not None and len(b) > 5000:
            m5data[s] = b
    print("=" * 106)
    print("CRT bias → 5m CISD+IFVG entry → FIXED RR · causal · costs+degradation · per HTF")
    print("=" * 106, flush=True)
    for tfn, rule, dur, _ in HTFS:
        for rr in (2.0, 3.0):
            allt = [run(s, m5data[s], rule, dur, rr) for s in m5data]
            raw = pd.concat([x for x in allt if len(x)], ignore_index=True)
            if not len(raw):
                print(f"  {tfn} {rr}R: no trades"); continue
            raw["entry_time"] = pd.to_datetime(raw["entry_time"])
            deg = degrade(raw.sort_values("entry_time"))
            st = edr.stats(deg); yrs = (raw.entry_time.max() - raw.entry_time.min()).days / 365.25
            print(f"\n  ══ {tfn} · {rr:.0f}R ══  N={st['n']} ({st['n']/yrs:.0f}/yr) · WR {st['wr']:.1f}% "
                  f"(BE {100/(1+rr):.0f}%) · PF {st['pf']:.2f} · {st['tot']:+.0f}R · DD {st['dd']:.0f}% · "
                  f"${st['bal']:,.0f}", flush=True)
            ok = True
            for name, a, b in SPLITS:
                g = deg[(deg.entry_time >= a) & (deg.entry_time < b)]; s2 = edr.stats(g)
                if s2 is None or s2["n"] < 30:
                    print(f"      {name}: few"); ok = False; continue
                print(f"      {name}: N={s2['n']:>4} WR {s2['wr']:>4.0f}% PF {s2['pf']:>5.2f} {s2['tot']:>+6.0f}R")
                if s2["pf"] < 1.25:
                    ok = False
            v = "✅ SURVIVOR" if ok and st["pf"] >= 1.3 else "❌ no edge"
            print(f"      VERDICT: {v}", flush=True)
    if have:
        mt5.shutdown()


if __name__ == "__main__":
    main()
