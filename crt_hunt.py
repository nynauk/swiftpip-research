"""
CRT (Candle Range Theory / Power-of-Three) — causal backtest, 1H / 4H / 1D SEPARATELY.

Per timeframe, for each consecutive triplet A(i), B(i+1), C(i+2):
  bearish: B.high > A.high AND B closes back INSIDE A's range  → at C.open go SHORT,
           TP = A.low (opposite extreme / draw on liquidity), SL = B.high (+spread+1pip)
  bullish: B.low  < A.low  AND B closes back inside            → at C.open go LONG,
           TP = A.high, SL = B.low (-spread-1pip)
Everything causal: A and B are complete before C opens; SL/TP known at entry.
One position at a time. Per-instrument spread on every fill + execution degradation.
Survivor gate: PF >= 1.25 in EVERY era. HTF spread drag is small — the fair test.
"""
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import frequency_tune as fq
import realistic_ema_dol_rsi as edr
import find_edge as fe

SYMS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "EURJPY", "GBPJPY", "AUDJPY",
        "XAUUSD", "US500.cash", "US100.cash", "GER40.cash", "UK100.cash",
        "USOIL.cash", "UKOIL.cash"]
SPLITS = [("2000-2011", "2000-01-01", "2012-01-01"),
          ("2012-2018", "2012-01-01", "2019-01-01"),
          ("2019-2026", "2019-01-01", "2027-01-01")]
TFS = [("1H", "1h", mt5.TIMEFRAME_H1), ("4H", "4h", mt5.TIMEFRAME_H4),
       ("1D", "1D", mt5.TIMEFRAME_D1)]
MAX_HOLD = 3          # candles to resolve (A→B→C; distribution should hit fast)


def run_crt(sym, df):
    spread, slip, _ = fq.SPEC[sym]; pad = edr.sl_pad(sym, spread)
    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    idx = df.index; N = len(df)
    trades = []; busy = -1
    for i in range(N - 3):
        ci = i + 2
        if ci <= busy:
            continue
        Ah, Al = h[i], l[i]
        Bh, Bl, Bc = h[i + 1], l[i + 1], c[i + 1]
        if Ah - Al <= 0:
            continue
        # bearish CRT
        rng = Ah - Al
        if Bh > Ah and Al < Bc < Ah:
            entry = o[ci] - slip; sl = Bh + pad; tp = Al; risk = sl - entry
            if risk <= 0 or tp >= entry or risk < 0.20 * rng:   # skip degenerate near-zero stops
                continue
            res = None
            for x in range(ci, min(ci + MAX_HOLD, N)):
                if h[x] + spread >= sl: res = (x, -1.0); break
                if l[x] + spread <= tp: res = (x, (entry - tp) / risk); break
            if res is None:
                x = min(ci + MAX_HOLD, N) - 1; res = (x, (entry - (c[x] + spread)) / risk)
            busy = res[0]
            trades.append(dict(symbol=sym, entry_time=idx[ci], exit_time=idx[res[0]],
                               direction="short", pnl_r=res[1], rr=(entry - tp) / risk,
                               spread_r=spread / risk,
                               outcome="win" if res[1] > 0.01 else ("loss" if res[1] < -0.01 else "be")))
        # bullish CRT
        elif Bl < Al and Al < Bc < Ah:
            entry = o[ci] + spread + slip; sl = Bl - pad; tp = Ah; risk = entry - sl
            if risk <= 0 or tp <= entry or risk < 0.20 * rng:
                continue
            res = None
            for x in range(ci, min(ci + MAX_HOLD, N)):
                if l[x] <= sl: res = (x, -1.0); break
                if h[x] >= tp: res = (x, (tp - entry) / risk); break
            if res is None:
                x = min(ci + MAX_HOLD, N) - 1; res = (x, (c[x] - entry) / risk)
            busy = res[0]
            trades.append(dict(symbol=sym, entry_time=idx[ci], exit_time=idx[res[0]],
                               direction="long", pnl_r=res[1], rr=(tp - entry) / risk,
                               spread_r=spread / risk,
                               outcome="win" if res[1] > 0.01 else ("loss" if res[1] < -0.01 else "be")))
    return pd.DataFrame(trades)


def degrade(df, seed=3):
    if not len(df):
        return df
    rng = np.random.default_rng(seed); d = df.reset_index(drop=True).copy()
    d = d[rng.random(len(d)) >= 0.05].reset_index(drop=True)      # HTF: easy fills, 5% miss
    r = d.pnl_r.to_numpy(float); sr = d.spread_r.to_numpy(float)
    r = r - 0.3 * sr
    r = r - np.where(rng.random(len(d)) < 0.15, 1.5 * sr, 0.0)
    r = r - np.where(r <= -0.90, 0.5 * sr, 0.0)
    r = np.where(rng.random(len(d)) < 0.002, -3.0, r)
    d["pnl_r"] = r
    d["outcome"] = np.where(r > 0.01, "win", np.where(r < -0.01, "loss", "be"))
    return d


def main():
    edr.SL_SPREAD_PAD = True
    have = mt5.initialize()
    print("=" * 104)
    print("CRT (Candle Range Theory) · causal · costs+degradation · 1H / 4H / 1D SEPARATELY")
    print("  survivor gate: PF >= 1.25 in EVERY era · target = opposite range extreme")
    print("=" * 104, flush=True)

    for tfn, rule, mtf in TFS:
        allt = []
        for s in SYMS:
            b = fe.load_tf(s, rule, mtf)
            if b is not None and len(b) > 300:
                allt.append(run_crt(s, b))
        raw = pd.concat([x for x in allt if len(x)], ignore_index=True)
        raw["entry_time"] = pd.to_datetime(raw["entry_time"])
        deg = degrade(raw.sort_values("entry_time"))
        st = edr.stats(deg); rr = raw.rr.median()
        yrs = (raw.entry_time.max() - raw.entry_time.min()).days / 365.25
        print(f"\n  ══ {tfn} ══   N={st['n']} ({st['n']/yrs:.0f}/yr) · median RR {rr:.2f} · "
              f"WR {st['wr']:.1f}% · PF {st['pf']:.2f} · {st['tot']:+.0f}R · DD {st['dd']:.0f}% · "
              f"end ${st['bal']:,.0f}", flush=True)
        ok = True
        for name, a, b in SPLITS:
            g = deg[(deg.entry_time >= a) & (deg.entry_time < b)]; s2 = edr.stats(g)
            if s2 is None or s2["n"] < 30:
                print(f"      {name}: few"); ok = False; continue
            print(f"      {name}: N={s2['n']:>5} WR {s2['wr']:>4.0f}% PF {s2['pf']:>5.2f} {s2['tot']:>+7.0f}R "
                  f"DD {s2['dd']:>5.0f}%")
            if s2["pf"] < 1.25:
                ok = False
        # best instruments this TF
        top = sorted([(s, edr.stats(deg[deg.symbol == s])) for s in SYMS
                      if len(deg[deg.symbol == s])], key=lambda x: -(x[1]["pf"] if x[1] else 0))[:5]
        print("      top instruments: " + " · ".join(
            f"{s} PF{st2['pf']:.2f}(n{st2['n']})" for s, st2 in top if st2), flush=True)
        v = "✅ SURVIVOR — worth developing" if ok and st["pf"] >= 1.3 else "❌ no edge after costs"
        print(f"      VERDICT: {v}", flush=True)
        os.makedirs(os.path.join("strategies", "crt"), exist_ok=True)
        deg.to_csv(os.path.join("strategies", "crt", f"crt_{tfn}.csv"), index=False)

    if have:
        mt5.shutdown()


if __name__ == "__main__":
    main()
