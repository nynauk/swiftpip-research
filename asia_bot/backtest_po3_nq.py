"""
Does the 10AM PO3 PREMISE hold on NQ? — hourly futures, 2024-02 → 2026-07.

This does NOT test the 1m CISD entry (hourly bars are far too coarse). It tests
the model's core claim, which is directional and testable on H1:

  "After the 10AM EST 4H candle opens, price MANIPULATES (sweeps the pre-10AM
   range), then DELIVERS in the opposite direction toward opposing liquidity."

Method (all causal):
  • accumulation range = 07:00-10:00 ET high/low (the liquidity pool)
  • from 10:00 ET, find the first H1 bar that sweeps that high or low
  • PO3 says: sweep the HIGH -> deliver DOWN ; sweep the LOW -> deliver UP
  • enter at that bar's close, SL beyond the sweep extreme (+pad), and measure:
        - fixed 2R / 3R outcomes
        - opposing-liquidity target (other side of the accumulation range)
  • also reports the pure directional stat: how often the post-sweep close
    (session end 16:00 ET) is on the delivery side — a filter-free read on
    whether the reversal premise has ANY signal.

    python -m asia_bot.backtest_po3_nq
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HD = os.path.join("ext_data", "FUT_NQ_1h.pkl")
ACC_START, ACC_END = 7, 10          # ET
NY_END = 16                         # cash close
SPREAD = 1.5                        # index points, round trip-ish
PAD = 2.0                           # SL pad in points


def main():
    df = pd.read_pickle(HD)
    df.columns = [str(c).lower() for c in df.columns]
    idx = df.index                                   # tz-aware America/New_York
    print("=" * 96)
    print(f"10AM PO3 PREMISE TEST · NQ futures H1 · {idx[0].date()} → {idx[-1].date()} (ET, DST-exact)")
    print("H1 is too coarse for the 1m CISD trigger — this tests the sweep→deliver PREMISE only")
    print("=" * 96)

    o = df["open"].values; h = df["high"].values
    l = df["low"].values;  c = df["close"].values
    dates = pd.DatetimeIndex(idx).date
    hours = pd.DatetimeIndex(idx).hour

    rows = []
    for d in sorted(set(dates)):
        m = dates == d
        ii = np.where(m)[0]
        if len(ii) < 8: continue
        acc = ii[(hours[ii] >= ACC_START) & (hours[ii] < ACC_END)]
        ny  = ii[(hours[ii] >= ACC_END) & (hours[ii] <= NY_END)]
        if len(acc) < 2 or len(ny) < 3: continue
        hi, lo = h[acc].max(), l[acc].min()

        # first sweep of the accumulation range after 10:00
        sw = None
        for k in ny:
            if h[k] > hi: sw = ("sell", k, h[k]); break     # swept high -> deliver DOWN
            if l[k] < lo: sw = ("buy",  k, l[k]); break     # swept low  -> deliver UP
        if sw is None: continue
        direction, k, ext = sw
        if k == ny[-1]: continue

        entry = c[k]
        if direction == "sell":
            sl = ext + PAD; risk = sl - entry; opp = lo
            if risk <= 0 or opp >= entry: continue
        else:
            sl = ext - PAD; risk = entry - sl; opp = hi
            if risk <= 0 or opp <= entry: continue

        fwd = ny[ny > k]
        res = {"date": str(d), "year": pd.Timestamp(d).year, "dir": direction,
               "risk_pts": round(risk, 1)}
        # fixed R targets + opposing liquidity
        for tag, rr in (("r2", 2.0), ("r3", 3.0)):
            tp = entry - rr*risk if direction == "sell" else entry + rr*risk
            out = 0.0
            for j in fwd:
                if direction == "sell":
                    if h[j] + SPREAD >= sl: out = -1.0; break
                    if l[j] + SPREAD <= tp: out = rr; break
                else:
                    if l[j] <= sl: out = -(1 + SPREAD/risk); break
                    if h[j] >= tp: out = rr - SPREAD/risk; break
            res[tag] = out
        # opposing liquidity
        rr_opp = abs(entry - opp) / risk
        out = 0.0
        for j in fwd:
            if direction == "sell":
                if h[j] + SPREAD >= sl: out = -1.0; break
                if l[j] + SPREAD <= opp: out = rr_opp; break
            else:
                if l[j] <= sl: out = -(1 + SPREAD/risk); break
                if h[j] >= opp: out = rr_opp - SPREAD/risk; break
        res["opp"] = out; res["rr_opp"] = round(rr_opp, 2)
        # pure direction: did the session close on the delivery side of entry?
        close_end = c[fwd[-1]]
        res["dir_ok"] = (close_end < entry) if direction == "sell" else (close_end > entry)
        rows.append(res)

    t = pd.DataFrame(rows)
    if not len(t):
        print("no setups"); return

    print(f"\n  sweeps found: {len(t)}  ({(t['dir']=='sell').sum()} after high-sweep, "
          f"{(t['dir']=='buy').sum()} after low-sweep)")
    print(f"  median stop: {t['risk_pts'].median():.0f} pts · median opp-RR: {t['rr_opp'].median():.2f}")

    print(f"\n  {'target':22} {'N':>4} {'WR':>6} {'PF':>5} {'totR':>8} {'exp':>9}")
    for tag, name in (("r2", "fixed 2R"), ("r3", "fixed 3R"), ("opp", "opposing liquidity")):
        v = t[t[tag] != 0.0]
        if not len(v): continue
        n = len(v); w = (v[tag] > 0).sum(); R = v[tag].sum()
        gw = v.loc[v[tag] > 0, tag].sum(); gl = -v.loc[v[tag] < 0, tag].sum()
        pf = gw/gl if gl > 0 else float("inf")
        print(f"  {name:22} {n:>4} {w/n*100:>5.1f}% {pf:>5.2f} {R:>+8.1f} {R/n:>+8.3f}R")

    dok = t["dir_ok"].mean()*100
    print(f"\n  PREMISE CHECK — after the 10AM sweep, session closed on the DELIVERY side: "
          f"{dok:.1f}%  (coin-flip = 50%)")
    print(f"     high-swept→closed lower: {t[t['dir']=='sell']['dir_ok'].mean()*100:.1f}%")
    print(f"     low-swept →closed higher: {t[t['dir']=='buy']['dir_ok'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
