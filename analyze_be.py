"""
Deep stats on break-even@1R and partial@1R management for the 3RR EURUSD setup.
Uses ONE fixed entry set → clean counterfactual across managements.
"""
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import main as bt
import eurusd_basic as eb

mt5.initialize()
df_h4 = bt.fetch_data("EURUSD", mt5.TIMEFRAME_H4,  bt.START_DATE, bt.END_DATE, tf_name="H4")
df_15 = bt.fetch_data("EURUSD", mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
mt5.shutdown()

h = df_15["high"].values; l = df_15["low"].values; idx = df_15.index

# One fixed entry set (plain run), then re-simulate each entry under 3 managements
base = eb.run(df_h4, df_15, rr=3.0)          # entries + plain outcome
print(f"Fixed entry set: {len(base)} trades\n")

def resim(mode):
    outs = []
    for _, t in base.iterrows():
        ei = int(t["entry_idx"]); d = t["direction"]
        e, sl, tp = t["entry"], t["sl"], t["tp"]
        be_at = 1.0 if mode == "BE" else None
        pa    = 1.0 if mode == "PARTIAL" else None
        oc, xt, xp, pr = eb.simulate(h, l, idx, ei, d, e, sl, tp, 3.0,
                                     be_at=be_at, partial_at=pa)
        outs.append((oc, pr, xt))
    r = base.copy()
    r["outcome"] = [o[0] for o in outs]
    r["pnl_r"]   = [o[1] for o in outs]
    r["exit_time"] = [o[2] for o in outs]
    return r

plain = resim("NONE"); be = resim("BE"); par = resim("PARTIAL")


def metrics(df):
    cl = df[df["outcome"] != "open"]
    n = len(cl)
    w = (cl["pnl_r"] > 0.01).sum(); lo = (cl["pnl_r"] < -0.01).sum(); b = n - w - lo
    wr = w / n * 100
    gw = cl.loc[cl["pnl_r"] > 0, "pnl_r"].sum(); gl = -cl.loc[cl["pnl_r"] < 0, "pnl_r"].sum()
    pf = gw / gl if gl else float("inf")
    tot = cl["pnl_r"].sum(); exp = cl["pnl_r"].mean(); sd = cl["pnl_r"].std()
    # equity, DD, streaks
    bal = 1e4; eq = [bal]; clc = bcl = cw = bcw = 0
    streaks = []
    for _, r in cl.iterrows():
        bal += bal * .01 * r["pnl_r"]; eq.append(bal)
        if r["pnl_r"] < -0.01:
            clc += 1; cw = 0
        else:
            if clc: streaks.append(clc)
            clc = 0
            if r["pnl_r"] > 0.01: cw += 1; bcw = max(bcw, cw)
        bcl = max(bcl, clc)
    if clc: streaks.append(clc)
    eqa = np.array(eq); peak = np.maximum.accumulate(eqa)
    ddser = (eqa - peak) / peak * 100
    maxdd = ddser.min()
    # longest underwater (bars in DD)
    uw = 0; longest_uw = 0
    for x in ddser:
        uw = uw + 1 if x < -0.01 else 0
        longest_uw = max(longest_uw, uw)
    return dict(n=n, w=w, be=b, lo=lo, wr=wr, pf=pf, tot=tot, exp=exp, sd=sd,
                sharpe=exp / sd if sd else 0, maxdd=maxdd, ls=bcl, ws=bcw,
                bal=bal, cl=cl, streaks=streaks, uw=longest_uw)


M = {"plain": metrics(plain), "BE@1R": metrics(be), "partial@1R": metrics(par)}

print("=" * 92)
print("3RR TRADE-MANAGEMENT DEEP STATS  ·  identical entry set  ·  EURUSD")
print("=" * 92)
labels = list(M.keys())
def row(name, fn):
    print(f"  {name:22}" + "".join(f"{fn(M[k]):>20}" for k in labels))
print(f"  {'metric':22}" + "".join(f"{k:>20}" for k in labels))
row("trades",           lambda m: m["n"])
row("wins / BE / losses", lambda m: f"{m['w']}/{m['be']}/{m['lo']}")
row("win rate",         lambda m: f"{m['wr']:.1f}%")
row("profit factor",    lambda m: f"{m['pf']:.2f}")
row("total R",          lambda m: f"{m['tot']:+.0f}R")
row("expectancy/trade", lambda m: f"{m['exp']:+.3f}R")
row("R std dev",        lambda m: f"{m['sd']:.2f}")
row("R Sharpe (exp/sd)", lambda m: f"{m['sharpe']:.3f}")
row("max drawdown",     lambda m: f"{m['maxdd']:.1f}%")
row("LONGEST loss streak", lambda m: m["ls"])
row("longest win streak",  lambda m: m["ws"])
row("longest underwater",  lambda m: f"{m['uw']} trades")
row("$10k →",           lambda m: f"${m['bal']:,.0f}")

# ── Counterfactual: what BE / partial did to each plain outcome ──
plain_out = np.where(plain["pnl_r"] > 0.01, "win", np.where(plain["pnl_r"] < -0.01, "loss", "be"))
be_out    = np.where(be["pnl_r"] > 0.01, "win", np.where(be["pnl_r"] < -0.01, "loss", "be"))
resc = int(((plain_out == "loss") & (be_out == "be")).sum())   # losers rescued to BE
clip = int(((plain_out == "win")  & (be_out == "be")).sum())   # winners clipped to BE
unch_w = int(((plain_out == "win")  & (be_out == "win")).sum())
unch_l = int(((plain_out == "loss") & (be_out == "loss")).sum())
print("\n── BE@1R counterfactual (vs plain, same trades) ──")
print(f"    losers rescued to break-even : {resc:4d}   (+{resc*1}R saved)")
print(f"    winners clipped to break-even: {clip:4d}   (-{clip*3}R given up)")
print(f"    winners kept full (+3R)      : {unch_w:4d}")
print(f"    losers unchanged (-1R)       : {unch_l:4d}")
print(f"    net R impact of BE           : {resc*1 - clip*3:+d}R  "
      f"({M['plain']['tot']:.0f} → {M['BE@1R']['tot']:.0f})")
print(f"    → for every winner clipped, {resc/clip:.1f} losers rescued")

# ── Streak length distribution ──
print("\n── Loss-streak length distribution (how often each run length happens) ──")
for name in ["plain", "BE@1R", "partial@1R"]:
    s = pd.Series(M[name]["streaks"])
    buckets = {"1-3": ((s>=1)&(s<=3)).sum(), "4-6": ((s>=4)&(s<=6)).sum(),
               "7-10": ((s>=7)&(s<=10)).sum(), "11-15": ((s>=11)&(s<=15)).sum(),
               "16+": (s>=16).sum()}
    print(f"    {name:12}: " + "  ".join(f"{k}:{v}" for k, v in buckets.items())
          + f"   (max {M[name]['ls']})")

# ── Per-year ──
print("\n── Per-year total R ──")
for name, df in [("plain", plain), ("BE@1R", be), ("partial@1R", par)]:
    df = df.copy(); df["yr"] = pd.to_datetime(df["entry_time"]).dt.year
    yr = df.groupby("yr")["pnl_r"].sum()
    print(f"    {name:12}: " + "  ".join(f"{int(y)}:{v:+.0f}" for y, v in yr.items()))

# ── Monthly consistency ──
print("\n── Monthly consistency ──")
for name, df in [("plain", plain), ("BE@1R", be), ("partial@1R", par)]:
    df = df.copy(); df["m"] = pd.to_datetime(df["entry_time"]).dt.to_period("M")
    mo = df.groupby("m")["pnl_r"].sum()
    pos = (mo > 0).sum(); tot = len(mo)
    print(f"    {name:12}: {pos}/{tot} green months ({pos/tot*100:.0f}%)   "
          f"best {mo.max():+.1f}R   worst {mo.min():+.1f}R   avg {mo.mean():+.2f}R")

# ── Partial@1R outcome breakdown ──
print("\n── Partial@1R R-value breakdown ──")
pv = par[par["outcome"] != "open"]["pnl_r"].round(2).value_counts().sort_index()
for rv, cnt in pv.items():
    print(f"    R={rv:+.2f} : {cnt}")
