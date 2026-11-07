"""
Deeper leak-hunt on the EURUSD dry-basic losers.
Reloads M15/H4 to compute excursions + HTF context per trade.
"""
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import main as bt

PIP = 0.0001
CSV = os.path.join("strategies", "eurusd_basic", "trades.csv")
tr = pd.read_csv(CSV, parse_dates=["sweep_time", "entry_time", "exit_time"])
tr = tr[tr["outcome"] != "open"].copy().reset_index(drop=True)

mt5.initialize()
df15 = bt.fetch_data("EURUSD", mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
df4  = bt.fetch_data("EURUSD", mt5.TIMEFRAME_H4,  bt.START_DATE, bt.END_DATE, tf_name="H4")
mt5.shutdown()

h = df15["high"].values; l = df15["low"].values; c = df15["close"].values; idx15 = df15.index
# 15M ATR(14)
tr_range = np.maximum(h[1:] - l[1:],
             np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
atr = pd.Series(tr_range).rolling(14).mean().values
atr = np.concatenate([[np.nan], atr])
# 4H EMA50 for HTF trend
ema50 = df4["close"].ewm(span=50, adjust=False).mean()
# rolling 20-bar 4H equilibrium for premium/discount
roll_hi = df4["high"].rolling(20).max()
roll_lo = df4["low"].rolling(20).min()

mfe_r = []; mae_r = []; went_green = []; htf_align = []; pd_zone = []; rr_vs_atr = []; pen_p = []
for _, t in tr.iterrows():
    ei = int(t["entry_idx"]); entry = t["entry"]; sl = t["sl"]; tp = t["tp"]
    risk = abs(entry - sl); direction = t["direction"]
    xi = int(np.searchsorted(idx15, t["exit_time"], side="left")); xi = min(xi, len(df15) - 1)
    seg_h = h[ei + 1:xi + 1]; seg_l = l[ei + 1:xi + 1]
    if len(seg_h) == 0:
        mfe_r.append(0); mae_r.append(0); went_green.append(False)
    else:
        if direction == "long":
            mfe = (seg_h.max() - entry) / risk; mae = (entry - seg_l.min()) / risk
        else:
            mfe = (entry - seg_l.min()) / risk; mae = (seg_h.max() - entry) / risk
        mfe_r.append(mfe); mae_r.append(mae); went_green.append(mfe >= 1.0)

    # HTF trend alignment at sweep
    s4 = min(int(np.searchsorted(df4.index, t["sweep_time"], side="left")), len(df4) - 1)
    up = df4["close"].iloc[s4] > ema50.iloc[s4]
    htf_align.append("with" if ((direction == "long") == up) else "against")

    # premium/discount: where entry sits in the rolling 4H range
    hi = roll_hi.iloc[s4]; lo = roll_lo.iloc[s4]
    if pd.notna(hi) and hi > lo:
        loc = (entry - lo) / (hi - lo)   # 0=bottom(discount) 1=top(premium)
        if direction == "long":
            pd_zone.append("discount(<0.5)" if loc < 0.5 else "premium(>0.5)")
        else:
            pd_zone.append("premium(>0.5)" if loc > 0.5 else "discount(<0.5)")
    else:
        pd_zone.append("na")

    reward = abs(tp - entry)
    a = atr[ei] if ei < len(atr) and not np.isnan(atr[ei]) else np.nan
    rr_vs_atr.append(reward / a if a and a > 0 else np.nan)
    pen_p.append(abs(entry - t["swept_level"]) / PIP)  # proxy

tr["mfe_r"] = mfe_r; tr["mae_r"] = mae_r; tr["went_green"] = went_green
tr["htf_align"] = htf_align; tr["pd_zone"] = pd_zone; tr["rr_vs_atr"] = rr_vs_atr
tr["is_win"] = (tr["outcome"] == "win").astype(int)
tr["hour"] = tr["entry_time"].dt.hour
tr["risk_p"] = (tr["entry"] - tr["sl"]).abs() / PIP
tr["ext_p"] = (tr["entry"] - tr["swept_level"]).abs() / PIP
base = tr["is_win"].mean() * 100
N = len(tr)
print("=" * 86)
print(f"DEEPER LEAK HUNT · {N} trades · baseline WR {base:.1f}%")
print("=" * 86)


def buck(col, bins=None, labels=None, cat=False):
    d = tr.copy()
    d["_b"] = d[col] if cat else pd.cut(d[col], bins=bins, labels=labels, include_lowest=True)
    g = d.groupby("_b", observed=True).agg(N=("is_win", "size"), WR=("is_win", "mean"))
    g["WR"] *= 100
    return g[g["N"] >= 40].sort_values("WR")


def show(title, g):
    print(f"\n── {title} ──  (base {base:.1f}%)")
    for b, r in g.iterrows():
        f = "  <<<" if r["WR"] < base - 5 else ("  (+)" if r["WR"] > base + 5 else "")
        print(f"    {str(b):24} N={int(r['N']):4d}  WR={r['WR']:5.1f}%{f}")


# 1. HTF trend alignment (reversal WITH vs AGAINST 4H trend)
show("HTF trend alignment (vs 4H EMA50)", buck("htf_align", cat=True))
# 2. Premium/discount at entry
show("Premium/Discount zone at entry", buck("pd_zone", cat=True))
# 3. Reward size vs ATR (is 2R unrealistic vs normal move?)
show("Reward ÷ ATR14", buck("rr_vs_atr",
     bins=[0,1,2,3,5,8,100], labels=["<1","1-2","2-3","3-5","5-8","8+"]))
# 4. Sweep side (BSL high sweep vs SSL low sweep)
tr["sweep_side"] = np.where(tr["bias"] == "bearish", "high(BSL)", "low(SSL)")
show("Sweep side", buck("sweep_side", cat=True))

# ── MFE analysis: how close did LOSERS get to TP? ──
losers = tr[tr["is_win"] == 0]
print("\n── LOSER excursions (how far they traveled before dying) ──")
print(f"    losers that reached >=1R MFE before losing : "
      f"{(losers['mfe_r']>=1).mean()*100:4.1f}%  ({(losers['mfe_r']>=1).sum()} of {len(losers)})")
print(f"    losers that reached >=1.5R MFE            : {(losers['mfe_r']>=1.5).mean()*100:4.1f}%")
print(f"    losers that NEVER went green (<0R MFE)    : {(losers['mfe_r']<=0.05).mean()*100:4.1f}%")
print(f"    median loser MFE                          : {losers['mfe_r'].median():.2f}R")
print(f"    median winner MAE (worst dip before win)  : {tr[tr['is_win']==1]['mae_r'].median():.2f}R")

# how many losers reached >=1R then reversed -> a partial/BE stop would save them
save = (losers["mfe_r"] >= 1.0).sum()
print(f"\n    -> {save} losers ({save/len(losers)*100:.0f}%) reached +1R before hitting SL.")
print(f"       A 'move SL to break-even at +1R' rule would turn those into BE, not losses.")

# ── 2-way interaction: risk bucket x extension ──
print("\n── Interaction: SL distance × entry extension ──  (base {:.1f}%)".format(base))
tr["risk_b"] = pd.cut(tr["risk_p"], [0,3,6,1000], labels=["tight<3","mid3-6","wide6+"])
tr["ext_b"]  = pd.cut(tr["ext_p"],  [0,10,20,1000], labels=["near<10","mid10-20","ext20+"])
piv = tr.pivot_table(index="risk_b", columns="ext_b", values="is_win", aggfunc="mean", observed=True) * 100
cnt = tr.pivot_table(index="risk_b", columns="ext_b", values="is_win", aggfunc="size", observed=True)
print("    WR% (N):")
print("    " + "".join(f"{str(c):>16}" for c in piv.columns))
for r in piv.index:
    row = "    " + f"{str(r):8}"
    for cc in piv.columns:
        row += f"  {piv.loc[r,cc]:5.1f}%({int(cnt.loc[r,cc]):4d})"
    print(row)

# ── worst combined cell ──
print("\n── 'went green then lost' by hour (gave-back losses) ──")
gb = losers[losers["mfe_r"] >= 1.0]
g = gb.groupby(gb["hour"]).size().sort_values(ascending=False).head(6)
for hr, n in g.items():
    print(f"    hour {int(hr):02d}: {n} give-back losses")

# ── Save enriched log ──
OUT = os.path.join("strategies", "eurusd_basic")
tr.to_csv(os.path.join(OUT, "trades_enriched.csv"), index=False)
print(f"\nEnriched log saved → {OUT}/trades_enriched.csv")
