"""Find patterns in the losing trades of the EURUSD dry-basic strategy."""
import os
import numpy as np
import pandas as pd

CSV = os.path.join("strategies", "eurusd_basic", "trades.csv")
df = pd.read_csv(CSV, parse_dates=["sweep_time", "entry_time", "exit_time"])
df = df[df["outcome"] != "open"].copy().reset_index(drop=True)
PIP = 0.0001

# ── Derived features ─────────────────────────────────────────────────────────
df["hour"]  = df["entry_time"].dt.hour                 # broker time (GMT+3)
df["dow"]   = df["entry_time"].dt.day_name()
df["month"] = df["entry_time"].dt.month
df["fvg_width_p"] = (df["fvg_top"] - df["fvg_bottom"]) / PIP
df["risk_p"]      = (df["entry"] - df["sl"]).abs() / PIP
df["dis_range_p"] = (df["dis_high"] - df["dis_low"]) / PIP
df["dis_mult"]    = (df["dis_high"] - df["dis_low"]) / (df["fvg_top"] - df["fvg_bottom"])
df["dur_min"]     = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 60
df["sweep_to_entry_min"] = (df["entry_time"] - df["sweep_time"]).dt.total_seconds() / 60
# distance entry sits from the swept level (how far price reversed before entry)
df["entry_vs_sweep_p"] = (df["entry"] - df["swept_level"]).abs() / PIP
df["is_win"] = (df["outcome"] == "win").astype(int)

N = len(df); base_wr = df["is_win"].mean() * 100
print("=" * 84)
print(f"LOSS PATTERN ANALYSIS · {N} trades · baseline WR {base_wr:.1f}%  "
      f"({df['is_win'].sum()}W / {(~df['is_win'].astype(bool)).sum()}L)")
print("=" * 84)


def by_bucket(col, bins=None, labels=None, cat=False, top=None):
    d = df.copy()
    if cat:
        d["_b"] = d[col]
    else:
        d["_b"] = pd.cut(d[col], bins=bins, labels=labels, include_lowest=True)
    g = d.groupby("_b", observed=True).agg(N=("is_win", "size"), WR=("is_win", "mean"))
    g["WR"] = g["WR"] * 100
    g = g[g["N"] >= 30]   # ignore tiny buckets
    return g.sort_values("WR")


def show(title, g):
    print(f"\n── {title} ──   (baseline {base_wr:.1f}%)")
    for b, r in g.iterrows():
        flag = "  <<< loss-heavy" if r["WR"] < base_wr - 5 else ("  (strong)" if r["WR"] > base_wr + 5 else "")
        print(f"    {str(b):22}  N={int(r['N']):4d}  WR={r['WR']:5.1f}%{flag}")


# 1. Entry type & direction
show("Entry type", by_bucket("entry_type", cat=True))
show("Direction", by_bucket("direction", cat=True))

# 2. Time of day (broker GMT+3)
show("Entry hour (broker/GMT+3)", by_bucket("hour", cat=True))

# 3. Session grouping
def sess(hr):
    if 0 <= hr < 8:   return "Asia 00-08"
    if 8 <= hr < 13:  return "London 08-13"
    if 13 <= hr < 17: return "NY-AM 13-17"
    if 17 <= hr < 22: return "NY-PM 17-22"
    return "Late 22-24"
df["session"] = df["hour"].map(sess)
show("Session", by_bucket("session", cat=True))

# 4. Day of week
show("Day of week", by_bucket("dow", cat=True))

# 5. FVG width
show("FVG width (pips)", by_bucket("fvg_width_p",
     bins=[0,1,2,3,5,8,1000], labels=["0-1","1-2","2-3","3-5","5-8","8+"]))

# 6. Risk / SL distance
show("Risk = SL distance (pips)", by_bucket("risk_p",
     bins=[0,2,4,6,10,15,1000], labels=["0-2","2-4","4-6","6-10","10-15","15+"]))

# 7. Disrespect candle multiple (how big the disrespect candle was)
show("Disrespect mult (candle/FVG)", by_bucket("dis_mult",
     bins=[0,1,1.5,2.5,4,8,1000], labels=["<1","1-1.5","1.5-2.5","2.5-4","4-8","8+"]))

# 8. How far price reversed before entry (entry vs swept level)
show("Entry distance from swept level (pips)", by_bucket("entry_vs_sweep_p",
     bins=[0,3,6,10,15,25,1000], labels=["0-3","3-6","6-10","10-15","15-25","25+"]))

# 9. Sweep→entry delay
show("Sweep→entry delay (min)", by_bucket("sweep_to_entry_min",
     bins=[0,30,60,120,240,480,100000], labels=["0-30","30-60","1-2h","2-4h","4-8h","8h+"]))

# 10. Year / month
show("Month", by_bucket("month", cat=True))

# ── Consecutive-loss clustering ──────────────────────────────────────────────
print("\n── Loss clustering ──")
outc = df.sort_values("entry_time")["outcome"].values
after_loss = []; after_win = []
for i in range(1, len(outc)):
    (after_loss if outc[i-1] == "loss" else after_win).append(1 if outc[i] == "win" else 0)
print(f"    WR after a LOSS : {np.mean(after_loss)*100:5.1f}%  (n={len(after_loss)})")
print(f"    WR after a WIN  : {np.mean(after_win)*100:5.1f}%  (n={len(after_win)})")

# ── Winners vs losers: mean feature comparison ───────────────────────────────
print("\n── Winners vs Losers — average feature values ──")
feats = ["fvg_width_p", "risk_p", "dis_range_p", "dis_mult",
         "entry_vs_sweep_p", "sweep_to_entry_min", "hour"]
w = df[df["is_win"] == 1]; ls = df[df["is_win"] == 0]
print(f"    {'feature':22} {'winners':>10} {'losers':>10} {'diff':>8}")
for f in feats:
    print(f"    {f:22} {w[f].mean():10.2f} {ls[f].mean():10.2f} {w[f].mean()-ls[f].mean():+8.2f}")

# ── 'What if we filtered' — impact of dropping the worst buckets ──────────────
print("\n── Filter impact (drop the loss-heavy conditions) ──")
def filt_impact(name, mask_keep):
    sub = df[mask_keep]
    if not len(sub):
        print(f"    {name:34}: (empty)"); return
    wr = sub["is_win"].mean()*100
    kept = len(sub); removed = N - kept
    tot_r = sub["is_win"].sum()*2 - (~sub["is_win"].astype(bool)).sum()
    print(f"    {name:34}: keep {kept:4d} ({kept/N*100:4.0f}%)  WR {wr:5.1f}%  totalR {tot_r:+.0f}")

filt_impact("ALL (baseline)", df.index >= 0)
filt_impact("drop Asia session (00-08)", df["session"] != "Asia 00-08")
filt_impact("keep FVG width >= 2 pips", df["fvg_width_p"] >= 2)
filt_impact("keep FVG width >= 3 pips", df["fvg_width_p"] >= 3)
filt_impact("keep risk >= 3 pips", df["risk_p"] >= 3)
filt_impact("keep dis_mult <= 4", df["dis_mult"] <= 4)
filt_impact("London+NY only (08-22)", df["hour"].between(8, 21))
filt_impact("drop Fri", df["dow"] != "Friday")
filt_impact("combo: FVG>=2 & London/NY & dis<=4",
            (df["fvg_width_p"] >= 2) & (df["hour"].between(8,21)) & (df["dis_mult"] <= 4))
