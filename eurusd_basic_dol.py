"""
Compare partial@1R management with different RUNNER targets:
  A. fixed 3R                 (baseline — current dry-basic)
  B. 15M-swing DOL runner     (nearest opposing 15M swing high/low)
  C. prior-level DOL runner   (nearest opposing PDH/PDL/PWH/PWL)
  D. mixed DOL runner         (nearest of B or C)

Same 2,516 entries used for all variants — only the TAKE PROFIT differs.
Partial 50% at +1R + move SL to BE is applied to all.
"""
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import main as bt
import eurusd_basic as eb

mt5.initialize()
df_h4 = bt.fetch_data("EURUSD", mt5.TIMEFRAME_H4,  bt.START_DATE, bt.END_DATE, tf_name="H4")
df_15 = bt.fetch_data("EURUSD", mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
mt5.shutdown()

# Baseline entry set (partial@1R + fixed 3R)
base = eb.run(df_h4, df_15, rr=3.0, partial_at=1.0)
base = base[base["outcome"] != "open"].sort_values("entry_time").reset_index(drop=True)
print(f"Entries: {len(base)}\n")

h = df_15["high"].values
l = df_15["low"].values
c = df_15["close"].values
idx = df_15.index

# ── 15M swing highs / lows (5-bar) ──
def swings(h_, l_, n=5):
    sh = np.full(len(h_), np.nan); sl = np.full(len(l_), np.nan)
    for i in range(n, len(h_) - n):
        if h_[i] >= h_[i - n:i + n + 1].max(): sh[i] = h_[i]
        if l_[i] <= l_[i - n:i + n + 1].min(): sl[i] = l_[i]
    return sh, sl
sh15, sl15 = swings(h, l, n=5)

# ── PDH/PDL/PWH/PWL series ──
d = df_15.resample("1D").agg({"high": "max", "low": "min"}).dropna()
pdh_ser = d["high"].shift(1); pdl_ser = d["low"].shift(1)
w = df_15.resample("W").agg({"high": "max", "low": "min"}).dropna()
pwh_ser = w["high"].shift(1); pwl_ser = w["low"].shift(1)

def get_prev_levels(ts):
    day = pd.Timestamp(ts).normalize()
    pdh = pdh_ser.get(day); pdl = pdl_ser.get(day)
    wp  = pd.Timestamp(ts).to_period("W")
    pwh = pwh_ser.get(wp); pwl = pwl_ser.get(wp)
    return {"PDH": pdh, "PDL": pdl, "PWH": pwh, "PWL": pwl}


def nearest_swing_above(entry_i, entry_price, lookback=400):
    lo = max(0, entry_i - lookback)
    cands = [sh15[i] for i in range(lo, entry_i)
             if not np.isnan(sh15[i]) and sh15[i] > entry_price]
    return min(cands) if cands else None


def nearest_swing_below(entry_i, entry_price, lookback=400):
    lo = max(0, entry_i - lookback)
    cands = [sl15[i] for i in range(lo, entry_i)
             if not np.isnan(sl15[i]) and sl15[i] < entry_price]
    return max(cands) if cands else None


def resim(entry_i, direction, entry, sl, tp, partial_at=1.0, partial_frac=0.5, max_bars=600):
    """Partial 50% at +partial_at·R + BE, remainder runs to tp. Returns realized R."""
    risk = abs(entry - sl)
    if risk <= 0 or tp is None:
        return 0.0
    sl_cur = sl
    armed = False
    realized = 0.0
    remaining = 1.0
    for j in range(entry_i + 1, min(entry_i + max_bars, len(h))):
        if direction == "long":
            if l[j] <= sl_cur:
                return realized + remaining * ((sl_cur - entry) / risk)
            if h[j] >= tp:
                return realized + remaining * ((tp - entry) / risk)
            if not armed and h[j] >= entry + partial_at * risk:
                realized += partial_frac * partial_at
                remaining -= partial_frac
                sl_cur = entry; armed = True
        else:
            if h[j] >= sl_cur:
                return realized + remaining * ((entry - sl_cur) / risk)
            if l[j] <= tp:
                return realized + remaining * ((entry - tp) / risk)
            if not armed and l[j] <= entry - partial_at * risk:
                realized += partial_frac * partial_at
                remaining -= partial_frac
                sl_cur = entry; armed = True
    return realized       # unresolved → return what we've realized


def choose_target(row, mode, min_rr=1.0, max_rr=15.0):
    """Pick TP for this row given mode. Return (tp_price, tp_r_multiple, source)."""
    ei = int(row["entry_idx"])
    entry = row["entry"]; sl = row["sl"]; risk = abs(entry - sl)
    long = row["direction"] == "long"

    if mode == "FIXED_3R":
        return (entry + 3 * risk) if long else (entry - 3 * risk), 3.0, "fixed"

    if mode == "SWING":
        dol = nearest_swing_above(ei, entry) if long else nearest_swing_below(ei, entry)
    elif mode == "PRIOR":
        lv = get_prev_levels(row["entry_time"])
        keys = ("PDH", "PWH") if long else ("PDL", "PWL")
        cands = [lv[k] for k in keys if lv[k] is not None and not np.isnan(lv[k])
                 and (lv[k] > entry if long else lv[k] < entry)]
        dol = (min(cands) if long else max(cands)) if cands else None
    elif mode == "MIXED":
        # nearest of swing OR prior
        swing = nearest_swing_above(ei, entry) if long else nearest_swing_below(ei, entry)
        lv = get_prev_levels(row["entry_time"])
        keys = ("PDH", "PWH") if long else ("PDL", "PWL")
        pri_cands = [lv[k] for k in keys if lv[k] is not None and not np.isnan(lv[k])
                     and (lv[k] > entry if long else lv[k] < entry)]
        pri = (min(pri_cands) if long else max(pri_cands)) if pri_cands else None
        opts = [x for x in (swing, pri) if x is not None]
        dol = (min(opts) if long else max(opts)) if opts else None
    else:
        return None, None, "none"

    if dol is None:
        return None, None, "no_dol"
    rr = ((dol - entry) if long else (entry - dol)) / risk
    if rr < min_rr or rr > max_rr:
        return None, None, "out_of_range"
    return dol, rr, mode


def run_variant(mode, min_rr=1.0):
    rows = []
    for _, row in base.iterrows():
        tp, tp_rr, src = choose_target(row, mode, min_rr=min_rr)
        if tp is None:
            # skip this trade if DOL not available — fair comparison
            continue
        pnl = resim(int(row["entry_idx"]), row["direction"], row["entry"], row["sl"], tp)
        rows.append({"entry_time": row["entry_time"], "pnl_r": pnl, "tp_rr": tp_rr, "src": src})
    df = pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)
    return df


def metrics(df, name):
    if not len(df):
        return None
    n = len(df)
    wins = (df["pnl_r"] > 0.01).sum()
    losses = (df["pnl_r"] < -0.01).sum()
    bes = n - wins - losses
    wr = wins / n * 100
    gw = df.loc[df["pnl_r"] > 0, "pnl_r"].sum()
    gl = -df.loc[df["pnl_r"] < 0, "pnl_r"].sum()
    pf = gw / gl if gl else float("inf")
    tot = df["pnl_r"].sum()
    exp = df["pnl_r"].mean()
    avg_rr = df["tp_rr"].mean()
    bal = 10000.0; eq = [bal]; cl = bcl = cw = bcw = 0
    for _, r in df.iterrows():
        bal += bal * 0.01 * float(r["pnl_r"]); eq.append(bal)
        if r["pnl_r"] < -0.01: cl += 1; cw = 0
        else:
            if r["pnl_r"] > 0.01: cw += 1; bcw = max(bcw, cw)
            cl = 0
        bcl = max(bcl, cl)
    eqa = np.array(eq)
    dd = ((eqa - np.maximum.accumulate(eqa)) / np.maximum.accumulate(eqa) * 100).min()
    return dict(name=name, n=n, w=wins, be=bes, lo=losses, wr=wr, pf=pf,
                tot=tot, exp=exp, avg_rr=avg_rr, dd=dd, ls=bcl, ws=bcw, bal=bal)


variants = [
    ("A. FIXED 3R (baseline)",       run_variant("FIXED_3R")),
    ("B. 15M-swing DOL (≥1R)",       run_variant("SWING",  min_rr=1.0)),
    ("B'. 15M-swing DOL (≥2R)",      run_variant("SWING",  min_rr=2.0)),
    ("C. Prior levels DOL (≥1R)",    run_variant("PRIOR",  min_rr=1.0)),
    ("C'. Prior levels DOL (≥2R)",   run_variant("PRIOR",  min_rr=2.0)),
    ("D. MIXED (nearest of B/C, ≥1R)", run_variant("MIXED",  min_rr=1.0)),
    ("D'. MIXED (≥2R)",              run_variant("MIXED",  min_rr=2.0)),
]

M = [metrics(df, name) for name, df in variants]

print("=" * 108)
print("PARTIAL@1R · DOL RUNNER COMPARISON · same entries")
print("=" * 108)
print(f"  {'variant':32} {'N':>4} {'W/BE/L':>14} {'WR':>6} {'PF':>5} {'exp':>7} "
      f"{'avgRR':>6} {'totR':>7} {'DD':>7} {'LStrk':>5} {'$10k→':>10}")
for m in M:
    if m is None:
        continue
    print(f"  {m['name']:32} {m['n']:>4} {m['w']:>4}/{m['be']:>3}/{m['lo']:>3}  "
          f"{m['wr']:>5.1f}% {m['pf']:>4.2f} {m['exp']:>+6.3f}R {m['avg_rr']:>5.2f} "
          f"{m['tot']:>+6.0f}R {m['dd']:>5.1f}% {m['ls']:>5}  ${m['bal']:>8,.0f}")

# Distribution of DOL runner distances (mixed ≥1R)
mix = variants[5][1]
print("\n── DOL distance distribution (variant D, mixed ≥1R) ──")
b = pd.cut(mix["tp_rr"], [1, 1.5, 2, 3, 5, 8, 100],
           labels=["1-1.5R","1.5-2R","2-3R","3-5R","5-8R","8R+"])
for bkt, cnt in b.value_counts().sort_index().items():
    print(f"    {bkt}: {cnt}")

# Per-year on the best DOL variant
best = max([m for m in M if m and m["name"].startswith(("B","C","D"))],
           key=lambda x: x["tot"])
print(f"\n── Per-year on best DOL variant ({best['name']}) ──")
df_best = variants[[m['name'] for m in M].index(best["name"])][1]
df_best = df_best.copy()
df_best["year"] = pd.to_datetime(df_best["entry_time"]).dt.year
for yr, g in df_best.groupby("year"):
    wr = (g["pnl_r"] > 0.01).mean() * 100
    print(f"    {yr}: N={len(g):>4}  WR={wr:>5.1f}%  R={g['pnl_r'].sum():+.0f}")
