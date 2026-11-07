"""
ICT "FOREVER MODEL" — grading-system backtest.

The Reddit setup grades trades A++ / A+ / A / B by how many confluences stack.
This script MECHANIZES the confluences, scores every base setup, assigns a
grade, and reports win-rate / profit-factor / R BY GRADE — to answer the real
question: does a higher grade actually mean a higher-quality trade?

Base setup (must be present to be a trade at all):
   • HTF (4H) liquidity sweep  — a 4H candle wicks past a 4H swing high/low and
     closes back inside (turtle soup / took HTF liquidity).
   • LTF (15M) reversal FVG that inverts (IFVG) in the bias direction.
   • LTF CISD — a 15M candle closes through the recent opposing swing.

Scored confluences (each adds a point):
   F2  HTF PD ARRAY   : the 4H sweep extreme sits inside a 4H FVG (HTF imbalance).
   F3  SMT (HTF)      : correlated pair (GBPUSD) did NOT confirm the sweep —
                        EURUSD made a new extreme, GBPUSD did not (divergence).
   F4  CISD (HTF)     : after the sweep a 4H candle closes through the recent
                        opposing 4H swing (HTF shift confirmed).
   F6  CLEAR DOL      : a clear opposing draw-on-liquidity (opposing prior level
                        or swing) sits >= 2R away → room to target.
   F7  RETRACEMENT    : entry taken on a retrace back into the LTF PD array after
                        the displacement (not chased at the CISD close).
   F8  SMT (LTF)      : correlated pair divergence on the LTF retracement.

Grade (by score of the 6 optional confluences):
   A++ : score >= 5      A+ : 4      A : 3      B : <= 2

Entry  = retrace into the IFVG (or CISD close if no retrace in window).
Stop   = beyond the swept extreme + buffer.
Target = 2R (also an OPP_LIQ variant tested).
Traded pair EURUSD, SMT correlate GBPUSD, HTF=H4, LTF=M15. News-blocked.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MetaTrader5 as mt5

import main as bt
import news_filter as nf
import plot_utils as pu


TRADED   = "EURUSD"
CORR     = "GBPUSD"      # positively-correlated pair for SMT
PIP      = 0.0001
MIN_FVG  = 2 * PIP
POST_SWEEP_WINDOW = 48   # M15 bars after sweep to find reversal FVG
IFVG_DELAY        = 12
CISD_WINDOW       = 12
CISD_LOOKBACK     = 5
RETRACE_WINDOW    = 24    # M15 bars to wait for retrace into the IFVG
SL_BUFFER         = 0.25
SMT_WINDOW        = 20    # bars around sweep to compare correlated pair
DOL_TOL_R         = 2.0   # a clear DOL must be >= this many R away


def _swings(h, l, n=5):
    sh = np.full(len(h), np.nan); slo = np.full(len(l), np.nan)
    for i in range(n, len(h) - n):
        if h[i] >= h[i - n:i + n + 1].max(): sh[i] = h[i]
        if l[i] <= l[i - n:i + n + 1].min(): slo[i] = l[i]
    return sh, slo


def simulate(h, l, idx, entry_i, direction, entry, sl, tp, rr_value, max_bars=600):
    last_j = entry_i
    for j in range(entry_i + 1, min(entry_i + max_bars, len(h))):
        last_j = j
        if direction == "long":
            if l[j] <= sl:  return {"outcome": "loss", "exit": sl, "exit_time": idx[j], "pnl_r": -1.0}
            if h[j] >= tp:  return {"outcome": "win",  "exit": tp, "exit_time": idx[j], "pnl_r":  rr_value}
        else:
            if h[j] >= sl:  return {"outcome": "loss", "exit": sl, "exit_time": idx[j], "pnl_r": -1.0}
            if l[j] <= tp:  return {"outcome": "win",  "exit": tp, "exit_time": idx[j], "pnl_r":  rr_value}
    # Unresolved within window → exit at the LAST bar checked (NOT end of dataset,
    # which would poison the 1-position concurrency cap).
    return {"outcome": "open", "exit": float("nan"), "exit_time": idx[last_j], "pnl_r": 0.0}


def build_prior_levels(df):
    d = df.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    w = df.resample("W").agg({"high": "max", "low": "min"}).dropna()
    m = df.resample("ME").agg({"high": "max", "low": "min"}).dropna()
    pdh, pdl = d["high"].shift(1).to_dict(), d["low"].shift(1).to_dict()
    pwh = {k.to_period("W"): v for k, v in w["high"].shift(1).items()}
    pwl = {k.to_period("W"): v for k, v in w["low"].shift(1).items()}
    pmh = {k.to_period("M"): v for k, v in m["high"].shift(1).items()}
    pml = {k.to_period("M"): v for k, v in m["low"].shift(1).items()}

    def lookup(ts):
        ts = pd.Timestamp(ts); day = ts.normalize()
        return {"PDH": pdh.get(day), "PDL": pdl.get(day),
                "PWH": pwh.get(ts.to_period("W")), "PWL": pwl.get(ts.to_period("W")),
                "PMH": pmh.get(ts.to_period("M")), "PML": pml.get(ts.to_period("M"))}
    return lookup


def run(tp_mode="FIXED", rr=2.0, news_block=True):
    df_h4  = bt.fetch_data(TRADED, mt5.TIMEFRAME_H4,  bt.START_DATE, bt.END_DATE, tf_name="H4")
    df_15  = bt.fetch_data(TRADED, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
    corr15 = bt.fetch_data(CORR,   mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")

    df_h4_sw = bt.find_swings(df_h4, bt.SWING_LOOKBACK)
    sweeps   = bt.find_sweeps(df_h4_sw)
    fvgs_4h  = bt.find_all_fvgs(df_h4_sw)
    fvgs_15  = bt.find_all_fvgs(df_15)

    h = df_15["high"].values; l = df_15["low"].values; c = df_15["close"].values
    idx = df_15.index
    ch = corr15["high"].reindex(idx, method="ffill").values
    cl = corr15["low"].reindex(idx, method="ffill").values

    # HTF arrays for CISD_HTF
    h4h = df_h4["high"].values; h4l = df_h4["low"].values; h4c = df_h4["close"].values
    h4idx = df_h4.index

    level_lookup = build_prior_levels(df_15)
    sh15, sl15 = _swings(h, l, n=5)
    news_set = nf.build_news_block_set(idx[0].date(), idx[-1].date()) if news_block else set()

    fvgs_sorted = sorted(fvgs_15, key=lambda f: f["form_idx"])
    form_idx_arr = np.array([f["form_idx"] for f in fvgs_sorted])

    trades = []
    used = set()
    open_until = []

    for sweep in sweeps:
        bias = sweep["direction"]   # bullish→long
        sig = (sweep["time"], bias)
        if sig in used:
            continue
        swept = sweep["swept_level"]
        si = int(np.searchsorted(idx, sweep["time"], side="left"))
        if si >= len(idx) or (si == 0 and sweep["time"] < idx[0]):
            continue

        # ── F3 SMT (HTF): did the correlated pair confirm the sweep? ──
        w0 = max(0, si - SMT_WINDOW)
        smt_htf = False
        if bias == "bullish":
            eur_new_low = l[si] <= l[w0:si + 1].min() + 1e-12
            gbp_new_low = cl[si] <= cl[w0:si + 1].min() + 1e-12
            smt_htf = eur_new_low and not gbp_new_low
        else:
            eur_new_high = h[si] >= h[w0:si + 1].max() - 1e-12
            gbp_new_high = ch[si] >= ch[w0:si + 1].max() - 1e-12
            smt_htf = eur_new_high and not gbp_new_high

        # ── F2 HTF PD array: sweep extreme inside a 4H FVG (bias direction) ──
        s4 = int(np.searchsorted(df_h4.index, sweep["time"], side="left"))
        sweep_extreme_4h = h4h[s4] if bias == "bearish" else h4l[s4]
        htf_pd = False
        for f4 in fvgs_4h:
            if h4idx[f4["form_idx"]] > sweep["time"]:
                continue
            if f4["bottom"] <= sweep_extreme_4h <= f4["top"]:
                if (bias == "bullish" and f4["direction"] == "bullish") or \
                   (bias == "bearish" and f4["direction"] == "bearish"):
                    htf_pd = True; break

        # ── F4 CISD (HTF): 4H close through recent opposing 4H swing after sweep ──
        cisd_htf = False
        for k in range(s4 + 1, min(s4 + 12, len(df_h4))):
            if bias == "bullish":
                if h4c[k] > h4h[max(0, k - 5):k].max():
                    cisd_htf = True; break
            else:
                if h4c[k] < h4l[max(0, k - 5):k].min():
                    cisd_htf = True; break

        # ── Reversal FVG (opposite bias) after sweep ──
        target_dir = "bearish" if bias == "bullish" else "bullish"
        fvg_end = min(si + POST_SWEEP_WINDOW, len(idx) - 1)
        lo_p = int(np.searchsorted(form_idx_arr, si, side="left"))
        hi_p = int(np.searchsorted(form_idx_arr, fvg_end, side="right"))

        for p in range(lo_p, hi_p):
            fvg = fvgs_sorted[p]
            if fvg["direction"] != target_dir or fvg["width"] < MIN_FVG:
                continue
            f_i = fvg["form_idx"]

            # IFVG inversion
            inv_end = min(f_i + IFVG_DELAY, len(idx) - 1)
            ifvg_i = None
            for di in range(f_i + 1, inv_end + 1):
                if fvg["direction"] == "bearish":
                    if c[di] > fvg["top"]: ifvg_i = di; break
                else:
                    if c[di] < fvg["bottom"]: ifvg_i = di; break
            if ifvg_i is None:
                continue

            # LTF CISD (base requirement, F5)
            cisd_end = min(ifvg_i + CISD_WINDOW, len(idx) - 1)
            cisd_i = None
            for k in range(ifvg_i + 1, cisd_end + 1):
                if k - CISD_LOOKBACK < 0: continue
                if bias == "bullish":
                    if c[k] > h[k - CISD_LOOKBACK:k].max(): cisd_i = k; break
                else:
                    if c[k] < l[k - CISD_LOOKBACK:k].min(): cisd_i = k; break
            if cisd_i is None:
                continue

            # ── F7 retracement entry: wait for pullback into the IFVG ──
            retr_end = min(cisd_i + RETRACE_WINDOW, len(idx) - 1)
            entry_i = None
            retr = False
            for k in range(cisd_i + 1, retr_end + 1):
                if bias == "bullish":
                    if l[k] <= fvg["top"]:   # pullback down into IFVG support
                        entry_i = k; retr = True; break
                else:
                    if h[k] >= fvg["bottom"]:
                        entry_i = k; retr = True; break
            if entry_i is None:
                entry_i = cisd_i     # no retrace → enter at CISD close (F7 False)

            if news_block and idx[entry_i].date() in news_set:
                continue
            open_until = [t for t in open_until if t > idx[entry_i]]
            if len(open_until) >= 1:
                continue

            # ── F8 SMT (LTF) on the retracement ──
            smt_ltf = False
            if retr:
                w = max(0, entry_i - SMT_WINDOW)
                if bias == "bullish":
                    smt_ltf = (l[entry_i] <= l[w:entry_i + 1].min() + 1e-12) and \
                              (cl[entry_i] > cl[w:entry_i + 1].min() + 1e-12)
                else:
                    smt_ltf = (h[entry_i] >= h[w:entry_i + 1].max() - 1e-12) and \
                              (ch[entry_i] < ch[w:entry_i + 1].max() - 1e-12)

            entry = (fvg["top"] if bias == "bullish" else fvg["bottom"]) if retr else c[cisd_i]
            deep_extreme = l[si:entry_i + 1].min() if bias == "bullish" else h[si:entry_i + 1].max()
            buf = fvg["width"] * SL_BUFFER
            if bias == "bullish":
                direction = "long"; sl = deep_extreme - buf; risk = entry - sl
            else:
                direction = "short"; sl = deep_extreme + buf; risk = sl - entry
            if risk <= 0:
                continue

            # ── F6 CLEAR DOL: opposing prior level / swing >= 2R away ──
            elv = level_lookup(idx[entry_i])
            dol_clear = False
            opp_target = None
            if direction == "long":
                opp = [elv[k] for k in ("PDH", "PWH", "PMH") if elv[k] and elv[k] > entry]
                opp += [sh15[k] for k in range(max(0, entry_i - 400), entry_i)
                        if not np.isnan(sh15[k]) and sh15[k] > entry]
                opp = [x for x in opp if (x - entry) >= DOL_TOL_R * risk]
                if opp:
                    dol_clear = True; opp_target = min(opp)
            else:
                opp = [elv[k] for k in ("PDL", "PWL", "PML") if elv[k] and elv[k] < entry]
                opp += [sl15[k] for k in range(max(0, entry_i - 400), entry_i)
                        if not np.isnan(sl15[k]) and sl15[k] < entry]
                opp = [x for x in opp if (entry - x) >= DOL_TOL_R * risk]
                if opp:
                    dol_clear = True; opp_target = max(opp)

            # ── TP ──
            if tp_mode == "OPP_LIQ" and opp_target is not None:
                tp = opp_target
                rr_val = (tp - entry) / risk if direction == "long" else (entry - tp) / risk
                if rr_val > 15:
                    tp = entry + rr * risk if direction == "long" else entry - rr * risk
                    rr_val = rr
            else:
                rr_val = rr
                tp = entry + rr * risk if direction == "long" else entry - rr * risk

            # ── Score & grade ──
            score = int(htf_pd) + int(smt_htf) + int(cisd_htf) + int(dol_clear) + int(retr) + int(smt_ltf)
            if score >= 5:   grade = "A++"
            elif score == 4: grade = "A+"
            elif score == 3: grade = "A"
            else:            grade = "B"

            result = simulate(h, l, idx, entry_i, direction, entry, sl, tp, rr_val)
            open_until.append(pd.Timestamp(result["exit_time"]))
            trades.append({
                "symbol": TRADED, "sweep_time": sweep["time"], "bias": bias,
                "entry_time": idx[entry_i], "entry_idx": entry_i, "direction": direction,
                "entry": round(entry, 6), "sl": round(sl, 6), "tp": round(tp, 6),
                "rr_target": round(rr_val, 2),
                "fvg_top": round(fvg["top"], 6), "fvg_bottom": round(fvg["bottom"], 6),
                "F2_HTF_PD": htf_pd, "F3_SMT_HTF": smt_htf, "F4_CISD_HTF": cisd_htf,
                "F6_DOL_clear": dol_clear, "F7_retrace": retr, "F8_SMT_LTF": smt_ltf,
                "score": score, "grade": grade,
                "outcome": result["outcome"],
                "exit": round(result["exit"], 6) if not np.isnan(result["exit"]) else np.nan,
                "exit_time": result["exit_time"], "pnl_r": float(result["pnl_r"]),
            })
            used.add(sig)
            break

    return pd.DataFrame(trades)


def grade_report(df, label):
    closed = df[df["outcome"] != "open"].copy()
    print(f"\n{'='*92}\n  {label}   (total {len(closed)} closed trades)\n{'='*92}")
    print(f"  {'Grade':6}  {'N':>4}  {'WR':>6}  {'R':>7}  {'PF':>5}  {'Exp/trd':>8}  {'AvgRR':>6}")
    order = {"A++": 0, "A+": 1, "A": 2, "B": 3}
    for g in sorted(closed["grade"].unique(), key=lambda x: order.get(x, 9)):
        sub = closed[closed["grade"] == g]
        n = len(sub); wr = (sub["outcome"] == "win").sum() / n * 100
        r = sub["pnl_r"].sum()
        gw = sub.loc[sub["outcome"] == "win", "pnl_r"].sum()
        gl = -sub.loc[sub["outcome"] == "loss", "pnl_r"].sum()
        pf = gw / gl if gl > 0 else float("inf")
        print(f"  {g:6}  {n:4d}  {wr:5.1f}%  {r:+6.1f}  {pf:5.2f}  {sub['pnl_r'].mean():+7.3f}R  {sub['rr_target'].mean():5.2f}")

    # By raw score too
    print(f"\n  By score:")
    for sc in sorted(closed["score"].unique(), reverse=True):
        sub = closed[closed["score"] == sc]
        n = len(sub); wr = (sub["outcome"] == "win").sum() / n * 100
        print(f"    score {sc}: N={n:4d}  WR={wr:5.1f}%  R={sub['pnl_r'].sum():+6.1f}  Exp={sub['pnl_r'].mean():+.3f}R")

    # Confluence hit-rate: WR when each flag is present vs absent
    print(f"\n  Single-confluence WR (present vs absent):")
    for flag in ["F2_HTF_PD", "F3_SMT_HTF", "F4_CISD_HTF", "F6_DOL_clear", "F7_retrace", "F8_SMT_LTF"]:
        on = closed[closed[flag]]; off = closed[~closed[flag]]
        wr_on = (on["outcome"] == "win").sum() / len(on) * 100 if len(on) else 0
        wr_off = (off["outcome"] == "win").sum() / len(off) * 100 if len(off) else 0
        print(f"    {flag:14}:  ON {wr_on:5.1f}% (n={len(on):4d})   OFF {wr_off:5.1f}% (n={len(off):4d})   Δ {wr_on-wr_off:+.1f}%")


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    OUT = os.path.join("strategies", "forever_model")
    os.makedirs(OUT, exist_ok=True)

    for tp_mode, tag in [("FIXED", "Fixed 2R"), ("OPP_LIQ", "Opposing-liquidity TP")]:
        df = run(tp_mode=tp_mode, rr=2.0, news_block=True)
        if not len(df):
            print(f"{tag}: no trades")
            continue
        grade_report(df, f"FOREVER MODEL · EURUSD (SMT vs GBPUSD) · H4→M15 · {tag}")
        df.to_csv(os.path.join(OUT, f"trades_{tp_mode}.csv"), index=False)

    mt5.shutdown()
    print(f"\nSaved to {OUT}/")


if __name__ == "__main__":
    main()
