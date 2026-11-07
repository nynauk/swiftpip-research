"""
Master comparison harness.
Runs every strategy with sensible parameter sweeps, candle-by-candle
simulation, on the 7 forex majors. Single leaderboard at the end.
"""
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import main as bt
import news_filter as nf
import plot_utils as pu

import silver_bullet as sb
import cisd_strategy as cisd
import ema_dol_strategy as ema_dol


MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]


def streak_dd(df, initial=10_000, risk_pct=1.0):
    bal = initial
    eq = [bal]
    cl = 0
    best_cl = 0
    for _, row in df.sort_values("entry_time").iterrows():
        bal += bal * (risk_pct / 100) * float(row["pnl_r"])
        eq.append(bal)
        if row["outcome"] == "loss":
            cl += 1
            best_cl = max(best_cl, cl)
        elif row["outcome"] == "win":
            cl = 0
    eq_arr = np.array(eq)
    dd = ((eq_arr - np.maximum.accumulate(eq_arr)) / np.maximum.accumulate(eq_arr) * 100).min()
    return bal, dd, best_cl


def run_combined(strategy_fn, kwargs, name):
    """Run on all majors, combine results."""
    dfs = []
    for s in MAJORS:
        try:
            r = strategy_fn(s, **kwargs)
            if r["n"]:
                dfs.append(r["df"])
        except Exception as e:
            print(f"  [!] {s} {name} failed: {e}")
    if not dfs:
        return None
    all_df = pd.concat(dfs, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    closed = all_df[all_df["outcome"] != "open"]
    if not len(closed):
        return None
    wr = len(closed[closed["outcome"] == "win"]) / len(closed) * 100
    total_r = closed["pnl_r"].sum()
    gross_w = closed.loc[closed["outcome"] == "win", "pnl_r"].sum()
    gross_l = -closed.loc[closed["outcome"] == "loss", "pnl_r"].sum()
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    bal, dd, streak = streak_dd(all_df)
    expectancy = closed["pnl_r"].mean()
    return {
        "name": name, "n": len(all_df), "wr": wr, "r": total_r,
        "pf": pf, "dd": dd, "streak": streak, "exp": expectancy,
        "bal": bal, "df": all_df,
    }


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    runs = []

    # ── 1. Silver Bullet (proper ICT) ──
    print("\n[1/4] Silver Bullet (proper rules)...")
    for label, kw in [
        ("SB · M15 · NYAM · RR=2",         dict(windows=["NYAM"], rr=2.0, tf="M15")),
        ("SB · M15 · NYAM · RR=3",         dict(windows=["NYAM"], rr=3.0, tf="M15")),
        ("SB · M15 · All-3 · RR=2",        dict(windows=["London", "NYAM", "NYPM"], rr=2.0, tf="M15")),
        ("SB · M5  · NYAM · RR=2",         dict(windows=["NYAM"], rr=2.0, tf="M5")),
        ("SB · M5  · All-3 · RR=2",        dict(windows=["London", "NYAM", "NYPM"], rr=2.0, tf="M5")),
        ("SB · M15 · NYAM · RR=2 · no-news", dict(windows=["NYAM"], rr=2.0, tf="M15", news_block=False)),
    ]:
        print(f"  → {label}")
        r = run_combined(sb.run_silver_bullet, kw, label)
        if r:
            runs.append(r)

    # ── 2. CISD ──
    print("\n[2/4] CISD strategy (sweep + 5M IFVG + CISD)...")
    for label, kw in [
        ("CISD · SL=sweep · RR=2",  dict(sl_mode="SWEEP_EXTREME", rr=2.0)),
        ("CISD · SL=ifvg  · RR=2",  dict(sl_mode="IFVG_EDGE", rr=2.0)),
        ("CISD · SL=sweep · RR=3",  dict(sl_mode="SWEEP_EXTREME", rr=3.0)),
    ]:
        print(f"  → {label}")
        r = run_combined(cisd.run_cisd, kw, label)
        if r:
            runs.append(r)

    # ── 3. EMA + DOL ──
    print("\n[3/4] EMA crossover + ICT DOL...")
    for label, kw in [
        ("EMA9/21 · DOL · 2R min",         dict(fast=9, slow=21, tp_mode="DOL", min_rr=2.0)),
        ("EMA9/21 · DOL · 3R min",         dict(fast=9, slow=21, tp_mode="DOL", min_rr=3.0)),
        ("EMA9/21 · Fixed 2R",             dict(fast=9, slow=21, tp_mode="FIXED", min_rr=2.0)),
        ("EMA12/26 · DOL · 2R min",        dict(fast=12, slow=26, tp_mode="DOL", min_rr=2.0)),
        ("EMA20/50 · DOL · 2R min",        dict(fast=20, slow=50, tp_mode="DOL", min_rr=2.0)),
        ("EMA50/200 · DOL · 2R min",       dict(fast=50, slow=200, tp_mode="DOL", min_rr=2.0)),
        ("EMA9/21 · DOL · no-news",        dict(fast=9, slow=21, tp_mode="DOL", min_rr=2.0, news_block=False)),
    ]:
        print(f"  → {label}")
        r = run_combined(ema_dol.run_ema_dol, kw, label)
        if r:
            runs.append(r)

    # ── 4. Sweep + IFVG (the original family from main.py) ──
    print("\n[4/4] Sweep + IFVG (legacy family)...")
    NO_RETRACE = 1e9

    def run_sweep_ifvg(symbol, *, sweep_mode, rr, sess_start, sess_end,
                       sl_mode, close_loc=0.3, news_block=True, min_w_mult=1.0):
        df_h4  = bt.fetch_data(symbol, mt5.TIMEFRAME_H4,  bt.START_DATE, bt.END_DATE, tf_name="H4")
        df_15m = bt.fetch_data(symbol, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
        df_4h_sw = bt.find_swings(df_h4, bt.SWING_LOOKBACK)
        if sweep_mode == "PDH_PDL":
            sweeps = bt.find_pdh_pdl_sweeps(df_15m)
        elif sweep_mode == "ASIAN_RANGE":
            sweeps = bt.find_session_range_sweeps(df_15m, 0, 7, 23, "Asian")
        elif sweep_mode == "LONDON_RANGE":
            sweeps = bt.find_session_range_sweeps(df_15m, 8, 12, 23, "London")
        else:
            sweeps = bt.find_sweeps(df_4h_sw)
        fvgs_15m = bt.find_all_fvgs(df_15m)
        fvgs_4h  = bt.find_all_fvgs(df_4h_sw)

        per_symbol_min_w = {
            "EURUSD": 0.0002, "GBPUSD": 0.0002, "USDCAD": 0.0002, "USDCHF": 0.0002,
            "USDJPY": 0.02,   "AUDUSD": 0.0002, "NZDUSD": 0.0002,
        }
        bt.DISRESPECT_MULT    = NO_RETRACE
        bt.MIN_FVG_WIDTH      = per_symbol_min_w.get(symbol, 0.0002) * min_w_mult
        bt.REQUIRE_4H_TOUCH   = False
        bt.MAX_CLOSE_LOC      = close_loc
        bt.RR                 = rr
        bt.SESSION_START_HOUR = sess_start
        bt.SESSION_END_HOUR   = sess_end
        bt.HTF_BIAS_MODE      = "OFF"
        bt.SL_MODE            = sl_mode
        bt.NEWS_BLOCK_ENABLED = news_block
        if news_block and not bt.NEWS_BLOCK_SET:
            bt.NEWS_BLOCK_SET = nf.build_news_block_set(df_15m.index[0].date(), df_15m.index[-1].date())

        trades, _ = bt.run_backtest(df_4h_sw, df_15m, sweeps, fvgs_15m, fvgs_4h)
        if not trades:
            return {"symbol": symbol, "n": 0, "wr": 0, "total_r": 0, "df": pd.DataFrame()}
        df = pd.DataFrame(trades)
        df["symbol"] = symbol
        closed = df[df["outcome"] != "open"]
        wr = len(closed[closed["outcome"] == "win"]) / len(closed) * 100 if len(closed) else 0
        return {"symbol": symbol, "n": len(df), "wr": wr,
                "total_r": closed["pnl_r"].sum(), "df": df}

    for label, kw in [
        ("H4-swing · 12-18 · IFVG · RR=2",  dict(sweep_mode="H4_SWING",     rr=2.0, sess_start=12, sess_end=18, sl_mode="IFVG_EDGE")),
        ("H4-swing · 24h · IFVG · RR=2",    dict(sweep_mode="H4_SWING",     rr=2.0, sess_start=0,  sess_end=24, sl_mode="IFVG_EDGE")),
        ("H4-swing · 12-16 · sweep_ext · RR=1", dict(sweep_mode="H4_SWING", rr=1.0, sess_start=12, sess_end=16, sl_mode="SWEEP_EXTREME")),
        ("PDH/PDL · 24h · IFVG · RR=2",     dict(sweep_mode="PDH_PDL",      rr=2.0, sess_start=0,  sess_end=24, sl_mode="IFVG_EDGE")),
        ("Asian rng · 7-23 · IFVG · RR=2",  dict(sweep_mode="ASIAN_RANGE",  rr=2.0, sess_start=7,  sess_end=23, sl_mode="IFVG_EDGE")),
        ("London rng · 13-23 · IFVG · RR=2", dict(sweep_mode="LONDON_RANGE", rr=2.0, sess_start=13, sess_end=23, sl_mode="IFVG_EDGE")),
    ]:
        print(f"  → {label}")
        r = run_combined(run_sweep_ifvg, kw, label)
        if r:
            runs.append(r)

    mt5.shutdown()

    # ── Sort + print leaderboard ──
    print("\n" + "=" * 120)
    print(f"MASTER LEADERBOARD — sorted by TOTAL R")
    print("=" * 120)
    print(f"{'Strategy':45}  {'N':>5}  {'WR':>6}  {'Exp':>7}  {'R':>9}  {'PF':>5}  {'DD':>7}  {'Streak':>7}  {'$bal':>10}")
    print("-" * 120)
    for r in sorted(runs, key=lambda x: x["r"], reverse=True):
        print(f"{r['name']:45}  {r['n']:5d}  {r['wr']:5.1f}%  "
              f"{r['exp']:+6.2f}R  {r['r']:+8.1f}R  {r['pf']:5.2f}  "
              f"{r['dd']:6.1f}%  {r['streak']:>7}  ${r['bal']:>9,.0f}")

    print("\n" + "=" * 120)
    print(f"SORTED BY WIN RATE")
    print("=" * 120)
    for r in sorted(runs, key=lambda x: x["wr"], reverse=True)[:8]:
        print(f"{r['name']:45}  {r['n']:5d}  {r['wr']:5.1f}%  R={r['r']:+7.1f}  PF={r['pf']:.2f}")

    print("\n" + "=" * 120)
    print(f"SORTED BY PROFIT FACTOR")
    print("=" * 120)
    for r in sorted(runs, key=lambda x: x["pf"], reverse=True)[:8]:
        print(f"{r['name']:45}  PF={r['pf']:.2f}  R={r['r']:+7.1f}  WR={r['wr']:.1f}%  N={r['n']}")

    print("\n" + "=" * 120)
    print(f"BEST RISK-ADJUSTED (Total R / |Max DD|)")
    print("=" * 120)
    for r in sorted(runs, key=lambda x: abs(x["r"] / x["dd"]) if x["dd"] != 0 else 0, reverse=True)[:8]:
        ratio = abs(r["r"] / r["dd"]) if r["dd"] != 0 else 0
        print(f"{r['name']:45}  R/|DD|={ratio:6.2f}  R={r['r']:+7.1f}  DD={r['dd']:.1f}%  Streak={r['streak']}")

    # Save the top-3 to CSVs
    top3 = sorted(runs, key=lambda x: x["r"], reverse=True)[:3]
    os.makedirs(os.path.join("strategies", "master"), exist_ok=True)
    for i, r in enumerate(top3, 1):
        safe = r["name"].replace("/", "-").replace(" ", "_").replace("·", "_")
        path = os.path.join("strategies", "master", f"top{i}_{safe}.csv")
        r["df"].to_csv(path, index=False)
        print(f"\nSaved top-{i}: {path}")


if __name__ == "__main__":
    main()
