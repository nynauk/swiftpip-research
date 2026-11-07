"""
Iteration runner. Loads cached data once, then runs the backtest under
multiple config combinations and prints a comparison table.

Usage:
    python iterate.py            # run all configs
    python iterate.py --once     # run a single config (override below)
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime

import main as bt


def run_config(df_4h, df_15m, **overrides):
    """Apply overrides to bt module globals, run, return summary dict."""
    saved = {}
    for k, v in overrides.items():
        if not hasattr(bt, k):
            raise AttributeError(f"unknown config key: {k}")
        saved[k] = getattr(bt, k)
        setattr(bt, k, v)

    try:
        df_4h_sw = bt.find_swings(df_4h, bt.SWING_LOOKBACK)
        if bt.SWEEP_MODE == "PDH_PDL":
            sweeps = bt.find_pdh_pdl_sweeps(df_15m)
        else:
            sweeps = bt.find_sweeps(df_4h_sw)
        fvgs_15m = bt.find_all_fvgs(df_15m)
        fvgs_4h  = bt.find_all_fvgs(df_4h_sw)
        daily_bias = bt.build_daily_bias(df_15m) if bt.HTF_BIAS_MODE == "DAILY" else {}
        trades, funnel = bt.run_backtest(df_4h_sw, df_15m, sweeps, fvgs_15m, fvgs_4h, daily_bias)

        # Time-window cut: last 4 years from the end of data
        end = df_15m.index[-1]
        start_4y = end - pd.Timedelta(days=365 * 4)

        if not trades:
            return {
                "overrides": overrides, "n": 0, "wr": 0.0, "total_r": 0.0,
                "max_dd": 0.0, "best_loss_streak": 0, "trades": [],
                "funnel": funnel, "n_4y": 0, "wr_4y": 0.0, "r_4y": 0.0,
            }

        df = pd.DataFrame(trades)
        closed = df[df["outcome"] != "open"]
        wr = len(closed[closed["outcome"] == "win"]) / len(closed) * 100 if len(closed) else 0
        stats = bt.compute_stats(trades)

        df4y = df[df["entry_time"] >= start_4y]
        c4y = df4y[df4y["outcome"] != "open"]
        wr4y = len(c4y[c4y["outcome"] == "win"]) / len(c4y) * 100 if len(c4y) else 0
        r4y = c4y["pnl_r"].sum()

        return {
            "overrides": overrides,
            "n": len(df), "n_closed": len(closed),
            "wr": wr, "total_r": closed["pnl_r"].sum(),
            "max_dd": stats["max_dd"],
            "best_loss_streak": stats["best_loss_streak"],
            "best_win_streak":  stats["best_win_streak"],
            "n_4y": len(df4y), "wr_4y": wr4y, "r_4y": r4y,
            "funnel": funnel,
            "df": df,
        }
    finally:
        for k, v in saved.items():
            setattr(bt, k, v)


def load_data():
    """Load cached H4 and M15. If no cache, prompt user to run main.py once."""
    import os
    h4 = os.path.join(bt.CACHE_DIR, f"{bt.SYMBOL}_H4.pkl")
    m15 = os.path.join(bt.CACHE_DIR, f"{bt.SYMBOL}_M15.pkl")
    if not (os.path.exists(h4) and os.path.exists(m15)):
        print(f"[!] No cache at {bt.CACHE_DIR}. Run `python main.py` once with MT5 open to build it.")
        sys.exit(1)
    return pd.read_pickle(h4), pd.read_pickle(m15)


def print_summary(label, r, by_hour=False, by_year=False):
    import pandas as pd
    print(f"\n── {label} ──")
    print(f"  overrides: {r['overrides']}")
    print(f"  total trades : {r['n']}  closed: {r.get('n_closed', 0)}")
    print(f"  win rate     : {r['wr']:.1f}%   total R: {r['total_r']:.1f}")
    print(f"  max DD       : {r['max_dd']:.1f}%   longest loss streak: {r['best_loss_streak']}")
    if "df" in r and len(r["df"]):
        df = r["df"]
        for side in ("long", "short"):
            sub = df[df["direction"] == side]
            closed = sub[sub["outcome"] != "open"]
            if len(closed):
                wr = len(closed[closed["outcome"] == "win"]) / len(closed) * 100
                print(f"    {side:5}: {len(sub):3d} trades, WR {wr:.1f}%, R {closed['pnl_r'].sum():+.1f}")
        if by_hour:
            df = df.copy()
            df["hour"] = pd.to_datetime(df["entry_time"]).dt.hour
            for hr in sorted(df["hour"].unique()):
                sub = df[df["hour"] == hr]
                closed = sub[sub["outcome"] != "open"]
                if len(closed):
                    wr = len(closed[closed["outcome"] == "win"]) / len(closed) * 100
                    print(f"    hr {hr:02d}: {len(sub):3d} trades, WR {wr:.1f}%, R {closed['pnl_r'].sum():+.1f}")
        if by_year:
            df = df.copy()
            df["year"] = pd.to_datetime(df["entry_time"]).dt.year
            for yr in sorted(df["year"].unique()):
                sub = df[df["year"] == yr]
                closed = sub[sub["outcome"] != "open"]
                if len(closed):
                    wr = len(closed[closed["outcome"] == "win"]) / len(closed) * 100
                    print(f"    {yr}: {len(sub):3d} trades, WR {wr:.1f}%, R {closed['pnl_r'].sum():+.1f}")


def main():
    df_4h, df_15m = load_data()
    print(f"Loaded H4: {len(df_4h):,} bars  ({df_4h.index[0]} -> {df_4h.index[-1]})")
    print(f"Loaded M15:{len(df_15m):,} bars  ({df_15m.index[0]} -> {df_15m.index[-1]})")

    # Iteration M: best realistic targets given the data.
    # Compare: PDH/PDL multi-RR · with vs without bias · vs the 4H-swing winner.
    NO_RETRACE = 1e9
    P = {"DISRESPECT_MULT": NO_RETRACE, "MIN_FVG_WIDTH": 0.0002,
         "REQUIRE_4H_TOUCH": False, "MAX_CLOSE_LOC": 0.3,
         "SESSION_START_HOUR": 8, "SESSION_END_HOUR": 16,
         "HTF_BIAS_MODE": "OFF"}
    configs = [
        ("M0 PDH/PDL RR=2.0",                     {**P, "RR": 2.0}),
        ("M1 PDH/PDL RR=1.5",                     {**P, "RR": 1.5}),
        ("M2 PDH/PDL RR=1.0",                     {**P, "RR": 1.0}),
        ("M3 PDH/PDL RR=1.5 + close=0.5",         {**P, "RR": 1.5, "MAX_CLOSE_LOC": 0.5}),
        ("M4 PDH/PDL RR=2.0 + bias + close=0.5",  {**P, "RR": 2.0, "HTF_BIAS_MODE": "DAILY", "MAX_CLOSE_LOC": 0.5}),
        ("M5 PDH/PDL RR=1.5 + bias + close=0.5",  {**P, "RR": 1.5, "HTF_BIAS_MODE": "DAILY", "MAX_CLOSE_LOC": 0.5}),
        ("M6 PDH/PDL RR=1.5 + 12-17",             {**P, "RR": 1.5, "SESSION_START_HOUR": 12, "SESSION_END_HOUR": 17}),
        ("M7 PDH/PDL RR=1.5 + buf=1.5",           {**P, "RR": 1.5, "SL_BUFFER_FRAC": 1.5}),
        ("M8 4H-swing J2 (our previous winner)",  {"SWEEP_MODE": "H4_SWING",
                                                    "DISRESPECT_MULT": NO_RETRACE,
                                                    "MIN_FVG_WIDTH": 0.0002,
                                                    "REQUIRE_4H_TOUCH": False,
                                                    "MAX_CLOSE_LOC": 0.3,
                                                    "RR": 1.0,
                                                    "SESSION_START_HOUR": 12,
                                                    "SESSION_END_HOUR": 16}),
        ("M9 4H-swing J2 + RR=2.0",               {"SWEEP_MODE": "H4_SWING",
                                                    "DISRESPECT_MULT": NO_RETRACE,
                                                    "MIN_FVG_WIDTH": 0.0002,
                                                    "REQUIRE_4H_TOUCH": False,
                                                    "MAX_CLOSE_LOC": 0.3,
                                                    "RR": 2.0,
                                                    "SESSION_START_HOUR": 12,
                                                    "SESSION_END_HOUR": 16}),
    ]

    results = []
    for label, cfg in configs:
        r = run_config(df_4h, df_15m, **cfg)
        results.append((label, r))
        # Show full breakdown for every config in J — we're checking stability
        print_summary(label, r, by_hour=True, by_year=True)


if __name__ == "__main__":
    main()
