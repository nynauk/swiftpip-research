"""
10AM PO3 model — objective, causal backtest. EURUSD · HistData 1min 2004-2024.

Codified from the PDF:
  • Accumulation = the pre-10AM range (07:00-10:00 EST) — its high/low is the
    liquidity pool / HTF-PDA proxy the 10AM manipulation sweeps.
  • Manipulation + entry window = 10:00-12:00 EST (NY AM delivery).
  • First sweep of the accumulation high/low after 10:00 = manipulation.
        swept the HIGH -> short bias ; swept the LOW -> long bias
  • LTF trigger = 1m CISD (a trigger the PDF explicitly lists), reusing the exact
    detector. SL beyond the swing.
  • Target: opposing liquidity (the opposite side of the accumulation range) —
    plus fixed 2.1R and 3R for comparison.

HistData is EST with no DST, so passing offset_h = 1 makes the detector's session
clock equal the raw EST index (utc1 = index - 1 + 1 = index). We override the
session-hour config locally so the LIVE bot config is untouched.

    python -m asia_bot.backtest_po3
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asia_bot import config as cfg
from asia_bot import detector as det

SYMBOL = "EURUSD"
HD = os.path.join("ext_data", "HD_EURUSD_1min_2004-2024.pkl")
SPREAD_PIPS = 1.0
WALK_BARS, WALK_HOURS = 3000, 8
OFFSET_EST = 1                       # makes detector session-time == EST index

# NY session windows (EST)
ACC_START, ACC_END = 7, 10           # accumulation / liquidity pool
NY_START, NY_END = 10, 12            # manipulation + entry (10AM PO3 window)


def sim(dfc, t1, s, pr, spread, mode):
    h = dfc["high"].values; l = dfc["low"].values
    E, slv, tp, rk = pr["entry_level"], pr["sl"], pr["tp"], pr["risk"]
    rr = pr.get("rr", cfg.RR)
    k = s.confirm_i
    for j in range(k + 1, min(k + 1 + WALK_BARS, len(dfc))):
        if s.direction == "sell":
            if h[j] + spread >= slv: return -1.0
            if l[j] + spread <= tp: return rr
        else:
            if l[j] <= slv: return -(1 + spread / rk)
            if h[j] >= tp: return rr - spread / rk
    return None


def run(df, utc1_all, mode):
    spread = SPREAD_PIPS * cfg.PIP[SYMBOL]
    days = pd.DatetimeIndex(utc1_all).normalize().unique()
    rows = []; last = None
    for day in days:
        d = day.date()
        if d.weekday() >= 5: continue
        sb = det.session_bounds_utc1(d)
        a = int(np.searchsorted(utc1_all, sb["asia_start"], "left"))
        b = int(np.searchsorted(utc1_all, sb["london_end"] + pd.Timedelta(hours=WALK_HOURS), "right"))
        if b - a < 120: continue
        sl_df = df.iloc[a:b]
        s = det.analyze_day(sl_df, SYMBOL, OFFSET_EST, session_date=d)
        if s is None or s.stage != "confirmed": continue
        if last is not None and s.confirm_time <= last: continue

        if mode == "opp":
            pr = det.build_asia_tp_prices(s, SYMBOL, spread)
        else:
            cfg.RR = 2.1 if mode == "r21" else 3.0
            pr = det.build_market_prices(s, SYMBOL, spread)
        if pr is None: continue
        dfc = sl_df.iloc[:-1]; t1 = det.utc1_index(dfc, OFFSET_EST)
        R = sim(dfc, t1, s, pr, spread, mode)
        if R is None: continue
        # advance the no-overlap cursor to this trade's exit
        k = s.confirm_i
        last = t1[min(k + 1, len(dfc) - 1)]
        rows.append({"date": str(d), "year": d.year, "dir": s.direction,
                     "rr": round(pr.get("rr", cfg.RR), 2), "R": round(R, 3), "win": R > 0})
    return pd.DataFrame(rows)


def summ(df, label):
    if not len(df):
        print(f"  {label:26} 0 trades"); return
    n = len(df); w = df["win"].sum(); R = df["R"].sum()
    gw = df.loc[df.R > 0, "R"].sum(); gl = -df.loc[df.R < 0, "R"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    yr = df.groupby("year")["R"].sum()
    extra = ""
    if "opp" in label.lower() or df["rr"].nunique() > 1:
        extra = f"  medRR={df['rr'].median():.2f}"
    print(f"  {label:26} N={n:>4} WR={w/n*100:5.1f}% PF={pf:4.2f} totR={R:+7.1f} "
          f"exp={R/n:+.3f}R green={(yr>0).sum()}/{len(yr)}{extra}")


def main():
    # override session hours to the NY/EST windows (live config untouched)
    cfg.ASIA_START_H, cfg.ASIA_END_H = ACC_START, ACC_END
    cfg.LONDON_START_H, cfg.LONDON_END_H = NY_START, NY_END

    df = pd.read_pickle(HD)
    utc1_all = det.utc1_index(df, OFFSET_EST)
    print("=" * 96)
    print(f"10AM PO3 · {SYMBOL} 1M · HistData 2004-2024 · EST · CISD trigger")
    print(f"accumulation {ACC_START:02d}:00-{ACC_END:02d}:00 EST · NY window "
          f"{NY_START:02d}:00-{NY_END:02d}:00 EST · spread {SPREAD_PIPS}p")
    print("=" * 96)
    for mode, name in (("opp", "TARGET opposing liq"), ("r21", "TARGET fixed 2.1R"),
                       ("r3", "TARGET fixed 3.0R")):
        t = run(df, utc1_all, mode)
        summ(t, name)
        if mode == "opp" and len(t):
            print(f"    achieved RR: min {t['rr'].min():.2f} · median {t['rr'].median():.2f} · "
                  f"mean {t['rr'].mean():.2f} · max {t['rr'].max():.2f}")
    print(f"\n  (PO3 PDF claims clean +3R to +8R moves are 'normal')")


if __name__ == "__main__":
    main()
