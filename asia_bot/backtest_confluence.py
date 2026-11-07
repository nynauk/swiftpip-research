"""
Reverse-engineer the discretionary "strong CISD" — codify the confluences and
stack them on the mechanical 1M CISD to see how far each lifts the win rate.

EURUSD · HistData 1min 2004-2024 · fixed 2.1R market-on-confirm (matches the bot
and the manual avg RR 2.01).

Confluences (all causal, known at entry):
  SWING (structure shift) — the confirmation must CLOSE beyond an M1 swing that
      formed during the manipulation: a short closes below a prior swing LOW, a
      long closes above a prior swing HIGH. Pivots use n bars each side and are
      only counted once confirmed (index <= confirm - n), so no lookahead.
  IFVG — the reversal (displacement) left a fair-value gap (setup.entry_source
      == 'fvg').

Reports the base and every filter combination so we can see the incremental lift.

    python -m asia_bot.backtest_confluence
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asia_bot import config as cfg
from asia_bot import detector as det

SYMBOL = "EURUSD"
HD = os.path.join("ext_data", "HD_EURUSD_1min_2004-2024.pkl")
OFFSET_H = -5
SPREAD_PIPS = 1.0
PIVOT_N = 3
WALK_BARS, WALK_HOURS = 3000, 60


def pivots(h, l, n):
    sh = np.zeros(len(h), bool); sl = np.zeros(len(l), bool)
    for i in range(n, len(h) - n):
        if h[i] >= h[i - n:i + n + 1].max(): sh[i] = True
        if l[i] <= l[i - n:i + n + 1].min(): sl[i] = True
    return sh, sl


def run():
    cfg.RR = 2.1
    df = pd.read_pickle(HD)
    utc1_all = det.utc1_index(df, OFFSET_H)
    spread = SPREAD_PIPS * cfg.PIP[SYMBOL]
    days = pd.DatetimeIndex(utc1_all).normalize().unique()
    rows = []; last = None

    for day in days:
        d = day.date()
        if d.weekday() >= 5: continue
        sb = det.session_bounds_utc1(d)
        a = int(np.searchsorted(utc1_all, sb["asia_start"], "left"))
        b = int(np.searchsorted(utc1_all, sb["london_end"] + pd.Timedelta(hours=WALK_HOURS), "right"))
        if b - a < 200: continue
        sl_df = df.iloc[a:b]
        s = det.analyze_day(sl_df, SYMBOL, OFFSET_H, session_date=d)
        if s is None or s.stage != "confirmed": continue
        if last is not None and s.confirm_time <= last: continue

        pr = det.build_market_prices(s, SYMBOL, spread)
        if pr is None: continue
        dfc = sl_df.iloc[:-1]; t1 = det.utc1_index(dfc, OFFSET_H)
        o = dfc["open"].values; h = dfc["high"].values; l = dfc["low"].values; c = dfc["close"].values
        k = s.confirm_i
        # simulate fixed-RR outcome
        E, slv, tp, rk = pr["entry_level"], pr["sl"], pr["tp"], pr["risk"]
        R = None; xi = None
        for j in range(k + 1, min(k + 1 + WALK_BARS, len(dfc))):
            if s.direction == "sell":
                if h[j] + spread >= slv: R = -1.0; xi = j; break
                if l[j] + spread <= tp: R = cfg.RR; xi = j; break
            else:
                if l[j] <= slv: R = -(1 + spread / rk); xi = j; break
                if h[j] >= tp: R = cfg.RR - spread / rk; xi = j; break
        if xi is None: continue
        last = t1[xi]

        # confluence flags (causal)
        sh, spl = pivots(h, l, PIVOT_N)
        swing_ok = False
        lo = max(0, s.sweep_i)
        hi = k - PIVOT_N                     # only pivots confirmed by now
        if s.direction == "sell":
            for p in range(hi, lo - 1, -1):
                if spl[p]:
                    swing_ok = c[k] < l[p]   # closed below a prior swing low
                    break
        else:
            for p in range(hi, lo - 1, -1):
                if sh[p]:
                    swing_ok = c[k] > h[p]   # closed above a prior swing high
                    break
        ifvg_ok = s.entry_source == "fvg"

        rows.append({"date": str(d), "year": d.year, "dir": s.direction,
                     "R": round(R, 3), "win": R > 0,
                     "swing": swing_ok, "ifvg": ifvg_ok})
    return pd.DataFrame(rows)


def summ(df, label):
    if not len(df):
        print(f"  {label:34} 0 trades"); return
    n = len(df); w = df["win"].sum(); R = df["R"].sum()
    gw = df.loc[df.R > 0, "R"].sum(); gl = -df.loc[df.R < 0, "R"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    yr = df.groupby("year")["R"].sum()
    print(f"  {label:34} N={n:>4}  WR={w/n*100:5.1f}%  PF={pf:4.2f}  totR={R:+7.1f}  "
          f"exp={R/n:+.3f}R  green={(yr>0).sum()}/{len(yr)}")


def main():
    t = run()
    print("=" * 96)
    print(f"CONFLUENCE STACK · {SYMBOL} 1M CISD · 2004-2024 · fixed 2.1R · pivots n={PIVOT_N}")
    print("=" * 96)
    summ(t, "BASE (every CISD)")
    summ(t[t.swing], "+ swing structure-shift")
    summ(t[t.ifvg], "+ IFVG")
    summ(t[t.swing & t.ifvg], "+ swing + IFVG (their setup)")
    print("  " + "-" * 92)
    summ(t[~t.swing], "(no swing — for contrast)")
    summ(t[~t.ifvg], "(no IFVG — for contrast)")
    # how selective is the full filter?
    full = t[t.swing & t.ifvg]
    print(f"\n  full filter keeps {len(full)}/{len(t)} = {len(full)/len(t)*100:.0f}% of CISDs "
          f"(~{len(full)/21:.0f}/yr)")
    print(f"  manual Casa reference: 75 trades / 6mo, 63.5% WR, PF 3.47")
    t.to_csv(os.path.join(os.path.dirname(__file__), "backtest", "confluence_1m.csv"), index=False)


if __name__ == "__main__":
    main()
