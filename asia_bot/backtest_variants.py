"""
1M CISD on EURUSD — three variants, both TP modes. HistData 2004-2024.

  A) BASELINE            — detector as currently configured.
  B) LEG-AFTER-SWEEP     — delivery run must start at/after the sweep bar.
                           Causal, knowable live (36% of legs currently start
                           before the sweep).
  C) LOOKAHEAD-EXTREME   — only CISDs whose manipulation extreme is STILL the
                           London high/low at 10:00. *** NOT TRADEABLE ***: at
                           entry you cannot know whether a higher high prints
                           later. Included ONLY as a theoretical upper bound.

    python -m asia_bot.backtest_variants
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
WALK_BARS = 3000
WALK_HOURS = 60
OUT = os.path.join(os.path.dirname(__file__), "backtest")


def sim(dfc, t1, setup, prices, spread, london_end, market):
    h = dfc["high"].values; l = dfc["low"].values
    E, sl, tp, r = prices["entry_level"], prices["sl"], prices["tp"], prices["risk"]
    rr = prices.get("rr", cfg.RR)
    n = len(dfc)
    if market:
        fill_i = setup.confirm_i; walk = fill_i + 1
    else:
        fill_i = None
        for j in range(setup.confirm_i + 1, n):
            if t1[j] >= london_end: break
            if setup.direction == "sell" and h[j] >= E: fill_i = j; break
            if setup.direction == "buy" and l[j] <= E: fill_i = j; break
        if fill_i is None: return "no_fill", 0.0, None, None
        walk = fill_i
    for j in range(walk, min(walk + WALK_BARS, n)):
        if setup.direction == "sell":
            if h[j] + spread >= sl: return "loss", -1.0, fill_i, j
            if l[j] + spread <= tp: return "win", rr, fill_i, j
        else:
            if l[j] <= sl: return "loss", -(1.0 + spread / r), fill_i, j
            if h[j] >= tp: return "win", rr - spread / r, fill_i, j
    return "open", 0.0, fill_i, None


def scan(df, utc1_all, lookahead_filter=False):
    """One detector pass -> trades for both TP modes."""
    spread = SPREAD_PIPS * cfg.PIP[SYMBOL]; pip = cfg.PIP[SYMBOL]
    days = pd.DatetimeIndex(utc1_all).normalize().unique()
    rows = {"fixed": [], "asia": []}
    last_exit = {"fixed": None, "asia": None}

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

        dfc = sl_df.iloc[:-1]
        t1 = det.utc1_index(dfc, OFFSET_H)

        if lookahead_filter:
            h = dfc["high"].values; l = dfc["low"].values
            lon = np.where((t1 >= sb["london_start"]) & (t1 < sb["london_end"]))[0]
            if not len(lon): continue
            lon0, lon_end = int(lon[0]), int(lon[-1])
            if s.direction == "sell":
                if s.sweep_extreme < h[lon0:lon_end + 1].max() - 1e-9: continue
            else:
                if s.sweep_extreme > l[lon0:lon_end + 1].min() + 1e-9: continue

        too_big = False
        if cfg.ENTRY_MODE == "market":
            s2l = abs(s.sweep_extreme - s.cisd_level)
            cb = abs(s.confirm_close - s.cisd_level)
            too_big = s2l > 0 and cb > cfg.CISD_BIG_CANDLE_MULT * s2l
        eo = s.cisd_level if too_big else None
        fill_market = (cfg.ENTRY_MODE == "market") and not too_big

        for mode in ("fixed", "asia"):
            if last_exit[mode] is not None and s.confirm_time <= last_exit[mode]:
                continue
            if mode == "asia":
                prices = det.build_asia_tp_prices(s, SYMBOL, spread, entry=eo)
            else:
                prices = (det.build_market_prices(s, SYMBOL, spread) if fill_market
                          else det.build_order_prices(s, SYMBOL, spread, entry_override=eo))
            if prices is None: continue
            oc, R, fi, xi = sim(dfc, t1, s, prices, spread, sb["london_end"], fill_market)
            if oc == "no_fill": continue
            if xi is not None: last_exit[mode] = t1[xi]
            rows[mode].append({"date": str(d), "year": d.year, "direction": s.direction,
                               "stop_p": round(prices["risk"] / pip, 1),
                               "rr": round(prices.get("rr", cfg.RR), 2),
                               "outcome": oc, "pnl_r": round(R, 3)})
    return {m: pd.DataFrame(v) for m, v in rows.items()}


def summ(df):
    cl = df[df["outcome"].isin(["win", "loss"])] if len(df) else df
    if not len(cl): return None
    n = len(cl); w = (cl["outcome"] == "win").sum(); R = cl["pnl_r"].sum()
    gw = cl.loc[cl["pnl_r"] > 0, "pnl_r"].sum(); gl = -cl.loc[cl["pnl_r"] < 0, "pnl_r"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    bal = 10000.0; eq = [bal]; ls = c = 0
    for _, t in cl.iterrows():
        ra = cfg.RISK_FLAT if bal < cfg.RISK_ANCHOR else bal * cfg.RISK_PCT / 100
        bal += ra * float(t["pnl_r"]); eq.append(bal)
        if t["pnl_r"] < 0: c += 1; ls = max(ls, c)
        else: c = 0
    e = np.array(eq); dd = ((e - np.maximum.accumulate(e)) / np.maximum.accumulate(e) * 100).min()
    yr = cl.groupby("year")["pnl_r"].sum()
    return dict(n=n, wr=w / n * 100, pf=pf, R=R, exp=R / n, dd=dd, ls=ls,
                green=(yr > 0).sum(), yrs=len(yr))


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_pickle(HD)
    utc1_all = det.utc1_index(df, OFFSET_H)
    print("=" * 104)
    print(f"1M CISD · {SYMBOL} · HistData 2004-2024 · entry={cfg.ENTRY_MODE} · RR {cfg.RR} · spread {SPREAD_PIPS}p")
    print("=" * 104)

    results = {}
    # A) baseline
    cfg.CISD_LEG_AFTER_SWEEP = False; cfg.CISD_LEG_MAKES_EXTREME = False
    results["A) baseline"] = scan(df, utc1_all)
    # D) causal: the 2+ delivery candles must MAKE the London high/low
    cfg.CISD_LEG_MAKES_EXTREME = True
    results["D) leg-makes-extreme (causal)"] = scan(df, utc1_all)
    cfg.CISD_LEG_MAKES_EXTREME = False
    # D+B) both causal filters stacked
    cfg.CISD_LEG_MAKES_EXTREME = True; cfg.CISD_LEG_AFTER_SWEEP = True
    results["D+B) makes-extreme + after-sweep"] = scan(df, utc1_all)
    cfg.CISD_LEG_MAKES_EXTREME = False; cfg.CISD_LEG_AFTER_SWEEP = False
    # C) lookahead upper bound
    results["C) lookahead-extreme (NOT TRADEABLE)"] = scan(df, utc1_all, lookahead_filter=True)

    for mode, mname in (("fixed", f"FIXED TP ({cfg.RR}R)"), ("asia", "ASIA TP (opposing liquidity)")):
        print(f"\n{'#'*104}\n#  {mname}\n{'#'*104}")
        print(f"  {'variant':40} {'N':>4} {'WR':>6} {'PF':>5} {'totR':>8} {'exp':>8} {'DD':>7} {'maxLS':>5} {'green':>7}")
        for vname, r in results.items():
            s = summ(r[mode])
            if not s:
                print(f"  {vname:40} no trades"); continue
            print(f"  {vname:40} {s['n']:>4} {s['wr']:>5.1f}% {s['pf']:>5.2f} {s['R']:>+8.1f} "
                  f"{s['exp']:>+7.3f}R {s['dd']:>6.1f}% {s['ls']:>5} {s['green']:>3}/{s['yrs']:<3}")
            r[mode].to_csv(os.path.join(OUT, f"var_{vname[0]}_{mode}.csv"), index=False)
    print(f"\nsaved → {OUT}/ (var_A_*.csv, var_B_*.csv, var_C_*.csv)")


if __name__ == "__main__":
    main()
