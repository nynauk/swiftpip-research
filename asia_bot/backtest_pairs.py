"""
asia_bot M5 CISD across the 7 FX majors, each tested SEPARATELY.
HistData M1 zips → M5 (2000-2024, EST no DST → offset_h = -5).

Same detector the live bot runs. Both take-profit modes are evaluated on the
same detected setups (analyze_day is the bottleneck, so it runs once per day and
both exits are priced off it):
  FIXED — TP at cfg.RR (config default 2.1R)
  ASIA  — TP at the opposing Asia liquidity

    python -m asia_bot.backtest_pairs            # all 7 majors
    python -m asia_bot.backtest_pairs EURUSD     # one pair
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asia_bot import config as cfg
from asia_bot import detector as det
from asia_bot.load_histdata import build_m5

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
SPREAD_PIPS = {"EURUSD": 1.0, "GBPUSD": 1.5, "USDJPY": 1.0, "USDCHF": 1.5,
               "AUDUSD": 1.5, "USDCAD": 1.5, "NZDUSD": 2.0}
OFFSET_H = -5
WALK_BARS = 600            # ~50h on M5
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


def run_pair(pair):
    df = build_m5(pair)
    if df is None:
        return None
    spread = SPREAD_PIPS[pair] * cfg.PIP[pair]
    pip = cfg.PIP[pair]
    utc1_all = det.utc1_index(df, OFFSET_H)
    days = pd.DatetimeIndex(utc1_all).normalize().unique()
    rows = {"fixed": [], "asia": []}
    last_exit = {"fixed": None, "asia": None}

    for day in days:
        d = day.date()
        if d.weekday() >= 5: continue
        sb = det.session_bounds_utc1(d)
        a = int(np.searchsorted(utc1_all, sb["asia_start"], "left"))
        b = int(np.searchsorted(utc1_all, sb["london_end"] + pd.Timedelta(hours=WALK_HOURS), "right"))
        if b - a < 60: continue
        sl_df = df.iloc[a:b]
        setup = det.analyze_day(sl_df, pair, OFFSET_H, session_date=d)
        if setup is None or setup.stage != "confirmed": continue

        too_big = False
        if cfg.ENTRY_MODE == "market":
            s2l = abs(setup.sweep_extreme - setup.cisd_level)
            cb = abs(setup.confirm_close - setup.cisd_level)
            too_big = s2l > 0 and cb > cfg.CISD_BIG_CANDLE_MULT * s2l
        eo = setup.cisd_level if too_big else None
        fill_market = (cfg.ENTRY_MODE == "market") and not too_big
        dfc = sl_df.iloc[:-1]
        t1 = det.utc1_index(dfc, OFFSET_H)

        for mode in ("fixed", "asia"):
            if last_exit[mode] is not None and setup.confirm_time <= last_exit[mode]:
                continue
            if mode == "asia":
                prices = det.build_asia_tp_prices(setup, pair, spread, entry=eo)
            else:
                prices = (det.build_market_prices(setup, pair, spread) if fill_market
                          else det.build_order_prices(setup, pair, spread, entry_override=eo))
            if prices is None: continue
            oc, R, fi, xi = sim(dfc, t1, setup, prices, spread, sb["london_end"], fill_market)
            if oc == "no_fill": continue
            if xi is not None: last_exit[mode] = t1[xi]
            rows[mode].append({"pair": pair, "date": str(d), "year": d.year,
                               "direction": setup.direction,
                               "stop_p": round(prices["risk"] / pip, 1),
                               "rr": round(prices.get("rr", cfg.RR), 2),
                               "outcome": oc, "pnl_r": round(R, 3)})
    return {m: pd.DataFrame(v) for m, v in rows.items()}


def summ(df):
    cl = df[df["outcome"].isin(["win", "loss"])]
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
                green=(yr > 0).sum(), yrs=len(yr), stop=cl["stop_p"].median())


def main():
    os.makedirs(OUT, exist_ok=True)
    pairs = sys.argv[1:] or PAIRS
    print("=" * 108)
    print(f"ASIA-SWEEP + LONDON M5-CISD · EACH PAIR SEPARATELY · HistData M5 2000-2024")
    print(f"entry={cfg.ENTRY_MODE} · RR {cfg.RR} · risk flat {cfg.RISK_FLAT:.0f} <{cfg.RISK_ANCHOR:.0f} else {cfg.RISK_PCT}%")
    print("=" * 108)
    hdr = (f"  {'pair':8} {'N':>4} {'WR':>6} {'PF':>5} {'totR':>8} {'exp':>8} "
           f"{'maxLS':>5} {'stop':>6} {'green':>7}")
    store = {"fixed": [], "asia": []}
    results = {}
    for p in pairs:
        r = run_pair(p)
        if r is None:
            print(f"  {p}: no data"); continue
        results[p] = {m: summ(d) for m, d in r.items()}
        for m in ("fixed", "asia"):
            if len(r[m]): store[m].append(r[m])

    for mode, name in (("fixed", f"FIXED TP ({cfg.RR}R)"), ("asia", "ASIA TP (opposing liquidity)")):
        print(f"\n{'#'*108}\n#  {name}\n{'#'*108}")
        print(hdr)
        for p in pairs:
            s = results.get(p, {}).get(mode)
            if not s: continue
            print(f"  {p:8} {s['n']:>4} {s['wr']:>5.1f}% {s['pf']:>5.2f} {s['R']:>+8.1f} "
                  f"{s['exp']:>+7.3f}R {s['ls']:>5} {s['stop']:>5.1f}p {s['green']:>3}/{s['yrs']:<3}")
        if store[mode]:
            allt = pd.concat(store[mode], ignore_index=True)
            s = summ(allt)
            print(f"  {'-'*100}")
            print(f"  {'ALL':8} {s['n']:>4} {s['wr']:>5.1f}% {s['pf']:>5.2f} {s['R']:>+8.1f} "
                  f"{s['exp']:>+7.3f}R {s['ls']:>5} {s['stop']:>5.1f}p")
            allt.to_csv(os.path.join(OUT, f"pairs_m5_{mode}.csv"), index=False)
    print(f"\nsaved → {OUT}/ (pairs_m5_fixed.csv, pairs_m5_asia.csv)")


if __name__ == "__main__":
    main()
