"""
asia_bot M5 CISD backtest — 2024 → present day (EURUSD).

Data is spliced from two sources onto ONE common UTC timeline:
  • 2024            → HistData 5min (index is EST, fixed, no DST)  → UTC = idx + 5
  • 2025-03-12 → now → MT5 M5 (index is broker EET/EEST, WITH DST) → UTC via
                       the Europe/Athens zone (GMT+2 winter / +3 summer)
A ~10-week gap (2025-01-01 → 2025-03-11) exists — neither source covers it.

Because everything is normalised to UTC, the detector runs with offset_h = 0
(its utc1_index does UTC + 1 = the bot's fixed UTC+1 session clock). This is the
only correct way to mix the two feeds: a single fixed offset would misplace every
winter session by an hour once DST is involved.

    python -m asia_bot.backtest_recent
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asia_bot import config as cfg
from asia_bot import detector as det

SYMBOL = "EURUSD"
SPREAD_PIPS = 1.0
OFFSET_H = 0                 # frames are already UTC
WALK_BARS = 600              # ~50h on M5
WALK_HOURS = 60
OUT = os.path.join(os.path.dirname(__file__), "backtest")


def build_utc_frame():
    # --- HistData 5min (EST fixed) -> UTC ---
    hd = pd.read_pickle(os.path.join("ext_data", "HD_EURUSD_5min_2000-2024.pkl"))
    hd = hd[hd.index >= "2024-01-01"]
    hd.index = hd.index + pd.Timedelta(hours=5)          # EST(-5) -> UTC
    hd = hd[["open", "high", "low", "close"]]

    # --- MT5 M5 (broker EET/EEST, DST-aware) -> UTC ---
    mt = pd.read_pickle(os.path.join("data_cache", "EURUSD_M5_mt5.pkl"))
    idx = mt.index.tz_localize("Europe/Athens", ambiguous=True,
                               nonexistent="shift_forward").tz_convert("UTC")
    mt = mt.copy()
    mt.index = idx.tz_localize(None)
    mt = mt[["open", "high", "low", "close"]]

    print(f"  HistData 2024 (UTC): {hd.index[0]} -> {hd.index[-1]}  ({len(hd):,} bars)")
    print(f"  MT5      (UTC): {mt.index[0]} -> {mt.index[-1]}  ({len(mt):,} bars)")
    gap0, gap1 = hd.index[-1], mt.index[0]
    print(f"  GAP: {gap0} -> {gap1}  ({(gap1-gap0).days} days with no data)")

    df = pd.concat([hd, mt]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def verify_utc(df):
    """FX week should close ~21:00-22:00 UTC Fri and reopen ~21:00-22:00 UTC Sun.
    If our tz maths were wrong this would sit at a nonsense hour."""
    d = df.index
    fri = d[d.dayofweek == 4]; sun = d[d.dayofweek == 6]
    fh = pd.Series(fri.hour).value_counts().sort_index()
    sh = pd.Series(sun.hour).value_counts().sort_index()
    print(f"  Fri last-bar hours (top): {list(fh.tail(3).index)}  | Sun first-bar hours: {list(sh.head(3).index)}")
    sat = d[d.dayofweek == 5]
    print(f"  Saturday bars: {len(sat)} (should be ~0)  → UTC alignment {'OK' if len(sat) < 500 else 'SUSPECT'}")


def sim_trade(dfc, t1, setup, prices, spread, london_end, market=False):
    h = dfc["high"].values; l = dfc["low"].values
    E, sl, tp, r = prices["entry_level"], prices["sl"], prices["tp"], prices["risk"]
    rr = prices.get("rr", cfg.RR)
    n = len(dfc)
    if market:
        fill_i = setup.confirm_i; walk_start = fill_i + 1
    else:
        fill_i = None
        for j in range(setup.confirm_i + 1, n):
            if t1[j] >= london_end: break
            if setup.direction == "sell" and h[j] >= E: fill_i = j; break
            if setup.direction == "buy" and l[j] <= E: fill_i = j; break
        if fill_i is None:
            return "no_fill", 0.0, None, None
        walk_start = fill_i
    for j in range(walk_start, min(walk_start + WALK_BARS, n)):
        if setup.direction == "sell":
            if h[j] + spread >= sl: return "loss", -1.0, fill_i, j
            if l[j] + spread <= tp: return "win", rr, fill_i, j
        else:
            if l[j] <= sl: return "loss", -(1.0 + spread / r), fill_i, j
            if h[j] >= tp: return "win", rr - spread / r, fill_i, j
    return "open", 0.0, fill_i, None


def run(df, utc1_all, tp_mode):
    spread = SPREAD_PIPS * cfg.PIP[SYMBOL]; pip = cfg.PIP[SYMBOL]
    days = pd.DatetimeIndex(utc1_all).normalize().unique()
    trades = []; last_exit_t = None
    for day in days:
        d = day.date()
        if d.weekday() >= 5: continue
        sb = det.session_bounds_utc1(d)
        a = int(np.searchsorted(utc1_all, sb["asia_start"], "left"))
        b = int(np.searchsorted(utc1_all, sb["london_end"] + pd.Timedelta(hours=WALK_HOURS), "right"))
        if b - a < 60: continue
        sl_df = df.iloc[a:b]
        setup = det.analyze_day(sl_df, SYMBOL, OFFSET_H, session_date=d)
        if setup is None or setup.stage != "confirmed": continue
        if last_exit_t is not None and setup.confirm_time <= last_exit_t: continue

        too_big = False
        if cfg.ENTRY_MODE == "market":
            s2l = abs(setup.sweep_extreme - setup.cisd_level)
            cb = abs(setup.confirm_close - setup.cisd_level)
            too_big = s2l > 0 and cb > cfg.CISD_BIG_CANDLE_MULT * s2l
        eo = setup.cisd_level if too_big else None
        fill_market = (cfg.ENTRY_MODE == "market") and not too_big
        if tp_mode == "asia":
            prices = det.build_asia_tp_prices(setup, SYMBOL, spread, entry=eo)
        else:
            prices = (det.build_market_prices(setup, SYMBOL, spread) if fill_market
                      else det.build_order_prices(setup, SYMBOL, spread, entry_override=eo))
        if prices is None: continue
        dfc = sl_df.iloc[:-1]; t1 = det.utc1_index(dfc, OFFSET_H)
        outcome, R, fi, xi = sim_trade(dfc, t1, setup, prices, spread, sb["london_end"], market=fill_market)
        if outcome == "no_fill": continue
        if xi is not None: last_exit_t = t1[xi]
        trades.append({"date": str(d), "year": d.year, "direction": setup.direction,
                       "entry": round(prices["entry_level"], 5), "sl": round(prices["sl"], 5),
                       "tp": round(prices["tp"], 5), "stop_p": round(prices["risk"] / pip, 1),
                       "rr": round(prices.get("rr", cfg.RR), 2), "outcome": outcome,
                       "pnl_r": round(R, 3), "fill_time": str(t1[fi])})
    out = pd.DataFrame(trades)
    return out.sort_values("fill_time").reset_index(drop=True) if len(out) else out


def summ(cl, tag):
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
    print(f"  {tag:24} N={n:>4} WR={w/n*100:5.1f}% PF={pf:4.2f} totR={R:+7.1f} "
          f"exp={R/n:+.3f}R DD={dd:6.1f}% maxLS={ls:>2} end=£{bal:,.0f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("building 2024 → present UTC M5 frame ...")
    df = build_utc_frame()
    verify_utc(df)
    utc1_all = det.utc1_index(df, OFFSET_H)
    print("\n" + "=" * 104)
    print(f"M5 CISD · {SYMBOL} · 2024 → PRESENT · entry={cfg.ENTRY_MODE} · RR {cfg.RR}")
    print(f"data: {utc1_all[0]} → {utc1_all[-1]} (UTC+1) · {len(df):,} bars · spread {SPREAD_PIPS}p")
    print("=" * 104)

    for mode, name in (("fixed", f"FIXED TP ({cfg.RR}R)"), ("asia", "ASIA TP (opposing liq)")):
        t = run(df, utc1_all, mode)
        if not len(t):
            print(f"  {name}: no trades"); continue
        cl = t[t["outcome"].isin(["win", "loss"])]
        print()
        summ(cl, name)
        print(f"    {'period':22} {'N':>4} {'WR':>6} {'PF':>5} {'totR':>7}")
        for per, g in [("2024 (HistData)", cl[cl.year == 2024]),
                       ("2025 (MT5, Mar+)", cl[cl.year == 2025]),
                       ("2026 YTD (MT5)", cl[cl.year == 2026])]:
            if not len(g): continue
            n = len(g); w = (g["outcome"] == "win").sum()
            gw = g.loc[g["pnl_r"] > 0, "pnl_r"].sum(); gl = -g.loc[g["pnl_r"] < 0, "pnl_r"].sum()
            pf = gw / gl if gl > 0 else 9.99
            print(f"    {per:22} {n:>4} {w/n*100:>5.1f}% {pf:>5.2f} {g['pnl_r'].sum():>+7.1f}")
        t.to_csv(os.path.join(OUT, f"recent_m5_{mode}.csv"), index=False)
    print(f"\nsaved → {OUT}/ (recent_m5_fixed.csv, recent_m5_asia.csv)")


if __name__ == "__main__":
    main()
