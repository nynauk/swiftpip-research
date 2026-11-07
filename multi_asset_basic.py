"""
Backtest the DRY BASIC strategy (prev-4H sweep + 15M IFVG, SL on disrespect
candle low/high, plain 3RR, no partial, no BE, no filters) across:
  · forex majors + crosses
  · indices  (GER40, UK100, US30, US500, US100 = NAS100)
  · commodities  (XAUUSD gold, XAGUSD silver, USOIL, UKOIL)
  · crypto  (BTCUSD, ETHUSD)

Runs candle-by-candle. Ranks by profit factor with a min-trade floor.
"""
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import main as bt
import eurusd_basic as eb


CATEGORIES = {
    "FX Major":     ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"],
    "FX Cross":     ["EURJPY", "GBPJPY", "EURGBP"],
    "Index":        ["US500.cash", "US30.cash", "US100.cash", "GER40.cash", "UK100.cash"],
    "Commodity":    ["XAUUSD", "XAGUSD", "USOIL.cash", "UKOIL.cash"],
    "Crypto":       ["BTCUSD", "ETHUSD"],
}

MIN_TRADES = 50            # floor to avoid crowning tiny-sample flukes


def streak_and_dd(df, initial=10_000, risk_pct=1.0):
    bal = initial; eq = [bal]; cl = cw = bcl = bcw = 0
    for _, r in df.sort_values("entry_time").iterrows():
        bal += bal * (risk_pct / 100) * float(r["pnl_r"]); eq.append(bal)
        if r["pnl_r"] < -0.01: cl += 1; cw = 0; bcl = max(bcl, cl)
        elif r["pnl_r"] > 0.01: cw += 1; cl = 0; bcw = max(bcw, cw)
    eqa = np.array(eq)
    dd = ((eqa - np.maximum.accumulate(eqa)) / np.maximum.accumulate(eqa) * 100).min()
    return bal, dd, bcl, bcw


def run_one(symbol):
    try:
        df_h4 = bt.fetch_data(symbol, mt5.TIMEFRAME_H4,  bt.START_DATE, bt.END_DATE, tf_name="H4")
        df_15 = bt.fetch_data(symbol, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}
    if len(df_h4) < 100 or len(df_15) < 200:
        return {"symbol": symbol, "error": "not enough data"}

    # Plain 3RR — no partial, no BE, no filters
    df = eb.run(df_h4, df_15, rr=3.0)
    df = df[df["outcome"] != "open"].copy()
    if not len(df):
        return {"symbol": symbol, "n": 0}

    n = len(df); wins = (df["outcome"] == "win").sum(); losses = n - wins
    wr = wins / n * 100
    gw = wins * 3.0; gl = losses
    pf = gw / gl if gl > 0 else float("inf")
    tot = df["pnl_r"].sum()
    exp = tot / n
    bal, dd, ls, ws = streak_and_dd(df)
    days = (pd.to_datetime(df["entry_time"]).max() - pd.to_datetime(df["entry_time"]).min()).days
    weeks = max(days / 7, 1)

    return {
        "symbol": symbol, "n": n, "wr": wr, "pf": pf, "tot": tot, "exp": exp,
        "dd": dd, "ls": ls, "ws": ws, "bal": bal,
        "per_week": n / weeks, "weeks": weeks,
    }


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    print("=" * 108)
    print("PLAIN 3RR — dry basic strategy across all asset classes")
    print("=" * 108)
    print(f"  {'category':10} {'symbol':13} {'N':>4} {'/wk':>4} "
          f"{'WR':>6} {'PF':>5} {'expR':>6} {'totR':>7} {'DD':>7} {'LStrk':>5} {'$10k→':>10}")

    rows = []
    for cat, syms in CATEGORIES.items():
        for s in syms:
            r = run_one(s)
            if "error" in r:
                print(f"  {cat:10} {s:13} · {r['error']}")
                continue
            if r["n"] == 0:
                print(f"  {cat:10} {s:13} · no trades")
                continue
            r["cat"] = cat
            rows.append(r)
            print(f"  {cat:10} {s:13} {r['n']:>4} {r['per_week']:>4.1f} "
                  f"{r['wr']:>5.1f}% {r['pf']:>4.2f} {r['exp']:>+5.2f}R {r['tot']:>+6.0f}R "
                  f"{r['dd']:>6.1f}% {r['ls']:>5} ${r['bal']:>8,.0f}")
        print()

    mt5.shutdown()

    # ── Rankings ──
    valid = [r for r in rows if r["n"] >= MIN_TRADES]
    if not valid:
        print("No results with min trades.")
        return

    print("=" * 108)
    print(f"TOP 10 BY PROFIT FACTOR  (min {MIN_TRADES} trades)")
    print("=" * 108)
    for r in sorted(valid, key=lambda x: x["pf"], reverse=True)[:10]:
        print(f"  {r['cat']:10} {r['symbol']:13} PF {r['pf']:>4.2f}  WR {r['wr']:>5.1f}%  "
              f"totR {r['tot']:>+6.0f}  expR {r['exp']:>+5.2f}  N {r['n']:>4}  DD {r['dd']:>5.1f}%")

    print("\n" + "=" * 108)
    print(f"TOP 10 BY TOTAL R")
    print("=" * 108)
    for r in sorted(valid, key=lambda x: x["tot"], reverse=True)[:10]:
        print(f"  {r['cat']:10} {r['symbol']:13} totR {r['tot']:>+6.0f}  PF {r['pf']:>4.2f}  "
              f"WR {r['wr']:>5.1f}%  N {r['n']:>4}  DD {r['dd']:>5.1f}%  LStrk {r['ls']}")

    print("\n" + "=" * 108)
    print(f"TOP 10 BY WIN RATE")
    print("=" * 108)
    for r in sorted(valid, key=lambda x: x["wr"], reverse=True)[:10]:
        print(f"  {r['cat']:10} {r['symbol']:13} WR {r['wr']:>5.1f}%  PF {r['pf']:>4.2f}  "
              f"totR {r['tot']:>+6.0f}  N {r['n']:>4}")

    print("\n" + "=" * 108)
    print("BEST-BY-CATEGORY (by total R)")
    print("=" * 108)
    for cat in CATEGORIES:
        subset = [r for r in valid if r["cat"] == cat]
        if not subset: continue
        b = max(subset, key=lambda x: x["tot"])
        print(f"  {cat:10} → {b['symbol']:13}  N={b['n']:>4}  WR={b['wr']:>5.1f}%  "
              f"PF={b['pf']:>4.2f}  totR={b['tot']:>+6.0f}  DD={b['dd']:>5.1f}%  "
              f"{b['per_week']:.1f}/wk")


if __name__ == "__main__":
    main()
