"""
Backtest of the DEPLOYED asia_bot logic — EURUSD only.

Per session day:
  Asia sweep → London 1M CISD (>=2-candle delivery, London-extreme guard) →
  market-execute at the confirmation close (or, if that candle is too big, rest
  a limit at the CISD level) → SL at the swing (CISD formation → confirmation)
  + pad + spread.

Two take-profit modes are compared:
  FIXED  — TP at cfg.RR (currently 2.1R). Constant R per win.
  ASIA   — TP at the OPPOSING Asia liquidity (high swept → target the Asia low,
           low swept → target the Asia high). R is variable per trade.

Realistic fills: bars are BID, a buy fills at ASK, so a buy WIN pays
(rr - spread/risk)R and a buy LOSS -(1 + spread/risk)R. Sells are already
spread-honest (SL widened by spread, exits tested at ask).

    python -m asia_bot.backtest
"""
import os, sys
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MetaTrader5 as mt5

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asia_bot import config as cfg
from asia_bot import detector as det

SYMBOL = "EURUSD"
SPREAD_PIPS = 1.0
OUT = os.path.join(os.path.dirname(__file__), "backtest")
CACHE = "data_cache"


def refresh_m1(sym):
    os.makedirs(CACHE, exist_ok=True)
    mt5.symbol_select(sym, True)
    bars = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M1, 0, 99_999)
    if bars is None or len(bars) == 0:
        return None
    df = pd.DataFrame(bars)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")[["open", "high", "low", "close"]]
    df.to_pickle(os.path.join(CACHE, f"{sym}_M1.pkl"))
    return df


def sim_trade(dfc, t1, setup, prices, spread, london_end, market=False):
    """Fill then walk to SL/TP. Returns (outcome, realized_R, fill_i, exit_i)."""
    h = dfc["high"].values; l = dfc["low"].values
    E, sl, tp, r = prices["entry_level"], prices["sl"], prices["tp"], prices["risk"]
    rr = prices.get("rr", cfg.RR)
    n = len(dfc)

    if market:
        fill_i = setup.confirm_i
        walk_start = fill_i + 1
    else:
        fill_i = None
        for j in range(setup.confirm_i + 1, n):
            if t1[j] >= london_end:
                break
            if setup.direction == "sell" and h[j] >= E:
                fill_i = j; break
            if setup.direction == "buy" and l[j] <= E:
                fill_i = j; break
        if fill_i is None:
            return "no_fill", 0.0, None, None
        walk_start = fill_i

    for j in range(walk_start, min(walk_start + 3000, n)):
        if setup.direction == "sell":
            if h[j] + spread >= sl:
                return "loss", -1.0, fill_i, j
            if l[j] + spread <= tp:
                return "win", rr, fill_i, j
        else:
            if l[j] <= sl:
                return "loss", -(1.0 + spread / r), fill_i, j
            if h[j] >= tp:
                return "win", rr - spread / r, fill_i, j
    return "open", 0.0, fill_i, None


def run(df, dfc, t1, offset_h, tp_mode):
    spread = SPREAD_PIPS * cfg.PIP[SYMBOL]
    pip = cfg.PIP[SYMBOL]
    days = sorted(set(pd.DatetimeIndex(t1).date))
    trades = []
    last_exit = -1
    n_open = 0
    for day in days:
        setup = det.analyze_day(df, SYMBOL, offset_h, session_date=day)
        if setup is None or setup.stage != "confirmed":
            continue
        if setup.confirm_i <= last_exit:
            continue

        too_big = False
        if cfg.ENTRY_MODE == "market":
            stop_to_level = abs(setup.sweep_extreme - setup.cisd_level)
            close_beyond = abs(setup.confirm_close - setup.cisd_level)
            too_big = stop_to_level > 0 and close_beyond > cfg.CISD_BIG_CANDLE_MULT * stop_to_level

        entry_override = setup.cisd_level if too_big else None
        fill_market = (cfg.ENTRY_MODE == "market") and not too_big

        if tp_mode == "asia":
            prices = det.build_asia_tp_prices(setup, SYMBOL, spread, entry=entry_override)
        else:
            prices = (det.build_market_prices(setup, SYMBOL, spread) if fill_market
                      else det.build_order_prices(setup, SYMBOL, spread,
                                                  entry_override=entry_override))
        if prices is None:
            continue

        sb = det.session_bounds_utc1(day)
        outcome, R, fi, xi = sim_trade(dfc, t1, setup, prices, spread,
                                        sb["london_end"], market=fill_market)
        if outcome == "no_fill":
            continue
        if outcome == "open":
            n_open += 1
        if xi is not None:
            last_exit = xi
        trades.append({
            "date": str(day), "direction": setup.direction,
            "entry_type": "limit(big)" if too_big else "market",
            "confirm": str(setup.confirm_time)[-8:-3],
            "entry": round(prices["entry_level"], 5), "sl": round(prices["sl"], 5),
            "tp": round(prices["tp"], 5),
            "stop_p": round(prices["risk"] / pip, 1),
            "rr": round(prices.get("rr", cfg.RR), 2),
            "outcome": outcome, "pnl_r": round(R, 3),
            "fill_time": str(t1[fi]),
        })
    out = pd.DataFrame(trades)
    if len(out):
        out = out.sort_values("fill_time").reset_index(drop=True)
    return out, n_open


def stats(df, label):
    cl = df[df["outcome"].isin(["win", "loss"])] if len(df) else df
    if not len(cl):
        print(f"  {label}: no closed trades"); return None
    n = len(cl)
    w = (cl["outcome"] == "win").sum()
    wr = w / n * 100
    R = cl["pnl_r"].sum()
    gw = cl.loc[cl["pnl_r"] > 0, "pnl_r"].sum()
    gl = -cl.loc[cl["pnl_r"] < 0, "pnl_r"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    bal = 10000.0; eq = [bal]; ls = cl_ = 0
    for _, t in cl.iterrows():
        risk_amt = cfg.RISK_FLAT if bal < cfg.RISK_ANCHOR else bal * (cfg.RISK_PCT / 100)
        bal += risk_amt * float(t["pnl_r"]); eq.append(bal)
        if t["pnl_r"] < 0: cl_ += 1; ls = max(ls, cl_)
        else: cl_ = 0
    eqa = np.array(eq)
    dd = ((eqa - np.maximum.accumulate(eqa)) / np.maximum.accumulate(eqa) * 100).min()
    print(f"  {label:28} N={n:>3}  WR={wr:5.1f}%  PF={pf:4.2f}  totR={R:+7.1f}  "
          f"exp={R/n:+.3f}R  DD={dd:5.1f}%  maxLS={ls:>2}  end=${bal:,.0f}")
    return eqa


def print_week(df, title):
    if not len(df):
        print(f"  {title}: no trades"); return
    latest = pd.Timestamp(max(df["date"])).date()
    wk_start = latest - timedelta(days=latest.weekday())
    wk = df[pd.to_datetime(df["date"]).dt.date >= wk_start]
    print(f"\n  {title} — week of {wk_start} ({len(wk)} trades)")
    if not len(wk):
        print("    (none)"); return
    print(f"    {'date':11} {'dir':5} {'type':11} {'conf':6} {'entry':>8} {'sl':>8} "
          f"{'tp':>8} {'stop':>6} {'rr':>5} {'out':5} {'R':>6} {'£@90':>8}")
    tot = 0.0
    for _, t in wk.iterrows():
        gbp = t["pnl_r"] * cfg.RISK_FLAT
        tot += gbp
        print(f"    {t['date']:11} {t['direction']:5} {t['entry_type']:11} {t['confirm']:6} "
              f"{t['entry']:>8} {t['sl']:>8} {t['tp']:>8} {t['stop_p']:>5.1f}p "
              f"{t['rr']:>5.2f} {t['outcome']:5} {t['pnl_r']:>+6.2f} {gbp:>+8.2f}")
    print(f"    {'':72} WEEK {wk['pnl_r'].sum():>+6.2f}R {tot:>+8.2f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    t = mt5.symbol_info_tick(SYMBOL)
    offset_h = det.broker_offset_hours(t.time) if t else 3
    print("refreshing M1 data from MT5 ...")
    df = refresh_m1(SYMBOL)
    mt5.shutdown()
    if df is None:
        print("no data"); return
    dfc = df.iloc[:-1]
    t1 = det.utc1_index(dfc, offset_h)

    print("=" * 100)
    print(f"ASIA-SWEEP + LONDON 1M-CISD · {SYMBOL} ONLY · entry={cfg.ENTRY_MODE}")
    print(f"risk: flat {cfg.RISK_FLAT:.0f} below {cfg.RISK_ANCHOR:.0f}, else {cfg.RISK_PCT}% "
          f"· spread {SPREAD_PIPS}p · data {min(pd.DatetimeIndex(t1).date)} → {max(pd.DatetimeIndex(t1).date)}")
    print("=" * 100)

    curves = {}
    for mode, name in (("fixed", f"FIXED TP ({cfg.RR}R)"),
                       ("asia", "ASIA TP (opposing liquidity)")):
        print(f"\n{'#'*100}\n#  {name}\n{'#'*100}")
        tdf, n_open = run(df, dfc, t1, offset_h, mode)
        if not len(tdf):
            print("  no trades"); continue
        eqa = stats(tdf, SYMBOL)
        curves[name] = eqa
        if mode == "asia":
            rr = tdf["rr"]
            print(f"    achieved RR → min {rr.min():.2f} · median {rr.median():.2f} · "
                  f"mean {rr.mean():.2f} · max {rr.max():.2f}")
            wins = tdf[tdf['outcome'] == 'win']
            if len(wins):
                print(f"    winners' RR → median {wins['rr'].median():.2f} · mean {wins['rr'].mean():.2f}")
        if n_open:
            print(f"    ({n_open} trade(s) never hit SL or TP within the walk window)")
        print_week(tdf, name)
        tdf.to_csv(os.path.join(OUT, f"eurusd_{mode}.csv"), index=False)

    BG, FG = "#0d1117", "#e6edf3"
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG); ax.set_facecolor(BG)
    for (name, eqa), col in zip(curves.items(), ["#58a6ff", "#3fb950"]):
        if eqa is not None:
            ax.plot(eqa, color=col, linewidth=1.5, label=name)
    ax.axhline(10000, color="#444c56", ls="--", lw=0.8)
    ax.set_title(f"asia_bot · {SYMBOL} · fixed {cfg.RR}R vs opposing-Asia-liquidity TP",
                 color=FG, fontweight="bold")
    ax.set_xlabel("Trade #", color=FG); ax.set_ylabel("Balance", color=FG)
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.legend(facecolor="#161b22", edgecolor="#444c56", labelcolor=FG)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "equity_eurusd.png"), dpi=140, facecolor=BG)
    plt.close(fig)
    print(f"\nsaved → {OUT}/ (eurusd_fixed.csv, eurusd_asia.csv, equity_eurusd.png)")


if __name__ == "__main__":
    main()
