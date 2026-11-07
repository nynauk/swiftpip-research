"""
asia_bot backtest on the FULL HistData 1-minute EURUSD feed (2004-2024, 7.4M bars).

Same detector the live bot runs: Asia sweep → London 1M CISD (>=2-candle
delivery, London-extreme guard) → market at the confirmation close (or a limit
at the CISD level if that candle is too big) → SL at the swing (CISD formation →
confirmation) + pad + spread.

Two take-profit modes compared:
  FIXED — TP at cfg.RR (2.1R), constant R per win.
  ASIA  — TP at the OPPOSING Asia liquidity (high swept → Asia low, and vice
          versa). R is variable per trade.

Timezone: HistData is EST with NO DST (verified via the Fri-17:00/Sun-17:00
weekly gap), so offset_h = -5 maps it onto the bot's fixed UTC+1 session clock
(UTC+1 = EST + 6) — matching how the live bot anchors its sessions.

Perf: the frame is sliced per session day (Asia start → London end + a forward
window for the exit walk) so analyze_day never rescans 7.4M rows.

    python -m asia_bot.backtest_hd
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asia_bot import config as cfg
from asia_bot import detector as det

SYMBOL = "EURUSD"
OFFSET_H = -5                    # HistData EST(-5) -> bot's UTC+1 clock
SPREAD_PIPS = 1.0
WALK_HOURS = 60                  # forward window kept in each slice for the exit walk
OUT = os.path.join(os.path.dirname(__file__), "backtest")

# timeframe → (data file, bars to walk for the exit ≈ 50h in each TF)
HD_FILES = {
    "1min": (os.path.join("ext_data", "HD_EURUSD_1min_2004-2024.pkl"), 3000),
    "5min": (os.path.join("ext_data", "HD_EURUSD_5min_2000-2024.pkl"), 600),
    "3min": (os.path.join("ext_data", "HD_EURUSD_3min_2000-2024.pkl"), 1000),
    "15min": (os.path.join("ext_data", "HD_EURUSD_15min_2000-2024.pkl"), 200),
}
TF = sys.argv[1] if len(sys.argv) > 1 else "1min"
HD_PATH, WALK_BARS = HD_FILES[TF]

# optional RR override for testing — does NOT touch config.py on disk.
# detector reads cfg.RR at call time, so mutating the module here is enough.
if len(sys.argv) > 2:
    cfg.RR = float(sys.argv[2])


def sim_trade(dfc, t1, setup, prices, spread, london_end, market=False):
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

    for j in range(walk_start, min(walk_start + WALK_BARS, n)):
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


def run(df, utc1_all, tp_mode):
    spread = SPREAD_PIPS * cfg.PIP[SYMBOL]
    pip = cfg.PIP[SYMBOL]
    days = pd.DatetimeIndex(utc1_all).normalize().unique()
    trades = []
    last_exit_t = None
    n_open = 0

    for day in days:
        d = day.date()
        if d.weekday() >= 5:
            continue
        sb = det.session_bounds_utc1(d)
        a = int(np.searchsorted(utc1_all, sb["asia_start"], "left"))
        b = int(np.searchsorted(utc1_all, sb["london_end"] + pd.Timedelta(hours=WALK_HOURS), "right"))
        if b - a < 60:
            continue
        sl_df = df.iloc[a:b]

        setup = det.analyze_day(sl_df, SYMBOL, OFFSET_H, session_date=d)
        if setup is None or setup.stage != "confirmed":
            continue
        if last_exit_t is not None and setup.confirm_time <= last_exit_t:
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

        dfc = sl_df.iloc[:-1]
        t1 = det.utc1_index(dfc, OFFSET_H)
        outcome, R, fi, xi = sim_trade(dfc, t1, setup, prices, spread,
                                        sb["london_end"], market=fill_market)
        if outcome == "no_fill":
            continue
        if outcome == "open":
            n_open += 1
        if xi is not None:
            last_exit_t = t1[xi]
        trades.append({
            "date": str(d), "year": d.year, "direction": setup.direction,
            "entry_type": "limit(big)" if too_big else "market",
            "entry": round(prices["entry_level"], 5), "sl": round(prices["sl"], 5),
            "tp": round(prices["tp"], 5), "stop_p": round(prices["risk"] / pip, 1),
            "rr": round(prices.get("rr", cfg.RR), 2),
            "outcome": outcome, "pnl_r": round(R, 3),
            "fill_time": str(t1[fi]),
        })
    out = pd.DataFrame(trades)
    return (out.sort_values("fill_time").reset_index(drop=True) if len(out) else out), n_open


def summarize(cl):
    n = len(cl)
    w = (cl["outcome"] == "win").sum()
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
    return dict(n=n, wr=w / n * 100, pf=pf, R=R, exp=R / n, dd=dd, ls=ls, bal=bal), eqa


def report(df, label):
    cl = df[df["outcome"].isin(["win", "loss"])] if len(df) else df
    if not len(cl):
        print(f"  {label}: no closed trades"); return None
    s, eqa = summarize(cl)
    print(f"  {label:30} N={s['n']:>4}  WR={s['wr']:5.1f}%  PF={s['pf']:4.2f}  "
          f"totR={s['R']:+8.1f}  exp={s['exp']:+.3f}R  DD={s['dd']:5.1f}%  maxLS={s['ls']:>2}")
    return eqa


def per_year(df, label):
    cl = df[df["outcome"].isin(["win", "loss"])]
    print(f"\n  {label} — by year")
    print(f"    {'year':6} {'N':>4} {'WR':>6} {'PF':>5} {'totR':>8}")
    green = 0; yrs = 0
    for y, g in cl.groupby("year"):
        s, _ = summarize(g)
        yrs += 1
        if s["R"] > 0: green += 1
        print(f"    {y:<6} {s['n']:>4} {s['wr']:>5.1f}% {s['pf']:>5.2f} {s['R']:>+8.1f}")
    print(f"    → {green}/{yrs} years profitable")


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"loading {HD_PATH} ...")
    df = pd.read_pickle(HD_PATH)
    utc1_all = det.utc1_index(df, OFFSET_H)
    print(f"  {len(df):,} M1 bars · {utc1_all[0]} → {utc1_all[-1]} (UTC+1)")

    print("=" * 104)
    print(f"ASIA-SWEEP + LONDON {TF.upper()}-CISD · {SYMBOL} · HistData {TF} · entry={cfg.ENTRY_MODE}")
    print(f"risk: flat {cfg.RISK_FLAT:.0f} below {cfg.RISK_ANCHOR:.0f}, else {cfg.RISK_PCT}% · "
          f"spread {SPREAD_PIPS}p · sessions (UTC+1) Asia {cfg.ASIA_START_H:02d}-{cfg.ASIA_END_H:02d}, "
          f"London {cfg.LONDON_START_H:02d}-{cfg.LONDON_END_H:02d}")
    print("=" * 104)

    curves = {}
    for mode, name in (("fixed", f"FIXED TP ({cfg.RR}R)"),
                       ("asia", "ASIA TP (opposing liquidity)")):
        print(f"\n{'#'*104}\n#  {name}\n{'#'*104}")
        tdf, n_open = run(df, utc1_all, mode)
        if not len(tdf):
            print("  no trades"); continue
        eqa = report(tdf, f"{SYMBOL} {TF}")
        curves[name] = eqa
        if mode == "asia":
            rr = tdf["rr"]
            print(f"    achieved RR → min {rr.min():.2f} · median {rr.median():.2f} · "
                  f"mean {rr.mean():.2f} · max {rr.max():.2f}")
        if n_open:
            print(f"    ({n_open} trade(s) never resolved in the walk window)")
        per_year(tdf, name)
        tdf.to_csv(os.path.join(OUT, f"hd_eurusd_{TF}_{mode}_rr{cfg.RR}.csv"), index=False)

    BG, FG = "#0d1117", "#e6edf3"
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG); ax.set_facecolor(BG)
    for (name, eqa), col in zip(curves.items(), ["#58a6ff", "#3fb950"]):
        if eqa is not None:
            ax.plot(eqa, color=col, linewidth=1.2, label=name)
    ax.axhline(10000, color="#444c56", ls="--", lw=0.8)
    ax.set_yscale("symlog")
    ax.set_title(f"asia_bot · {SYMBOL} · HistData M1 2004-2024 · fixed {cfg.RR}R vs opposing-Asia TP",
                 color=FG, fontweight="bold")
    ax.set_xlabel("Trade #", color=FG); ax.set_ylabel("Balance (symlog)", color=FG)
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.legend(facecolor="#161b22", edgecolor="#444c56", labelcolor=FG)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "equity_hd_eurusd.png"), dpi=140, facecolor=BG)
    plt.close(fig)
    print(f"\nsaved → {OUT}/ (hd_eurusd_fixed.csv, hd_eurusd_asia.csv, equity_hd_eurusd.png)")


if __name__ == "__main__":
    main()
