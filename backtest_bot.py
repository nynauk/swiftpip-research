"""
Backtest THE DEPLOYED BOT'S strategy — parameters pulled straight from
live/config.py so this is provably the same config the live bot runs:
  EMA cfg.EMA_FAST/EMA_SLOW · FVG cfg.FVG_LOOKUP_WINDOW · retest cfg.RETEST_WINDOW_EMA
  · SL swing±cfg.SL_BUFFER_FRAC×FVG + spread+1pip · DOL target in [MIN_RR,MAX_RR]
  · RSI cfg.RSI_LONG_MIN/RSI_SHORT_MAX · news cfg.NEWS_FILTER · gradient exit
  (lb cfg.GRAD_LOOKBACK, profit-only cfg.GRADIENT_PROFIT_ONLY) · cfg.RISK_PCT sizing.
Realistic per-instrument spread, M15, ~4.6yr. Overall + per-instrument +
year-by-year + monthly frequency + FTMO prop-firm sim + equity curve.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import MetaTrader5 as mt5

import main as bt
import frequency_tune as fq
import test_cross_exit as tc
import test_gradient_exit as tg
import realistic_ema_dol_rsi as edr
from live import config as cfg

OUT = os.path.join("strategies", "bot_strategy")
BG, FG = "#0d1117", "#e6edf3"


def load(s):
    p = os.path.join("data_cache", f"{s}_M15.pkl")
    try:
        return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
    except Exception:
        if os.path.exists(p): os.remove(p)
        return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    os.makedirs(OUT, exist_ok=True)
    edr.SL_SPREAD_PAD = True                       # spread+1pip stop = the honest, live-matching model

    # ── config pulled from the live bot ──
    ema = (cfg.EMA_FAST, cfg.EMA_SLOW)
    lb, po = cfg.GRAD_LOOKBACK, cfg.GRADIENT_PROFIT_ONLY
    p = dict(rsi_band=(cfg.RSI_LONG_MIN, 100, 0, cfg.RSI_SHORT_MAX),
             fvg_window=cfg.FVG_LOOKUP_WINDOW, retest_window=cfg.RETEST_WINDOW_EMA,
             sl_buf=cfg.SL_BUFFER_FRAC, min_rr=cfg.MIN_RR, max_rr=cfg.MAX_RR,
             news_on=cfg.NEWS_FILTER)
    syms = [s for s in cfg.SYMBOLS if s in fq.SPEC]
    risk_usd = edr.START_BAL * cfg.RISK_PCT / 100.0     # 1% of $100k = $1,000

    bases = {s: fq.precompute_base(s, load(s)) for s in syms}
    df = tc.combine([tg.run_gradient(s, bases[s], ema, p, grad_lb=lb, profit_only=po) for s in syms])
    cl = df[df.outcome != "open"]
    d0 = min(b["idx"][0] for b in bases.values()); d1 = max(b["idx"][-1] for b in bases.values())
    months = (d1 - d0).days / 30.44
    st = edr.stats(df)
    w = cl.loc[cl.pnl_r > 0.01, "pnl_r"]

    print("=" * 100)
    print("BACKTEST · THE DEPLOYED BOT STRATEGY (params from live/config.py)")
    print("=" * 100)
    print(f"  EMA {ema[0]}/{ema[1]} · FVG {p['fvg_window']} · retest {p['retest_window']} · "
          f"SL swing±{p['sl_buf']}×FVG +spread+1pip · DOL [{p['min_rr']},{p['max_rr']}]R")
    print(f"  RSI {cfg.RSI_LONG_MIN}/{cfg.RSI_SHORT_MAX} · news {p['news_on']} · gradient exit "
          f"lb={lb} profit_only={po} · risk {cfg.RISK_PCT}% (${risk_usd:,.0f}/$100k)")
    print(f"  {len(syms)} instruments · M15 · {d0.date()} → {d1.date()} ({months:.1f} months) · realistic spread")
    print("-" * 100)
    print(f"  Trades        : {st['n']}   ({st['n']/(months/12*52):.2f}/wk · {st['n']/months:.1f}/mo)")
    print(f"  Win rate      : {st['wr']:.1f}%")
    print(f"  Profit factor : {st['pf']:.2f}")
    print(f"  Total R       : {st['tot']:+.1f}R   →  end ${st['bal']:,.0f}  (+{st['bal']/edr.START_BAL-1:.0%})")
    print(f"  Max drawdown  : {st['dd']:.1f}%   loss streak {st['ls']}   win streak {st['ws']}")
    print(f"  Avg / median win : +{w.mean():.2f}R / +{w.median():.2f}R   avg hold {cl.hold_h.mean():.1f}h")

    print("\n  ── per instrument ──")
    print(f"  {'sym':12} {'N':>4} {'WR':>6} {'PF':>6} {'totR':>8} {'DD':>7} {'LS':>4}")
    for s in syms:
        g = df[df.symbol == s]
        if len(g):
            s2 = edr.stats(g)
            print(f"  {s:12} {s2['n']:>4} {s2['wr']:>5.1f}% {s2['pf']:>5.2f} "
                  f"{s2['tot']:>+7.1f}R {s2['dd']:>6.1f}% {s2['ls']:>4}")

    print("\n  ── year by year ──")
    df["yr"] = pd.to_datetime(df.entry_time).dt.year
    for yr in sorted(df.yr.unique()):
        s2 = edr.stats(df[df.yr == yr])
        if s2:
            print(f"    {yr}: N={s2['n']:>4}  WR {s2['wr']:>4.0f}%  PF {s2['pf']:>5.2f}  "
                  f"{s2['tot']:>+6.1f}R  DD {s2['dd']:>5.1f}%  LS{s2['ls']}")

    # ── FTMO prop-firm sim at the live risk % ──
    ps = edr.prop_sim(df, risk_usd=risk_usd)
    print("\n  ── FTMO prop-firm sim ($100k · +10% payout&reset · -5% daily · -10% overall · "
          f"${risk_usd:,.0f}/trade · ${edr.CHALLENGE_FEE:.0f}/blowup) ──")
    print(f"    Payouts {ps['passes']}  ·  Banked ${ps['banked']:,.0f}  ·  "
          f"Blown {ps['blown']} ({ps['blown_daily']} daily / {ps['blown_overall']} overall)  ·  "
          f"Fees ${ps['fees']:,.0f}  ·  NET ${ps['net']:,.0f}")

    df.drop(columns="yr").to_csv(os.path.join(OUT, "bot_trades.csv"), index=False)

    # ── equity curve ──
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(st["times"], st["eqa"], color="#3fb950", linewidth=1.6, drawstyle="steps-post")
    ax.axhline(edr.START_BAL, color="#444c56", linestyle="--", linewidth=0.8)
    ax.axhline(edr.START_BAL * 0.90, color="#f85149", linestyle=":", linewidth=1.0, label="FTMO -10% floor")
    ax.set_title(f"Deployed bot strategy · EMA {ema[0]}/{ema[1]} gradient · buffered · "
                 f"PF {st['pf']:.2f} · WR {st['wr']:.1f}% · DD {st['dd']:.1f}% · "
                 f"{cfg.RISK_PCT:.0f}% risk · end ${st['bal']:,.0f}",
                 color=FG, fontsize=12, fontweight="bold")
    ax.set_xlabel("Date", color=FG); ax.set_ylabel("Balance ($)", color=FG)
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "bot_equity.png"), dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nSaved → {OUT}/  (bot_equity.png, bot_trades.csv)")
    mt5.shutdown()


if __name__ == "__main__":
    main()
