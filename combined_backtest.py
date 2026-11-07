"""
COMBINED backtest — run BOTH gradient engines as one portfolio:
  • M15 gradient  (the deployed bot)
  • M5  gradient  (the latest, higher-frequency engine)
Same champion config (EMA 7/18, lb1, profit-only), buffer ON (spread+1pip),
17 instruments, flat $1,000 risk/trade on a shared $100k account.

Reports each engine standalone, the combined portfolio over the FULL span and
over the OVERLAP window (where both actually run together), the daily-return
CORRELATION between the two engines (diversification), and combined drawdown.
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

EMA = (7, 18)
OUT = os.path.join("strategies", "combined")
BG, FG = "#0d1117", "#e6edf3"
C_M15, C_M5, C_COMB = "#58a6ff", "#d29922", "#3fb950"


def load_m15(s):
    p = os.path.join("data_cache", f"{s}_M15.pkl")
    try:
        return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
    except Exception:
        if os.path.exists(p): os.remove(p)
        return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")


def fetch_m5(symbol):
    mt5.symbol_select(symbol, True)
    bars = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 99999)
    if bars is None or not len(bars):
        return None
    df = pd.DataFrame(bars)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.set_index("time")[["open", "high", "low", "close"]]


def daily_pnl(df):
    d = df[df.outcome != "open"].copy()
    d["day"] = pd.to_datetime(d["exit_time"]).dt.normalize()
    return d.groupby("day")["pnl_r"].sum() * edr.RISK_USD


def dd_of(cum_series):
    eq = edr.START_BAL + cum_series.cumsum()
    peak = eq.cummax()
    return ((eq - peak) / peak * 100).min()


def summ(df):
    st = edr.stats(df)
    cl = df[df.outcome != "open"]
    wk = max((pd.to_datetime(cl.entry_time).max() - pd.to_datetime(cl.entry_time).min()).days / 7, 1)
    return st, len(cl) / wk


def prow(tag, df):
    st, perwk = summ(df)
    print(f"  {tag:26} {st['n']:>5} {perwk:>6.2f} {st['wr']:>6.1f}% {st['pf']:>5.2f} "
          f"{st['tot']:>+8.1f} {st['dd']:>7.1f}% {st['ls']:>4} ${st['bal']:>10,.0f}")
    return st


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    os.makedirs(OUT, exist_ok=True)
    edr.SL_SPREAD_PAD = True                     # buffered = honest numbers
    syms = fq.UNIVERSE

    # ── engines ──
    m15 = {s: load_m15(s) for s in syms}
    b15 = {s: fq.precompute_base(s, m15[s]) for s in syms}
    dfA = tc.combine([tg.run_gradient(s, b15[s], EMA, fq.BASEP, 1, True) for s in syms])
    dfA["engine"] = "M15"

    m5 = {}
    for s in syms:
        d = fetch_m5(s)
        if d is not None and len(d) >= 1000: m5[s] = d
    b5 = {s: fq.precompute_base(s, m5[s]) for s in m5}
    dfB = tc.combine([tg.run_gradient(s, b5[s], EMA, fq.BASEP, 1, True) for s in m5])
    dfB["engine"] = "M5"

    comb_full = tc.combine([dfA, dfB])

    # overlap window (both engines live)
    o0 = max(pd.to_datetime(dfA.entry_time).min(), pd.to_datetime(dfB.entry_time).min())
    o1 = min(pd.to_datetime(dfA.entry_time).max(), pd.to_datetime(dfB.entry_time).max())
    dfA_o = dfA[(pd.to_datetime(dfA.entry_time) >= o0) & (pd.to_datetime(dfA.entry_time) <= o1)]
    dfB_o = dfB[(pd.to_datetime(dfB.entry_time) >= o0) & (pd.to_datetime(dfB.entry_time) <= o1)]
    comb_o = tc.combine([dfA_o, dfB_o])

    hdr = (f"  {'stream':26} {'N':>5} {'/wk':>6} {'WR':>7} {'PF':>5} {'totR':>8} "
           f"{'DD':>8} {'LS':>4} {'End$':>11}")
    print("=" * 100)
    print("COMBINED PORTFOLIO · M15 gradient + M5 gradient · buffer ON · flat $1k/$100k · 17 instruments")
    print("=" * 100)
    print("\n  ── standalone (each over its own full span) ──")
    print(hdr)
    stA = prow("M15 gradient (deployed)", dfA)
    stB = prow("M5 gradient (latest)", dfB)
    print("\n  ── COMBINED, full span (M15 ~4.6yr; M5 only in last ~1.6yr) ──")
    print(hdr)
    stC = prow("combined (both, one acct)", comb_full)

    print(f"\n  ── OVERLAP window {o0.date()} → {o1.date()} (both engines live together) ──")
    print(hdr)
    prow("M15 only (overlap)", dfA_o)
    prow("M5 only (overlap)", dfB_o)
    stCO = prow("COMBINED (overlap)", comb_o)

    # ── correlation on the overlap ──
    da = daily_pnl(dfA_o); db = daily_pnl(dfB_o)
    alld = pd.date_range(min(da.index.min(), db.index.min()), max(da.index.max(), db.index.max()))
    a = da.reindex(alld, fill_value=0.0); b = db.reindex(alld, fill_value=0.0)
    both = (a != 0) | (b != 0)
    corr = np.corrcoef(a[both], b[both])[0, 1]
    comb_daily = (a + b)
    print("\n  ── diversification (overlap, daily $ P&L) ──")
    print(f"  Pearson corr M15 vs M5 (active days) : {corr:+.3f}")
    print(f"  M15  daily vol ${a[both].std():,.0f}   DD {dd_of(a):.1f}%")
    print(f"  M5   daily vol ${b[both].std():,.0f}   DD {dd_of(b):.1f}%")
    print(f"  COMBINED daily vol ${comb_daily[both].std():,.0f}   DD {dd_of(comb_daily):.1f}%   "
          f"(exit-ordered; sum-of-parts vol would be ${a[both].std()+b[both].std():,.0f})")
    print(f"  Combined net over overlap: ${comb_daily.sum():,.0f}  "
          f"(M15 ${a.sum():,.0f} + M5 ${b.sum():,.0f})")

    comb_full.to_csv(os.path.join(OUT, "combined_trades.csv"), index=False)

    # ── charts ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), facecolor=BG,
                                   gridspec_kw={"height_ratios": [1.7, 1]})
    # top: full-span equity, three streams
    for df, col, lab in [(dfA, C_M15, "M15 gradient"), (dfB, C_M5, "M5 gradient"),
                         (comb_full, C_COMB, "COMBINED")]:
        st = edr.stats(df)
        ax1.plot(st["times"], st["eqa"], color=col, linewidth=1.7 if lab == "COMBINED" else 1.3,
                 drawstyle="steps-post",
                 label=f"{lab}: N={st['n']}, PF {st['pf']:.2f}, DD {st['dd']:.1f}%, ${st['bal']:,.0f}")
    ax1.axvspan(o0, o1, color=FG, alpha=0.05, label=f"overlap (both live)")
    ax1.axhline(edr.START_BAL, color="#444c56", linestyle="--", linewidth=0.8)
    ax1.set_title("Combined portfolio equity · M15 + M5 gradient · flat $1k/$100k",
                  color=FG, fontsize=12, fontweight="bold")
    ax1.set_facecolor(BG); ax1.tick_params(colors=FG); ax1.grid(True, alpha=0.1, color=FG)
    ax1.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
    ax1.set_ylabel("Balance ($)", color=FG)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.xaxis.set_major_locator(mdates.YearLocator()); ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in ax1.spines.values(): sp.set_edgecolor("#30363d")
    # bottom: daily P&L scatter (overlap)
    ax2.scatter(b[both], a[both], s=12, color="#7d8590", alpha=0.55)
    ax2.axhline(0, color="#444c56", lw=0.6); ax2.axvline(0, color="#444c56", lw=0.6)
    ax2.set_title(f"Daily P&L: M5 vs M15 (overlap) · corr {corr:+.2f} · "
                  f"low correlation = the two engines smooth each other",
                  color=FG, fontsize=11, fontweight="bold")
    ax2.set_xlabel("M5 engine daily $", color=FG); ax2.set_ylabel("M15 engine daily $", color=FG)
    ax2.set_facecolor(BG); ax2.tick_params(colors=FG); ax2.grid(True, alpha=0.1, color=FG)
    for sp in ax2.spines.values(): sp.set_edgecolor("#30363d")
    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "combined_equity.png"), dpi=140, facecolor=BG)
    plt.close(fig)

    print(f"\nSaved -> {OUT}/  (combined_equity.png, combined_trades.csv)")
    mt5.shutdown()


if __name__ == "__main__":
    main()
