"""
Timeframe sweep of the champion gradient strategy (EMA 7/18, gradient exit lb1
profit-only, DOL>=2R, RSI 45/55, news ON, spread+1pip buffer, 17 instruments,
flat $1k/$100k). Runs it on M1, M3, M5 (each = MT5's max 99,999 bars) and the
deployed M15 (full history) and compares. READ-ONLY — does not touch the live bot.

Note: MT5 caps history at 99,999 bars per TF, so faster TFs cover SHORTER windows
(M1 ~2-3 months, M3 ~6-7 months, M5 ~1.5yr, M15 ~4.6yr). Smaller/newer samples on
the fast TFs — treat them as indicative, not as robust as M15.
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
OUT = os.path.join("strategies", "timeframe_compare")
BG, FG = "#0d1117", "#e6edf3"
COLS = {"M1": "#f85149", "M3": "#d29922", "M5": "#58a6ff", "M15": "#3fb950"}


def fetch(symbol, tf_const, n=99999):
    mt5.symbol_select(symbol, True)
    bars = mt5.copy_rates_from_pos(symbol, tf_const, 0, n)
    if bars is None or not len(bars):
        return None
    df = pd.DataFrame(bars)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.set_index("time")[["open", "high", "low", "close"]]


def load_m15_full(s):
    p = os.path.join("data_cache", f"{s}_M15.pkl")
    try:
        return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
    except Exception:
        if os.path.exists(p): os.remove(p)
        return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")


def run_tf(name, loader):
    """loader(symbol) -> M15-style OHLC df for this timeframe."""
    allt, spans = [], []
    for s in fq.UNIVERSE:
        df = loader(s)
        if df is None or len(df) < 1000:
            continue
        spans.append((df.index[0], df.index[-1]))
        base = fq.precompute_base(s, df)
        allt.append(tg.run_gradient(s, base, EMA, fq.BASEP, grad_lb=1, profit_only=True))
    df = tc.combine(allt)
    d0 = min(s[0] for s in spans); d1 = max(s[1] for s in spans)
    tdays = max((d1 - d0).days * 5 / 7, 1)
    st = edr.stats(df)
    cl = df[df.outcome != "open"]
    w = cl.loc[cl.pnl_r > 0.01, "pnl_r"]
    return dict(name=name, df=df, st=st, d0=d0, d1=d1, tdays=tdays,
                perday=st["n"] / tdays, perwk=st["n"] / (tdays / 5),
                avgw=w.mean() if len(w) else 0, medw=w.median() if len(w) else 0,
                hold=cl.hold_h.mean())


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    os.makedirs(OUT, exist_ok=True)
    edr.SL_SPREAD_PAD = True

    runs = []
    runs.append(run_tf("M1", lambda s: fetch(s, mt5.TIMEFRAME_M1)))
    runs.append(run_tf("M3", lambda s: fetch(s, mt5.TIMEFRAME_M3)))
    runs.append(run_tf("M5", lambda s: fetch(s, mt5.TIMEFRAME_M5)))
    runs.append(run_tf("M15", load_m15_full))

    print("=" * 118)
    print("TIMEFRAME COMPARISON · champion gradient strategy · buffered (spread+1pip) · "
          "17 instruments · flat $1k/$100k")
    print("=" * 118)
    print(f"  {'TF':4} {'window':25} {'days':>5} {'N':>5} {'/day':>6} {'/wk':>6} {'WR':>6} "
          f"{'PF':>6} {'totR':>8} {'DD':>7} {'LS':>4} {'avgW/med':>10} {'hold':>6} {'End$':>11}")
    print("-" * 118)
    for r in runs:
        st = r["st"]
        print(f"  {r['name']:4} {str(r['d0'].date())+'→'+str(r['d1'].date()):25} {r['tdays']:>5.0f} "
              f"{st['n']:>5} {r['perday']:>6.2f} {r['perwk']:>6.1f} {st['wr']:>5.1f}% {st['pf']:>6.2f} "
              f"{st['tot']:>+8.1f} {st['dd']:>6.1f}% {st['ls']:>4} "
              f"+{r['avgw']:>3.2f}/{r['medw']:>3.2f} {r['hold']:>5.1f}h ${st['bal']:>10,.0f}")

    # per-instrument PF for each TF (indices are the engine — see how it holds up)
    print("\n  ── profit factor by instrument across timeframes ──")
    print(f"  {'symbol':12} {'M1':>7} {'M3':>7} {'M5':>7} {'M15':>7}")
    for s in fq.UNIVERSE:
        cells = []
        for r in runs:
            g = r["df"][r["df"].symbol == s]
            cells.append(f"{edr.stats(g)['pf']:>7.2f}" if len(g) else f"{'—':>7}")
        print(f"  {s:12} " + " ".join(cells))

    for r in runs:
        r["df"].drop(columns=[c for c in ("yr",) if c in r["df"].columns], errors="ignore") \
            .to_csv(os.path.join(OUT, f"trades_{r['name']}.csv"), index=False)

    # ── equity curves, one panel per TF (vs trade #, since windows differ) ──
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor=BG)
    for ax, r in zip(axes.flat, runs):
        st = r["st"]
        bal = 100000.0; eq = [bal]
        for pr in r["df"].sort_values("entry_time")["pnl_r"]:
            bal += 1000.0 * float(pr); eq.append(bal)
        ax.plot(eq, color=COLS[r["name"]], linewidth=1.5)
        ax.axhline(100000, color="#444c56", linestyle="--", linewidth=0.8)
        ax.set_title(f"{r['name']} · {r['d0'].date()}→{r['d1'].date()} · N={st['n']} "
                     f"({r['perday']:.1f}/day) · PF {st['pf']:.2f} · WR {st['wr']:.1f}% · "
                     f"DD {st['dd']:.1f}% · ${st['bal']:,.0f}",
                     color=FG, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Trade #", color=FG); ax.set_ylabel("Balance ($)", color=FG)
        ax.set_facecolor(BG); ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    fig.suptitle("Gradient strategy across timeframes · buffered · flat $1k/$100k",
                 color=FG, fontsize=13, fontweight="bold")
    fig.patch.set_facecolor(BG)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(OUT, "timeframe_equity.png"), dpi=135, facecolor=BG)
    plt.close(fig)
    print(f"\nSaved → {OUT}/  (timeframe_equity.png + per-TF trade CSVs)")
    mt5.shutdown()


if __name__ == "__main__":
    main()
