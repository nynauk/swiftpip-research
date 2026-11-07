"""
FULL 15-YEAR out-of-sample test on HistData (2010-2024), across M1/M3/M5/M15.
Fetches each instrument's cleaned M1 once, resamples to all four timeframes, runs
the champion gradient (buffered) on each, and compares the timeframes over 15
years + 5+ market regimes. FX goes back to 2010; indices to whenever HistData
has them (per-instrument span reported). READ-ONLY; does not touch the live bot.
"""
import os
import gc
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import frequency_tune as fq
import test_cross_exit as tc
import test_gradient_exit as tg
import realistic_ema_dol_rsi as edr
import histdata_fetch as hd

EMA = (7, 18)
YEARS = list(range(2010, 2025))            # 2010-2024 inclusive
SYMS = ["EURUSD", "USDJPY", "GBPUSD", "US500.cash", "US100.cash", "GER40.cash", "UK100.cash"]
TFS = [("M1", "1min"), ("M3", "3min"), ("M5", "5min"), ("M15", "15min")]
OHLC = {"open": "first", "high": "max", "low": "min", "close": "last"}
OUT = os.path.join("strategies", "histdata_15yr")
BG, FG = "#0d1117", "#e6edf3"
COLS = {"M1": "#f85149", "M3": "#d29922", "M5": "#58a6ff", "M15": "#3fb950"}


def main():
    os.makedirs(OUT, exist_ok=True)
    edr.SL_SPREAD_PAD = True
    print("=" * 116, flush=True)
    print(f"HISTDATA FULL 15-YEAR · {YEARS[0]}-{YEARS[-1]} · gradient · buffered · $1k/$100k · "
          f"M1/M3/M5/M15 · {len(SYMS)} instruments", flush=True)
    print("=" * 116, flush=True)

    tf_trades = {tf: [] for tf, _ in TFS}
    span0 = span1 = None
    for s in SYMS:
        try:
            m1 = hd.fetch(s, YEARS, tf="1min")
        except Exception as e:
            print(f"  {s:12} FAILED: {e!r}", flush=True); continue
        if len(m1) < 5000:
            print(f"  {s:12} too few bars", flush=True); continue
        span0 = m1.index[0] if span0 is None else min(span0, m1.index[0])
        span1 = m1.index[-1] if span1 is None else max(span1, m1.index[-1])
        row = []
        for tf, rule in TFS:
            bars = m1 if tf == "M1" else m1.resample(rule).agg(OHLC).dropna()
            base = fq.precompute_base(s, bars)
            t = tg.run_gradient(s, base, EMA, fq.BASEP, 1, True)
            if len(t):
                tf_trades[tf].append(t)
            row.append(f"{tf} {len(t)}")
            del bars, base
        print(f"    {s:12} {m1.index[0].date()}→{m1.index[-1].date()} · " + " · ".join(row), flush=True)
        del m1; gc.collect()

    days = (span1 - span0).days * 5 / 7
    print("\n" + "=" * 116, flush=True)
    print(f"  span {span0.date()} → {span1.date()}  (~{days:.0f} trading days · {days/252:.1f} yrs)", flush=True)
    print(f"  {'TF':4} {'N':>5} {'/day':>6} {'/wk':>6} {'WR':>7} {'PF':>6} {'totR':>9} "
          f"{'DD':>7} {'LS':>4} {'avgW/med':>10} {'End$':>11}", flush=True)
    print("-" * 116, flush=True)
    keep = {}
    for tf, _ in TFS:
        df = tc.combine(tf_trades[tf]) if tf_trades[tf] else pd.DataFrame()
        keep[tf] = df
        st = edr.stats(df)
        if st is None:
            print(f"  {tf:4} no trades", flush=True); continue
        cl = df[df.outcome != "open"]; w = cl.loc[cl.pnl_r > 0.01, "pnl_r"]
        print(f"  {tf:4} {st['n']:>5} {st['n']/days:>6.2f} {st['n']/(days/5):>6.1f} "
              f"{st['wr']:>6.1f}% {st['pf']:>6.2f} {st['tot']:>+9.1f} {st['dd']:>6.1f}% {st['ls']:>4} "
              f"+{w.mean():>3.2f}/{w.median():>3.2f} ${st['bal']:>10,.0f}", flush=True)

    # per-instrument (M5) + year-by-year (M5)
    d5 = keep["M5"]
    if len(d5):
        print("\n  ── M5 per instrument (15yr) ──", flush=True)
        for s in SYMS:
            g = d5[d5.symbol == s]
            if len(g):
                st = edr.stats(g)
                print(f"    {s:12} N={st['n']:>4} WR {st['wr']:>4.0f}% PF {st['pf']:>6.2f} "
                      f"{st['tot']:>+7.1f}R DD {st['dd']:>5.1f}%", flush=True)
        print("\n  ── M5 year-by-year (15yr) ──", flush=True)
        d5 = d5.copy(); d5["yr"] = pd.to_datetime(d5.entry_time).dt.year
        for yr in sorted(d5.yr.unique()):
            st = edr.stats(d5[d5.yr == yr])
            if st:
                print(f"    {yr}: N={st['n']:>4} WR {st['wr']:>4.0f}% PF {st['pf']:>6.2f} "
                      f"{st['tot']:>+6.1f}R DD {st['dd']:>5.1f}% LS{st['ls']}", flush=True)

    for tf, _ in TFS:
        if len(keep[tf]):
            keep[tf].to_csv(os.path.join(OUT, f"trades_{tf}.csv"), index=False)

    # equity 2x2
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor=BG)
    for ax, (tf, _) in zip(axes.flat, TFS):
        df = keep[tf]; st = edr.stats(df)
        if st is None:
            ax.set_visible(False); continue
        bal = 100000.0; eq = [bal]
        for pr in df.sort_values("entry_time")["pnl_r"]:
            bal += 1000.0 * float(pr); eq.append(bal)
        ax.plot(eq, color=COLS[tf], linewidth=1.4)
        ax.axhline(100000, color="#444c56", linestyle="--", linewidth=0.8)
        ax.set_title(f"{tf} · N={st['n']} · PF {st['pf']:.2f} · WR {st['wr']:.1f}% · "
                     f"DD {st['dd']:.1f}% · ${st['bal']:,.0f}", color=FG, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Trade #", color=FG); ax.set_ylabel("Balance ($)", color=FG)
        ax.set_facecolor(BG); ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    fig.suptitle(f"HistData 15-year out-of-sample ({span0.date()}→{span1.date()}) · gradient · buffered",
                 color=FG, fontsize=13, fontweight="bold")
    fig.patch.set_facecolor(BG)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(OUT, "equity_15yr.png"), dpi=135, facecolor=BG)
    plt.close(fig)
    print(f"\nSaved → {OUT}/  (equity_15yr.png + per-TF CSVs)", flush=True)


if __name__ == "__main__":
    main()
