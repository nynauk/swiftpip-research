"""
DEEP out-of-sample validation on HistData.com (5 years, 2020-2024).
Runs the champion gradient (EMA 7/18, buffered spread+1pip, DOL>=2R, RSI 45/55,
news ON) on M5 and M15 resampled from HistData M1, for FX + index engines, then
compares to the MT5-feed results. 5 years spans COVID '20, bull '21, bear '22,
recovery '23-24 — if the edge holds here it isn't a feed/regime curve-fit.
"""
import os
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
YEARS = [2020, 2021, 2022, 2023, 2024]
SYMS = ["EURUSD", "USDJPY", "GBPUSD", "US500.cash", "US100.cash", "GER40.cash", "UK100.cash"]
OUT = os.path.join("strategies", "histdata_validation")
BG, FG = "#0d1117", "#e6edf3"


def summarize(name, df, days):
    st = edr.stats(df)
    if st is None:
        return None
    cl = df[df.outcome != "open"]; w = cl.loc[cl.pnl_r > 0.01, "pnl_r"]
    return dict(name=name, st=st, perday=st["n"] / days, perwk=st["n"] / (days / 5),
                avgw=w.mean() if len(w) else 0, medw=w.median() if len(w) else 0,
                hold=cl.hold_h.mean(), df=df)


def main():
    os.makedirs(OUT, exist_ok=True)
    edr.SL_SPREAD_PAD = True
    print("=" * 112)
    print(f"HISTDATA OUT-OF-SAMPLE · {YEARS[0]}-{YEARS[-1]} · gradient · buffered · $1k/$100k · "
          f"{len(SYMS)} instruments")
    print("=" * 112)

    m5_all, m15_all, span0, span1 = [], [], None, None
    for s in SYMS:
        try:
            m5 = hd.fetch(s, YEARS, tf="5min")
        except Exception as e:
            print(f"  {s:12} download FAILED: {e!r}"); continue
        if len(m5) < 2000:
            print(f"  {s:12} too few bars"); continue
        span0 = m5.index[0] if span0 is None else min(span0, m5.index[0])
        span1 = m5.index[-1] if span1 is None else max(span1, m5.index[-1])
        m15 = m5.resample("15min").agg({"open": "first", "high": "max",
                                        "low": "min", "close": "last"}).dropna()
        b5 = fq.precompute_base(s, m5); b15 = fq.precompute_base(s, m15)
        t5 = tg.run_gradient(s, b5, EMA, fq.BASEP, 1, True)
        t15 = tg.run_gradient(s, b15, EMA, fq.BASEP, 1, True)
        if len(t5): m5_all.append(t5)
        if len(t15): m15_all.append(t15)
        print(f"    {s:12} → M5 {len(t5)} trades · M15 {len(t15)} trades", flush=True)

    d5 = tc.combine(m5_all); d15 = tc.combine(m15_all)
    days = (span1 - span0).days * 5 / 7
    r5 = summarize("HISTDATA M5", d5, days); r15 = summarize("HISTDATA M15", d15, days)

    print("\n" + "=" * 112)
    print(f"  span {span0.date()} → {span1.date()}  (~{days:.0f} trading days)")
    print(f"  {'stream':22} {'N':>5} {'/day':>6} {'/wk':>6} {'WR':>7} {'PF':>6} {'totR':>8} "
          f"{'DD':>7} {'LS':>4} {'avgW/med':>10} {'End$':>11}")
    print("-" * 112)
    for r in (r5, r15):
        if r is None:
            continue
        st = r["st"]
        print(f"  {r['name']:22} {st['n']:>5} {r['perday']:>6.2f} {r['perwk']:>6.1f} "
              f"{st['wr']:>6.1f}% {st['pf']:>6.2f} {st['tot']:>+8.1f} {st['dd']:>6.1f}% {st['ls']:>4} "
              f"+{r['avgw']:>3.2f}/{r['medw']:>3.2f} ${st['bal']:>10,.0f}")
    print("\n  reference (MT5 feed, buffered, same config):")
    print(f"  {'MT5 M5  (24-12→26-07)':22} {'204':>5} {'0.50':>6} {'2.5':>6} {'77.5%':>7} "
          f"{'6.86':>6} {'+131.3':>8} {'-1.9%':>7} {'3':>4} {'+0.97/0.39':>10} ${'231,319':>10}")
    print(f"  {'MT5 M15 (21-11→26-07)':22} {'278':>5} {'0.23':>6} {'1.1':>6} {'76.3%':>7} "
          f"{'5.50':>6} {'+218.9':>8} {'-2.8%':>7} {'4':>4} {'+1.26/0.48':>10} ${'318,931':>10}")

    print("\n  ── HISTDATA M5 per instrument ──")
    for s in SYMS:
        g = d5[d5.symbol == s]
        if len(g):
            st = edr.stats(g)
            print(f"    {s:12} N={st['n']:>4} WR {st['wr']:>4.0f}% PF {st['pf']:>6.2f} "
                  f"{st['tot']:>+7.1f}R DD {st['dd']:>5.1f}%")

    # year-by-year (M5)
    print("\n  ── HISTDATA M5 year-by-year ──")
    d5["yr"] = pd.to_datetime(d5.entry_time).dt.year
    for yr in sorted(d5.yr.unique()):
        st = edr.stats(d5[d5.yr == yr])
        if st:
            print(f"    {yr}: N={st['n']:>4} WR {st['wr']:>4.0f}% PF {st['pf']:>6.2f} "
                  f"{st['tot']:>+6.1f}R DD {st['dd']:>5.1f}% LS{st['ls']}")

    d5.drop(columns="yr").to_csv(os.path.join(OUT, "hd_m5_trades.csv"), index=False)
    d15.to_csv(os.path.join(OUT, "hd_m15_trades.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=BG)
    for ax, r, col in [(axes[0], r5, "#58a6ff"), (axes[1], r15, "#3fb950")]:
        if r is None:
            continue
        st = r["st"]; bal = 100000.0; eq = [bal]
        for pr in r["df"].sort_values("entry_time")["pnl_r"]:
            bal += 1000.0 * float(pr); eq.append(bal)
        ax.plot(eq, color=col, linewidth=1.4)
        ax.axhline(100000, color="#444c56", linestyle="--", linewidth=0.8)
        ax.set_title(f"{r['name']} · {span0.date()}→{span1.date()} · N={st['n']} · PF {st['pf']:.2f} · "
                     f"WR {st['wr']:.1f}% · DD {st['dd']:.1f}% · ${st['bal']:,.0f}",
                     color=FG, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Trade #", color=FG); ax.set_ylabel("Balance ($)", color=FG)
        ax.set_facecolor(BG); ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    fig.suptitle("Out-of-sample on HistData (2020-2024, different feed + 5 regimes)",
                 color=FG, fontsize=13, fontweight="bold")
    fig.patch.set_facecolor(BG)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUT, "hd_equity.png"), dpi=135, facecolor=BG)
    plt.close(fig)
    print(f"\nSaved → {OUT}/  (hd_equity.png, hd_m5_trades.csv, hd_m15_trades.csv)")


if __name__ == "__main__":
    main()
