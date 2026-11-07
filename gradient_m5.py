"""
The gradient strategy on M5 (intraday frequency test): same logic as the
deployed M15 bot (EMA 7/18 -> FVG retest -> RSI 45/55 -> DOL>=2R filter -> SL at
swing -> gradient exit), just run on 5-minute bars. Goal: does it keep PF > 3
while firing multiple times a day? Realistic per-instrument spread, $1k/$100k.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MetaTrader5 as mt5

import frequency_tune as fq
import test_cross_exit as tc
import test_gradient_exit as tg
import realistic_ema_dol_rsi as edr

OUT = os.path.join("strategies", "gradient_m5")


def fetch_m5(symbol):
    mt5.symbol_select(symbol, True)
    bars = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 99999)
    if bars is None or not len(bars):
        return None
    df = pd.DataFrame(bars)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.set_index("time")[["open", "high", "low", "close"]]


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    os.makedirs(OUT, exist_ok=True)

    allt = []; spans = []
    for s in fq.UNIVERSE:
        df = fetch_m5(s)
        if df is None or len(df) < 1000:
            print(f"  {s}: no M5"); continue
        spans.append((df.index[0], df.index[-1]))
        base = fq.precompute_base(s, df)
        allt.append(tg.run_gradient(s, base, (7, 18), fq.BASEP, grad_lb=1, profit_only=True))
    df = tc.combine(allt)
    cl = df[df.outcome != "open"]
    d0 = min(s[0] for s in spans); d1 = max(s[1] for s in spans)
    tdays = (d1 - d0).days * 5 / 7
    st = edr.stats(cl)
    w = cl.loc[cl.pnl_r > 0.01, "pnl_r"]

    print("=" * 92)
    print(f"GRADIENT STRATEGY on M5 · {len(spans)} instruments · realistic · $1k/$100k")
    print(f"  data {d0.date()} -> {d1.date()}  (~{tdays:.0f} trading days)")
    print("=" * 92)
    print(f"  Trades        : {st['n']}   ({st['n']/tdays:.1f}/day · {st['n']/(tdays/5):.1f}/week)")
    print(f"  Win rate      : {st['wr']:.1f}%")
    print(f"  Profit factor : {st['pf']:.2f}   {'✅ PF > 3' if st['pf'] >= 3 else '❌ below 3'}")
    print(f"  Total R       : {st['tot']:+.1f}R   (end ${st['bal']:,.0f} from $100k)")
    print(f"  Max drawdown  : {st['dd']:.1f}%   loss streak {st['ls']}")
    print(f"  Avg win / median win : +{w.mean():.2f}R / +{w.median():.2f}R   avg hold {cl.hold_h.mean():.1f}h")

    print("\n  Per instrument:")
    print(f"  {'sym':12} {'N':>4} {'WR':>6} {'PF':>5} {'totR':>8}")
    for s in fq.UNIVERSE:
        g = cl[cl.symbol == s]
        if len(g):
            s2 = edr.stats(g)
            print(f"  {s:12} {s2['n']:>4} {s2['wr']:>5.1f}% {s2['pf']:>4.2f} {s2['tot']:>+7.1f}R")

    cl.to_csv(os.path.join(OUT, "trades.csv"), index=False)
    bal = 100000.0; eq = [bal]
    for r in cl.sort_values("entry_time")["pnl_r"]:
        bal += 1000.0 * float(r); eq.append(bal)
    BG, FG = "#0d1117", "#e6edf3"
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(eq, color="#3fb950" if st['pf'] >= 1 else "#f85149", linewidth=1.3)
    ax.axhline(100000, color="#444c56", linestyle="--", linewidth=0.8)
    ax.set_title(f"Gradient on M5 · PF {st['pf']:.2f} · WR {st['wr']:.1f}% · "
                 f"{st['n']/tdays:.1f} trades/day · DD {st['dd']:.1f}% · end ${st['bal']:,.0f}",
                 color=FG, fontsize=12, fontweight="bold")
    ax.set_xlabel("Trade #", color=FG); ax.set_ylabel("Balance ($)", color=FG)
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "equity.png"), dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nSaved -> {OUT}/")
    mt5.shutdown()


if __name__ == "__main__":
    main()
