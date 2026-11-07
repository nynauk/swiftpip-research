"""
Prop-firm sim comparison: DOL exit vs EMA cross-back exit (both on DOL>=2R
selected setups, EMA 7/18, 17 instruments). No break-evens, no partials —
each trade is one entry, one exit, full position. $100k account, +10% cash-out
+ reset, -5% daily (24h), -10% overall, $550 lost per blowup.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import MetaTrader5 as mt5

import main as bt
import realistic_ema_dol_rsi as edr
import frequency_tune as fq
import test_cross_exit as tc

EMA = (7, 18)
OUT = os.path.join("strategies", "ema718_final")


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    os.makedirs(OUT, exist_ok=True)
    bases = {}
    for s in fq.UNIVERSE:
        df = bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
        bases[s] = fq.precompute_base(s, df)

    dol_df = tc.combine([fq.run_freq(s, bases[s], EMA, fq.BASEP) for s in fq.UNIVERSE])
    x_df = tc.combine([tc.run_cross(s, bases[s], EMA, fq.BASEP, require_dol=True) for s in fq.UNIVERSE])

    print("=" * 104)
    print("PROP-FIRM SIM · $100k · +10% cash-out+reset · -5% daily · -10% overall · "
          "$1k/trade · $550/blowup · NO BE/partials")
    print("=" * 104)
    print(f"  {'Exit':16} {'Trades':>7} {'Payouts':>8} {'Banked $':>11} "
          f"{'Blown(d)':>9} {'Blown(a)':>9} {'BlownTot':>9} {'Fees $':>8} {'NET $':>11}")
    props = {}
    for label, df in [("DOL target", dol_df), ("EMA cross-back", x_df)]:
        ps = edr.prop_sim(df)
        props[label] = ps
        print(f"  {label:16} {len(df):>7} {ps['passes']:>8} ${ps['banked']:>10,.0f} "
              f"{ps['blown_daily']:>9} {ps['blown_overall']:>9} {ps['blown']:>9} "
              f"${ps['fees']:>7,.0f} ${ps['net']:>10,.0f}")

    d, x = props["DOL target"], props["EMA cross-back"]
    print(f"\n  DOL      : {d['passes']} payouts, {d['blown']} blown -> NET ${d['net']:,.0f}")
    print(f"  Cross-back: {x['passes']} payouts, {x['blown']} blown -> NET ${x['net']:,.0f}")
    win = "EMA cross-back" if x['net'] > d['net'] else "DOL target"
    print(f"  >> Higher NET in the funded model: {win}")

    # sawtooth comparison
    BG, FG = "#0d1117", "#e6edf3"
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), facecolor=BG, sharex=True)
    for ax, (label, ps, col) in zip(axes, [("DOL target", d, "#3fb950"),
                                            ("EMA cross-back", x, "#58a6ff")]):
        ax.set_facecolor(BG)
        ax.plot(ps["trace_t"], ps["trace_b"], color=col, linewidth=0.8)
        ax.axhline(100_000, color="#7d8590", linestyle="--", linewidth=0.6)
        ax.axhline(110_000, color="#3fb950", linestyle=":", linewidth=0.8)
        ax.axhline(90_000, color="#f85149", linestyle=":", linewidth=0.8)
        if ps["passes_t"]:
            ax.scatter([t for t, _ in ps["passes_t"]], [110_000] * len(ps["passes_t"]),
                       marker="^", s=22, color="#3fb950", zorder=5)
        if ps["blows_t"]:
            ax.scatter([t for t, _ in ps["blows_t"]], [90_000] * len(ps["blows_t"]),
                       marker="x", s=40, color="#f85149", zorder=5)
        ax.set_title(f"{label} · {ps['passes']} payouts · {ps['blown']} blown · "
                     f"banked ${ps['banked']:,.0f} · NET ${ps['net']:,.0f}",
                     color=FG, fontsize=11, fontweight="bold")
        ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
        ax.set_ylabel("Account $", color=FG)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    axes[1].set_xlabel("Date", color=FG)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "prop_dol_vs_cross.png"), dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nSaved -> {OUT}/prop_dol_vs_cross.png")
    mt5.shutdown()


if __name__ == "__main__":
    main()
