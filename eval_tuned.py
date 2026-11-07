"""Evaluate combined tuned configs vs base, with year-by-year robustness."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import MetaTrader5 as mt5

import main as bt
import realistic_ema_dol_rsi as edr
import finetune_ema_dol as ft

NO_METALS = [s for s in ft.SYMBOLS if s not in ("XAUUSD", "XAGUSD")]


def yr_table(df, label):
    print(f"\n  {label} — year by year:")
    print(f"  {'Year':6} {'N':>5} {'WR':>6} {'PF':>5} {'R':>8} {'DD':>7} {'LS':>4}")
    d = df.copy(); d["yr"] = pd.to_datetime(d["entry_time"]).dt.year
    for yr in sorted(d["yr"].unique()):
        st = edr.stats(d[d["yr"] == yr])
        if st:
            print(f"  {yr:6} {st['n']:>5} {st['wr']:>5.1f}% {st['pf']:>4.2f} "
                  f"{st['tot']:>+7.1f}R {st['dd']:>6.1f}% {st['ls']:>4}")


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    pcs = {}
    for s in ft.SYMBOLS:
        df = bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
        pcs[s] = edr.precompute(s, df)

    BASE = ft.BASE
    TUNED = dict(BASE); TUNED["fvg_window"] = 16; TUNED["min_rr"] = 2.5
    TUNED_R2 = dict(BASE); TUNED_R2["fvg_window"] = 16   # tighter FVG only, keep 2.0 floor

    configs = [
        ("BASE (fvg24, rr2.0, 16 instr)", BASE, ft.SYMBOLS),
        ("TUNED-A (fvg16, rr2.0, 16 instr)", TUNED_R2, ft.SYMBOLS),
        ("TUNED-B (fvg16, rr2.5, 16 instr)", TUNED, ft.SYMBOLS),
        ("TUNED-B no metals (14 instr)", TUNED, NO_METALS),
    ]

    print("=" * 100)
    print("COMBINED TUNED CONFIGS · realistic · trade-by-trade · flat $1,000 / $100k")
    print("=" * 100)
    print(f"  {'Config':34} {'N':>5} {'WR':>6} {'PF':>5} {'R':>8} {'DD':>7} {'LS':>4} {'End$':>10}")
    results = {}
    for label, p, syms in configs:
        df = ft.combine([ft.run_ft(s, pcs[s], p) for s in syms])
        st = edr.stats(df)
        results[label] = (df, st)
        print(f"  {label:34} {st['n']:>5} {st['wr']:>5.1f}% {st['pf']:>4.2f} "
              f"{st['tot']:>+7.1f}R {st['dd']:>6.1f}% {st['ls']:>4} ${st['bal']:>9,.0f}")

    # year-by-year for the winner (TUNED-B no metals)
    yr_table(results["TUNED-B no metals (14 instr)"][0], "TUNED-B no metals")

    # equity comparison chart
    OUT = os.path.join("strategies", "finetune_ema_dol"); os.makedirs(OUT, exist_ok=True)
    BG, FG = "#0d1117", "#e6edf3"
    cols = {"BASE (fvg24, rr2.0, 16 instr)": "#7d8590",
            "TUNED-A (fvg16, rr2.0, 16 instr)": "#58a6ff",
            "TUNED-B (fvg16, rr2.5, 16 instr)": "#d29922",
            "TUNED-B no metals (14 instr)": "#3fb950"}
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=BG); ax.set_facecolor(BG)
    for label, (_, st) in results.items():
        ax.plot(st["times"], st["eqa"], color=cols[label], linewidth=1.6, drawstyle="steps-post",
                label=f"{label}: PF {st['pf']:.2f}, DD {st['dd']:.1f}%, ${st['bal']:,.0f}")
    ax.axhline(edr.START_BAL, color="#444c56", linestyle="--", linewidth=0.8)
    ax.set_title("Fine-tuned EMA+DOL momentum · config comparison · realistic",
                 color=FG, fontsize=12, fontweight="bold")
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
    ax.set_xlabel("Date", color=FG); ax.set_ylabel("Balance ($)", color=FG)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "equity_tuned_compare.png"), dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    results["TUNED-B no metals (14 instr)"][0].to_csv(os.path.join(OUT, "trades_tuned_final.csv"), index=False)
    print(f"\nSaved -> {OUT}/")
    mt5.shutdown()


if __name__ == "__main__":
    main()
