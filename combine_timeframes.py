"""
Combine the gradient strategy's M3 + M5 + M15 results into ONE portfolio, using
the 15-year HistData out-of-sample trades (2010-2024, same span for all three).
Each timeframe = a separate engine, flat $1,000 risk/trade on one $100k account.
Reports each engine standalone, the combined portfolio, and the pairwise daily-
return correlation (does stacking timeframes diversify or just pile on risk?).
(M1 excluded — it's a net loser over 15yr, PF 0.97.)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import realistic_ema_dol_rsi as edr

DIR = os.path.join("strategies", "histdata_15yr")
OUT = os.path.join("strategies", "combined_tf")
RISK = 1000.0
BG, FG = "#0d1117", "#e6edf3"
COL = {"M3": "#d29922", "M5": "#58a6ff", "M15": "#3fb950", "COMBINED": "#ff7b72"}


def load(tf):
    df = pd.read_csv(os.path.join(DIR, f"trades_{tf}.csv"))
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["tf"] = tf
    return df


def daily(df):
    d = df[df.outcome != "open"].copy()
    d["day"] = d["exit_time"].dt.normalize()
    return d.groupby("day")["pnl_r"].sum() * RISK


def dd_of(cum):
    eq = 100000 + cum.cumsum(); peak = eq.cummax()
    return ((eq - peak) / peak * 100).min()


def main():
    os.makedirs(OUT, exist_ok=True)
    m3, m5, m15 = load("M3"), load("M5"), load("M15")
    comb = pd.concat([m3, m5, m15]).sort_values("entry_time").reset_index(drop=True)

    print("=" * 100)
    print("GRADIENT · M3 + M5 + M15 COMBINED PORTFOLIO · HistData 15yr (2010-2024) · $1k/$100k each")
    print("=" * 100)
    print(f"  {'engine':14} {'N':>5} {'WR':>7} {'PF':>6} {'totR':>8} {'DD':>7} {'LS':>4} "
          f"{'avgW/med':>10} {'End$':>11}")
    print("-" * 100)
    for name, df in [("M3", m3), ("M5", m5), ("M15", m15), ("COMBINED", comb)]:
        st = edr.stats(df)
        cl = df[df.outcome != "open"]; w = cl.loc[cl.pnl_r > 0.01, "pnl_r"]
        print(f"  {name:14} {st['n']:>5} {st['wr']:>6.1f}% {st['pf']:>6.2f} {st['tot']:>+8.1f} "
              f"{st['dd']:>6.1f}% {st['ls']:>4} +{w.mean():>3.2f}/{w.median():>3.2f} "
              f"${st['bal']:>10,.0f}")

    # ── diversification: pairwise daily-return correlation ──
    d3, d5, d15 = daily(m3), daily(m5), daily(m15)
    alld = pd.date_range(min(d3.index.min(), d5.index.min(), d15.index.min()),
                         max(d3.index.max(), d5.index.max(), d15.index.max()))
    a3 = d3.reindex(alld, fill_value=0.0); a5 = d5.reindex(alld, fill_value=0.0)
    a15 = d15.reindex(alld, fill_value=0.0)
    active = (a3 != 0) | (a5 != 0) | (a15 != 0)

    def corr(x, y):
        m = active & ((x != 0) | (y != 0))
        return np.corrcoef(x[m], y[m])[0, 1]

    combd = a3 + a5 + a15
    print("\n  ── diversification (daily $ P&L, active days) ──")
    print(f"  corr  M3–M5  {corr(a3,a5):+.3f}   M3–M15 {corr(a3,a15):+.3f}   "
          f"M5–M15 {corr(a5,a15):+.3f}")
    print(f"  daily vol   M3 ${a3[active].std():,.0f}  M5 ${a5[active].std():,.0f}  "
          f"M15 ${a15[active].std():,.0f}")
    print(f"  COMBINED daily vol ${combd[active].std():,.0f}  "
          f"(sum-of-parts would be ${a3[active].std()+a5[active].std()+a15[active].std():,.0f})")
    print(f"  DD (realized, exit-ordered):  M3 {dd_of(a3):.1f}%  M5 {dd_of(a5):.1f}%  "
          f"M15 {dd_of(a15):.1f}%  →  COMBINED {dd_of(combd):.1f}%")
    print(f"  Combined net: ${combd.sum():,.0f}  (M3 ${a3.sum():,.0f} + M5 ${a5.sum():,.0f} "
          f"+ M15 ${a15.sum():,.0f})")

    comb.to_csv(os.path.join(OUT, "combined_tf_trades.csv"), index=False)

    # ── charts: equity (entry-ordered) + combined realized equity over time ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), facecolor=BG,
                                   gridspec_kw={"height_ratios": [1.4, 1]})
    for name, df in [("M3", m3), ("M5", m5), ("M15", m15), ("COMBINED", comb)]:
        st = edr.stats(df)
        ax1.plot(st["times"], st["eqa"], color=COL[name],
                 linewidth=2.0 if name == "COMBINED" else 1.3, drawstyle="steps-post",
                 label=f"{name}: N={st['n']}, PF {st['pf']:.2f}, DD {st['dd']:.1f}%, ${st['bal']:,.0f}")
    ax1.axhline(100000, color="#444c56", linestyle="--", linewidth=0.8)
    ax1.set_title("Gradient · M3+M5+M15 combined portfolio · HistData 15yr · $1k/$100k each",
                  color=FG, fontsize=12, fontweight="bold")
    ax1.set_facecolor(BG); ax1.tick_params(colors=FG); ax1.grid(True, alpha=0.1, color=FG)
    ax1.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
    ax1.set_ylabel("Balance ($)", color=FG)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for sp in ax1.spines.values(): sp.set_edgecolor("#30363d")

    ceq = 100000 + combd.cumsum()
    ax2.plot(ceq.index, ceq.values, color=COL["COMBINED"], linewidth=1.3)
    ax2.axhline(100000, color="#444c56", linestyle="--", linewidth=0.8)
    ax2.set_title(f"Combined realized equity over time · DD {dd_of(combd):.1f}% · "
                  f"end ${ceq.values[-1]:,.0f}", color=FG, fontsize=11, fontweight="bold")
    ax2.set_facecolor(BG); ax2.tick_params(colors=FG); ax2.grid(True, alpha=0.1, color=FG)
    ax2.set_xlabel("Date", color=FG); ax2.set_ylabel("Balance ($)", color=FG)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for sp in ax2.spines.values(): sp.set_edgecolor("#30363d")
    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "combined_tf_equity.png"), dpi=140, facecolor=BG)
    plt.close(fig)
    print(f"\nSaved → {OUT}/  (combined_tf_equity.png, combined_tf_trades.csv)")


if __name__ == "__main__":
    main()
