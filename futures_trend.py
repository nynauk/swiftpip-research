"""
FUTURES TREND-FOLLOWING SYSTEM — the one real edge, built properly.

Design (all causal, standard managed-futures construction):
  · Signal  : ENSEMBLE of Donchian breakout states at 4 lookbacks (40/80/160/320d).
              State = +1 after an up-breakout, -1 after a down-breakout (carried).
              Composite = mean of the four → continuous position in [-1, +1].
  · Sizing  : VOLATILITY TARGETING — each market scaled to TARGET_VOL annualised
              (position ∝ 1/realized-vol), so a quiet bond and a wild NatGas
              contribute equal risk. Leverage capped.
  · Portfolio: equal-weight the active markets each day. Costs on turnover.
  · Metrics : CAGR, annual vol, Sharpe, MaxDD, per-era — portfolio lens.
Robustness: each single lookback reported too (a real edge is a plateau, not a spike).
Real CME futures daily, 2000-2026.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import frequency_tune as fq
import futures_test as ft            # FUT list + cached daily loader + spec injection

LOOKBACKS = [120, 240, 480]         # slow-trend ensemble (robustly best; fast lookbacks are noise)
TARGET_VOL = 0.20                    # annualised per-market vol target
VOL_WIN = 60                         # realized-vol lookback (days)
LEV_CAP = 4.0
OUT = os.path.join("strategies", "futures_trend")
BG, FG = "#0d1117", "#e6edf3"
ERAS = [("2000-2007", "2000-01-01", "2008-01-01"),
        ("2008-2011", "2008-01-01", "2012-01-01"),
        ("2012-2018", "2012-01-01", "2019-01-01"),
        ("2019-2026", "2019-01-01", "2027-01-01")]


def don_state(df, L):
    h, l, c = df.high.values, df.low.values, df.close.values
    rh = pd.Series(h).rolling(L).max().shift(1).values
    rl = pd.Series(l).rolling(L).min().shift(1).values
    st = np.zeros(len(df)); s = 0.0
    for i in range(len(df)):
        if not np.isnan(rh[i]):
            if c[i] > rh[i]: s = 1.0
            elif c[i] < rl[i]: s = -1.0
        st[i] = s
    return st


def inst_returns(name, df, lookbacks):
    """Vol-targeted daily strategy returns for one market (causal)."""
    spread = fq.SPEC[name][0]
    ret = df.close.pct_change().values
    rv = (pd.Series(ret).rolling(VOL_WIN).std().shift(1).values) * np.sqrt(252)
    scale = np.clip(np.where(rv > 0, TARGET_VOL / rv, 0.0), 0, LEV_CAP)
    comp = np.mean([don_state(df, L) for L in lookbacks], axis=0)   # -1..1
    pos = pd.Series(comp).shift(1).values                          # trade next day (causal)
    r = np.nan_to_num(pos) * np.nan_to_num(scale) * np.nan_to_num(ret)
    # cost on turnover
    turn = np.abs(np.diff(np.concatenate([[0], np.nan_to_num(pos)])))
    cost = turn * np.nan_to_num(scale) * (2 * spread / df.close.values)
    return pd.Series(r - cost, index=df.index)


def metrics(port):
    p = port.dropna()
    if len(p) < 100:
        return None
    eq = (1 + p).cumprod()
    yrs = (p.index[-1] - p.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = p.std() * np.sqrt(252)
    sharpe = p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else 0
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, dd=dd, eq=eq, n=len(p), yrs=yrs)


def main():
    os.makedirs(OUT, exist_ok=True)
    dd = {}
    for y, nm, *_ in ft.FUT:
        d = ft.daily(y, nm)
        if d is not None and len(d) > 400:
            dd[nm] = d
    print(f"markets: {len(dd)}  ({', '.join(dd)})", flush=True)

    # ── robustness: single-lookback portfolios ──
    print("\n" + "=" * 92)
    print("ROBUSTNESS · single-lookback trend portfolios (vol-targeted, all markets)")
    print("=" * 92)
    print(f"  {'lookback':>10} {'Sharpe':>8} {'CAGR':>8} {'vol':>7} {'MaxDD':>8}")
    for L in LOOKBACKS + [20, 240]:
        R = pd.DataFrame({nm: inst_returns(nm, df, [L]) for nm, df in dd.items()})
        m = metrics(R.mean(axis=1))
        if m:
            print(f"  {L:>10} {m['sharpe']:>8.2f} {m['cagr']:>7.1%} {m['vol']:>6.1%} {m['dd']:>8.1%}", flush=True)

    # ── the deployed ensemble ──
    R = pd.DataFrame({nm: inst_returns(nm, df, LOOKBACKS) for nm, df in dd.items()})
    port = R.mean(axis=1)
    m = metrics(port)
    print("\n" + "=" * 92)
    print(f"ENSEMBLE (lookbacks {LOOKBACKS}) · vol-target {TARGET_VOL:.0%}/market · {len(dd)} markets")
    print("=" * 92)
    print(f"  Sharpe {m['sharpe']:.2f} · CAGR {m['cagr']:.1%} · vol {m['vol']:.1%} · "
          f"MaxDD {m['dd']:.1%} · {m['yrs']:.0f}yr", flush=True)
    print("\n  per era:")
    for name, a, b in ERAS:
        me = metrics(port[(port.index >= a) & (port.index < b)])
        if me:
            print(f"    {name}: Sharpe {me['sharpe']:>5.2f} · CAGR {me['cagr']:>6.1%} · "
                  f"MaxDD {me['dd']:>6.1%}", flush=True)

    # per-market Sharpe (which markets carry it)
    print("\n  per-market Sharpe:")
    ms = sorted([(nm, metrics(R[nm])) for nm in dd], key=lambda x: -(x[1]["sharpe"] if x[1] else -9))
    print("    " + " · ".join(f"{nm} {mm['sharpe']:.2f}" for nm, mm in ms if mm), flush=True)

    # ── FTMO-sizing view: scale the whole book so MaxDD ~ -8% (buffer under -10%) ──
    k = 0.08 / abs(m["dd"])
    print(f"\n  FTMO-sized (scale ×{k:.2f} → MaxDD ≈ -8%): "
          f"CAGR {m['cagr']*k:.1%} · vol {m['vol']*k:.1%} · same Sharpe {m['sharpe']:.2f}", flush=True)

    port.to_frame("ret").to_csv(os.path.join(OUT, "portfolio_returns.csv"))

    # ── equity chart ──
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8), facecolor=BG, sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    eq = m["eq"]
    a1.plot(eq.index, eq.values, color="#3fb950", linewidth=1.3)
    a1.set_yscale("log")
    a1.set_title(f"Futures trend ensemble · Sharpe {m['sharpe']:.2f} · CAGR {m['cagr']:.1%} · "
                 f"MaxDD {m['dd']:.1%} · {TARGET_VOL:.0%} vol-target/market",
                 color=FG, fontsize=12, fontweight="bold")
    a1.set_ylabel("Growth of $1 (log)", color=FG)
    uw = (eq - eq.cummax()) / eq.cummax() * 100
    a2.fill_between(uw.index, uw.values, 0, color="#f85149", alpha=0.5)
    a2.set_ylabel("Drawdown %", color=FG); a2.set_xlabel("Year", color=FG)
    a2.set_title("Underwater", color=FG, fontsize=10, fontweight="bold")
    for ax in (a1, a2):
        ax.set_facecolor(BG); ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
        ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "futures_trend_equity.png"), dpi=140, facecolor=BG)
    plt.close(fig)
    print(f"\nSaved → {OUT}/  (futures_trend_equity.png, portfolio_returns.csv)", flush=True)


if __name__ == "__main__":
    main()
