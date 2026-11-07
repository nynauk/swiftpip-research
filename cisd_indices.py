"""
INDICES-ONLY 5m CISD (US100 + US500 + GER40) — the winning engine, validated.
  · full stats (recent 2019-2026): WR, PF, expectancy, equity, streaks, drawdown
  · FTMO harvester TWO ways:
       i.i.d. bootstrap (optimistic)  vs  SEQUENTIAL/chronological (preserves the
       real loss-streak CLUSTERING — the honest number)
  · out-of-sample: full-history 2010-2024 harvester (does the index edge predate
    the recent window?)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import MetaTrader5 as mt5

import realistic_ema_dol_rsi as edr
import all_pairs_2000_2026 as ap
from cisd_dev import run, degrade
from cisd_stats import streaks

edr.SL_SPREAD_PAD = True
IDX = ["US100.cash", "US500.cash", "GER40.cash"]
OUT = os.path.join("strategies", "cisd_dev")
BG, FG, GREEN, RED = "#0d1117", "#e6edf3", "#3fb950", "#f85149"
START, FEE, SPLIT, TPD, MAXT, MAXPAY = 10_000.0, 96.0, 0.80, 3, 1500, 30


def load5(sym, since):
    m5 = pd.read_pickle(os.path.join("ext_data", f"HD_{sym}_5min_2000-2024.pkl"))
    if since >= "2019-01-01":
        tail = ap.mt5_bars(ap.U[sym][1], mt5.TIMEFRAME_M5, n=99999)
        if tail is not None:
            m5 = pd.concat([m5[m5.index < "2025-01-01"], tail[tail.index >= "2025-01-01"]])
        return m5[m5.index >= since]
    return m5[(m5.index >= since) & (m5.index < "2025-01-01")]


def harvest_iid(R, risk, n=10000, seed=0):
    rng = np.random.default_rng(seed); R = np.asarray(R)

    def ph(target):
        bal = 0.0
        for _ in range(MAXT // TPD):
            day = 0.0
            for _ in range(TPD):
                x = R[rng.integers(len(R))] * risk; bal += x; day += x
                if bal <= -0.10: return "blow"
                if bal >= target: return "pass"
            if day <= -0.05: return "blow"
        return "timeout"
    nets = np.empty(n); fu = 0
    for a in range(n):
        net = -FEE
        if ph(0.10) == "pass" and ph(0.05) == "pass":
            fu += 1; first = True
            for _ in range(MAXPAY):
                if ph(0.10) == "pass":
                    net += SPLIT * 0.10 * START
                    if first: net += FEE; first = False
                else: break
        nets[a] = net
    return nets.mean(), fu / n, (nets > 0).mean()


def harvest_seq(R, risk, n=10000, seed=0):
    """Walk the REAL chronological trade order (circular) — preserves clustering."""
    rng = np.random.default_rng(seed); R = np.asarray(R); M = len(R)
    ptr = [0]

    def ph(target):
        bal = 0.0
        for _ in range(MAXT // TPD):
            day = 0.0
            for _ in range(TPD):
                x = R[ptr[0] % M] * risk; ptr[0] += 1; bal += x; day += x
                if bal <= -0.10: return "blow"
                if bal >= target: return "pass"
            if day <= -0.05: return "blow"
        return "timeout"
    nets = np.empty(n); fu = 0
    for a in range(n):
        ptr[0] = int(rng.integers(M))
        net = -FEE
        if ph(0.10) == "pass" and ph(0.05) == "pass":
            fu += 1; first = True
            for _ in range(MAXPAY):
                if ph(0.10) == "pass":
                    net += SPLIT * 0.10 * START
                    if first: net += FEE; first = False
                else: break
        nets[a] = net
    return nets.mean(), fu / n, (nets > 0).mean()


def stat_block(df):
    r = df.pnl_r.astype(float).values
    w = r[r > 0.01]; lo = r[r < -0.01]
    ls, ws = streaks(r)
    cum = np.cumsum(r); dd = (cum - np.maximum.accumulate(cum)).min()
    d = df.copy(); d["day"] = pd.to_datetime(d.entry_time).dt.normalize()
    daily = d.groupby("day").pnl_r.sum(); sh = daily.mean() / daily.std() * np.sqrt(252)
    return dict(n=len(r), wr=len(w) / len(r) * 100, pf=w.sum() / -lo.sum(), exp=r.mean(),
                tot=cum[-1], dd=dd, maxls=max(ls), sharpe=sh, cum=cum)


def main():
    os.makedirs(OUT, exist_ok=True)
    have = mt5.initialize()
    rec = {s: load5(s, "2019-01-01") for s in IDX}
    full = {s: load5(s, "2010-01-01") for s in IDX}
    if have:
        mt5.shutdown()

    raw = pd.concat([run(s, rec[s]) for s in IDX], ignore_index=True)
    raw["entry_time"] = pd.to_datetime(raw["entry_time"])
    df = degrade(raw.sort_values("entry_time")).reset_index(drop=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    st = stat_block(df); yrs = (df.entry_time.max() - df.entry_time.min()).days / 365.25

    print("=" * 84)
    print("INDICES-ONLY 5m CISD (US100 + US500 + GER40) · recent 2019-2026 · realistic")
    print("=" * 84)
    print(f"  Trades {st['n']:,} ({st['n']/yrs:.0f}/yr) · WR {st['wr']:.1f}% (BE 33%) · PF {st['pf']:.3f} · "
          f"expectancy {st['exp']:+.4f}R")
    print(f"  Total {st['tot']:+.0f}R · Sharpe {st['sharpe']:.2f} · max losing streak {st['maxls']} · "
          f"max DD {st['dd']:.0f}R")
    print("  per instrument:")
    for s in IDX:
        g = df[df.symbol == s]; s2 = edr.stats(g)
        print(f"    {s:12} N={s2['n']:>5} WR {s2['wr']:>4.0f}% PF {s2['pf']:>5.2f} {s2['tot']:>+6.0f}R")
    df["yr"] = df.entry_time.dt.year
    print("  per year PF: " + " · ".join(f"{y}:{edr.stats(df[df.yr==y])['pf']:.2f}" for y in sorted(df.yr.unique())))

    print("\n  ── FTMO HARVESTER (recent) ──")
    print(f"  {'risk':>6} {'  i.i.d. EV':>12} {'funded':>7}   {'SEQ (clustered) EV':>19} {'funded':>7} {'P(profit)':>10}")
    for risk in (0.005, 0.01, 0.015):
        ei, fi, _ = harvest_iid(df.pnl_r.values, risk)
        es, fs, ps = harvest_seq(df.pnl_r.values, risk)
        print(f"  {risk:>5.1%} ${ei:>+11.0f} {fi:>6.1%}   ${es:>+17.0f} {fs:>6.1%} {ps:>9.1%}"
              f"{'  <<< +EV both' if ei>0 and es>0 else ''}", flush=True)

    # ── OOS: full history 2010-2024 ──
    rawf = pd.concat([run(s, full[s]) for s in IDX], ignore_index=True)
    rawf["entry_time"] = pd.to_datetime(rawf["entry_time"])
    dff = degrade(rawf.sort_values("entry_time"))
    sf = stat_block(dff)
    print("\n  ── OUT-OF-SAMPLE · full history 2010-2024 ──")
    print(f"  N {sf['n']:,} · PF {sf['pf']:.3f} · expectancy {sf['exp']:+.4f}R · max DD {sf['dd']:.0f}R")
    ei, fi, _ = harvest_iid(dff.pnl_r.values, 0.01)
    es, fs, ps = harvest_seq(dff.pnl_r.values, 0.01)
    print(f"  harvester @1%: i.i.d. ${ei:+.0f} ({fi:.0%} funded) · SEQ ${es:+.0f} ({fs:.0%} funded)", flush=True)

    # ── equity chart ──
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8), facecolor=BG, sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(df.entry_time.values, st["cum"], color=GREEN, lw=1.3)
    a1.axhline(0, color="#444c56", ls="--", lw=0.7)
    a1.set_title(f"Indices-only 5m CISD · {st['tot']:+.0f}R · PF {st['pf']:.2f} · Sharpe {st['sharpe']:.2f} · "
                 f"maxLS {st['maxls']} · 2019-2026", color=FG, fontsize=12, fontweight="bold")
    a1.set_ylabel("Cumulative R", color=FG)
    ddser = st["cum"] - np.maximum.accumulate(st["cum"])
    a2.fill_between(df.entry_time.values, ddser, 0, color=RED, alpha=0.5)
    a2.set_title(f"Underwater (R) · max {st['dd']:.0f}R", color=FG, fontsize=10, fontweight="bold")
    a2.set_ylabel("Drawdown (R)", color=FG); a2.set_xlabel("Year", color=FG)
    for a in (a1, a2):
        a.set_facecolor(BG); a.tick_params(colors=FG); a.grid(True, alpha=0.1, color=FG)
        a.xaxis.set_major_locator(mdates.YearLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for sp in a.spines.values(): sp.set_edgecolor("#30363d")
    fig.patch.set_facecolor(BG); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "cisd_indices_equity.png"), dpi=140, facecolor=BG)
    plt.close(fig)
    print(f"\nSaved → {OUT}/cisd_indices_equity.png")


if __name__ == "__main__":
    main()
