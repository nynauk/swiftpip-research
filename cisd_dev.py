"""
5-minute CISD development pass (recent data 2019-2026):
  · deeper stats (expectancy, avg/med win-loss, streaks, Sharpe, per-instrument, per-year)
  · variations: base(close entry) · +MSS filter · FVG-retrace entry · +news-block · combo
  · FTMO harvester: feed the real CISD trade distribution → EV per challenge (0.5/1/2% risk)
  · plot one representative trade
Causal, per-instrument spread + degradation.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import MetaTrader5 as mt5

import frequency_tune as fq
import realistic_ema_dol_rsi as edr
import news_filter as nf
import all_pairs_2000_2026 as ap
from request1_htf import conf_swings

edr.SL_SPREAD_PAD = True
SYMS = ["EURUSD", "GBPUSD", "USDJPY", "US500.cash", "US100.cash", "GER40.cash", "UK100.cash"]
RR = 2.0; TRIG = 60; HOLD = 96; RETR = 24
OUT = os.path.join("strategies", "cisd_dev")
BG, FG, GREEN, RED, BLUE, AMBER = "#0d1117", "#e6edf3", "#3fb950", "#f85149", "#58a6ff", "#d29922"
START = 10_000.0; FEE = 96.0; SPLIT = 0.80; TPD = 3; MAXT = 1500; MAXPAY = 30


def load5(sym):
    m5 = pd.read_pickle(os.path.join("ext_data", f"HD_{sym}_5min_2000-2024.pkl"))
    tail = ap.mt5_bars(ap.U[sym][1], mt5.TIMEFRAME_M5, n=99999)
    if tail is not None:
        m5 = pd.concat([m5[m5.index < "2025-01-01"], tail[tail.index >= "2025-01-01"]])
    return m5[m5.index >= "2019-01-01"]


def run(sym, bars, entry="close", mss=False, news_block=False, news_set=None, rec_geo=False):
    spread, slip, _ = fq.SPEC[sym]; pad = edr.sl_pad(sym, spread)
    o, h, l, c = bars.open.values, bars.high.values, bars.low.values, bars.close.values
    idx = bars.index; M = len(bars)
    shv, shi, slv, sli = conf_swings(bars, 3)
    trades = []; busy = -1; last_sl = -1; last_sh = -1
    for i in range(5, M):
        if i <= busy:
            continue
        for side in ("long", "short"):
            if side == "long":
                sj = int(sli[i]) if sli[i] == sli[i] else -1
                if not (sj > last_sl and sj >= 0):
                    continue
                last_sl = sj; m0 = sj
                while m0 > 0 and c[m0 - 1] < o[m0 - 1]:
                    m0 -= 1
                cisd = o[m0]; sw = l[sj]
                g = next((gg for gg in range(max(sj + 3, i), min(sj + TRIG, M)) if c[gg] > cisd), None)
                if g is None:
                    continue
                if mss and not (not np.isnan(shv[g]) and c[g] > shv[g]):
                    continue
                if news_block and idx[g].date() in news_set:
                    continue
                if entry == "fvg":
                    fk = None
                    for k in range(m0 + 2, g + 1):
                        if l[k] > h[k - 2]:
                            fk = k
                    if fk is None:
                        continue
                    lvl = l[fk]
                    r = next((rr for rr in range(g + 1, min(g + RETR, M)) if l[rr] <= lvl), None)
                    if r is None:
                        continue
                    ei = r; entry_px = lvl + spread + slip
                else:
                    ei = g; entry_px = c[g] + spread + slip
                sl = sw - pad; risk = entry_px - sl
                if risk <= 0:
                    continue
                tp = entry_px + rr * risk if False else entry_px + RR * risk
                res = None
                for x in range(ei + 1, min(ei + HOLD, M)):
                    if l[x] <= sl: res = (x, -1.0); break
                    if h[x] >= tp: res = (x, RR); break
                if res is None:
                    x = min(ei + HOLD, M) - 1; res = (x, (c[x] - entry_px) / risk)
                busy = res[0]
                t = dict(symbol=sym, entry_time=idx[ei], exit_time=idx[res[0]], direction="long",
                         pnl_r=res[1], spread_r=spread / risk,
                         outcome="win" if res[1] > 0.01 else ("loss" if res[1] < -0.01 else "be"))
                if rec_geo:
                    t.update(ei=ei, xj=res[0], entry=entry_px, sl=sl, tp=tp, cisd=cisd, swi=sj)
                trades.append(t)
                break
            else:
                sj = int(shi[i]) if shi[i] == shi[i] else -1
                if not (sj > last_sh and sj >= 0):
                    continue
                last_sh = sj; m0 = sj
                while m0 > 0 and c[m0 - 1] > o[m0 - 1]:
                    m0 -= 1
                cisd = o[m0]; sw = h[sj]
                g = next((gg for gg in range(max(sj + 3, i), min(sj + TRIG, M)) if c[gg] < cisd), None)
                if g is None:
                    continue
                if mss and not (not np.isnan(slv[g]) and c[g] < slv[g]):
                    continue
                if news_block and idx[g].date() in news_set:
                    continue
                if entry == "fvg":
                    fk = None
                    for k in range(m0 + 2, g + 1):
                        if h[k] < l[k - 2]:
                            fk = k
                    if fk is None:
                        continue
                    lvl = h[fk]
                    r = next((rr for rr in range(g + 1, min(g + RETR, M)) if h[rr] >= lvl), None)
                    if r is None:
                        continue
                    ei = r; entry_px = lvl - slip
                else:
                    ei = g; entry_px = c[g] - slip
                sl = sw + pad; risk = sl - entry_px
                if risk <= 0:
                    continue
                tp = entry_px - RR * risk
                res = None
                for x in range(ei + 1, min(ei + HOLD, M)):
                    if h[x] + spread >= sl: res = (x, -1.0); break
                    if l[x] + spread <= tp: res = (x, RR); break
                if res is None:
                    x = min(ei + HOLD, M) - 1; res = (x, (entry_px - (c[x] + spread)) / risk)
                busy = res[0]
                t = dict(symbol=sym, entry_time=idx[ei], exit_time=idx[res[0]], direction="short",
                         pnl_r=res[1], spread_r=spread / risk,
                         outcome="win" if res[1] > 0.01 else ("loss" if res[1] < -0.01 else "be"))
                if rec_geo:
                    t.update(ei=ei, xj=res[0], entry=entry_px, sl=sl, tp=tp, cisd=cisd, swi=sj)
                trades.append(t)
                break
    return pd.DataFrame(trades)


def degrade(df, seed=41):
    if not len(df):
        return df
    rng = np.random.default_rng(seed); d = df.reset_index(drop=True).copy()
    d = d[rng.random(len(d)) >= 0.10].reset_index(drop=True)
    r = d.pnl_r.to_numpy(float); sr = d.spread_r.to_numpy(float)
    r = r - 0.3 * sr
    r = r - np.where(rng.random(len(d)) < 0.15, 1.5 * sr, 0.0)
    r = r - np.where(r <= -0.90, 0.5 * sr, 0.0)
    r = np.where(rng.random(len(d)) < 0.002, -3.0, r)
    d["pnl_r"] = r
    d["outcome"] = np.where(r > 0.01, "win", np.where(r < -0.01, "loss", "be"))
    return d


def full_stats(deg, raw):
    st = edr.stats(deg); cl = deg[deg.outcome != "open"]; r = cl.pnl_r.astype(float)
    w = r[r > 0.01]; ls = r[r < -0.01]
    yrs = (raw.entry_time.max() - raw.entry_time.min()).days / 365.25
    d = cl.copy(); d["day"] = pd.to_datetime(d.exit_time).dt.normalize()
    daily = d.groupby("day").pnl_r.sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    return dict(n=st["n"], peryr=st["n"] / yrs, wr=st["wr"], pf=st["pf"], exp=r.mean(),
                med=r.median(), avgw=w.mean() if len(w) else 0, avgl=ls.mean() if len(ls) else 0,
                dd=st["dd"], ls=st["ls"], ws=st["ws"], sharpe=sharpe, bal=st["bal"])


def harvest_ev(r_samples, risk, n=12000, seed=0):
    rng = np.random.default_rng(seed); R = np.asarray(r_samples, float)

    def phase(target):
        bal = 0.0
        for t in range(0, MAXT, TPD):
            day = 0.0
            for _ in range(TPD):
                bal += R[rng.integers(len(R))] * risk; day += 0
                if bal <= -0.10: return "blow"
                if bal >= target: return "pass"
            # daily: recompute this day's contribution
        return "timeout"
    # simpler daily-aware version
    def phase2(target):
        bal = 0.0
        for _ in range(MAXT // TPD):
            day = 0.0
            for _ in range(TPD):
                x = R[rng.integers(len(R))] * risk; bal += x; day += x
                if bal <= -0.10: return "blow"
                if bal >= target: return "pass"
            if day <= -0.05: return "blow"
        return "timeout"
    nets = np.empty(n); funded = 0
    for a in range(n):
        net = -FEE
        if phase2(0.10) == "pass" and phase2(0.05) == "pass":
            funded += 1; first = True
            for _ in range(MAXPAY):
                if phase2(0.10) == "pass":
                    net += SPLIT * 0.10 * START
                    if first: net += FEE; first = False
                else:
                    break
        nets[a] = net
    return nets.mean(), funded / n, (nets > 0).mean()


def plot_trade(bars, t, path):
    a = max(t["ei"] - 40, 0); b = min(t["xj"] + 10, len(bars) - 1)
    o = bars.open.values[a:b + 1]; hi = bars.high.values[a:b + 1]
    lo = bars.low.values[a:b + 1]; cl = bars.close.values[a:b + 1]
    x = np.arange(len(o))
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=BG); ax.set_facecolor(BG)
    for k in range(len(o)):
        col = GREEN if cl[k] >= o[k] else RED
        ax.plot([k, k], [lo[k], hi[k]], color=col, lw=0.7, zorder=2)
        b0, b1 = sorted((o[k], cl[k]))
        ax.add_patch(Rectangle((k - 0.3, b0), 0.6, max(b1 - b0, 1e-9), facecolor=col, edgecolor=col, lw=0.4, zorder=3))
    ax.axhline(t["cisd"], color=AMBER, ls=":", lw=1.2, label="CISD line")
    ax.axhline(t["entry"], color=FG, ls="-", lw=0.8, alpha=0.7, label="entry")
    ax.axhline(t["sl"], color=RED, ls="--", lw=1.0, label="SL (swing)")
    ax.axhline(t["tp"], color=GREEN, ls="--", lw=1.0, label="TP 2R")
    up = t["direction"] == "long"
    ax.scatter([t["ei"] - a], [t["entry"]], marker="^" if up else "v", s=110, color=GREEN if up else RED,
               edgecolor=FG, zorder=6)
    ax.scatter([t["xj"] - a], [bars.close.values[t["xj"]]], marker="o", s=60, color=FG, edgecolor="#000", zorder=6)
    res = "WIN +2R" if t["pnl_r"] > 0.01 else ("LOSS -1R" if t["pnl_r"] < -0.01 else "BE")
    ax.set_title(f"{t['symbol']} 5m CISD {t['direction'].upper()} · {res} · {t['entry_time']:%Y-%m-%d %H:%M}",
                 color=GREEN if t["pnl_r"] > 0 else RED, fontsize=12, fontweight="bold")
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout(); plt.savefig(path, dpi=140, facecolor=BG); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    have = mt5.initialize()
    data = {s: load5(s) for s in SYMS}
    if have:
        mt5.shutdown()
    news = nf.build_news_block_set(pd.Timestamp("2019-01-01").date(), pd.Timestamp("2027-01-01").date())
    print("=" * 100)
    print("5m CISD · RECENT (2019-2026) · variations · causal+costs")
    print("=" * 100)
    print(f"  {'variant':22} {'N':>6} {'WR':>6} {'PF':>6} {'exp/tr':>8} {'Sharpe':>7} {'DD':>7} {'LS':>4}")
    variants = [("base (close entry)", dict()),
                ("+MSS filter", dict(mss=True)),
                ("FVG-retrace entry", dict(entry="fvg")),
                ("+news block", dict(news_block=True, news_set=news)),
                ("MSS+FVG+news", dict(mss=True, entry="fvg", news_block=True, news_set=news))]
    base_deg = None
    for label, kw in variants:
        raw = pd.concat([run(s, data[s], **kw) for s in SYMS], ignore_index=True)
        if not len(raw):
            print(f"  {label:22} no trades"); continue
        raw["entry_time"] = pd.to_datetime(raw["entry_time"])
        deg = degrade(raw.sort_values("entry_time"))
        fs = full_stats(deg, raw)
        if label.startswith("base"):
            base_deg = deg
        print(f"  {label:22} {fs['n']:>6} {fs['wr']:>5.1f}% {fs['pf']:>6.2f} {fs['exp']:>+7.3f}R "
              f"{fs['sharpe']:>7.2f} {fs['dd']:>6.0f}% {fs['ls']:>4}", flush=True)

    # deeper stats on base
    raw = pd.concat([run(s, data[s], rec_geo=True) for s in SYMS], ignore_index=True)
    raw["entry_time"] = pd.to_datetime(raw["entry_time"]); deg = degrade(raw.sort_values("entry_time"))
    fs = full_stats(deg, raw)
    print("\n  ── BASE deeper stats ──")
    print(f"    N {fs['n']} ({fs['peryr']:.0f}/yr) · WR {fs['wr']:.1f}% · PF {fs['pf']:.2f} · "
          f"expectancy {fs['exp']:+.3f}R · avgWin +{fs['avgw']:.2f}R · avgLoss {fs['avgl']:.2f}R · "
          f"Sharpe {fs['sharpe']:.2f} · maxLS {fs['ls']} · maxWS {fs['ws']}")
    print("    per-instrument PF: " + " · ".join(
        f"{s} {edr.stats(deg[deg.symbol==s])['pf']:.2f}" for s in SYMS if len(deg[deg.symbol==s]) > 20))
    deg["yr"] = pd.to_datetime(deg.entry_time).dt.year
    print("    per-year PF: " + " · ".join(
        f"{y}:{edr.stats(deg[deg.yr==y])['pf']:.2f}" for y in sorted(deg.yr.unique()) if len(deg[deg.yr==y]) > 20))

    # ── FTMO harvester on the base CISD distribution ──
    print("\n  ── FTMO HARVESTER on base 5m CISD trade distribution ──")
    R = deg.pnl_r.values
    print(f"    strategy expectancy {R.mean():+.3f}R/trade over {len(R)} trades")
    print(f"    {'risk':>6} {'funded%':>8} {'EV/challenge':>13} {'P(profit)':>10}")
    for risk in (0.005, 0.01, 0.02):
        ev, fr, pp = harvest_ev(R, risk)
        flag = "  <<< +EV" if ev > 0 else ""
        print(f"    {risk:>5.1%} {fr:>7.1%} ${ev:>+11.0f} {pp:>9.1%}{flag}", flush=True)

    # ── plot one representative winning trade (GER40 preferred) ──
    for s in ["GER40.cash", "UK100.cash", "US500.cash"]:
        tr = run(s, data[s], rec_geo=True)
        wins = tr[(tr.pnl_r > 1.9) & (tr.direction == "long")]
        if len(wins):
            t = wins.iloc[len(wins) // 2].to_dict()
            plot_trade(data[s], t, os.path.join(OUT, "sample_trade.png"))
            print(f"\n  plotted {s} trade @ {t['entry_time']} → {OUT}/sample_trade.png", flush=True)
            break


if __name__ == "__main__":
    main()
