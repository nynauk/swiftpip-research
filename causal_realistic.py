"""
THE REALISTIC BACKTEST — all five degradation factors modeled, 2000-2026 dual-feed.

F1 CAUSAL SWINGS (exact fix, changes which trades exist):
   swing at index j is only usable from bar j+SWING_LOOKBACK (=5) onward, for BOTH
   the SL anchor and the DOL>=2R selection filter — identical to the live detector.
F2 SELECTION GLOW: report curated-21 AND full-40; plus WALK-FORWARD curation
   (instruments picked on 2000-2013 only, tested 2014-2026).
F3 EXECUTION (seeded RNG, applied per trade):
   - 10% of entries MISSED (offline / closed-bar tag / drift tolerance)
   - poll-lag cost 0.3×(spread/risk) R on every fill
   - 15% of trades pay 2.5× spread (rollover/news widening): extra 1.5×(spread/risk) R
   - stop slippage 0.5×(spread/risk) R on stopped-out trades
F4 FTMO FRICTIONS: sim breaches at -4% daily / -9% overall (floating-loss slack),
   on the 4/day-capped stream, $10k account, €89 re-buys, 80% split.
F5 TAIL EVENTS: 0.2% of trades (~1-2/yr) gap through the stop to -3R.
"""
import os
import gc
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ema_dol_strategy as ed
import frequency_tune as fq
import test_gradient_exit as tg
import test_cross_exit as tc
import realistic_ema_dol_rsi as edr
import histdata_fetch as hd
import all_pairs_2000_2026 as ap          # U universe + spec injections + mt5_bars

EMA = (7, 18)
CONF = ed.SWING_LOOKBACK                   # 5 — bars until a swing is confirmable
OUT = os.path.join("strategies", "realistic")
BG, FG = "#0d1117", "#e6edf3"
KEEP = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "EURJPY", "GBPJPY", "AUDJPY",
        "CADJPY", "NZDJPY", "EURAUD", "GBPAUD", "XAUUSD",
        "US500.cash", "US100.cash", "GER40.cash", "UK100.cash",
        "FRA40.cash", "JP225.cash", "HK50.cash", "USOIL.cash", "UKOIL.cash"]


# ───────────────────────── F1: strictly causal engine ─────────────────────────
def run_causal(symbol, base, ema, p, grad_lb=1, profit_only=True):
    """tg.run_gradient with CAUSAL swing usage: at entry bar ei only swings formed
    at index <= ei-CONF exist (confirmed live). Also records risk & spread_r."""
    long_lo, long_hi, short_lo, short_hi = p["rsi_band"]
    spread, slippage, min_w = fq.SPEC[symbol]
    h, l, c, idx = base["h"], base["l"], base["c"], base["idx"]
    sh, sl_arr = base["sh"], base["sl_arr"]
    pdh_map, pdl_map, rsi_arr = base["pdh"], base["pdl"], base["rsi"]
    news_set = base["news"] if p["news_on"] else set()
    ef = pd.Series(c, index=idx).ewm(span=ema[0], adjust=False).mean().values
    cu, cd, nb, nbr = fq.cross_arrays(c, idx, ema[0], ema[1])
    N = len(idx)
    FVGW, RTW, SLB = p["fvg_window"], p["retest_window"], p["sl_buf"]

    trades = []; open_until = []
    for i in range(N):
        if not (cu[i] or cd[i]):
            continue
        bias = "long" if cu[i] else "short"
        tdir = "bullish" if bias == "long" else "bearish"
        opp = nbr[i + 1] if bias == "long" else nb[i + 1]
        end_idx = min(i + FVGW, opp - 1, N - 1)
        if end_idx < i + 2:
            continue
        fvg = ed.find_aligned_fvg(h, l, i, end_idx, tdir, min_w)
        if fvg is None:
            continue
        rt_end = min(fvg["form_idx"] + RTW, opp - 1, N - 1)
        ei = None
        for k in range(fvg["form_idx"] + 1, rt_end + 1):
            if bias == "long" and l[k] <= fvg["top"]:    ei = k; break
            if bias == "short" and h[k] >= fvg["bottom"]: ei = k; break
        if ei is None or ei - CONF < 1:
            continue
        open_until = [t for t in open_until if t > idx[ei]]
        if len(open_until) >= 1:
            continue
        if idx[ei].date() in news_set:
            continue
        rv = rsi_arr[ei]
        if np.isnan(rv):
            continue
        if bias == "long" and not (long_lo <= rv <= long_hi):
            continue
        if bias == "short" and not (short_lo <= rv <= short_hi):
            continue
        d_today = pd.Timestamp(idx[ei]).normalize()
        pdh_t = pdh_map.get(d_today); pdl_t = pdl_map.get(d_today)
        buf = fvg["width"] * SLB
        pad = edr.sl_pad(symbol, spread)
        # ---- CAUSAL: only swings confirmed by bar ei (formed at <= ei-CONF) ----
        if bias == "long":
            entry = fvg["top"] + spread + slippage
            swlow = ed.find_recent_swing_low(sl_arr, ei - CONF, lookback=80)
            anchor = swlow if swlow is not None and swlow < fvg["bottom"] else fvg["bottom"]
            sl = anchor - buf - pad; risk = entry - sl; direction = "long"
            if risk <= 0: continue
            dol = ed.next_liquidity_above(entry, sh, ei - CONF + 1, pdh_t)
            if dol is None or not (2.0 <= (dol - entry) / risk <= 10.0): continue
        else:
            entry = fvg["bottom"] - slippage
            swhigh = ed.find_recent_swing_high(sh, ei - CONF, lookback=80)
            anchor = swhigh if swhigh is not None and swhigh > fvg["top"] else fvg["top"]
            sl = anchor + buf + pad; risk = sl - entry; direction = "short"
            if risk <= 0: continue
            dol = ed.next_liquidity_below(entry, sl_arr, ei - CONF + 1, pdl_t)
            if dol is None or not (2.0 <= (entry - dol) / risk <= 10.0): continue
        oc, xj, pr = tg.sim_gradient(h, l, c, ef, ei, direction, entry, sl, risk, opp,
                                     spread, grad_lb, profit_only)
        open_until.append(pd.Timestamp(idx[xj]))
        trades.append(dict(symbol=symbol, entry_time=idx[ei], direction=direction,
                           outcome=oc, exit_time=idx[xj], pnl_r=pr,
                           spread_r=spread / risk,
                           hold_h=(idx[xj] - idx[ei]).total_seconds() / 3600))
    return pd.DataFrame(trades)


# ───────────────────────── F3+F5: execution degradation ─────────────────────────
def degrade(df, seed=42):
    rng = np.random.default_rng(seed)
    d = df.copy().reset_index(drop=True)
    keep = rng.random(len(d)) >= 0.10                 # 10% of entries missed
    d = d[keep].reset_index(drop=True)
    r = d["pnl_r"].to_numpy(float)
    sr = d["spread_r"].to_numpy(float)
    r = r - 0.30 * sr                                  # poll-lag fill cost, every trade
    wide = rng.random(len(d)) < 0.15                   # rollover/news spread widening
    r = r - np.where(wide, 1.5 * sr, 0.0)
    stopped = r <= -0.90
    r = r - np.where(stopped, 0.5 * sr, 0.0)           # stop slippage
    tail = rng.random(len(d)) < 0.002                  # gap-through-stop disasters
    r = np.where(tail, -3.0, r)
    d["pnl_r"] = r
    d["outcome"] = np.where(r > 0.01, "win", np.where(r < -0.01, "loss", "be"))
    return d


def row(tag, df, days):
    st = edr.stats(df)
    if st is None:
        print(f"  {tag:26} — no trades", flush=True)
        return None
    cl = df[df.outcome != "open"]; w = cl.loc[cl.pnl_r > 0.01, "pnl_r"]
    print(f"  {tag:26} {st['n']:>6} {st['n']/days:>6.2f} {st['wr']:>6.1f}% {st['pf']:>5.2f} "
          f"{st['tot']:>+9.1f} {st['dd']:>6.1f}% {st['ls']:>4} ${st['bal']:>11,.0f}", flush=True)
    return st


def main():
    os.makedirs(OUT, exist_ok=True)
    edr.SL_SPREAD_PAD = True
    csv = os.path.join(OUT, "causal_trades.csv")

    if not os.path.exists(csv):
        print("ENGINE: causal re-run · 40 instruments × M3/M5/M15 · 2000-2026", flush=True)
        parts = []
        have_mt5 = mt5.initialize()
        for sym, (code, cands, spec, pip) in ap.U.items():
            for tf, rule, tfc in ap.TFS:
                try:
                    bars = hd.fetch(sym, ap.YEARS, tf=rule)
                except Exception:
                    bars = None
                if bars is not None and len(bars) > 2000:
                    base = fq.precompute_base(sym, bars)
                    t = run_causal(sym, base, EMA, fq.BASEP)
                    if len(t):
                        t = t[pd.to_datetime(t.entry_time) <= ap.HD_END]
                    if len(t):
                        t["tf"] = tf; parts.append(t)
                    del base
                del bars; gc.collect()
                if have_mt5:
                    dfm = ap.mt5_bars(cands, tfc)
                    if dfm is not None:
                        base = fq.precompute_base(sym, dfm)
                        t = run_causal(sym, base, EMA, fq.BASEP)
                        if len(t):
                            t = t[pd.to_datetime(t.entry_time) >= ap.CUT]
                        if len(t):
                            t["tf"] = tf; parts.append(t)
                        del base
                    del dfm; gc.collect()
            print(f"  {sym} done", flush=True)
        if have_mt5:
            mt5.shutdown()
        causal = pd.concat(parts, ignore_index=True).sort_values("entry_time")
        causal.to_csv(csv, index=False)
    else:
        causal = pd.read_csv(csv)

    causal["entry_time"] = pd.to_datetime(causal["entry_time"])
    causal["exit_time"] = pd.to_datetime(causal["exit_time"])
    baseline = pd.read_csv(os.path.join("strategies", "all_pairs", "all_pairs_trades.csv"))
    baseline["entry_time"] = pd.to_datetime(baseline["entry_time"])
    baseline["exit_time"] = pd.to_datetime(baseline["exit_time"])

    days = (causal.entry_time.max() - causal.entry_time.min()).days * 5 / 7
    base_cur = baseline[baseline.symbol.isin(KEEP)]
    cur = causal[causal.symbol.isin(KEEP)]
    cur_deg = degrade(cur)
    full_deg = degrade(causal)

    print("\n" + "=" * 114, flush=True)
    print("REALISTIC BACKTEST · degradation waterfall · 2000-2026 · $1k/$100k", flush=True)
    print("=" * 114, flush=True)
    print(f"  {'stream':26} {'N':>6} {'/day':>6} {'WR':>7} {'PF':>5} {'totR':>9} "
          f"{'DD':>7} {'LS':>4} {'End$':>12}", flush=True)
    print("-" * 114, flush=True)
    row("1 baseline curated21", base_cur, days)
    row("2 +causal swings", cur, days)
    row("3 +execution+tails", cur_deg, days)
    row("4 same, full40 (no glow)", full_deg, days)

    # ── F2: walk-forward curation (pick on 2000-2013, test 2014-2026) ──
    split = pd.Timestamp("2014-01-01")
    ins = causal[causal.entry_time < split]
    wf_keep = []
    for s in causal.symbol.unique():
        g = ins[ins.symbol == s]
        st = edr.stats(g) if len(g) >= 30 else None
        if st and st["pf"] >= 2.0:
            wf_keep.append(s)
    oos = degrade(causal[(causal.entry_time >= split) & causal.symbol.isin(wf_keep)], seed=7)
    oos_cur = degrade(causal[(causal.entry_time >= split) & causal.symbol.isin(KEEP)], seed=7)
    d2 = (causal.entry_time.max() - split).days * 5 / 7
    print(f"\n  ── walk-forward curation: picked on 2000-2013 (PF>=2, N>=30) → "
          f"{len(wf_keep)} instruments → tested 2014-2026, degraded ──", flush=True)
    row("WF basket 2014-26", oos, d2)
    row("(hindsight 21, same win)", oos_cur, d2)
    print(f"    WF basket: {sorted(wf_keep)}", flush=True)

    # ── F4: FTMO sim, deployed reality ──
    print(f"\n  ── FTMO sim · $10k · 1% ($100/trade) · 4/day cap · breach at -4% daily / -9% overall "
          f"(floating slack) · €89 ──", flush=True)
    for tag, stream in [("causal+exec curated", cur_deg), ("causal+exec full40", full_deg)]:
        s = stream.sort_values("entry_time").copy()
        s["eday"] = s.entry_time.dt.normalize()
        s = s.groupby("eday", group_keys=False).head(4)
        ps = edr.prop_sim(s, start=10_000, daily_pct=0.04, overall_pct=0.09, risk_usd=100.0)
        yrs = days / 252
        net = ps["banked"] * 0.80 - ps["blown"] * 89.0
        print(f"    {tag:22}: {ps['passes']} payouts ({ps['passes']/yrs:.1f}/yr) · "
              f"blown {ps['blown']} ({ps['blown_daily']}d/{ps['blown_overall']}o) · "
              f"NET ${net:,.0f} (${net/yrs:,.0f}/yr)", flush=True)

    cur_deg.to_csv(os.path.join(OUT, "realistic_curated_trades.csv"), index=False)

    # ── chart: the honesty waterfall ──
    fig, ax = plt.subplots(figsize=(13.5, 6.5), facecolor=BG); ax.set_facecolor(BG)
    for tag, df, col in [("baseline curated", base_cur, "#7d8590"),
                         ("+causal swings", cur, "#d29922"),
                         ("+execution+tails (REALISTIC)", cur_deg, "#3fb950")]:
        st = edr.stats(df)
        ax.plot(st["times"], st["eqa"], color=col, linewidth=1.5, drawstyle="steps-post",
                label=f"{tag}: N={st['n']}, WR {st['wr']:.0f}%, PF {st['pf']:.2f}, "
                      f"DD {st['dd']:.1f}%, ${st['bal']:,.0f}")
    ax.axhline(100000, color="#444c56", linestyle="--", linewidth=0.8)
    ax.set_title("The honesty waterfall · curated 21 · M3+M5+M15 · 2000-2026",
                 color=FG, fontsize=12, fontweight="bold")
    ax.set_xlabel("Date", color=FG); ax.set_ylabel("Balance ($)", color=FG)
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "honesty_waterfall.png"), dpi=140, facecolor=BG)
    plt.close(fig)
    print(f"\nSaved → {OUT}/  (honesty_waterfall.png, realistic_curated_trades.csv)", flush=True)


if __name__ == "__main__":
    main()
