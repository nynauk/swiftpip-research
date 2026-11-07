"""
Asia liquidity model — full stack, strictly causal:
  · Asian range (Tokyo window). SKIP the day if the range is WIDE (> 1.3× the
    rolling-median of the last 20 Asian ranges) — avoid high-volatility days.
  · London: sweep of the Asian high (→short) or low (→long), close back inside.
  · Require a 1m BOS in the reversal direction (break a confirmed 1m swing).
  · Entry on the 1m IFVG (displacement FVG that price retraces into).
  · TP = OPPOSING liquidity (the other side of the Asian range). SL = swept swing.
  · AVOID NEWS: skip red-folder days (news_filter set).
FX majors, M1 2010-2024, per-instrument spread + degradation + era splits.
"""
import os
import numpy as np
import pandas as pd

import frequency_tune as fq
import realistic_ema_dol_rsi as edr
import histdata_fetch as hd
import news_filter as nf
from request1_htf import conf_swings

edr.SL_SPREAD_PAD = True
SYMS = ["EURUSD", "GBPUSD", "USDCHF", "USDJPY"]
YEARS = list(range(2010, 2025))
SPLITS = [("2010-2014", "2010-01-01", "2015-01-01"),
          ("2015-2019", "2015-01-01", "2020-01-01"),
          ("2020-2024", "2020-01-01", "2025-01-01")]
WIDE_K = 1.3; BOS_WIN = 60; FVG_WIN = 40; MAX_HOLD = 480; MIN_RR = 1.0


def run(sym, m1, news_set):
    spread, slip, _ = fq.SPEC[sym]; pad = edr.sl_pad(sym, spread)
    o, h, l, c = m1.open.values, m1.high.values, m1.low.values, m1.close.values
    idx = m1.index
    sess = idx + pd.Timedelta(hours=7)
    sday = sess.normalize().values.astype("datetime64[D]")
    shour = sess.hour.values
    M = len(m1)
    _, _, csl_v, _ = conf_swings(m1, 5)       # for BOS-down (break swing low)
    csh_v, _, _, _ = conf_swings(m1, 5)       # for BOS-up (break swing high)
    starts = np.where(np.concatenate([[True], sday[1:] != sday[:-1]]))[0]
    trades = []; recent = []
    for di in range(len(starts)):
        s0 = starts[di]; s1 = starts[di + 1] if di + 1 < len(starts) else M
        sh = shour[s0:s1]
        am = (sh >= 2) & (sh <= 9)
        if am.sum() < 30:
            continue
        ah = h[s0:s1][am].max(); al = l[s0:s1][am].min(); arange = ah - al
        med = np.median(recent[-20:]) if len(recent) >= 10 else None
        recent.append(arange)
        if med is not None and arange > WIDE_K * med:      # WIDE day → skip
            continue
        lon = np.where((sh >= 10) & (sh <= 16))[0] + s0
        if len(lon) < 20:
            continue
        day = pd.Timestamp(idx.values[lon[0]]).date()
        if day in news_set:                                # AVOID NEWS
            continue
        took = False
        for k in lon:
            if took:
                break
            bias = None
            if h[k] > ah and c[k] < ah:
                bias = "short"; ext = h[k]; tp_liq = al
            elif l[k] < al and c[k] > al:
                bias = "long"; ext = l[k]; tp_liq = ah
            if bias is None:
                continue
            # ── BOS in reversal direction ──
            if bias == "short":
                bos = next((b for b in range(k + 1, min(k + BOS_WIN, M))
                            if not np.isnan(csl_v[b]) and c[b] < csl_v[b]), None)
            else:
                bos = next((b for b in range(k + 1, min(k + BOS_WIN, M))
                            if not np.isnan(csh_v[b]) and c[b] > csh_v[b]), None)
            if bos is None:
                continue
            # ── IFVG entry ──
            got = None
            for j in range(bos, min(bos + FVG_WIN, M)):
                if j < 3:
                    continue
                if bias == "short" and h[j] < l[j - 2] and c[j] < o[j]:
                    for r in range(j + 1, min(j + FVG_WIN, M)):
                        if h[r] >= h[j]:
                            entry = h[j] - slip; sl = ext + pad; risk = sl - entry
                            if risk >= 0.15 * arange and entry - 10 * risk < tp_liq < entry - MIN_RR * risk:
                                got = (r, entry, sl, tp_liq, risk); break
                    if got:
                        break
                if bias == "long" and l[j] > h[j - 2] and c[j] > o[j]:
                    for r in range(j + 1, min(j + FVG_WIN, M)):
                        if l[r] <= l[j]:
                            entry = l[j] + spread + slip; sl = ext - pad; risk = entry - sl
                            if risk >= 0.15 * arange and entry + MIN_RR * risk < tp_liq < entry + 10 * risk:
                                got = (r, entry, sl, tp_liq, risk); break
                    if got:
                        break
            if not got:
                continue
            r, entry, sl, tp, risk = got; rr = abs(tp - entry) / risk
            res = None
            for x in range(r + 1, min(r + MAX_HOLD, M)):
                if bias == "short":
                    if h[x] + spread >= sl: res = (x, -1.0); break
                    if l[x] + spread <= tp: res = (x, rr); break
                else:
                    if l[x] <= sl: res = (x, -1.0); break
                    if h[x] >= tp: res = (x, rr); break
            if res is None:
                x = min(r + MAX_HOLD, M) - 1
                res = (x, ((entry - (c[x] + spread)) if bias == "short" else (c[x] - entry)) / risk)
            trades.append(dict(symbol=sym, entry_time=idx[r], exit_time=idx[res[0]], direction=bias,
                               pnl_r=res[1], rr=rr, spread_r=spread / risk,
                               outcome="win" if res[1] > 0.01 else ("loss" if res[1] < -0.01 else "be")))
            took = True
    return pd.DataFrame(trades)


def degrade(df, seed=17):
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


def main():
    news = nf.build_news_block_set(pd.Timestamp("2010-01-01").date(), pd.Timestamp("2025-01-01").date())
    print(f"news block days: {len(news)}", flush=True)
    allt = []
    for s in SYMS:
        m1 = hd.fetch(s, YEARS, tf="1min")
        t = run(s, m1, news)
        print(f"  {s}: {len(t)} setups", flush=True)
        allt.append(t)
    raw = pd.concat(allt, ignore_index=True)
    raw["entry_time"] = pd.to_datetime(raw["entry_time"])
    deg = degrade(raw.sort_values("entry_time"))
    st = edr.stats(deg); yrs = (raw.entry_time.max() - raw.entry_time.min()).days / 365.25
    print("\n" + "=" * 96)
    print("ASIA FULL STACK · sweep+wide-filter+BOS+IFVG+opposing-liquidity+news · FX majors · causal")
    print("=" * 96)
    print(f"  N={st['n']} ({st['n']/yrs:.0f}/yr) · median RR {raw.rr.median():.2f} · WR {st['wr']:.1f}% · "
          f"PF {st['pf']:.2f} · {st['tot']:+.0f}R · DD {st['dd']:.0f}% · ${st['bal']:,.0f}", flush=True)
    ok = True
    for name, a, b in SPLITS:
        g = deg[(deg.entry_time >= a) & (deg.entry_time < b)]; s2 = edr.stats(g)
        if s2 is None or s2["n"] < 30:
            print(f"    {name}: few ({0 if s2 is None else s2['n']})"); ok = False; continue
        print(f"    {name}: N={s2['n']:>4} WR {s2['wr']:>4.0f}% PF {s2['pf']:>5.2f} {s2['tot']:>+6.0f}R")
        if s2["pf"] < 1.25:
            ok = False
    for s in SYMS:
        g = deg[deg.symbol == s]; s2 = edr.stats(g)
        if s2:
            print(f"    {s}: N={s2['n']:>4} WR {s2['wr']:>4.0f}% PF {s2['pf']:>5.2f}")
    print(f"\n  VERDICT: {'✅ SURVIVOR — develop it' if ok and st['pf'] >= 1.3 else '❌ no edge after costs'}", flush=True)


if __name__ == "__main__":
    main()
