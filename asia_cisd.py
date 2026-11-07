"""
ASIA-SWEEP + 1m CISD retrace — strictly causal backtest, EURUSD + USDCHF, M1.

Setup (all causal — nothing uses a future bar):
  1. Asian range = high/low over the Tokyo window (completed before we act).
  2. London window: a bar sweeps the Asian high (wick above, close back below)
     → bearish bias; or sweeps the Asian low (close back above) → bullish.
  3. CISD level = open of the FIRST candle of the run that pushed into the sweep
     (change-in-state-of-delivery origin). CISD confirmed when a later bar CLOSES
     back through that level (displacement).
  4. Wait for price to RETRACE to the CISD level → enter there.
  5. SL = the swept swing extreme (+spread+1pip). TP = 2R.
One trade per instrument per session day. Costs on every fill + degradation.
Session clock = EST+7 (HistData is EST, no DST): shour 0 = 17:00 EST rollover.
Asian range = shour 2-9 (19:00-02:00 EST) · London/entry = shour 10-16.
"""
import os
import numpy as np
import pandas as pd

import frequency_tune as fq
import realistic_ema_dol_rsi as edr
import histdata_fetch as hd

edr.SL_SPREAD_PAD = True
SYMS = ["EURUSD", "USDCHF"]
YEARS = list(range(2004, 2025))
SPLITS = [("2004-2011", "2004-01-01", "2012-01-01"),
          ("2012-2018", "2012-01-01", "2019-01-01"),
          ("2019-2024", "2019-01-01", "2025-01-01")]
CISD_WIN = 40         # bars after sweep to confirm CISD
RETR_WIN = 40         # bars after CISD to fill the retrace
MAX_HOLD = 480        # bars to resolve the trade (8h on M1)


def run(sym, m1):
    spread, slip, _ = fq.SPEC[sym]; pad = edr.sl_pad(sym, spread)
    o = m1.open.values; h = m1.high.values; l = m1.low.values; c = m1.close.values
    idx = m1.index
    sess = idx + pd.Timedelta(hours=7)
    sday = sess.normalize().values.astype("datetime64[D]")
    shour = sess.hour.values
    N = len(m1)
    starts = np.where(np.concatenate([[True], sday[1:] != sday[:-1]]))[0]
    trades = []
    for di in range(len(starts)):
        s0 = starts[di]; s1 = starts[di + 1] if di + 1 < len(starts) else N
        sh = shour[s0:s1]
        am = (sh >= 2) & (sh <= 9)
        if am.sum() < 30:
            continue
        ah = h[s0:s1][am].max(); al = l[s0:s1][am].min()
        lon = np.where((sh >= 10) & (sh <= 16))[0] + s0
        if len(lon) < 20:
            continue
        took = False
        for k in lon:
            if took:
                break
            # ── bearish: sweep Asian high + reject ──
            if h[k] > ah and c[k] < ah:
                m0 = k
                while m0 > s0 and c[m0 - 1] > o[m0 - 1]:
                    m0 -= 1
                lvl = o[m0]; ext = h[k]
                j = next((jj for jj in range(k + 1, min(k + CISD_WIN, s1 - 1) + 1)
                          if c[jj] < lvl), None)
                if j is None:
                    continue
                r = next((rr for rr in range(j + 1, min(j + RETR_WIN, s1 - 1) + 1)
                          if h[rr] >= lvl), None)
                if r is None:
                    continue
                entry = lvl - slip; sl = ext + pad; risk = sl - entry
                if risk <= 0:
                    continue
                tp = entry - 2 * risk; res = None
                for x in range(r + 1, min(r + MAX_HOLD, s1)):
                    if h[x] + spread >= sl: res = (x, -1.0); break
                    if l[x] + spread <= tp: res = (x, 2.0); break
                if res is None:
                    x = min(r + MAX_HOLD, s1) - 1; res = (x, (entry - (c[x] + spread)) / risk)
                trades.append(dict(symbol=sym, entry_time=idx[r], exit_time=idx[res[0]],
                                   direction="short", pnl_r=res[1], spread_r=spread / risk,
                                   outcome="win" if res[1] > 0.01 else ("loss" if res[1] < -0.01 else "be")))
                took = True
            # ── bullish: sweep Asian low + reject ──
            elif l[k] < al and c[k] > al:
                m0 = k
                while m0 > s0 and c[m0 - 1] < o[m0 - 1]:
                    m0 -= 1
                lvl = o[m0]; ext = l[k]
                j = next((jj for jj in range(k + 1, min(k + CISD_WIN, s1 - 1) + 1)
                          if c[jj] > lvl), None)
                if j is None:
                    continue
                r = next((rr for rr in range(j + 1, min(j + RETR_WIN, s1 - 1) + 1)
                          if l[rr] <= lvl), None)
                if r is None:
                    continue
                entry = lvl + spread + slip; sl = ext - pad; risk = entry - sl
                if risk <= 0:
                    continue
                tp = entry + 2 * risk; res = None
                for x in range(r + 1, min(r + MAX_HOLD, s1)):
                    if l[x] <= sl: res = (x, -1.0); break
                    if h[x] >= tp: res = (x, 2.0); break
                if res is None:
                    x = min(r + MAX_HOLD, s1) - 1; res = (x, (c[x] - entry) / risk)
                trades.append(dict(symbol=sym, entry_time=idx[r], exit_time=idx[res[0]],
                                   direction="long", pnl_r=res[1], spread_r=spread / risk,
                                   outcome="win" if res[1] > 0.01 else ("loss" if res[1] < -0.01 else "be")))
                took = True
    return pd.DataFrame(trades)


def degrade(df, seed=11):
    if not len(df):
        return df
    rng = np.random.default_rng(seed)
    d = df.reset_index(drop=True).copy()
    keep = rng.random(len(d)) >= 0.10                    # missed entries
    d = d[keep].reset_index(drop=True)
    r = d.pnl_r.to_numpy(float); sr = d.spread_r.to_numpy(float)
    r = r - 0.3 * sr                                     # poll-lag on fill
    wide = rng.random(len(d)) < 0.15
    r = r - np.where(wide, 1.5 * sr, 0.0)               # spread widening
    stopped = r <= -0.90
    r = r - np.where(stopped, 0.5 * sr, 0.0)            # stop slippage
    tail = rng.random(len(d)) < 0.002
    r = np.where(tail, -3.0, r)                          # gap through stop
    d["pnl_r"] = r
    d["outcome"] = np.where(r > 0.01, "win", np.where(r < -0.01, "loss", "be"))
    return d


def main():
    print("Loading M1 (EURUSD, USDCHF, 2004-2024)...", flush=True)
    allt = []
    for s in SYMS:
        m1 = hd.fetch(s, YEARS, tf="1min")
        t = run(s, m1)
        print(f"  {s}: {len(t)} raw setups", flush=True)
        allt.append(t)
    raw = pd.concat(allt, ignore_index=True)
    raw["entry_time"] = pd.to_datetime(raw["entry_time"])
    deg = degrade(raw.sort_values("entry_time"))

    print("\n" + "=" * 104)
    print("ASIA-SWEEP + 1m CISD · EURUSD+USDCHF · causal · 2R/swing · costs+degradation")
    print("=" * 104)
    print(f"  {'stream':16} {'N':>6} {'/wk':>6} {'WR':>7} {'PF':>6} {'totR':>8} {'DD':>7} {'LS':>4} {'End$':>11}")
    print("-" * 104, flush=True)
    for tag, df in [("raw (no exec)", raw), ("realistic", deg)]:
        st = edr.stats(df)
        wk = max((raw.entry_time.max() - raw.entry_time.min()).days / 7, 1)
        print(f"  {tag:16} {st['n']:>6} {st['n']/wk:>6.2f} {st['wr']:>6.1f}% {st['pf']:>6.2f} "
              f"{st['tot']:>+8.1f} {st['dd']:>6.1f}% {st['ls']:>4} ${st['bal']:>10,.0f}", flush=True)

    print("\n  ── per era (realistic) · survivor gate: PF>=1.25 in EVERY era ──", flush=True)
    ok = True
    for name, a, b in SPLITS:
        g = deg[(deg.entry_time >= a) & (deg.entry_time < b)]
        st = edr.stats(g)
        if st is None or st["n"] < 30:
            print(f"    {name}: too few ({0 if st is None else st['n']})"); ok = False; continue
        print(f"    {name}: N={st['n']:>4} WR {st['wr']:>4.0f}% PF {st['pf']:>5.2f} "
              f"{st['tot']:>+6.1f}R DD {st['dd']:>5.1f}%", flush=True)
        if st["pf"] < 1.25:
            ok = False
    for s in SYMS:
        g = deg[deg.symbol == s]; st = edr.stats(g)
        if st:
            print(f"    {s}: N={st['n']:>4} WR {st['wr']:>4.0f}% PF {st['pf']:>5.2f} {st['tot']:>+6.1f}R", flush=True)

    verdict = "✅ SURVIVOR — build the bot" if ok and edr.stats(deg)["pf"] >= 1.3 else "❌ no edge after costs"
    print(f"\n  VERDICT: {verdict}", flush=True)
    os.makedirs(os.path.join("strategies", "asia_cisd"), exist_ok=True)
    deg.to_csv(os.path.join("strategies", "asia_cisd", "trades.csv"), index=False)


if __name__ == "__main__":
    main()
