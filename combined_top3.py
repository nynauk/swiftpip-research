"""
Combined backtest: TOP 3 symbols scanned simultaneously.

Symbols (top 3 by PF+DD from multi_asset_basic):
  · US500.cash    (best PF + best DD)
  · GBPJPY        (best PF + best loss streak)
  · UKOIL.cash    (highest total R)

Rules:
  · prev-4H candle sweep (same as dry basic)
  · 15M IFVG entry (close, or retest on 1.5x rule)
  · SL at SWING = sweep extreme + 0.25 × FVG width buffer (deeper than dry basic)
  · TP = 2R
  · News filter (red-folder days blocked)
  · 1 concurrent position per SYMBOL (independent slots — simulates
    running 3 bots or one multi-symbol bot)
"""
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import main as bt
import news_filter as nf


SYMBOLS       = ["US500.cash", "GBPJPY", "UKOIL.cash"]
RR            = 3.0                # ← back to 3RR (Option A)
REV_WINDOW    = 24
DIS_DELAY     = 7
RETR_WINDOW   = 48
DIS_MULT      = 1.5
# SL = disrespect-candle low (long) / high (short) — the DRY BASIC SL,
# not the sweep extreme.


def sweeps_prev4h(df_15, df_h4):
    prevH = df_h4["high"].shift(1); prevL = df_h4["low"].shift(1)
    floor4h = df_15.index.floor("4h")
    ph = prevH.reindex(floor4h).values
    pl = prevL.reindex(floor4h).values
    h = df_15["high"].values; l = df_15["low"].values; c = df_15["close"].values
    idx = df_15.index
    sweeps = []; cur = None; hd = ld = False
    for i in range(len(df_15)):
        p = floor4h[i]
        if p != cur:
            cur = p; hd = ld = False
        if np.isnan(ph[i]):
            continue
        if not hd and h[i] > ph[i] and c[i] < ph[i]:
            sweeps.append({"time": idx[i], "direction": "bearish",
                           "swept_level": float(ph[i])}); hd = True
        if not ld and l[i] < pl[i] and c[i] > pl[i]:
            sweeps.append({"time": idx[i], "direction": "bullish",
                           "swept_level": float(pl[i])}); ld = True
    return sweeps


def simulate(h, l, idx, ei, direction, entry, sl, tp, rr, max_bars=400):
    last = ei
    for j in range(ei + 1, min(ei + max_bars, len(h))):
        last = j
        if direction == "long":
            if l[j] <= sl:  return "loss", idx[j], sl, -1.0
            if h[j] >= tp:  return "win",  idx[j], tp, rr
        else:
            if h[j] >= sl:  return "loss", idx[j], sl, -1.0
            if l[j] <= tp:  return "win",  idx[j], tp, rr
    return "open", idx[last], float("nan"), 0.0


def run_symbol(symbol, news_set):
    df_h4 = bt.fetch_data(symbol, mt5.TIMEFRAME_H4,  bt.START_DATE, bt.END_DATE, tf_name="H4")
    df_15 = bt.fetch_data(symbol, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
    sweeps = sweeps_prev4h(df_15, df_h4)
    fvgs = sorted(bt.find_all_fvgs(df_15), key=lambda f: f["form_idx"])
    fia = np.array([f["form_idx"] for f in fvgs])
    h = df_15["high"].values; l = df_15["low"].values; c = df_15["close"].values
    idx = df_15.index

    trades = []
    used = set()
    open_until = None       # per-symbol concurrency slot

    for sw in sweeps:
        bias = sw["direction"]; sig = (sw["time"], bias)
        if sig in used: continue
        si = int(np.searchsorted(idx, sw["time"], "left"))
        if si >= len(idx) or (si == 0 and sw["time"] < idx[0]):
            continue
        tdir = "bearish" if bias == "bullish" else "bullish"
        rev_end = min(si + REV_WINDOW, len(idx) - 1)
        lo = int(np.searchsorted(fia, si, "left"))
        hi = int(np.searchsorted(fia, rev_end, "right"))

        done = False
        for p in range(hi - 1, lo - 1, -1):
            fvg = fvgs[p]
            if fvg["direction"] != tdir or fvg["width"] <= 0:
                continue
            fi = fvg["form_idx"]
            for di in range(fi + 1, min(fi + DIS_DELAY, len(idx) - 1) + 1):
                inv = c[di] > fvg["top"] if fvg["direction"] == "bearish" else c[di] < fvg["bottom"]
                if not inv:
                    continue

                dis_low, dis_high = float(l[di]), float(h[di])
                dis_range = dis_high - dis_low
                too_big = dis_range > DIS_MULT * fvg["width"]

                if not too_big:
                    ei = di; entry = c[di]; etype = "close"
                else:
                    rend = min(di + RETR_WINDOW, len(idx) - 1); ei = None
                    for rj in range(di + 1, rend + 1):
                        if bias == "bullish" and l[rj] <= fvg["top"]:
                            ei = rj; entry = fvg["top"]; break
                        if bias == "bearish" and h[rj] >= fvg["bottom"]:
                            ei = rj; entry = fvg["bottom"]; break
                    if ei is None:
                        break
                    etype = "retest"

                # NEWS FILTER — block only the specific 4H window containing the release
                if nf.is_blocked_h4(idx[ei], news_set):
                    break

                # CONCURRENCY (1 per symbol)
                if open_until is not None and pd.Timestamp(open_until) > pd.Timestamp(idx[ei]):
                    break

                # DISRESPECT-CANDLE SL — like the dry-basic setup
                if bias == "bullish":
                    direction = "long"; sl = dis_low; risk = entry - sl
                    tp = entry + RR * risk
                else:
                    direction = "short"; sl = dis_high; risk = sl - entry
                    tp = entry - RR * risk
                if risk <= 0:
                    break

                oc, xt, xp, pr = simulate(h, l, idx, ei, direction, entry, sl, tp, RR)
                if oc != "open":
                    open_until = pd.Timestamp(xt)

                trades.append({
                    "symbol": symbol, "sweep_time": sw["time"],
                    "entry_time": idx[ei], "direction": direction,
                    "entry_type": etype, "entry": round(entry, 5),
                    "sl": round(sl, 5), "tp": round(tp, 5),
                    "outcome": oc,
                    "exit": round(xp, 5) if not np.isnan(xp) else np.nan,
                    "exit_time": xt, "pnl_r": pr,
                })
                used.add(sig); done = True; break
            if done: break
    return trades


def stats(df, label):
    cl = df[df["outcome"] != "open"].copy()
    if not len(cl):
        print(f"  {label}: no closed trades"); return None
    n = len(cl); wins = (cl["outcome"] == "win").sum(); losses = n - wins
    wr = wins / n * 100
    gw = wins * RR; gl = losses
    pf = gw / gl if gl > 0 else float("inf")
    tot = cl["pnl_r"].sum()
    # equity, DD, streaks
    bal = 10000.0; eq = [bal]; cs = cw = bcl = bcw = 0
    for _, r in cl.sort_values("entry_time").iterrows():
        bal += bal * 0.01 * float(r["pnl_r"]); eq.append(bal)
        if r["pnl_r"] < -0.01: cs += 1; cw = 0; bcl = max(bcl, cs)
        elif r["pnl_r"] > 0.01: cw += 1; cs = 0; bcw = max(bcw, cw)
    eqa = np.array(eq)
    dd = ((eqa - np.maximum.accumulate(eqa)) / np.maximum.accumulate(eqa) * 100).min()
    return dict(n=n, wr=wr, pf=pf, tot=tot, dd=dd, ls=bcl, ws=bcw, bal=bal, cl=cl)


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    print("=" * 96)
    print(f"TOP-3 COMBINED  ·  RR={RR}  ·  SL = disrespect candle  ·  news-blocked  (Option A)")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print("=" * 96)

    # Compute a common H4-window block set from a probe fetch
    probe = bt.fetch_data("EURUSD", mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
    news_set = nf.build_news_block_windows(probe.index[0].date(), probe.index[-1].date())
    print(f"  News-blocked H4 windows: {len(news_set)}  (vs old full-day: "
          f"{len(nf.build_news_block_set(probe.index[0].date(), probe.index[-1].date()))} days)\n")

    per_sym = {}
    all_trades = []
    for s in SYMBOLS:
        trades = run_symbol(s, news_set)
        per_sym[s] = trades
        all_trades += trades

    mt5.shutdown()

    all_df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)

    print(f"  {'symbol':13} {'N':>5} {'WR':>6} {'PF':>5} {'totR':>7} {'DD':>7} "
          f"{'LStrk':>5} {'WStrk':>5} {'$10k→':>10}")
    for s in SYMBOLS:
        sdf = pd.DataFrame(per_sym[s])
        st = stats(sdf, s)
        if st:
            print(f"  {s:13} {st['n']:>5} {st['wr']:>5.1f}% {st['pf']:>4.2f} "
                  f"{st['tot']:>+6.0f}R {st['dd']:>6.1f}% {st['ls']:>5} {st['ws']:>5} "
                  f"${st['bal']:>8,.0f}")

    print("-" * 96)
    st = stats(all_df, "COMBINED")
    print(f"  {'COMBINED':13} {st['n']:>5} {st['wr']:>5.1f}% {st['pf']:>4.2f} "
          f"{st['tot']:>+6.0f}R {st['dd']:>6.1f}% {st['ls']:>5} {st['ws']:>5} "
          f"${st['bal']:>8,.0f}")

    # Frequency
    weeks = (all_df["entry_time"].max() - all_df["entry_time"].min()).days / 7
    print(f"\n  Trades/week (combined): {st['n']/weeks:.1f}")
    dur_hrs = (pd.to_datetime(st['cl']['exit_time']) - pd.to_datetime(st['cl']['entry_time'])).dt.total_seconds().mean() / 3600
    print(f"  Avg trade duration: {dur_hrs:.1f} hours")

    # Per-year
    cl = st["cl"].copy()
    cl["year"] = pd.to_datetime(cl["entry_time"]).dt.year
    print("\n  Per-year (combined):")
    for yr, g in cl.groupby("year"):
        wr = (g["outcome"] == "win").sum() / len(g) * 100
        r = g["pnl_r"].sum()
        print(f"    {yr}: N={len(g):>4}  WR={wr:>5.1f}%  R={r:+.0f}")

    # Concurrency profile: how many trades open at any given moment
    events = []
    for _, r in cl.iterrows():
        events.append((pd.Timestamp(r["entry_time"]), +1))
        events.append((pd.Timestamp(r["exit_time"]), -1))
    events.sort()
    active = 0; peak = 0
    hist = {0: 0, 1: 0, 2: 0, 3: 0}
    prev_t = None
    for t, d in events:
        if prev_t is not None:
            hist[active] = hist.get(active, 0) + (t - prev_t).total_seconds() / 3600
        active += d
        peak = max(peak, active)
        prev_t = t
    total_h = sum(hist.values())
    print("\n  Concurrent-position exposure (% of time with N open):")
    for k in sorted(hist):
        print(f"    {k} open: {hist[k]/total_h*100:>5.1f}%")
    print(f"    peak concurrent: {peak}")

    # Save combined trade log
    fn = f"combined_top3_{int(RR)}R_trades.csv"
    all_df.to_csv(fn, index=False)
    print(f"\n  Saved trade log → {fn} ({len(all_df)} trades)")


if __name__ == "__main__":
    main()
