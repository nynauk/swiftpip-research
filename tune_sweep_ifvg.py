"""
Grid-search tuner for the 4H-swing sweep + 15M IFVG strategy.

FUNDAMENTALS (fixed):
   • 4H swing high/low sweep (5-bar swings)
   • 15M reversal FVG that inverts (IFVG) in the bias direction

EVERYTHING ELSE is a tunable variable:
   sl_mode        SWEEP_EXT | IFVG_EDGE | DISP  (displacement candle)
   rr             1.0 | 1.5 | 2.0 | 3.0
   min_fvg_pips   1 | 2 | 4
   max_close_loc  1.0(off) | 0.5 | 0.3     (inversion candle strong close)
   session        (0,24) | (7,16) | (12,18)   broker hours for entry
   news_block     False | True
   htf_cisd       False | True             (4H confirms shift after sweep)
   entry_mode     RULE(1.5x) | CLOSE | RETRACE

Per symbol we precompute sweeps / FVGs / per-sweep HTF-CISD / news set ONCE,
then replay each parameter combo cheaply. Ranked by win rate with a minimum
trade-count floor so we don't crown a 5-trade fluke.
"""
import os
import itertools
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import main as bt
import news_filter as nf

MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]
PIP = {"EURUSD":0.0001,"GBPUSD":0.0001,"USDCAD":0.0001,"USDCHF":0.0001,
       "USDJPY":0.01,"AUDUSD":0.0001,"NZDUSD":0.0001}
REV_WINDOW   = 24
DIS_DELAY    = 7
RETR_WINDOW  = 48
DIS_MULT     = 1.5
SL_BUF       = 0.25
MIN_TRADES   = 200   # floor for ranking (combined across majors)


def simulate(h, l, idx, ei, direction, entry, sl, tp, rr, max_bars=400):
    last = ei
    for j in range(ei + 1, min(ei + max_bars, len(h))):
        last = j
        if direction == "long":
            if l[j] <= sl: return "loss", idx[j], -1.0
            if h[j] >= tp: return "win",  idx[j], rr
        else:
            if h[j] >= sl: return "loss", idx[j], -1.0
            if l[j] <= tp: return "win",  idx[j], rr
    return "open", idx[last], 0.0


def build_ctx(symbol):
    """Precompute everything param-independent for one symbol."""
    df_h4 = bt.fetch_data(symbol, mt5.TIMEFRAME_H4, bt.START_DATE, bt.END_DATE, tf_name="H4")
    df_15 = bt.fetch_data(symbol, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
    df_sw = bt.find_swings(df_h4, bt.SWING_LOOKBACK)
    sweeps = bt.find_sweeps(df_sw)
    fvgs = sorted(bt.find_all_fvgs(df_15), key=lambda f: f["form_idx"])
    fia = np.array([f["form_idx"] for f in fvgs])

    h4h, h4l, h4c = df_h4["high"].values, df_h4["low"].values, df_h4["close"].values

    # per-sweep HTF CISD flag: after sweep, a 4H close through recent opposing swing
    htf_cisd = {}
    for sw in sweeps:
        s4 = int(np.searchsorted(df_h4.index, sw["time"], side="left"))
        flag = False
        for k in range(s4 + 1, min(s4 + 12, len(df_h4))):
            if sw["direction"] == "bullish":
                if h4c[k] > h4h[max(0, k - 5):k].max(): flag = True; break
            else:
                if h4c[k] < h4l[max(0, k - 5):k].min(): flag = True; break
        htf_cisd[sw["time"]] = flag

    news_set = nf.build_news_block_set(df_15.index[0].date(), df_15.index[-1].date())

    return {
        "symbol": symbol, "df_15": df_15,
        "h": df_15["high"].values, "l": df_15["low"].values, "c": df_15["close"].values,
        "idx": df_15.index, "sweeps": sweeps, "fvgs": fvgs, "fia": fia,
        "htf_cisd": htf_cisd, "news_set": news_set, "pip": PIP.get(symbol, 0.0001),
    }


def run_combo(ctx, *, sl_mode, rr, min_fvg_pips, max_close_loc, session,
              news_block, htf_cisd_req, entry_mode, collect=False):
    h, l, c, idx = ctx["h"], ctx["l"], ctx["c"], ctx["idx"]
    fvgs, fia = ctx["fvgs"], ctx["fia"]
    min_fvg = min_fvg_pips * ctx["pip"]
    s0, s1 = session
    hrs = idx.hour
    news_set = ctx["news_set"]
    htf = ctx["htf_cisd"]

    wins = losses = 0
    used = set(); open_until = []
    rec = []
    for sw in ctx["sweeps"]:
        bias = sw["direction"]; sig = (sw["time"], bias)
        if sig in used:
            continue
        if htf_cisd_req and not htf.get(sw["time"], False):
            continue
        si = int(np.searchsorted(idx, sw["time"], side="left"))
        if si >= len(idx) or (si == 0 and sw["time"] < idx[0]):
            continue

        tdir = "bearish" if bias == "bullish" else "bullish"
        rev_end = min(si + REV_WINDOW, len(idx) - 1)
        lo = int(np.searchsorted(fia, si, "left")); hi = int(np.searchsorted(fia, rev_end, "right"))

        done = False
        for p in range(hi - 1, lo - 1, -1):
            fvg = fvgs[p]
            if fvg["direction"] != tdir or fvg["width"] < min_fvg:
                continue
            fi = fvg["form_idx"]
            for di in range(fi + 1, min(fi + DIS_DELAY, len(idx) - 1) + 1):
                inv = c[di] > fvg["top"] if fvg["direction"] == "bearish" else c[di] < fvg["bottom"]
                if not inv:
                    continue

                rng = h[di] - l[di]
                # strong-close filter on the inversion candle
                if max_close_loc < 1.0 and rng > 0:
                    if bias == "bullish":
                        if (c[di] - l[di]) / rng < 1.0 - max_close_loc: break
                    else:
                        if (h[di] - c[di]) / rng < 1.0 - max_close_loc: break

                too_big = rng > DIS_MULT * fvg["width"]
                if entry_mode == "CLOSE":
                    use_retrace = False
                elif entry_mode == "RETRACE":
                    use_retrace = True
                else:   # RULE
                    use_retrace = too_big

                buf = fvg["width"] * SL_BUF
                sweep_ext = l[si:di + 1].min() if bias == "bullish" else h[si:di + 1].max()

                if not use_retrace:
                    ei = di; entry = c[di]
                else:
                    rend = min(di + RETR_WINDOW, len(idx) - 1); ei = None
                    for rj in range(di + 1, rend + 1):
                        if bias == "bullish" and l[rj] <= fvg["top"]: ei = rj; entry = fvg["top"]; break
                        if bias == "bearish" and h[rj] >= fvg["bottom"]: ei = rj; entry = fvg["bottom"]; break
                    if ei is None:
                        break

                # session + news on entry
                if not (s0 <= hrs[ei] < s1): break
                if news_block and idx[ei].date() in news_set: break

                open_until = [t for t in open_until if t > idx[ei]]
                if len(open_until) >= 1:
                    done = True; break

                if sl_mode == "IFVG_EDGE":
                    sl = (fvg["bottom"] - buf) if bias == "bullish" else (fvg["top"] + buf)
                elif sl_mode == "DISP":
                    disp = fi - 1
                    sl = (l[disp] - buf) if bias == "bullish" else (h[disp] + buf)
                else:
                    sl = (sweep_ext - buf) if bias == "bullish" else (sweep_ext + buf)

                if bias == "bullish":
                    direction = "long"; risk = entry - sl; tp = entry + rr * risk
                else:
                    direction = "short"; risk = sl - entry; tp = entry - rr * risk
                if risk <= 0:
                    break

                outcome, xt, pr = simulate(h, l, idx, ei, direction, entry, sl, tp, rr)
                open_until.append(pd.Timestamp(xt))
                if outcome == "win": wins += 1
                elif outcome == "loss": losses += 1
                if collect:
                    rec.append({"symbol": ctx["symbol"], "entry_time": idx[ei],
                                "direction": direction, "outcome": outcome, "pnl_r": pr})
                used.add(sig); done = True; break
            if done:
                break
    if collect:
        return wins, losses, rec
    return wins, losses


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    print("Precomputing per-symbol context...")
    ctxs = [build_ctx(s) for s in MAJORS]

    grid = {
        "sl_mode":       ["SWEEP_EXT", "IFVG_EDGE", "DISP"],
        "rr":            [1.0, 1.5, 2.0, 3.0],
        "min_fvg_pips":  [1, 2, 4],
        "max_close_loc": [1.0, 0.5, 0.3],
        "session":       [(0, 24), (7, 16), (12, 18)],
        "news_block":    [False, True],
        "htf_cisd_req":  [False, True],
        "entry_mode":    ["RULE"],   # keep the 1.5x fundamental
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"Testing {len(combos)} parameter combos across {len(MAJORS)} majors...\n")

    rows = []
    for ci, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        W = L = 0
        for ctx in ctxs:
            w, l = run_combo(ctx, **params)
            W += w; L += l
        n = W + L
        if n == 0:
            continue
        wr = W / n * 100
        # expectancy at this RR (win=+rr, loss=-1)
        exp = (W * params["rr"] - L) / n
        rows.append({**params, "N": n, "WR": wr, "expR": exp, "totalR": W * params["rr"] - L})
        if (ci + 1) % 50 == 0:
            print(f"  ...{ci+1}/{len(combos)} done")

    res = pd.DataFrame(rows)
    OUT = os.path.join("strategies", "sweep_15m_ifvg")
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, "tuning_grid.csv"), index=False)

    def show(title, df):
        print("\n" + "=" * 118)
        print(title)
        print("=" * 118)
        print(f"  {'WR':>6} {'N':>5} {'expR':>7} {'totR':>7}  {'SL':10} {'RR':>4} "
              f"{'minFVG':>6} {'close':>5} {'session':>9} {'news':>5} {'htfCISD':>7} {'entry':>7}")
        for _, r in df.iterrows():
            print(f"  {r['WR']:5.1f}% {int(r['N']):5d} {r['expR']:+6.3f} {r['totalR']:+6.0f}  "
                  f"{r['sl_mode']:10} {r['rr']:4.2f} {int(r['min_fvg_pips']):6d} "
                  f"{r['max_close_loc']:5.2f} {str(r['session']):>9} {str(r['news_block']):>5} "
                  f"{str(r['htf_cisd_req']):>7} {r['entry_mode']:>7}")

    # Rank by WR with trade-count floor
    elig = res[res["N"] >= MIN_TRADES].copy()
    show(f"TOP 15 BY WIN RATE  (min {MIN_TRADES} trades)", elig.sort_values("WR", ascending=False).head(15))
    # Also best expectancy and best total R for context
    show("TOP 10 BY EXPECTANCY (R/trade)", elig.sort_values("expR", ascending=False).head(10))
    show("TOP 10 BY TOTAL R", elig.sort_values("totalR", ascending=False).head(10))

    # Absolute highest WR ignoring floor (to show the tradeoff)
    show("TOP 8 BY WIN RATE (no trade floor — watch small N)", res.sort_values("WR", ascending=False).head(8))

    print(f"\nFull grid saved → {OUT}/tuning_grid.csv  ({len(res)} rows)")

    # ── Detailed report on the best WR config (eligible) ──
    best = elig.sort_values("WR", ascending=False).iloc[0]
    bp = {k: best[k] for k in keys}
    print("\n" + "=" * 90)
    print("DETAILED — best win-rate config:")
    print(f"  {bp}")
    print("=" * 90)
    allrec = []
    for ctx in ctxs:
        _, _, rec = run_combo(ctx, **bp, collect=True)
        allrec += rec
    d = pd.DataFrame(allrec).sort_values("entry_time").reset_index(drop=True)
    d.to_csv(os.path.join(OUT, "best_wr_trades.csv"), index=False)

    # streaks + equity
    bal = 10000.0; eq = [bal]; cl = cw = bcl = bcw = 0
    for _, r in d.iterrows():
        bal += bal * 0.01 * r["pnl_r"]; eq.append(bal)
        if r["outcome"] == "loss": cl += 1; cw = 0; bcl = max(bcl, cl)
        elif r["outcome"] == "win": cw += 1; cl = 0; bcw = max(bcw, cw)
    eqa = np.array(eq); dd = ((eqa - np.maximum.accumulate(eqa)) / np.maximum.accumulate(eqa) * 100).min()
    n = len(d); wins = (d["outcome"] == "win").sum()
    gw = wins * bp["rr"]; gl = (n - wins)
    print(f"  Trades {n} | WR {wins/n*100:.1f}% | PF {gw/gl:.2f} | "
          f"Max DD {dd:.1f}% | WinStreak {bcw} | LossStreak {bcl} | $10k→${bal:,.0f}")
    print("  Per-year:")
    d["year"] = pd.to_datetime(d["entry_time"]).dt.year
    for yr, g in d.groupby("year"):
        wr = (g["outcome"] == "win").sum() / len(g) * 100
        print(f"    {yr}: N={len(g):3d}  WR={wr:5.1f}%")
    print("  Per-pair:")
    for sym, g in d.groupby("symbol"):
        wr = (g["outcome"] == "win").sum() / len(g) * 100
        print(f"    {sym:8}: N={len(g):3d}  WR={wr:5.1f}%")
    try:
        import plot_utils as pu
        pu.plot_equity(d, os.path.join(OUT, "best_wr_equity.png"),
                       f"Best WR: {bp['sl_mode']} RR{bp['rr']} minFVG{bp['min_fvg_pips']} "
                       f"close{bp['max_close_loc']} HTFcisd{bp['htf_cisd_req']}")
        print(f"  equity → {OUT}/best_wr_equity.png")
    except Exception as e:
        print(f"  (equity plot skipped: {e})")

    mt5.shutdown()


if __name__ == "__main__":
    main()
