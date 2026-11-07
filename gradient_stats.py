"""
Deep stats on the gradient-exit config (EMA 7/18, DOL>=2R filter, RSI 45/55,
gradient exit lb=1 profit-only, 17 instruments, realistic, $1k/$100k).
Captures the EXIT REASON per trade (stop / gradient / cross-fallback) plus the
full distribution, long/short, per-year, per-instrument, and monthly breakdowns.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MetaTrader5 as mt5

import main as bt
import ema_dol_strategy as ed
import realistic_ema_dol_rsi as edr
import frequency_tune as fq

EMA = (7, 18)
RISK_USD = 1000.0
OUT = os.path.join("strategies", "ema718_final")


def collect(symbol, base):
    spread, slippage, min_w = fq.SPEC[symbol]
    h, l, c, idx = base["h"], base["l"], base["c"], base["idx"]
    sh, sl_arr = base["sh"], base["sl_arr"]
    pdh_map, pdl_map, rsi_arr = base["pdh"], base["pdl"], base["rsi"]
    news_set = base["news"]
    ef = pd.Series(c, index=idx).ewm(span=EMA[0], adjust=False).mean().values
    cu, cd, nb, nbr = fq.cross_arrays(c, idx, EMA[0], EMA[1])
    N = len(idx)
    p = fq.BASEP

    trades = []; open_until = []
    for i in range(N):
        if not (cu[i] or cd[i]):
            continue
        bias = "long" if cu[i] else "short"
        tdir = "bullish" if bias == "long" else "bearish"
        opp = nbr[i + 1] if bias == "long" else nb[i + 1]
        end_idx = min(i + p["fvg_window"], opp - 1, N - 1)
        if end_idx < i + 2:
            continue
        fvg = ed.find_aligned_fvg(h, l, i, end_idx, tdir, min_w)
        if fvg is None:
            continue
        rt_end = min(fvg["form_idx"] + p["retest_window"], opp - 1, N - 1)
        ei = None
        for k in range(fvg["form_idx"] + 1, rt_end + 1):
            if bias == "long" and l[k] <= fvg["top"]:    ei = k; break
            if bias == "short" and h[k] >= fvg["bottom"]: ei = k; break
        if ei is None:
            continue
        open_until = [t for t in open_until if t > idx[ei]]
        if len(open_until) >= 1:
            continue
        if idx[ei].date() in news_set:
            continue
        rv = rsi_arr[ei]
        if np.isnan(rv):
            continue
        if bias == "long" and not (45 <= rv <= 100):
            continue
        if bias == "short" and not (0 <= rv <= 55):
            continue
        d_today = pd.Timestamp(idx[ei]).normalize()
        pdh_t = pdh_map.get(d_today); pdl_t = pdl_map.get(d_today)
        buf = fvg["width"] * p["sl_buf"]
        if bias == "long":
            entry = fvg["top"] + spread + slippage
            swlow = ed.find_recent_swing_low(sl_arr, ei, lookback=80)
            anchor = swlow if swlow is not None and swlow < fvg["bottom"] else fvg["bottom"]
            sl = anchor - buf; risk = entry - sl; direction = "long"
            if risk <= 0: continue
            dol = ed.next_liquidity_above(entry, sh, ei, pdh_t)
            if dol is None or not (2.0 <= (dol - entry) / risk <= 10.0): continue
        else:
            entry = fvg["bottom"] - slippage
            swhigh = ed.find_recent_swing_high(sh, ei, lookback=80)
            anchor = swhigh if swhigh is not None and swhigh > fvg["top"] else fvg["top"]
            sl = anchor + buf; risk = sl - entry; direction = "short"
            if risk <= 0: continue
            dol = ed.next_liquidity_below(entry, sl_arr, ei, pdl_t)
            if dol is None or not (2.0 <= (entry - dol) / risk <= 10.0): continue

        reason, xj, pr = _sim(h, l, c, ef, ei, direction, entry, sl, risk, opp, spread)
        open_until.append(pd.Timestamp(idx[xj]))
        trades.append(dict(symbol=symbol, dir=direction, entry_time=idx[ei], exit_time=idx[xj],
                           pnl_r=pr, reason=reason, hold_h=(idx[xj] - idx[ei]).total_seconds() / 3600))
    return trades


def _sim(h, l, c, ef, ei, direction, entry, sl, risk, opp, spread, max_bars=2000):
    end = min(opp, ei + max_bars, len(h) - 1)
    for j in range(ei + 1, end + 1):
        if direction == "long":
            if l[j] <= sl: return "stop", j, (sl - entry) / risk
        else:
            if h[j] + spread >= sl: return "stop", j, (entry - sl) / risk
        if j - 1 >= 0:
            grad = ef[j] - ef[j - 1]
            inp = (c[j] > entry) if direction == "long" else (c[j] + spread < entry)
            if inp and ((direction == "long" and grad < 0) or (direction == "short" and grad > 0)):
                px = c[j] if direction == "long" else c[j] + spread
                return "gradient", j, ((px - entry) if direction == "long" else (entry - px)) / risk
        if j == opp:
            px = c[j] if direction == "long" else c[j] + spread
            return "cross", j, ((px - entry) if direction == "long" else (entry - px)) / risk
    j = end; px = c[j] if direction == "long" else c[j] + spread
    return "end", j, ((px - entry) if direction == "long" else (entry - px)) / risk


def block(df, title):
    n = len(df)
    if n == 0:
        return f"  {title:22} -"
    w = (df.pnl_r > 0.01).sum(); ls = (df.pnl_r < -0.01).sum()
    wr = w / n * 100
    gw = df.loc[df.pnl_r > 0, "pnl_r"].sum(); gl = -df.loc[df.pnl_r < 0, "pnl_r"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    return (f"  {title:22} {n:>5} {wr:>5.1f}% {pf:>5.2f} {df.pnl_r.sum():>+8.1f}R "
            f"{df.pnl_r.mean():>+5.2f}R")


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    def load(s):
        pth = os.path.join("data_cache", f"{s}_M15.pkl")
        try:
            return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
        except Exception:
            if os.path.exists(pth): os.remove(pth)
            return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")

    allt = []
    for s in fq.UNIVERSE:
        allt += collect(s, fq.precompute_base(s, load(s)))
    df = pd.DataFrame(allt).sort_values("entry_time").reset_index(drop=True)
    r = df["pnl_r"]

    # streaks + DD + equity ($1k/$100k)
    bal = 100000.0; eq = [bal]; cl = cw = bcl = bcw = 0
    for x in r:
        bal += x * RISK_USD; eq.append(bal)
        if x < -0.01: cl += 1; cw = 0; bcl = max(bcl, cl)
        elif x > 0.01: cw += 1; cl = 0; bcw = max(bcw, cw)
    eqa = np.array(eq)
    dd = ((eqa - np.maximum.accumulate(eqa)) / np.maximum.accumulate(eqa) * 100).min()
    dd_usd = (eqa - np.maximum.accumulate(eqa)).min()

    d0, d1 = pd.to_datetime(df.entry_time).min(), pd.to_datetime(df.entry_time).max()
    weeks = (d1 - d0).days / 7
    wins = r[r > 0.01]; losses = r[r < -0.01]
    n = len(df)

    print("=" * 78)
    print("GRADIENT-EXIT DEEP STATS · EMA 7/18 · 17 instruments · realistic · $1k/$100k")
    print("=" * 78)
    print(f"  Period                : {d0.date()} -> {d1.date()}  ({weeks/52:.1f} yr)")
    print(f"  Trades                : {n}   ({n/weeks:.2f}/week · {n/(weeks/4.345):.1f}/month)")
    print(f"  Win rate              : {len(wins)/n*100:.1f}%   ({len(wins)}W / {len(losses)}L / {n-len(wins)-len(losses)}BE)")
    gw, gl = wins.sum(), -losses.sum()
    print(f"  Profit factor         : {gw/gl:.2f}")
    print(f"  Total R               : {r.sum():+.1f}R   (end ${eqa[-1]:,.0f} from $100k)")
    print(f"  Expectancy / trade    : {r.mean():+.3f}R  (${r.mean()*RISK_USD:+.0f})")
    print(f"  Avg win / avg loss    : +{wins.mean():.2f}R / {losses.mean():.2f}R   (payoff {wins.mean()/abs(losses.mean()):.2f})")
    print(f"  Largest win / loss    : +{r.max():.1f}R / {r.min():.1f}R")
    print(f"  Std dev of R          : {r.std():.2f}R   (per-trade Sharpe {r.mean()/r.std():.2f}, "
          f"annualised ~{r.mean()/r.std()*np.sqrt(n/(weeks/52)):.2f})")
    print(f"  Max win streak        : {bcw}")
    print(f"  Max loss streak       : {bcl}")
    print(f"  Max drawdown          : {dd:.1f}%  (${dd_usd:,.0f})")
    print(f"  Avg hold time         : {df.hold_h.mean():.1f}h  (median {df.hold_h.median():.1f}h, max {df.hold_h.max():.0f}h)")

    print("\n  --- EXIT REASON (how the gradient exit actually closes trades) ---")
    print(f"  {'reason':22} {'N':>5} {'WR':>6} {'PF':>5} {'totR':>8} {'avgR':>6}  share")
    for rs in ["gradient", "stop", "cross", "end"]:
        sub = df[df.reason == rs]
        if len(sub):
            print(block(sub, rs) + f"   {len(sub)/n*100:4.0f}%")

    print("\n  --- LONG vs SHORT ---")
    print(f"  {'':22} {'N':>5} {'WR':>6} {'PF':>5} {'totR':>8} {'avgR':>6}")
    for d in ["long", "short"]:
        print(block(df[df.dir == d], d))

    print("\n  --- PER YEAR ---")
    print(f"  {'':22} {'N':>5} {'WR':>6} {'PF':>5} {'totR':>8} {'avgR':>6}")
    df["yr"] = pd.to_datetime(df.entry_time).dt.year
    for y in sorted(df.yr.unique()):
        print(block(df[df.yr == y], str(y)))

    GROUPS = [("FX majors", ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]),
              ("FX crosses", ["EURGBP", "EURJPY", "GBPJPY"]),
              ("Indices", ["US500.cash", "US30.cash", "GER40.cash", "UK100.cash", "US100.cash"]),
              ("Energy", ["UKOIL.cash", "USOIL.cash"])]
    print("\n  --- BY GROUP ---")
    print(f"  {'':22} {'N':>5} {'WR':>6} {'PF':>5} {'totR':>8} {'avgR':>6}")
    for gname, syms in GROUPS:
        sub = df[df.symbol.isin(syms)]
        if len(sub):
            print(block(sub, gname))

    print("\n  --- PER INSTRUMENT ---")
    print(f"  {'':22} {'N':>5} {'WR':>6} {'PF':>5} {'totR':>8} {'avgR':>6}")
    for s in fq.UNIVERSE:
        print(block(df[df.symbol == s], s))

    # monthly
    df["ym"] = pd.to_datetime(df.entry_time).dt.to_period("M")
    m = df.groupby("ym")["pnl_r"].sum()
    print("\n  --- MONTHLY ---")
    print(f"  Months traded         : {len(m)}   ({(m>0).mean()*100:.0f}% green months)")
    print(f"  Best / worst month    : +{m.max():.1f}R / {m.min():.1f}R")
    print(f"  Avg / median month    : +{m.mean():.1f}R / +{m.median():.1f}R")

    print("\n  --- R DISTRIBUTION ---")
    buckets = [(-99, -1, "≤ -1R (full stop)"), (-1, 0, "-1 to 0R"), (0, 1, "0 to +1R"),
               (1, 2, "+1 to +2R"), (2, 3, "+2 to +3R"), (3, 5, "+3 to +5R"), (5, 99, "+5R and up")]
    for lo, hi, lab in buckets:
        cnt = ((r > lo) & (r <= hi)).sum()
        print(f"  {lab:22} {cnt:>5}  {cnt/n*100:4.0f}%  " + "#" * int(cnt / n * 60))

    # charts: R histogram + monthly bars
    BG, FG = "#0d1117", "#e6edf3"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    for ax in (a1, a2):
        ax.set_facecolor(BG); ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    a1.hist(r, bins=40, color="#58a6ff", edgecolor="#0d1117")
    a1.axvline(0, color="#7d8590", linestyle="--", linewidth=0.8)
    a1.set_title("R distribution per trade", color=FG, fontweight="bold")
    a1.set_xlabel("R multiple", color=FG); a1.set_ylabel("trades", color=FG)
    mc = ["#3fb950" if v > 0 else "#f85149" for v in m.values]
    a2.bar(range(len(m)), m.values, color=mc)
    a2.axhline(0, color="#7d8590", linewidth=0.8)
    a2.set_title("Monthly R", color=FG, fontweight="bold")
    a2.set_xlabel("month #", color=FG); a2.set_ylabel("R", color=FG)
    fig.suptitle("Gradient-exit stats · EMA 7/18 · realistic", color=FG, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "gradient_stats.png"), dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nSaved -> {OUT}/gradient_stats.png")
    mt5.shutdown()


if __name__ == "__main__":
    main()
