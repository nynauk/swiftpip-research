"""
Increase trade frequency toward ~2/week while holding the edge.
Primary lever: EMA speed (faster cross = more setups). Also expands the universe
to 17 instruments (14 no-metals + UK100 + US100 + USOIL) and offers an RSI-band
frequency dial. Realistic, trade-by-trade, flat $1,000 / $100k.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import MetaTrader5 as mt5

import main as bt
import ema_dol_strategy as ed
import realistic_ema_dol_rsi as edr

# per-instrument (spread, slippage, min_fvg_width)
EXTRA_SPEC = {
    "UK100.cash": (1.2, 0.2, 2.0),
    "US100.cash": (2.0, 0.4, 5.0),
    "USOIL.cash": (0.04, 0.01, 0.08),
}
SPEC = {**edr.SPEC, **EXTRA_SPEC}

NO_METALS = [s for s in edr.SYMBOLS if s not in ("XAUUSD", "XAGUSD")]
UNIVERSE = NO_METALS + ["UK100.cash", "US100.cash", "USOIL.cash"]   # 17 instruments

BASEP = dict(rsi_band=(45, 100, 0, 55), fvg_window=24, retest_window=48,
             sl_buf=0.25, min_rr=2.0, max_rr=10.0, news_on=True)
OUT = os.path.join("strategies", "finetune_ema_dol")


def precompute_base(symbol, df):
    h = df["high"].values; l = df["low"].values; c = df["close"].values
    idx = df.index
    sh, sl_arr = ed.find_swings_arr(h, l, n=ed.SWING_LOOKBACK)
    pdh_map, pdl_map = ed.build_pdh_pdl(df)
    news_set = nf_set(idx)
    return dict(h=h, l=l, c=c, idx=idx, sh=sh, sl_arr=sl_arr,
                pdh=pdh_map, pdl=pdl_map, news=news_set, rsi=edr.rsi(c))


def nf_set(idx):
    import news_filter as nf
    return nf.build_news_block_set(idx[0].date(), idx[-1].date())


def cross_arrays(c, idx, fast, slow):
    ema_f = pd.Series(c, index=idx).ewm(span=fast, adjust=False).mean().values
    ema_s = pd.Series(c, index=idx).ewm(span=slow, adjust=False).mean().values
    above = ema_f > ema_s
    n = len(c)
    cu = np.zeros(n, bool); cd = np.zeros(n, bool)
    cu[1:] = above[1:] & ~above[:-1]; cd[1:] = ~above[1:] & above[:-1]
    nb = np.full(n + 1, n, int); nbr = np.full(n + 1, n, int)
    for k in range(n - 1, -1, -1):
        nb[k] = k if cu[k] else nb[k + 1]
        nbr[k] = k if cd[k] else nbr[k + 1]
    return cu, cd, nb, nbr


def run_freq(symbol, base, ema, p):
    long_lo, long_hi, short_lo, short_hi = p["rsi_band"]
    spread, slippage, min_w = SPEC[symbol]
    h, l, c, idx = base["h"], base["l"], base["c"], base["idx"]
    sh, sl_arr = base["sh"], base["sl_arr"]
    pdh_map, pdl_map, rsi_arr = base["pdh"], base["pdl"], base["rsi"]
    news_set = base["news"] if p["news_on"] else set()
    cu, cd, nb, nbr = cross_arrays(c, idx, ema[0], ema[1])
    N = len(idx)
    FVGW, RTW, SLB, MINRR, MAXRR = (p["fvg_window"], p["retest_window"],
                                    p["sl_buf"], p["min_rr"], p["max_rr"])

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
        if bias == "long" and not (long_lo <= rv <= long_hi):
            continue
        if bias == "short" and not (short_lo <= rv <= short_hi):
            continue
        d_today = pd.Timestamp(idx[ei]).normalize()
        pdh_t = pdh_map.get(d_today); pdl_t = pdl_map.get(d_today)
        buf = fvg["width"] * SLB
        pad = edr.sl_pad(symbol, spread)                  # spread+1pip stop buffer
        if bias == "long":
            entry = fvg["top"] + spread + slippage
            swlow = ed.find_recent_swing_low(sl_arr, ei, lookback=80)
            anchor = swlow if swlow is not None and swlow < fvg["bottom"] else fvg["bottom"]
            sl = anchor - buf - pad; risk = entry - sl
            if risk <= 0: continue
            dol = ed.next_liquidity_above(entry, sh, ei, pdh_t)
            if dol is None: continue
            tp = dol; rr = (tp - entry) / risk; direction = "long"
        else:
            entry = fvg["bottom"] - slippage
            swhigh = ed.find_recent_swing_high(sh, ei, lookback=80)
            anchor = swhigh if swhigh is not None and swhigh > fvg["top"] else fvg["top"]
            sl = anchor + buf + pad; risk = sl - entry
            if risk <= 0: continue
            dol = ed.next_liquidity_below(entry, sl_arr, ei, pdl_t)
            if dol is None: continue
            tp = dol; rr = (entry - tp) / risk; direction = "short"
        if rr < MINRR or rr > MAXRR:
            continue
        oc, xt, pr = edr.sim_realistic(h, l, idx, ei, direction, entry, sl, tp, rr, spread)
        open_until.append(pd.Timestamp(xt))
        trades.append(dict(symbol=symbol, entry_time=idx[ei], direction=direction,
                           outcome=oc, exit_time=xt, pnl_r=pr))
    return pd.DataFrame(trades)


def combine(lst):
    return pd.concat(lst, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def per_week(df):
    d = pd.to_datetime(df["entry_time"])
    weeks = max((d.max() - d.min()).days / 7.0, 1)
    return len(df) / weeks


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    bases = {}
    for s in UNIVERSE:
        df = bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
        bases[s] = precompute_base(s, df)

    print("=" * 100)
    print(f"FREQUENCY TUNING · {len(UNIVERSE)} instruments (no metals + UK100/US100/USOIL) · "
          f"realistic · flat $1k/$100k")
    print("target ~2 trades/week; hold RSI 45/55, FVG24, news ON, most-recent-cross")
    print("=" * 100)

    def show(label, df):
        st = edr.stats(df)
        if st is None:
            print(f"  {label:22} no trades"); return None
        tpw = per_week(df)
        print(f"  {label:22} {st['n']:>5} {tpw:>5.2f}/wk {st['wr']:>5.1f}% "
              f"{st['pf']:>4.2f} {st['tot']:>+7.1f}R {st['dd']:>6.1f}% LS{st['ls']:>3} ${st['bal']:>9,.0f}")
        return st

    print(f"\n  === EMA speed sweep (fvg24, rr2.0, RSI 45/55) ===")
    print(f"  {'EMA fast/slow':22} {'N':>5} {'/wk':>6} {'WR':>6} {'PF':>4} {'R':>8} {'DD':>7} {'streak':>5} {'End$':>10}")
    ema_grid = [(9, 21), (8, 21), (7, 18), (6, 15), (5, 13), (5, 10)]
    keep = {}
    for ema in ema_grid:
        df = combine([run_freq(s, bases[s], ema, BASEP) for s in UNIVERSE])
        keep[ema] = df
        show(f"EMA {ema[0]}/{ema[1]}", df)

    # RSI band dial at a chosen fast EMA (5/13) to add more if needed
    print(f"\n  === RSI band dial at EMA 5/13 (loosen for more trades) ===")
    for band, name in [((45, 100, 0, 55), "45/55"), ((42, 100, 0, 58), "42/58"),
                       ((40, 100, 0, 60), "40/60"), ((0, 100, 0, 100), "none")]:
        p = dict(BASEP); p["rsi_band"] = band
        df = combine([run_freq(s, bases[s], (5, 13), p) for s in UNIVERSE])
        show(f"RSI {name}", df)

    # Year-by-year of a good ~2/week candidate: EMA 6/15, RSI 45/55
    cand = combine([run_freq(s, bases[s], (6, 15), BASEP) for s in UNIVERSE])
    print(f"\n  === candidate EMA 6/15, RSI 45/55 — year by year ===")
    cand["yr"] = pd.to_datetime(cand["entry_time"]).dt.year
    for yr in sorted(cand["yr"].unique()):
        st = edr.stats(cand[cand["yr"] == yr])
        if st:
            print(f"    {yr}: N={st['n']:>4} PF {st['pf']:>4.2f} {st['tot']:>+6.1f}R "
                  f"DD {st['dd']:>5.1f}% LS{st['ls']}")

    # equity: base 9/21 vs faster candidates
    BG, FG = "#0d1117", "#e6edf3"
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=BG); ax.set_facecolor(BG)
    cols = {(9, 21): "#7d8590", (6, 15): "#3fb950", (5, 13): "#58a6ff"}
    for ema, col in cols.items():
        st = edr.stats(keep.get(ema, cand if ema == (6, 15) else None) if ema in keep else cand)
        df = keep[ema] if ema in keep else cand
        st = edr.stats(df)
        ax.plot(st["times"], st["eqa"], color=col, linewidth=1.6, drawstyle="steps-post",
                label=f"EMA {ema[0]}/{ema[1]}: {per_week(df):.2f}/wk, PF {st['pf']:.2f}, "
                      f"DD {st['dd']:.1f}%, ${st['bal']:,.0f}")
    ax.axhline(edr.START_BAL, color="#444c56", linestyle="--", linewidth=0.8)
    ax.set_title(f"Frequency tuning · EMA speed · {len(UNIVERSE)} instruments · realistic",
                 color=FG, fontsize=12, fontweight="bold")
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
    ax.set_xlabel("Date", color=FG); ax.set_ylabel("Balance ($)", color=FG)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "frequency_tune.png"), dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nSaved -> {OUT}/frequency_tune.png")
    mt5.shutdown()


if __name__ == "__main__":
    main()
