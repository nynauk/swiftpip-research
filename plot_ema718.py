"""
EMA 7/18 final config — equity curve + sample trades.
Config: EMA 7/18, FVG window 24, retest 48, SL swing+0.25xFVG, DOL floor 2.0R,
RSI 45/55, news ON, most-recent-cross, 17 instruments (no metals). Realistic,
trade-by-trade, flat $1,000 / $100k.
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
import news_filter as nf
import ema_dol_strategy as ed
import realistic_ema_dol_rsi as edr
from plot_ema_dol_trade import plot_trade

FAST, SLOW = 7, 18
RSI_BAND = (45, 100, 0, 55)
EXTRA_SPEC = {"UK100.cash": (1.2, 0.2, 2.0), "US100.cash": (2.0, 0.4, 5.0),
              "USOIL.cash": (0.04, 0.01, 0.08)}
SPEC = {**edr.SPEC, **EXTRA_SPEC}
NO_METALS = [s for s in edr.SYMBOLS if s not in ("XAUUSD", "XAGUSD")]
UNIVERSE = NO_METALS + ["UK100.cash", "US100.cash", "USOIL.cash"]
OUT = os.path.join("strategies", "ema718_final")


def collect(df, symbol):
    spread, slippage, MIN_W = SPEC[symbol]
    h = df["high"].values; l = df["low"].values; c = df["close"].values; idx = df.index
    ema_f = pd.Series(c, index=idx).ewm(span=FAST, adjust=False).mean().values
    ema_s = pd.Series(c, index=idx).ewm(span=SLOW, adjust=False).mean().values
    above = ema_f > ema_s
    N = len(df)
    cu = np.zeros(N, bool); cd = np.zeros(N, bool)
    cu[1:] = above[1:] & ~above[:-1]; cd[1:] = ~above[1:] & above[:-1]
    nb = np.full(N + 1, N, int); nbr = np.full(N + 1, N, int)
    for k in range(N - 1, -1, -1):
        nb[k] = k if cu[k] else nb[k + 1]; nbr[k] = k if cd[k] else nbr[k + 1]
    sh, sl_arr = ed.find_swings_arr(h, l, n=ed.SWING_LOOKBACK)
    pdh_map, pdl_map = ed.build_pdh_pdl(df)
    rsi_arr = edr.rsi(c)
    news_set = nf.build_news_block_set(idx[0].date(), idx[-1].date())
    trades = []; open_until = []
    for i in range(N):
        if not (cu[i] or cd[i]): continue
        bias = "long" if cu[i] else "short"
        tdir = "bullish" if bias == "long" else "bearish"
        opp = nbr[i + 1] if bias == "long" else nb[i + 1]
        end_idx = min(i + ed.FVG_LOOKUP_WINDOW, opp - 1, N - 1)
        if end_idx < i + 2: continue
        fvg = ed.find_aligned_fvg(h, l, i, end_idx, tdir, MIN_W)
        if fvg is None: continue
        rt_end = min(fvg["form_idx"] + ed.RETEST_WINDOW, opp - 1, N - 1)
        ei = None
        for k in range(fvg["form_idx"] + 1, rt_end + 1):
            if bias == "long" and l[k] <= fvg["top"]:    ei = k; break
            if bias == "short" and h[k] >= fvg["bottom"]: ei = k; break
        if ei is None: continue
        open_until = [t for t in open_until if t > idx[ei]]
        if len(open_until) >= 1: continue
        if idx[ei].date() in news_set: continue
        rv = rsi_arr[ei]
        if np.isnan(rv): continue
        llo, lhi, slo, shi = RSI_BAND
        if bias == "long" and not (llo <= rv <= lhi): continue
        if bias == "short" and not (slo <= rv <= shi): continue
        d_today = pd.Timestamp(idx[ei]).normalize()
        pdh_t = pdh_map.get(d_today); pdl_t = pdl_map.get(d_today)
        buf = fvg["width"] * ed.SL_BUFFER_FRAC
        if bias == "long":
            entry = fvg["top"] + spread + slippage
            swlow = ed.find_recent_swing_low(sl_arr, ei, lookback=80)
            anchor = swlow if swlow is not None and swlow < fvg["bottom"] else fvg["bottom"]
            sl = anchor - buf; risk = entry - sl
            if risk <= 0: continue
            dol = ed.next_liquidity_above(entry, sh, ei, pdh_t)
            if dol is None: continue
            tp = dol; rr = (tp - entry) / risk; direction = "long"
        else:
            entry = fvg["bottom"] - slippage
            swhigh = ed.find_recent_swing_high(sh, ei, lookback=80)
            anchor = swhigh if swhigh is not None and swhigh > fvg["top"] else fvg["top"]
            sl = anchor + buf; risk = sl - entry
            if risk <= 0: continue
            dol = ed.next_liquidity_below(entry, sl_arr, ei, pdl_t)
            if dol is None: continue
            tp = dol; rr = (entry - tp) / risk; direction = "short"
        if rr < ed.MIN_RR_TO_TAKE or rr > ed.MAX_RR_CAP: continue
        outcome, exit_i = "open", min(ei + 400, N - 1)
        for j in range(ei + 1, min(ei + 400, N)):
            if direction == "long":
                if l[j] <= sl: outcome, exit_i = "loss", j; break
                if h[j] >= tp: outcome, exit_i = "win", j; break
            else:
                if h[j] + spread >= sl: outcome, exit_i = "loss", j; break
                if l[j] + spread <= tp: outcome, exit_i = "win", j; break
        open_until.append(idx[exit_i])
        pnl = rr if outcome == "win" else (-1.0 if outcome == "loss" else 0.0)
        trades.append(dict(symbol=symbol, cross_i=i, fvg_i=fvg["form_idx"], fvg_top=fvg["top"],
                           fvg_bottom=fvg["bottom"], entry_i=ei, entry=entry, sl=sl, tp=tp, rr=rr,
                           direction=direction, rsi=float(rv), exit_i=exit_i, outcome=outcome,
                           entry_time=idx[ei], exit_time=idx[exit_i], pnl_r=pnl))
    return trades, ema_f, ema_s, rsi_arr


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    os.makedirs(OUT, exist_ok=True)
    data = {}; alltr = []
    for s in UNIVERSE:
        df = bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
        tr, ef, es, rs = collect(df, s)
        data[s] = (df, ef, es, rs)
        alltr += tr

    tdf = pd.DataFrame(alltr).sort_values("entry_time").reset_index(drop=True)
    st = edr.stats(tdf)
    weeks = (pd.to_datetime(tdf["entry_time"]).max() - pd.to_datetime(tdf["entry_time"]).min()).days / 7
    print(f"EMA 7/18 · N={st['n']} ({st['n']/weeks:.2f}/wk) · WR {st['wr']:.1f}% · PF {st['pf']:.2f} "
          f"· {st['tot']:+.1f}R · DD {st['dd']:.1f}% · LS {st['ls']} · end ${st['bal']:,.0f}")

    # ── Equity curve (time axis, fixed $) ──
    BG, FG = "#0d1117", "#e6edf3"
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(st["times"], st["eqa"], color="#3fb950", linewidth=1.7, drawstyle="steps-post")
    ax.axhline(edr.START_BAL, color="#444c56", linestyle="--", linewidth=0.8)
    ax.axhline(edr.START_BAL * 0.90, color="#f85149", linestyle=":", linewidth=1.0,
               label="FTMO -10% floor")
    ax.set_title(f"EMA 7/18 final · 17 instruments · realistic · flat $1k/$100k · "
                 f"PF {st['pf']:.2f} · DD {st['dd']:.1f}% · end ${st['bal']:,.0f}",
                 color=FG, fontsize=12, fontweight="bold")
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
    ax.set_xlabel("Date", color=FG); ax.set_ylabel("Balance ($)", color=FG)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "equity_ema718.png"), dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    # ── Sample trades: biggest win, a typical win, a loss (prefer FX for readability) ──
    fx = [t for t in alltr if "." not in t["symbol"]]
    wins = [t for t in fx if t["outcome"] == "win"]
    losses = [t for t in fx if t["outcome"] == "loss"]
    biggest = max(wins, key=lambda t: t["rr"])
    typ = sorted([t for t in wins if 2.5 <= t["rr"] <= 4.0 and t["cross_i"] > 60
                  and 10 <= (t["exit_i"] - t["cross_i"]) <= 80], key=lambda t: abs(t["rr"] - 3.0))
    typical = typ[len(typ) // 3] if typ else biggest
    losc = sorted([t for t in losses if t["cross_i"] > 60 and 8 <= (t["exit_i"] - t["cross_i"]) <= 120],
                  key=lambda t: abs(t["rr"] - 3.0))
    a_loss = losc[len(losc) // 2] if losc else losses[0]

    for t, name in [(biggest, "sample_biggest_win.png"), (typical, "sample_typical_win.png"),
                    (a_loss, "sample_loss.png")]:
        df, ef, es, rs = data[t["symbol"]]
        plot_trade(t, df, ef, es, rs, os.path.join(OUT, name), fast=FAST, slow=SLOW)
        print(f"  {name}: {t['symbol']} {t['direction']} {t['rr']:.1f}R -> {t['outcome']} "
              f"(RSI {t['rsi']:.0f}, {df.index[t['entry_i']].date()})")
    print(f"Saved -> {OUT}/")
    mt5.shutdown()


if __name__ == "__main__":
    main()
