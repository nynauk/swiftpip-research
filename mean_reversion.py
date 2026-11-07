"""
RSI-2 mean-reversion (Connors-style pullback fade), realistic, 17 instruments.
  Trend filter : trade only WITH the 200-SMA (long above, short below)
  Entry        : fade the short-term extreme — long when RSI(2) < LO, short when > HI
  Stop         : FIXED at entry = STOP_ATR x ATR(14)  (never moves)
  Exit         : RSI(2) reverts through 50, OR stop, OR max-hold
  Spread-aware : long fills at ask, short at bid; SHORT triggers vs ask.
Then measures the DAILY-RETURN correlation vs the deployed gradient bot to see if
it's a genuine de-correlated 2nd engine.
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
import frequency_tune as fq
import test_cross_exit as tc
import test_gradient_exit as tg
import realistic_ema_dol_rsi as edr

OUT = os.path.join("strategies", "mean_reversion")
RISK_USD = 1000.0
SMA_N, RSI_N, ATR_N = 200, 2, 14
RSI_LO, RSI_HI = 10.0, 90.0       # entry extremes
EXIT_RSI = 60.0                    # revert-to-mean exit (RSI2 back >= 60 long / <= 40 short)
STOP_ATR = 2.5                     # fixed stop = 2.5 x ATR
MAX_HOLD = 10                      # bars time stop (on the trade timeframe)
TF = "D1"                          # RSI-2 timeframe (classic Connors = daily + 200-day SMA)


def resample(df, rule):
    o = df.resample(rule).agg({"open": "first", "high": "max",
                               "low": "min", "close": "last"}).dropna()
    return o


def rsi(c, n):
    s = pd.Series(c); d = s.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return (100 - 100 / (1 + ag / al.replace(0, np.nan))).values


def atr(h, l, c, n):
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().values


def run_mr(symbol, df, lo=RSI_LO, hi=RSI_HI, exit_rsi=EXIT_RSI, stop_atr=STOP_ATR):
    spread, slippage, _ = fq.SPEC[symbol]
    h = df["high"].values; l = df["low"].values; c = df["close"].values; idx = df.index
    sma = pd.Series(c).rolling(SMA_N).mean().values
    r2 = rsi(c, RSI_N); a = atr(h, l, c, ATR_N)
    N = len(df)
    trades = []; busy_until = -1
    for i in range(SMA_N + ATR_N, N - 1):
        if i <= busy_until:
            continue
        if np.isnan(sma[i]) or np.isnan(r2[i]) or np.isnan(a[i]) or a[i] <= 0:
            continue
        long_sig = c[i] > sma[i] and r2[i] < lo
        short_sig = c[i] < sma[i] and r2[i] > hi
        if not (long_sig or short_sig):
            continue
        risk = stop_atr * a[i] + edr.sl_pad(symbol, spread)   # ATR stop + spread+1pip buffer
        if risk < 3 * spread:                       # spread would dominate — skip
            continue
        if long_sig:
            entry = c[i] + spread + slippage; sl = entry - risk; direction = "long"
        else:
            entry = c[i] - slippage; sl = entry + risk; direction = "short"
        oc, xj, pr = _sim(h, l, c, r2, i, direction, entry, sl, risk, spread, exit_rsi)
        busy_until = xj
        trades.append(dict(symbol=symbol, entry_time=idx[i], exit_time=idx[xj],
                           direction=direction, outcome=oc, pnl_r=pr,
                           hold_h=(idx[xj] - idx[i]).total_seconds() / 3600))
    return pd.DataFrame(trades)


def _sim(h, l, c, r2, ei, direction, entry, sl, risk, spread, exit_rsi):
    end = min(ei + MAX_HOLD, len(h) - 1)
    for j in range(ei + 1, end + 1):
        if direction == "long":
            if l[j] <= sl: return "loss", j, (sl - entry) / risk
            if r2[j] >= exit_rsi:
                return _lbl((c[j] - entry) / risk), j, (c[j] - entry) / risk
        else:
            if h[j] + spread >= sl: return "loss", j, (entry - sl) / risk
            if r2[j] <= (100 - exit_rsi):
                px = c[j] + spread
                return _lbl((entry - px) / risk), j, (entry - px) / risk
    j = end
    px = c[j] if direction == "long" else c[j] + spread
    r = (px - entry) / risk if direction == "long" else (entry - px) / risk
    return _lbl(r), j, r


def _lbl(r):
    return "win" if r > 0.01 else ("loss" if r < -0.01 else "be")


def daily_pnl(df):
    d = df[df.outcome != "open"].copy()
    d["day"] = pd.to_datetime(d["exit_time"]).dt.normalize()
    return d.groupby("day")["pnl_r"].sum() * RISK_USD


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    os.makedirs(OUT, exist_ok=True)

    def load(s):
        p = os.path.join("data_cache", f"{s}_M15.pkl")
        try: return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
        except Exception:
            if os.path.exists(p): os.remove(p)
            return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")

    data = {s: load(s) for s in fq.UNIVERSE}
    bases = {s: fq.precompute_base(s, data[s]) for s in fq.UNIVERSE}

    rule = {"H1": "1h", "H4": "4h", "D1": "1D"}[TF]
    res = {s: resample(data[s], rule) for s in fq.UNIVERSE}

    # ── parameter sweep to find a profitable edge ──
    print("=" * 92)
    print(f"RSI-2 PARAMETER SWEEP ({TF}, 17 instruments, realistic)")
    print(f"  {'lo/hi':8} {'exit':>5} {'stopATR':>8} {'N':>6} {'WR':>6} {'PF':>5} {'totR':>9} {'DD':>7}")
    best = None; best_pf = 0
    for lo, hi in [(10, 90), (5, 95), (2, 98)]:
        for ex in [50, 60, 70]:
            for st in [1.5, 2.5]:
                d = tc.combine([run_mr(s, res[s], lo, hi, ex, st) for s in fq.UNIVERSE])
                stt = edr.stats(d)
                print(f"  {lo}/{hi:<4} {ex:>5} {st:>8.1f} {stt['n']:>6} {stt['wr']:>5.1f}% "
                      f"{stt['pf']:>4.2f} {stt['tot']:>+8.1f}R {stt['dd']:>6.1f}%")
                if stt['pf'] > best_pf:
                    best_pf = stt['pf']; best = (lo, hi, ex, st, d)
    lo, hi, ex, st, mr = best
    print(f"\n  >> best config: RSI {lo}/{hi}, exit {ex}, stop {st}xATR (PF {best_pf:.2f})")
    grad = tc.combine([tg.run_gradient(s, bases[s], (7, 18), fq.BASEP, 1, True) for s in fq.UNIVERSE])

    def report(name, df):
        st = edr.stats(df); cl = df[df.outcome != "open"]
        wk = (pd.to_datetime(cl.entry_time).max() - pd.to_datetime(cl.entry_time).min()).days / 7
        print(f"  {name:16} N={st['n']:>4} ({st['n']/wk:.2f}/wk)  WR {st['wr']:.1f}%  PF {st['pf']:.2f}  "
              f"{st['tot']:+.1f}R  avgR {cl.pnl_r.mean():+.2f}  DD {st['dd']:.1f}%  LS {st['ls']}  "
              f"end ${st['bal']:,.0f}  hold {cl.hold_h.mean():.1f}h")
        return st

    print("=" * 100)
    print("RSI-2 MEAN-REVERSION vs GRADIENT BOT · 17 instruments · realistic · $1k/$100k")
    print(f"  (SMA{SMA_N} trend · RSI{RSI_N} {RSI_LO:.0f}/{RSI_HI:.0f} · stop {STOP_ATR}xATR · "
          f"exit RSI{EXIT_RSI:.0f} · max-hold {MAX_HOLD} bars)")
    print("=" * 100)
    smr = report("mean-reversion", mr)
    sgr = report("gradient bot", grad)

    # ── daily-return correlation ──
    dmr = daily_pnl(mr); dgr = daily_pnl(grad)
    alld = pd.date_range(min(dmr.index.min(), dgr.index.min()), max(dmr.index.max(), dgr.index.max()))
    a = dmr.reindex(alld, fill_value=0.0); b = dgr.reindex(alld, fill_value=0.0)
    both = (a != 0) | (b != 0)                    # days at least one traded
    corr_all = np.corrcoef(a, b)[0, 1]
    corr_active = np.corrcoef(a[both], b[both])[0, 1]
    comb = a + b
    def dd_of(series):
        eq = 100000 + series.cumsum(); peak = eq.cummax()
        return ((eq - peak) / peak * 100).min()
    print("\n  --- DAILY-RETURN CORRELATION (the whole point) ---")
    print(f"  Pearson corr (all days)     : {corr_all:+.3f}")
    print(f"  Pearson corr (active days)  : {corr_active:+.3f}")
    print(f"  Gradient-only  DD : {dd_of(b):.1f}%   vol/day ${b[both].std():,.0f}")
    print(f"  MeanRev-only   DD : {dd_of(a):.1f}%   vol/day ${a[both].std():,.0f}")
    print(f"  COMBINED       DD : {dd_of(comb):.1f}%   vol/day ${comb[both].std():,.0f}")
    print(f"  Combined net P&L  : ${comb.sum():,.0f}  (grad ${b.sum():,.0f} + MR ${a.sum():,.0f})")

    mr.to_csv(os.path.join(OUT, "trades_mr.csv"), index=False)

    # ── charts ──
    BG, FG = "#0d1117", "#e6edf3"
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 9), facecolor=BG, gridspec_kw={"height_ratios": [1.5, 1]})
    a1.set_facecolor(BG); a2.set_facecolor(BG)
    for lbl, series, col in [("gradient bot", b, "#3fb950"), ("mean-reversion", a, "#f2cc60"),
                             ("combined", comb, "#58a6ff")]:
        eq = 100000 + series.cumsum()
        a1.plot(eq.index, eq.values, color=col, linewidth=1.5,
                label=f"{lbl}: end ${eq.values[-1]:,.0f}, DD {dd_of(series):.1f}%")
    a1.axhline(100000, color="#444c56", linestyle="--", linewidth=0.8)
    a1.set_title(f"Two engines · equity ($1k/$100k) · daily-return corr {corr_active:+.2f}",
                 color=FG, fontsize=12, fontweight="bold")
    a1.tick_params(colors=FG); a1.grid(True, alpha=0.1, color=FG)
    a1.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    a1.xaxis.set_major_locator(mdates.YearLocator()); a1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in a1.spines.values(): sp.set_edgecolor("#30363d")
    a2.scatter(b[both], a[both], s=10, color="#7d8590", alpha=0.5)
    a2.axhline(0, color="#444c56", lw=0.6); a2.axvline(0, color="#444c56", lw=0.6)
    a2.set_title(f"Daily P&L scatter · corr {corr_active:+.2f} (each dot = one day)", color=FG, fontsize=11, fontweight="bold")
    a2.set_xlabel("gradient bot daily $", color=FG); a2.set_ylabel("mean-rev daily $", color=FG)
    a2.tick_params(colors=FG); a2.grid(True, alpha=0.1, color=FG)
    for sp in a2.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "meanrev_vs_gradient.png"), dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nSaved -> {OUT}/")
    mt5.shutdown()


if __name__ == "__main__":
    main()
