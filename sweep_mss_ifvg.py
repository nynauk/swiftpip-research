"""
ICT intraday model on M5 (multiple trades/day target):
  liquidity SWEEP of a swing high/low
  -> MSS / CHOCH / BOS (structure breaks the other way)
  -> CISD / IFVG (displacement leaves a fair-value gap that price inverts+retests)
  -> ENTER on the IFVG retest
  -> SL at the swept SWING HIGH/LOW (+ buffer)
  -> TP fixed 3RR.
Realistic per-instrument spread. Causal swings (confirmed SWING_N bars later).
Goal check: multiple trades/day, profit factor ~3, FTMO-safe drawdown.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import MetaTrader5 as mt5

import frequency_tune as fq
import realistic_ema_dol_rsi as edr

SYMS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "GBPJPY",
        "US500.cash", "US30.cash", "GER40.cash", "US100.cash", "UK100.cash"]
RR = 3.0
SWING_N = 5
MSS_WINDOW = 30        # M5 bars after sweep to confirm MSS
RETEST_WINDOW = 24     # M5 bars after IFVG to fill the entry
SL_BUF = 0.05          # buffer beyond the swept swing, as frac of stop
OUT = os.path.join("strategies", "sweep_mss_ifvg")


def fetch_m5(symbol):
    mt5.symbol_select(symbol, True)
    bars = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 99999)
    if bars is None or not len(bars):
        return None
    df = pd.DataFrame(bars)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.set_index("time")[["open", "high", "low", "close"]]


def swings(h, l, n):
    sh = np.full(len(h), np.nan); sl = np.full(len(l), np.nan)
    for i in range(n, len(h) - n):
        if h[i] >= h[i - n:i + n + 1].max(): sh[i] = h[i]
        if l[i] <= l[i - n:i + n + 1].min(): sl[i] = l[i]
    return sh, sl


def last_confirmed(sh, sl, n):
    """Most recent swing high/low CONFIRMED at each bar (swing at j known at j+n)."""
    N = len(sh)
    shv = np.full(N, np.nan); shi = np.full(N, -1, int)
    slv = np.full(N, np.nan); sli = np.full(N, -1, int)
    csh = np.nan; cshi = -1; csl = np.nan; csli = -1
    for i in range(N):
        j = i - n
        if j >= 0:
            if not np.isnan(sh[j]): csh, cshi = sh[j], j
            if not np.isnan(sl[j]): csl, csli = sl[j], j
        shv[i], shi[i], slv[i], sli[i] = csh, cshi, csl, csli
    return shv, shi, slv, sli


def run(symbol, df):
    spread, slippage, _ = fq.SPEC[symbol]
    h = df["high"].values; l = df["low"].values; c = df["close"].values; idx = df.index
    sh, sl = swings(h, l, SWING_N)
    shv, shi, slv, sli = last_confirmed(sh, sl, SWING_N)
    N = len(df)
    trades = []; busy = -1; swept_hi = set(); swept_lo = set()

    for i in range(SWING_N + 3, N - 1):
        if i <= busy:
            continue
        # ── 1. sweep of the most recent confirmed swing ──
        bias = None
        if not np.isnan(shv[i]) and h[i] > shv[i] and c[i] < shv[i] and shi[i] not in swept_hi:
            bias = "short"; swept_hi.add(shi[i]); sweep_ext = h[i]; struct = slv[i]
        elif not np.isnan(slv[i]) and l[i] < slv[i] and c[i] > slv[i] and sli[i] not in swept_lo:
            bias = "long"; swept_lo.add(sli[i]); sweep_ext = l[i]; struct = shv[i]
        if bias is None or np.isnan(struct):
            continue
        si = i

        # ── 2. MSS / BOS: close beyond the opposing structure after the sweep ──
        mend = min(si + MSS_WINDOW, N - 1)
        mi = None
        for k in range(si + 1, mend + 1):
            if bias == "short" and c[k] < struct: mi = k; break
            if bias == "long" and c[k] > struct: mi = k; break
        if mi is None:
            continue

        # ── 3. CISD/IFVG: last displacement FVG in the MSS leg ──
        fvg = None
        for k in range(mi, si, -1):
            if k < 2:
                break
            if bias == "short" and h[k] < l[k - 2]:
                fvg = (float(l[k - 2]), float(h[k])); break     # bearish gap (top, bottom)
            if bias == "long" and l[k] > h[k - 2]:
                fvg = (float(h[k - 2]), float(l[k])); break      # bullish gap (bottom, top)
        if fvg is None:
            continue

        # ── 4. entry on IFVG retest ──
        rend = min(mi + RETEST_WINDOW, N - 1); ei = None
        if bias == "short":
            edge = fvg[1]                       # bottom of bearish gap = first touch from below
            for k in range(mi + 1, rend + 1):
                if h[k] >= edge: ei = k; break
        else:
            edge = fvg[1]                       # top of bullish gap = first touch from above
            for k in range(mi + 1, rend + 1):
                if l[k] <= edge: ei = k; break
        if ei is None:
            continue

        # ── 5. SL at swept swing extreme (+buffer +spread+1pip), TP 3RR ──
        pad = edr.sl_pad(symbol, spread)               # spread+1pip stop buffer
        if bias == "short":
            entry = edge - slippage
            sl_px = sweep_ext + SL_BUF * (sweep_ext - entry) + pad
            risk = sl_px - entry
            if risk <= 0: continue
            tp = entry - RR * risk
        else:
            entry = edge + spread + slippage
            sl_px = sweep_ext - SL_BUF * (entry - sweep_ext) - pad
            risk = entry - sl_px
            if risk <= 0: continue
            tp = entry + RR * risk

        oc, xj, pr = _sim(h, l, ei, bias, entry, sl_px, tp, spread)
        busy = xj
        trades.append(dict(symbol=symbol, entry_time=idx[ei], exit_time=idx[xj],
                           direction=bias, outcome=oc, pnl_r=pr,
                           risk=risk, hold_h=(idx[xj] - idx[ei]).total_seconds() / 3600))
    return pd.DataFrame(trades)


def _sim(h, l, ei, bias, entry, sl_px, tp, spread, max_bars=600):
    for j in range(ei + 1, min(ei + max_bars, len(h))):
        if bias == "long":
            if l[j] <= sl_px: return "loss", j, -1.0
            if h[j] >= tp: return "win", j, RR
        else:
            if h[j] + spread >= sl_px: return "loss", j, -1.0
            if l[j] + spread <= tp: return "win", j, RR
    return "open", min(ei + max_bars - 1, len(h) - 1), 0.0


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    os.makedirs(OUT, exist_ok=True)
    allt = []; spans = []
    for s in SYMS:
        df = fetch_m5(s)
        if df is None or len(df) < 500:
            print(f"  {s}: no M5 data"); continue
        spans.append((df.index[0], df.index[-1]))
        allt.append(run(s, df))
    df = pd.concat(allt, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    cl = df[df.outcome != "open"]
    d0 = min(s[0] for s in spans); d1 = max(s[1] for s in spans)
    days = (d1 - d0).days
    tdays = days * 5 / 7                      # approx trading days
    st = edr.stats(cl)

    print("=" * 92)
    print("SWEEP -> MSS/BOS -> CISD/IFVG -> 3RR (SL at swing) · M5 · realistic · "
          f"{len(SYMS)} instruments")
    print(f"  data {d0.date()} -> {d1.date()} (~{tdays:.0f} trading days)")
    print("=" * 92)
    print(f"  Trades        : {st['n']}   ({st['n']/tdays:.1f}/day · {st['n']/(tdays/5):.1f}/week)")
    print(f"  Win rate      : {st['wr']:.1f}%   (breakeven for 3RR = 25%; PF 3 needs 50%)")
    print(f"  Profit factor : {st['pf']:.2f}   {'✅ HIT PF 3' if st['pf']>=3 else '❌ below PF 3'}")
    print(f"  Total R       : {st['tot']:+.1f}R   (end ${st['bal']:,.0f} from $10k @1%)")
    print(f"  Max drawdown  : {st['dd']:.1f}%   loss streak {st['ls']}")
    print(f"  Avg hold      : {cl.hold_h.mean():.1f}h")

    print("\n  Per instrument:")
    print(f"  {'sym':12} {'N':>4} {'WR':>6} {'PF':>5} {'totR':>8}")
    for s in SYMS:
        g = cl[cl.symbol == s]
        if len(g):
            s2 = edr.stats(g)
            print(f"  {s:12} {s2['n']:>4} {s2['wr']:>5.1f}% {s2['pf']:>4.2f} {s2['tot']:>+7.1f}R")

    cl.to_csv(os.path.join(OUT, "trades.csv"), index=False)
    # equity
    bal = 10000.0; eq = [bal]
    for r in cl.sort_values("entry_time")["pnl_r"]:
        bal += bal * 0.01 * float(r); eq.append(bal)
    BG, FG = "#0d1117", "#e6edf3"
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(eq, color="#3fb950" if st['pf'] >= 1 else "#f85149", linewidth=1.4)
    ax.axhline(10000, color="#444c56", linestyle="--", linewidth=0.8)
    ax.set_title(f"Sweep+MSS+IFVG · 3RR · M5 · PF {st['pf']:.2f} · WR {st['wr']:.1f}% · "
                 f"{st['n']/tdays:.1f} trades/day · DD {st['dd']:.1f}%",
                 color=FG, fontsize=12, fontweight="bold")
    ax.set_xlabel("Trade #", color=FG); ax.set_ylabel("Balance ($)", color=FG)
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "equity.png"), dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nSaved -> {OUT}/")
    mt5.shutdown()


if __name__ == "__main__":
    main()
