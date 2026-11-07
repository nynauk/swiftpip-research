"""
ICT SESSION-SWEEP + HTF PD-ARRAY + MSS/CHOCH + FVG entry, fixed 2R.
Realistic conditions, same 16 instruments as the EMA+DOL work, and compared
directly against EMA+DOL (baseline) and EMA+DOL momentum 45/55.

Mechanical chain (M15 entry TF, 4H = HTF derived by resample):
  1. SESSION SWEEP — take the Asia-range high/low (00:00-08:00 broker) during
     the 08:00-20:00 sweep window (wick beyond level, close back inside).
        Asia HIGH swept -> SHORT bias ;  Asia LOW swept -> LONG bias
  2. HTF PD ARRAY — the sweep's extreme must tap an UNMITIGATED 4H FVG in the
     reversal direction (short->bearish 4H FVG overhead, long->bullish below).
     No HTF array in confluence => no trade. (This is the key filter.)
  3. MSS / CHOCH — after the sweep, price must break the last M15 structural
     point against the sweep (short: close < pre-sweep swing low; long: >high).
  4. ENTRY — retest of the displacement FVG left by the MSS leg
     (short: bearish FVG bottom; long: bullish FVG top).
  5. SL beyond the swept extreme (+10% buffer); TP = fixed 2R.
News filter on, 1 position/symbol, realistic per-instrument spread+slippage.
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
import ema_dol_strategy as ed        # DOL helpers + swings + PDH/PDL
import realistic_ema_dol_rsi as edr   # reuse SPEC, GROUPS, SYMBOLS, stats, run, precompute

SPEC   = edr.SPEC
GROUPS = edr.GROUPS
SYMBOLS = edr.SYMBOLS

ASIA_START, ASIA_END   = 0, 8      # broker time
SWEEP_START, SWEEP_END = 8, 20
RR            = 2.0
MIN_RR        = 2.0                 # DOL must be at least 2R away
MAX_RR        = 10.0
MSS_LOOKBACK  = 10                 # M15 bars before sweep for the structural ref
MSS_WINDOW    = 24                 # bars after sweep to confirm MSS
RETEST_WINDOW = 16                 # bars after displacement FVG to fill the entry
SL_BUF        = 0.10               # buffer beyond sweep extreme, as frac of raw risk


def resample_4h(df15):
    o = df15.resample("4h").agg({"open": "first", "high": "max",
                                 "low": "min", "close": "last"}).dropna()
    return o


def four_h_fvgs(df4h, min_w):
    h = df4h["high"].values; l = df4h["low"].values; c = df4h["close"].values
    idx = df4h.index
    out = []
    for k in range(2, len(df4h)):
        if l[k] > h[k - 2] and (l[k] - h[k - 2]) >= min_w:
            out.append(dict(dir="bullish", top=float(l[k]), bottom=float(h[k - 2]), k=k))
        elif h[k] < l[k - 2] and (l[k - 2] - h[k]) >= min_w:
            out.append(dict(dir="bearish", top=float(l[k - 2]), bottom=float(h[k]), k=k))
    return out, c, idx


def htf_confluence(fvgs4h, h4c, idx4h, sweep_time, reach, want_dir):
    """Unmitigated 4H FVG of want_dir, formed before the sweep, containing `reach`."""
    s4 = int(np.searchsorted(idx4h, pd.Timestamp(sweep_time), "right")) - 1
    if s4 < 1:
        return False
    for f in fvgs4h:
        if f["k"] >= s4 or f["dir"] != want_dir:
            continue
        if not (f["bottom"] <= reach <= f["top"]):
            continue
        seg = h4c[f["k"] + 1:s4 + 1]
        if want_dir == "bearish" and len(seg) and (seg > f["top"]).any():
            continue
        if want_dir == "bullish" and len(seg) and (seg < f["bottom"]).any():
            continue
        return True
    return False


def last_fvg(h, l, i0, i1, direction, min_w):
    """Most recent FVG of `direction` completing in [i0, i1]."""
    found = None
    for k in range(i0 + 2, i1 + 1):
        if direction == "bullish" and l[k] > h[k - 2] and (l[k] - h[k - 2]) >= min_w:
            found = dict(top=float(l[k]), bottom=float(h[k - 2]), k=k)
        elif direction == "bearish" and h[k] < l[k - 2] and (l[k - 2] - h[k]) >= min_w:
            found = dict(top=float(l[k - 2]), bottom=float(h[k]), k=k)
    return found


def sim_realistic(h, l, idx, ei, direction, entry, sl, tp, rr_val, spread, max_bars=400):
    for j in range(ei + 1, min(ei + max_bars, len(h))):
        if direction == "long":
            if l[j] <= sl: return "loss", idx[j], -1.0
            if h[j] >= tp: return "win",  idx[j],  rr_val
        else:
            if h[j] + spread >= sl: return "loss", idx[j], -1.0
            if l[j] + spread <= tp: return "win",  idx[j],  rr_val
    return "open", idx[min(ei + max_bars - 1, len(h) - 1)], 0.0


def run_sweep(symbol, df15, realistic=True, tp_mode="FIXED", rsi_band=None):
    spread_px, slip_px, min_w = SPEC[symbol]
    spread   = spread_px if realistic else 0.0
    slippage = slip_px   if realistic else 0.0

    h = df15["high"].values; l = df15["low"].values; c = df15["close"].values
    idx = df15.index; hours = idx.hour; dates = idx.normalize()
    news_set = nf.build_news_block_set(idx[0].date(), idx[-1].date())

    df4h = resample_4h(df15)
    fvgs4h, h4c, idx4h = four_h_fvgs(df4h, min_w * 2)

    # DOL + RSI machinery (same as EMA+DOL): swing arrays, PDH/PDL, RSI(14)
    sh, sl_arr = ed.find_swings_arr(h, l, n=ed.SWING_LOOKBACK)
    pdh_map, pdl_map = ed.build_pdh_pdl(df15)
    rsi_arr = edr.rsi(c)

    trades = []; open_until = []
    udays, dstart = np.unique(dates, return_index=True)
    dend = np.append(dstart[1:], len(df15))

    for di, day in enumerate(udays):
        a, b = dstart[di], dend[di]
        dh = hours[a:b]
        am = (dh >= ASIA_START) & (dh < ASIA_END)
        if not am.any():
            continue
        asia_hi = h[a:b][am].max(); asia_lo = l[a:b][am].min()
        sw_local = np.where((dh >= SWEEP_START) & (dh < SWEEP_END))[0]
        if not len(sw_local):
            continue

        # 1. first sweep in the window
        sdir = si = sext = None
        for j in sw_local:
            i = a + j
            if h[i] > asia_hi and c[i] < asia_hi:
                sdir, si, sext = "short", i, float(h[i]); break
            if l[i] < asia_lo and c[i] > asia_lo:
                sdir, si, sext = "long", i, float(l[i]); break
        if sdir is None or si < MSS_LOOKBACK:
            continue

        # 2. HTF PD-array confluence (unmitigated 4H FVG in reversal dir at the sweep)
        want = "bearish" if sdir == "short" else "bullish"
        if not htf_confluence(fvgs4h, h4c, idx4h, idx[si], sext, want):
            continue

        # 3. MSS / CHOCH after the sweep
        if sdir == "short":
            ref = l[si - MSS_LOOKBACK:si].min()
        else:
            ref = h[si - MSS_LOOKBACK:si].max()
        mss_end = min(si + MSS_WINDOW, len(df15) - 1)
        mi = None
        for k in range(si + 1, mss_end + 1):
            if sdir == "short" and c[k] < ref: mi = k; break
            if sdir == "long"  and c[k] > ref: mi = k; break
        if mi is None:
            continue

        # 4. displacement FVG from the MSS leg + retest entry
        fvg = last_fvg(h, l, si, min(mi + 1, len(df15) - 1), want, min_w)
        if fvg is None:
            continue
        entry_lvl = fvg["bottom"] if sdir == "short" else fvg["top"]
        rt_end = min(fvg["k"] + RETEST_WINDOW, len(df15) - 1)
        ei = None
        for k in range(fvg["k"] + 1, rt_end + 1):
            if sdir == "short" and h[k] >= entry_lvl: ei = k; break
            if sdir == "long"  and l[k] <= entry_lvl: ei = k; break
        if ei is None:
            continue
        if hours[ei] < SWEEP_START or hours[ei] >= SWEEP_END + 3:
            continue

        # 5. fills + SL beyond the swept extreme
        if sdir == "short":
            direction = "short"
            real_entry = entry_lvl - slippage
            sl = sext + abs(sext - real_entry) * SL_BUF
            risk = sl - real_entry
        else:
            direction = "long"
            real_entry = entry_lvl + spread + slippage
            sl = sext - abs(real_entry - sext) * SL_BUF
            risk = real_entry - sl
        if risk <= 0:
            continue

        open_until = [t for t in open_until if t > idx[ei]]
        if len(open_until) >= 1:
            continue
        if idx[ei].date() in news_set:
            continue

        # RSI momentum filter at the entry bar (long trend-side / short trend-side)
        if rsi_band is not None:
            rv = rsi_arr[ei]
            if np.isnan(rv):
                continue
            llo, lhi, slo, shi = rsi_band
            if direction == "long" and not (llo <= rv <= lhi):
                continue
            if direction == "short" and not (slo <= rv <= shi):
                continue

        # 6. TP: fixed 2R, or DOL = next opposing liquidity (>= MIN_RR)
        if tp_mode == "DOL":
            d_today = pd.Timestamp(idx[ei]).normalize()
            pdh_t = pdh_map.get(d_today); pdl_t = pdl_map.get(d_today)
            if direction == "long":
                dol = ed.next_liquidity_above(real_entry, sh, ei, pdh_t)
                if dol is None: continue
                tp = dol; rr_val = (tp - real_entry) / risk
            else:
                dol = ed.next_liquidity_below(real_entry, sl_arr, ei, pdl_t)
                if dol is None: continue
                tp = dol; rr_val = (real_entry - tp) / risk
            if rr_val < MIN_RR or rr_val > MAX_RR:
                continue
        else:
            rr_val = RR
            tp = real_entry + RR * risk if direction == "long" else real_entry - RR * risk

        oc, xt, pr = sim_realistic(h, l, idx, ei, direction, real_entry, sl, tp, rr_val, spread)
        open_until.append(pd.Timestamp(xt))
        trades.append({"symbol": symbol, "entry_time": idx[ei], "direction": direction,
                       "entry": real_entry, "sl": sl, "tp": tp, "achieved_rr": round(rr_val, 2),
                       "outcome": oc, "exit_time": xt, "pnl_r": pr})
    return pd.DataFrame(trades)


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    OUT = os.path.join("strategies", "realistic_sweep_mss")
    os.makedirs(OUT, exist_ok=True)

    # Load M15 once; build EMA+DOL precompute alongside for the comparison
    data = {}; pcs = {}
    for s in SYMBOLS:
        df = bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
        data[s] = df; pcs[s] = edr.precompute(s, df)

    print("=" * 116)
    print("STRATEGY COMPARISON · realistic · 16 instruments · fixed $1,000/trade on $100k")
    print("  A) Sweep+HTF+MSS+FVG  ->  fixed 2R")
    print("  B) Sweep+HTF+MSS+FVG  ->  DOL target (>=2R)")
    print("  C) Sweep+HTF+MSS+FVG  ->  DOL target (>=2R) + RSI 45/55")
    print("  D) EMA(9/21)+DOL      + RSI momentum 45/55  (reigning champion)")
    print("=" * 116)

    def combine(lst):
        return pd.concat(lst, ignore_index=True).sort_values("entry_time").reset_index(drop=True)

    A = combine([run_sweep(s, data[s], True, tp_mode="FIXED") for s in SYMBOLS])
    B = combine([run_sweep(s, data[s], True, tp_mode="DOL") for s in SYMBOLS])
    C = combine([run_sweep(s, data[s], True, tp_mode="DOL", rsi_band=(45, 100, 0, 55)) for s in SYMBOLS])
    D = combine([edr.run(s, pcs[s], (45, 100, 0, 55), realistic=True) for s in SYMBOLS])
    sA, sB, sC, sD = edr.stats(A), edr.stats(B), edr.stats(C), edr.stats(D)

    print(f"  {'Strategy':40} {'N':>5} {'WR':>6} {'PF':>5} {'R':>8} {'DD':>7} "
          f"{'LossStrk':>8} {'End$100k':>11}")
    rows = [("A) Sweep -> fixed 2R", sA),
            ("B) Sweep -> DOL (>=2R)", sB),
            ("C) Sweep -> DOL + RSI 45/55", sC),
            ("D) EMA+DOL momentum 45/55", sD)]
    for lbl, st in rows:
        if st is None:
            print(f"  {lbl:40} no trades"); continue
        print(f"  {lbl:40} {st['n']:>5} {st['wr']:>5.1f}% {st['pf']:>4.2f} "
              f"{st['tot']:>+7.1f}R {st['dd']:>6.1f}% {st['ls']:>8} ${st['bal']:>10,.0f}")

    # Per-group for the best sweep config (C = DOL + RSI)
    print("\n  Sweep -> DOL + RSI 45/55 — by group:")
    print(f"  {'Group':11} {'N':>4} {'WR':>6} {'PF':>5} {'R':>8} {'LossStrk':>8}")
    for gname, syms in GROUPS:
        g = C[C["symbol"].isin(syms)]
        st = edr.stats(g)
        if st is None:
            print(f"  {gname:11}  no trades"); continue
        print(f"  {gname:11} {st['n']:>4} {st['wr']:>5.1f}% {st['pf']:>4.2f} "
              f"{st['tot']:>+7.1f}R {st['ls']:>8}")

    C.to_csv(os.path.join(OUT, "trades_sweep_dol_rsi.csv"), index=False)

    # Equity comparison
    BG, FG = "#0d1117", "#e6edf3"
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=BG); ax.set_facecolor(BG)
    for lbl, st, col in [("A) Sweep fixed 2R", sA, "#f778ba"),
                         ("B) Sweep DOL", sB, "#d29922"),
                         ("C) Sweep DOL + RSI 45/55", sC, "#58a6ff"),
                         ("D) EMA+DOL momentum 45/55", sD, "#3fb950")]:
        if st is None: continue
        ax.plot(st["times"], st["eqa"], color=col, linewidth=1.6,
                drawstyle="steps-post",
                label=f"{lbl}: N={st['n']}, PF {st['pf']:.2f}, DD {st['dd']:.1f}%, "
                      f"end ${st['bal']:,.0f}")
    ax.axhline(edr.START_BAL, color="#444c56", linestyle="--", linewidth=0.8)
    ax.set_title("Sweep+HTF+MSS+FVG variants vs EMA+DOL · realistic · 16 instruments · $1k/trade",
                 color=FG, fontsize=12, fontweight="bold")
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
    ax.set_xlabel("Date", color=FG); ax.set_ylabel("Balance ($)", color=FG)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "sweep_vs_emadol.png"), dpi=140,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nSaved -> {OUT}/")
    mt5.shutdown()


if __name__ == "__main__":
    main()
