"""
Realistic-conditions test on the LIVE BOT STRATEGY (prev-4H sweep + 15M IFVG,
SL on disrespect candle). Adds:
  • spread (per-symbol pip cost paid at entry)
  • slippage (0.3 pips adverse fill on market entries)
  • MT5 historical OHLC = bid → LONG fills at ASK (close+spread), SHORT fills
    at bid (close). SHORT SL/TP triggers offset by spread (adverse).

Also tests the SAME strategy at 2RR and 3RR, and with partial@1R management.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MetaTrader5 as mt5

import main as bt


# LIVE BOT config uses these three
LIVE_SYMBOLS = ["US500.cash", "GBPJPY", "UKOIL.cash"]
# All 7 forex majors for the "raw signal" comparison
MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]

# Absolute spread (in PRICE units) per symbol — matches typical FTMO/ICMarkets
# retail conditions. Slippage is adverse fill on market entries.
SPREAD_PRICE = {
    # FX (in price units, not pips)
    "EURUSD":     1.0 * 1e-4,
    "GBPUSD":     1.5 * 1e-4,
    "USDJPY":     1.0 * 1e-2,   # JPY quote
    "USDCHF":     1.5 * 1e-4,
    "USDCAD":     1.5 * 1e-4,
    "AUDUSD":     1.5 * 1e-4,
    "NZDUSD":     2.0 * 1e-4,
    "GBPJPY":     2.0 * 1e-2,   # JPY quote, ~2 pips
    # Indices / CFDs
    "US500.cash": 0.5,          # 0.5 index points (~$5 on 1 lot)
    "UKOIL.cash": 0.05,         # 5 cents (~$5 on 1 lot)
}
SLIPPAGE_PRICE = {
    "EURUSD": 0.3 * 1e-4, "GBPUSD": 0.3 * 1e-4, "USDJPY": 0.3 * 1e-2,
    "USDCHF": 0.3 * 1e-4, "USDCAD": 0.3 * 1e-4, "AUDUSD": 0.3 * 1e-4,
    "NZDUSD": 0.3 * 1e-4, "GBPJPY": 0.3 * 1e-2,
    "US500.cash": 0.1, "UKOIL.cash": 0.01,
}
DIS_MULT        = 1.5
REV_WINDOW_M15  = 24
DIS_DELAY_M15   = 7
RETR_WINDOW_M15 = 48


def min_sl_for(sym):
    """Microscopic FVG protection — reject setups with SL under this."""
    if sym in ("US500.cash",):    return 1.5
    if sym in ("UKOIL.cash",):    return 0.15
    if "JPY" in sym:              return 1.5 * 1e-2
    return 1.5 * 1e-4


def prev4h_sweeps(df_15, df_h4):
    """Sweeps of the PREVIOUS closed 4H candle's H/L, on M15 bars."""
    prevH = df_h4["high"].shift(1); prevL = df_h4["low"].shift(1)
    floor4h = df_15.index.floor("4h")
    ph = prevH.reindex(floor4h).values
    pl = prevL.reindex(floor4h).values
    h, l, c = df_15["high"].values, df_15["low"].values, df_15["close"].values
    idx = df_15.index
    sweeps = []; cur = None; hd = ld = False
    for i in range(len(df_15)):
        p = floor4h[i]
        if p != cur: cur = p; hd = ld = False
        if np.isnan(ph[i]): continue
        if not hd and h[i] > ph[i] and c[i] < ph[i]:
            sweeps.append({"time": idx[i], "direction": "bearish",
                           "swept_level": float(ph[i]), "idx": i}); hd = True
        if not ld and l[i] < pl[i] and c[i] > pl[i]:
            sweeps.append({"time": idx[i], "direction": "bullish",
                           "swept_level": float(pl[i]), "idx": i}); ld = True
    return sweeps


def sim_realistic(h, l, idx, ei, direction, real_entry, sl, tp, rr,
                  spread, partial_at=None, max_bars=400):
    """
    Realistic bar-by-bar. Historical OHLC = bid.
      LONG:  bid low ≤ SL → stop. bid high ≥ TP → win.
      SHORT: bid high + spread ≥ SL → stop. bid low + spread ≤ TP → win.
    Optional partial: at +partial_at·R, close half + move SL to real_entry (BE).
    Returns (outcome, exit_time, pnl_r).
    """
    risk = abs(real_entry - sl)
    sl_cur = sl; armed = False; realized = 0.0; remain = 1.0
    last = ei
    for j in range(ei + 1, min(ei + max_bars, len(h))):
        last = j
        if direction == "long":
            if l[j] <= sl_cur:
                r = realized + remain * ((sl_cur - real_entry) / risk)
                return _lbl(r), idx[j], r
            if h[j] >= tp:
                r = realized + remain * rr
                return "win", idx[j], r
            if partial_at is not None and not armed and h[j] >= real_entry + partial_at * risk:
                realized += 0.5 * partial_at; remain -= 0.5
                sl_cur = real_entry; armed = True
        else:
            if h[j] + spread >= sl_cur:
                r = realized + remain * ((real_entry - sl_cur) / risk)
                return _lbl(r), idx[j], r
            if l[j] + spread <= tp:
                r = realized + remain * rr
                return "win", idx[j], r
            if partial_at is not None and not armed and l[j] + spread <= real_entry - partial_at * risk:
                realized += 0.5 * partial_at; remain -= 0.5
                sl_cur = real_entry; armed = True
    return "open", idx[last], realized


def _lbl(r):
    if r > 0.01: return "win"
    if r < -0.01: return "loss"
    return "be"


def run(df_h4, df_15, symbol, rr, realistic, partial_at=None):
    spread   = SPREAD_PRICE[symbol]   if realistic else 0.0
    slippage = SLIPPAGE_PRICE[symbol] if realistic else 0.0
    min_sl   = min_sl_for(symbol)

    sweeps = prev4h_sweeps(df_15, df_h4)
    fvgs = sorted(bt.find_all_fvgs(df_15), key=lambda f: f["form_idx"])
    fia = np.array([f["form_idx"] for f in fvgs])
    h, l, c = df_15["high"].values, df_15["low"].values, df_15["close"].values
    idx = df_15.index
    trades = []; used = set(); open_until = []

    for sw in sweeps:
        bias = sw["direction"]; sig = (sw["time"], bias)
        if sig in used: continue
        si = sw["idx"]
        if si >= len(idx) - 2: continue
        tdir = "bearish" if bias == "bullish" else "bullish"
        rev_end = min(si + REV_WINDOW_M15, len(idx) - 1)
        lo = int(np.searchsorted(fia, si, "left"))
        hi = int(np.searchsorted(fia, rev_end, "right"))
        done = False
        for p in range(hi - 1, lo - 1, -1):
            fvg = fvgs[p]
            if fvg["direction"] != tdir or fvg["width"] < min_sl: continue
            fi = fvg["form_idx"]
            for di in range(fi + 1, min(fi + DIS_DELAY_M15, len(idx) - 1) + 1):
                inv = c[di] > fvg["top"] if fvg["direction"] == "bearish" else c[di] < fvg["bottom"]
                if not inv: continue

                dis_low = float(l[di]); dis_high = float(h[di])
                rng = dis_high - dis_low
                too_big = rng > DIS_MULT * fvg["width"]

                if not too_big:
                    signal_close = float(c[di])
                    ei = di
                    etype = "close"
                else:
                    rend = min(di + RETR_WINDOW_M15, len(idx) - 1); ei = None
                    for rj in range(di + 1, rend + 1):
                        if bias == "bullish" and l[rj] <= fvg["top"]:
                            ei = rj; signal_close = float(fvg["top"]); break
                        if bias == "bearish" and h[rj] >= fvg["bottom"]:
                            ei = rj; signal_close = float(fvg["bottom"]); break
                    if ei is None: break
                    etype = "retest"

                open_until = [t for t in open_until if t > idx[ei]]
                if len(open_until) >= 1:
                    done = True; break

                # Real fills with spread + slippage
                if bias == "bullish":
                    real_entry = signal_close + spread + slippage
                    sl = dis_low
                    if real_entry - sl <= 0: break
                    tp = real_entry + rr * (real_entry - sl)
                    direction = "long"
                    risk_p = real_entry - sl
                else:
                    real_entry = signal_close - slippage
                    sl = dis_high
                    if sl - real_entry <= 0: break
                    tp = real_entry - rr * (sl - real_entry)
                    direction = "short"
                    risk_p = sl - real_entry

                oc, xt, pr = sim_realistic(h, l, idx, ei, direction, real_entry, sl, tp,
                                            rr, spread, partial_at=partial_at)
                open_until.append(pd.Timestamp(xt))
                trades.append({
                    "symbol": symbol, "entry_time": idx[ei], "entry_idx": ei,
                    "direction": direction, "entry_type": etype,
                    "entry": round(real_entry, 6), "sl": round(sl, 6),
                    "tp": round(tp, 6), "risk_p": risk_p,
                    "outcome": oc, "exit_time": xt, "pnl_r": pr,
                })
                used.add(sig); done = True; break
            if done: break
    return pd.DataFrame(trades)


def stats(df):
    cl = df[df["outcome"] != "open"]
    if not len(cl): return None
    n = len(cl)
    wins = (cl["pnl_r"] > 0.01).sum()
    losses = (cl["pnl_r"] < -0.01).sum()
    bes = n - wins - losses
    wr = wins / n * 100
    tot = cl["pnl_r"].sum()
    gw = cl.loc[cl["pnl_r"] > 0, "pnl_r"].sum()
    gl = -cl.loc[cl["pnl_r"] < 0, "pnl_r"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    bal = 10000.0; eq = [bal]; cl_ = cw = bcl = bcw = 0
    for _, r in cl.iterrows():
        bal += bal * 0.01 * float(r["pnl_r"]); eq.append(bal)
        if r["pnl_r"] < -0.01: cl_ += 1; cw = 0; bcl = max(bcl, cl_)
        elif r["pnl_r"] > 0.01: cw += 1; cl_ = 0; bcw = max(bcw, cw)
    eqa = np.array(eq)
    dd = ((eqa - np.maximum.accumulate(eqa)) / np.maximum.accumulate(eqa) * 100).min()
    return dict(n=n, wins=wins, be=bes, loss=losses, wr=wr, pf=pf, tot=tot,
                dd=dd, ls=bcl, ws=bcw, bal=bal, eqa=eqa)


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    OUT = os.path.join("strategies", "realistic_live")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 108)
    print("LIVE BOT STRATEGY - prev-4H sweep + 15M IFVG - SL on disrespect candle")
    print("Symbols: US500.cash, GBPJPY, UKOIL.cash (the ACTUAL live bot config)")
    print("Naive (no costs) vs Realistic (spread + slippage) - 2RR & 3RR - plain & partial@1R")
    print("=" * 108)

    # Build the data once per symbol
    data = {}
    for s in LIVE_SYMBOLS:
        df_h4 = bt.fetch_data(s, mt5.TIMEFRAME_H4,  bt.START_DATE, bt.END_DATE, tf_name="H4")
        df_15 = bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
        data[s] = (df_h4, df_15)

    variants = [
        ("2RR · naive · plain",           2.0, False, None),
        ("2RR · realistic · plain",       2.0, True,  None),
        ("2RR · realistic · partial@1R",  2.0, True,  1.0),
        ("3RR · naive · plain",           3.0, False, None),
        ("3RR · realistic · plain",       3.0, True,  None),
        ("3RR · realistic · partial@1R",  3.0, True,  1.0),
    ]

    print(f"\n  Combined across LIVE BOT symbols (US500.cash + GBPJPY + UKOIL.cash):")
    print(f"  {'Variant':32} {'N':>5} {'WR':>6} {'PF':>5} {'R':>9} {'DD':>7} {'LStrk':>5}  End $10k")
    keep = {}
    for label, rr, real, pa in variants:
        allt = []
        for s in LIVE_SYMBOLS:
            df_h4, df_15 = data[s]
            df = run(df_h4, df_15, s, rr, real, pa)
            allt.append(df)
        combined = pd.concat(allt, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
        stt = stats(combined)
        if stt is None: continue
        keep[label] = (combined, stt)
        print(f"  {label:32} {stt['n']:>5} {stt['wr']:>5.1f}% {stt['pf']:>4.2f} "
              f"{stt['tot']:>+7.1f}R {stt['dd']:>6.1f}% {stt['ls']:>5}  ${stt['bal']:>10,.0f}")

    # Per-symbol breakdown for the realistic 3RR plain (matches live bot config exactly)
    label = "3RR · realistic · plain"
    if label in keep:
        combined, _ = keep[label]
        print(f"\n  Per-symbol breakdown for LIVE BOT CONFIG (3RR, plain, realistic):")
        print(f"  {'Symbol':12}  {'N':>4}  {'WR':>6}  {'R':>8}")
        for sym in LIVE_SYMBOLS:
            g = combined[combined["symbol"] == sym]
            cg = g[g["outcome"] != "open"]
            if not len(cg): continue
            wr = (cg["pnl_r"] > 0.01).sum() / len(cg) * 100
            print(f"  {sym:12}  {len(cg):>4}  {wr:>5.1f}%  {cg['pnl_r'].sum():>+7.1f}R")

    # Save equity curves for the 4 realistic variants
    BG, FG, GRN, RED = "#0d1117", "#e6edf3", "#3fb950", "#f85149"
    for lbl in [k for k in keep if "realistic" in k]:
        combined, stt = keep[lbl]
        fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG); ax.set_facecolor(BG)
        eqa = stt["eqa"]
        ax.plot(eqa, color=GRN, linewidth=1.4)
        ax.axhline(10000, color="#444c56", linestyle="--", linewidth=0.8)
        ax.fill_between(range(len(eqa)), 10000, eqa,
                        where=eqa >= 10000, alpha=0.15, color=GRN)
        ax.fill_between(range(len(eqa)), 10000, eqa,
                        where=eqa < 10000, alpha=0.18, color=RED)
        ax.set_title(f"{lbl}  |  {stt['n']} trades  |  end ${stt['bal']:,.0f}  |  DD {stt['dd']:.1f}%",
                     color=FG, fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
        ax.set_xlabel("Trade #", color=FG); ax.set_ylabel("Balance ($)", color=FG)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        plt.tight_layout()
        fn = "equity_" + lbl.replace("·", "").replace(" ", "_").replace("@", "at") + ".png"
        plt.savefig(os.path.join(OUT, fn), dpi=140, bbox_inches="tight", facecolor=BG)
        plt.close(fig)

    # Save the live-bot config trade log
    if "3RR · realistic · plain" in keep:
        keep["3RR · realistic · plain"][0].to_csv(
            os.path.join(OUT, "trades_live_bot_config.csv"), index=False)
    print(f"\nSaved equity curves + trades.csv -> {OUT}/")

    mt5.shutdown()


if __name__ == "__main__":
    main()
