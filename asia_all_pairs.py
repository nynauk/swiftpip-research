"""
Asia sweep + 1M CISD on ALL forex majors — MT5 data (broker time).

Session hours in broker time (GMT+3):
  Asia range (Tokyo/Sydney window):  03:00 - 10:00 broker  = 00:00 - 07:00 UTC
  London entry window:               10:00 - 13:00 broker  = 07:00 - 10:00 UTC

Strategy = CISD_ONLY mode (winner from prior EURUSD scan on the UTC CSV):
  1. Compute Asia range high/low per day
  2. In London window, wait for first sweep of Asia H/L
  3. Then wait for 1M CISD (close through recent opposing 5-bar swing)
  4. Enter at CISD close, SL beyond swept extreme + buffer + min 3 pips,
     TP at 2R.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MetaTrader5 as mt5


# ── Broker-time session hours ──
ASIA_START      = 3         # broker (00:00 UTC)
ASIA_END        = 10        # broker (07:00 UTC)
LONDON_START    = 10        # broker (07:00 UTC)
LONDON_END      = 13        # broker (10:00 UTC)

# ── Strategy params ──
CISD_LOOKBACK   = 5
CISD_WINDOW     = 60        # bars after sweep to find CISD
SL_BUFFER       = 0.10      # fraction of sweep-to-entry distance
RR              = 3.0       # 3RR target
MIN_SL_PIPS     = 3         # in pips (0.0003 for majors, 0.03 for JPY pairs)

# ── Realistic market conditions ──
# Round-trip spread cost per symbol (in pips). Values are conservative retail
# spreads on EURUSD-type accounts (paper FTMO would be tighter).
SPREAD_PIPS = {
    "EURUSD": 1.0, "GBPUSD": 1.5, "USDJPY": 1.0, "USDCHF": 1.5,
    "USDCAD": 1.5, "AUDUSD": 1.5, "NZDUSD": 2.0,
}
# Slippage on market entry (additional pips of adverse fill). Realistic on M1.
SLIPPAGE_PIPS = 0.3
# Commission per round-turn in R units — set 0.0 (most retail spreads-inclusive)
COMMISSION_R = 0.0

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]


def pip_for(symbol: str) -> float:
    return 0.01 if "JPY" in symbol else 0.0001


def fetch_m1(symbol: str, force: bool = False) -> pd.DataFrame | None:
    """Load or fetch 99,999 M1 bars for a symbol."""
    cache = os.path.join("data_cache", f"{symbol}_M1.pkl")
    os.makedirs("data_cache", exist_ok=True)
    if os.path.exists(cache) and not force:
        return pd.read_pickle(cache)
    mt5.symbol_select(symbol, True)
    bars = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 99_999)
    if bars is None or len(bars) == 0:
        return None
    df = pd.DataFrame(bars)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")[["open", "high", "low", "close"]]
    df.to_pickle(cache)
    return df


def simulate(h, l, idx, ei, direction, entry, sl, tp, max_bars=600, spread=0.0):
    """
    Realistic sim. MT5 historical OHLC = BID prices.
      • LONG fill at ASK  = close + spread (adverse). SL/TP tested on bid as-is.
      • SHORT fill at BID = close (no adjustment on entry). But SL triggers on
        ASK ≥ SL (i.e. bid high ≥ SL - spread) and TP on ASK ≤ TP (bid low ≤
        TP - spread) — both shifted by spread against you.
    Returns (outcome, exit_time, exit_price, R). R is +RR for win, -1 for loss.
    Position sizing already accounts for the real entry, so R is consistent.
    """
    last = ei
    for j in range(ei + 1, min(ei + max_bars, len(h))):
        last = j
        if direction == "long":
            if l[j] <= sl: return "loss", idx[j], sl, -1.0
            if h[j] >= tp: return "win",  idx[j], tp, RR
        else:
            # SHORT: SL/TP tested against ask = bid + spread
            if h[j] + spread >= sl: return "loss", idx[j], sl, -1.0
            if l[j] + spread <= tp: return "win",  idx[j], tp, RR
    return "open", idx[last], float("nan"), 0.0


def run_cisd_only(df_1m: pd.DataFrame, symbol: str) -> pd.DataFrame:
    pip = pip_for(symbol)
    min_sl = MIN_SL_PIPS * pip
    spread = SPREAD_PIPS.get(symbol, 1.0) * pip
    slippage = SLIPPAGE_PIPS * pip

    h = df_1m["high"].values
    l = df_1m["low"].values
    c = df_1m["close"].values
    idx = df_1m.index
    hours = idx.hour
    dates = idx.normalize()

    trades = []
    uniq_days, day_starts = np.unique(dates, return_index=True)
    day_ends = np.append(day_starts[1:], len(df_1m))

    for di_day, day in enumerate(uniq_days):
        a, b = day_starts[di_day], day_ends[di_day]
        day_h = hours[a:b]

        # Asia range
        asia_mask = (day_h >= ASIA_START) & (day_h < ASIA_END)
        if not asia_mask.any():
            continue
        asia_hi = float(h[a:b][asia_mask].max())
        asia_lo = float(l[a:b][asia_mask].min())

        # London window
        lon_mask = (day_h >= LONDON_START) & (day_h < LONDON_END)
        lon_local = np.where(lon_mask)[0]
        if not len(lon_local):
            continue

        # First sweep in London window
        sweep_dir = None
        sweep_i = None
        sweep_extreme = None
        for j in lon_local:
            i = a + j
            if h[i] > asia_hi and c[i] < asia_hi:
                sweep_dir = "short"; sweep_i = i; sweep_extreme = float(h[i]); break
            if l[i] < asia_lo and c[i] > asia_lo:
                sweep_dir = "long";  sweep_i = i; sweep_extreme = float(l[i]); break
        if sweep_dir is None:
            continue

        # CISD scan
        cisd_end = min(sweep_i + CISD_WINDOW, len(df_1m) - 1)
        cisd_idx = None
        for k in range(sweep_i + 1, cisd_end + 1):
            if k - CISD_LOOKBACK < 0:
                continue
            if sweep_dir == "short":
                ref = l[k - CISD_LOOKBACK:k].min()
                if c[k] < ref:
                    cisd_idx = k; break
            else:
                ref = h[k - CISD_LOOKBACK:k].max()
                if c[k] > ref:
                    cisd_idx = k; break
        if cisd_idx is None:
            continue

        # Enforce London-window constraint on entry
        if hours[cisd_idx] < LONDON_START or hours[cisd_idx] >= LONDON_END:
            continue

        # ── Entry / SL / TP with realistic spread + slippage ──
        signal_close = float(c[cisd_idx])
        if sweep_dir == "short":
            direction = "short"
            # SELL fills at bid ≈ close, minus slippage (adverse)
            entry = signal_close - slippage
            base = sweep_extreme - entry
            sl = sweep_extreme + abs(base) * SL_BUFFER
            if sl - entry < min_sl:
                sl = entry + min_sl
            risk = sl - entry
            if risk <= 0:
                continue
            tp = entry - RR * risk
        else:
            direction = "long"
            # BUY fills at ask = close + spread, plus slippage (adverse)
            entry = signal_close + spread + slippage
            base = entry - sweep_extreme
            sl = sweep_extreme - abs(base) * SL_BUFFER
            if entry - sl < min_sl:
                sl = entry - min_sl
            risk = entry - sl
            if risk <= 0:
                continue
            tp = entry + RR * risk

        oc, xt, xp, pr = simulate(h, l, idx, cisd_idx, direction, entry, sl, tp,
                                   spread=spread)
        trades.append({
            "symbol":       symbol,
            "date":         pd.Timestamp(day).date(),
            "sweep_time":   idx[sweep_i],
            "sweep_dir":    sweep_dir,
            "sweep_extreme": round(sweep_extreme, 6),
            "asia_high":    round(asia_hi, 6),
            "asia_low":     round(asia_lo, 6),
            "entry_time":   idx[cisd_idx],
            "entry_idx":    cisd_idx,
            "direction":    direction,
            "entry":        round(entry, 6),
            "sl":           round(sl, 6),
            "tp":           round(tp, 6),
            "outcome":      oc,
            "exit":         round(xp, 6) if not np.isnan(xp) else np.nan,
            "exit_time":    xt,
            "pnl_r":        pr,
        })
    return pd.DataFrame(trades)


def streak_dd(df):
    bal = 10000.0; eq = [bal]; cl = cw = bcl = bcw = 0
    for _, r in df.sort_values("entry_time").iterrows():
        bal += bal * 0.01 * float(r["pnl_r"]); eq.append(bal)
        if r["outcome"] == "loss": cl += 1; cw = 0; bcl = max(bcl, cl)
        elif r["outcome"] == "win": cw += 1; cl = 0; bcw = max(bcw, cw)
    eq_arr = np.array(eq)
    dd = ((eq_arr - np.maximum.accumulate(eq_arr)) / np.maximum.accumulate(eq_arr) * 100).min()
    return bal, dd, bcl, bcw, eq_arr


def plot_equity(eq_arr, title, out_path):
    BG, FG, GRN, RED = "#0d1117", "#e6edf3", "#3fb950", "#f85149"
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(eq_arr, color=GRN, linewidth=1.4)
    ax.axhline(10000, color="#444c56", linestyle="--", linewidth=0.8)
    ax.fill_between(range(len(eq_arr)), 10000, eq_arr,
                    where=eq_arr >= 10000, alpha=0.15, color=GRN)
    ax.fill_between(range(len(eq_arr)), 10000, eq_arr,
                    where=eq_arr < 10000, alpha=0.18, color=RED)
    peak = np.maximum.accumulate(eq_arr)
    max_dd = ((eq_arr - peak) / peak * 100).min()
    ax.set_title(f"{title}  |  {len(eq_arr)-1} trades  |  end ${eq_arr[-1]:,.0f}  |  DD {max_dd:.1f}%",
                 color=FG, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Trade #", color=FG)
    ax.set_ylabel("Balance ($)", color=FG)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.tick_params(colors=FG)
    ax.grid(True, alpha=0.1, color=FG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_sample_trade(df_1m, trade, out_path):
    BG, FG = "#0d1117", "#e6edf3"
    GRN, RED, YLW = "#3fb950", "#f85149", "#d29922"
    BLU = "#58a6ff"
    UP, DN = "#26a69a", "#ef5350"

    entry_idx = int(trade["entry_idx"])
    exit_idx = df_1m.index.get_indexer([pd.Timestamp(trade["exit_time"])], method="nearest")[0]
    sweep_idx = df_1m.index.get_indexer([pd.Timestamp(trade["sweep_time"])], method="nearest")[0]
    lo = max(0, sweep_idx - 20)
    hi = min(len(df_1m), exit_idx + 20)
    win = df_1m.iloc[lo:hi].copy().reset_index()
    win.rename(columns={win.columns[0]: "time"}, inplace=True)

    ex = entry_idx - lo
    xx = exit_idx - lo
    sx = sweep_idx - lo
    is_short = trade["direction"] == "short"
    entry, sl, tp = float(trade["entry"]), float(trade["sl"]), float(trade["tp"])

    fig, ax = plt.subplots(figsize=(18, 10), facecolor=BG)
    ax.set_facecolor(BG)

    # candles
    for k, row in win.iterrows():
        col = UP if row["close"] >= row["open"] else DN
        ax.plot([k, k], [row["low"], row["high"]], color=col, linewidth=0.8, zorder=2)
        body_lo = min(row["open"], row["close"])
        hh = abs(row["close"] - row["open"])
        if hh == 0:
            hh = (win["high"].max() - win["low"].min()) * 0.0005
        ax.add_patch(plt.Rectangle((k - 0.35, body_lo), 0.7, hh,
                                   facecolor=col, edgecolor=col, zorder=3))

    # Asia H/L level
    ax.axhline(float(trade["sweep_extreme"]), color=BLU, linewidth=1.2, linestyle="--", alpha=0.85)
    lbl = "Asia HIGH swept" if is_short else "Asia LOW swept"
    ax.text(0.5, float(trade["sweep_extreme"]), f"  {lbl}",
            color=BLU, fontsize=10, va="bottom", fontweight="bold")

    # Sweep marker
    if 0 <= sx < len(win):
        ax.axvline(sx, color=BLU, linestyle=":", alpha=0.5, linewidth=1.0)
        ax.text(sx, win["high"].max(), "  Sweep",
                color=BLU, fontsize=9, va="top", rotation=90, fontweight="bold")

    # Position boxes
    box_w = max(xx - ex, 6)
    tlo, thi = sorted([entry, tp])
    ax.add_patch(plt.Rectangle((ex, tlo), box_w, thi - tlo,
                                facecolor=GRN, alpha=0.15, edgecolor=GRN, zorder=4))
    slo, shi = sorted([entry, sl])
    ax.add_patch(plt.Rectangle((ex, slo), box_w, shi - slo,
                                facecolor=RED, alpha=0.15, edgecolor=RED, zorder=4))

    # Level labels
    span = max(win["high"].max() - win["low"].min(), 1e-6)
    lvls = sorted([
        (entry, FG, f"Entry {entry}"),
        (tp,    GRN, f"TP ({int(RR)}R) {tp}"),
        (sl,    RED, f"SL {sl}"),
    ], key=lambda x: -x[0])
    last_y = None
    for price, col, lbl in lvls:
        ax.axhline(price, color=col, linewidth=1.0, alpha=0.85, zorder=5)
        y = price
        if last_y is not None and (last_y - y) < span * 0.03:
            y = last_y - span * 0.03
        last_y = y
        ax.text(ex + box_w + 0.5, y, f"  {lbl}", color=col, fontsize=10,
                va="center", fontweight="bold")

    # Entry / exit markers
    ax.scatter([ex], [entry], marker="v" if is_short else "^", s=200,
               color=YLW, edgecolor="black", zorder=7, label="Entry (CISD)")
    ec = GRN if trade["outcome"] == "win" else RED
    ax.scatter([xx], [float(trade["exit"])], marker="o", s=170, color=ec,
               edgecolor="black", zorder=7, label=f"Exit ({trade['outcome'].upper()})")

    step = max(1, len(win) // 10)
    ticks = list(range(0, len(win), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([win["time"].iloc[t].strftime("%m-%d %H:%M") for t in ticks],
                       rotation=30, ha="right", fontsize=9, color=FG)

    dur = (pd.to_datetime(trade["exit_time"]) - pd.to_datetime(trade["entry_time"])).total_seconds() / 60
    ax.set_title(f"ASIA SWEEP + 1M CISD · {trade['symbol']}  |  {trade['direction'].upper()}  |  "
                 f"{trade['outcome'].upper()} ({trade['pnl_r']:+.1f}R)  |  "
                 f"{pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d %H:%M')}  |  {dur:.0f} min",
                 color=FG, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Price", color=FG)
    ax.tick_params(colors=FG)
    ax.grid(True, alpha=0.1, color=FG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.legend(facecolor="#161b22", edgecolor="#444c56", labelcolor=FG, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    OUT = os.path.join("strategies", "asia_all_pairs")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 96)
    print(f"ASIA SWEEP + 1M CISD · 7 forex majors · RR={RR:.0f} · REALISTIC (spread + slippage)")
    print(f"Asia range: {ASIA_START:02d}-{ASIA_END:02d} broker (00-07 UTC) · "
          f"London entry: {LONDON_START:02d}-{LONDON_END:02d} broker (07-10 UTC)")
    print(f"Spreads (pips): " + ", ".join(f"{k}={v}" for k, v in SPREAD_PIPS.items()) +
          f"  ·  slippage {SLIPPAGE_PIPS} pips")
    print("=" * 96)

    all_trades = []
    per_pair = []
    for sym in SYMBOLS:
        print(f"  fetching {sym}...")
        df_1m = fetch_m1(sym)
        if df_1m is None or len(df_1m) < 100:
            print(f"    ✗ no data for {sym}")
            continue
        span_days = (df_1m.index[-1] - df_1m.index[0]).days
        trades = run_cisd_only(df_1m, sym)
        if not len(trades):
            print(f"    {sym}: no trades")
            continue
        closed = trades[trades["outcome"] != "open"]
        n = len(closed)
        wins = (closed["outcome"] == "win").sum()
        wr = wins / n * 100 if n else 0
        r = closed["pnl_r"].sum()
        _, dd, ls, ws, _ = streak_dd(closed)
        per_pair.append({
            "symbol": sym, "n": n, "wr": wr, "r": r, "dd": dd,
            "ls": ls, "ws": ws, "days": span_days,
            "range": f"{df_1m.index[0].date()} → {df_1m.index[-1].date()}",
        })
        all_trades.append(trades)

    if not all_trades:
        print("No trades at all.")
        mt5.shutdown()
        return

    # ── Per-pair table ──
    print("\n" + "=" * 96)
    print("PER-PAIR RESULTS")
    print("=" * 96)
    print(f"  {'Symbol':8}  {'Days':>4}  {'N':>4}  {'WR':>6}  {'R':>7}  "
          f"{'DD':>7}  {'LStrk':>5}  {'WStrk':>5}  Date range")
    for p in per_pair:
        print(f"  {p['symbol']:8}  {p['days']:>4}  {p['n']:>4}  {p['wr']:>5.1f}%  "
              f"{p['r']:>+6.1f}R  {p['dd']:>6.1f}%  {p['ls']:>5}  {p['ws']:>5}  {p['range']}")

    # ── Combined ──
    combined = pd.concat(all_trades, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    combined_closed = combined[combined["outcome"] != "open"]
    N = len(combined_closed)
    W = (combined_closed["outcome"] == "win").sum()
    L = N - W
    WR = W / N * 100
    R = combined_closed["pnl_r"].sum()
    bal, dd, ls, ws, eq_arr = streak_dd(combined_closed)
    gw = W * RR
    gl = L
    pf = gw / gl if gl else float("inf")
    print("\n" + "=" * 96)
    print("COMBINED (all majors, mixed chronologically)")
    print("=" * 96)
    print(f"  Total trades       : {N} ({W} W / {L} L)")
    print(f"  Win rate           : {WR:.1f}%")
    print(f"  Profit factor      : {pf:.2f}")
    print(f"  Total R            : {R:+.1f}R")
    print(f"  Max drawdown       : {dd:.1f}%")
    print(f"  Longest win streak : {ws}")
    print(f"  Longest loss streak: {ls}")
    print(f"  $10k → ${bal:,.0f}  ({(bal/10000-1)*100:+.1f}%)")

    # ── Save + plot ──
    combined.to_csv(os.path.join(OUT, "trades.csv"), index=False)
    plot_equity(eq_arr, f"Asia sweep + 1M CISD · 7 majors · RR={int(RR)} · realistic",
                os.path.join(OUT, "equity.png"))
    print(f"\n  Saved trades.csv, equity.png → {OUT}/")

    # ── Sample trade: pick best pair's most recent win with nice stop distance ──
    best_pair = max(per_pair, key=lambda p: p["r"])
    print(f"\n  Best pair: {best_pair['symbol']}  ({best_pair['r']:+.1f}R, "
          f"WR {best_pair['wr']:.1f}%)")
    best_df_1m = fetch_m1(best_pair["symbol"])
    best_trades = combined[combined["symbol"] == best_pair["symbol"]]
    wins = best_trades[best_trades["outcome"] == "win"].sort_values("entry_time")
    sample = wins.iloc[-1] if len(wins) else best_trades.iloc[-1]
    plot_sample_trade(best_df_1m, sample.to_dict(),
                      os.path.join(OUT, "sample_trade.png"))
    print(f"  Sample winning trade: {best_pair['symbol']} {sample['direction']} "
          f"{sample['entry_time']} · {sample['pnl_r']:+.1f}R")
    print(f"  Saved sample_trade.png → {OUT}/")

    mt5.shutdown()


if __name__ == "__main__":
    main()
