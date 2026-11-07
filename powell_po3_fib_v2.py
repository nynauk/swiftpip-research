"""
Powell's PO3 + Fib OTE — V2 with streak reducers and futures.

FUNDAMENTALS UNCHANGED:
  · HTF bias from previous D1 direction
  · 5M rejection block at 10am NY 4H window
  · Fib 79% OTE entry after impulse confirmation
  · SL at manipulation extreme, TP at 5R

QUALITY FILTERS ADDED (increase WR, reduce streaks):
  · F1: manipulation must occur in FIRST HOUR of the window (16:00-17:00 broker)
        — the classic PO3 interpretation: manipulation happens near the open
  · F2: impulse threshold bumped 2× → 3× rejection range
        — only high-conviction reversals
  · F3: PORTFOLIO DAILY CAP = 5 trades / day (earliest signals across all symbols)
        — prevents "all 21 symbols fail together" cluster losses

FUTURES ADDED:
  · Tries common naming across brokers (ES, NQ, GC, CL, 6E, etc.)
"""
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import main as bt

RR                        = 5.0
NY_10AM_START_HOUR        = 16
NY_10AM_END_HOUR          = 20
MANIPULATION_HOUR_END     = 17     # F1: manipulation must fire in [16:00, 17:00)
LOOKBACK_SWING            = 4
IMPULSE_MULT              = 3.0    # F2: was 2.0
OTE                       = 0.79
INVALIDATION_BARS         = 48
MAX_HOLD_BARS             = 800
MAX_TRADES_PER_DAY        = 5      # F3: portfolio cap

ALL21 = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "NZDJPY",
    "EURAUD", "GBPAUD",
    "XAUUSD",
    "US500.cash", "US100.cash", "GER40.cash", "UK100.cash",
    "FRA40.cash", "JP225.cash", "HK50.cash",
    "USOIL.cash", "UKOIL.cash",
]

# Try common futures symbol conventions — filtered at runtime by what broker offers
FUTURES_CANDIDATES = [
    "ES", "NQ", "YM", "RTY",              # US index futures
    "MES", "MNQ", "MYM", "M2K",           # micro US index
    "ES.f", "NQ.f", "YM.f", "RTY.f",      # with .f suffix
    "SPX500", "SPX", "SPX.f",             # alt naming
    "USTEC", "USTECH", "USTECH100",
    "GC", "SI", "HG", "PA", "PL",         # metals futures
    "GC.f", "SI.f",
    "CL", "BZ", "NG", "HO", "RB",         # energy futures
    "CL.f", "NG.f",
    "6E", "6B", "6J", "6A", "6C", "6S",   # FX futures
    "ZN", "ZB", "ZF", "ZT",               # bond futures
    "NKD",                                 # Nikkei futures
    "VX",                                  # VIX futures
    "BTC.f", "ETH.f",                     # crypto futures
]

COSTS = {
    "EURUSD":     (0.00015, 0.00003),
    "GBPUSD":     (0.00020, 0.00003),
    "USDJPY":     (0.020,   0.003),
    "USDCHF":     (0.00020, 0.00003),
    "EURJPY":     (0.025,   0.005),
    "GBPJPY":     (0.021,   0.005),
    "AUDJPY":     (0.025,   0.005),
    "CADJPY":     (0.025,   0.005),
    "NZDJPY":     (0.030,   0.005),
    "EURAUD":     (0.00030, 0.00005),
    "GBPAUD":     (0.00040, 0.00005),
    "XAUUSD":     (0.30,    0.05),
    "US500.cash": (0.55,    0.05),
    "US100.cash": (2.00,    0.20),
    "GER40.cash": (1.50,    0.20),
    "UK100.cash": (1.00,    0.15),
    "FRA40.cash": (1.50,    0.20),
    "JP225.cash": (10.0,    2.0),
    "HK50.cash":  (10.0,    2.0),
    "USOIL.cash": (0.05,    0.01),
    "UKOIL.cash": (0.063,   0.01),
}


def default_cost(sym: str):
    """Guess cost for unknown futures by symbol family."""
    sym_u = sym.upper()
    if any(x in sym_u for x in ["ES", "SPX", "SPY"]):    return (0.25, 0.05)
    if any(x in sym_u for x in ["NQ", "USTEC", "MNQ"]):  return (1.00, 0.20)
    if any(x in sym_u for x in ["YM", "MYM"]):           return (2.00, 0.50)
    if any(x in sym_u for x in ["RTY", "M2K"]):          return (0.20, 0.05)
    if "GC" in sym_u:                                    return (0.30, 0.05)
    if "SI" in sym_u:                                    return (0.015, 0.005)
    if "CL" in sym_u or "BZ" in sym_u:                   return (0.05, 0.01)
    if sym_u.startswith("6E") or sym_u.startswith("6B"): return (0.00015, 0.00003)
    if sym_u.startswith("6J"):                           return (0.00003, 0.00001)
    if "NKD" in sym_u:                                   return (10.0, 2.0)
    if "ZN" in sym_u or "ZB" in sym_u:                   return (0.015, 0.005)
    return (0.5, 0.1)   # fallback


def compute_bias(df_d1: pd.DataFrame) -> dict:
    prev_open  = df_d1["open"].shift(1)
    prev_close = df_d1["close"].shift(1)
    bias = np.where(prev_close > prev_open, "bullish", "bearish")
    return {d: str(b) for d, b in zip(df_d1.index.date, bias)}


def find_setups_v2(df_5m: pd.DataFrame, bias_map: dict, sym: str):
    o = df_5m["open"].values; h = df_5m["high"].values
    l = df_5m["low"].values;  c = df_5m["close"].values
    idx = df_5m.index; hours = df_5m.index.hour

    # F1: manipulation only in [16:00, 17:00) broker time
    in_manip_window = (hours >= NY_10AM_START_HOUR) & (hours < MANIPULATION_HOUR_END)

    setups = []
    last_date = None; used_today = False

    for i in range(LOOKBACK_SWING + 1, len(df_5m) - INVALIDATION_BARS - 1):
        d = idx[i].date()
        if d != last_date:
            last_date = d; used_today = False
        if used_today or not in_manip_window[i]:
            continue
        bias = bias_map.get(d)
        if bias is None:
            continue

        prior_lo = l[i - LOOKBACK_SWING:i].min()
        prior_hi = h[i - LOOKBACK_SWING:i].max()

        if bias == "bullish":
            if not (l[i] < prior_lo and c[i] > o[i]):
                continue
            L = float(l[i]); rc = float(c[i]); rr = rc - L
            if rr <= 0: continue

            H_max = float(h[i])
            impulse_target = rc + IMPULSE_MULT * rr    # F2: 3× threshold
            impulse_confirmed = False
            entry_price = None; entry_idx = None
            for j in range(i + 1, min(i + INVALIDATION_BARS, len(df_5m))):
                if l[j] <= L: break
                H_max = max(H_max, float(h[j]))
                if not impulse_confirmed:
                    if H_max >= impulse_target: impulse_confirmed = True
                    else: continue
                ote_price = OTE * L + (1 - OTE) * H_max
                if l[j] <= ote_price and ote_price > L:
                    entry_price = ote_price; entry_idx = j; break
            if entry_price is None: continue
            risk = entry_price - L
            if risk <= 0: continue
            setups.append({
                "symbol": sym, "signal_time": idx[i], "entry_time": idx[entry_idx],
                "direction": "long",
                "entry": entry_price, "sl": L,
                "tp": entry_price + RR * risk,
                "entry_idx": entry_idx, "risk_price": risk,
            })
            used_today = True

        else:
            if not (h[i] > prior_hi and c[i] < o[i]):
                continue
            H = float(h[i]); rc = float(c[i]); rr = H - rc
            if rr <= 0: continue

            L_min = float(l[i])
            impulse_target = rc - IMPULSE_MULT * rr    # F2: 3× threshold
            impulse_confirmed = False
            entry_price = None; entry_idx = None
            for j in range(i + 1, min(i + INVALIDATION_BARS, len(df_5m))):
                if h[j] >= H: break
                L_min = min(L_min, float(l[j]))
                if not impulse_confirmed:
                    if L_min <= impulse_target: impulse_confirmed = True
                    else: continue
                ote_price = OTE * H + (1 - OTE) * L_min
                if h[j] >= ote_price and ote_price < H:
                    entry_price = ote_price; entry_idx = j; break
            if entry_price is None: continue
            risk = H - entry_price
            if risk <= 0: continue
            setups.append({
                "symbol": sym, "signal_time": idx[i], "entry_time": idx[entry_idx],
                "direction": "short",
                "entry": entry_price, "sl": H,
                "tp": entry_price - RR * risk,
                "entry_idx": entry_idx, "risk_price": risk,
            })
            used_today = True
    return setups


def simulate(df_5m: pd.DataFrame, setup: dict):
    h = df_5m["high"].values; l = df_5m["low"].values; idx = df_5m.index
    ei = setup["entry_idx"]; d = setup["direction"]
    sl = setup["sl"]; tp = setup["tp"]
    end = min(ei + MAX_HOLD_BARS, len(df_5m))
    for j in range(ei + 1, end):
        if d == "long":
            if l[j] <= sl:  return "loss", idx[j], sl, -1.0
            if h[j] >= tp:  return "win",  idx[j], tp,  RR
        else:
            if h[j] >= sl:  return "loss", idx[j], sl, -1.0
            if l[j] <= tp:  return "win",  idx[j], tp,  RR
    return "open", idx[end - 1], np.nan, 0.0


def fetch_symbol_data(sym: str):
    """Try to fetch D1 + M5 for a symbol; return None if unavailable."""
    info = mt5.symbol_info(sym)
    if info is None:
        return None, None
    if not mt5.symbol_select(sym, True):
        return None, None
    try:
        df_d1 = bt.fetch_data(sym, mt5.TIMEFRAME_D1, bt.START_DATE, bt.END_DATE, tf_name="D1")
        df_5m = bt.fetch_data(sym, mt5.TIMEFRAME_M5, bt.START_DATE, bt.END_DATE, tf_name="M5")
    except Exception:
        return None, None
    if len(df_d1) < 30 or len(df_5m) < 500:
        return None, None
    return df_d1, df_5m


def stats_naive(df: pd.DataFrame):
    cl = df[df["outcome"] != "open"].copy()
    if not len(cl): return None
    n = len(cl); wins = (cl["outcome"] == "win").sum(); losses = n - wins
    wr = wins / n * 100
    pf = (wins * RR) / losses if losses else float("inf")
    tot = cl["pnl_r"].sum()
    bal = 10_000.0; eq = [bal]; cs = cw = bcl = bcw = 0
    for _, r in cl.sort_values("entry_time").iterrows():
        bal += bal * 0.01 * float(r["pnl_r"]); eq.append(bal)
        if r["pnl_r"] < -0.01: cs += 1; cw = 0; bcl = max(bcl, cs)
        elif r["pnl_r"] > 0.01: cw += 1; cs = 0; bcw = max(bcw, cw)
    eqa = np.array(eq)
    dd = ((eqa - np.maximum.accumulate(eqa)) / np.maximum.accumulate(eqa) * 100).min()
    return dict(n=n, wr=wr, pf=pf, tot=tot, dd=dd, ls=bcl, ws=bcw, bal=bal)


def stats_realistic(df: pd.DataFrame, cost_map: dict):
    cl = df[df["outcome"] != "open"].copy()
    if not len(cl): return None
    tot_r = 0.0; per = []; wins = losses = 0
    for _, r in cl.iterrows():
        sp, sl_ = cost_map.get(r["symbol"], default_cost(r["symbol"]))
        rp = float(r["risk_price"])
        if rp <= 0: continue
        if r["outcome"] == "win":
            v = (RR * rp - sp) / rp; wins += 1
        else:
            v = -(rp + sp + sl_) / rp; losses += 1
        tot_r += v; per.append(v)
    n = wins + losses
    if n == 0: return None
    wr = wins / n * 100
    gw = sum(x for x in per if x > 0); gl = -sum(x for x in per if x < 0)
    pf = gw / gl if gl else float("inf")
    return dict(n=n, wr=wr, pf=pf, tot=tot_r)


def streaks(outcomes):
    runs = []; cur_t = None; cur_l = 0
    for o in outcomes:
        t = "win" if o == "win" else "loss"
        if t == cur_t: cur_l += 1
        else:
            if cur_t is not None: runs.append((cur_t, cur_l))
            cur_t = t; cur_l = 1
    runs.append((cur_t, cur_l))
    return runs


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    # ── Cache all symbol data + collect all setups ──
    print("=" * 108)
    print("POWELL PO3 + FIB OTE V2  ·  filters: manip-first-hour, impulse=3×, daily-cap=5")
    print("=" * 108)

    # Discover which futures are available
    futures_avail = []
    for f in FUTURES_CANDIDATES:
        if mt5.symbol_info(f) is not None:
            futures_avail.append(f)
    if futures_avail:
        print(f"  Futures available on broker: {futures_avail}")
    else:
        print(f"  No futures symbols matched from candidate list — broker doesn't offer them")

    ALL_SYMBOLS = ALL21 + futures_avail
    print(f"  Total symbols scanning: {len(ALL_SYMBOLS)}")
    print()

    cache = {}    # sym → df_5m
    all_setups = []
    for sym in ALL_SYMBOLS:
        df_d1, df_5m = fetch_symbol_data(sym)
        if df_5m is None:
            print(f"  {sym:14} · unavailable / insufficient data — skipped")
            continue
        cache[sym] = df_5m
        bias_map = compute_bias(df_d1)
        setups = find_setups_v2(df_5m, bias_map, sym)
        all_setups.extend(setups)
        print(f"  {sym:14} · raw setups {len(setups):>4}")

    print()
    print(f"  Total raw setups across all symbols: {len(all_setups)}")

    # F3: portfolio daily cap — sort by entry time, keep at most N per date
    all_setups.sort(key=lambda s: s["entry_time"])
    kept = []; per_day_count = {}
    for s in all_setups:
        d = s["entry_time"].date()
        c = per_day_count.get(d, 0)
        if c >= MAX_TRADES_PER_DAY: continue
        per_day_count[d] = c + 1
        kept.append(s)
    print(f"  After daily cap of {MAX_TRADES_PER_DAY}: {len(kept)} setups "
          f"(dropped {len(all_setups) - len(kept)} late-in-day duplicates)")

    # Simulate each kept setup on its symbol's cached df
    trades = []
    for s in kept:
        df_5m = cache[s["symbol"]]
        oc, xt, xp, pr = simulate(df_5m, s)
        trades.append({
            "symbol": s["symbol"], "signal_time": s["signal_time"],
            "entry_time": s["entry_time"], "direction": s["direction"],
            "entry": s["entry"], "sl": s["sl"], "tp": s["tp"],
            "risk_price": s["risk_price"],
            "outcome": oc, "exit_time": xt, "exit": xp, "pnl_r": pr,
        })

    df_all = pd.DataFrame(trades)
    df_all = df_all[df_all["outcome"] != "open"].sort_values("entry_time").reset_index(drop=True)

    # Cost map — merge user COSTS with default_cost for futures
    cost_map = {**COSTS}
    for s in futures_avail:
        cost_map[s] = default_cost(s)

    # Report
    st_naive = stats_naive(df_all)
    st_real  = stats_realistic(df_all, cost_map)

    print("\n" + "=" * 108)
    print("HEADLINE RESULTS")
    print("=" * 108)
    print(f"  Trades      : {st_naive['n']}")
    print(f"  Win rate    : {st_naive['wr']:.2f}%")
    print(f"  PF (naive)  : {st_naive['pf']:.2f}   R_naive    +{st_naive['tot']:.0f}R")
    print(f"  PF (real)   : {st_real['pf']:.2f}   R_realistic {st_real['tot']:+.0f}R")
    print(f"  DD @ 1%     : {st_naive['dd']:.1f}%")

    runs = streaks(df_all["outcome"].values)
    loss_lens = [L for t, L in runs if t == "loss"]
    win_lens  = [L for t, L in runs if t == "win"]
    print(f"\n  Loss streaks: total {len(loss_lens)}  "
          f"longest {max(loss_lens)}  mean {np.mean(loss_lens):.2f}  median {int(np.median(loss_lens))}")
    print(f"  Win streaks : total {len(win_lens)}  "
          f"longest {max(win_lens)}  mean {np.mean(win_lens):.2f}")

    print("\n  Loss streak distribution:")
    buckets = {"1":0, "2-3":0, "4-6":0, "7-10":0, "11-15":0, "16+":0}
    for L in loss_lens:
        if L == 1: buckets["1"] += 1
        elif L <= 3: buckets["2-3"] += 1
        elif L <= 6: buckets["4-6"] += 1
        elif L <= 10: buckets["7-10"] += 1
        elif L <= 15: buckets["11-15"] += 1
        else: buckets["16+"] += 1
    for k, v in buckets.items():
        print(f"    {k:>6}: {v:>4}  ({v/len(loss_lens)*100:>5.1f}%)")

    # Per-symbol realistic
    print("\n  Per-symbol (kept trades):")
    print(f"    {'symbol':14} {'N':>4} {'WR':>6} {'PF_r':>6} {'R_r':>7}")
    for sym, g in df_all.groupby("symbol"):
        st = stats_realistic(g, cost_map)
        if st is None: continue
        print(f"    {sym:14} {st['n']:>4}  {st['wr']:>4.1f}%  {st['pf']:>4.2f}  {st['tot']:>+6.0f}R")

    # Comparison to V1
    print("\n" + "=" * 108)
    print("V1 vs V2 COMPARISON (ALL-21, realistic)")
    print("=" * 108)
    print(f"  {'metric':22} {'V1':>10} {'V2':>10} {'Δ':>10}")
    v1 = dict(n=3336, wr=32.3, pf=1.24, tot=906, ls=25, dd=46.4)
    v2 = dict(n=st_naive['n'], wr=st_real['wr'], pf=st_real['pf'],
              tot=st_real['tot'], ls=max(loss_lens), dd=st_naive['dd'])
    for m, fmt in [("Trades", "{:.0f}"), ("WR (%)", "{:.1f}"),
                   ("PF realistic", "{:.2f}"), ("Total R realistic", "{:+.0f}"),
                   ("Longest streak", "{:.0f}"), ("DD @1% (%)", "{:.1f}")]:
        k = m.split()[0].lower()
        if k == "trades": v1v, v2v = v1["n"], v2["n"]
        elif k == "wr":   v1v, v2v = v1["wr"], v2["wr"]
        elif k == "pf":   v1v, v2v = v1["pf"], v2["pf"]
        elif k == "total":v1v, v2v = v1["tot"], v2["tot"]
        elif k == "longest": v1v, v2v = v1["ls"], v2["ls"]
        elif k == "dd":   v1v, v2v = v1["dd"], v2["dd"]
        delta = v2v - v1v
        print(f"  {m:22} {fmt.format(v1v):>10} {fmt.format(v2v):>10} {fmt.format(delta):>10}")

    df_all.to_csv("powell_po3_fib_v2_trades.csv", index=False)
    print(f"\n  Saved → powell_po3_fib_v2_trades.csv ({len(df_all)} trades)")

    mt5.shutdown()


if __name__ == "__main__":
    main()
