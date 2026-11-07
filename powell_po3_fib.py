"""
Powell's PO3 — FIB OTE version.

Rules:
  · HTF bias   = direction of previous D1 candle
  · Setup window = broker 16:00-20:00 (= 10am NY on FTMO)
  · 5M rejection bar identifies the manipulation extreme (L for long, H for short)
  · After rejection, track the impulse swing:
      - long : need highest_high >= rejection_close + 2 × (rejection_close - L)
      - short: need lowest_low  <= rejection_close - 2 × (H - rejection_close)
  · Once impulse is validated, place LIMIT at 79% OTE fib:
      - long : OTE = 0.79 × L + 0.21 × H_max
      - short: OTE = 0.79 × H + 0.21 × L_min
  · Setup dies if price returns to L (long) or H (short) BEFORE the OTE fills
  · Setup times out after 48 bars (4 hours)
  · SL = manipulation extreme; TP = 5R
  · One trade per broker day maximum
  · Realistic cost model applied post-hoc (spread + slip)
"""
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import main as bt

RR                 = 5.0
NY_10AM_START_HOUR = 16
NY_10AM_END_HOUR   = 20
LOOKBACK_SWING     = 4
IMPULSE_MULT       = 2.0        # must reach 2× rejection-body range for confirmation
OTE                = 0.79       # fib 79% retracement
INVALIDATION_BARS  = 48         # bars to wait for OTE fill
MAX_HOLD_BARS      = 800

TOP3 = ["US500.cash", "GBPJPY", "UKOIL.cash"]
ALL21 = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "NZDJPY",
    "EURAUD", "GBPAUD",
    "XAUUSD",
    "US500.cash", "US100.cash", "GER40.cash", "UK100.cash",
    "FRA40.cash", "JP225.cash", "HK50.cash",
    "USOIL.cash", "UKOIL.cash",
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


def compute_bias(df_d1: pd.DataFrame) -> dict:
    prev_open  = df_d1["open"].shift(1)
    prev_close = df_d1["close"].shift(1)
    bias = np.where(prev_close > prev_open, "bullish", "bearish")
    return {d: str(b) for d, b in zip(df_d1.index.date, bias)}


def find_setups(df_5m: pd.DataFrame, bias_map: dict):
    o = df_5m["open"].values
    h = df_5m["high"].values
    l = df_5m["low"].values
    c = df_5m["close"].values
    idx = df_5m.index
    hours = df_5m.index.hour
    in_window = (hours >= NY_10AM_START_HOUR) & (hours < NY_10AM_END_HOUR)

    setups = []
    last_date = None
    used_today = False

    for i in range(LOOKBACK_SWING + 1, len(df_5m) - INVALIDATION_BARS - 1):
        d = idx[i].date()
        if d != last_date:
            last_date = d; used_today = False
        if used_today or not in_window[i]:
            continue
        bias = bias_map.get(d)
        if bias is None:
            continue

        prior_lo = l[i - LOOKBACK_SWING:i].min()
        prior_hi = h[i - LOOKBACK_SWING:i].max()

        if bias == "bullish":
            # Rejection bar: sweeps swing low, closes bullish
            if not (l[i] < prior_lo and c[i] > o[i]):
                continue
            L = float(l[i])
            rejection_close = float(c[i])
            rejection_range = rejection_close - L
            if rejection_range <= 0:
                continue

            # Track impulse: need highest_high >= rejection_close + IMPULSE_MULT * range
            H_max = float(h[i])
            impulse_target = rejection_close + IMPULSE_MULT * rejection_range
            impulse_confirmed = False
            entry_price = None; entry_idx = None
            for j in range(i + 1, min(i + INVALIDATION_BARS, len(df_5m))):
                # invalidation
                if l[j] <= L:
                    break
                H_max = max(H_max, float(h[j]))
                if not impulse_confirmed:
                    if H_max >= impulse_target:
                        impulse_confirmed = True
                    else:
                        continue                # still waiting for impulse
                # Impulse confirmed → look for OTE pullback
                ote_price = OTE * L + (1 - OTE) * H_max
                if l[j] <= ote_price and ote_price > L:
                    entry_price = ote_price; entry_idx = j; break
            if entry_price is None:
                continue
            risk = entry_price - L
            if risk <= 0:
                continue
            setups.append({
                "time": idx[i], "entry_time": idx[entry_idx],
                "direction": "long",
                "entry": entry_price, "sl": L,
                "tp": entry_price + RR * risk,
                "entry_idx": entry_idx, "risk_price": risk,
                "H_max": H_max,
            })
            used_today = True

        else:  # bearish
            if not (h[i] > prior_hi and c[i] < o[i]):
                continue
            H = float(h[i])
            rejection_close = float(c[i])
            rejection_range = H - rejection_close
            if rejection_range <= 0:
                continue

            L_min = float(l[i])
            impulse_target = rejection_close - IMPULSE_MULT * rejection_range
            impulse_confirmed = False
            entry_price = None; entry_idx = None
            for j in range(i + 1, min(i + INVALIDATION_BARS, len(df_5m))):
                if h[j] >= H:
                    break
                L_min = min(L_min, float(l[j]))
                if not impulse_confirmed:
                    if L_min <= impulse_target:
                        impulse_confirmed = True
                    else:
                        continue
                ote_price = OTE * H + (1 - OTE) * L_min
                if h[j] >= ote_price and ote_price < H:
                    entry_price = ote_price; entry_idx = j; break
            if entry_price is None:
                continue
            risk = H - entry_price
            if risk <= 0:
                continue
            setups.append({
                "time": idx[i], "entry_time": idx[entry_idx],
                "direction": "short",
                "entry": entry_price, "sl": H,
                "tp": entry_price - RR * risk,
                "entry_idx": entry_idx, "risk_price": risk,
                "L_min": L_min,
            })
            used_today = True

    return setups


def simulate(df_5m: pd.DataFrame, setup: dict):
    h = df_5m["high"].values; l = df_5m["low"].values; idx = df_5m.index
    ei = setup["entry_idx"]
    d = setup["direction"]; sl = setup["sl"]; tp = setup["tp"]
    end = min(ei + MAX_HOLD_BARS, len(df_5m))
    for j in range(ei + 1, end):
        if d == "long":
            if l[j] <= sl:  return "loss", idx[j], sl, -1.0
            if h[j] >= tp:  return "win",  idx[j], tp,  RR
        else:
            if h[j] >= sl:  return "loss", idx[j], sl, -1.0
            if l[j] <= tp:  return "win",  idx[j], tp,  RR
    return "open", idx[end - 1], np.nan, 0.0


def run_symbol(sym: str):
    try:
        df_d1 = bt.fetch_data(sym, mt5.TIMEFRAME_D1, bt.START_DATE, bt.END_DATE, tf_name="D1")
        df_5m = bt.fetch_data(sym, mt5.TIMEFRAME_M5, bt.START_DATE, bt.END_DATE, tf_name="M5")
    except Exception as e:
        return {"symbol": sym, "error": str(e)}
    if len(df_d1) < 30 or len(df_5m) < 500:
        return {"symbol": sym, "error": "insufficient data"}

    bias_map = compute_bias(df_d1)
    setups = find_setups(df_5m, bias_map)
    trades = []
    for s in setups:
        oc, xt, xp, pr = simulate(df_5m, s)
        trades.append({
            "symbol": sym,
            "signal_time": s["time"], "entry_time": s["entry_time"],
            "direction": s["direction"],
            "entry": s["entry"], "sl": s["sl"], "tp": s["tp"],
            "risk_price": s["risk_price"],
            "outcome": oc, "exit_time": xt, "exit": xp, "pnl_r": pr,
        })
    return {"symbol": sym, "trades": trades, "n_days": len(bias_map)}


def stats_naive(trades_df: pd.DataFrame):
    cl = trades_df[trades_df["outcome"] != "open"].copy()
    if not len(cl): return None
    n = len(cl); wins = (cl["outcome"] == "win").sum(); losses = n - wins
    wr = wins / n * 100
    gw = wins * RR; gl = losses
    pf = gw / gl if gl > 0 else float("inf")
    tot = cl["pnl_r"].sum()

    bal = 10_000.0; eq = [bal]; cs = cw = bcl = bcw = 0
    for _, r in cl.sort_values("entry_time").iterrows():
        bal += bal * 0.01 * float(r["pnl_r"]); eq.append(bal)
        if r["pnl_r"] < -0.01: cs += 1; cw = 0; bcl = max(bcl, cs)
        elif r["pnl_r"] > 0.01: cw += 1; cs = 0; bcw = max(bcw, cw)
    eqa = np.array(eq)
    dd = ((eqa - np.maximum.accumulate(eqa)) / np.maximum.accumulate(eqa) * 100).min()
    return dict(n=n, wr=wr, pf=pf, tot=tot, dd=dd, ls=bcl, ws=bcw, bal=bal)


def stats_realistic(trades_df: pd.DataFrame):
    cl = trades_df[trades_df["outcome"] != "open"].copy()
    if not len(cl): return None
    tot_r_after = 0.0
    per_row = []
    wins = losses = 0
    for _, r in cl.iterrows():
        sym = r["symbol"]
        if sym not in COSTS: continue
        spread, slip = COSTS[sym]
        rp = float(r["risk_price"])
        if rp <= 0: continue
        if r["outcome"] == "win":
            r_after = (RR * rp - spread) / rp
            wins += 1
        else:
            r_after = -(rp + spread + slip) / rp
            losses += 1
        tot_r_after += r_after; per_row.append(r_after)
    n = wins + losses
    if n == 0: return None
    wr = wins / n * 100
    gw = sum(x for x in per_row if x > 0)
    gl = -sum(x for x in per_row if x < 0)
    pf = gw / gl if gl > 0 else float("inf")
    return dict(n=n, wr=wr, pf=pf, tot=tot_r_after)


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    for label, symbols in [("TOP-3", TOP3), ("ALL-21", ALL21)]:
        print("=" * 108)
        print(f"POWELL'S PO3 + FIB OTE · {label} · RR={RR} · impulse={IMPULSE_MULT}×  · OTE={int(OTE*100)}%")
        print("=" * 108)
        print(f"  {'symbol':13} {'N':>4} {'/wk':>5} {'WR':>6} {'PF':>5} "
              f"{'totR':>7} {'DD':>7} {'LStrk':>5} {'realistic':>25}")

        all_trades = []
        for s in symbols:
            r = run_symbol(s)
            if "error" in r:
                print(f"  {s:13} · {r['error']}")
                continue
            df = pd.DataFrame(r["trades"])
            if not len(df):
                print(f"  {s:13} · no setups"); continue
            st_naive = stats_naive(df)
            st_real  = stats_realistic(df)
            if st_naive is None: continue
            per_wk = st_naive["n"] / (r["n_days"] / 7)
            real_str = (f"WR{st_real['wr']:.0f}% PF{st_real['pf']:.2f} R{st_real['tot']:+.0f}"
                        if st_real else "n/a")
            print(f"  {s:13} {st_naive['n']:>4} {per_wk:>4.1f} "
                  f"{st_naive['wr']:>5.1f}% {st_naive['pf']:>4.2f} "
                  f"{st_naive['tot']:>+6.0f}R {st_naive['dd']:>6.1f}% "
                  f"{st_naive['ls']:>5}  {real_str:>25}")
            all_trades.extend(r["trades"])

        if all_trades:
            all_df = pd.DataFrame(all_trades)
            all_df = all_df[all_df["outcome"] != "open"].sort_values("entry_time")
            print("-" * 108)
            st_all_naive = stats_naive(all_df)
            st_all_real  = stats_realistic(all_df)
            print(f"  COMBINED     {st_all_naive['n']:>4}       "
                  f"{st_all_naive['wr']:>5.1f}% {st_all_naive['pf']:>4.2f} "
                  f"{st_all_naive['tot']:>+6.0f}R {st_all_naive['dd']:>6.1f}%  "
                  f"realistic: WR{st_all_real['wr']:.0f}% PF{st_all_real['pf']:.2f} R{st_all_real['tot']:+.0f}")
            fn = f"powell_po3_fib_{label.lower()}_trades.csv"
            all_df.to_csv(fn, index=False)
            print(f"  Saved → {fn}")
        print()

    mt5.shutdown()


if __name__ == "__main__":
    main()
