"""
Bollinger Band + RSI mean-reversion backtest — honest (real spread, causal).

Signal (all on CLOSED bars, indicators use only trailing data):
  bands  = SMA(close, BB_N) ± BB_K · rollingstd(close, BB_N)
  rsi    = Wilder RSI(RSI_N)
  LONG   = close pierces below lower band AND rsi < RSI_OS   (fresh cross only)
  SHORT  = close pierces above upper band AND rsi > RSI_OB   (fresh cross only)

Entry at the signal bar close. One position at a time (flat before re-entry).
  STOP = signal-bar extreme ± buffer (+ spread on the adverse side)
  Exits compared:
    MEAN   — take profit when price returns to the middle band (variable R)
    FIXED  — take profit at BB_FIXED_RR · risk
  Long fills at ask, short exits at ask → spread charged honestly, same model as
  the CISD backtests.

    python -m asia_bot.backtest_bbrsi                 # EURUSD M5
    python -m asia_bot.backtest_bbrsi EURUSD GBPUSD   # several pairs
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asia_bot import config as cfg
from asia_bot.load_histdata import build_m5

PAIRS = ["EURUSD"]
SPREAD_PIPS = {"EURUSD": 1.0, "GBPUSD": 1.5, "USDJPY": 1.0, "USDCHF": 1.5,
               "AUDUSD": 1.5, "USDCAD": 1.5, "NZDUSD": 2.0}
OUT = os.path.join(os.path.dirname(__file__), "backtest")

BB_N = 20
BB_K = 2.0
RSI_N = 14
RSI_OS = 30
RSI_OB = 70
TREND_EMA_N = 200      # trend filter: longs only above it, shorts only below
TREND_FILTER = True    # buy dips in uptrends / sell rallies in downtrends
BUF_PIPS = 1.0
MIN_SL_PIPS = 5.0       # floor — you can't really risk 1% on a sub-pip stop
                        # (spread+slippage+lot granularity); without this, tiny
                        # signal candles produce absurd R and fake PF.
BB_FIXED_RR = 1.5
MAX_HOLD = 288          # M5 bars ≈ 24h


def rsi_wilder(close, n):
    d = np.diff(close, prepend=close[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1 / n, adjust=False).mean().values
    rd = pd.Series(dn).ewm(alpha=1 / n, adjust=False).mean().values
    rs = np.where(rd == 0, np.inf, ru / rd)
    return 100 - 100 / (1 + rs)


def run_pair(pair, exit_mode):
    df = build_m5(pair)
    if df is None:
        return None
    pip = cfg.PIP[pair]
    spread = SPREAD_PIPS[pair] * pip
    buf = BUF_PIPS * pip
    o = df["open"].values; h = df["high"].values
    l = df["low"].values; c = df["close"].values
    years = df.index.year.values
    n = len(c)

    s = pd.Series(c)
    mid = s.rolling(BB_N).mean().values
    sd = s.rolling(BB_N).std(ddof=0).values
    lower = mid - BB_K * sd
    upper = mid + BB_K * sd
    rsi = rsi_wilder(c, RSI_N)

    long_sig = (c < lower) & (rsi < RSI_OS)
    short_sig = (c > upper) & (rsi > RSI_OB)
    fresh_long = long_sig & ~np.r_[False, long_sig[:-1]]
    fresh_short = short_sig & ~np.r_[False, short_sig[:-1]]

    if TREND_FILTER:
        ema = s.ewm(span=TREND_EMA_N, adjust=False).mean().values
        fresh_long = fresh_long & (c > ema)      # buy dips only in uptrend
        fresh_short = fresh_short & (c < ema)     # sell rallies only in downtrend

    trades = []
    i = BB_N + 1
    while i < n - 1:
        direction = None
        if fresh_long[i]: direction = "long"
        elif fresh_short[i]: direction = "short"
        if direction is None:
            i += 1; continue

        min_sl = MIN_SL_PIPS * pip
        if direction == "long":
            fill = c[i] + spread
            sl = min(l[i] - buf, fill - min_sl)      # floor the stop distance
            risk = fill - sl
        else:
            fill = c[i]
            sl = max(h[i] + buf, fill + min_sl)
            risk = sl - fill
        if risk <= 0:
            i += 1; continue

        outcome = "open"; R = 0.0; exit_i = None
        end = min(i + MAX_HOLD, n)
        for j in range(i + 1, end):
            if direction == "long":
                if l[j] <= sl:
                    R = -(1.0 + spread / risk); outcome = "loss"; exit_i = j; break
                tp = (mid[j] if exit_mode == "mean"
                      else upper[j] if exit_mode == "band"
                      else fill + BB_FIXED_RR * risk)
                if h[j] >= tp:
                    R = (tp - fill) / risk; outcome = "win"; exit_i = j; break
            else:
                if h[j] + spread >= sl:
                    R = -1.0; outcome = "loss"; exit_i = j; break
                tp = (mid[j] if exit_mode == "mean"
                      else lower[j] if exit_mode == "band"
                      else fill - BB_FIXED_RR * risk)
                if l[j] + spread <= tp:
                    R = (fill - tp) / risk; outcome = "win"; exit_i = j; break
        if outcome == "open":
            exit_i = end - 1
            mtm = (c[exit_i] - fill) / risk if direction == "long" else (fill - c[exit_i]) / risk
            R = mtm
        trades.append({"pair": pair, "year": int(years[i]), "direction": direction,
                       "stop_p": round(risk / pip, 1), "outcome": outcome,
                       "pnl_r": round(R, 3)})
        i = exit_i + 1        # flat until the trade closes, then continue
    return pd.DataFrame(trades)


def summ(df):
    cl = df[df["outcome"].isin(["win", "loss"])] if len(df) else df
    if not len(cl):
        return None
    n = len(cl); w = (cl["outcome"] == "win").sum(); R = cl["pnl_r"].sum()
    gw = cl.loc[cl["pnl_r"] > 0, "pnl_r"].sum(); gl = -cl.loc[cl["pnl_r"] < 0, "pnl_r"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    bal = 10000.0; eq = [bal]; ls = c = 0
    for _, t in cl.iterrows():
        ra = cfg.RISK_FLAT if bal < cfg.RISK_ANCHOR else bal * cfg.RISK_PCT / 100
        bal += ra * float(t["pnl_r"]); eq.append(bal)
        if t["pnl_r"] < 0: c += 1; ls = max(ls, c)
        else: c = 0
    e = np.array(eq); dd = ((e - np.maximum.accumulate(e)) / np.maximum.accumulate(e) * 100).min()
    yr = cl.groupby("year")["pnl_r"].sum()
    return dict(n=n, wr=w / n * 100, pf=pf, R=R, exp=R / n, dd=dd, ls=ls,
                green=(yr > 0).sum(), yrs=len(yr), aw=cl.loc[cl.pnl_r > 0, "pnl_r"].mean(),
                al=cl.loc[cl.pnl_r < 0, "pnl_r"].mean())


def main():
    os.makedirs(OUT, exist_ok=True)
    pairs = sys.argv[1:] or PAIRS
    print("=" * 108)
    tf = f"TREND FILTER: {'ON (EMA%d)' % TREND_EMA_N if TREND_FILTER else 'OFF'}"
    print(f"BOLLINGER({BB_N},{BB_K}) + RSI({RSI_N}) · M5 · HistData 2000-2024 · "
          f"OS/OB {RSI_OS}/{RSI_OB} · {tf}")
    print(f"stop = signal extreme +{BUF_PIPS}p +spread · one position at a time · flat-to-reenter")
    print("=" * 108)
    for exit_mode, name in (("mean", "EXIT: middle band"),
                            ("band", "EXIT: opposite band (ride the trend)"),
                            ("fixed", f"EXIT: fixed {BB_FIXED_RR}R")):
        print(f"\n{'#'*108}\n#  {name}\n{'#'*108}")
        print(f"  {'pair':8} {'N':>5} {'WR':>6} {'PF':>5} {'totR':>8} {'exp':>8} "
              f"{'avgW':>6} {'avgL':>6} {'DD':>7} {'maxLS':>5} {'green':>7}")
        allrows = []
        for p in pairs:
            df = run_pair(p, exit_mode)
            if df is None or not len(df):
                print(f"  {p:8} no data/trades"); continue
            allrows.append(df)
            s = summ(df)
            print(f"  {p:8} {s['n']:>5} {s['wr']:>5.1f}% {s['pf']:>5.2f} {s['R']:>+8.1f} "
                  f"{s['exp']:>+7.3f}R {s['aw']:>+5.2f} {s['al']:>+5.2f} {s['dd']:>6.1f}% "
                  f"{s['ls']:>5} {s['green']:>3}/{s['yrs']:<3}")
        if len(allrows) > 1:
            allt = pd.concat(allrows, ignore_index=True)
            s = summ(allt)
            print(f"  {'-'*100}")
            print(f"  {'ALL':8} {s['n']:>5} {s['wr']:>5.1f}% {s['pf']:>5.2f} {s['R']:>+8.1f} "
                  f"{s['exp']:>+7.3f}R {s['aw']:>+5.2f} {s['al']:>+5.2f} {s['dd']:>6.1f}% {s['ls']:>5}")
    print(f"\nparams: BB {BB_N}/{BB_K}, RSI {RSI_N} {RSI_OS}/{RSI_OB}, stop sig-extreme+{BUF_PIPS}p, "
          f"fixed target {BB_FIXED_RR}R, max-hold {MAX_HOLD} bars")


if __name__ == "__main__":
    main()
