"""
Bollinger Band + RSI mean-reversion backtest — EURUSD, HistData 2000-2024.

Rules (all computed on CLOSED bars, strictly causal):
  BB(20, 2) on close, RSI(14) Wilder, ATR(14) Wilder.
  LONG  : close < lower band AND RSI < RSI_LO
  SHORT : close > upper band AND RSI > RSI_HI
  Entry : market at that candle's close.
  Stop  : ATR_MULT x ATR(14).
  Exit  : "mid" -> the middle band (SMA20) captured at entry  (classic MR target)
          "rr"  -> fixed RR x risk
  One position at a time; max hold ~5 trading days.

Fill/spread model (identical convention to asia_bot, already validated):
  bars are BID. LONG fills at ASK (close+spread); its SL/TP are bid levels.
  SHORT fills at BID; its SL/TP are shifted +spread and exits tested at ask.
  So a loss is exactly -1R and a win is +(reward/risk)R, spread included.

    python -m bb_rsi_backtest
"""
import os
import numpy as np
import pandas as pd

from asia_bot import config as cfg

HD5 = os.path.join("ext_data", "HD_EURUSD_5min_2000-2024.pkl")
SYMBOL = "EURUSD"
PIP = 1e-4
SPREAD = 1.0 * PIP

BB_N, BB_K = 20, 2.0
RSI_N, RSI_LO, RSI_HI = 14, 30, 70
ATR_N, ATR_MULT = 14, 1.5
RR_FIXED = 2.0

TFS = {"M5": ("5min", 1440), "M15": ("15min", 480), "H1": ("1h", 120), "H4": ("4h", 30)}


def wilder_rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    au = up.ewm(alpha=1 / n, adjust=False).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def atr(df, n):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def run(df, tp_mode, max_hold):
    c = df["close"]
    ma = c.rolling(BB_N).mean()
    sd = c.rolling(BB_N).std(ddof=0)
    upper, lower = ma + BB_K * sd, ma - BB_K * sd
    rsi = wilder_rsi(c, RSI_N)
    a = atr(df, ATR_N)

    long_sig = (c < lower) & (rsi < RSI_LO)
    short_sig = (c > upper) & (rsi > RSI_HI)

    H = df["high"].values; L = df["low"].values; C = c.values
    MA = ma.values; A = a.values
    ls = long_sig.values; ss = short_sig.values
    idx = df.index
    n = len(df)
    start = max(BB_N, RSI_N, ATR_N) + 2

    trades = []
    i = start
    while i < n - 1:
        if not (ls[i] or ss[i]) or not np.isfinite(A[i]) or A[i] <= 0:
            i += 1; continue
        is_long = bool(ls[i])
        stop_dist = ATR_MULT * A[i]

        if is_long:
            fill = C[i] + SPREAD
            sl = C[i] - stop_dist
            risk = fill - sl
            tp = MA[i] if tp_mode == "mid" else fill + RR_FIXED * risk
            if risk <= 0 or tp <= fill:
                i += 1; continue
        else:
            fill = C[i]
            sl = C[i] + stop_dist + SPREAD
            risk = sl - fill
            tp = MA[i] if tp_mode == "mid" else fill - RR_FIXED * risk
            if risk <= 0 or tp >= fill:
                i += 1; continue
        rr = (tp - fill) / risk if is_long else (fill - tp) / risk

        outcome, R, xi = "open", 0.0, None
        for j in range(i + 1, min(i + max_hold, n)):
            if is_long:
                if L[j] <= sl: outcome, R, xi = "loss", -1.0, j; break
                if H[j] >= tp: outcome, R, xi = "win", rr, j; break
            else:
                if H[j] + SPREAD >= sl: outcome, R, xi = "loss", -1.0, j; break
                if L[j] + SPREAD <= tp: outcome, R, xi = "win", rr, j; break
        if outcome == "open":
            i += 1; continue
        trades.append({"time": idx[i], "year": idx[i].year,
                       "dir": "long" if is_long else "short",
                       "stop_p": round(risk / PIP, 1), "rr": round(rr, 2),
                       "outcome": outcome, "pnl_r": round(R, 3)})
        i = xi + 1                      # one position at a time
    return pd.DataFrame(trades)


def summ(df, label):
    cl = df[df["outcome"].isin(["win", "loss"])] if len(df) else df
    if not len(cl):
        print(f"  {label:26} no trades"); return
    n = len(cl); w = (cl["outcome"] == "win").sum(); R = cl["pnl_r"].sum()
    gw = cl.loc[cl["pnl_r"] > 0, "pnl_r"].sum(); gl = -cl.loc[cl["pnl_r"] < 0, "pnl_r"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    bal = 10000.0; eq = [bal]; ls = ccount = 0
    for _, t in cl.iterrows():
        ra = cfg.RISK_FLAT if bal < cfg.RISK_ANCHOR else bal * cfg.RISK_PCT / 100
        bal += ra * float(t["pnl_r"]); eq.append(bal)
        if t["pnl_r"] < 0: ccount += 1; ls = max(ls, ccount)
        else: ccount = 0
    e = np.array(eq); dd = ((e - np.maximum.accumulate(e)) / np.maximum.accumulate(e) * 100).min()
    yr = cl.groupby("year")["pnl_r"].sum()
    print(f"  {label:26} N={n:>5} WR={w/n*100:5.1f}% PF={pf:4.2f} totR={R:+8.1f} "
          f"exp={R/n:+.3f}R DD={dd:6.1f}% maxLS={ls:>2} stop={cl['stop_p'].median():5.1f}p "
          f"rr={cl['rr'].median():4.2f} green={(yr>0).sum()}/{len(yr)}")


def main():
    print("loading EURUSD M5 HistData ...")
    m5 = pd.read_pickle(HD5)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    print("=" * 118)
    print(f"BOLLINGER({BB_N},{BB_K}) + RSI({RSI_N}) {RSI_LO}/{RSI_HI} mean reversion · {SYMBOL} · "
          f"HistData 2000-2024 · stop {ATR_MULT}xATR({ATR_N}) · spread 1.0p")
    print("=" * 118)
    for tfname, (rule, hold) in TFS.items():
        df = m5 if rule == "5min" else m5.resample(rule).agg(agg).dropna()
        print(f"\n{tfname}  ({len(df):,} bars)")
        for mode, lbl in (("mid", "TP = middle band"), ("rr", f"TP = fixed {RR_FIXED}R")):
            summ(run(df, mode, hold), lbl)


if __name__ == "__main__":
    main()
