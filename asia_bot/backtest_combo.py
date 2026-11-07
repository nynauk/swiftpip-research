"""
Pair the BB+RSI (trend-filtered) strategy with the Asia-CISD strategy.
EURUSD · M5 · HistData 2000-2024 · honest spread.

Legs (each already tested standalone):
  BB   = Bollinger(20,2)+RSI(14) dip-in-trend, EMA200 filter, exit at opposite band
  CISD = Asia sweep → London M5 CISD, market@close, SL swing, TP fixed 2.1R

Combinations reported:
  PORTFOLIO   = both legs run as one book; trades merged chronologically, each
                sized off the running balance (flat 90 <10k, else 0.9%).
  CONFLUENCE  = only CISD trades whose direction agrees with the EMA200 trend
                at the confirmation bar (the BB strat's regime filter applied to
                CISD).

    python -m asia_bot.backtest_combo
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asia_bot import config as cfg
from asia_bot import detector as det
from asia_bot.load_histdata import build_m5

PAIR = "EURUSD"
OFFSET_H = -5
SPREAD_PIPS = 1.0
# BB+RSI params
BB_N, BB_K, RSI_N, RSI_OS, RSI_OB = 20, 2.0, 14, 30, 70
TREND_EMA_N = 200
BUF_PIPS, MIN_SL_PIPS, MAX_HOLD = 1.0, 5.0, 288
# CISD
CISD_RR = 2.1
WALK_BARS, WALK_HOURS = 600, 60
OUT = os.path.join(os.path.dirname(__file__), "backtest")


def rsi_wilder(c, n):
    d = np.diff(c, prepend=c[0])
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().values
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().values
    return 100 - 100/(1 + np.where(rd == 0, np.inf, ru/rd))


def bb_leg(df, pip, spread):
    o=df["open"].values; h=df["high"].values; l=df["low"].values; c=df["close"].values
    idx = df.index
    s = pd.Series(c)
    mid = s.rolling(BB_N).mean().values
    sd = s.rolling(BB_N).std(ddof=0).values
    lower, upper = mid - BB_K*sd, mid + BB_K*sd
    rsi = rsi_wilder(c, RSI_N)
    ema = s.ewm(span=TREND_EMA_N, adjust=False).mean().values
    ls = (c < lower) & (rsi < RSI_OS); ss = (c > upper) & (rsi > RSI_OB)
    fl = ls & ~np.r_[False, ls[:-1]] & (c > ema)
    fs = ss & ~np.r_[False, ss[:-1]] & (c < ema)
    buf, min_sl = BUF_PIPS*pip, MIN_SL_PIPS*pip
    n = len(c); trades = []; i = BB_N + 1
    while i < n - 1:
        d = "long" if fl[i] else "short" if fs[i] else None
        if d is None: i += 1; continue
        if d == "long":
            fill = c[i]+spread; sl = min(l[i]-buf, fill-min_sl); risk = fill-sl
        else:
            fill = c[i]; sl = max(h[i]+buf, fill+min_sl); risk = sl-fill
        if risk <= 0: i += 1; continue
        R = 0.0; xi = None; end = min(i+MAX_HOLD, n)
        for j in range(i+1, end):
            if d == "long":
                if l[j] <= sl: R = -(1+spread/risk); xi=j; break
                if h[j] >= upper[j]: R = (upper[j]-fill)/risk; xi=j; break
            else:
                if h[j]+spread >= sl: R = -1.0; xi=j; break
                if l[j]+spread <= lower[j]: R = (fill-lower[j])/risk; xi=j; break
        if xi is None:
            xi = end-1
            R = (c[xi]-fill)/risk if d=="long" else (fill-c[xi])/risk
        trades.append({"t": idx[i], "dir": d, "R": round(R, 3), "leg": "BB"})
        i = xi + 1
    return pd.DataFrame(trades)


def cisd_leg(df, utc1_all, pip, spread):
    ema = pd.Series(df["close"].values).ewm(span=TREND_EMA_N, adjust=False).mean().values
    days = pd.DatetimeIndex(utc1_all).normalize().unique()
    trades = []; last_exit = None
    for day in days:
        d = day.date()
        if d.weekday() >= 5: continue
        sb = det.session_bounds_utc1(d)
        a = int(np.searchsorted(utc1_all, sb["asia_start"], "left"))
        b = int(np.searchsorted(utc1_all, sb["london_end"]+pd.Timedelta(hours=WALK_HOURS), "right"))
        if b-a < 60: continue
        sl_df = df.iloc[a:b]
        s = det.analyze_day(sl_df, PAIR, OFFSET_H, session_date=d)
        if s is None or s.stage != "confirmed": continue
        if last_exit is not None and s.confirm_time <= last_exit: continue
        cfg.RR = CISD_RR
        pr = det.build_market_prices(s, PAIR, spread)
        if pr is None: continue
        dfc = sl_df.iloc[:-1]; t1 = det.utc1_index(dfc, OFFSET_H)
        h=dfc["high"].values; l=dfc["low"].values
        E, slv, tp, r = pr["entry_level"], pr["sl"], pr["tp"], pr["risk"]
        k = s.confirm_i; R=0.0; xi=None
        for j in range(k+1, min(k+1+WALK_BARS, len(dfc))):
            if s.direction == "sell":
                if h[j]+spread >= slv: R=-1.0; xi=j; break
                if l[j]+spread <= tp: R=CISD_RR; xi=j; break
            else:
                if l[j] <= slv: R=-(1+spread/r); xi=j; break
                if h[j] >= tp: R=CISD_RR-spread/r; xi=j; break
        if xi is None: continue
        last_exit = t1[xi]
        # trend agreement at confirm bar (absolute index into df)
        abs_i = a + k
        trend_up = df["close"].values[abs_i] > ema[abs_i]
        agree = (s.direction == "buy" and trend_up) or (s.direction == "sell" and not trend_up)
        trades.append({"t": s.confirm_time, "dir": "long" if s.direction=="buy" else "short",
                       "R": round(R, 3), "leg": "CISD", "agree": agree})
    return pd.DataFrame(trades)


def stats(df, label):
    if not len(df): print(f"  {label:32} no trades"); return
    n=len(df); w=(df["R"]>0).sum(); R=df["R"].sum()
    gw=df.loc[df["R"]>0,"R"].sum(); gl=-df.loc[df["R"]<0,"R"].sum()
    pf = gw/gl if gl>0 else float("inf")
    d2 = df.sort_values("t")
    bal=10000.0; eq=[bal]; ls=cc=0; yrs={}
    for _,t in d2.iterrows():
        ra = cfg.RISK_FLAT if bal<cfg.RISK_ANCHOR else bal*cfg.RISK_PCT/100
        bal += ra*float(t["R"]); eq.append(bal)
        y = pd.Timestamp(t["t"]).year; yrs[y]=yrs.get(y,0)+float(t["R"])
        if t["R"]<0: cc+=1; ls=max(ls,cc)
        else: cc=0
    e=np.array(eq); dd=((e-np.maximum.accumulate(e))/np.maximum.accumulate(e)*100).min()
    green=sum(1 for v in yrs.values() if v>0)
    print(f"  {label:32} N={n:>5} WR={w/n*100:5.1f}% PF={pf:4.2f} totR={R:+8.1f} "
          f"exp={R/n:+.3f}R DD={dd:6.1f}% maxLS={ls:>2} green={green}/{len(yrs)}")


def main():
    os.makedirs(OUT, exist_ok=True)
    df = build_m5(PAIR)
    utc1_all = det.utc1_index(df, OFFSET_H)
    pip = cfg.PIP[PAIR]; spread = SPREAD_PIPS*pip
    print("=" * 104)
    print(f"BB+RSI (trend, band-exit)  ×  Asia-CISD (M5, {CISD_RR}R)  ·  {PAIR}  ·  2000-2024")
    print("=" * 104)

    bb = bb_leg(df, pip, spread)
    ci = cisd_leg(df, utc1_all, pip, spread)

    print("\n  — individual legs —")
    stats(bb, "BB+RSI trend (band exit)")
    stats(ci, "CISD (2.1R)")

    print("\n  — 1) PORTFOLIO: both legs as one book —")
    port = pd.concat([bb, ci[["t","dir","R","leg"]]], ignore_index=True)
    stats(port, "BB + CISD portfolio")

    print("\n  — 2) CONFLUENCE: CISD gated by EMA200 trend —")
    stats(ci[ci["agree"]], "CISD in-trend only")
    stats(ci[~ci["agree"]], "CISD counter-trend (for ref)")
    # portfolio using the trend-gated CISD
    port2 = pd.concat([bb, ci[ci["agree"]][["t","dir","R","leg"]]], ignore_index=True)
    stats(port2, "BB + in-trend-CISD portfolio")

    port.sort_values("t").to_csv(os.path.join(OUT, "combo_portfolio.csv"), index=False)
    print(f"\nsaved → {OUT}/combo_portfolio.csv")


if __name__ == "__main__":
    main()
