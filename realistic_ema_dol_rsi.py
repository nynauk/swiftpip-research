"""
EMA(9/21) + DOL + RSI confluence — realistic, EXPANDED universe (16 instruments).

Universe: 7 FX majors + 3 FX crosses + 2 metals + 3 indices + 1 energy.
Each instrument carries its own (spread, slippage, min-FVG-width) in SPEC.

EMA+DOL is trend-continuation, so we test RSI(14) bands at the entry bar:
  baseline       — no RSI
  momentum 50    — long RSI>=50, short RSI<=50   (trend agreement — the winner on FX)
  not-exhausted  — long RSI<=70, short RSI>=30
  strong 55/45   — long RSI>=55, short RSI<=45
  reversal 40/60 — long RSI<=40, short RSI>=60   (mean-reversion logic — control)

Reports combined + per-instrument, WITH longest loss/win streaks everywhere.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MetaTrader5 as mt5

import main as bt
import news_filter as nf
import ema_dol_strategy as ed


# symbol: (spread_price, slippage_price, min_fvg_width) — realistic retail/FTMO
SPEC = {
    # ── FX majors ──
    "EURUSD":     (1.0e-4, 0.3e-4, 2.0e-4),
    "GBPUSD":     (1.5e-4, 0.3e-4, 2.0e-4),
    "USDJPY":     (1.0e-2, 0.3e-2, 2.0e-2),
    "USDCHF":     (1.5e-4, 0.3e-4, 2.0e-4),
    "USDCAD":     (1.5e-4, 0.3e-4, 2.0e-4),
    "AUDUSD":     (1.5e-4, 0.3e-4, 2.0e-4),
    "NZDUSD":     (2.0e-4, 0.3e-4, 2.0e-4),
    # ── FX crosses ──
    "EURGBP":     (1.5e-4, 0.3e-4, 2.0e-4),
    "EURJPY":     (1.8e-2, 0.3e-2, 2.0e-2),
    "GBPJPY":     (2.5e-2, 0.4e-2, 3.0e-2),
    # ── metals ──
    "XAUUSD":     (0.25,   0.05,   0.30),
    "XAGUSD":     (0.025,  0.005,  0.015),
    # ── indices ──
    "US500.cash": (0.6,    0.1,    1.5),
    "US30.cash":  (3.0,    0.5,    8.0),
    "GER40.cash": (1.5,    0.3,    3.0),
    # ── energy ──
    "UKOIL.cash": (0.04,   0.01,   0.08),
}
GROUPS = [
    ("FX majors",  ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]),
    ("FX crosses", ["EURGBP", "EURJPY", "GBPJPY"]),
    ("Metals",     ["XAUUSD", "XAGUSD"]),
    ("Indices",    ["US500.cash", "US30.cash", "GER40.cash"]),
    ("Energy",     ["UKOIL.cash"]),
]
SYMBOLS = [s for _, syms in GROUPS for s in syms]
RSI_PERIOD = 14

# Fixed-dollar sizing: risk a flat $1,000 per trade on a $100k account
# (non-compounding — matches how an FTMO evaluation account behaves).
START_BAL = 100_000.0
RISK_USD  = 1_000.0
CHALLENGE_FEE = 550.0   # cost to (re)buy a $100k challenge — lost on each blowup

# ── Broker-realism stop buffer ──────────────────────────────────────────────
# A resting stop is filled on the WORSE side of the book (bid for a long, ask
# for a short) and can gap/slip a touch past the level. So the true risk distance
# is the chart distance PLUS the spread PLUS a 1-pip cushion. We widen every
# modeled SL by (spread + 1 pip): a stop-out then reflects that real fill, and
# marginal setups whose reward:risk no longer clears 2R correctly drop out.
# Defined here (the lowest-level module) so every backtest shares one toggle.
PIP = {
    "EURUSD": 1e-4, "GBPUSD": 1e-4, "USDCHF": 1e-4, "USDCAD": 1e-4,
    "AUDUSD": 1e-4, "NZDUSD": 1e-4, "EURGBP": 1e-4,
    "USDJPY": 1e-2, "EURJPY": 1e-2, "GBPJPY": 1e-2,
    "XAUUSD": 0.10, "XAGUSD": 0.01,
    "US500.cash": 1.0, "US30.cash": 1.0, "GER40.cash": 1.0,
    "US100.cash": 1.0, "UK100.cash": 1.0,
    "UKOIL.cash": 0.01, "USOIL.cash": 0.01,
}
SL_SPREAD_PAD = True    # master toggle for the spread+1pip stop buffer (before/after tests)


def sl_pad(symbol, spread):
    """Extra stop distance (price units) = broker spread + 1 pip. 0 if disabled.
    `spread` is passed in so this works with any SPEC table (edr's or fq's)."""
    if not SL_SPREAD_PAD:
        return 0.0
    return spread + PIP.get(symbol, spread)

# Momentum gradient (tightest -> loosest) to show the trade-count vs quality
# frontier. band = (long_lo, long_hi, short_lo, short_hi); RSI at entry must fit.
VARIANTS = [
    ("strong 55/45",      (55, 100, 0, 45)),
    ("momentum 50/50",    (50, 100, 0, 50)),
    ("momentum 45/55",    (45, 100, 0, 55)),
    ("momentum 40/60",    (40, 100, 0, 60)),
    ("baseline (no RSI)", (0, 100, 0, 100)),
]


def rsi(close, period=RSI_PERIOD):
    s = pd.Series(close)
    d = s.diff()
    ag = d.clip(lower=0).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100/(1+rs)).values


def sim_realistic(h, l, idx, ei, direction, entry, sl, tp, rr, spread, max_bars=400):
    for j in range(ei + 1, min(ei + max_bars, len(h))):
        if direction == "long":
            if l[j] <= sl: return "loss", idx[j], -1.0
            if h[j] >= tp: return "win",  idx[j],  rr
        else:
            if h[j] + spread >= sl: return "loss", idx[j], -1.0
            if l[j] + spread <= tp: return "win",  idx[j],  rr
    return "open", idx[min(ei + max_bars - 1, len(h) - 1)], 0.0


def precompute(symbol, df):
    h = df["high"].values; l = df["low"].values; c = df["close"].values
    idx = df.index
    ema_f = pd.Series(c, index=idx).ewm(span=ed.EMA_FAST, adjust=False).mean().values
    ema_s = pd.Series(c, index=idx).ewm(span=ed.EMA_SLOW, adjust=False).mean().values
    above = ema_f > ema_s
    cu = np.zeros(len(df), bool); cd = np.zeros(len(df), bool)
    cu[1:] = above[1:] & ~above[:-1]; cd[1:] = ~above[1:] & above[:-1]
    # index of the next bull / bear cross at or after each bar (n = "none")
    n = len(df)
    next_bull = np.full(n + 1, n, dtype=int)
    next_bear = np.full(n + 1, n, dtype=int)
    for k in range(n - 1, -1, -1):
        next_bull[k] = k if cu[k] else next_bull[k + 1]
        next_bear[k] = k if cd[k] else next_bear[k + 1]
    sh, sl_arr = ed.find_swings_arr(h, l, n=ed.SWING_LOOKBACK)
    pdh_map, pdl_map = ed.build_pdh_pdl(df)
    news_set = nf.build_news_block_set(idx[0].date(), idx[-1].date())
    return dict(h=h, l=l, c=c, idx=idx, cu=cu, cd=cd, sh=sh, sl_arr=sl_arr,
                next_bull=next_bull, next_bear=next_bear,
                pdh=pdh_map, pdl=pdl_map, news=news_set, rsi=rsi(c))


def run(symbol, pc, band, realistic=True):
    long_lo, long_hi, short_lo, short_hi = band
    spread_px, slip_px, min_w = SPEC[symbol]
    spread   = spread_px if realistic else 0.0
    slippage = slip_px   if realistic else 0.0
    h, l, c, idx = pc["h"], pc["l"], pc["c"], pc["idx"]
    cu, cd, sh, sl_arr = pc["cu"], pc["cd"], pc["sh"], pc["sl_arr"]
    next_bull, next_bear = pc["next_bull"], pc["next_bear"]
    pdh_map, pdl_map, news_set, rsi_arr = pc["pdh"], pc["pdl"], pc["news"], pc["rsi"]
    N = len(idx)

    trades = []; open_until = []; vetoed = 0
    for i in range(N):
        if not (cu[i] or cd[i]):
            continue
        bias = "long" if cu[i] else "short"
        target_dir = "bullish" if bias == "long" else "bearish"
        # MOST-RECENT-CROSS RULE: this cross dies the moment an OPPOSITE cross
        # fires. FVG must form (and entry must fill) strictly before that flip.
        opp = next_bear[i + 1] if bias == "long" else next_bull[i + 1]
        end_idx = min(i + ed.FVG_LOOKUP_WINDOW, opp - 1, N - 1)
        if end_idx < i + 2:
            continue                       # trend flipped before any FVG could form
        fvg = ed.find_aligned_fvg(h, l, i, end_idx, target_dir, min_w)
        if fvg is None:
            continue                       # no aligned FVG before the flip -> no trade
        retest_end = min(fvg["form_idx"] + ed.RETEST_WINDOW, opp - 1, N - 1)
        entry_i = None
        for k in range(fvg["form_idx"] + 1, retest_end + 1):
            if bias == "long" and l[k] <= fvg["top"]:    entry_i = k; break
            if bias == "short" and h[k] >= fvg["bottom"]: entry_i = k; break
        if entry_i is None:
            continue
        open_until = [t for t in open_until if t > idx[entry_i]]
        if len(open_until) >= 1:
            continue
        if idx[entry_i].date() in news_set:
            continue

        rv = rsi_arr[entry_i]
        if np.isnan(rv):
            continue
        if bias == "long" and not (long_lo <= rv <= long_hi):
            vetoed += 1; continue
        if bias == "short" and not (short_lo <= rv <= short_hi):
            vetoed += 1; continue

        d_today = pd.Timestamp(idx[entry_i]).normalize()
        pdh_t = pdh_map.get(d_today); pdl_t = pdl_map.get(d_today)
        buf = fvg["width"] * ed.SL_BUFFER_FRAC
        pad = sl_pad(symbol, spread) if realistic else 0.0   # spread+1pip stop buffer

        if bias == "long":
            real_entry = fvg["top"] + spread + slippage
            swing_low = ed.find_recent_swing_low(sl_arr, entry_i, lookback=80)
            anchor = swing_low if swing_low is not None and swing_low < fvg["bottom"] else fvg["bottom"]
            sl = anchor - buf - pad
            risk = real_entry - sl
            if risk <= 0: continue
            dol = ed.next_liquidity_above(real_entry, sh, entry_i, pdh_t)
            if dol is None: continue
            tp = dol; arr_ = (tp - real_entry) / risk; direction = "long"
        else:
            real_entry = fvg["bottom"] - slippage
            swing_high = ed.find_recent_swing_high(sh, entry_i, lookback=80)
            anchor = swing_high if swing_high is not None and swing_high > fvg["top"] else fvg["top"]
            sl = anchor + buf + pad
            risk = sl - real_entry
            if risk <= 0: continue
            dol = ed.next_liquidity_below(real_entry, sl_arr, entry_i, pdl_t)
            if dol is None: continue
            tp = dol; arr_ = (real_entry - tp) / risk; direction = "short"

        if arr_ < ed.MIN_RR_TO_TAKE or arr_ > ed.MAX_RR_CAP:
            continue

        oc, xt, pr = sim_realistic(h, l, idx, entry_i, direction, real_entry, sl, tp, arr_, spread)
        open_until.append(pd.Timestamp(xt))
        trades.append({"symbol": symbol, "entry_time": idx[entry_i], "direction": direction,
                       "rsi_entry": round(float(rv), 1), "achieved_rr": round(arr_, 2),
                       "outcome": oc, "exit_time": xt, "pnl_r": pr})
    out = pd.DataFrame(trades)
    out.attrs["vetoed"] = vetoed
    return out


def stats(df):
    """Fixed-dollar sizing: flat $1,000 risk/trade on a $100k account (no compounding)."""
    cl = df[df["outcome"] != "open"] if len(df) else df
    if not len(cl):
        return None
    cl = cl.sort_values("entry_time")
    n = len(cl); wins = (cl["pnl_r"] > 0.01).sum()
    wr = wins / n * 100; tot = cl["pnl_r"].sum()
    gw = cl.loc[cl["pnl_r"] > 0, "pnl_r"].sum(); gl = -cl.loc[cl["pnl_r"] < 0, "pnl_r"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    avg_win = cl.loc[cl["pnl_r"] > 0, "pnl_r"].mean() if wins else 0
    bal = START_BAL; eq = [bal]; cl_ = cw = bcl = bcw = 0
    for r in cl["pnl_r"]:
        bal += RISK_USD * float(r); eq.append(bal)          # fixed $ per R
        if r < -0.01: cl_ += 1; cw = 0; bcl = max(bcl, cl_)
        elif r > 0.01: cw += 1; cl_ = 0; bcw = max(bcw, cw)
    eqa = np.array(eq)
    dd = ((eqa - np.maximum.accumulate(eqa)) / np.maximum.accumulate(eqa) * 100).min()
    # timestamps aligned to equity points (prepend first trade time for the $100k start)
    times = [pd.Timestamp(cl["entry_time"].iloc[0])] + list(pd.to_datetime(cl["entry_time"]))
    return dict(n=n, wr=wr, pf=pf, tot=tot, dd=dd, ls=bcl, ws=bcw,
                bal=bal, avg_win=avg_win, eqa=eqa, times=times)


def prop_sim(df, start=100_000.0, target_pct=0.10, daily_pct=0.05,
             overall_pct=0.10, risk_usd=RISK_USD):
    """
    Simulate a rolling FTMO-style $100k account through the trade stream.
      • +10% ($110k)  -> cash out the profit, BANK it, reset to $100k (a 'payout')
      • -5% within 24h of the first loss  -> account BLOWN, reset, tally
      • -10% overall  (<= $90k)           -> account BLOWN, reset, tally
    Trades consumed in EXIT-time order (P&L realized on exit). Flat $ risk/trade.
    """
    cl = df[df["outcome"] != "open"].copy()
    if not len(cl):
        return None
    cl["exit_time"] = pd.to_datetime(cl["exit_time"])
    cl = cl.sort_values("exit_time")

    target = start * (1 + target_pct)
    daily_limit = start * daily_pct
    floor = start * (1 - overall_pct)

    bal = start; banked = 0.0
    passes = blown_daily = blown_overall = 0
    danchor = None; dpnl = 0.0
    trace_t = [cl["exit_time"].iloc[0]]; trace_b = [bal]
    passes_t = []; blows_t = []

    for t, pr in zip(cl["exit_time"], cl["pnl_r"]):
        pnl = float(pr) * risk_usd
        # roll the 24h daily window if it has expired
        if danchor is not None and (t - danchor).total_seconds() > 86400:
            danchor = None; dpnl = 0.0
        bal += pnl
        # a loss opens (or extends) the 24h window
        if danchor is None:
            if pnl < 0:
                danchor = t; dpnl = pnl
        else:
            dpnl += pnl

        if dpnl <= -daily_limit:                      # -5% daily breach
            blown_daily += 1; blows_t.append((t, "daily"))
            bal = start; danchor = None; dpnl = 0.0
        elif bal <= floor:                            # -10% overall breach
            blown_overall += 1; blows_t.append((t, "overall"))
            bal = start; danchor = None; dpnl = 0.0
        elif bal >= target:                           # +10% payout
            banked += (bal - start); passes += 1; passes_t.append((t, bal))
            bal = start; danchor = None; dpnl = 0.0
        trace_t.append(t); trace_b.append(bal)

    blown = blown_daily + blown_overall
    fees = blown * CHALLENGE_FEE
    return dict(banked=banked, passes=passes,
                blown_daily=blown_daily, blown_overall=blown_overall,
                blown=blown, final=bal, fees=fees,
                net=banked - fees,                       # true net business P&L
                total=banked + (bal - start),
                trace_t=trace_t, trace_b=trace_b,
                passes_t=passes_t, blows_t=blows_t)


class _Acct:
    __slots__ = ("bal", "danchor", "dpnl")
    def __init__(self):
        self.bal = 0.0; self.danchor = None; self.dpnl = 0.0


def fleet_sim(df, acct_size=10_000.0, fee=89.0, split=0.80,
              risk_pct=0.01, target_pct=0.10, daily_pct=0.05, overall_pct=0.10,
              cap_accounts=100):
    """
    Reinvestment loop. Start with ONE $10k account; each account risks 1%/trade.
      • +10% -> withdraw 80% profit ($800) to cash, reset & keep the account
      • -5% daily (24h) or -10% overall -> account dead (fee already sunk)
      • cash >= fee -> buy another $10k account (up to cap_accounts)
    All accounts take the SAME signals, so a -5% day can cull the whole fleet
    at once (correlated risk). Returns fleet + net-profit trajectory over time.
    """
    cl = df[df["outcome"] != "open"].copy()
    if not len(cl):
        return None
    cl["exit_time"] = pd.to_datetime(cl["exit_time"])
    cl = cl.sort_values("exit_time")

    risk_usd = acct_size * risk_pct          # $100/trade
    target_usd = acct_size * target_pct      # +$1,000
    daily_usd = acct_size * daily_pct        # -$500
    floor = -acct_size * overall_pct         # -$1,000 from base
    payout = split * target_usd              # $800 to trader

    fleet = [_Acct()]                        # start with the one account you have
    cash = 0.0
    withdrawn = 0.0
    fees = fee                               # you paid for the first one
    bought = 1; blown = 0; payouts = 0
    peak_fleet = 1
    times = []; fleet_n = []; net_t = []; cap_t = []

    for t, pr in zip(cl["exit_time"], cl["pnl_r"]):
        pnl = float(pr) * risk_usd
        survivors = []
        for a in fleet:
            if a.danchor is not None and (t - a.danchor).total_seconds() > 86400:
                a.danchor = None; a.dpnl = 0.0
            a.bal += pnl
            if a.danchor is None:
                if pnl < 0: a.danchor = t; a.dpnl = pnl
            else:
                a.dpnl += pnl
            if a.dpnl <= -daily_usd or a.bal <= floor:
                blown += 1; continue          # account dead
            if a.bal >= target_usd:
                cash += payout; withdrawn += payout; payouts += 1
                a.bal = 0.0; a.danchor = None; a.dpnl = 0.0
            survivors.append(a)
        fleet = survivors
        while cash >= fee and len(fleet) < cap_accounts:
            cash -= fee; fees += fee; fleet.append(_Acct()); bought += 1
        peak_fleet = max(peak_fleet, len(fleet))
        times.append(t); fleet_n.append(len(fleet))
        net_t.append(withdrawn - fees); cap_t.append(len(fleet) * acct_size)

    return dict(bought=bought, blown=blown, payouts=payouts,
                final_fleet=len(fleet), peak_fleet=peak_fleet,
                withdrawn=withdrawn, fees=fees, net=withdrawn - fees,
                times=times, fleet_n=fleet_n, net_t=net_t, cap_t=cap_t)


def rotation_sim(df, n_accounts=3, acct_size=10_000.0, fee=89.0, split=0.80,
                 risk_pct=0.01, target_pct=0.10, daily_pct=0.05, overall_pct=0.10,
                 rotate=True, replace_blown=False):
    """
    3 fixed $10k accounts, NO reinvestment (profits banked as cash).
      rotate=True  -> exactly ONE account is live each calendar week (1->2->3->1...)
                      so a bad week can only hit one account.
      rotate=False -> all 3 accounts trade every week (correlated, for comparison).
    Per account: +10% -> bank 80% ($800) & reset; -5% daily or -10% overall -> dead.
    """
    cl = df[df["outcome"] != "open"].copy()
    if not len(cl):
        return None
    cl["exit_time"] = pd.to_datetime(cl["exit_time"])
    cl = cl.sort_values("exit_time")
    risk = acct_size * risk_pct
    target = acct_size * target_pct
    daily = acct_size * daily_pct
    floor = -acct_size * overall_pct
    payout = split * target
    t0 = cl["exit_time"].iloc[0]

    accts = [_Acct() for _ in range(n_accounts)]
    alive = [True] * n_accounts
    banked = 0.0; payouts = 0; blown = 0; fees = n_accounts * fee
    pa_pay = [0] * n_accounts; pa_blown = [0] * n_accounts
    traces = [[(t0, acct_size)] for _ in range(n_accounts)]

    for t, pr in zip(cl["exit_time"], cl["pnl_r"]):
        week = (t - t0).days // 7
        pnl = float(pr) * risk
        targets = [week % n_accounts] if rotate else list(range(n_accounts))
        for i in targets:
            if not alive[i]:
                continue
            a = accts[i]
            if a.danchor is not None and (t - a.danchor).total_seconds() > 86400:
                a.danchor = None; a.dpnl = 0.0
            a.bal += pnl
            if a.danchor is None:
                if pnl < 0: a.danchor = t; a.dpnl = pnl
            else:
                a.dpnl += pnl
            if a.dpnl <= -daily or a.bal <= floor:
                blown += 1; pa_blown[i] += 1
                traces[i].append((t, acct_size + a.bal))
                if replace_blown:
                    accts[i] = _Acct(); fees += fee
                    traces[i].append((t, acct_size))
                else:
                    alive[i] = False
                continue
            if a.bal >= target:
                banked += payout; payouts += 1; pa_pay[i] += 1
                traces[i].append((t, acct_size + a.bal))     # show the +10% peak
                a.bal = 0.0; a.danchor = None; a.dpnl = 0.0
                traces[i].append((t, acct_size))             # reset to base
            else:
                traces[i].append((t, acct_size + a.bal))

    return dict(banked=banked, payouts=payouts, blown=blown, fees=fees,
                net=banked - fees, alive=sum(alive),
                pa_pay=pa_pay, pa_blown=pa_blown, traces=traces)


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    OUT = os.path.join("strategies", "realistic_ema_dol_rsi")
    os.makedirs(OUT, exist_ok=True)

    pcs = {}
    for s in SYMBOLS:
        df = bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
        pcs[s] = precompute(s, df)

    print("=" * 116)
    print(f"EMA(9/21)+DOL + RSI · REALISTIC · {len(SYMBOLS)} instruments "
          f"(7 FX majors + 3 crosses + 2 metals + 3 indices + 1 energy) · ~4yr M15")
    print("=" * 116)
    print(f"  (sizing: flat ${RISK_USD:,.0f} risk/trade on ${START_BAL:,.0f} account, no compounding)")
    print(f"  {'Variant':20} {'':9} {'N':>4} {'WR':>6} {'PF':>5} {'R':>8} {'DD':>7} "
          f"{'LossStrk':>8} {'WinStrk':>7} {'avgWin':>6}  {'End$100k':>11}")

    keep = {}
    for vlabel, band in VARIANTS:
        for realistic in (False, True):
            allc = []
            for s in SYMBOLS:
                allc.append(run(s, pcs[s], band, realistic=realistic))
            combined = pd.concat(allc, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
            st = stats(combined)
            keep[(vlabel, realistic)] = (combined, st)
            tag = "REALISTIC" if realistic else "naive"
            if st is None:
                print(f"  {vlabel:20} {tag:9} no trades"); continue
            print(f"  {vlabel:20} {tag:9} {st['n']:>4} {st['wr']:>5.1f}% {st['pf']:>4.2f} "
                  f"{st['tot']:>+7.1f}R {st['dd']:>6.1f}% {st['ls']:>8} {st['ws']:>7} "
                  f"{st['avg_win']:>5.2f}R  ${st['bal']:>9,.0f}")
        print()

    # Per-instrument for baseline + momentum (realistic), WITH loss/win streaks
    for vlabel in ["baseline (no RSI)", "momentum 50/50"]:
        combined, _ = keep[(vlabel, True)]
        print("=" * 116)
        print(f"PER-INSTRUMENT · {vlabel} · REALISTIC")
        print(f"  {'Group':11} {'Symbol':11} {'N':>4} {'WR':>6} {'PF':>5} {'R':>8} "
              f"{'DD':>7} {'LossStrk':>8} {'WinStrk':>7}")
        for gname, syms in GROUPS:
            for s in syms:
                g = combined[combined["symbol"] == s]
                st = stats(g)
                if st is None:
                    print(f"  {gname:11} {s:11}  no trades"); continue
                print(f"  {gname:11} {s:11} {st['n']:>4} {st['wr']:>5.1f}% {st['pf']:>4.2f} "
                      f"{st['tot']:>+7.1f}R {st['dd']:>6.1f}% {st['ls']:>8} {st['ws']:>7}")
        print()

    # Group-level summary for momentum
    combined, _ = keep[("momentum 50/50", True)]
    print("=" * 116)
    print("BY GROUP · momentum 50/50 · REALISTIC")
    print(f"  {'Group':11} {'N':>4} {'WR':>6} {'PF':>5} {'R':>8} {'DD':>7} {'LossStrk':>8} {'WinStrk':>7}")
    for gname, syms in GROUPS:
        g = combined[combined["symbol"].isin(syms)]
        st = stats(g)
        if st is None:
            print(f"  {gname:11}  no trades"); continue
        print(f"  {gname:11} {st['n']:>4} {st['wr']:>5.1f}% {st['pf']:>4.2f} "
              f"{st['tot']:>+7.1f}R {st['dd']:>6.1f}% {st['ls']:>8} {st['ws']:>7}")

    # Equity over TIME (x-axis = date), fixed-$ sizing. All variants span the
    # same ~4yr window, so the momentum lines no longer "stop early".
    import matplotlib.dates as mdates
    BG, FG = "#0d1117", "#e6edf3"
    palette = {"strong 55/45": "#d29922", "momentum 50/50": "#3fb950",
               "momentum 45/55": "#2ea043", "momentum 40/60": "#7ee787",
               "baseline (no RSI)": "#7d8590"}
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=BG); ax.set_facecolor(BG)
    for vlabel, band in VARIANTS:
        st = keep[(vlabel, True)][1]
        if st is None: continue
        lw = 2.1 if vlabel.startswith("baseline") else 1.5
        ax.plot(st["times"], st["eqa"], color=palette[vlabel], linewidth=lw,
                drawstyle="steps-post",
                label=f"{vlabel}: N={st['n']}, PF {st['pf']:.2f}, LossStrk {st['ls']}, "
                      f"end ${st['bal']:,.0f}")
    ax.axhline(START_BAL, color="#444c56", linestyle="--", linewidth=0.8)
    ax.axhline(START_BAL * 0.90, color="#f85149", linestyle=":", linewidth=1.1,
               label="FTMO -10% floor ($90,000)")
    ax.set_title(f"EMA(9/21)+DOL + RSI · {len(SYMBOLS)} instruments · REALISTIC · "
                 f"flat {RISK_USD:,.0f} USD/trade on {START_BAL/1000:.0f}k account",
                 color=FG, fontsize=12, fontweight="bold")
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
    ax.set_xlabel("Date", color=FG); ax.set_ylabel("Balance ($)", color=FG)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "equity_rsi_variants.png"),
                dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    keep[("momentum 50/50", True)][0].to_csv(
        os.path.join(OUT, "trades_momentum_realistic.csv"), index=False)

    # ── TRADE FREQUENCY (over the dataset span) ──
    data_start = min(pc["idx"][0] for pc in pcs.values())
    data_end = max(pc["idx"][-1] for pc in pcs.values())
    span_months = (data_end - data_start).days / 30.44
    print("\n" + "=" * 116)
    print(f"TRADE FREQUENCY · dataset {data_start.date()} -> {data_end.date()} "
          f"({span_months:.1f} months across {len(SYMBOLS)} instruments)")
    print(f"  {'Variant':20} {'Trades':>7} {'Trades/mo':>10} {'Trades/wk':>10} "
          f"{'span(mo)':>9} {'first trade':>12} {'last trade':>12}")
    freq = {}
    for vlabel, band in VARIANTS:
        cl = keep[(vlabel, True)][0]
        cl = cl[cl["outcome"] != "open"]
        if not len(cl):
            continue
        f0 = pd.to_datetime(cl["entry_time"]).min()
        f1 = pd.to_datetime(cl["entry_time"]).max()
        vmonths = max((f1 - f0).days / 30.44, 0.1)
        tpm = len(cl) / vmonths
        freq[vlabel] = tpm
        print(f"  {vlabel:20} {len(cl):>7} {tpm:>10.1f} {tpm*12/52:>10.1f} "
              f"{vmonths:>9.1f} {str(f0.date()):>12} {str(f1.date()):>12}")

    # ── PROP-FIRM SIMULATION (net of challenge fees) ──
    print("\n" + "=" * 116)
    print(f"PROP-FIRM SIM · ${START_BAL:,.0f} acct · +10% target (cash out+reset) · "
          f"-5% daily (24h) · -10% overall · flat ${RISK_USD:,.0f}/trade · "
          f"blowup = -${CHALLENGE_FEE:,.0f} fee")
    print("=" * 116)
    print(f"  {'Variant':20} {'Tr/mo':>6} {'Payouts':>8} {'Banked $':>11} "
          f"{'Blown':>6} {'Fees $':>8} {'NET $':>12}")
    props = {}
    for vlabel, band in VARIANTS:
        df = keep[(vlabel, True)][0]
        ps = prop_sim(df)
        props[vlabel] = ps
        if ps is None:
            print(f"  {vlabel:20} no trades"); continue
        print(f"  {vlabel:20} {freq.get(vlabel,0):>6.1f} {ps['passes']:>8} "
              f"${ps['banked']:>10,.0f} {ps['blown']:>6} ${ps['fees']:>7,.0f} "
              f"${ps['net']:>11,.0f}")

    best = max(props, key=lambda v: props[v]["net"] if props[v] else -1e18)
    worst = max(props, key=lambda v: props[v]["blown"] if props[v] else -1)
    safest = min(props, key=lambda v: props[v]["blown"] if props[v] else 1e9)
    print(f"\n  >> MOST NET $  : '{best}'  ->  ${props[best]['net']:,.0f} net "
          f"(${props[best]['banked']:,.0f} banked - ${props[best]['fees']:,.0f} fees, "
          f"{props[best]['passes']} payouts, {props[best]['blown']} blown)")
    print(f"  >> MOST BLOWN  : '{worst}'  ->  {props[worst]['blown']} accounts blown "
          f"({props[worst]['blown_daily']} daily, {props[worst]['blown_overall']} overall), "
          f"-${props[worst]['fees']:,.0f} in fees")
    print(f"  >> SAFEST      : '{safest}'  ->  {props[safest]['blown']} blown, "
          f"${props[safest]['net']:,.0f} net")

    # Chart A: account sawtooth over time for the best variant
    bt_ps = props[best]
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(bt_ps["trace_t"], bt_ps["trace_b"], color="#3fb950", linewidth=0.9)
    ax.axhline(START_BAL, color="#7d8590", linestyle="--", linewidth=0.7)
    ax.axhline(START_BAL * 1.10, color="#3fb950", linestyle=":", linewidth=1.0)
    ax.axhline(START_BAL * 0.90, color="#f85149", linestyle=":", linewidth=1.0)
    if bt_ps["passes_t"]:
        ax.scatter([t for t, _ in bt_ps["passes_t"]],
                   [START_BAL * 1.10] * len(bt_ps["passes_t"]),
                   marker="^", s=36, color="#3fb950", zorder=5,
                   label=f"payout ({len(bt_ps['passes_t'])})")
    if bt_ps["blows_t"]:
        ax.scatter([t for t, _ in bt_ps["blows_t"]],
                   [START_BAL * 0.90] * len(bt_ps["blows_t"]),
                   marker="x", s=44, color="#f85149", zorder=5,
                   label=f"blown ({len(bt_ps['blows_t'])})")
    ax.set_title(f"Prop-firm account cycle · '{best}' · "
                 f"banked ${bt_ps['banked']:,.0f} · {bt_ps['passes']} payouts · "
                 f"{bt_ps['blown']} blown", color=FG, fontsize=12, fontweight="bold")
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
    ax.set_xlabel("Date", color=FG); ax.set_ylabel("Account balance ($)", color=FG)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "prop_sawtooth_best.png"),
                dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    # Chart B: banked $ vs accounts blown, per variant
    labels = [v for v, _ in VARIANTS]
    banked_vals = [props[v]["banked"] for v in labels]
    blown_vals = [props[v]["blown"] for v in labels]
    fig, ax1 = plt.subplots(figsize=(12, 6), facecolor=BG); ax1.set_facecolor(BG)
    x = np.arange(len(labels))
    ax1.bar(x - 0.2, banked_vals, 0.4, color="#3fb950", label="Banked $")
    ax1.set_ylabel("Banked $ (cashed out)", color="#3fb950")
    ax1.tick_params(axis="y", colors="#3fb950")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, blown_vals, 0.4, color="#f85149", label="Accounts blown")
    ax2.set_ylabel("Accounts blown", color="#f85149")
    ax2.tick_params(axis="y", colors="#f85149")
    for xi, (b, bl, p) in enumerate(zip(banked_vals, blown_vals, [props[v]["passes"] for v in labels])):
        ax1.text(xi - 0.2, b, f"{p}p", ha="center", va="bottom", color=FG, fontsize=8)
        ax2.text(xi + 0.2, bl, str(bl), ha="center", va="bottom", color=FG, fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, color=FG, fontsize=9, rotation=15)
    ax1.set_title(f"Prop-firm outcome by RSI variant · {START_BAL/1000:.0f}k accounts · "
                  f"flat {RISK_USD:,.0f} USD/trade  (label = payouts)",
                  color=FG, fontsize=12, fontweight="bold")
    ax1.tick_params(axis="x", colors=FG); ax1.grid(True, alpha=0.1, color=FG, axis="y")
    for sp in ax1.spines.values(): sp.set_edgecolor("#30363d")
    for sp in ax2.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "prop_compare.png"),
                dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    # ── REINVESTMENT LOOP (fleet of $10k accounts) ──
    print("\n" + "=" * 116)
    print("REINVESTMENT LOOP · start with ONE $10k account · 1% risk/trade · "
          "+10% -> withdraw 80% ($800), buy more $10k accounts ($89 each, cap 100)")
    print("=" * 116)
    print(f"  {'Variant':16} {'Payouts':>8} {'Bought':>7} {'Blown':>6} {'PeakFleet':>10} "
          f"{'EndFleet':>9} {'Withdrawn$':>12} {'Fees$':>9} {'NET $':>12}")
    fleet_variants = ["momentum 50/50", "momentum 45/55", "momentum 40/60"]
    fleets = {}
    for vlabel in fleet_variants:
        fs = fleet_sim(keep[(vlabel, True)][0])
        fleets[vlabel] = fs
        print(f"  {vlabel:16} {fs['payouts']:>8} {fs['bought']:>7} {fs['blown']:>6} "
              f"{fs['peak_fleet']:>10} {fs['final_fleet']:>9} ${fs['withdrawn']:>11,.0f} "
              f"${fs['fees']:>8,.0f} ${fs['net']:>11,.0f}")

    fbest = max(fleets, key=lambda v: fleets[v]["net"])
    print(f"\n  >> BEST BUSINESS: '{fbest}' -> ${fleets[fbest]['net']:,.0f} net profit, "
          f"{fleets[fbest]['final_fleet']} accounts running at end "
          f"(${fleets[fbest]['final_fleet']*10_000:,.0f} capital), {fleets[fbest]['blown']} blown")

    # Chart: fleet size + net profit over time
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(13, 8), facecolor=BG, sharex=True)
    cols = {"momentum 50/50": "#3fb950", "momentum 45/55": "#58a6ff", "momentum 40/60": "#d29922"}
    for vlabel in fleet_variants:
        fs = fleets[vlabel]
        axA.plot(fs["times"], fs["fleet_n"], color=cols[vlabel], linewidth=1.4,
                 label=f"{vlabel}: peak {fs['peak_fleet']}, end {fs['final_fleet']} accts, "
                       f"{fs['blown']} blown")
        axB.plot(fs["times"], fs["net_t"], color=cols[vlabel], linewidth=1.4,
                 label=f"{vlabel}: ${fs['net']:,.0f} net")
    for ax in (axA, axB):
        ax.set_facecolor(BG); ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
        ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=8, loc="upper left")
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axA.set_title("Reinvestment loop · fleet of $10k accounts (cap 100)",
                  color=FG, fontsize=12, fontweight="bold")
    axA.set_ylabel("Accounts running", color=FG)
    axB.set_ylabel("Net profit ($)", color=FG)
    axB.set_xlabel("Date", color=FG)
    axB.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "reinvestment_fleet.png"),
                dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    # ── 3-ACCOUNT WEEKLY ROTATION (no reinvestment) ──
    print("\n" + "=" * 116)
    print("3x $10k ACCOUNTS · WEEKLY ROTATION (one live account per week) · NO reinvestment · "
          "1% risk · +10% banks $800")
    print("=" * 116)
    print(f"  {'Variant':16} {'Mode':14} {'Payouts':>8} {'Blown':>6} {'Alive':>6} "
          f"{'Banked $':>11} {'Net $':>10}  per-account payouts")
    rot_variants = ["momentum 50/50", "momentum 45/55", "momentum 40/60"]
    rots = {}
    for vlabel in rot_variants:
        for rotate, mlabel in [(True, "rotate 1/wk"), (False, "all-3 (compare)")]:
            rs = rotation_sim(keep[(vlabel, True)][0], rotate=rotate)
            rots[(vlabel, rotate)] = rs
            print(f"  {vlabel:16} {mlabel:14} {rs['payouts']:>8} {rs['blown']:>6} "
                  f"{rs['alive']:>6} ${rs['banked']:>10,.0f} ${rs['net']:>9,.0f}  "
                  f"{rs['pa_pay']}  blown{rs['pa_blown']}")
        print()

    print("  Read: 'rotate 1/wk' spreads one week's trades onto a single account, so a bad")
    print("  week can only hit one of the three. 'all-3' trades every account every week")
    print("  (3x the payouts but correlated — a -5% day can blow all three at once).")

    # Chart: the 3 account balances over time, weekly rotation, recommended variant
    rec = "momentum 45/55"
    rs = rots[(rec, True)]
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG); ax.set_facecolor(BG)
    acct_cols = ["#3fb950", "#58a6ff", "#d29922"]
    for i, tr in enumerate(rs["traces"]):
        ts = [t for t, _ in tr]; bs = [b for _, b in tr]
        ax.plot(ts, bs, color=acct_cols[i], linewidth=1.0, drawstyle="steps-post",
                label=f"Account {i+1}: {rs['pa_pay'][i]} payouts, {rs['pa_blown'][i]} blown")
    ax.axhline(10_000, color="#7d8590", linestyle="--", linewidth=0.7)
    ax.axhline(11_000, color="#3fb950", linestyle=":", linewidth=0.9)
    ax.axhline(9_000, color="#f85149", linestyle=":", linewidth=0.9, label="blow floor ($9,000)")
    ax.set_title(f"3x 10k accounts · weekly rotation · {rec} · banked {rs['banked']:,.0f} USD · "
                 f"{rs['blown']} blown / {rs['alive']} alive", color=FG, fontsize=12, fontweight="bold")
    ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=9, loc="upper left")
    ax.set_xlabel("Date", color=FG); ax.set_ylabel("Account balance ($)", color=FG)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "rotation_3accounts.png"),
                dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    print("\nSaved -> " + OUT + "/")
    mt5.shutdown()


if __name__ == "__main__":
    main()
