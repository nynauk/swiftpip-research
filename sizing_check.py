"""
Position-sizing feasibility check for the live account.
For each of the 17 instruments: pull the broker's REAL contract specs
(volume_min/step, tick_size, tick_value), derive the strategy's median stop
distance from an actual backtest pass, then compute — using the SAME calc_lots
math the bot uses — what fraction of that instrument's trades can be sized at the
live risk budget (1% of the current balance). Instruments whose min lot already
overrisks the budget get 'lots = 0' and are silently skipped by the bot.
"""
import math
import numpy as np
import MetaTrader5 as mt5

import frequency_tune as fq
import realistic_ema_dol_rsi as edr
import spread_analysis as sa
from live import config as cfg

EMA = (7, 18)


def lots_for(info, risk_usd, sl_dist):
    """Replica of broker.calc_lots with a cached symbol_info."""
    if sl_dist <= 0 or risk_usd <= 0:
        return 0.0
    dllr = (sl_dist / info.trade_tick_size) * info.trade_tick_value
    if dllr <= 0:
        return 0.0
    step = info.volume_step or 0.01
    lots = math.floor((risk_usd / dllr) / step) * step
    if lots < info.volume_min:
        return 0.0
    return min(lots, info.volume_max)


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    edr.SL_SPREAD_PAD = True
    acct = mt5.account_info()
    bal = acct.balance
    budget = bal * cfg.RISK_PCT / 100.0
    print("=" * 104)
    print(f"SIZING CHECK · account #{acct.login} · balance ${bal:,.2f} · "
          f"risk {cfg.RISK_PCT}% = ${budget:.2f}/trade")
    print("=" * 104)
    print(f"  {'symbol':12} {'Ntr':>4} {'volMin':>7} {'medStop':>10} {'minLot$risk':>12} "
          f"{'minLot%acct':>11} {'tradeable':>10}  verdict")
    print("-" * 104)

    tradeable, blocked = [], []
    for s in cfg.SYMBOLS:
        info = mt5.symbol_info(s)
        if info is None:
            print(f"  {s:12}  — no symbol_info (not offered by broker?)")
            blocked.append(s); continue
        mt5.symbol_select(s, True)
        info = mt5.symbol_info(s)
        try:
            df = sa.load_m15(s)
            base = fq.precompute_base(s, df)
            trs = sa.collect(s, df, base, EMA, fq.BASEP, 1, True)
        except Exception as e:
            print(f"  {s:12}  — backtest failed: {e!r}")
            continue
        if not trs:
            print(f"  {s:12}  — no trades to sample"); continue
        risks = np.array([t["risk"] for t in trs], float)
        med = float(np.median(risks))
        frac_ok = float(np.mean([lots_for(info, budget, r) > 0 for r in risks])) * 100
        dllr_med = (med / info.trade_tick_size) * info.trade_tick_value
        minlot_risk = info.volume_min * dllr_med          # $ risk if forced to min lot at median stop
        minlot_pct = minlot_risk / bal * 100
        ok = frac_ok >= 50                                 # can take at least half its trades
        (tradeable if ok else blocked).append(s)
        verdict = "OK" if frac_ok >= 95 else ("PARTIAL" if ok else "BLOCKED")
        print(f"  {s:12} {len(trs):>4} {info.volume_min:>7.2f} {med:>10.5f} "
              f"${minlot_risk:>10.2f} {minlot_pct:>10.1f}% {frac_ok:>9.0f}%  {verdict}")

    print("-" * 104)
    print(f"  Tradeable at ${budget:.0f}/trade (≥50% of trades sizable): {len(tradeable)}/"
          f"{len(cfg.SYMBOLS)}  →  {tradeable}")
    print(f"  BLOCKED (min lot overrisks the budget): {len(blocked)}  →  {blocked}")

    # what risk budget / account each blocked instrument needs
    print("\n  To trade the blocked instruments you'd need (per trade, at median stop):")
    for s in blocked:
        info = mt5.symbol_info(s)
        if info is None:
            print(f"    {s:12} not offered by broker"); continue
        df = sa.load_m15(s); base = fq.precompute_base(s, df)
        trs = sa.collect(s, df, base, EMA, fq.BASEP, 1, True)
        if not trs:
            continue
        med = float(np.median([t["risk"] for t in trs]))
        need = info.volume_min * (med / info.trade_tick_size) * info.trade_tick_value
        print(f"    {s:12} ≥ ${need:,.2f}/trade  → ${need/(cfg.RISK_PCT/100):,.0f} account at "
              f"{cfg.RISK_PCT:.0f}%  (or {need/bal*100:.1f}% risk on ${bal:,.0f})")
    mt5.shutdown()


if __name__ == "__main__":
    main()
