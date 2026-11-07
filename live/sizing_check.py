"""
Show what the SPREAD-ADJUSTED sizing produces for each symbol RIGHT NOW,
across a range of hypothetical SL distances. Compare to the old naive sizing.
NO ORDERS PLACED.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaTrader5 as mt5
from live import config as cfg
from live import broker


def main():
    print(f"── SIZING SANITY · risk={cfg.RISK_PCT}% · symbols={cfg.SYMBOLS} ──\n")
    if not broker.connect():
        return

    bal = broker.account_balance()
    risk_dollars = bal * cfg.RISK_PCT / 100
    print(f"  Balance ${bal:,.2f}  →  target risk ${risk_dollars:.2f}/trade\n")

    for sym in cfg.SYMBOLS:
        if not broker.ensure_symbol(sym):
            continue
        bid, ask, spread = broker.get_tick(sym)
        info = mt5.symbol_info(sym)
        tv = info.trade_tick_value; ts = info.trade_tick_size
        print(f"── {sym} · bid {bid} · ask {ask} · spread {spread:.5f} "
              f"· tick_val ${tv} · tick_size {ts} ──")
        print(f"  {'close-SL':>10} {'lots(naive)':>12} {'$risk(naive)':>13}  "
              f"{'lots(fixed)':>12} {'$risk(fixed)':>13} {'overshoot':>10}")

        # A range of hypothetical close-to-SL distances (in symbol's native price)
        # For FX: 5, 10, 20, 50 pips = 0.0005, 0.001, 0.002, 0.005
        # For US500: 0.5, 1, 2, 5 (points)
        # For UKOIL: 0.05, 0.10, 0.20, 0.50
        if "JPY" in sym:
            distances = [0.05, 0.10, 0.20, 0.50, 1.00]
        elif "cash" in sym and "500" in sym:
            distances = [0.5, 1.0, 2.0, 5.0, 10.0]
        elif "OIL" in sym or "oil" in sym.lower():
            distances = [0.05, 0.10, 0.20, 0.50, 1.00]
        else:
            distances = [0.0005, 0.001, 0.002, 0.005, 0.010]

        for close_sl in distances:
            # NAIVE: size on close_sl. What actually happens: fill at ASK for
            # a BUY, so real risk = close_sl + spread. Dollar loss = real_risk × lots × tick_val / ts.
            lots_naive = broker.calc_lots(sym, risk_dollars, close_sl)
            real_sl_for_naive = close_sl + spread
            risk_naive = lots_naive * real_sl_for_naive / ts * tv

            # FIXED: size on close_sl + spread from the start
            fixed_sl = close_sl + spread
            lots_fixed = broker.calc_lots(sym, risk_dollars, fixed_sl)
            risk_fixed = lots_fixed * fixed_sl / ts * tv

            overshoot = (risk_naive - risk_dollars) / risk_dollars * 100 if risk_naive > 0 else 0
            print(f"  {close_sl:>10.5f} {lots_naive:>12.2f} ${risk_naive:>11.2f}  "
                  f"{lots_fixed:>12.2f} ${risk_fixed:>11.2f} {overshoot:>+9.1f}%")
        print()

    broker.disconnect()


if __name__ == "__main__":
    main()
