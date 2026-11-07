"""
Candle-by-candle trace of a single session for one symbol.

Prints every London M1 candle from the CISD confirmation onward, showing when
the limit at the entry level (FVG edge or CISD level) is TAPPED, and then which
of SL / TP is hit first — so you can eyeball exactly how the bot fills.

    python -m asia_bot.trace EURUSD
    python -m asia_bot.trace EURUSD 2026-07-08
"""
import os, sys
from datetime import timedelta
import pandas as pd
import MetaTrader5 as mt5

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asia_bot import config as cfg
from asia_bot import detector as det

SPREAD_PIPS = {"EURUSD": 1.0, "USDCHF": 1.5}


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    sym = "EURUSD"; want = None
    for a in sys.argv[1:]:
        (want := a) if "-" in a else (sym := a.upper())

    t = mt5.symbol_info_tick(sym)
    offset_h = det.broker_offset_hours(t.time) if t else 3
    mt5.symbol_select(sym, True)
    bars = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M1, 0, 40_000)
    mt5.shutdown()
    df = pd.DataFrame(bars); df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")[["open", "high", "low", "close"]]

    today = pd.Timestamp(want).date() if want else \
        (df.index[-1] - pd.Timedelta(hours=offset_h - 1)).date()

    setup = None
    for back in range(0, 15):
        day = today - timedelta(days=back)
        if day.weekday() >= 5:
            continue
        s = det.analyze_day(df, sym, offset_h, session_date=day)
        if s and s.stage == "confirmed":
            setup = s; break
    if setup is None:
        print("No confirmed setup in last 15 sessions."); return

    pip = cfg.PIP[sym]
    spread = SPREAD_PIPS.get(sym, 1.0) * pip
    prices = det.build_order_prices(setup, sym, spread)
    entry, sl, tp = prices["entry"], prices["sl"], prices["tp"]
    lvl = prices.get("entry_level", entry)

    dfc = df.iloc[:-1]
    t1 = det.utc1_index(dfc, offset_h)
    sb = det.session_bounds_utc1(setup.session_date)
    is_buy = setup.direction == "buy"

    print(f"\n{'='*84}")
    print(f"{sym} {setup.session_date} · {setup.direction.upper()} · "
          f"entry {lvl:.5f} ({setup.entry_source}) · SL {sl:.5f} · TP {tp:.5f} "
          f"· spread {spread/pip:.1f}p")
    print(f"Asia {setup.asia_low:.5f}-{setup.asia_high:.5f} · sweep {setup.sweep_time} "
          f"· CISD confirm {setup.confirm_time}")
    print(f"{'='*84}")
    print(f"{'time (UTC+1)':16} {'open':>8} {'high':>8} {'low':>8} {'close':>8}  event")
    print(f"{'-'*84}")

    filled = False; done = False
    for j in range(setup.confirm_i, len(dfc)):
        if t1[j] >= sb["london_end"] + pd.Timedelta(minutes=5):
            break
        o, h, l, c = dfc["open"].iloc[j], dfc["high"].iloc[j], dfc["low"].iloc[j], dfc["close"].iloc[j]
        ev = ""
        if j == setup.confirm_i:
            ev = "◄ CISD confirmed (limit placed at entry level)"
        elif not filled:
            tap = (l <= lvl) if is_buy else (h >= lvl)
            if tap:
                filled = True
                ev = f"✅ TAP → FILLED @ {lvl:.5f}"
        if filled and not done and j > setup.confirm_i:
            if is_buy:
                if l <= sl: ev += "  ✖ SL hit → LOSS -1R"; done = True
                elif h >= tp: ev += "  ★ TP hit → WIN +2R"; done = True
            else:
                if h + spread >= sl: ev += "  ✖ SL hit → LOSS -1R"; done = True
                elif l + spread <= tp: ev += "  ★ TP hit → WIN +2R"; done = True
        star = "  *fill*" if (ev.startswith("✅")) else ""
        print(f"{t1[j].strftime('%Y-%m-%d %H:%M'):16} {o:8.5f} {h:8.5f} {l:8.5f} {c:8.5f}  {ev}")
        if done:
            break
    if not filled:
        print("\n→ limit NEVER tapped before London close (no fill).")
    elif not done:
        print("\n→ filled but neither SL nor TP hit before London close (still open).")


if __name__ == "__main__":
    main()
