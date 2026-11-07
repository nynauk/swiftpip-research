"""Dry-run for the gradient bot: connect to MT5, iterate SYMBOLS, print
detection + sizing + EMA/exit state — NO ORDERS PLACED."""
import os, sys
from datetime import datetime, timezone
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live import config as cfg
from live import broker
from live.detector_ema_dol import detect_ema_dol_signals
from live.bot import NEWS_DAYS


def main():
    print(f"── SMOKE TEST · strategy={cfg.STRATEGY} · exit={cfg.EXIT_MODE} · "
          f"{len(cfg.SYMBOLS)} symbols · risk={cfg.RISK_PCT}% · "
          f"RSI {cfg.RSI_LONG_MIN}/{cfg.RSI_SHORT_MAX} · NO ORDERS ──\n")
    if not broker.connect():
        return
    bal = broker.account_balance(); eq = broker.account_equity()
    print(f"  Account balance ${bal:,.2f} · equity ${eq:,.2f}\n")

    for sym in cfg.SYMBOLS:
        if not broker.ensure_symbol(sym):
            print(f"── {sym} ── ensure_symbol FAILED"); continue
        bid, ask, spread = broker.get_tick(sym)
        df_15 = broker.get_bars(sym, "M15", cfg.M15_BARS_BACK)
        n = len(df_15)
        # live EMA state (for the manager / exit)
        emastate = "-"
        if n > cfg.EMA_SLOW * 4:
            c = df_15.iloc[:-1]["close"].values
            ef = pd.Series(c).ewm(span=cfg.EMA_FAST, adjust=False).mean().values
            es = pd.Series(c).ewm(span=cfg.EMA_SLOW, adjust=False).mean().values
            slope = ef[-1] - ef[-1 - cfg.GRAD_LOOKBACK]
            emastate = f"EMA{cfg.EMA_FAST} {'>' if ef[-1] > es[-1] else '<'} EMA{cfg.EMA_SLOW}, slope {slope:+.5f}"
        print(f"── {sym} ── bid {bid} ask {ask} spread {spread:.5f} · M15 {n} bars · {emastate}")

        if n < 130:
            print("  → not enough bars"); print(); continue
        closed = df_15.iloc[:-1]
        signals = detect_ema_dol_signals(
            closed, set(), bid=bid, ask=ask,
            fast=cfg.EMA_FAST, slow=cfg.EMA_SLOW, swing_n=cfg.SWING_LOOKBACK,
            fvg_window=cfg.FVG_LOOKUP_WINDOW, retest_window=cfg.RETEST_WINDOW_EMA,
            sl_buffer_frac=cfg.SL_BUFFER_FRAC, min_rr=cfg.MIN_RR, max_rr=cfg.MAX_RR,
            min_fvg_width=cfg.MIN_FVG_WIDTH.get(sym, 0.0002),
            rsi_period=cfg.RSI_PERIOD, rsi_long_min=cfg.RSI_LONG_MIN, rsi_short_max=cfg.RSI_SHORT_MAX,
        )
        if not signals:
            print("  → no fresh setup right now")
        for s in signals:
            if s.entry_mode == "market":
                rd = bal * (cfg.RISK_PCT / 100)
                lots = broker.calc_lots(sym, rd, s.risk_price)
                safety = (ask + cfg.SAFETY_TP_R * s.risk_price) if s.direction == "buy" \
                    else (bid - cfg.SAFETY_TP_R * s.risk_price)
                print(f"  ✅ {s.direction.upper()} MARKET · edge {s.edge_price} · sl {s.sl_price} "
                      f"· dol-rr {s.rr:.2f} · safetyTP {safety:.5f} · would-size {lots} lots (${rd:.2f})")
            else:
                print(f"  · {s.direction.upper()} {s.entry_mode}: {s.reason}")

        t = broker.get_tick(sym)
        import MetaTrader5 as mt5
        tk = mt5.symbol_info_tick(sym)
        if tk:
            d = datetime.fromtimestamp(tk.time, tz=timezone.utc).date()
            print(f"  news-blocked today ({d})? {d in NEWS_DAYS}")
        print()

    broker.disconnect()


if __name__ == "__main__":
    main()
