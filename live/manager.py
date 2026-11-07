"""Position management for the gradient-exit strategy.

The broker holds the SL and a far safety TP. This manager runs the live GRADIENT
EXIT for every timeframe engine (M3/M5/M15, each with its own magic): on each
poll it recomputes the fast EMA on CLOSED bars of the position's OWN timeframe
and closes at market when the fast-EMA slope has turned against the position
(once in profit) — or when the fast/slow EMAs have crossed back (fallback).
Matches the backtest exit.

All exit inputs — the gradient, the EMA cross, AND the in-profit test — are read
from the last CLOSED candle, never the live tick, so a forming candle's
fluctuation can't prematurely scratch a trade (the backtest judges the gradient
exit on the bar close too).
"""
import pandas as pd
import MetaTrader5 as mt5

from . import broker


def manage(symbol: str, state: dict, cfg, log):
    """Manage exits for EVERY timeframe engine on this symbol. Each TF has its
    own magic; its positions are exited against ITS OWN timeframe's EMAs."""
    tf_magic = getattr(cfg, "TF_MAGIC", {"M15": cfg.MAGIC})
    all_live = set()
    for tf, magic in tf_magic.items():
        all_live |= {p.ticket for p in broker.open_positions(symbol, magic)}

    # prune state entries whose positions are gone (SL / safety-TP fired on broker)
    for tk in [t for t, m in list(state["positions"].items())
               if m.get("symbol") == symbol and int(t) not in all_live]:
        state["positions"].pop(tk, None)
        log(f"[{symbol}] position #{tk} closed on broker (SL/TP) — pruned")

    if getattr(cfg, "EXIT_MODE", "gradient") != "gradient":
        return
    for tf, magic in tf_magic.items():
        _manage_tf(symbol, tf, magic, state, cfg, log)


def _manage_tf(symbol: str, tf: str, magic: int, state: dict, cfg, log):
    positions = broker.open_positions(symbol, magic)
    if not positions:
        return

    df = broker.get_bars(symbol, tf, getattr(cfg, "BARS_BACK", cfg.M15_BARS_BACK))
    if len(df) < cfg.EMA_SLOW * 4 + 2:
        return
    # Everything the exit reads comes from the LAST CLOSED candle (the "previous"
    # candle). We drop the currently-forming bar so intra-candle tick fluctuation
    # can't flip the gradient or scratch the trade. c[-1] is that settled close —
    # matching the backtest, which evaluates the gradient exit on the bar CLOSE.
    c = df.iloc[:-1]["close"].values                       # completed bars only
    ema_f = pd.Series(c).ewm(span=cfg.EMA_FAST, adjust=False).mean().values
    ema_s = pd.Series(c).ewm(span=cfg.EMA_SLOW, adjust=False).mean().values
    lb = cfg.GRAD_LOOKBACK
    if len(ema_f) < lb + 2:
        return
    grad = ema_f[-1] - ema_f[-1 - lb]                       # fast-EMA slope over lb closed bars
    cross_up   = ema_f[-1] > ema_s[-1] and ema_f[-2] <= ema_s[-2]
    cross_down = ema_f[-1] < ema_s[-1] and ema_f[-2] >= ema_s[-2]
    prev_close = float(c[-1])                               # settled close of the previous candle

    bid, ask, spread = broker.get_tick(symbol)
    if bid is None or ask is None:
        return

    for p in positions:
        is_long = p.type == mt5.POSITION_TYPE_BUY
        if is_long:
            inprofit = prev_close > p.price_open            # judged on the closed candle, not a live tick
            grad_against = grad < 0
            cross_against = cross_down
            px = bid                                        # actual exit still fills at market
        else:
            inprofit = prev_close + (spread or 0.0) < p.price_open   # short pays spread to close (backtest parity)
            grad_against = grad > 0
            cross_against = cross_up
            px = ask

        reason = None
        if grad_against and (inprofit or not cfg.GRADIENT_PROFIT_ONLY):
            reason = "gradient flip" + (" (in profit)" if inprofit else "")
        elif cross_against:
            reason = "opposite EMA cross"

        if reason and broker.close_position(p.ticket, symbol):
            log(f"[{symbol} {tf}] GRAD-EXIT close #{p.ticket} "
                f"({'LONG' if is_long else 'SHORT'}) @ {px} — {reason}")
            state["positions"].pop(str(p.ticket), None)
