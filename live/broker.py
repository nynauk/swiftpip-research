"""MT5 wrappers — connect, fetch bars, get tick, size positions, place orders."""
import math
import time
from typing import Optional
import pandas as pd
import MetaTrader5 as mt5


TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M3":  mt5.TIMEFRAME_M3,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}


def connect() -> bool:
    if not mt5.initialize():
        print(f"[broker] MT5 init FAILED: {mt5.last_error()}")
        return False
    print(f"[broker] MT5 connected · account #{mt5.account_info().login} "
          f"· balance ${mt5.account_info().balance:,.2f}")
    return True


def disconnect():
    mt5.shutdown()


def ensure_symbol(symbol: str) -> bool:
    """Subscribe the symbol in Market Watch."""
    if not mt5.symbol_select(symbol, True):
        print(f"[broker] symbol_select({symbol}) FAILED: {mt5.last_error()}")
        return False
    return True


def get_bars(symbol: str, tf: str, count: int) -> pd.DataFrame:
    """Returns the last `count` bars — INCLUDES the currently-forming bar as the last row."""
    tf_const = TF_MAP[tf]
    bars = mt5.copy_rates_from_pos(symbol, tf_const, 0, count)
    if bars is None or len(bars) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")[["open", "high", "low", "close"]]
    return df


def get_tick(symbol: str):
    """Returns (bid, ask, spread_price)."""
    t = mt5.symbol_info_tick(symbol)
    if t is None:
        return None, None, None
    return t.bid, t.ask, (t.ask - t.bid)


def account_balance() -> float:
    a = mt5.account_info()
    return float(a.balance) if a else 0.0


def account_equity() -> float:
    a = mt5.account_info()
    return float(a.equity) if a else 0.0


def calc_lots(symbol: str, risk_dollars: float, sl_distance_price: float) -> float:
    """Convert (risk in $ , stop distance in price units) → lots, respecting broker limits."""
    info = mt5.symbol_info(symbol)
    if info is None or sl_distance_price <= 0 or risk_dollars <= 0:
        return 0.0
    ticks = sl_distance_price / info.trade_tick_size
    dollar_loss_per_lot = ticks * info.trade_tick_value
    if dollar_loss_per_lot <= 0:
        return 0.0
    raw = risk_dollars / dollar_loss_per_lot
    # round DOWN to volume step
    step = info.volume_step or 0.01
    lots = math.floor(raw / step) * step
    lots = max(lots, 0.0)
    if lots < info.volume_min:
        return 0.0                     # too small to trade
    lots = min(lots, info.volume_max)
    return round(lots, 2)


def _order_result_ok(res) -> bool:
    if res is None:
        return False
    return res.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED)


def market_order(symbol: str, side: str, lots: float, sl: float, tp: float,
                 magic: int, comment: str) -> Optional[int]:
    """side = 'buy' or 'sell'. Returns position ticket on success, None on fail."""
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if side == "buy" else tick.bid
    req = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lots,
        "type":         mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL,
        "price":        price,
        "sl":           round(sl, info.digits),
        "tp":           round(tp, info.digits),
        "deviation":    20,
        "magic":        magic,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if not _order_result_ok(res):
        print(f"[broker] market_order FAILED: {res}")
        return None
    return res.order


def limit_order(symbol: str, side: str, price: float, lots: float,
                sl: float, tp: float, expire_ts: int,
                magic: int, comment: str) -> Optional[int]:
    """side = 'buy' or 'sell'. expire_ts = unix seconds. Returns order ticket."""
    info = mt5.symbol_info(symbol)
    req = {
        "action":       mt5.TRADE_ACTION_PENDING,
        "symbol":       symbol,
        "volume":       lots,
        "type":         mt5.ORDER_TYPE_BUY_LIMIT if side == "buy" else mt5.ORDER_TYPE_SELL_LIMIT,
        "price":        round(price, info.digits),
        "sl":           round(sl, info.digits),
        "tp":           round(tp, info.digits),
        "deviation":    20,
        "magic":        magic,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_SPECIFIED,
        "expiration":   expire_ts,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    res = mt5.order_send(req)
    if not _order_result_ok(res):
        print(f"[broker] limit_order FAILED: {res}")
        return None
    return res.order


def modify_sl(position_ticket: int, new_sl: float, tp: float, symbol: str) -> bool:
    info = mt5.symbol_info(symbol)
    req = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": position_ticket,
        "sl":       round(new_sl, info.digits),
        "tp":       round(tp, info.digits),
    }
    res = mt5.order_send(req)
    return _order_result_ok(res)


def close_partial(position_ticket: int, lots: float, symbol: str) -> bool:
    """Partially close by opening an opposite market order for `lots` volume."""
    pos = [p for p in mt5.positions_get(symbol=symbol) if p.ticket == position_ticket]
    if not pos:
        return False
    pos = pos[0]
    side = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(symbol)
    price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
    info = mt5.symbol_info(symbol)
    step = info.volume_step or 0.01
    lots = math.floor(min(lots, pos.volume) / step) * step
    if lots < info.volume_min:
        return False
    req = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       round(lots, 2),
        "type":         side,
        "position":     position_ticket,
        "price":        price,
        "deviation":    20,
        "magic":        pos.magic,
        "comment":      "partial",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    return _order_result_ok(res)


def close_position(position_ticket: int, symbol: str) -> bool:
    """Close a full position at market (opposite-side DEAL). Used by the gradient exit."""
    pos = [p for p in (mt5.positions_get(symbol=symbol) or []) if p.ticket == position_ticket]
    if not pos:
        return False
    pos = pos[0]
    side = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(symbol)
    price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
    req = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       pos.volume,
        "type":         side,
        "position":     position_ticket,
        "price":        price,
        "deviation":    20,
        "magic":        pos.magic,
        "comment":      "gradexit",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    return _order_result_ok(res)


def open_positions(symbol: str, magic: int) -> list:
    positions = mt5.positions_get(symbol=symbol) or []
    return [p for p in positions if p.magic == magic]


def pending_orders(symbol: str, magic: int) -> list:
    orders = mt5.orders_get(symbol=symbol) or []
    return [o for o in orders if o.magic == magic]


def cancel_order(order_ticket: int) -> bool:
    req = {"action": mt5.TRADE_ACTION_REMOVE, "order": order_ticket}
    res = mt5.order_send(req)
    return _order_result_ok(res)


def cancel_all_pending(symbol: str, magic: int, comment: Optional[str] = None) -> int:
    """Cancel every pending order on `symbol` with this magic (optionally also
    matching `comment`). Returns how many were cancelled."""
    n = 0
    for o in pending_orders(symbol, magic):
        if comment is not None and o.comment != comment:
            continue
        if cancel_order(o.ticket):
            n += 1
    return n


def place_heartbeat(symbol: str, price: float, magic: int, comment: str) -> Optional[int]:
    """Place a never-fill BUY-LIMIT (min volume, no SL/TP, GTC) as a visible
    'bot is alive' marker in the terminal. `price` is set absurdly low so it can
    never trigger. Returns the order ticket, or None on failure (non-fatal)."""
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"[broker] heartbeat: symbol_info({symbol}) is None")
        return None
    vol = info.volume_min or 0.01
    req = {
        "action":       mt5.TRADE_ACTION_PENDING,
        "symbol":       symbol,
        "volume":       round(vol, 2),
        "type":         mt5.ORDER_TYPE_BUY_LIMIT,
        "price":        round(price, info.digits),
        "magic":        magic,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    res = mt5.order_send(req)
    if not _order_result_ok(res):
        print(f"[broker] heartbeat order FAILED: {res}")
        return None
    return res.order


def is_market_open(symbol: str) -> bool:
    """Rough check via last tick recency (broker weekend closed → tick > minutes old)."""
    t = mt5.symbol_info_tick(symbol)
    if t is None:
        return False
    age = time.time() - t.time
    return age < 300           # tick within 5 min = market active
