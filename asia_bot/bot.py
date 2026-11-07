"""
Asia-sweep + London 1M-CISD morning bot (EURUSD).

Run every morning (or leave running — it sleeps outside the session):
    .venv\\Scripts\\python.exe -m asia_bot.bot

Per symbol, per session (one trade per pair per session):
  1. Asia range 00:00-07:00 UTC+1
  2. First sweep of Asia high/low during London 07:00-10:00 UTC+1
  3. Asia HIGH swept → wait for bearish 1M CISD; LOW swept → bullish CISD
  4. On CISD confirmation: LIMIT order at the CISD level ("enter off the
     level"), SL at the manipulation swing high/low, TP at 2R
  5. Unfilled limits are cancelled at London end; positions run to SL/TP
  6. Every event is appended to asia_bot/trades.csv

Protections: dynamic %-risk sizing · daily loss stop · overall drawdown halt ·
max-spread guard · min-SL guard · one position per symbol · weekend/market
guard · broker-offset auto-detection (DST-proof sessions) · state survives
restarts.
"""
import os
import sys
import time
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asia_bot import config as cfg
from asia_bot import detector as det
from live import broker


# ─────────────────────────────────────────────────────────────────────────────
# Logging / state / CSV
# ─────────────────────────────────────────────────────────────────────────────

# Force UTF-8 on stdout/stderr so log glyphs (── ✅ ⛔) never crash the bot when
# output is redirected to a cp1252 file/console on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    line = f"[{ts}] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    try:
        with open(cfg.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


DEFAULT_STATE = {
    "session_date": None,          # UTC+1 date the pairs dict refers to
    "pairs": {},                   # symbol -> {status, ticket, ...}
    "day_start_balance": None,
    "anchor_balance": None,        # for the overall-DD floor
    "halted": False,
}


def load_state() -> dict:
    if os.path.exists(cfg.STATE_FILE):
        try:
            with open(cfg.STATE_FILE) as f:
                return {**DEFAULT_STATE, **json.load(f)}
        except Exception as e:
            log(f"[state] load failed ({e}) — starting fresh")
    return dict(DEFAULT_STATE)


def save_state(state: dict):
    tmp = cfg.STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    for i in range(3):
        try:
            os.replace(tmp, cfg.STATE_FILE)
            return
        except PermissionError:
            time.sleep(0.2 * (i + 1))


CSV_COLS = ["session_date", "symbol", "direction", "asia_high", "asia_low",
            "sweep_time_utc1", "confirm_time_utc1", "cisd_level", "entry_source",
            "entry", "sl", "tp", "risk_pips", "lots", "risk_usd",
            "ticket", "status", "fill_price", "fill_time",
            "outcome", "pnl_usd", "pnl_r", "close_time"]


def csv_upsert(row: dict):
    """Insert or update the (session_date, symbol) row in trades.csv."""
    if os.path.exists(cfg.TRADES_CSV):
        df = pd.read_csv(cfg.TRADES_CSV, dtype=str)
    else:
        df = pd.DataFrame(columns=CSV_COLS)
    for col in CSV_COLS:
        if col not in df.columns:
            df[col] = ""
    mask = (df["session_date"] == str(row["session_date"])) & \
           (df["symbol"] == row["symbol"])
    if mask.any():
        i = df.index[mask][0]
        for k, v in row.items():
            df.loc[i, k] = str(v)
    else:
        full = {k: str(row.get(k, "")) for k in CSV_COLS}
        df = pd.concat([df, pd.DataFrame([full])], ignore_index=True)
    tmp = cfg.TRADES_CSV + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, cfg.TRADES_CSV)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def now_utc1(offset_h: int):
    """Current UTC+1 time derived from a broker tick (fallback: wall clock)."""
    return pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None) \
           + pd.Timedelta(hours=1)


def broker_offset(anchor_symbol: str) -> int:
    t = mt5.symbol_info_tick(anchor_symbol)
    if t is None:
        return 3                                  # sane fallback (EEST)
    return det.broker_offset_hours(t.time)


def day_pnl_pct(state) -> float:
    bal = broker.account_equity()
    start = state.get("day_start_balance") or bal
    return (bal - start) / start * 100 if start else 0.0


def check_overall_dd(state) -> bool:
    """True = halted."""
    anchor = cfg.INITIAL_BALANCE or state.get("anchor_balance")
    if not anchor:
        return False
    floor = anchor * (1 - cfg.MAX_OVERALL_DD_PCT / 100)
    if broker.account_equity() <= floor:
        if not state["halted"]:
            log(f"⛔ OVERALL DD HALT — equity ${broker.account_equity():,.2f} "
                f"<= floor ${floor:,.2f}. No further trading.")
        state["halted"] = True
    return state["halted"]


def reconcile_closed(state):
    """Detect closed positions, fetch the closing deal, log outcome to CSV."""
    for sym, meta in list(state["pairs"].items()):
        if meta.get("status") != "filled":
            continue
        ticket = int(meta["position_ticket"])
        still_open = any(p.ticket == ticket
                         for p in broker.open_positions(sym, cfg.MAGIC))
        if still_open:
            continue
        # find closing deal
        deals = mt5.history_deals_get(position=ticket) or []
        out_deals = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT]
        pnl = sum(d.profit + d.commission + d.swap for d in out_deals) if out_deals else 0.0
        risk_usd = float(meta.get("risk_usd") or 0) or 1.0
        outcome = "win" if pnl > 0 else "loss"
        close_t = datetime.fromtimestamp(out_deals[-1].time, tz=timezone.utc) \
            if out_deals else None
        log(f"[{sym}] position #{ticket} closed → {outcome.upper()} "
            f"{pnl:+.2f} USD ({pnl / risk_usd:+.2f}R)")
        csv_upsert({"session_date": meta["session_date"], "symbol": sym,
                    "status": "closed", "outcome": outcome,
                    "pnl_usd": round(pnl, 2), "pnl_r": round(pnl / risk_usd, 2),
                    "close_time": close_t})
        meta["status"] = "closed"
        save_state(state)


def manage_profit_close(state):
    """Primary exit: market-close any of OUR open positions once floating profit
    (profit + swap) reaches cfg.PROFIT_CLOSE_USD. The broker TP at cfg.RR is the
    backstop if the bot misses the tick. reconcile_closed() logs the result."""
    target = getattr(cfg, "PROFIT_CLOSE_USD", 0) or 0
    if target <= 0:
        return
    for sym in cfg.SYMBOLS:
        for p in broker.open_positions(sym, cfg.MAGIC):
            floating = p.profit + p.swap
            if floating >= target:
                if broker.close_position(p.ticket, sym):
                    log(f"[{sym}] 💰 PROFIT-CLOSE #{p.ticket} at ${floating:,.2f} "
                        f"(>= ${target:.0f} target)")
                else:
                    log(f"[{sym}] profit-close FAILED for #{p.ticket} "
                        f"(floating ${floating:,.2f}) — retry next loop")


def promote_fills(state):
    """Pending limit filled → track the position + log fill."""
    for sym, meta in list(state["pairs"].items()):
        if meta.get("status") != "pending":
            continue
        order_ticket = int(meta["ticket"])
        pend = [o.ticket for o in broker.pending_orders(sym, cfg.MAGIC)]
        if order_ticket in pend:
            continue                              # still resting
        # order gone → filled or cancelled/expired. A new position?
        pos = broker.open_positions(sym, cfg.MAGIC)
        if pos:
            p = pos[0]
            meta["status"] = "filled"
            meta["position_ticket"] = p.ticket
            log(f"[{sym}] limit FILLED #{p.ticket} @ {p.price_open} ({p.volume} lots)")
            csv_upsert({"session_date": meta["session_date"], "symbol": sym,
                        "status": "filled", "fill_price": p.price_open,
                        "fill_time": datetime.now(timezone.utc), "ticket": p.ticket})
        else:
            meta["status"] = "expired"
            log(f"[{sym}] limit #{order_ticket} gone without fill (expired/cancelled)")
            csv_upsert({"session_date": meta["session_date"], "symbol": sym,
                        "status": "expired", "outcome": "no_fill"})
        save_state(state)


def cancel_pendings_after_london(state, offset_h):
    for sym, meta in list(state["pairs"].items()):
        if meta.get("status") != "pending":
            continue
        for o in broker.pending_orders(sym, cfg.MAGIC):
            if broker.cancel_order(o.ticket):
                log(f"[{sym}] cancelled unfilled limit #{o.ticket} (London over)")
        meta["status"] = "expired"
        csv_upsert({"session_date": meta["session_date"], "symbol": sym,
                    "status": "expired", "outcome": "no_fill"})
        save_state(state)


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

def try_enter(symbol: str, state: dict, offset_h: int, session_date):
    meta = state["pairs"].get(symbol, {})
    if meta.get("status") in ("pending", "filled", "closed", "expired", "skipped"):
        return                                    # one trade per pair per session
    if len(broker.open_positions(symbol, cfg.MAGIC)) >= 1:
        return

    df = broker.get_bars(symbol, "M1", cfg.M1_BARS)
    if len(df) < 100:
        return
    setup = det.analyze_day(df, symbol, offset_h, session_date=session_date)
    if setup is None or setup.stage != "confirmed":
        return

    bid, ask, spread = broker.get_tick(symbol)
    if bid is None:
        return
    pip = cfg.PIP.get(symbol, 1e-4)

    def _skip(reason):
        log(f"[{symbol}] setup CONFIRMED but skipped: {reason}")
        state["pairs"][symbol] = {"status": "skipped",
                                  "session_date": str(session_date)}
        csv_upsert({"session_date": session_date, "symbol": symbol,
                    "direction": setup.direction, "asia_high": setup.asia_high,
                    "asia_low": setup.asia_low, "sweep_time_utc1": setup.sweep_time,
                    "confirm_time_utc1": setup.confirm_time,
                    "cisd_level": setup.cisd_level,
                    "status": "skipped", "outcome": reason})
        save_state(state)

    # ── Protections ──
    if spread > cfg.MAX_SPREAD_PIPS * pip:
        return _skip(f"spread {spread/pip:.1f}p > {cfg.MAX_SPREAD_PIPS}p")
    if day_pnl_pct(state) <= -cfg.DAILY_STOP_PCT:
        return _skip(f"daily stop hit ({day_pnl_pct(state):.1f}%)")

    market_mode = cfg.ENTRY_MODE == "market"

    # "Too big" test (market mode only): if the confirmation candle closed more
    # than CISD_BIG_CANDLE_MULT × the (level→swing) stop distance BEYOND the CISD
    # level, it's too extended to chase at market — rest a limit at the CISD
    # level instead so we enter on the retrace.
    too_big = False
    if market_mode:
        stop_to_level = abs(setup.sweep_extreme - setup.cisd_level)
        close_beyond  = abs(setup.confirm_close - setup.cisd_level)
        too_big = stop_to_level > 0 and close_beyond > cfg.CISD_BIG_CANDLE_MULT * stop_to_level

    if market_mode and not too_big:
        prices = det.build_market_prices(setup, symbol, spread)
        entry_desc = "market @ CISD close"
    elif too_big:
        prices = det.build_order_prices(setup, symbol, spread, entry_override=setup.cisd_level)
        entry_desc = "LIMIT @ CISD level (candle too big)"
    else:
        prices = det.build_order_prices(setup, symbol, spread)
        entry_desc = "FVG " + ("high" if setup.direction == "buy" else "low") \
            if setup.entry_source == "fvg" else "CISD level"
    if prices is None:
        return _skip("risk<=0")
    entry, sl, tp, risk = prices["entry"], prices["sl"], prices["tp"], prices["risk"]
    entry_level = prices["entry_level"]     # bid-chart level (confirm close / CISD / FVG)

    balance = broker.account_balance()
    # Flat $95 risk below the anchor (→ $200 at TP, 2.105R), 0.95% once past it.
    risk_usd = (cfg.RISK_FLAT if balance < cfg.RISK_ANCHOR
                else balance * (cfg.RISK_PCT / 100))
    lots = broker.calc_lots(symbol, risk_usd, risk)
    if lots <= 0:
        return _skip(f"lots=0 for risk ${risk_usd:.2f} / stop {risk/pip:.1f}p")

    # London end (broker unix seconds) for the limit expiry
    sb = det.session_bounds_utc1(session_date)
    exp_broker = det.utc1_to_broker(sb["london_end"], offset_h)
    expire_ts = int(exp_broker.timestamp())

    log(f"[{symbol}] CISD {setup.direction.upper()} confirmed {setup.confirm_time} "
        f"· entry {entry:.5f} @ {entry_desc} · swing "
        f"{'high' if setup.direction=='sell' else 'low'} {setup.sweep_extreme:.5f} "
        f"· sl {sl:.5f} · tp {tp:.5f} ({cfg.RR:.1f}R, {risk/pip:.1f}p stop) "
        f"· {lots} lots (${risk_usd:.2f})")

    # BUY limit fills when ASK <= its price → place at entry_level+spread so it
    # triggers when bid taps the level. SELL limit fills when BID >= its price →
    # place at entry_level. If the bid is already through it, market-in now.
    if setup.direction == "buy":
        limit_price = entry_level + spread
        marketable = bid <= entry_level
    else:
        limit_price = entry_level
        marketable = bid >= entry_level

    do_market = (market_mode and not too_big) or marketable
    if do_market:
        if marketable and not (market_mode and not too_big):
            log(f"[{symbol}] bid already at the entry level — market-in instead of limit")
        tk = broker.market_order(symbol, setup.direction, lots, sl, tp,
                                 cfg.MAGIC, cfg.COMMENT)
        status = "filled" if tk else None
    else:
        tk = broker.limit_order(symbol, setup.direction, limit_price, lots, sl, tp,
                                expire_ts, cfg.MAGIC, cfg.COMMENT)
        status = "pending" if tk else None
    if tk is None:
        log(f"[{symbol}] order placement FAILED — will retry next poll")
        return

    state["pairs"][symbol] = {
        "status": status, "ticket": tk, "session_date": str(session_date),
        "risk_usd": risk_usd, "direction": setup.direction,
    }
    if status == "filled":
        time.sleep(1.0)
        pos = broker.open_positions(symbol, cfg.MAGIC)
        if pos:
            state["pairs"][symbol]["position_ticket"] = pos[0].ticket
    csv_upsert({"session_date": session_date, "symbol": symbol,
                "direction": setup.direction, "asia_high": setup.asia_high,
                "asia_low": setup.asia_low, "sweep_time_utc1": setup.sweep_time,
                "confirm_time_utc1": setup.confirm_time,
                "cisd_level": setup.cisd_level, "entry_source": setup.entry_source,
                "entry": entry, "sl": round(sl, 6), "tp": round(tp, 6),
                "risk_pips": round(risk / pip, 1), "lots": lots,
                "risk_usd": round(risk_usd, 2), "ticket": tk, "status": status})
    save_state(state)
    log(f"[{symbol}] {'MARKET order' if status=='filled' else 'LIMIT resting'} "
        f"#{tk} @ {entry:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# Gold heartbeat — a never-fill BUY LIMIT so you can SEE the bot is alive
# ─────────────────────────────────────────────────────────────────────────────

def place_heartbeat():
    """Place a never-fill gold buy-limit. Returns (symbol, ticket) or (None, None)."""
    if not cfg.HEARTBEAT_ENABLED:
        return None, None
    for sym in cfg.HEARTBEAT_SYMBOLS:
        if not mt5.symbol_select(sym, True):
            continue
        info = mt5.symbol_info(sym)
        if info is None:
            continue
        # cancel any stale heartbeat from a previous run first
        for o in (mt5.orders_get(symbol=sym) or []):
            if o.magic == cfg.HEARTBEAT_MAGIC:
                mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket})
        req = {
            "action":       mt5.TRADE_ACTION_PENDING,
            "symbol":       sym,
            "volume":       info.volume_min,
            "type":         mt5.ORDER_TYPE_BUY_LIMIT,
            "price":        cfg.HEARTBEAT_PRICE,
            "magic":        cfg.HEARTBEAT_MAGIC,
            "comment":      cfg.HEARTBEAT_COMMENT,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        res = mt5.order_send(req)
        if res is not None and res.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            log(f"heartbeat placed on {sym} #{res.order} @ {cfg.HEARTBEAT_PRICE} "
                f"({info.volume_min} lots, never fills) — bot is ALIVE")
            return sym, res.order
        log(f"heartbeat on {sym} failed: {res.retcode if res else 'None'} — trying next")
    log("heartbeat: no gold symbol available — bot runs without alive-marker")
    return None, None


def cancel_heartbeat(sym, ticket):
    if sym and ticket:
        mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": ticket})
        log(f"heartbeat cancelled on {sym} #{ticket}")


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not broker.connect():
        return
    for s in cfg.SYMBOLS:
        if not broker.ensure_symbol(s):
            log(f"failed to subscribe {s} — aborting")
            return

    hb_sym, hb_ticket = place_heartbeat()

    state = load_state()
    if state.get("anchor_balance") is None:
        state["anchor_balance"] = cfg.INITIAL_BALANCE or broker.account_balance()
        save_state(state)

    anchor = cfg.SYMBOLS[0]
    offset_h = broker_offset(anchor)
    log(f"asia_bot starting · {cfg.SYMBOLS} · risk {cfg.RISK_PCT}%/trade · RR {cfg.RR} "
        f"· entry={cfg.ENTRY_MODE} · broker offset UTC+{offset_h} · "
        f"sessions (UTC+1): Asia {cfg.ASIA_START_H:02d}-{cfg.ASIA_END_H:02d}, "
        f"London {cfg.LONDON_START_H:02d}-{cfg.LONDON_END_H:02d}")

    try:
        while True:
            try:
                if not broker.is_market_open(anchor):
                    time.sleep(60)
                    continue
                offset_h = broker_offset(anchor)
                now1 = now_utc1(offset_h)
                today = now1.date()
                sb = det.session_bounds_utc1(today)

                # New session day → reset per-pair state + daily anchor
                if state["session_date"] != str(today):
                    state["session_date"] = str(today)
                    state["pairs"] = {}
                    state["day_start_balance"] = broker.account_balance()
                    save_state(state)
                    log(f"── new session {today} · balance "
                        f"${state['day_start_balance']:,.2f} ──")

                promote_fills(state)
                manage_profit_close(state)
                reconcile_closed(state)

                if check_overall_dd(state):
                    time.sleep(300)
                    continue

                if now1 < sb["london_start"]:
                    wait = min((sb["london_start"] - now1).total_seconds(), 300)
                    time.sleep(max(wait, cfg.LOOP_SEC))
                    continue

                if now1 >= sb["london_end"]:
                    cancel_pendings_after_london(state, offset_h)
                    open_any = any(m.get("status") in ("pending", "filled")
                                   for m in state["pairs"].values())
                    # poll fast while a position is still open so the $200
                    # profit-close stays responsive after London ends
                    time.sleep(cfg.LOOP_SEC if open_any else 300)
                    continue

                # London is live — hunt for entries
                for symbol in cfg.SYMBOLS:
                    try_enter(symbol, state, offset_h, today)

            except Exception as e:
                log(f"LOOP ERROR: {e!r}")
            time.sleep(cfg.LOOP_SEC)
    except KeyboardInterrupt:
        log("interrupted — saving state")
        save_state(state)
    finally:
        cancel_heartbeat(hb_sym, hb_ticket)
        broker.disconnect()


if __name__ == "__main__":
    main()
