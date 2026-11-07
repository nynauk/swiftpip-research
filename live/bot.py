"""
Live bot — multi-symbol Option A deployment.

Runs 24/7. For each symbol in cfg.SYMBOLS:
  1. Ensure fresh H4 + M15 bars
  2. Prune closed positions from state
  3. Reap expired retest limit orders
  4. Detect the newest ready sweep-setup we haven't traded
  5. News-filter check
  6. Size @ RISK_PCT, place market or retest-limit order
  7. Persist state

Run from project root:
    .venv\\Scripts\\python.exe -m live.bot
"""
import os
import sys
import time
from datetime import date, datetime, timezone, timedelta
import MetaTrader5 as mt5

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live import config as cfg
from live import broker
from live import state as st
from live.detector import detect_latest_setup
from live.detector_ema_dol import detect_ema_dol_signals
from live import manager
import news_filter as nf


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(cfg.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ── News block windows: build once at boot, cover ±2 years for margin ──
# Blocks only the specific 4H candle window containing each release, not the
# whole day. Windows are (date, h4_start_hour) tuples in broker time.
def _build_news_windows():
    if not cfg.NEWS_FILTER:
        return set()
    today = date.today()
    return nf.build_news_block_windows(today - timedelta(days=730),
                                        today + timedelta(days=730))


# Full-day block set — the EMA+DOL backtest skipped entire red-folder days,
# so the ema_dol strategy must match that (not the H4-window variant).
def _build_news_days():
    if not cfg.NEWS_FILTER:
        return set()
    today = date.today()
    return nf.build_news_block_set(today - timedelta(days=730),
                                   today + timedelta(days=730))


NEWS_WINDOWS = _build_news_windows()
NEWS_DAYS = _build_news_days()


def _today_broker(anchor_symbol: str) -> str:
    t = mt5.symbol_info_tick(anchor_symbol)
    if t is None:
        return datetime.now(timezone.utc).date().isoformat()
    return datetime.fromtimestamp(t.time, tz=timezone.utc).date().isoformat()


def _update_daily(state: dict, anchor_symbol: str):
    today = _today_broker(anchor_symbol)
    if state["day"] != today:
        state["day"] = today
        state["day_start_balance"] = broker.account_balance()
        state["day_pnl_pct"] = 0.0
        state["day_trades"] = 0                       # reset daily trade cap at (broker) midnight
        log(f"── new day {today} · start balance ${state['day_start_balance']:,.2f} · "
            f"daily trade cap reset (max {cfg.MAX_TRADES_PER_DAY})")
    else:
        bal = broker.account_equity()
        start = state["day_start_balance"] or bal
        state["day_pnl_pct"] = (bal - start) / start * 100 if start else 0.0


_OVERALL_HALTED = False


def _overall_ok(state: dict) -> bool:
    """FTMO overall-loss guard: block NEW entries if equity draws to 
    -MAX_OVERALL_DD_PCT from the initial balance (static, like FTMO max loss).
    Open positions are still managed to their gradient exit."""
    global _OVERALL_HALTED
    override = getattr(cfg, "INITIAL_BALANCE", None)
    if override is not None:
        state["init_balance"] = float(override)          # FTMO static starting balance
    elif state.get("init_balance") is None:
        state["init_balance"] = broker.account_balance()  # auto-anchor on first connect
    floor = state["init_balance"] * (1 - getattr(cfg, "MAX_OVERALL_DD_PCT", 9.0) / 100)
    eq = broker.account_equity()
    if eq <= floor:
        if not _OVERALL_HALTED:
            log(f"⛔ overall drawdown guard: equity ${eq:,.2f} <= floor ${floor:,.2f} "
                f"(-{cfg.MAX_OVERALL_DD_PCT}% of ${state['init_balance']:,.2f}) — halting new entries")
            _OVERALL_HALTED = True
        return False
    _OVERALL_HALTED = False
    return True


def _reap_expired_retests(symbol: str, state: dict):
    live_orders = {o.ticket for o in broker.pending_orders(symbol, cfg.MAGIC)}
    now = int(time.time())
    to_del = []
    for tk, meta in list(state["pending_retests"].items()):
        if meta.get("symbol") != symbol:
            continue
        tk_i = int(tk)
        if tk_i not in live_orders:
            to_del.append(tk)
            continue
        if meta["expires_at"] < now:
            if broker.cancel_order(tk_i):
                log(f"[{symbol}] cancelled expired retest limit #{tk_i}")
            to_del.append(tk)
    for k in to_del:
        state["pending_retests"].pop(k, None)


def _promote_filled_retests(symbol: str, state: dict):
    positions = broker.open_positions(symbol, cfg.MAGIC)
    known = set(state["positions"].keys())
    for p in positions:
        key = str(p.ticket)
        if key in known:
            continue
        direction = "buy" if p.type == mt5.POSITION_TYPE_BUY else "sell"
        state["positions"][key] = {
            "symbol": symbol,
            "entry": p.price_open, "sl": p.sl, "tp": p.tp,
            "risk_price": abs(p.price_open - p.sl) if p.sl else 0.0,
            "direction": direction,
            "original_lots": p.volume,
            "sweep_time": None,
        }
        log(f"[{symbol}] tracking new position #{p.ticket} ({direction} {p.volume} @ {p.price_open})")


def _positions_for_symbol(state: dict, symbol: str) -> int:
    return sum(1 for m in state["positions"].values() if m.get("symbol") == symbol)


def _try_enter_ema_dol(symbol: str, state: dict, tf: str = "M15"):
    """EMA(7/18) + DOL entry for ONE timeframe engine — market-in the moment
    price tags the FVG edge, with SL/RR computed at that instant (matches the
    backtest's retest-bar evaluation). Each TF engine has its own magic, its own
    per-symbol dedup bucket, and 1 position per symbol per TF."""
    magic = cfg.TF_MAGIC[tf]
    dedup_key = f"{symbol}|{tf}"          # per-(symbol, TF) consumed-cross bucket
    if len(broker.open_positions(symbol, magic)) >= cfg.MAX_CONCURRENT_PER_SYMBOL:
        return
    if state["day_pnl_pct"] <= -cfg.DAILY_STOP_PCT:
        return
    if state.get("day_trades", 0) >= cfg.MAX_TRADES_PER_DAY:
        return                                        # hard daily trade cap reached

    df_15 = broker.get_bars(symbol, tf, cfg.BARS_BACK)
    if len(df_15) < 130:
        return
    closed = df_15.iloc[:-1]              # drop the currently-forming bar

    bid, ask, spread = broker.get_tick(symbol)
    if bid is None or ask is None:
        return

    already = set(st.traded_sweeps_for(state, dedup_key))
    # Widen the SL by (live spread + 1 pip) — parity with the backtest's edr.sl_pad.
    sl_pad = max(spread, 0.0) + cfg.PIP.get(symbol, spread)
    signals = detect_ema_dol_signals(
        closed, already, bid=bid, ask=ask,
        fast=cfg.EMA_FAST, slow=cfg.EMA_SLOW,
        swing_n=cfg.SWING_LOOKBACK,
        fvg_window=cfg.FVG_LOOKUP_WINDOW,
        retest_window=cfg.RETEST_WINDOW_EMA,
        sl_buffer_frac=cfg.SL_BUFFER_FRAC,
        min_rr=cfg.MIN_RR, max_rr=cfg.MAX_RR,
        min_fvg_width=cfg.MIN_FVG_WIDTH.get(symbol, 0.0002),
        rsi_period=cfg.RSI_PERIOD, rsi_long_min=cfg.RSI_LONG_MIN,
        rsi_short_max=cfg.RSI_SHORT_MAX,
        sl_pad=sl_pad,
    )
    if not signals:
        return

    entered = False
    for setup in signals:
        key = str(setup.cross_time)

        if setup.entry_mode == "missed":
            log(f"[{symbol} {tf}] cross {key}: {setup.reason} — consumed, no trade")
            st.traded_sweeps_for(state, dedup_key).append(key)
            continue
        if setup.entry_mode == "reject":
            log(f"[{symbol} {tf}] cross {key}: {setup.reason} — consumed, no trade")
            st.traded_sweeps_for(state, dedup_key).append(key)
            continue
        if entered:
            continue                      # 1 entry per poll; others rescan next loop

        # News filter — EMA+DOL backtest blocked the ENTIRE red-folder day
        tick = mt5.symbol_info_tick(symbol)
        if cfg.NEWS_FILTER and tick is not None:
            broker_date = datetime.fromtimestamp(tick.time, tz=timezone.utc).date()
            if broker_date in NEWS_DAYS:
                log(f"[{symbol} {tf}] tag on {broker_date} is a red-folder day — consumed")
                st.traded_sweeps_for(state, dedup_key).append(key)
                continue

        balance = broker.account_balance()
        risk_dollars = balance * (cfg.RISK_PCT / 100)
        lots = broker.calc_lots(symbol, risk_dollars, setup.risk_price)
        if lots <= 0:
            log(f"[{symbol} {tf}] computed lots=0 (risk ${risk_dollars:.2f}, "
                f"stop {setup.risk_price:.5f}) — consumed")
            st.traded_sweeps_for(state, dedup_key).append(key)
            continue

        # Gradient exit is managed live (manager.py). Place a FAR safety TP
        # (SAFETY_TP_R · risk) as bot-death insurance — rarely hit; the manager
        # closes the trade when the fast-EMA slope flips in profit / opposite cross.
        if setup.direction == "buy":
            safety_tp = ask + cfg.SAFETY_TP_R * setup.risk_price
        else:
            safety_tp = bid - cfg.SAFETY_TP_R * setup.risk_price
        log(f"[{symbol} {tf}] GRAD {setup.direction.upper()} @ edge {setup.edge_price} "
            f"· sl {setup.sl_price:.5f} · safetyTP {safety_tp:.5f} "
            f"· dol-rr {setup.rr:.2f} · lots {lots} · spread {spread:.5f}")
        tk = broker.market_order(symbol, setup.direction, lots,
                                 setup.sl_price, safety_tp, magic, cfg.TF_COMMENT[tf])
        if tk is None:
            continue                      # order failed — do NOT consume; retry next poll
        st.traded_sweeps_for(state, dedup_key).append(key)
        state["day_trades"] = state.get("day_trades", 0) + 1
        st.save(cfg.STATE_FILE, state)
        entered = True
        log(f"[{symbol} {tf}] ENTERED {setup.direction} · daily trade "
            f"{state['day_trades']}/{cfg.MAX_TRADES_PER_DAY}")

        time.sleep(1.0)
        for p in broker.open_positions(symbol, magic):
            if str(p.ticket) not in state["positions"]:
                state["positions"][str(p.ticket)] = {
                    "symbol": symbol, "tf": tf,
                    "entry": p.price_open, "sl": setup.sl_price, "tp": safety_tp,
                    "risk_price": abs(p.price_open - setup.sl_price),
                    "direction": setup.direction, "original_lots": p.volume,
                    "sweep_time": key,
                }
                log(f"[{symbol} {tf}] filled #{p.ticket} @ {p.price_open}")
                break


def _try_enter(symbol: str, state: dict):
    # Per-symbol concurrency
    live_count = len(broker.open_positions(symbol, cfg.MAGIC))
    if live_count >= cfg.MAX_CONCURRENT_PER_SYMBOL:
        return
    # Account-wide daily stop
    if state["day_pnl_pct"] <= -cfg.DAILY_STOP_PCT:
        return

    df_15 = broker.get_bars(symbol, "M15", cfg.M15_BARS_BACK)
    df_h4 = broker.get_bars(symbol, "H4",  cfg.H4_BARS_BACK)
    if len(df_15) < 30 or len(df_h4) < 3:
        return

    already = set(st.traded_sweeps_for(state, symbol))
    setup = detect_latest_setup(
        df_15, df_h4, already,
        rev_window=cfg.REV_WINDOW_M15,
        dis_delay=cfg.DIS_DELAY_M15,
        retr_window=cfg.RETR_WINDOW_M15,
        dis_mult=cfg.DIS_MULT,
        rr=cfg.RR,
    )
    if setup is None:
        return

    # News filter — block ONLY the 4H window containing the news release
    # (not the whole day). Uses broker-time from the last tick.
    tick = mt5.symbol_info_tick(symbol)
    if tick is not None:
        # tick.time is broker time as unix seconds (MT5 quirk: broker time treated
        # as if UTC). Convert to naive datetime → matches broker-time semantics
        # used by is_blocked_h4.
        tick_dt = datetime.fromtimestamp(tick.time, tz=timezone.utc).replace(tzinfo=None)
        if nf.is_blocked_h4(tick_dt, NEWS_WINDOWS):
            h4_start = (tick_dt.hour // 4) * 4
            log(f"[{symbol}] setup found but H4 window {tick_dt.date()} "
                f"{h4_start:02d}:00-{(h4_start+4)%24:02d}:00 is news-blocked — skipping")
            st.traded_sweeps_for(state, symbol).append(str(setup.sweep_time))
            return

    # ── SPREAD-ADJUSTED SIZING ──
    # The setup uses close-of-disrespect-candle as "entry", but the market fills
    # a BUY at ASK (> close) and closes a BUY at BID. So real risk from fill to
    # SL = ASK - SL, which is bigger than close - SL. Size on the real distance
    # so 0.33% actually means 0.33%, not 0.5%+.
    bid, ask, spread = broker.get_tick(symbol)
    if bid is None or ask is None:
        return
    if setup.direction == "buy":
        real_entry = ask
        real_risk_price = real_entry - setup.sl_price
        real_tp = real_entry + cfg.RR * real_risk_price
    else:
        real_entry = bid
        real_risk_price = setup.sl_price - real_entry
        real_tp = real_entry - cfg.RR * real_risk_price

    if real_risk_price <= 0:
        log(f"[{symbol}] real risk_price ≤ 0 after spread adjustment "
            f"(spread {spread:.5f} bigger than intended SL distance {setup.risk_price:.5f}) — skipping")
        st.traded_sweeps_for(state, symbol).append(str(setup.sweep_time))
        return

    balance = broker.account_balance()
    risk_dollars = balance * (cfg.RISK_PCT / 100)
    lots = broker.calc_lots(symbol, risk_dollars, real_risk_price)
    if lots <= 0:
        log(f"[{symbol}] setup found but computed lots={lots} "
            f"(risk ${risk_dollars:.2f}, real_stop {real_risk_price:.5f}) — skipping")
        st.traded_sweeps_for(state, symbol).append(str(setup.sweep_time))
        return

    # For the actual order we pass the recomputed TP so realized RR = cfg.RR
    setup_entry = real_entry
    setup_tp = real_tp

    overshoot_pct = (real_risk_price - setup.risk_price) / setup.risk_price * 100
    log(f"[{symbol}] SETUP · {setup.direction.upper()} · {setup.entry_type} · "
        f"close_entry {setup.entry_price} → fill_entry {real_entry} · sl {setup.sl_price} "
        f"· tp {real_tp:.5f} · spread {spread:.5f} "
        f"(+{overshoot_pct:.0f}% vs close-risk) · lots {lots} · target ${risk_dollars:.2f}")

    if setup.entry_type == "retest_pending":
        # A retest limit fills at setup.entry_price (the FVG edge) — no spread
        # adjustment needed for entry, but keep the original TP relative to that
        # limit price so RR is preserved on fill.
        expire_ts = int(time.time()) + cfg.RETR_WINDOW_M15 * 15 * 60
        tk = broker.limit_order(
            symbol, setup.direction, setup.entry_price, lots,
            setup.sl_price, setup.tp_price, expire_ts, cfg.MAGIC, cfg.COMMENT
        )
        if tk is not None:
            state["pending_retests"][str(tk)] = {
                "symbol": symbol,
                "sweep_time": str(setup.sweep_time),
                "expires_at": expire_ts,
                "entry": setup.entry_price, "sl": setup.sl_price, "tp": setup.tp_price,
                "risk_price": setup.risk_price, "direction": setup.direction,
                "lots": lots,
            }
            st.traded_sweeps_for(state, symbol).append(str(setup.sweep_time))
            log(f"[{symbol}] placed retest limit #{tk} @ {setup.entry_price}")
    else:
        # Market order: pass the spread-adjusted TP so realized RR = cfg.RR.
        tk = broker.market_order(
            symbol, setup.direction, lots,
            setup.sl_price, setup_tp, cfg.MAGIC, cfg.COMMENT
        )
        if tk is not None:
            # CRITICAL: mark traded + persist BEFORE the position lookup,
            # so a slow lookup can't let the same sweep fire again next poll.
            st.traded_sweeps_for(state, symbol).append(str(setup.sweep_time))
            st.save(cfg.STATE_FILE, state)

            time.sleep(1.0)
            positions = broker.open_positions(symbol, cfg.MAGIC)
            newp = None
            for p in positions:
                if str(p.ticket) not in state["positions"]:
                    newp = p; break
            if newp is not None:
                actual_risk_price = abs(newp.price_open - setup.sl_price)
                actual_risk_dollars = actual_risk_price / mt5.symbol_info(symbol).trade_tick_size \
                                      * mt5.symbol_info(symbol).trade_tick_value * newp.volume
                state["positions"][str(newp.ticket)] = {
                    "symbol": symbol,
                    "entry": newp.price_open, "sl": setup.sl_price, "tp": setup_tp,
                    "risk_price": actual_risk_price, "direction": setup.direction,
                    "original_lots": newp.volume,
                    "sweep_time": str(setup.sweep_time),
                }
                log(f"[{symbol}] MARKET filled #{newp.ticket} @ {newp.price_open} "
                    f"· actual $ risk if SL hits: ${actual_risk_dollars:.2f} (target ${risk_dollars:.2f})")
            else:
                log(f"[{symbol}] market order #{tk} placed but no matching position yet (marked traded)")


def _place_heartbeat():
    """Drop a never-fill XAUUSD buy-limit as a visible 'bot is running' marker."""
    if not getattr(cfg, "HEARTBEAT_ENABLED", False):
        return
    sym = cfg.HEARTBEAT_SYMBOL
    if not broker.ensure_symbol(sym):
        log(f"heartbeat: could not subscribe {sym} — skipping marker (bot continues)")
        return
    hb_magic = getattr(cfg, "HEARTBEAT_MAGIC", cfg.MAGIC)
    broker.cancel_all_pending(sym, hb_magic)           # clear any stale marker first
    broker.cancel_all_pending(sym, cfg.MAGIC, cfg.HEARTBEAT_COMMENT)  # legacy-magic marker
    tk = broker.place_heartbeat(sym, cfg.HEARTBEAT_PRICE, hb_magic, cfg.HEARTBEAT_COMMENT)
    if tk:
        log(f"heartbeat: {sym} buy-limit @ {cfg.HEARTBEAT_PRICE} placed (#{tk}) "
            f"— visible marker that the bot is alive")
    else:
        log(f"heartbeat: FAILED to place {sym} marker (broker may reject a price this far "
            f"from market) — bot continues without it")


def _cancel_heartbeat():
    """Remove the heartbeat marker so its absence means the bot is stopped."""
    if not getattr(cfg, "HEARTBEAT_ENABLED", False):
        return
    hb_magic = getattr(cfg, "HEARTBEAT_MAGIC", cfg.MAGIC)
    n = broker.cancel_all_pending(cfg.HEARTBEAT_SYMBOL, hb_magic)
    if n:
        log(f"heartbeat: cancelled {n} {cfg.HEARTBEAT_SYMBOL} marker order(s) — bot stopped")


def _resolve_symbols() -> list:
    """Subscribe each configured symbol, trying broker-name aliases; skip (with a
    loud warning) anything the broker doesn't offer instead of aborting."""
    out = []
    aliases = getattr(cfg, "SYMBOL_ALIASES", {})
    for s in cfg.SYMBOLS:
        if broker.ensure_symbol(s):
            out.append(s); continue
        resolved = None
        for alt in aliases.get(s, []):
            if broker.ensure_symbol(alt):
                resolved = alt; break
        if resolved:
            log(f"symbol {s} not offered — using broker name {resolved}")
            out.append(resolved)
        else:
            log(f"⚠️ symbol {s} NOT offered by this broker (no alias worked) — SKIPPED")
    return out


def main():
    if not broker.connect():
        return
    symbols = _resolve_symbols()
    if not symbols:
        log("no tradeable symbols — aborting")
        return

    if cfg.STRATEGY in ("ema_dol", "gradient"):
        tfs = getattr(cfg, "TIMEFRAMES", ["M15"])
        log(f"bot starting · STRATEGY={cfg.STRATEGY} · {len(symbols)} symbols · "
            f"TFs {'+'.join(tfs)} ({len(symbols)*len(tfs)} engines) · "
            f"risk {cfg.RISK_PCT}%/trade · exit={getattr(cfg,'EXIT_MODE','dol')} · "
            f"RSI {getattr(cfg,'RSI_LONG_MIN','-')}/{getattr(cfg,'RSI_SHORT_MAX','-')} · "
            f"daily-stop {cfg.DAILY_STOP_PCT}% · overall-stop {getattr(cfg,'MAX_OVERALL_DD_PCT','-')}% · "
            f"trade-cap {cfg.MAX_TRADES_PER_DAY}/day · news={cfg.NEWS_FILTER} · loop {cfg.LOOP_SEC}s")
    else:
        log(f"bot starting · STRATEGY=sweep_ifvg · {len(cfg.SYMBOLS)} symbols {cfg.SYMBOLS} · "
            f"risk {cfg.RISK_PCT}% · RR {cfg.RR} · news_filter={cfg.NEWS_FILTER} "
            f"(blocked H4 windows in ±2y: {len(NEWS_WINDOWS)}) · loop {cfg.LOOP_SEC}s")

    _place_heartbeat()             # visible 'bot is running' marker

    state = st.load(cfg.STATE_FILE)
    # If the persisted state belongs to a DIFFERENT account, discard it — a stale
    # balance/day baseline from another account would false-trip the DD guards.
    _acct = mt5.account_info()
    _login = _acct.login if _acct else None
    if _login is not None and state.get("account") != _login:
        if state.get("account") is not None:
            log(f"account changed #{state.get('account')} → #{_login}: resetting per-account state")
        else:
            log(f"anchoring state to account #{_login}")
        state = st.new(_login)
        st.save(cfg.STATE_FILE, state)

    anchor = symbols[0]             # for daily-window + market-open check

    try:
        while True:
            try:
                if not broker.is_market_open(anchor):
                    time.sleep(cfg.LOOP_SEC * 4)
                    continue

                _update_daily(state, anchor)
                entries_ok = _overall_ok(state)          # FTMO -10% overall guard

                for symbol in symbols:
                    if not broker.is_market_open(symbol):
                        continue
                    _reap_expired_retests(symbol, state)
                    _promote_filled_retests(symbol, state)
                    manager.manage(symbol, state, cfg, log)   # ALWAYS manage exits (all TFs)
                    if not entries_ok:
                        continue                              # halted — no new entries
                    if cfg.STRATEGY in ("ema_dol", "gradient"):
                        for tf in getattr(cfg, "TIMEFRAMES", ["M15"]):
                            _try_enter_ema_dol(symbol, state, tf)
                    else:
                        _try_enter(symbol, state)

                st.trim_traded_sweeps(state)
                st.save(cfg.STATE_FILE, state)

            except Exception as e:
                log(f"LOOP ERROR: {e!r}")

            time.sleep(cfg.LOOP_SEC)
    except KeyboardInterrupt:
        log("interrupted — saving state and shutting down")
        st.save(cfg.STATE_FILE, state)
    finally:
        _cancel_heartbeat()        # remove marker so its absence = bot stopped
        broker.disconnect()


if __name__ == "__main__":
    main()
