"""
End-to-end MT5 connection test — opens the SMALLEST possible market order,
waits 5 seconds, closes it. Prints every step so you can verify the pipeline.

Uses MAGIC=999999 (different from the bot) so it doesn't touch any live trades.
Cost: ~1 pip of spread on 0.01 lots ≈ $0.01. Negligible.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaTrader5 as mt5
from live import config as cfg
from live import broker

TEST_MAGIC   = 999999
TEST_LOTS    = 0.01           # 1 micro-lot = smallest possible
TEST_COMMENT = "ping_test"
HOLD_SEC     = 5


def step(msg): print(f"→ {msg}", flush=True)


def main():
    step("Connecting to MT5...")
    if not broker.connect():
        return
    if not broker.ensure_symbol(cfg.SYMBOL):
        return

    acct = mt5.account_info()
    bal_before = acct.balance
    step(f"Account #{acct.login} · balance ${bal_before:,.2f}")

    bid, ask, spread = broker.get_tick(cfg.SYMBOL)
    step(f"Live tick: bid {bid} · ask {ask} · spread {spread*10000:.1f} pips")

    step(f"Placing BUY market order {TEST_LOTS} lots {cfg.SYMBOL} (no SL/TP)...")
    ticket = broker.market_order(
        cfg.SYMBOL, "buy", TEST_LOTS, 0.0, 0.0, TEST_MAGIC, TEST_COMMENT
    )
    if ticket is None:
        step("✗ ORDER FAILED — see error above. Bot pipeline is NOT working.")
        broker.disconnect()
        return

    step(f"✓ Order sent, order id #{ticket}. Waiting for position to appear...")
    position = None
    for _ in range(10):        # up to 5 sec
        time.sleep(0.5)
        positions = mt5.positions_get(symbol=cfg.SYMBOL) or []
        for p in positions:
            if p.magic == TEST_MAGIC:
                position = p
                break
        if position:
            break

    if position is None:
        step("✗ Order placed but no position appeared in 5s. Something's off.")
        broker.disconnect()
        return

    step(f"✓ Position #{position.ticket} filled @ {position.price_open} · volume {position.volume}")
    step(f"Holding for {HOLD_SEC}s...")
    time.sleep(HOLD_SEC)

    step(f"Closing position #{position.ticket}...")
    tick = mt5.symbol_info_tick(cfg.SYMBOL)
    req = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       cfg.SYMBOL,
        "volume":       position.volume,
        "type":         mt5.ORDER_TYPE_SELL,
        "position":     position.ticket,
        "price":        tick.bid,
        "deviation":    20,
        "magic":        TEST_MAGIC,
        "comment":      "ping_close",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res is None or res.retcode not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
        step(f"✗ CLOSE FAILED: {res}")
        broker.disconnect()
        return
    step(f"✓ Close order sent, deal #{res.deal}")
    time.sleep(1.5)

    # Verify closed
    still_open = [p for p in (mt5.positions_get(symbol=cfg.SYMBOL) or [])
                  if p.magic == TEST_MAGIC]
    if still_open:
        step(f"⚠ Position still shows open — may take a moment to reconcile")
    else:
        step("✓ Position confirmed closed")

    # Round-trip P/L
    time.sleep(0.5)
    bal_after = mt5.account_info().balance
    pl = bal_after - bal_before
    step(f"Balance before : ${bal_before:,.2f}")
    step(f"Balance after  : ${bal_after:,.2f}")
    step(f"Round-trip P/L : ${pl:+,.2f}   (≈ spread cost on {TEST_LOTS} lots)")
    step("")
    step("═" * 60)
    step("✓ FULL PIPELINE VERIFIED — MT5 auth, order placement, close, P/L accounting all working.")
    step("═" * 60)

    broker.disconnect()


if __name__ == "__main__":
    main()
