"""
Walk-forward parity test: replay live/detector_ema_dol.py over cached history
the way the 24/7 bot would see it, and compare the trades it generates against
the vectorized realistic backtest on the same data slice.

Poll emulation per bar t:
  • closed window = bars [.. t-1]
  • if bar t's range tags a waiting setup's edge, we poll at the FIRST TOUCH:
    bid = edge (price moves continuously, the bot's 15s loop catches the touch)
  • otherwise poll at the bar open (no tag → no action)

This is the proof that the live port implements the same strategy that
scored PF 1.91 / +240R in the realistic backtest.

Run from project root:  python -m live.parity_test
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import main as bt
import news_filter as nf
import realistic_ema_dol as re_dol
from live import config as cfg
from live.detector_ema_dol import detect_ema_dol_signals


SYMBOL = "EURUSD"
BARS   = 25_000          # ~10 months of M15
SPREAD   = re_dol.SPREAD_PRICE[SYMBOL]
SLIPPAGE = re_dol.SLIPPAGE_PRICE[SYMBOL]
WARMUP = 450


def _detect(window, already, bid):
    return detect_ema_dol_signals(
        window, already, bid=bid, ask=bid + SPREAD,
        fast=cfg.EMA_FAST, slow=cfg.EMA_SLOW,
        swing_n=cfg.SWING_LOOKBACK,
        fvg_window=cfg.FVG_LOOKUP_WINDOW,
        retest_window=cfg.RETEST_WINDOW_EMA,
        sl_buffer_frac=cfg.SL_BUFFER_FRAC,
        min_rr=cfg.MIN_RR, max_rr=cfg.MAX_RR,
        min_fvg_width=cfg.MIN_FVG_WIDTH.get(SYMBOL, 2e-4),
    )


def walk_forward(df):
    h = df["high"].values; l = df["low"].values
    o = df["open"].values
    idx = df.index
    news_set = nf.build_news_block_set(idx[0].date(), idx[-1].date())

    already = set()
    open_until = None
    entries = []

    for t in range(WARMUP, len(df)):
        now = idx[t]
        if open_until is not None and now >= open_until:
            open_until = None
        if open_until is not None:
            continue                       # trade open → bot won't enter (conc=1)

        window = df.iloc[max(0, t - 399):t]     # closed bars only (t-1 last)

        # DISCOVERY (side-effect free): probe at open/low/high to surface every
        # setup this bar could touch. The detector only READS `already`; keys
        # are consumed exclusively in the action pass below.
        candidates = {}
        for probe_bid in {o[t], l[t], h[t]}:
            for s in _detect(window, already, probe_bid):
                candidates.setdefault(str(s.cross_time), s)

        # ACTION: handle each candidate at its true FIRST-TOUCH price.
        # The live bot polls every 15s, so it sees the touch (bid ≈ edge)
        # before price can run beyond the drift tolerance — polling at the
        # bar extreme would wrongly discard those setups as 'missed'.
        entered = False
        for key in sorted(candidates):          # oldest cross first
            if open_until is not None:
                break
            s = candidates[key]
            if s.entry_mode in ("missed", "reject") and s.reason != "":
                # genuine closed-bar miss / structural reject — mode is
                # bid-independent only for closed-bar tags; re-check at touch
                pass
            if s.bias == "long":
                if l[t] > s.edge_price and o[t] > s.edge_price:
                    continue                    # bar never reached the edge
                touch_bid = min(o[t], s.edge_price)
            else:
                if h[t] < s.edge_price and o[t] < s.edge_price:
                    continue
                touch_bid = max(o[t], s.edge_price)

            sigs = [x for x in _detect(window, already, touch_bid)
                    if str(x.cross_time) == key]
            if not sigs:
                continue                        # not actionable at touch — wait
            s2 = sigs[0]
            if s2.entry_mode in ("missed", "reject"):
                already.add(key)
                continue
            if entered:
                continue                        # concurrency: consumed next bar
            already.add(key)
            if now.date() in news_set:
                continue
            d = "long" if s2.direction == "buy" else "short"
            fill = (touch_bid + SPREAD + SLIPPAGE) if d == "long" \
                   else (touch_bid - SLIPPAGE)
            risk = (fill - s2.sl_price) if d == "long" else (s2.sl_price - fill)
            if risk <= 0:
                continue
            rr = (s2.tp_price - fill) / risk if d == "long" \
                 else (fill - s2.tp_price) / risk
            oc, xt, pr = re_dol.sim_realistic(
                h, l, idx, t, d, fill, s2.sl_price, s2.tp_price, rr, SPREAD)
            entries.append({
                "entry_time": now, "direction": d, "entry": fill,
                "sl": s2.sl_price, "tp": s2.tp_price, "rr": rr,
                "outcome": oc, "pnl_r": pr,
            })
            open_until = pd.Timestamp(xt)
            entered = True
    return pd.DataFrame(entries)


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    df_full = bt.fetch_data(SYMBOL, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
    df = df_full.iloc[-BARS:]
    mt5.shutdown()

    print(f"-- PARITY TEST · {SYMBOL} · {len(df)} bars "
          f"({df.index[0]} -> {df.index[-1]}) --\n")

    bt_trades = re_dol.run_realistic(SYMBOL, realistic=True, df=df)
    lv_trades = walk_forward(df)

    def summarize(name, tr):
        cl = tr[tr["outcome"] != "open"] if len(tr) else tr
        if not len(cl):
            print(f"  {name:12} 0 trades"); return
        wr = (cl["pnl_r"] > 0).mean() * 100
        print(f"  {name:12} N={len(cl):>3}  WR={wr:.1f}%  R={cl['pnl_r'].sum():+.1f}")

    summarize("backtest", bt_trades)
    summarize("live-replay", lv_trades)

    matched = 0
    sl_diffs, tp_diffs = [], []
    if len(bt_trades) and len(lv_trades):
        used = set()
        for _, b in bt_trades.iterrows():
            bt_t = pd.Timestamp(b["entry_time"])
            for j, v in lv_trades.iterrows():
                if j in used or v["direction"] != b["direction"]:
                    continue
                if abs(v["entry"] - b["entry"]) < 3e-4 and \
                   abs((pd.Timestamp(v["entry_time"]) - bt_t).total_seconds()) < 12 * 3600:
                    matched += 1
                    used.add(j)
                    sl_diffs.append(abs(v["sl"] - b["sl"]) / 1e-4)
                    tp_diffs.append(abs(v["tp"] - b["tp"]) / 1e-4)
                    break
        print(f"\n  matched {matched}/{len(bt_trades)} backtest trades "
              f"({matched / len(bt_trades) * 100:.0f}%)"
              f" · extra live trades: {len(lv_trades) - matched}")
        if sl_diffs:
            print(f"  matched-trade SL diff: median {np.median(sl_diffs):.1f} pips, "
                  f"max {max(sl_diffs):.1f}")
            print(f"  matched-trade TP diff: median {np.median(tp_diffs):.1f} pips, "
                  f"max {max(tp_diffs):.1f}")


if __name__ == "__main__":
    main()
