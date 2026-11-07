"""
Pure detection logic for the Asia-sweep + London 1M-CISD strategy.

All functions take M1 bars indexed in BROKER time plus the broker's UTC
offset, convert to UTC+1 ("session time"), and work there:

  Asia range :  00:00-07:00 UTC+1   (range high/low)
  London     :  07:00-10:00 UTC+1   (sweep -> CISD -> entry, all inside)

CISD (change in state of delivery), bearish case after an Asia-high sweep:
  • manipulation high = highest high from the sweep bar onward
  • delivery series   = the last run of consecutive UP-CLOSE candles at or
    before the bar that made that high
  • CISD level        = OPEN of the first candle in that series
  • confirmed         = an M1 candle CLOSES below the level
  • entry             = the level itself (retrace entry, "off the level")
  • stop              = the manipulation (swing) high
Bullish after an Asia-low sweep is the exact mirror.
"""
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional
import numpy as np
import pandas as pd

from asia_bot import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# Time handling
# ─────────────────────────────────────────────────────────────────────────────

def broker_offset_hours(tick_unix: float) -> int:
    """Broker-vs-UTC offset in whole hours, from a live tick timestamp.
    MT5 tick.time is broker wall-clock encoded as if it were UTC unix."""
    now_utc = datetime.now(timezone.utc).timestamp()
    return round((tick_unix - now_utc) / 3600)


def utc1_index(df: pd.DataFrame, offset_h: int) -> pd.DatetimeIndex:
    """Convert a broker-time index to UTC+1 session time."""
    return df.index - pd.Timedelta(hours=offset_h) + pd.Timedelta(hours=1)


def session_bounds_utc1(session_date) -> dict:
    """Session window timestamps (UTC+1, naive) for a given calendar date."""
    d = pd.Timestamp(session_date).normalize()
    return {
        "asia_start":   d + pd.Timedelta(hours=cfg.ASIA_START_H),
        "asia_end":     d + pd.Timedelta(hours=cfg.ASIA_END_H),
        "london_start": d + pd.Timedelta(hours=cfg.LONDON_START_H),
        "london_end":   d + pd.Timedelta(hours=cfg.LONDON_END_H),
    }


def utc1_to_broker(ts_utc1: pd.Timestamp, offset_h: int) -> pd.Timestamp:
    return ts_utc1 - pd.Timedelta(hours=1) + pd.Timedelta(hours=offset_h)


# ─────────────────────────────────────────────────────────────────────────────
# Setup dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DaySetup:
    symbol: str
    session_date: str            # UTC+1 calendar date
    asia_high: float
    asia_low: float
    stage: str                   # 'no_asia' | 'no_sweep' | 'swept' | 'confirmed'
    direction: Optional[str] = None      # 'sell' / 'buy'
    sweep_time: Optional[pd.Timestamp] = None    # UTC+1
    sweep_extreme: float = 0.0           # manipulation high/low so far
    cisd_level: float = 0.0
    confirm_time: Optional[pd.Timestamp] = None  # UTC+1
    confirm_close: float = 0.0
    # entry: FVG near-edge if the displacement produced one, else the CISD level
    entry_price: float = 0.0
    entry_source: str = "cisd"           # 'fvg' | 'cisd'
    fvg_top: float = 0.0                  # 0 if no displacement FVG
    fvg_bottom: float = 0.0
    fvg_i: int = -1                       # form index (3rd candle) of the FVG
    # plot support (indices into the day's UTC+1-converted frame)
    sweep_i: int = -1
    confirm_i: int = -1
    series_start_i: int = -1


# ─────────────────────────────────────────────────────────────────────────────
# Core detection
# ─────────────────────────────────────────────────────────────────────────────

def _delivery_series_start(o, c, end_i, direction, max_walkback, min_len, lo_limit=None):
    """
    First index of the consecutive delivery run that made the swing:
      bearish CISD → the last run of UP-close candles at/before end_i
      bullish CISD → the last run of DOWN-close candles at/before end_i
    The run must be at least `min_len` candles (a valid CISD needs >= 2 up/down
    candles). Returns the run's start index, or None if there is no qualifying
    run of that length within the walkback. The CISD level is o[start].
    """
    up = direction == "sell"        # bearish CISD reverses an UP delivery
    lo_bound = max(0, end_i - max_walkback)
    if lo_limit is not None:        # leg may not start before this bar (the sweep)
        lo_bound = max(lo_bound, lo_limit)

    # Strict mode: the run must END on end_i (the bar that made the extreme) —
    # the last 2+ candles are the ones that CREATE the London high/low. No
    # walking back to an earlier run.
    if getattr(cfg, "CISD_LEG_MAKES_EXTREME", False):
        if (c[end_i] > o[end_i]) != up:
            return None                       # extreme bar isn't a delivery candle
        start = end_i
        while start - 1 >= lo_bound and (c[start - 1] > o[start - 1]) == up:
            start -= 1
        return start if (end_i - start + 1) >= min_len else None

    j = end_i
    # Scan back through the delivery leg, skipping opposite candles between runs,
    # and return the first run that reaches `min_len` consecutive candles.
    while j >= lo_bound:
        if (c[j] > o[j]) != up:     # not a delivery-direction candle → skip
            j -= 1
            continue
        run_end = j
        start = j
        while start - 1 >= lo_bound and (c[start - 1] > o[start - 1]) == up:
            start -= 1
        if run_end - start + 1 >= min_len:
            return start
        j = start - 1               # run too short → keep looking further back
    return None


def _displacement_fvg(h, l, lo_i, hi_i, direction, min_width):
    """
    Find the FVG left by the CISD displacement, scanning 3-candle windows whose
    3rd candle is in [lo_i, hi_i]. Returns (top, bottom, form_idx) of the most
    recent aligned FVG, or None.

      buy  (bullish FVG): low[j] > high[j-2] → gap [high[j-2] .. low[j]];
                          entry edge = FVG HIGH = low[j]  (top)
      sell (bearish FVG): high[j] < low[j-2] → gap [high[j] .. low[j-2]];
                          entry edge = FVG LOW  = high[j] (bottom)

    "Most recent" (largest j) = the gap closest to the confirmation, i.e. the
    one the displacement actually produced.
    """
    best = None
    for j in range(max(lo_i, 2), hi_i + 1):
        if direction == "buy":
            if l[j] > h[j - 2]:
                width = l[j] - h[j - 2]
                if width >= min_width:
                    best = (float(l[j]), float(h[j - 2]), j)
        else:
            if h[j] < l[j - 2]:
                width = l[j - 2] - h[j]
                if width >= min_width:
                    best = (float(l[j - 2]), float(h[j]), j)
    return best


def analyze_day(df_m1: pd.DataFrame, symbol: str, offset_h: int,
                session_date=None) -> Optional[DaySetup]:
    """
    Run the full day logic on M1 bars (broker time). Uses ONLY closed bars for
    CISD confirmation; the caller passes df including the forming bar and we
    drop it here. Returns the day's state (whatever stage it reached).
    """
    if df_m1 is None or len(df_m1) < 30:
        return None
    df = df_m1.iloc[:-1]                          # closed bars only
    t1 = utc1_index(df, offset_h)

    if session_date is None:
        session_date = t1[-1].date()
    sb = session_bounds_utc1(session_date)

    o = df["open"].values; h = df["high"].values
    l = df["low"].values;  c = df["close"].values

    asia_mask = (t1 >= sb["asia_start"]) & (t1 < sb["asia_end"])
    if asia_mask.sum() < 30:                      # Asia session not (fully) there
        return DaySetup(symbol, str(session_date), 0.0, 0.0, "no_asia")
    asia_high = float(h[asia_mask].max())
    asia_low  = float(l[asia_mask].min())

    lon_mask = (t1 >= sb["london_start"]) & (t1 < sb["london_end"])
    lon_idx = np.where(lon_mask)[0]
    if not len(lon_idx):
        return DaySetup(symbol, str(session_date), asia_high, asia_low, "no_sweep")

    # First sweep of the Asia range inside London
    sweep_i = None; direction = None
    for i in lon_idx:
        if h[i] > asia_high:
            sweep_i = int(i); direction = "sell"; break
        if l[i] < asia_low:
            sweep_i = int(i); direction = "buy"; break
    if sweep_i is None:
        return DaySetup(symbol, str(session_date), asia_high, asia_low, "no_sweep")

    setup = DaySetup(symbol, str(session_date), asia_high, asia_low, "swept",
                     direction=direction, sweep_time=t1[sweep_i], sweep_i=sweep_i)

    # CISD scan on closed London bars after the sweep. The manipulation
    # extreme and the CISD level both update while the move extends.
    pip = cfg.PIP.get(symbol, 1e-4)
    fvg_min = cfg.FVG_MIN_WIDTH_PIPS * pip
    lon_start_i = int(lon_idx[0])
    lon_end_i = int(lon_idx[-1])
    ext_i = sweep_i
    for k in range(sweep_i + 1, lon_end_i + 1):
        if direction == "sell":
            if h[k] >= h[ext_i]:
                ext_i = k
            start = _delivery_series_start(o, c, ext_i, "sell",
                                           cfg.CISD_MAX_WALKBACK, cfg.CISD_MIN_SERIES,
                                           lo_limit=sweep_i if getattr(cfg, "CISD_LEG_AFTER_SWEEP", False) else None)
            if start is None:
                continue
            level = float(o[start])
            if c[k] < level:
                # London-extreme constraint: the manipulation HIGH the CISD
                # reverses must be the highest high of the London session so far
                # (i.e. the session high sits inside the CISD window, not before
                # it). Rejects a CISD that forms after price already peaked.
                sess_hi_i = lon_start_i + int(np.argmax(h[lon_start_i:k + 1]))
                if sess_hi_i < start:
                    continue
                # SL swing = highest high between the CISD formation (delivery
                # series start) and the confirmation candle.
                setup.sweep_extreme = float(h[start:k + 1].max())
                _finalize(setup, o, h, l, c, t1, "sell", start, k, ext_i, level, fvg_min)
                return setup
        else:
            if l[k] <= l[ext_i]:
                ext_i = k
            start = _delivery_series_start(o, c, ext_i, "buy",
                                           cfg.CISD_MAX_WALKBACK, cfg.CISD_MIN_SERIES,
                                           lo_limit=sweep_i if getattr(cfg, "CISD_LEG_AFTER_SWEEP", False) else None)
            if start is None:
                continue
            level = float(o[start])
            if c[k] > level:
                # London-extreme constraint (mirror): the manipulation LOW must
                # be the lowest low of the London session so far.
                sess_lo_i = lon_start_i + int(np.argmin(l[lon_start_i:k + 1]))
                if sess_lo_i < start:
                    continue
                # SL swing = lowest low between the CISD formation (delivery
                # series start) and the confirmation candle.
                setup.sweep_extreme = float(l[start:k + 1].min())
                _finalize(setup, o, h, l, c, t1, "buy", start, k, ext_i, level, fvg_min)
                return setup

    setup.sweep_extreme = float(h[sweep_i:lon_end_i + 1].max()) if direction == "sell" \
                          else float(l[sweep_i:lon_end_i + 1].min())
    return setup


def _finalize(setup, o, h, l, c, t1, direction, start, k, ext_i, level, fvg_min):
    """Fill in a confirmed setup: CISD level, confirmation, and the entry — the
    displacement FVG near-edge if one was produced, else the CISD level."""
    setup.stage = "confirmed"
    setup.cisd_level = level
    setup.confirm_time = t1[k]
    setup.confirm_close = float(c[k])
    setup.confirm_i = k
    setup.series_start_i = start

    # Look for an FVG left by the reversal leg (swing extreme → confirmation).
    fvg = _displacement_fvg(h, l, ext_i, k, direction, fvg_min)
    if fvg is not None:
        top, bottom, fi = fvg
        setup.fvg_top, setup.fvg_bottom, setup.fvg_i = top, bottom, fi
        # buy → FVG HIGH (top); sell → FVG LOW (bottom)
        setup.entry_price = top if direction == "buy" else bottom
        setup.entry_source = "fvg"
    else:
        setup.entry_price = level
        setup.entry_source = "cisd"


def _prices_from(entry: float, setup: DaySetup, symbol: str, live_spread: float) -> Optional[dict]:
    """Given an entry price, build SL/TP. SL = the swing (highest high / lowest
    low between the CISD formation and the confirmation) padded by SL_PAD_PIPS
    only — the live spread is NOT added, so the stop sits exactly at the swing
    +/- SL_PAD_PIPS. TP = RR from the entry. (`live_spread` is retained in the
    signature for caller compatibility but no longer affects the stop.)"""
    pip = cfg.PIP.get(symbol, 1e-4)
    min_sl = cfg.MIN_SL_PIPS * pip
    pad = cfg.SL_PAD_PIPS * pip

    if setup.direction == "sell":
        sl = setup.sweep_extreme + pad
        risk = sl - entry
        if risk < min_sl:
            sl = entry + min_sl
            risk = min_sl
        tp = entry - cfg.RR * risk
    else:
        sl = setup.sweep_extreme - pad
        risk = entry - sl
        if risk < min_sl:
            sl = entry - min_sl
            risk = min_sl
        tp = entry + cfg.RR * risk
    if risk <= 0:
        return None
    return {"entry": entry, "entry_level": entry, "sl": sl, "tp": tp, "risk": risk}


def build_order_prices(setup: DaySetup, symbol: str, live_spread: float,
                       entry_override: float = None) -> Optional[dict]:
    """Limit entry. entry_override forces the entry price (used for the too-big
    candle → limit at the CISD level); otherwise the FVG edge / CISD level."""
    entry = entry_override if entry_override is not None else (setup.entry_price or setup.cisd_level)
    return _prices_from(entry, setup, symbol, live_spread)


def build_market_prices(setup: DaySetup, symbol: str, live_spread: float) -> Optional[dict]:
    """Market-on-confirm entry: fill at the CISD confirmation close, SL at the
    swing, TP at 2R from that fill (anchored to the actual fill, not the level)."""
    return _prices_from(setup.confirm_close, setup, symbol, live_spread)


def build_asia_tp_prices(setup: DaySetup, symbol: str, live_spread: float,
                         entry: float = None) -> Optional[dict]:
    """Same entry + SL as usual, but TP = the OPPOSING Asia liquidity:
    Asia high swept (sell) → target the Asia LOW; Asia low swept (buy) → target
    the Asia HIGH. Realized RR is variable and returned as 'rr'. Returns None if
    the opposing level has already been taken (TP on the wrong side of entry)."""
    pip = cfg.PIP.get(symbol, 1e-4)
    min_sl = cfg.MIN_SL_PIPS * pip
    pad = cfg.SL_PAD_PIPS * pip
    if entry is None:
        entry = setup.confirm_close

    if setup.direction == "sell":
        sl = setup.sweep_extreme + pad + live_spread
        risk = sl - entry
        if risk < min_sl:
            sl = entry + min_sl
            risk = min_sl
        tp = setup.asia_low
        if tp >= entry:
            return None
    else:
        sl = setup.sweep_extreme - pad - live_spread
        risk = entry - sl
        if risk < min_sl:
            sl = entry - min_sl
            risk = min_sl
        tp = setup.asia_high
        if tp <= entry:
            return None
    if risk <= 0:
        return None
    return {"entry": entry, "entry_level": entry, "sl": sl, "tp": tp,
            "risk": risk, "rr": abs(entry - tp) / risk}
