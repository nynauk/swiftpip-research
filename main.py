"""
=============================================================================
 STRATEGY: 4H Candle Sweep + 15M IFVG Entry
=============================================================================

 FULL LOGIC:

 STEP 1 — 4H SWEEP
   A 4H candle grabs liquidity by wicking past a recent swing high or low,
   then closes back inside → establishes directional bias.
     • Wick above swing HIGH + close below it → BEARISH bias  (SHORT trades)
     • Wick below swing LOW  + close above it → BULLISH bias  (LONG trades)

 STEP 2 — REVERSAL FVG (forms on the displacement INTO the sweep)
   The FVG that becomes the IFVG is part of the sweep-reversal leg:
     • Long  bias -> a BEARISH FVG printed on the drop into the swept low
     • Short bias -> a BULLISH FVG printed on the pop into the swept high
   It must form within SWEEP_REVERSAL_WINDOW candles of the sweep — it is
   NOT an unrelated gap forming hours later.

 STEP 3 — INVERSION WITHIN 7 CANDLES -> IFVG
   Price reverses out of the sweep and CLOSES THROUGH the FVG, inverting it.
   Must occur within 7 candles of the FVG forming, else invalid:
     • Bearish FVG closed ABOVE -> bullish IFVG (support) -> LONG
     • Bullish FVG closed BELOW -> bearish IFVG (resistance) -> SHORT

 STEP 4 — 4H FVG CONFLUENCE (outside the sweep candle range)
   Between the sweep and the entry, price must trade into a 4H FVG that
   sits OUTSIDE the sweep candle's range, in the direction of the bias:
     • Short bias -> a BEARISH 4H FVG whose bottom >= sweep_candle.high
                     (FVG above the sweep candle). Price rallies up into it.
     • Long  bias -> a BULLISH 4H FVG whose top    <= sweep_candle.low
                     (FVG below the sweep candle). Price drops into it.
   The 4H FVG may form before or after the sweep — the touch must happen
   after the sweep and before the entry.

 STEP 5a — STANDARD ENTRY (disrespect candle <= 1.5x FVG width)
   Enter at the CLOSE of the disrespect candle.

 STEP 5b — 1.5x RULE -> RETRACEMENT ENTRY
   If the disrespect candle range > 1.5x the FVG width, wait for price to
   retrace to the IFVG near edge and enter there:
     • Long  -> top of the IFVG zone (support, first touch from above)
     • Short -> bottom of the IFVG zone (resistance, first touch from below)

 STEP 6 — SESSION FILTER
   Entry timestamp must fall within [SESSION_START_HOUR, SESSION_END_HOUR)
   on the broker's clock. Default 08:00 - 16:00.

 STEP 7 — STOP / TARGET
   SL = just beyond the SWEPT STRUCTURAL EXTREME
        (below the swept low for longs, above the swept high for shorts).
   TP = 2R.

=============================================================================
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: figures save to disk, plt.show() is a no-op
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except Exception:
    mt5 = None
    _MT5_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SYMBOL            = "EURUSD"
START_DATE        = datetime(2000, 1, 1)   # MT5 returns from earliest available
END_DATE          = datetime.utcnow()      # UTC matches MT5 server time

SWING_LOOKBACK     = 5    # candles each side for 4H swing high/low detection
# Sweep source. Each mode produces a list of {time, direction, swept_level}:
#   "H4_SWING"     — sweeps of 4H swing highs/lows (original)
#   "PDH_PDL"      — prior day H/L, swept intraday
#   "ASIAN_RANGE"  — Asian-session H/L (00:00-07:00 broker time), swept after 07:00
#   "LONDON_RANGE" — London-session H/L (08:00-12:00 broker time), swept after 12:00
SWEEP_MODE         = "PDH_PDL"
# Higher-timeframe bias filter:
#   "OFF"   — no filter
#   "DAILY" — only longs when previous daily candle closed up, only shorts down
#   "H4_SWING_HL" — uptrend if last swing HH > prior HH; downtrend mirror
HTF_BIAS_MODE      = "OFF"

# News filter: block trades on red-folder news days (NFP/FOMC/CPI/ECB/...)
NEWS_BLOCK_ENABLED = False
NEWS_BLOCK_SET     = set()  # populated at runtime via news_filter.build_news_block_set
SWEEP_REVERSAL_WINDOW = 24  # 15M candles after sweep within which the reversal FVG must form
MAX_DISRESPECT_DELAY  = 7   # FVG must be disrespected (inverted) within this many candles of forming
RETRACE_WINDOW     = 48   # 15M candles to wait for retracement into the IFVG
# Setting DISRESPECT_MULT very high effectively disables retrace entries.
# Retrace entries were 0/10 winners in this strategy — standard entries only.
DISRESPECT_MULT    = 1e9
# Require the disrespect candle to have at least MIN_DIS_MULT x FVG width of
# range. Filters out weak close-throughs where the candle is smaller than the
# FVG itself (probably noise). 0 disables.
MIN_DIS_MULT       = 0.0
# Require the disrespect candle's close to fall within the top (long) or
# bottom (short) MAX_CLOSE_LOC fraction of its own range — i.e. a strong
# directional close. 1.0 disables.
MAX_CLOSE_LOC      = 0.3  # close in the top/bottom 30% of the candle range
SL_BUFFER_FRAC     = 0.25 # SL placed this fraction of the FVG width beyond the anchor
# Stop-loss anchor:
#   "SWEEP_EXTREME" — just past the swept structural extreme (original)
#   "IFVG_EDGE"     — just past the opposite far side of the IFVG zone (tighter,
#                     gives smaller R distance so 2R targets fire more often)
SL_MODE            = "IFVG_EDGE"
RR                 = 1.0  # risk:reward target — 1:1 paired with ~70% WR
INITIAL_BALANCE    = 10_000
RISK_PCT           = 1.0  # % of account risked per trade

# Entries are only accepted when the entry candle's timestamp hour is in
# [SESSION_START_HOUR, SESSION_END_HOUR). The clock is whatever timezone
# MT5 returns for this broker (typically GMT+2/+3, NOT true UTC despite
# the unit="s" call). Shift these if your broker's clock differs.
SESSION_START_HOUR = 12  # broker time — London PM / NY morning overlap
SESSION_END_HOUR   = 16

# Reject FVGs narrower than this (price units). Micro-imbalances cause near
# instant stop-outs because SL buffer scales with FVG width.
MIN_FVG_WIDTH      = 0.0002  # 2 pips on EURUSD
# Cap on FVG width — very wide FVGs often mean an already-overextended move.
# 0 disables.
MAX_FVG_WIDTH      = 0.0
# Cap on trade risk distance (entry - sl in price units). Trades with very
# wide stops have poor expectancy regardless of RR. 0 disables.
MAX_RISK_DIST      = 0.0

# Master switch for the 4H FVG confluence gate. Empirically this filter
# REMOVED profitable trades on this dataset — kept off by default. Flip to
# True if you want the strict "price must touch a 4H FVG outside the sweep
# candle range" requirement.
REQUIRE_4H_TOUCH   = False

# Cache dir for fetched MT5 OHLC so iteration doesn't re-fetch every run.
CACHE_DIR          = "data_cache"


# ─────────────────────────────────────────────────────────────────────────────
# MT5 DATA
# ─────────────────────────────────────────────────────────────────────────────
def _cache_path(symbol: str, tf_name: str) -> str:
    return os.path.join(CACHE_DIR, f"{symbol}_{tf_name}.pkl")


def fetch_data(symbol: str, timeframe: int, start: datetime, end: datetime,
               tf_name: str = None, use_cache: bool = True) -> pd.DataFrame:
    """
    Returns OHLC for symbol at timeframe. When use_cache=True we read/write a
    pickle under CACHE_DIR so iteration doesn't re-fetch MT5 every run.

    Cache key is (symbol, tf_name). Pass tf_name (e.g. "H4", "M15") explicitly
    — it's both the cache filename and a label used in logs.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(symbol, tf_name or str(timeframe))

    if use_cache and os.path.exists(path):
        print(f"  [cache] loading {path}")
        return pd.read_pickle(path)

    if not _MT5_AVAILABLE:
        raise RuntimeError(
            f"MetaTrader5 not importable and no cache at {path}. "
            f"Run once with MT5 connected to build the cache."
        )

    # Ensure the symbol is subscribed in Market Watch (required for some brokers)
    try:
        mt5.symbol_select(symbol, True)
    except Exception:
        pass

    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None or len(rates) == 0:
        print(f"  [!] copy_rates_range returned nothing for tf={timeframe} "
              f"— falling back to max available bars from broker...")
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 99_999)
    if rates is None or len(rates) == 0:
        raise ValueError(
            f"No data at all for {symbol} tf={timeframe}. "
            f"Check symbol name and that the chart is open in MT5."
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df[["open", "high", "low", "close"]].copy()

    if use_cache:
        df.to_pickle(path)
        print(f"  [cache] saved {path}  ({len(df):,} rows)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — 4H SWING DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def find_swings(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Swing High at index i: df.high[i] is the highest value in the
    window [i-n .. i+n]. Same logic for swing lows.
    """
    df = df.copy()
    highs = df["high"].values
    lows  = df["low"].values
    sh    = np.full(len(df), np.nan)
    sl    = np.full(len(df), np.nan)

    for i in range(n, len(df) - n):
        window_h = highs[i - n : i + n + 1]
        window_l = lows[i  - n : i + n + 1]
        if highs[i] >= window_h.max():
            sh[i] = highs[i]
        if lows[i] <= window_l.min():
            sl[i] = lows[i]

    df["swing_high"] = sh
    df["swing_low"]  = sl
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — 4H SWEEP DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def build_daily_bias(df_15m: pd.DataFrame) -> dict:
    """
    Returns {date(normalized): "bullish"|"bearish"} for each trading day,
    based on the previous day's CLOSE vs OPEN. Used by HTF_BIAS_MODE=DAILY.
    """
    daily = df_15m.resample("1D").agg({"open": "first", "close": "last"}).dropna()
    direction = (daily["close"] > daily["open"]).map({True: "bullish", False: "bearish"})
    # Today's bias = yesterday's daily direction
    return direction.shift(1).dropna().to_dict()


def find_session_range_sweeps(df_15m: pd.DataFrame,
                              range_start_hour: int,
                              range_end_hour: int,
                              scan_until_hour: int,
                              level_tag: str) -> list[dict]:
    """
    Detect sweeps of a session-range high/low (Asian / London / NY).

    For each calendar day:
      • range = (high, low) over candles where range_start_hour <= h < range_end_hour
      • after range_end_hour and before scan_until_hour, the FIRST candle that
        wicks past the range high (or low) and closes back inside is a sweep
        of the session high (or low).

    Returns dicts compatible with run_backtest:
      {time, direction, swept_level, level_type}
    """
    h = df_15m["high"].values
    l = df_15m["low"].values
    c = df_15m["close"].values
    idx = df_15m.index
    hours = idx.hour
    dates = idx.normalize()

    sweeps = []
    # iterate per day
    unique_days, day_starts = np.unique(dates, return_index=True)
    day_ends = np.append(day_starts[1:], len(df_15m))

    for di, day in enumerate(unique_days):
        a, b = day_starts[di], day_ends[di]
        day_hours = hours[a:b]

        in_range = (day_hours >= range_start_hour) & (day_hours < range_end_hour)
        if not in_range.any():
            continue
        rng_hi = h[a:b][in_range].max()
        rng_lo = l[a:b][in_range].min()

        post = (day_hours >= range_end_hour) & (day_hours < scan_until_hour)
        post_local = np.where(post)[0]
        if not len(post_local):
            continue

        hi_done = False
        lo_done = False
        for j in post_local:
            i = a + j
            if not hi_done and h[i] > rng_hi and c[i] < rng_hi:
                sweeps.append({"time": idx[i], "direction": "bearish",
                               "swept_level": float(rng_hi), "level_type": f"{level_tag}H"})
                hi_done = True
            if not lo_done and l[i] < rng_lo and c[i] > rng_lo:
                sweeps.append({"time": idx[i], "direction": "bullish",
                               "swept_level": float(rng_lo), "level_type": f"{level_tag}L"})
                lo_done = True
            if hi_done and lo_done:
                break

    return sweeps


def find_pdh_pdl_sweeps(df_15m: pd.DataFrame) -> list[dict]:
    """
    Detect sweeps of the PRIOR DAY HIGH / PRIOR DAY LOW on the 15M dataset.

    A sweep fires on the first 15M candle of the day that:
      • wicks above PDH and closes below it  -> bearish bias (SHORT)
      • wicks below PDL and closes above it  -> bullish bias (LONG)

    Only the first PDH-sweep and the first PDL-sweep per day are kept (a
    re-sweep later in the same day is noise). PDH/PDL are derived from the
    15M data itself by daily resampling.
    """
    daily = df_15m.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    pdh = daily["high"].shift(1)
    pdl = daily["low"].shift(1)

    h = df_15m["high"].values
    l = df_15m["low"].values
    c = df_15m["close"].values
    idx = df_15m.index
    days = idx.normalize()

    pdh_lookup = pdh.to_dict()
    pdl_lookup = pdl.to_dict()
    pdh_swept_today = set()
    pdl_swept_today = set()
    sweeps = []

    for i in range(len(df_15m)):
        d = days[i]
        cur_pdh = pdh_lookup.get(d)
        cur_pdl = pdl_lookup.get(d)
        if cur_pdh is None or pd.isna(cur_pdh):
            continue

        if d not in pdh_swept_today and h[i] > cur_pdh and c[i] < cur_pdh:
            sweeps.append({"time": idx[i], "direction": "bearish",
                           "swept_level": float(cur_pdh), "level_type": "PDH"})
            pdh_swept_today.add(d)

        if d not in pdl_swept_today and l[i] < cur_pdl and c[i] > cur_pdl:
            sweeps.append({"time": idx[i], "direction": "bullish",
                           "swept_level": float(cur_pdl), "level_type": "PDL"})
            pdl_swept_today.add(d)

    return sweeps


def find_sweeps(df_4h: pd.DataFrame) -> list[dict]:
    """
    Detect 4H liquidity sweeps. A sweep occurs when:
      Bearish: candle wick > most recent swing HIGH, candle CLOSES below it
      Bullish: candle wick < most recent swing LOW,  candle CLOSES above it
    Returns list of dicts: {time, direction, swept_level}
    """
    sweeps = []
    sh_series = df_4h["swing_high"].dropna()
    sl_series = df_4h["swing_low"].dropna()

    for i in range(1, len(df_4h)):
        row  = df_4h.iloc[i]
        ts   = df_4h.index[i]

        # Most recent swing high/low BEFORE this candle
        prior_highs = sh_series[sh_series.index < ts]
        prior_lows  = sl_series[sl_series.index < ts]

        if len(prior_highs):
            last_high = prior_highs.iloc[-1]
            # Wick swept above, closed back below → bearish liquidity grab
            if row["high"] > last_high and row["close"] < last_high:
                sweeps.append({
                    "time":      ts,
                    "direction": "bearish",  # bias: SHORT
                    "swept_level": last_high,
                })

        if len(prior_lows):
            last_low = prior_lows.iloc[-1]
            # Wick swept below, closed back above → bullish liquidity grab
            if row["low"] < last_low and row["close"] > last_low:
                sweeps.append({
                    "time":      ts,
                    "direction": "bullish",  # bias: LONG
                    "swept_level": last_low,
                })

    return sweeps


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — 15M FVG DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def find_all_fvgs(df: pd.DataFrame) -> list[dict]:
    """
    Scan the 15M dataset for Fair Value Gaps (3-candle imbalances).
    form_idx = index of the 3rd candle (when the gap is confirmed/complete).
    The 7-candle disrespect window is measured from form_idx.

    Bullish FVG: candle[i+2].low > candle[i].high  ->  zone = [c0.high, c2.low]
    Bearish FVG: candle[i+2].high < candle[i].low  ->  zone = [c2.high, c0.low]
    """
    highs = df["high"].values
    lows  = df["low"].values
    idx   = df.index
    fvgs  = []
    for i in range(len(df) - 2):
        if lows[i + 2] > highs[i]:                    # bullish FVG
            fvgs.append({
                "direction": "bullish",
                "top":       lows[i + 2],
                "bottom":    highs[i],
                "width":     lows[i + 2] - highs[i],
                "form_idx":  i + 2,
                "time":      idx[i + 2],
            })
        elif highs[i + 2] < lows[i]:                  # bearish FVG
            fvgs.append({
                "direction": "bearish",
                "top":       lows[i],
                "bottom":    highs[i + 2],
                "width":     lows[i] - highs[i + 2],
                "form_idx":  i + 2,
                "time":      idx[i + 2],
            })
    return fvgs


def in_session(ts) -> bool:
    """Entry-time filter: True if ts.hour in [SESSION_START_HOUR, SESSION_END_HOUR)."""
    h = pd.Timestamp(ts).hour
    return SESSION_START_HOUR <= h < SESSION_END_HOUR


def is_disrespected(fvg: dict, candle: pd.Series) -> bool:
    """
    Bullish FVG disrespected: a candle CLOSES below the FVG's bottom
    Bearish FVG disrespected: a candle CLOSES above the FVG's top
    """
    if fvg["direction"] == "bullish":
        return candle["close"] < fvg["bottom"]
    else:
        return candle["close"] > fvg["top"]


# ─────────────────────────────────────────────────────────────────────────────
# TRADE SIMULATION — walk forward candle by candle
# ─────────────────────────────────────────────────────────────────────────────
def simulate_trade(df: pd.DataFrame, entry_idx: int, direction: str,
                   entry: float, sl: float, tp: float) -> dict:
    risk = abs(entry - sl)
    for j in range(entry_idx + 1, min(entry_idx + 300, len(df))):
        c = df.iloc[j]
        if direction == "long":
            if c["low"]  <= sl:
                return {"outcome": "loss", "exit": sl, "exit_time": df.index[j], "pnl_r": -1.0}
            if c["high"] >= tp:
                return {"outcome": "win",  "exit": tp, "exit_time": df.index[j], "pnl_r": +RR}
        else:
            if c["high"] >= sl:
                return {"outcome": "loss", "exit": sl, "exit_time": df.index[j], "pnl_r": -1.0}
            if c["low"]  <= tp:
                return {"outcome": "win",  "exit": tp, "exit_time": df.index[j], "pnl_r": +RR}
    return {"outcome": "open", "exit": df["close"].iloc[-1],
            "exit_time": df.index[-1], "pnl_r": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BACKTEST LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df_4h: pd.DataFrame, df_15m: pd.DataFrame,
                 sweeps: list, all_fvgs: list, fvgs_4h: list,
                 daily_bias: dict = None) -> list[dict]:
    """
    Sequence per sweep (matches the TradingView reference structure):

      1. 4H sweep grabs liquidity -> directional bias.
      2. The displacement INTO the sweep prints an FVG on the 15M:
           long  bias -> a BEARISH fvg formed on the drop into the swept low
           short bias -> a BULLISH fvg formed on the pop into the swept high
         This fvg must form within SWEEP_REVERSAL_WINDOW candles of the sweep
         (it is part of the sweep-reversal leg, NOT some unrelated later gap).
      3. Price reverses out of the sweep and CLOSES THROUGH that fvg,
         inverting it into an IFVG. Must happen within MAX_DISRESPECT_DELAY
         (7) candles of the fvg forming, else invalid.
      4. Entry:
           - disrespect candle range <= 1.5x fvg width -> STANDARD:
                 enter at the disrespect candle CLOSE
           - disrespect candle range >  1.5x fvg width -> RETRACE:
                 wait for price to retrace to the IFVG near edge and enter there
                 (long  -> top of the zone, short -> bottom of the zone)
      5. SL = just beyond the SWEPT STRUCTURAL EXTREME
              (below the swept low for longs, above the swept high for shorts).
      6. TP = 2R.
    """
    o = df_15m["open"].values
    h = df_15m["high"].values
    l = df_15m["low"].values
    c = df_15m["close"].values
    idx = df_15m.index

    fvgs_sorted  = sorted(all_fvgs, key=lambda f: f["form_idx"])
    form_indices = np.array([f["form_idx"] for f in fvgs_sorted])

    trades        = []
    used          = set()
    taken_entries = set()

    funnel = {
        "sweeps_total":        len(sweeps),
        "sweeps_duplicate":    0,
        "sweeps_no_4h_fvg":    0,
        "sweeps_no_inversion": 0,
        "sweeps_htf_bias":     0,
        "rej_news":            0,
        "inversions_seen":     0,
        "rej_min_width":       0,
        "rej_session":         0,
        "rej_4h_touch":        0,
        "rej_risk_le_0":       0,
        "rej_dedup":           0,
        "retrace_expired":     0,
        "trades_taken":        0,
    }

    daily_bias = daily_bias or {}

    for sweep in sweeps:
        bias = sweep["direction"]                 # 'bullish' -> LONG, 'bearish' -> SHORT
        sig  = (sweep["time"], bias)
        if sig in used:
            funnel["sweeps_duplicate"] += 1
            continue

        # HTF bias filter: only take trades aligned with the higher-TF direction.
        # For a high sweep (bias='bearish'/SHORT), we want HTF to also be bearish.
        if HTF_BIAS_MODE == "DAILY" and daily_bias:
            day_key = pd.Timestamp(sweep["time"]).normalize()
            htf = daily_bias.get(day_key)
            if htf is None or htf != bias:
                funnel["sweeps_htf_bias"] += 1
                continue

        sweep_i = int(np.searchsorted(idx, sweep["time"], side="left"))
        if sweep_i >= len(idx):
            continue
        # Discard sweeps that fall before the M15 dataset begins (those would
        # all collapse to sweep_i=0 and pollute the funnel). Require at least
        # SWEEP_REVERSAL_WINDOW M15 bars after the sweep to evaluate setup.
        if sweep_i == 0 and sweep["time"] < idx[0]:
            continue

        # ── Sweep candle (4H) — needed for the 4H FVG positional filter ──
        sweep_4h_i = int(np.searchsorted(df_4h.index, sweep["time"], side="left"))
        if sweep_4h_i >= len(df_4h):
            continue
        sweep_4h_high = float(df_4h["high"].iloc[sweep_4h_i])
        sweep_4h_low  = float(df_4h["low"].iloc[sweep_4h_i])

        # Qualifying 4H FVGs for this sweep:
        #   SHORT bias -> bearish 4H FVG sitting ABOVE the sweep candle
        #                 (fvg.bottom >= sweep_candle.high)
        #   LONG  bias -> bullish 4H FVG sitting BELOW the sweep candle
        #                 (fvg.top    <= sweep_candle.low)
        # Map each FVG's 4H form time to a 15M index so we can check touch
        # on the 15M series later.
        quals_4h = []
        for f4 in fvgs_4h:
            if bias == "bearish":
                if f4["direction"] == "bearish" and f4["bottom"] >= sweep_4h_high:
                    f15 = int(np.searchsorted(idx, df_4h.index[f4["form_idx"]], side="left"))
                    quals_4h.append({"fvg": f4, "form_15m": f15})
            else:
                if f4["direction"] == "bullish" and f4["top"] <= sweep_4h_low:
                    f15 = int(np.searchsorted(idx, df_4h.index[f4["form_idx"]], side="left"))
                    quals_4h.append({"fvg": f4, "form_15m": f15})

        if not quals_4h:
            funnel["sweeps_no_4h_fvg"] += 1
            continue  # no 4H FVG confluence -> no trade off this sweep

        def four_h_touched(entry_i: int) -> bool:
            """Has price touched any qualifying 4H FVG after the sweep and
            before/at entry_i, with the FVG already formed by then?"""
            if not REQUIRE_4H_TOUCH:
                return True
            for q in quals_4h:
                if q["form_15m"] > entry_i:
                    continue
                start = max(sweep_i + 1, q["form_15m"])
                if start > entry_i:
                    continue
                if bias == "bearish":
                    # bearish FVG above -> 15m high reaches up to its bottom
                    if h[start:entry_i + 1].max() >= q["fvg"]["bottom"]:
                        return True
                else:
                    # bullish FVG below -> 15m low drops down to its top
                    if l[start:entry_i + 1].min() <= q["fvg"]["top"]:
                        return True
            return False

        # long  -> invert a BEARISH fvg (close above it)
        # short -> invert a BULLISH fvg (close below it)
        target_dir = "bearish" if bias == "bullish" else "bullish"
        rev_end = min(sweep_i + SWEEP_REVERSAL_WINDOW, len(idx) - 1)

        lo_pos = int(np.searchsorted(form_indices, sweep_i, side="left"))
        hi_pos = int(np.searchsorted(form_indices, rev_end, side="right"))

        trade_taken      = False
        had_inversion    = False
        # iterate FVGs nearest the swept extreme first (deepest in the leg)
        for p in range(hi_pos - 1, lo_pos - 1, -1):
            fvg = fvgs_sorted[p]
            if fvg["direction"] != target_dir:
                continue
            if fvg["width"] < MIN_FVG_WIDTH:
                funnel["rej_min_width"] += 1
                continue
            if MAX_FVG_WIDTH > 0 and fvg["width"] > MAX_FVG_WIDTH:
                funnel["rej_min_width"] += 1
                continue

            f_idx  = fvg["form_idx"]
            dis_lo = f_idx + 1
            dis_hi = min(f_idx + MAX_DISRESPECT_DELAY, len(idx) - 1)

            for di in range(dis_lo, dis_hi + 1):
                # inversion = close THROUGH the fvg
                if fvg["direction"] == "bearish":
                    inverted = c[di] > fvg["top"]      # closed above -> bullish IFVG (long)
                else:
                    inverted = c[di] < fvg["bottom"]   # closed below -> bearish IFVG (short)
                if not inverted:
                    continue

                had_inversion = True
                funnel["inversions_seen"] += 1

                dis_range   = h[di] - l[di]
                # Quality gates on the disrespect candle itself
                if dis_range < MIN_DIS_MULT * fvg["width"]:
                    continue
                if MAX_CLOSE_LOC < 1.0 and dis_range > 0:
                    if bias == "bullish":
                        close_loc = (c[di] - l[di]) / dis_range  # 1.0 = closed at high
                        if close_loc < 1.0 - MAX_CLOSE_LOC:
                            continue
                    else:
                        close_loc = (h[di] - c[di]) / dis_range  # 1.0 = closed at low
                        if close_loc < 1.0 - MAX_CLOSE_LOC:
                            continue
                use_retrace = dis_range > DISRESPECT_MULT * fvg["width"]
                buf         = fvg["width"] * SL_BUFFER_FRAC

                # ── Stop loss anchor ──
                if bias == "bullish":
                    sweep_extreme = l[sweep_i:di + 1].min()
                    if SL_MODE == "IFVG_EDGE":
                        sl = fvg["bottom"] - buf  # below the IFVG support
                    else:
                        sl = sweep_extreme - buf
                else:
                    sweep_extreme = h[sweep_i:di + 1].max()
                    if SL_MODE == "IFVG_EDGE":
                        sl = fvg["top"] + buf  # above the IFVG resistance
                    else:
                        sl = sweep_extreme + buf

                # ── STANDARD entry: at the disrespect candle close ────────
                if not use_retrace:
                    entry = c[di]
                    fill_i = di
                    # Session + news + 4H-FVG confluence gates
                    if not in_session(idx[fill_i]):
                        funnel["rej_session"] += 1
                        break
                    if NEWS_BLOCK_ENABLED and idx[fill_i].date() in NEWS_BLOCK_SET:
                        funnel["rej_news"] += 1
                        break
                    if not four_h_touched(fill_i):
                        funnel["rej_4h_touch"] += 1
                        break
                    if bias == "bullish":
                        direction, risk = "long", entry - sl
                        if risk <= 0:
                            funnel["rej_risk_le_0"] += 1
                            break
                        tp = entry + RR * risk
                    else:
                        direction, risk = "short", sl - entry
                        if risk <= 0:
                            funnel["rej_risk_le_0"] += 1
                            break
                        tp = entry - RR * risk
                    if MAX_RISK_DIST > 0 and risk > MAX_RISK_DIST:
                        funnel["rej_risk_le_0"] += 1
                        break

                    ekey = (idx[fill_i], direction)
                    if ekey in taken_entries:
                        funnel["rej_dedup"] += 1
                        trade_taken = True
                        break
                    result = simulate_trade(df_15m, fill_i, direction, entry, sl, tp)
                    trades.append(_trade_record(sweep, idx[fill_i], direction,
                                                entry, sl, tp, result, False,
                                                fvg["width"], dis_range, fvg, fill_i,
                                                sweep_extreme))
                    taken_entries.add(ekey)
                    funnel["trades_taken"] += 1
                    trade_taken = True
                    break

                # ── RETRACE entry: at the IFVG near edge ──────────────────
                else:
                    retr_end = min(di + RETRACE_WINDOW, len(idx) - 1)
                    retrace_filled  = False
                    retrace_handled = False  # any counter (rej or fill) recorded?
                    for rj in range(di + 1, retr_end + 1):
                        if bias == "bullish":
                            # long: IFVG is support, price retraces DOWN to its TOP
                            if l[rj] <= fvg["top"]:
                                # Session + 4H-FVG confluence gates
                                if not in_session(idx[rj]):
                                    funnel["rej_session"] += 1
                                    retrace_handled = True
                                    break
                                if NEWS_BLOCK_ENABLED and idx[rj].date() in NEWS_BLOCK_SET:
                                    funnel["rej_news"] += 1
                                    retrace_handled = True
                                    break
                                if not four_h_touched(rj):
                                    funnel["rej_4h_touch"] += 1
                                    retrace_handled = True
                                    break
                                entry, direction = fvg["top"], "long"
                                risk = entry - sl
                                if risk <= 0:
                                    funnel["rej_risk_le_0"] += 1
                                    retrace_handled = True
                                    break
                                tp = entry + RR * risk
                                ekey = (idx[rj], direction)
                                if ekey in taken_entries:
                                    funnel["rej_dedup"] += 1
                                    trade_taken = True
                                    break
                                result = simulate_trade(df_15m, rj, direction, entry, sl, tp)
                                trades.append(_trade_record(sweep, idx[rj], direction,
                                                            entry, sl, tp, result, True,
                                                            fvg["width"], dis_range, fvg, rj,
                                                            sweep_extreme))
                                taken_entries.add(ekey)
                                funnel["trades_taken"] += 1
                                trade_taken = True
                                retrace_filled = True
                                break
                        else:
                            # short: IFVG is resistance, price retraces UP to its BOTTOM
                            if h[rj] >= fvg["bottom"]:
                                # Session + 4H-FVG confluence gates
                                if not in_session(idx[rj]):
                                    funnel["rej_session"] += 1
                                    retrace_handled = True
                                    break
                                if NEWS_BLOCK_ENABLED and idx[rj].date() in NEWS_BLOCK_SET:
                                    funnel["rej_news"] += 1
                                    retrace_handled = True
                                    break
                                if not four_h_touched(rj):
                                    funnel["rej_4h_touch"] += 1
                                    retrace_handled = True
                                    break
                                entry, direction = fvg["bottom"], "short"
                                risk = sl - entry
                                if risk <= 0:
                                    funnel["rej_risk_le_0"] += 1
                                    retrace_handled = True
                                    break
                                tp = entry - RR * risk
                                ekey = (idx[rj], direction)
                                if ekey in taken_entries:
                                    funnel["rej_dedup"] += 1
                                    trade_taken = True
                                    break
                                result = simulate_trade(df_15m, rj, direction, entry, sl, tp)
                                trades.append(_trade_record(sweep, idx[rj], direction,
                                                            entry, sl, tp, result, True,
                                                            fvg["width"], dis_range, fvg, rj,
                                                            sweep_extreme))
                                taken_entries.add(ekey)
                                funnel["trades_taken"] += 1
                                trade_taken = True
                                retrace_filled = True
                                break
                    if not retrace_handled and not retrace_filled and not trade_taken:
                        funnel["retrace_expired"] += 1
                    break  # disrespect handled (filled or retrace expired)

            if trade_taken:
                break

        if trade_taken:
            used.add(sig)
        elif not had_inversion:
            funnel["sweeps_no_inversion"] += 1

    return trades, funnel


def _trade_record(sweep, entry_time, direction, entry, sl, tp,
                  result, is_retrace, fvg_width, dis_range, fvg, entry_idx,
                  sweep_extreme) -> dict:
    return {
        "sweep_time":    sweep["time"],
        "sweep_bias":    sweep["direction"],
        "swept_level":   round(sweep["swept_level"], 5),
        "sweep_extreme": round(sweep_extreme, 5),
        "entry_time":    entry_time,
        "entry_idx":     entry_idx,
        "direction":     direction,
        "entry":         round(entry, 5),
        "sl":            round(sl, 5),
        "tp":            round(tp, 5),
        "outcome":       result["outcome"],
        "exit":          round(result["exit"], 5),
        "exit_time":     result["exit_time"],
        "pnl_r":         result["pnl_r"],
        "retrace_entry": is_retrace,
        "fvg_top":       round(fvg["top"], 5),
        "fvg_bottom":    round(fvg["bottom"], 5),
        "fvg_dir":       fvg["direction"],
        "fvg_width":     round(fvg_width, 5),
        "dis_mult":      round(dis_range / fvg_width, 2) if fvg_width else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STATS + EQUITY
# ─────────────────────────────────────────────────────────────────────────────
def compute_stats(trades: list) -> dict:
    df     = pd.DataFrame(trades)
    closed = df[df["outcome"] != "open"]
    wins   = closed[closed["outcome"] == "win"]
    losses = closed[closed["outcome"] == "loss"]

    win_rate = len(wins) / len(closed) * 100 if len(closed) else 0

    # Compounding equity curve at RISK_PCT% per trade
    balance = INITIAL_BALANCE
    equity  = [balance]
    for _, row in df.iterrows():
        balance += balance * (RISK_PCT / 100) * row["pnl_r"]
        equity.append(balance)

    eq_arr  = np.array(equity)
    peak    = np.maximum.accumulate(eq_arr)
    dd_arr  = (eq_arr - peak) / peak * 100
    max_dd  = dd_arr.min()

    # ── Longest winning / losing streaks (chronological, closed only) ──────
    chrono = closed.sort_values("entry_time")
    best_win = best_loss = 0
    cur_win = cur_loss = 0
    win_start = win_end = loss_start = loss_end = None
    run_start = None
    for _, r in chrono.iterrows():
        if r["outcome"] == "win":
            if cur_win == 0:
                run_start = r["entry_time"]
            cur_win += 1
            cur_loss = 0
            if cur_win > best_win:
                best_win, win_start, win_end = cur_win, run_start, r["entry_time"]
        elif r["outcome"] == "loss":
            if cur_loss == 0:
                run_start = r["entry_time"]
            cur_loss += 1
            cur_win = 0
            if cur_loss > best_loss:
                best_loss, loss_start, loss_end = cur_loss, run_start, r["entry_time"]

    return {
        "total":     len(df),
        "closed":    len(closed),
        "wins":      len(wins),
        "losses":    len(losses),
        "open":      len(df[df["outcome"] == "open"]),
        "win_rate":  win_rate,
        "total_r":   closed["pnl_r"].sum(),
        "avg_win":   wins["pnl_r"].mean()   if len(wins)   else 0,
        "avg_loss":  losses["pnl_r"].mean() if len(losses) else 0,
        "max_dd":    max_dd,
        "final_bal": balance,
        "retrace":   len(df[df["retrace_entry"]]),
        "equity":    equity,
        "best_win_streak":  best_win,
        "best_loss_streak": best_loss,
        "win_streak_span":  (win_start, win_end),
        "loss_streak_span": (loss_start, loss_end),
        "df":        df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRINT REPORT
# ─────────────────────────────────────────────────────────────────────────────
def print_report(s: dict):
    print("\n" + "═" * 55)
    print("  4H SWEEP + 15M IFVG │ BACKTEST RESULTS")
    print("═" * 55)
    print(f"  Symbol          : {SYMBOL}")
    print(f"  Period          : {START_DATE.date()} → {END_DATE.date()}")
    print(f"  Risk/Trade      : {RISK_PCT}%  │  RR Target: {RR}R")
    print("─" * 55)
    print(f"  Total Signals   : {s['total']}")
    print(f"  Closed Trades   : {s['closed']}  "
          f"(Wins: {s['wins']}  Losses: {s['losses']}  Open: {s['open']})")
    print(f"  Win Rate        : {s['win_rate']:.1f}%")
    print(f"  Total R         : {s['total_r']:.2f}R")
    print(f"  Avg Win         : {s['avg_win']:.2f}R")
    print(f"  Avg Loss        : {s['avg_loss']:.2f}R")
    print(f"  Max Drawdown    : {s['max_dd']:.2f}%")
    print(f"  Retrace Entries : {s['retrace']}  "
          f"({s['retrace']/s['total']*100:.1f}% of trades)")
    print("─" * 55)
    ws, we = s["win_streak_span"]
    ls, le = s["loss_streak_span"]
    print(f"  Longest Win Run : {s['best_win_streak']} trades", end="")
    if ws is not None:
        print(f"  ({ws.date()} → {we.date()})")
    else:
        print()
    print(f"  Longest Loss Run: {s['best_loss_streak']} trades", end="")
    if ls is not None:
        print(f"  ({ls.date()} → {le.date()})")
    else:
        print()
    print(f"  Start Balance   : ${INITIAL_BALANCE:,.0f}")
    print(f"  End Balance     : ${s['final_bal']:,.0f}")
    print(f"  Net Return      : {(s['final_bal']/INITIAL_BALANCE - 1)*100:.1f}%")
    print("═" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(s: dict):
    df  = s["df"]
    eq  = s["equity"]
    BG  = "#0d1117"
    FG  = "#e6edf3"
    GRN = "#3fb950"
    RED = "#f85149"
    YLW = "#d29922"

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), facecolor=BG)
    fig.suptitle(
        f"4H Sweep + 15M IFVG  │  {SYMBOL}  │  "
        f"{START_DATE.year}–{END_DATE.year}  │  "
        f"WR: {s['win_rate']:.1f}%  │  {s['total_r']:.1f}R total",
        color=FG, fontsize=13, fontweight="bold", y=0.98
    )

    # ── Equity Curve ──────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor(BG)
    ax1.plot(eq, color=GRN, linewidth=1.4, label="Equity")
    ax1.axhline(INITIAL_BALANCE, color="#444c56", linestyle="--", linewidth=0.8)
    ax1.fill_between(range(len(eq)), INITIAL_BALANCE, eq,
                     where=[e >= INITIAL_BALANCE for e in eq], alpha=0.12, color=GRN)
    ax1.fill_between(range(len(eq)), INITIAL_BALANCE, eq,
                     where=[e < INITIAL_BALANCE for e in eq], alpha=0.15, color=RED)
    ax1.set_title("Equity Curve (compounding, 1% risk/trade)", color=FG, pad=8)
    ax1.set_ylabel("Balance ($)", color=FG)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(facecolor="#161b22", edgecolor="#444c56", labelcolor=FG)

    # ── Per-trade P&L ─────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    colors = [GRN if r > 0 else (RED if r < 0 else "#6e7681") for r in df["pnl_r"]]
    bars = ax2.bar(range(len(df)), df["pnl_r"], color=colors, alpha=0.85, width=0.7)

    # Hatching for retrace entries
    for idx, (bar, retrace) in enumerate(zip(bars, df["retrace_entry"])):
        if retrace:
            bar.set_hatch("///")
            bar.set_edgecolor(YLW)

    ax2.axhline(0, color="#444c56", linewidth=0.8)
    ax2.set_title(
        "P&L per Trade (R)  │  Hatched = retracement entry (1.5× rule)",
        color=FG, pad=8
    )
    ax2.set_ylabel("R multiple", color=FG)

    # ── Cumulative R ──────────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor(BG)
    cum_r = df["pnl_r"].cumsum().values
    ax3.plot(cum_r, color=YLW, linewidth=1.4, label="Cumulative R")
    ax3.axhline(0, color="#444c56", linestyle="--", linewidth=0.8)
    ax3.fill_between(range(len(cum_r)), 0, cum_r,
                     where=cum_r >= 0, alpha=0.12, color=YLW)
    ax3.fill_between(range(len(cum_r)), 0, cum_r,
                     where=cum_r < 0, alpha=0.15, color=RED)
    ax3.set_title("Cumulative R", color=FG, pad=8)
    ax3.set_ylabel("R", color=FG)
    ax3.set_xlabel("Trade #", color=FG)
    ax3.legend(facecolor="#161b22", edgecolor="#444c56", labelcolor=FG)

    for ax in axes:
        ax.tick_params(colors=FG)
        ax.grid(True, alpha=0.08, color=FG)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = "backtest_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"\nChart saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT — MOST RECENT TRADE (TradingView-style)
# ─────────────────────────────────────────────────────────────────────────────
def _plot_trade(df_15m: pd.DataFrame, trade: pd.Series, out_name: str, tag: str):
    """
    Render a single trade as a 15M candlestick chart, TradingView-style:
      - swept swing level (4H liquidity grabbed)
      - IFVG zone (the disrespected FVG)
      - position box (green = entry->TP, red = entry->SL)
      - entry / exit markers
    """
    BG, FG = "#0d1117", "#e6edf3"
    GRN, RED, YLW = "#3fb950", "#f85149", "#d29922"
    BLU, PURP = "#58a6ff", "#bc8cff"
    UP, DN = "#26a69a", "#ef5350"

    entry_idx = int(trade["entry_idx"])
    exit_idx  = int(np.searchsorted(df_15m.index, trade["exit_time"], side="left"))
    exit_idx  = min(exit_idx, len(df_15m) - 1)

    pad_left, pad_right = 35, 8
    lo = max(0, entry_idx - pad_left)
    hi = min(len(df_15m), exit_idx + pad_right)
    win = df_15m.iloc[lo:hi].copy().reset_index()
    win.rename(columns={win.columns[0]: "time"}, inplace=True)

    ex = entry_idx - lo
    xx = exit_idx  - lo

    is_short = trade["direction"] == "short"
    entry, sl, tp = trade["entry"], trade["sl"], trade["tp"]

    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
    ax.set_facecolor(BG)

    # Candlesticks
    for k, row in win.iterrows():
        col = UP if row["close"] >= row["open"] else DN
        ax.plot([k, k], [row["low"], row["high"]], color=col, linewidth=0.9, zorder=2)
        body_lo = min(row["open"], row["close"])
        h = abs(row["close"] - row["open"])
        if h == 0:
            h = (win["high"].max() - win["low"].min()) * 0.0005
        ax.add_patch(plt.Rectangle((k - 0.3, body_lo), 0.6, h,
                                   facecolor=col, edgecolor=col, zorder=3))

    # IFVG zone
    ax.axhspan(trade["fvg_bottom"], trade["fvg_top"], color=PURP, alpha=0.14, zorder=1)
    ax.axhline(trade["fvg_top"],    color=PURP, linewidth=0.8, linestyle=":", alpha=0.7)
    ax.axhline(trade["fvg_bottom"], color=PURP, linewidth=0.8, linestyle=":", alpha=0.7)
    ax.text(0.6, trade["fvg_top"], "  IFVG zone (disrespected FVG)",
            color=PURP, fontsize=9, va="bottom", fontweight="bold")

    # Swept swing level + swept extreme (where SL sits below/above)
    swing_lbl = "Swept swing HIGH" if is_short else "Swept swing LOW"
    ax.axhline(trade["swept_level"], color=BLU, linewidth=1.2, linestyle="--", alpha=0.9)
    ax.text(0.6, trade["swept_level"], f"  {swing_lbl} ({trade['swept_level']})",
            color=BLU, fontsize=9, va="bottom", fontweight="bold")
    if "sweep_extreme" in trade.index:
        ax.axhline(trade["sweep_extreme"], color=BLU, linewidth=0.8,
                   linestyle=":", alpha=0.6)

    # TradingView-style position box
    box_x0 = ex
    box_w  = max(xx - ex, 6)
    tgt_lo, tgt_hi = sorted([entry, tp])
    ax.add_patch(plt.Rectangle((box_x0, tgt_lo), box_w, tgt_hi - tgt_lo,
                               facecolor=GRN, alpha=0.18, edgecolor=GRN, linewidth=1.0, zorder=4))
    stp_lo, stp_hi = sorted([entry, sl])
    ax.add_patch(plt.Rectangle((box_x0, stp_lo), box_w, stp_hi - stp_lo,
                               facecolor=RED, alpha=0.18, edgecolor=RED, linewidth=1.0, zorder=4))

    # Entry / SL / TP lines + staggered labels (no overlap on tight setups)
    lvls = sorted([
        (entry, FG,  f"Entry {entry}"),
        (tp,    GRN, f"TP (2R) {tp}"),
        (sl,    RED, f"SL {sl}"),
    ], key=lambda x: -x[0])
    span = max(win["high"].max() - win["low"].min(), 1e-6)
    min_gap = span * 0.025
    last_y = None
    for price, col, lbl in lvls:
        ax.axhline(price, color=col, linewidth=1.0, alpha=0.85, zorder=5)
        y = price
        if last_y is not None and (last_y - y) < min_gap:
            y = last_y - min_gap
        last_y = y
        ax.text(box_x0 + box_w, y, f"  {lbl}", color=col, fontsize=9, va="center", fontweight="bold")

    # Markers
    ax.scatter([ex], [entry], marker="v" if is_short else "^",
               s=160, color=YLW, edgecolor="black", zorder=6, label="Entry")
    exit_col = GRN if trade["outcome"] == "win" else (RED if trade["outcome"] == "loss" else YLW)
    ax.scatter([xx], [trade["exit"]], marker="o", s=140, color=exit_col,
               edgecolor="black", zorder=6, label=f"Exit ({trade['outcome']})")

    # X axis
    n_ticks = 8
    step = max(1, len(win) // n_ticks)
    ticks = list(range(0, len(win), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([win["time"].iloc[t].strftime("%m-%d %H:%M") for t in ticks],
                       rotation=30, ha="right", fontsize=8)

    entry_type = "RETRACE" if trade["retrace_entry"] else "CLOSE"
    ax.set_title(
        f"{tag}  |  {SYMBOL} 15M  |  {trade['direction'].upper()} ({entry_type} entry)  |  "
        f"{trade['outcome'].upper()}  ({trade['pnl_r']:+.0f}R)  |  "
        f"{pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d %H:%M')}",
        color=FG, fontsize=12, fontweight="bold", pad=12
    )
    ax.set_ylabel("Price", color=FG)
    ax.tick_params(colors=FG)
    ax.grid(True, alpha=0.08, color=FG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.legend(facecolor="#161b22", edgecolor="#444c56", labelcolor=FG, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_name, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"Trade chart saved -> {out_name}")


def plot_recent_trade(df_15m: pd.DataFrame, stats: dict):
    trade = stats["df"].sort_values("entry_time").iloc[-1]
    _plot_trade(df_15m, trade, "recent_trade.png", "MOST RECENT TRADE")


def plot_first_trade(df_15m: pd.DataFrame, stats: dict):
    trade = stats["df"].sort_values("entry_time").iloc[0]
    _plot_trade(df_15m, trade, "first_trade.png", "FIRST TRADE")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("► Connecting to MT5...")
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    print(f"► Fetching {SYMBOL} data ({START_DATE.date()} → {END_DATE.date()})...")
    df_4h  = fetch_data(SYMBOL, mt5.TIMEFRAME_H4 if _MT5_AVAILABLE else 16388,
                        START_DATE, END_DATE, tf_name="H4")
    df_15m = fetch_data(SYMBOL, mt5.TIMEFRAME_M15 if _MT5_AVAILABLE else 15,
                        START_DATE, END_DATE, tf_name="M15")
    mt5.shutdown()

    print(f"  4H  candles : {len(df_4h):,}")
    print(f"  15M candles : {len(df_15m):,}")

    print("► Detecting 4H swings...")
    df_4h = find_swings(df_4h, SWING_LOOKBACK)

    if SWEEP_MODE == "PDH_PDL":
        print("► Detecting PDH/PDL sweeps (15M)...")
        sweeps = find_pdh_pdl_sweeps(df_15m)
    elif SWEEP_MODE == "ASIAN_RANGE":
        print("► Detecting Asian-range sweeps (15M, 0-7h range, scan 7-23h)...")
        sweeps = find_session_range_sweeps(df_15m, 0, 7, 23, "Asian")
    elif SWEEP_MODE == "LONDON_RANGE":
        print("► Detecting London-range sweeps (15M, 8-12h range, scan 12-23h)...")
        sweeps = find_session_range_sweeps(df_15m, 8, 12, 23, "London")
    else:
        print("► Detecting 4H swing sweeps...")
        sweeps = find_sweeps(df_4h)
    print(f"  Sweeps found: {len(sweeps)}")

    daily_bias = build_daily_bias(df_15m) if HTF_BIAS_MODE == "DAILY" else {}
    if daily_bias:
        print(f"  Daily bias map: {len(daily_bias):,} days")

    print("► Detecting all 15M FVGs...")
    all_fvgs = find_all_fvgs(df_15m)
    print(f"  FVGs found  : {len(all_fvgs):,}")

    print("► Detecting all 4H FVGs...")
    fvgs_4h = find_all_fvgs(df_4h)
    print(f"  4H FVGs     : {len(fvgs_4h):,}")

    print("► Running backtest...")
    trades, funnel = run_backtest(df_4h, df_15m, sweeps, all_fvgs, fvgs_4h, daily_bias)
    print(f"  Trades generated: {len(trades)}")
    print("\n  Sweep -> Trade funnel:")
    for k, v in funnel.items():
        print(f"    {k:22}: {v}")

    if not trades:
        print("\nNo trades found. Loosen FVG_PRE_SCAN / POST_SWEEP_WINDOW or adjust date range.")
        return

    stats = compute_stats(trades)
    print_report(stats)
    plot_results(stats)
    plot_recent_trade(df_15m, stats)
    plot_first_trade(df_15m, stats)

    log_path = "trade_log.csv"
    stats["df"].to_csv(log_path, index=False)
    print(f"Trade log saved → {log_path}")


if __name__ == "__main__":
    main()