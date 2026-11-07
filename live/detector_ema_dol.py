"""
EMA(9/21) + DOL live detector — mirrors ema_dol_strategy.py backtest logic.

Signal chain (all mechanical, computed on CLOSED M15 bars only):
  1. EMA(FAST) crosses EMA(SLOW)            → bias (up-cross = long)
  2. First aligned FVG within FVG_WINDOW bars after the cross
  3. Entry = the MOMENT price tags the FVG edge (top for long, bottom for
     short). The bot polls every 15s; when the live bid touches the edge we
     market-in — the realistic backtest modeled exactly this fill
     (edge + spread + slippage).
  4. SL  = recent swing low/high (80-bar lookback) buffered by 0.25×FVG width
  5. TP  = Draw on Liquidity: next opposing swing (or PDH/PDL if closer)
  6. RR gate [MIN_RR, MAX_RR] evaluated ONCE — at the tag moment — exactly
     like the backtest evaluates it at the retest bar. A cross that fails
     the gate is consumed, never re-evaluated.

Returned entry modes (only states that require caller action):
  'market' — price is tagging the edge right now, within drift tolerance → trade
  'reject' — tagging now but RR outside [MIN_RR, MAX_RR] → consume, no trade
  'missed' — the tag happened on a closed bar (bot offline) or price ran
             more than DRIFT_TOL×risk past the edge → consume, no trade
Setups still waiting for their tag are NOT returned (rescanned next poll).

Faithfulness note: the backtest computes swings over the full dataset, so a
swing forming within 5 bars of entry could use up to 4 future bars. Live we
only see confirmed swings (strictly causal) — never less conservative.
"""
from dataclasses import dataclass
from typing import List, Set
import numpy as np
import pandas as pd


DRIFT_TOL_FRAC = 0.25   # max adverse drift past the edge, as fraction of risk


@dataclass
class EmaDolSetup:
    cross_time: pd.Timestamp     # dedup key
    bias: str                    # 'long' / 'short'
    fvg_time: pd.Timestamp
    fvg_top: float
    fvg_bottom: float
    fvg_width: float
    direction: str               # 'buy' / 'sell'
    edge_price: float            # the FVG edge that was tagged
    sl_price: float
    tp_price: float              # DOL target
    rr: float                    # RR from the live fill price
    risk_price: float            # |fill - sl| at detection
    entry_mode: str              # 'market' | 'reject' | 'missed'
    reason: str = ""


def _find_swings(h: np.ndarray, l: np.ndarray, n: int):
    sh = np.full(len(h), np.nan)
    sl = np.full(len(l), np.nan)
    for i in range(n, len(h) - n):
        if h[i] >= h[i - n:i + n + 1].max():
            sh[i] = h[i]
        if l[i] <= l[i - n:i + n + 1].min():
            sl[i] = l[i]
    return sh, sl


def _find_aligned_fvg(h, l, start_idx, end_idx, direction, min_width):
    for k in range(start_idx + 2, end_idx + 1):
        if direction == "bullish":
            if l[k] > h[k - 2]:
                width = float(l[k] - h[k - 2])
                if width >= min_width:
                    return {"direction": "bullish", "top": float(l[k]),
                            "bottom": float(h[k - 2]), "width": width, "form_idx": k}
        else:
            if h[k] < l[k - 2]:
                width = float(l[k - 2] - h[k])
                if width >= min_width:
                    return {"direction": "bearish", "top": float(l[k - 2]),
                            "bottom": float(h[k]), "width": width, "form_idx": k}
    return None


def _next_liquidity_above(price, swing_highs, until_idx, pdh):
    candidates = []
    for i in range(until_idx - 1, max(-1, until_idx - 200), -1):
        if not np.isnan(swing_highs[i]) and swing_highs[i] > price:
            candidates.append(swing_highs[i])
    target = min(candidates) if candidates else None
    if pdh is not None and not pd.isna(pdh) and pdh > price:
        if target is None or pdh < target:
            target = pdh
    return target


def _next_liquidity_below(price, swing_lows, until_idx, pdl):
    candidates = []
    for i in range(until_idx - 1, max(-1, until_idx - 200), -1):
        if not np.isnan(swing_lows[i]) and swing_lows[i] < price:
            candidates.append(swing_lows[i])
    target = max(candidates) if candidates else None
    if pdl is not None and not pd.isna(pdl) and pdl < price:
        if target is None or pdl > target:
            target = pdl
    return target


def _recent_swing_low(swing_lows, until_idx, lookback=80):
    for i in range(until_idx, max(0, until_idx - lookback) - 1, -1):
        if not np.isnan(swing_lows[i]):
            return float(swing_lows[i])
    return None


def _recent_swing_high(swing_highs, until_idx, lookback=80):
    for i in range(until_idx, max(0, until_idx - lookback) - 1, -1):
        if not np.isnan(swing_highs[i]):
            return float(swing_highs[i])
    return None


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's RSI via EWM (alpha = 1/period). Matches edr.rsi in the backtest."""
    s = pd.Series(close)
    d = s.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).values


def _prev_day_levels(df: pd.DataFrame):
    """PDH/PDL relative to the date of the LAST closed bar."""
    daily = df.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    if len(daily) < 2:
        return None, None
    last_day = df.index[-1].normalize()
    prior = daily[daily.index < last_day]
    if not len(prior):
        return None, None
    return float(prior["high"].iloc[-1]), float(prior["low"].iloc[-1])


def detect_ema_dol_signals(df_15: pd.DataFrame,
                           already: Set[str],
                           *,
                           bid: float, ask: float,
                           fast: int = 9, slow: int = 21,
                           swing_n: int = 5,
                           fvg_window: int = 24,
                           retest_window: int = 48,
                           sl_buffer_frac: float = 0.25,
                           min_rr: float = 2.0,
                           max_rr: float = 10.0,
                           min_fvg_width: float = 0.0002,
                           rsi_period: int = 14,
                           rsi_long_min: float = 45.0,
                           rsi_short_max: float = 55.0,
                           sl_pad: float = 0.0) -> List[EmaDolSetup]:
    """
    df_15 must contain ONLY closed bars (caller drops the forming bar).
    `already` holds str(cross_time) keys previously consumed.
    Returns every cross needing action right now (usually 0 or 1 items).
    """
    out: List[EmaDolSetup] = []
    if len(df_15) < max(slow * 4, 120):
        return out

    h = df_15["high"].values
    l = df_15["low"].values
    c = df_15["close"].values
    idx = df_15.index
    n = len(df_15)
    last = n - 1

    ema_f = pd.Series(c, index=idx).ewm(span=fast, adjust=False).mean().values
    ema_s = pd.Series(c, index=idx).ewm(span=slow, adjust=False).mean().values
    above = ema_f > ema_s
    rsi = _rsi(c, rsi_period)

    sh, sl_arr = _find_swings(h, l, n=swing_n)
    pdh, pdl = _prev_day_levels(df_15)

    horizon = fvg_window + retest_window + 4
    lo = max(slow, n - horizon)

    for i in range(last, lo - 1, -1):
        if i < 1:
            break
        crossed_up = above[i] and not above[i - 1]
        crossed_dn = (not above[i]) and above[i - 1]
        if not (crossed_up or crossed_dn):
            continue
        key = str(idx[i])
        if key in already:
            continue
        bias = "long" if crossed_up else "short"
        target_dir = "bullish" if bias == "long" else "bearish"

        # MOST-RECENT-CROSS RULE: if an OPPOSITE cross has fired since this one,
        # the trend flipped — this setup is dead, don't hunt its FVG.
        flipped = False
        for j in range(i + 1, last + 1):
            o_up = above[j] and not above[j - 1]
            o_dn = (not above[j]) and above[j - 1]
            if (bias == "long" and o_dn) or (bias == "short" and o_up):
                flipped = True
                break
        if flipped:
            continue

        fvg = _find_aligned_fvg(h, l, i, min(i + fvg_window, last), target_dir, min_fvg_width)
        if fvg is None:
            continue                       # FVG window may still be open — rescan
        fi = fvg["form_idx"]
        if last > fi + retest_window:
            continue                       # retest window expired — dead cross

        edge = fvg["top"] if bias == "long" else fvg["bottom"]

        def _mk(mode, sl=0.0, tp=0.0, rr=0.0, risk=0.0, reason=""):
            return EmaDolSetup(
                cross_time=idx[i], bias=bias,
                fvg_time=idx[fi], fvg_top=fvg["top"], fvg_bottom=fvg["bottom"],
                fvg_width=fvg["width"],
                direction="buy" if bias == "long" else "sell",
                edge_price=float(edge), sl_price=float(sl), tp_price=float(tp),
                rr=float(rr), risk_price=float(risk),
                entry_mode=mode, reason=reason,
            )

        # Tag already happened on a CLOSED bar → the backtest entry is gone
        # (bot was offline or blocked at the time).
        closed_tag = False
        for k in range(fi + 1, last + 1):
            if bias == "long" and l[k] <= edge:
                closed_tag = True; break
            if bias == "short" and h[k] >= edge:
                closed_tag = True; break
        if closed_tag:
            out.append(_mk("missed", reason="edge tagged on a closed bar"))
            continue

        # Is the LIVE price tagging the edge right now? (bars are bid OHLC,
        # so the tag reference is the bid on both sides)
        tagging = (bid <= edge) if bias == "long" else (bid >= edge)
        if not tagging:
            continue                       # still waiting — no action

        # RSI momentum filter (backtest: long needs RSI >= 45, short needs <= 55)
        rv = rsi[last]
        if np.isnan(rv):
            continue
        if (bias == "long" and rv < rsi_long_min) or (bias == "short" and rv > rsi_short_max):
            out.append(_mk("reject", reason=f"RSI {rv:.0f} against momentum"))
            continue

        # SL from recent swing (strictly causal), buffered by 0.25×FVG width
        # AND widened by (spread + 1 pip) via sl_pad — matches the backtest's
        # edr.sl_pad, so a stop-out reflects the real worse fill.
        buf = fvg["width"] * sl_buffer_frac
        if bias == "long":
            swing_low = _recent_swing_low(sl_arr, last)
            anchor = swing_low if swing_low is not None and swing_low < fvg["bottom"] else fvg["bottom"]
            sl = anchor - buf - sl_pad
            fill = ask                     # market buy fills at ask
            risk = fill - sl
            if risk <= 0:
                out.append(_mk("reject", sl=sl, reason="risk<=0 at tag"))
                continue
            drift = edge - bid             # how far bid ran past the edge
            if drift > DRIFT_TOL_FRAC * risk:
                out.append(_mk("missed", sl=sl, risk=risk,
                               reason=f"price ran {drift:.5f} past edge"))
                continue
            dol = _next_liquidity_above(fill, sh, last, pdh)
            if dol is None:
                out.append(_mk("reject", sl=sl, risk=risk, reason="no DOL above"))
                continue
            tp = dol
            rr = (tp - fill) / risk
        else:
            swing_high = _recent_swing_high(sh, last)
            anchor = swing_high if swing_high is not None and swing_high > fvg["top"] else fvg["top"]
            sl = anchor + buf + sl_pad
            fill = bid                     # market sell fills at bid
            risk = sl - fill
            if risk <= 0:
                out.append(_mk("reject", sl=sl, reason="risk<=0 at tag"))
                continue
            drift = bid - edge
            if drift > DRIFT_TOL_FRAC * risk:
                out.append(_mk("missed", sl=sl, risk=risk,
                               reason=f"price ran {drift:.5f} past edge"))
                continue
            dol = _next_liquidity_below(fill, sl_arr, last, pdl)
            if dol is None:
                out.append(_mk("reject", sl=sl, risk=risk, reason="no DOL below"))
                continue
            tp = dol
            rr = (fill - tp) / risk

        if rr < min_rr or rr > max_rr:
            out.append(_mk("reject", sl=sl, tp=tp, rr=rr, risk=risk,
                           reason=f"rr {rr:.2f} outside [{min_rr},{max_rr}]"))
            continue

        out.append(_mk("market", sl=sl, tp=tp, rr=rr, risk=risk))

    # Oldest cross first — matches the backtest's chronological processing
    out.sort(key=lambda s: s.cross_time)
    return out
