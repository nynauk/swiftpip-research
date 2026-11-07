"""
Signal detector — same logic as the backtest, applied to a rolling live window.

Detects the LATEST completed prev-4H sweep + 15M IFVG setup that is ready
to trade. Returns None if no fresh setup exists.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

REV_WINDOW_DEFAULT   = 24     # 15M bars after sweep to find reversal FVG
DIS_DELAY_DEFAULT    = 7      # 15M bars for FVG to invert
DIS_MULT_DEFAULT     = 1.5    # too-big threshold


@dataclass
class Setup:
    sweep_time:    pd.Timestamp
    bias:          str        # 'bullish' or 'bearish'
    swept_level:   float
    fvg_top:       float
    fvg_bottom:    float
    fvg_width:     float
    disrespect_time: pd.Timestamp
    disrespect_low:  float
    disrespect_high: float
    dis_range:     float
    entry_type:    str        # 'close' or 'retest'
    entry_price:   float
    sl_price:      float
    tp_price:      float      # 3R
    direction:     str        # 'buy' or 'sell'
    risk_price:    float


def _find_fvgs(h: np.ndarray, l: np.ndarray):
    """Return list of (form_idx, direction, top, bottom, width)."""
    fvgs = []
    for i in range(len(h) - 2):
        # bullish: c[i+2].low > c[i].high
        if l[i + 2] > h[i]:
            fvgs.append((i + 2, "bullish", float(l[i + 2]), float(h[i]),
                         float(l[i + 2] - h[i])))
        elif h[i + 2] < l[i]:
            fvgs.append((i + 2, "bearish", float(l[i]), float(h[i + 2]),
                         float(l[i] - h[i + 2])))
    return fvgs


def _prev4h_sweeps(df_15: pd.DataFrame, df_h4: pd.DataFrame):
    """Detect prev-4H-candle sweeps on 15M. Uses only CLOSED bars."""
    if len(df_h4) < 2 or len(df_15) < 2:
        return []
    # Use closed bars only: drop last row (currently forming)
    h4_closed = df_h4.iloc[:-1]
    if len(h4_closed) < 2:
        return []
    prevH = h4_closed["high"].shift(1)
    prevL = h4_closed["low"].shift(1)
    # For each 15M timestamp, find its 4H "floor" period and look up prev H/L
    floor4h = df_15.index.floor("4h")
    ph_map = {ts: prevH.loc[ts] for ts in h4_closed.index if ts in prevH.index}
    pl_map = {ts: prevL.loc[ts] for ts in h4_closed.index if ts in prevL.index}
    # Also allow lookup even if the 4H bar hasn't been fetched: fall back to
    # last available prev H/L before the 15M bar.
    valid_h4 = h4_closed.index

    h = df_15["high"].values
    l = df_15["low"].values
    c = df_15["close"].values
    idx = df_15.index
    sweeps = []
    cur = None; hd = ld = False
    for i in range(len(df_15)):
        p = floor4h[i]
        if p != cur:
            cur = p; hd = ld = False
        # find prev-4H H/L applicable to this bar
        # need the 4H that STARTED at or before p (that's the "current" 4H)
        # and its shift(1) is the prev candle.
        applicable = valid_h4[valid_h4 <= p]
        if len(applicable) == 0:
            continue
        this4h = applicable[-1]
        ph = ph_map.get(this4h)
        pl = pl_map.get(this4h)
        if ph is None or pl is None or np.isnan(ph) or np.isnan(pl):
            continue
        if not hd and h[i] > ph and c[i] < ph:
            sweeps.append({"time": idx[i], "direction": "bearish",
                           "swept_level": float(ph), "idx": i})
            hd = True
        if not ld and l[i] < pl and c[i] > pl:
            sweeps.append({"time": idx[i], "direction": "bullish",
                           "swept_level": float(pl), "idx": i})
            ld = True
    return sweeps


def detect_latest_setup(df_15: pd.DataFrame, df_h4: pd.DataFrame,
                        already_traded_sweep_times: set,
                        rev_window: int = REV_WINDOW_DEFAULT,
                        dis_delay: int = DIS_DELAY_DEFAULT,
                        dis_mult:  float = DIS_MULT_DEFAULT,
                        retr_window: int = 48,
                        rr: float = 3.0) -> Optional[Setup]:
    """
    Scan the recent history for the MOST RECENT ready-to-trade setup that hasn't
    been traded yet. Returns None if nothing fresh.

    Uses CLOSED bars only (drops the currently-forming last bar of df_15).
    """
    if len(df_15) < 30:
        return None
    df_closed = df_15.iloc[:-1]        # closed 15M bars
    h = df_closed["high"].values
    l = df_closed["low"].values
    c = df_closed["close"].values
    idx = df_closed.index

    fvgs = _find_fvgs(h, l)
    fvgs_sorted = sorted(fvgs, key=lambda f: f[0])
    fia = np.array([f[0] for f in fvgs_sorted])

    sweeps = _prev4h_sweeps(df_closed, df_h4)
    if not sweeps:
        return None

    # Walk newest → oldest sweep; return the first that has a valid ready setup.
    # NOTE: `already_traded_sweep_times` is a set of STRINGS (from JSON),
    # while sw["time"] is a pd.Timestamp — must stringify to match.
    for sw in reversed(sweeps):
        if str(sw["time"]) in already_traded_sweep_times:
            continue
        bias = sw["direction"]
        si = sw["idx"]
        if si >= len(idx) - 2:
            continue                    # not enough post-sweep bars yet
        tdir = "bearish" if bias == "bullish" else "bullish"
        rev_end = min(si + rev_window, len(idx) - 1)
        # search FVGs in window (nearest the sweep first, matching backtest)
        lo = int(np.searchsorted(fia, si, "left"))
        hi = int(np.searchsorted(fia, rev_end, "right"))
        for p in range(hi - 1, lo - 1, -1):
            f_i, fdir, ftop, fbot, fwid = fvgs_sorted[p]
            if fdir != tdir or fwid <= 0:
                continue
            for di in range(f_i + 1, min(f_i + dis_delay, len(idx) - 1) + 1):
                inv = c[di] > ftop if fdir == "bearish" else c[di] < fbot
                if not inv:
                    continue

                dis_low = float(l[di]); dis_high = float(h[di])
                dis_range = dis_high - dis_low
                too_big = dis_range > dis_mult * fwid

                # sweep structural extreme (for reference; not used as SL here)
                sweep_ext = float(l[si:di + 1].min()) if bias == "bullish" else float(h[si:di + 1].max())

                if not too_big:
                    # standard entry — but only fires if the disrespect candle
                    # is the MOST RECENT closed bar (else the ship sailed).
                    if di != len(idx) - 1:
                        break
                    entry = float(c[di])
                    entry_type = "close"
                    entry_idx = di
                else:
                    # retest entry — wait for pullback into FVG edge
                    retr_end = min(di + retr_window, len(idx) - 1)
                    entry_idx = None
                    entry = None
                    for rj in range(di + 1, retr_end + 1):
                        if bias == "bullish" and l[rj] <= ftop:
                            entry_idx = rj; entry = ftop; break
                        if bias == "bearish" and h[rj] >= fbot:
                            entry_idx = rj; entry = fbot; break
                    if entry_idx is None:
                        # retest not filled yet; if we're still inside the retest
                        # window, this is a LIMIT-order setup at fvg edge
                        if len(idx) - 1 - di <= retr_window:
                            entry = ftop if bias == "bullish" else fbot
                            entry_idx = None    # signals: pending limit
                            entry_type = "retest_pending"
                        else:
                            break
                    else:
                        # retest filled in the past — only trade if it filled on the
                        # LATEST bar (else we missed it)
                        if entry_idx != len(idx) - 1:
                            break
                        entry_type = "retest"

                # SL = disrespect candle low/high
                if bias == "bullish":
                    sl = dis_low; risk = entry - sl
                    direction = "buy"
                    tp = entry + rr * risk
                else:
                    sl = dis_high; risk = sl - entry
                    direction = "sell"
                    tp = entry - rr * risk
                if risk <= 0:
                    break

                return Setup(
                    sweep_time=sw["time"], bias=bias, swept_level=sw["swept_level"],
                    fvg_top=ftop, fvg_bottom=fbot, fvg_width=fwid,
                    disrespect_time=idx[di], disrespect_low=dis_low, disrespect_high=dis_high,
                    dis_range=dis_range, entry_type=entry_type,
                    entry_price=float(entry), sl_price=float(sl), tp_price=float(tp),
                    direction=direction, risk_price=float(risk),
                )
    return None
