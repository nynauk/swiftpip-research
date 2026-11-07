"""JSON persistence for the live bot — survives restarts."""
import copy
import json
import os
from datetime import datetime, timezone


DEFAULT = {
    "account":           None,     # MT5 login this state belongs to (guards against stale-account carryover)
    "traded_sweeps":     {},       # {symbol: [iso-str]}  — per-symbol dedup
    "positions":         {},       # ticket → {symbol, entry, sl, tp, risk_price, direction, sweep_time, original_lots}
    "pending_retests":   {},       # order ticket → {symbol, sweep_time, expires_at, ...}
    "day":               None,     # YYYY-MM-DD
    "day_start_balance": None,
    "day_pnl_pct":       0.0,
    "day_trades":        0,         # NEW entries taken this broker day (hard-capped, resets at midnight)
    "init_balance":      None,      # baseline for the FTMO overall-loss floor
    "last_poll":         None,
}


def new(account=None) -> dict:
    """A clean state for a given account (fresh nested dicts, no carryover)."""
    d = copy.deepcopy(DEFAULT)
    d["account"] = account
    return d


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _migrate(data: dict) -> dict:
    """Upgrade older state formats in place."""
    ts = data.get("traded_sweeps")
    if isinstance(ts, list):
        # legacy single-symbol format — bucket under 'legacy'
        data["traded_sweeps"] = {"legacy": ts}
    return data


def load(path: str) -> dict:
    if not os.path.exists(path):
        return {**DEFAULT}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        out = {**DEFAULT, **data}
        out = _migrate(out)
        return out
    except Exception as e:
        print(f"[state] load failed ({e}); starting fresh")
        return {**DEFAULT}


def save(path: str, data: dict, retries: int = 3, backoff: float = 0.25):
    """Atomic write with Windows-file-lock retry (Explorer/editor may hold it)."""
    import time as _t
    data["last_poll"] = _now_iso()
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        print(f"[state] write to {tmp} FAILED: {e}")
        return
    for i in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            _t.sleep(backoff * (i + 1))
    print(f"[state] os.replace({path}) failed after {retries} retries — "
          f"state may be stale until next successful save")


def traded_sweeps_for(data: dict, symbol: str) -> list:
    """Return the (possibly empty) per-symbol traded-sweep list. Creates it lazily."""
    if symbol not in data["traded_sweeps"]:
        data["traded_sweeps"][symbol] = []
    return data["traded_sweeps"][symbol]


def trim_traded_sweeps(data: dict, keep: int = 200):
    """Keep only the most recent N sweeps per symbol."""
    for sym, lst in data["traded_sweeps"].items():
        if len(lst) > keep:
            data["traded_sweeps"][sym] = lst[-keep:]
