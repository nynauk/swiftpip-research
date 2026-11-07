"""
Live bot configuration — the deployed gradient-exit strategy.

DEPLOYED: the CURATED 21-instrument, THREE-TIMEFRAME (M3+M5+M15) portfolio,
validated 2000-2026 on HistData+MT5 dual feed, spread-aware (per-instrument
spread on fills + stop widened by live spread+1pip):
  combined PF 2.95 · 1.39 trades/day · WR 69.5% · max DD -5.0% · LS 8 ·
  27/27 years green · FTMO sim @1%: 260 payouts, $3.10M banked, 0 blown.
Each timeframe is an independent engine (own magic, 1 position per symbol per
TF, 1% risk per trade). Strategy per engine:
  EMA 7/18 cross -> first aligned FVG (win 24 bars) -> retest entry -> RSI 45/55
  momentum -> SL at recent swing +0.25xFVG +(spread+1pip) -> DOL>=2R selection
  filter -> gradient exit (fast-EMA slope flips while in profit) / opposite
  cross / SL.
"""
import os

STRATEGY = "gradient"          # gradient-exit EMA+DOL

# ── Symbols: the CURATED 21 (PF>=2 over 26yr; losers like AUDNZD/XAGUSD/EU50 cut) ──
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",                                # FX majors
    "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "NZDJPY",                      # JPY crosses
    "EURAUD", "GBPAUD",                                                    # trend crosses
    "XAUUSD",                                                              # gold
    "US500.cash", "US100.cash", "GER40.cash", "UK100.cash",                # core indices
    "FRA40.cash", "JP225.cash", "HK50.cash",                               # extra indices
    "USOIL.cash", "UKOIL.cash",                                            # energy
]
SYMBOL = SYMBOLS[0]            # legacy alias
# alternate broker names, tried in order if the primary name isn't offered
SYMBOL_ALIASES = {
    "FRA40.cash": ["FR40.cash", "FRA40", "FR40"],
    "JP225.cash": ["JPN225.cash", "JP225"],
    "HK50.cash":  ["HKG50.cash", "HK50"],
}

# ── Timeframe engines: each runs the full strategy independently ──
TIMEFRAMES = ["M3", "M5", "M15"]
TF_MAGIC   = {"M3": 442021, "M5": 442022, "M15": 442018}   # M15 keeps the legacy magic
TF_COMMENT = {"M3": "grad718-m3", "M5": "grad718-m5", "M15": "grad718"}

# ── Entry parameters (must match the backtest) ──
EMA_FAST          = 7
EMA_SLOW          = 18
SWING_LOOKBACK    = 5
FVG_LOOKUP_WINDOW = 24         # M15 bars after cross to find aligned FVG
RETEST_WINDOW_EMA = 48         # M15 bars after FVG to wait for edge retest
SL_BUFFER_FRAC    = 0.25       # SL buffer as fraction of FVG width
MIN_RR            = 2.0        # DOL must be >= 2R away (selection filter)
MAX_RR            = 10.0       # ... and <= 10R
RSI_PERIOD        = 14
RSI_LONG_MIN      = 45         # long allowed only if RSI >= 45  (momentum)
RSI_SHORT_MAX     = 55         # short allowed only if RSI <= 55
DRIFT_TOL_FRAC    = 0.25       # skip if live price ran >0.25R past the FVG edge

# ── Gradient exit (managed by manager.py, no fixed TP) ──
EXIT_MODE         = "gradient"
GRAD_LOOKBACK     = 1          # fast-EMA slope over 1 closed bar
GRADIENT_PROFIT_ONLY = True    # only arm the gradient exit once in profit
SAFETY_TP_R       = 15.0       # far broker-side TP as bot-death insurance (rarely hit)

# reject microscopic FVGs (price units) — from the backtest SPEC (all_pairs U)
MIN_FVG_WIDTH = {
    "EURUSD": 2.0e-4, "GBPUSD": 2.0e-4, "USDJPY": 2.0e-2, "USDCHF": 2.0e-4,
    "EURJPY": 2.0e-2, "GBPJPY": 3.0e-2, "AUDJPY": 2.0e-2, "CADJPY": 2.0e-2,
    "NZDJPY": 2.5e-2, "EURAUD": 3.0e-4, "GBPAUD": 4.0e-4,
    "XAUUSD": 0.30,
    "US500.cash": 1.5, "US100.cash": 5.0, "GER40.cash": 3.0, "UK100.cash": 2.0,
    "FRA40.cash": 3.0, "JP225.cash": 15.0, "HK50.cash": 15.0,
    "USOIL.cash": 0.08, "UKOIL.cash": 0.08,
}

# per-instrument pip size (price units) — the SL is widened by (live spread + 1 pip)
# for broker realism, matching edr.PIP / sl_pad in the backtest.
PIP = {
    "EURUSD": 1e-4, "GBPUSD": 1e-4, "USDCHF": 1e-4,
    "EURAUD": 1e-4, "GBPAUD": 1e-4,
    "USDJPY": 1e-2, "EURJPY": 1e-2, "GBPJPY": 1e-2, "AUDJPY": 1e-2,
    "CADJPY": 1e-2, "NZDJPY": 1e-2,
    "XAUUSD": 0.1,
    "US500.cash": 1.0, "US100.cash": 1.0, "GER40.cash": 1.0, "UK100.cash": 1.0,
    "FRA40.cash": 1.0, "JP225.cash": 1.0, "HK50.cash": 1.0,
    "USOIL.cash": 0.01, "UKOIL.cash": 0.01,
}

# ── Risk (FTMO-sized) ──
RISK_PCT           = 1.0       # % of balance risked per trade
# ⚠️ Set to THIS account's STARTING balance so the overall-loss floor matches
# FTMO's static max-loss rule (floor = INITIAL_BALANCE × (1 - MAX_OVERALL_DD_PCT/100)).
# e.g. 10_000 for a $10k account, 25_000 for $25k. None = auto-anchor to the
# balance when the bot first connects to the account.
INITIAL_BALANCE    = 10_000.0
DAILY_STOP_PCT     = 4.0       # halt trading for the day at -4% (buffer under FTMO -5%)
MAX_OVERALL_DD_PCT = 9.0       # halt entirely if account draws to -9% (buffer under -10%)
MAX_CONCURRENT_PER_SYMBOL = 1  # one position per symbol
MAX_TRADES_PER_DAY = 4         # hard cap on NEW entries per broker day (resets at midnight)

# ── News filter (full red-folder day block, matches backtest) ──
NEWS_FILTER = True

# ── Heartbeat marker ──
# A never-fill BUY-LIMIT on XAUUSD at an absurd price so you can SEE at a glance
# in the terminal that the bot is alive. Placed on startup, cancelled on shutdown.
HEARTBEAT_ENABLED = True
HEARTBEAT_SYMBOL  = "XAUUSD"
HEARTBEAT_PRICE   = 1.0        # far below gold's price → can never fill
HEARTBEAT_COMMENT = "BOT-ALIVE"
HEARTBEAT_MAGIC   = 442099     # own magic — XAUUSD is now a TRADED symbol, keep the marker separate

# ── Bot mechanics ──
LOOP_SEC   = 15
MAGIC      = 442018            # legacy/M15 magic (sweep path + heartbeat fallback)
COMMENT    = "grad718"
STATE_FILE = os.path.join(os.path.dirname(__file__), "state_gradient.json")
LOG_FILE   = os.path.join(os.path.dirname(__file__), "bot_gradient.log")

# ── Data window (bars fetched per poll, same count for every TF engine) ──
BARS_BACK     = 400
M15_BARS_BACK = 400            # legacy alias (sweep strategy path)
