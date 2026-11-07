"""
Asia-sweep + London 1M-CISD morning bot — configuration.

Sessions are anchored to UTC+1 (your local clock), per spec:
  Asia range :  00:00 - 07:00 UTC+1
  London     :  07:00 - 10:00 UTC+1   (sweep + CISD + entry must all happen here)
The bot converts to broker time automatically by measuring the broker's UTC
offset from a live tick, so a broker DST change can never break the sessions.
"""
import os

# ── Symbols ──
# USDCHF removed 2026-07-16: over 25y of M5 HistData it was one of the worst
# majors (PF 0.84 fixed / 0.80 asia-TP, 4/25 green years).
SYMBOLS = ["EURUSD"]

# ── Sessions (hours, UTC+1) ──
ASIA_START_H   = 0      # 00:00 UTC+1
ASIA_END_H     = 7      # 07:00 UTC+1
LONDON_START_H = 7      # 07:00 UTC+1
LONDON_END_H   = 10     # 10:00 UTC+1

# ── Strategy ──
RR              = 2.2    # broker-side take-profit at 2.2R (~$209 on $95 risk). This
                         # is the BACKSTOP TP — the bot also manually market-closes
                         # the position the moment floating profit hits
                         # PROFIT_CLOSE_USD, so 2.2R only fills if we miss that tick.
PROFIT_CLOSE_USD = 200.0 # primary exit: manually close any open position once its
                         # floating profit (profit+swap) reaches this many USD.
CISD_MIN_SERIES = 2      # delivery leg must be >= this many consecutive same-close
                         # candles (a valid CISD needs >= 2 up/down candles). The
                         # CISD level = open of the FIRST candle in that run.
CISD_MAX_WALKBACK = 60   # max bars to walk back when finding the delivery series
CISD_LEG_MAKES_EXTREME = False  # True = the delivery run must END exactly on the
                                # bar that creates the London high/low, i.e. the
                                # last 2+ candles are the ones that MAKE the
                                # extreme. If the extreme bar is an opposite-body
                                # candle the setup is rejected outright (no
                                # walking back to an earlier run). Causal.
CISD_LEG_AFTER_SWEEP = False  # True = the delivery run must start at/after the
                              # sweep bar (leg must be the post-sweep manipulation).
                              # Diagnostic: 36% of legs currently start BEFORE the
                              # sweep, i.e. the CISD level comes off a pre-sweep
                              # candle. Causal + knowable live. Default off.
FVG_MIN_WIDTH_PIPS = 0.3  # min 3-candle gap to treat a displacement FVG as real;
                          # if the CISD move leaves a gap >= this, the limit rests
                          # on the FVG near-edge (FVG high for buys / low for sells)
                          # instead of the CISD level
MIN_SL_PIPS     = 3.0    # widen SL to at least this many pips (microscopic-stop guard)
SL_PAD_PIPS     = 0.0    # SL locked EXACTLY at the swing high/low — no buffer,
                         # no spread. (Min-SL floor below still applies if the
                         # swing sits < MIN_SL_PIPS from entry.)
ENTRY_MODE      = "market"  # "market" = fill every CISD at the confirmation close
                            #            (backtests better: PF 0.49 vs 0.35, no
                            #             missed runners like the 2026-07-09 EURUSD)
                            # "limit"  = rest at the CISD/FVG level (retrace entry;
                            #            better price but misses trades that don't pull back)
CISD_BIG_CANDLE_MULT = 1.0  # market mode only: if the confirmation candle closes
                            # more than this multiple of the (level→swing) stop
                            # distance BEYOND the CISD level, the candle is "too
                            # big" to chase — rest a LIMIT at the CISD level instead
                            # of market-executing at the extended close.

# ── Risk & protections ──
RISK_PCT           = 0.95   # % of balance risked per trade when balance >= RISK_ANCHOR
RISK_ANCHOR        = 10_000.0  # below this balance → risk a FLAT amount (RISK_FLAT);
                               # at/above it → risk RISK_PCT% of the live balance.
# Flat amount is DERIVED from the anchor so the two rules always meet exactly at
# RISK_ANCHOR (no jump). At 0.95% of 10k that's a flat 95 → a 200/95 R win = 200.
RISK_FLAT          = RISK_ANCHOR * RISK_PCT / 100
DAILY_STOP_PCT     = 4.0    # no NEW entries once day P&L <= -4%
MAX_OVERALL_DD_PCT = 9.0    # halt bot entirely if equity <= start balance x (1-9%)
INITIAL_BALANCE    = None   # set to the account's starting balance for a fixed
                            # FTMO-style floor; None = anchor to balance at first run
MAX_SPREAD_PIPS    = 2.5    # skip entry if live spread wider than this
ONE_TRADE_PER_PAIR_PER_SESSION = True

# ── Gold heartbeat marker ──
# A never-fill BUY-LIMIT on gold at an absurdly low price so you can SEE in the
# MT5 terminal that the bot is alive (placed on startup, cancelled on shutdown).
# Never fills → costs nothing, holds no risk.
HEARTBEAT_ENABLED = True
HEARTBEAT_SYMBOLS = ["XAUUSD", "GOLD", "XAUUSD.cash"]   # first one the broker offers
HEARTBEAT_PRICE   = 1.0        # far below gold's price → can never fill
HEARTBEAT_COMMENT = "ASIABOT-ALIVE"
HEARTBEAT_MAGIC   = 442031

# ── Bot mechanics ──
LOOP_SEC    = 3          # M1 strategy → poll fast during London
MAGIC       = 442030
COMMENT     = "asiacisd"
STATE_FILE  = os.path.join(os.path.dirname(__file__), "state.json")
LOG_FILE    = os.path.join(os.path.dirname(__file__), "bot.log")
TRADES_CSV  = os.path.join(os.path.dirname(__file__), "trades.csv")

M1_BARS     = 2000       # bars fetched per poll (covers the whole session day)

# Live bot trades SYMBOLS only; the rest are here so backtests across other
# pairs get the right pip size (JPY quotes are 1e-2).
PIP = {
    "EURUSD": 1e-4, "USDCHF": 1e-4, "GBPUSD": 1e-4, "AUDUSD": 1e-4,
    "USDCAD": 1e-4, "NZDUSD": 1e-4, "EURGBP": 1e-4,
    "USDJPY": 1e-2, "EURJPY": 1e-2, "GBPJPY": 1e-2,
}
