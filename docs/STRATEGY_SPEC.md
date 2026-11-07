# ICT Sweep-IFVG Strategy — Full Specification

_A complete review document covering the trading logic, the backtest engine, and the live MT5 bot deployment (Option A: top-3 combined @ 0.33% risk)._

---

## 1. The Strategy — "Dry Basic 3RR"

The strategy is a single, mechanical, no-discretion ICT pattern:

> **When price sweeps the previous 4-hour candle's high or low, then closes back inside, wait for a 15-minute Inverted Fair Value Gap (IFVG) forming in the opposite direction. Enter on the disrespect candle (or on its retest if it's oversized). Stop at the disrespect candle's low/high. Target 3× that risk.**

### 1.1 Concepts

| Term | Definition |
|---|---|
| **Prev-4H sweep** | Current 15M bar's wick exceeds the previous 4H candle's high (or low), then the same bar closes back below it (or above). Signals liquidity taken. |
| **Fair Value Gap (FVG)** | A 3-candle imbalance: a bullish FVG exists when candle N+2's low > candle N's high (a "gap" in the middle candle). Bearish FVG is mirror. |
| **Inverted FVG (IFVG)** | An FVG that price then closes THROUGH in the opposite direction. Bullish sweep → we want a bearish FVG that gets closed above (the imbalance flipped). |
| **Disrespect candle** | The 15M bar that closes through the FVG — the one whose close inverts it. |
| **Disrespect SL** | Stop loss placed at the LOW of the disrespect candle (for longs) or its HIGH (for shorts). |
| **1.5× rule** | If the disrespect candle's range > 1.5 × FVG width, we treat it as too greedy — wait for a retest of the FVG edge instead of entering at candle close. |
| **3RR** | Take-profit at 3 × the price distance to the stop. |

### 1.2 The Five Rules

1. **Sweep detection.** On each new 15M bar, check if `high > prev_4H_high AND close < prev_4H_high` (bearish bias sweep — expect a move down). Mirror for bullish. Only one sweep per 4H window per direction.
2. **FVG search.** After the sweep, scan the next 24 M15 bars for an FVG in the OPPOSITE direction of the sweep (bearish sweep → look for bearish FVG). Nearest-to-sweep first.
3. **Inversion (disrespect).** Wait up to 7 M15 bars for a candle to close THROUGH the FVG (close > FVG top for bearish sweep). That's the disrespect candle.
4. **Entry.**
   - If disrespect range ≤ 1.5 × FVG width → enter at that candle's CLOSE (market entry).
   - If > 1.5 × width → place a LIMIT at the FVG edge (top for bullish, bottom for bearish), valid for 48 M15 bars (12 hours).
5. **SL / TP.** SL at disrespect low/high. TP at entry ± 3 × (entry − SL).

### 1.3 Why these choices

- **4H sweep, 15M entry.** The 4H timeframe defines a meaningful liquidity pool (institutional interest); 15M is granular enough to catch the reversal without overwhelming noise. This ratio (16:1) is a classic ICT pairing.
- **Opposite-direction FVG.** A bullish sweep implies buy-side liquidity was taken and price should reverse down. The FVG in the OPPOSITE direction (bearish) is the mechanical proof that the reversal is underway.
- **Disrespect SL (not sweep extreme).** SL at the sweep extreme gives more room but also worse RR — backtest showed disrespect SL nets ~30% more R across 21 symbols.
- **Plain 3RR (no partial, no BE).** Early testing tried partial-at-1R + break-even — winners averaged only 1.4R instead of 3R. Removed both to preserve pure 3RR expectancy.

### 1.4 Selected symbols (Option A)

From a 21-symbol scan (FX majors, crosses, indices, commodities, crypto), the three highest-quality symbols by PF+DD were:

| Symbol | PF | Max DD | Total R | Reasoning |
|---|---:|---:|---:|---|
| **US500.cash** | 1.35 | -23.3% | +508R | Best PF, cleanest curve |
| **GBPJPY** | 1.33 | -22.3% | +468R | Best PF among FX, high vol respects levels |
| **UKOIL.cash** | 1.32 | -30.0% | +524R | Highest total R, uncorrelated to equity/FX |

Running all three simultaneously → **6,572 trades, PF 1.33, +1,500R over 4.5 years.**

---

## 2. The Backtest Engine

### 2.1 Data source

MetaTrader 5 Python API (`MetaTrader5.copy_rates_from_pos`) via the user's live broker connection. Both H4 and M15 timeframes pulled for each symbol from `START_DATE` in [main.py](main.py) through today.

### 2.2 Core simulator — `combined_top3.py`

The file [combined_top3.py](combined_top3.py) is the Option A backtest. Key functions:

#### `sweeps_prev4h(df_15, df_h4)`

Iterates every 15M bar. For each:
- Floor to its 4H window.
- Look up the prev 4H high/low (shifted by 1 from the current 4H bar).
- If not-yet-swept in this 4H window AND bar's wick pierces prev H/L AND bar closes back inside → append a sweep.
- Only ONE sweep per direction per 4H window (the `hd`/`ld` flags in [combined_top3.py:44-55](combined_top3.py:44)).

Returns `[{"time", "direction", "swept_level"}, ...]`.

#### `simulate(h, l, idx, ei, direction, entry, sl, tp, rr, max_bars=400)`

Bar-by-bar walk forward from the entry bar. On each subsequent bar:
- For long: if `low ≤ sl` → loss; if `high ≥ tp` → win.
- For short: if `high ≥ sl` → loss; if `low ≤ tp` → win.
- Whichever hits first wins (which is realistic — worst case in backtesting is assuming SL hits first on same bar, we do this via order of the `if`s).
- If neither hits in 400 bars → `"open"` (trade left running past window; discarded from stats).

Returns `(outcome, exit_time, exit_price, pnl_r)`.

#### `run_symbol(symbol, news_set)`

For each sweep:
1. Find the FIRST valid FVG in the reversal direction within the search window.
2. Wait for inversion; classify as `"close"` or `"retest"` entry.
3. Apply news filter (if `entry.date() in news_set` → skip entire setup).
4. Apply concurrency (per-symbol cap of 1 open position).
5. Compute disrespect SL, 3R TP.
6. Simulate outcome, record trade.

### 2.3 News filter — [news_filter.py](news_filter.py)

Hardcoded red-folder days for 2022-2026:
- **NFP** — first Friday of every month (algorithmic)
- **FOMC** — 8×/year rate decisions
- **US CPI** — mid-month release
- **US Retail Sales** — mid-month release
- **ECB** — ~monthly rate decisions

`build_news_block_set(start_date, end_date)` returns a `set` of `datetime.date` objects. Any trade entry whose date is in the set is skipped entirely (not just delayed).

**Impact:** blocks 198 days across the 4.5-year backtest.

### 2.4 Combined-feed statistics

The magic in `stats()` (called on `all_df` with all 3 symbols) sorts by `entry_time` and treats the combined feed as a single stream. Streaks, max DD, R totals are computed on this chronologically-ordered stream — that's why "28 consecutive losses" is the worst combined-feed streak.

### 2.5 Concurrency exposure profile

For every trade, emit an `+1` event at `entry_time` and a `-1` at `exit_time`. Sort. Walk chronologically maintaining an "active" counter. Bucket hours-with-N-open. Result: 68% of time idle, 27% one open, 5% two or three.

### 2.6 What the backtest CANNOT model

- **Spread.** Backtest uses close-price entry with no slippage. Real fills lose ~0.1-0.5R over 6,000+ trades depending on symbol.
- **Broker-side stop-out timing.** MT5 stops out at last tick, not the bar's high/low. Some fills may be slightly better or worse than the OHLC assumes.
- **Weekend gaps.** Backtest treats Sunday open as continuous. In practice, gap-through-SL is possible on high-impact weekend events.
- **Requote / rejection.** Not modeled; assumed to always fill.

These are real drags but don't threaten the edge — expect ~15-20% slippage on total R vs. the theoretical +1,500R.

---

## 3. The Live Bot

### 3.1 Architecture

```
live/
├── config.py          — All tunable parameters
├── broker.py          — MT5 API wrappers (connect, get_bars, order_send, etc.)
├── detector.py        — Signal detection on a rolling live window
├── manager.py         — State cleanup (removes pruned positions)
├── state.py           — JSON persistence with Windows-lock retry
├── bot.py             — Main event loop
├── smoke_test.py      — Dry-run: no orders placed
├── test_trade.py      — Places a 0.01-lot buy, closes in 5s (end-to-end sanity)
└── state.json         — Persistent state (dedup, positions, day P&L)
```

### 3.2 Configuration — [config.py](live/config.py)

Current values (Option A):

```python
SYMBOLS = ["US500.cash", "GBPJPY", "UKOIL.cash"]
RR = 3.0
RISK_PCT = 0.33          # % of account balance per trade
DIS_MULT = 1.5           # too-big threshold
REV_WINDOW_M15 = 24      # bars after sweep to find FVG
DIS_DELAY_M15 = 7        # bars for FVG to invert
RETR_WINDOW_M15 = 48     # bars to wait for a retest fill (= 12h)
NEWS_FILTER = True
DAILY_STOP_PCT = 4.0     # account-wide daily stop
MAX_CONCURRENT_PER_SYMBOL = 1
LOOP_SEC = 15
MAGIC = 442015           # identifies this bot's orders
```

### 3.3 Detector — [detector.py](live/detector.py)

Called every 15 seconds per symbol. Same logic as the backtest but rolling:

1. **`_find_fvgs(h, l)`** — Scans the last 400 M15 bars for all 3-bar FVGs.
2. **`_prev4h_sweeps(df_15, df_h4)`** — Detects sweeps on CLOSED bars only (drops the currently-forming bar).
3. **`detect_latest_setup(...)`** — Walks newest sweep → oldest. For each untraded sweep:
   - Find first valid opposite-direction FVG in the search window.
   - Look for inversion within `DIS_DELAY_M15` bars.
   - Only fires if the disrespect candle IS THE MOST RECENTLY CLOSED BAR. This prevents entering setups whose "ship has sailed" while the bot was offline.
   - Classifies entry as `"close"`, `"retest"`, or `"retest_pending"` (limit not yet filled).
4. Returns a `Setup` dataclass with entry, SL, TP, direction, and metadata.

**Dedup:** `already_traded_sweep_times` is a set of ISO string timestamps loaded from `state.json`. `sw["time"]` (a `pd.Timestamp`) is stringified for comparison. This is critical — before the fix, the type mismatch meant every sweep looked "new" and the bot fired duplicates.

### 3.4 State — [state.py](live/state.py)

Persisted to `live/state.json`:

```json
{
  "traded_sweeps":  { "US500.cash": ["..."], "GBPJPY": ["..."], "UKOIL.cash": ["..."] },
  "positions":      { "12345": {symbol, entry, sl, tp, risk_price, direction, original_lots, sweep_time} },
  "pending_retests": { "67890": {symbol, sweep_time, expires_at, entry, sl, tp, ...} },
  "day": "YYYY-MM-DD",
  "day_start_balance": 100000.0,
  "day_pnl_pct": -1.23,
  "last_poll": "ISO timestamp"
}
```

**Migration:** If loading an old format where `traded_sweeps` was a bare list (single-symbol days), it's silently bucketed as `{"legacy": [...]}`.

**Windows lock retry:** `os.replace()` occasionally fails with `PermissionError` when Explorer or a text editor holds the file. `save()` retries 3× with exponential backoff.

**Trimming:** Keeps only the last 200 sweeps per symbol so the file doesn't grow forever.

### 3.5 Main event loop — [bot.py](live/bot.py)

Every `LOOP_SEC` (15s):

```
if market_closed → sleep 60s, continue

update_daily(state, anchor_symbol)     # account-wide day P&L

for each symbol in SYMBOLS:
    if !market_open(symbol) → skip
    reap_expired_retests(symbol, state)      # cancel dead limits
    promote_filled_retests(symbol, state)    # migrate filled limits into positions
    manager.manage(symbol, state, cfg, log)  # prune broker-closed positions
    try_enter(symbol, state)                  # look for new setup

trim_traded_sweeps(state)
save(state)
```

### 3.6 Entry flow — `_try_enter(symbol, state)`

1. **Concurrency check.** If `len(open_positions(symbol)) >= MAX_CONCURRENT_PER_SYMBOL` → return.
2. **Daily stop.** If today's P&L% ≤ `-DAILY_STOP_PCT` (-4%) → return.
3. **Fetch bars.** 400 M15 + 40 H4.
4. **Detect.** `detect_latest_setup(...)` with the symbol's traded-sweep dedup set.
5. **News check.** If entry-day date ∈ `NEWS_SET` → mark sweep traded (so we don't retry), return.
6. **Size.** `risk_dollars = balance × RISK_PCT/100`, then `broker.calc_lots(symbol, risk_dollars, sl_distance_price)`.
   - `calc_lots` uses MT5's `symbol_info.trade_tick_size` and `trade_tick_value` to compute $/tick and back out lots, rounded DOWN to `volume_step`.
7. **Place order:**
   - `"retest_pending"` → `broker.limit_order(...)` with 12h expiry. Sweep marked traded.
   - `"close"` or `"retest"` → `broker.market_order(...)`. Sweep marked traded + state saved IMMEDIATELY (before position lookup, to prevent double-fire).

### 3.7 Duplicate prevention — the critical guarantees

Three layers ensure ONE trade per setup:

1. **Per-symbol sweep dedup.** Stringified `sweep_time` added to `traded_sweeps[symbol]` at order-placement time.
2. **Save-before-lookup.** After `market_order` returns, we append to traded_sweeps and call `state.save()` BEFORE sleeping 1s and doing `open_positions()`. If the lookup ever takes >15s, the next poll won't see the sweep as untraded.
3. **Per-symbol concurrency cap.** Even if dedup somehow failed, `MAX_CONCURRENT_PER_SYMBOL = 1` blocks a second entry while any position is open.

### 3.8 Limit orders that never fill

Covered above but summarized here:
- Placed with MT5 native `expiration = now + 12h`. Broker auto-cancels.
- `_reap_expired_retests` runs every 15s: any pending order past its `expires_at` OR gone from broker's list → cleaned from state.
- Sweep is marked traded at PLACEMENT — expired limits do NOT re-fire.

### 3.9 Position management — [manager.py](live/manager.py)

Plain 3RR mode = minimal management. TP/SL live on broker side and close positions autonomously. `manage()` only does state cleanup: if a position we tracked is no longer in `broker.open_positions(symbol)` → prune from `state["positions"]`. Prevents state bloat.

_(Legacy partial+BE logic was disabled — see feedback in [config.py](live/config.py) where `PARTIAL_AT_R = None`, `BE_AT_R = None`.)_

### 3.10 Daily P&L stop

- On day change (`state["day"] != today`), snapshot `broker.account_balance()` as `day_start_balance`.
- On every loop, compute `day_pnl_pct = (equity - start) / start × 100`.
- If ≤ `-DAILY_STOP_PCT` (-4%) → skip all new entries for the rest of the day. Existing positions still run.
- Auto-resets at midnight (broker-server time, taken from the anchor symbol's last tick).

---

## 4. Prop-Firm Sizing Rationale

Backtest streak data over 4.5 years:

| Metric | Value |
|---|---:|
| Longest combined loss streak | 28 in a row |
| Worst 50-trade window | -34R (46 losses / 4 wins) |
| Worst 7-day calendar window | -27R |
| Worst 1-day | -13R |

At **0.33% risk per trade**:

| Scenario | Loss | vs FTMO limit |
|---|---:|---|
| Worst 1 day | -4.3% | Inside -5% daily |
| Worst 7 days | -8.9% | Inside -10% max DD (tight) |
| Worst 50-trade window | -11.2% | ~1% breach possible (once in 4.5 years) |

At **0.25% risk per trade**:

| Scenario | Loss | vs FTMO limit |
|---|---:|---|
| Worst 1 day | -3.3% | Inside -5% daily |
| Worst 7 days | -6.8% | Inside -10% max DD |
| Worst 50-trade window | -8.5% | Inside -10% max DD |

**0.33% is the recommended real-world risk for FTMO.** 0.25% is the fully-conservative variant.

If the user is re-buying blown accounts (bull-scaling strategy), 0.5-1% is fine because the ~16-30% drawdowns only appear once per run — most accounts survive.

---

## 5. Operational Checklist

### 5.1 Before first run

- [ ] MT5 terminal running, logged into paper account #1513873586
- [ ] All 3 symbols visible in Market Watch (US500.cash, GBPJPY, UKOIL.cash)
- [ ] `state.json` cleared or migrated (auto)
- [ ] `.venv` activated

### 5.2 Start the bot

```
.venv\Scripts\python.exe -m live.bot
```

### 5.3 Verify (smoke test, no orders)

```
.venv\Scripts\python.exe -m live.smoke_test
```

Expects: 3 symbols authenticate, 400 M15 + 40 H4 bars per symbol, "no fresh setup" or a valid setup with sizing.

### 5.4 What to watch in `bot.log`

- `bot starting · 3 symbols [...] · risk 0.33% · RR 3.0 · news_filter=True (128 blocked days)` — boot confirmation
- `── new day 2026-07-03 · start balance $99,156.79` — daily reset
- `[GBPJPY] SETUP · BUY · close · entry 214.9 · sl 214.8 · tp 215.2 · risk 0.100 · lots 0.15 · $327.22`
- `[GBPJPY] MARKET filled #12345 @ 214.905`
- `[GBPJPY] position #12345 closed (not on broker) — pruning` — TP or SL hit
- `[UKOIL.cash] setup found but 2026-07-15 is news-blocked — skipping` — NFP/CPI day

### 5.5 Kill switch

`Ctrl+C` triggers a clean shutdown with state save. Open positions continue to run on broker side (SL/TP already placed).

To close everything: use MT5 GUI or run a manual `close_all` script (not included — deliberately requires human action for safety).

---

## 6. File Map (deliverables)

### Strategy / research

- [main.py](main.py) — Base backtest utilities (fetch_data, find_all_fvgs)
- [eurusd_basic.py](eurusd_basic.py) — Core dry-basic simulator (single symbol, no news filter)
- [multi_asset_basic.py](multi_asset_basic.py) — 21-symbol screen, plain 3RR
- [combined_top3.py](combined_top3.py) — **Option A backtest** (top 3 combined, 3RR, news filter, disrespect SL)
- [analyze_top3.py](analyze_top3.py) — Streak distributions + equity/DD charts
- [combined_streak_deep.py](combined_streak_deep.py) — Rolling window & calendar-window worst cases
- [news_filter.py](news_filter.py) — Hardcoded red-folder day set

### Live bot

- [live/config.py](live/config.py) — All tunable parameters
- [live/broker.py](live/broker.py) — MT5 API wrappers
- [live/detector.py](live/detector.py) — Rolling signal detection
- [live/manager.py](live/manager.py) — Position cleanup
- [live/state.py](live/state.py) — JSON persistence
- [live/bot.py](live/bot.py) — Main loop
- [live/smoke_test.py](live/smoke_test.py) — Dry run
- [live/test_trade.py](live/test_trade.py) — End-to-end sanity: 0.01 buy + close in 5s

### Outputs

- [combined_top3_3R_trades.csv](combined_top3_3R_trades.csv) — 6,572-trade Option A log
- [strategies/top3_optionA/equity_and_drawdown.png](strategies/top3_optionA/equity_and_drawdown.png)
- [strategies/top3_optionA/loss_streak_histogram.png](strategies/top3_optionA/loss_streak_histogram.png)
- [strategies/top3_optionA/worst_streaks_highlighted.png](strategies/top3_optionA/worst_streaks_highlighted.png)

---

## 7. Known Gotchas

- **US500.cash and index-cash symbols close during broker maintenance windows** (typically 21:00-22:00 UTC weekdays, all Saturday). The bot handles this via `is_market_open()` — if the last tick is >5min old, skip that symbol for the loop.
- **`broker.calc_lots` returns 0 if the required size < `volume_min`.** For an $10k account at 0.33% risk on UKOIL with a small SL distance, this can happen. The bot logs `"setup found but computed lots=0 — skipping"` and marks the sweep traded to avoid retrying every 15s.
- **`sweep_time` timezone is broker-server time (not UTC).** The stringification is done consistently on both sides, so dedup works, but visual comparisons to news_filter dates use the tick's UTC date — this may be off by 2-3 hours on days that straddle midnight.
- **State file lock on Windows.** If you have `state.json` open in an editor or Explorer preview pane during a save attempt, expect a brief warning in log; the retry loop handles it.
