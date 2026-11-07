# Live Bot — prev-4H sweep + 15M IFVG

Real-time deployable version of the dry-basic strategy.

## What it does
- Monitors the configured symbol on MT5
- Detects when the **previous 4H candle's high/low** is swept
- Waits for a 15M FVG to form and get inverted (IFVG)
- Places a trade — market entry if the disrespect candle is small, or a limit
  order at the FVG edge for the retest if it's too big (> 1.5× FVG width)
- **Dynamically sizes each trade** so the dollar loss on the stop is exactly
  `RISK_PCT` of the current account balance
- Manages the trade: at +1R takes 50% off + moves stop to break-even; runner
  targets +3R
- Runs a 24/7 event loop; polls every 15s; skips when market is closed
- Persists state to `state.json` so restarts don't lose open positions or
  re-trade the same sweep twice
- Logs to `bot.log`

## Files
- `config.py` — every knob (symbol, risk %, RR, partial, daily stop, etc.)
- `broker.py` — MT5 wrappers (bars, ticks, orders, sizing math)
- `detector.py` — sweep + FVG + IFVG detection (matches the backtest exactly)
- `manager.py` — partial-TP + break-even at +1R
- `state.py` — JSON persistence
- `bot.py` — main event loop

## Running

1. Open MT5 and log into your paper account.
2. From the **project root**:
```
.venv\Scripts\python.exe -m live.bot
```
3. Bot logs each poll to console + `live/bot.log`. State to `live/state.json`.
4. `Ctrl+C` to stop safely (state saved on exit).

## Configuration
Edit `live/config.py`:

| Key | Default | Meaning |
|---|---|---|
| `SYMBOL` | `EURUSD` | trading symbol |
| `RISK_PCT` | `0.75` | % of balance risked per trade |
| `RR` | `3.0` | take-profit at 3× risk |
| `PARTIAL_AT_R` | `1.0` | move to break-even + take partial here |
| `PARTIAL_FRAC` | `0.5` | fraction of position closed at +1R |
| `DAILY_STOP_PCT` | `4.0` | pause new entries after -4% today |
| `MAGIC` | `442015` | ID for bot's orders (leave unless colliding) |
| `LOOP_SEC` | `15` | polling interval |

## Position sizing (worked example)
Balance $10,000 · risk 0.75% → **$75** at risk.
Setup: buy EURUSD at 1.1000, stop 1.0990 → **10 pips** stop.
1 lot EURUSD → **$10 per pip** → 10 pips = **$100/lot loss**.
Lot size = $75 / $100 = **0.75 lots** (rounded to broker volume step).
Every trade is sized this way — same % risk regardless of stop distance.

## Safety notes
- The bot **only touches trades with its `MAGIC` number**. Other manual trades
  on the same account are untouched.
- `MAX_CONCURRENT = 1` — one position on this symbol at a time.
- If bot restarts mid-trade, it re-adopts open positions from the broker into
  `state.json` on the next poll — partial-TP will still fire correctly.
- Retest limit orders auto-expire after `RETR_WINDOW_M15 * 15 min` (default
  12 hours) and are cancelled if not filled.

## Verifying it's working
- Console prints every setup detected + every order/action.
- `bot.log` keeps the full history.
- Check `state.json` to see what the bot thinks its open positions are.
- MT5 terminal → View → Toolbox → Trade tab to see actual positions.

## Known limits
- No news blackout (would need a live economic calendar feed).
- Uses BROKER time — sessions/killzones would need broker-TZ awareness.
- Does not restart on hard MT5 disconnect — wrap in a `while True` shell loop
  or a Windows scheduled task for auto-restart.
