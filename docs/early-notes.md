# ICT Strategy Backtesting Suite

Mechanical backtests of ICT-style strategies on MT5 forex/index data.
Every trade is simulated candle-by-candle (SL checked before TP each bar,
no look-ahead). Results are compounded at 1% risk per trade on a $10k account.

## Folder layout

All Python modules live in the project root (they import each other flatly:
`import main as bt`, `import news_filter as nf`, `import plot_utils as pu`).
They are grouped below by ROLE for readability. Data, cache, and outputs are
in dedicated folders.

```
ftmotest/
├── README.md              ← this file
├── data/                  External price CSVs (EURUSD_M1.csv = UTC 1-min)
├── data_cache/            Pickled MT5 OHLC (H4/M15/M5/M1 per symbol) — instant re-runs
├── strategies/<name>/     Per-strategy OUTPUTS (trades.csv, equity.png, sample charts)
└── archive/               Superseded/legacy output files
```

### Engine (shared, root)
| File | Role |
|---|---|
| `main.py` | Core engine: data fetch/cache, swing & sweep detection, FVG detection, IFVG backtest loop, stats, plotting. Imported as `bt`. |
| `news_filter.py` | Red-folder calendar (NFP/FOMC/CPI/ECB/Retail). |
| `plot_utils.py` | Shared candlestick trade chart + equity curve. |

### Strategy scripts (root) → outputs in `strategies/<name>/`
| File | Strategy | Output folder |
|---|---|---|
| `ema_dol_strategy.py` ★ | EMA9/21 cross + FVG entry + Draw-on-Liquidity TP | `strategies/ema_dol/` |
| `silver_bullet.py` | ICT Silver Bullet (sweep→FVG→retest, killzones) | `strategies/silver_bullet/` |
| `cisd_strategy.py` | PDH/PDL sweep + 5M IFVG + CISD | `strategies/cisd/` |
| `asia_london.py` | Asia range → HTF 4H FVG → London reversal | `strategies/asia_london/` |
| `asia_1m_cisd.py` | Asia sweep + 1M IFVG/CISD (uses `data/EURUSD_M1.csv`) | `strategies/asia_1m_cisd/` |
| `h4_swing_displacement.py` | 4H swing sweep + IFVG, SL at displacement candle | `strategies/h4_displacement/` |
| `confluence_sweep.py` | 4H sweep + confluent prior-level / 4H-FVG + IFVG+CISD (NEW) | `strategies/confluence_sweep/` |
| `multi_symbol.py` | Sweep+IFVG family across many symbols | `strategies/sweep_ifvg/` |

### Analysis / tooling (root)
| File | Role |
|---|---|
| `master_compare.py` | Runs every strategy+tuning → single leaderboard (`strategies/master/`) |
| `iterate.py` | Single-symbol parameter sweeper (sweep+IFVG) |
| `analyze_ema_dol.py` | Deep stats on the EMA+DOL trade log |
| `report_top2.py` | Biggest-win + most-recent plots + full stats (top 2) |
| `plot_biggest_win.py` | Renders the EMA+DOL biggest win with EMAs |
| `plot_h4_displacement_sample.py` | Renders H4-displacement sample trades |

## Results leaderboard (7 FX majors, 4yr where data allows, RR≥2, news-blocked)

| Strategy | Trades | WR | Total R | Profit Factor | Max DD | Notes |
|---|---:|---:|---:|---:|---:|---|
| **EMA9/21 + DOL** ★ | 542 | 41.5% | +409.6R | 2.29 | -8.6% | best all-round; DOL TP is the alpha |
| H4-swing 24h + IFVG (2R) | 832 | 41.0% | +192R | 1.39 | -16.8% | high frequency |
| Silver Bullet (NY killzone) | 289 | 39.0% | +49R | 1.28 | -10.9% | classic ICT |
| H4-swing + displacement SL | 478 | 42.4% | +130R | 1.47 | -22.8% | wider SL |
| Asia sweep + CISD (1M, EURUSD) | 112 | 39.3% | +20R | 1.29 | -13.4% | 6.5-month sample only |
| CISD (5M) | 74 | 29.4% | -8R | 0.83 | -10.7% | discard |

## Timezone notes
- MT5 broker feed = **GMT+3** (EEST). Killzone hours in scripts are broker time
  unless noted (e.g. Silver Bullet NY Open = 16:00-17:00 broker).
- `data/EURUSD_M1.csv` = **UTC**. Asia (00-07) / London (07-10) in that file are
  the textbook GMT session hours.

## How to run
Always run from the project root so relative paths (`data/`, `strategies/`) resolve:
```
cd ftmotest
.venv/Scripts/python.exe ema_dol_strategy.py
.venv/Scripts/python.exe confluence_sweep.py
.venv/Scripts/python.exe master_compare.py
```
Outputs land in `strategies/<name>/`.
