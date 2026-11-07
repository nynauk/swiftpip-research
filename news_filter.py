"""
Red-folder news filter.

Returns a set of dates (YYYY-MM-DD) on which we BLOCK trading because a
high-impact US event is scheduled. These events move USD majors and all
US indices, which covers everything in our `winners` symbol list.

Sources:
  • NFP — first Friday of every month (algorithmic)
  • FOMC rate decisions — hardcoded from the Fed schedule
  • CPI / PPI / Retail Sales — hardcoded from BLS schedule (the events
    consistently classified red folder on ForexFactory)
  • ECB rate decisions — hardcoded from ECB schedule (EU events also
    drive EURUSD / GBPUSD around 12:45 GMT)

Extend the hardcoded lists below as future dates are published.
"""
from datetime import date, datetime, timedelta


# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded high-impact event dates (2022-2026)
# ─────────────────────────────────────────────────────────────────────────────

# FOMC rate decisions (Fed funds rate, 8x/year, 18:00 UTC Wed)
FOMC_DATES = [
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]

# US CPI release dates (12:30/13:30 UTC, 2nd Tuesday/Wednesday of month)
US_CPI_DATES = [
    "2022-01-12", "2022-02-10", "2022-03-10", "2022-04-12",
    "2022-05-11", "2022-06-10", "2022-07-13", "2022-08-10",
    "2022-09-13", "2022-10-13", "2022-11-10", "2022-12-13",
    "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12",
    "2023-05-10", "2023-06-13", "2023-07-12", "2023-08-10",
    "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12",
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10",
    "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
    "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12",
    "2025-09-11", "2025-10-15", "2025-11-13", "2025-12-10",
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-15",
    "2026-05-13", "2026-06-10",
]

# US Retail Sales (red folder, 12:30 UTC mid-month)
US_RETAIL_DATES = [
    # 2022 - retail dates ~14-16 of month
    "2022-01-14", "2022-02-16", "2022-03-16", "2022-04-14",
    "2022-05-17", "2022-06-15", "2022-07-15", "2022-08-17",
    "2022-09-15", "2022-10-14", "2022-11-16", "2022-12-15",
    "2023-01-18", "2023-02-15", "2023-03-15", "2023-04-14",
    "2023-05-16", "2023-06-15", "2023-07-18", "2023-08-15",
    "2023-09-14", "2023-10-17", "2023-11-15", "2023-12-14",
    "2024-01-17", "2024-02-15", "2024-03-14", "2024-04-15",
    "2024-05-15", "2024-06-18", "2024-07-16", "2024-08-15",
    "2024-09-17", "2024-10-17", "2024-11-15", "2024-12-17",
    "2025-01-16", "2025-02-14", "2025-03-17", "2025-04-16",
    "2025-05-15", "2025-06-17", "2025-07-17", "2025-08-15",
    "2025-09-16", "2025-10-16", "2025-11-14", "2025-12-17",
    "2026-01-16", "2026-02-17", "2026-03-17", "2026-04-15",
    "2026-05-15", "2026-06-16",
]

# ECB rate decisions (12:15 UTC, ~monthly)
ECB_DATES = [
    "2022-02-03", "2022-03-10", "2022-04-14", "2022-06-09",
    "2022-07-21", "2022-09-08", "2022-10-27", "2022-12-15",
    "2023-02-02", "2023-03-16", "2023-05-04", "2023-06-15",
    "2023-07-27", "2023-09-14", "2023-10-26", "2023-12-14",
    "2024-01-25", "2024-03-07", "2024-04-11", "2024-06-06",
    "2024-07-18", "2024-09-12", "2024-10-17", "2024-12-12",
    "2025-01-30", "2025-03-06", "2025-04-17", "2025-06-05",
    "2025-07-24", "2025-09-11", "2025-10-30", "2025-12-18",
    "2026-01-29", "2026-03-12", "2026-04-30", "2026-06-04",
]


def first_friday_of_month(year: int, month: int) -> date:
    """NFP day — first Friday of the given month."""
    d = date(year, month, 1)
    # weekday(): Mon=0 ... Fri=4 ... Sun=6
    days_until_fri = (4 - d.weekday()) % 7
    return d + timedelta(days=days_until_fri)


# ── H4-window release times (BROKER TIME, ~UTC+2/+3 on FTMO servers) ──
# H4 candle windows on the broker align to hours 0/4/8/12/16/20.
# Most releases (12:30 UTC → 14:30 broker) land in the 12:00 broker window.
# FOMC (18:00 UTC → 20:00 broker) lands in the 20:00 broker window.
H4_MOST     = 12    # NFP, CPI, Retail Sales, ECB — 12:00-16:00 broker window
H4_FOMC     = 20    # FOMC — 20:00-00:00 broker window


def build_news_block_set(start: date, end: date) -> set:
    """Return the set of `date` objects on which we BLOCK trading (legacy full-day)."""
    blocked = set()
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        nfp = first_friday_of_month(y, m)
        if start <= nfp <= end:
            blocked.add(nfp)
        if m == 12: y, m = y + 1, 1
        else: m += 1
    for s in FOMC_DATES + US_CPI_DATES + US_RETAIL_DATES + ECB_DATES:
        d = datetime.strptime(s, "%Y-%m-%d").date()
        if start <= d <= end:
            blocked.add(d)
    return blocked


def build_news_block_windows(start: date, end: date) -> set:
    """
    Return the set of (date, h4_hour) tuples where trading is blocked.
    Only the specific 4H candle containing the news release is blocked,
    not the whole day. That preserves trading opportunity while still
    dodging the actual volatility spike.
    """
    blocked = set()

    # NFP — first Friday of every month, 12:00 broker window (12:30 UTC release)
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        nfp = first_friday_of_month(y, m)
        if start <= nfp <= end:
            blocked.add((nfp, H4_MOST))
        if m == 12: y, m = y + 1, 1
        else: m += 1

    # 12:00 broker window releases
    for s in US_CPI_DATES + US_RETAIL_DATES + ECB_DATES:
        d = datetime.strptime(s, "%Y-%m-%d").date()
        if start <= d <= end:
            blocked.add((d, H4_MOST))

    # 20:00 broker window releases (FOMC)
    for s in FOMC_DATES:
        d = datetime.strptime(s, "%Y-%m-%d").date()
        if start <= d <= end:
            blocked.add((d, H4_FOMC))

    return blocked


def is_blocked(ts, blocked_set) -> bool:
    """True if the timestamp's calendar date is in the blocked set (legacy)."""
    import pandas as pd
    d = pd.Timestamp(ts).date()
    return d in blocked_set


def is_blocked_h4(ts, blocked_windows) -> bool:
    """True if the timestamp's 4H candle window is blocked."""
    import pandas as pd
    t = pd.Timestamp(ts)
    d = t.date()
    h4 = (t.hour // 4) * 4
    return (d, h4) in blocked_windows
