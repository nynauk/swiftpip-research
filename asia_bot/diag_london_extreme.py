"""
Diagnostic: is the London-extreme guard actually doing anything?

For every confirmed 1M CISD on EURUSD it checks, independently of the detector:
  A) does the London session high/low (measured UP TO the confirmation candle —
     i.e. causally) already sit inside the CISD window [series_start, confirm]?
  B) does the delivery series start BEFORE the sweep (leg not part of the
     post-sweep manipulation)?
  C) is the manipulation extreme still the London extreme at the END of the
     session (07:00-10:00)?  <-- this one needs HINDSIGHT and is NOT tradeable.

    python -m asia_bot.diag_london_extreme
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asia_bot import config as cfg
from asia_bot import detector as det

SYMBOL = "EURUSD"
HD = os.path.join("ext_data", "HD_EURUSD_1min_2004-2024.pkl")
OFFSET_H = -5
WALK_HOURS = 60


def main():
    df = pd.read_pickle(HD)
    utc1_all = det.utc1_index(df, OFFSET_H)
    days = pd.DatetimeIndex(utc1_all).normalize().unique()

    n_conf = 0
    a_inside = 0                 # session extreme (causal) inside CISD window
    b_before_sweep = 0           # delivery start before the sweep
    c_still_extreme = 0          # extreme survives to end of London (hindsight)
    c_checked = 0

    for day in days:
        d = day.date()
        if d.weekday() >= 5:
            continue
        sb = det.session_bounds_utc1(d)
        a = int(np.searchsorted(utc1_all, sb["asia_start"], "left"))
        b = int(np.searchsorted(utc1_all, sb["london_end"] + pd.Timedelta(hours=WALK_HOURS), "right"))
        if b - a < 200:
            continue
        sl_df = df.iloc[a:b]
        s = det.analyze_day(sl_df, SYMBOL, OFFSET_H, session_date=d)
        if s is None or s.stage != "confirmed":
            continue
        n_conf += 1

        dfc = sl_df.iloc[:-1]
        t1 = det.utc1_index(dfc, OFFSET_H)
        h = dfc["high"].values; l = dfc["low"].values
        lon = np.where((t1 >= sb["london_start"]) & (t1 < sb["london_end"]))[0]
        if not len(lon):
            continue
        lon0, lon_end = int(lon[0]), int(lon[-1])
        k = s.confirm_i

        # A) causal: session extreme up to confirmation, inside [start, k]?
        if s.direction == "sell":
            ext_i = lon0 + int(np.argmax(h[lon0:k + 1]))
        else:
            ext_i = lon0 + int(np.argmin(l[lon0:k + 1]))
        if s.series_start_i <= ext_i <= k:
            a_inside += 1

        # B) delivery leg starts before the sweep?
        if s.series_start_i < s.sweep_i:
            b_before_sweep += 1

        # C) hindsight: is that extreme still the extreme at London close?
        if lon_end > k:
            c_checked += 1
            if s.direction == "sell":
                full = h[lon0:lon_end + 1].max()
                if s.sweep_extreme >= full - 1e-9:
                    c_still_extreme += 1
            else:
                full = l[lon0:lon_end + 1].min()
                if s.sweep_extreme <= full + 1e-9:
                    c_still_extreme += 1

    print("=" * 92)
    print(f"LONDON-EXTREME DIAGNOSTIC · {SYMBOL} 1M · HistData 2004-2024")
    print("=" * 92)
    print(f"  confirmed CISDs                                  : {n_conf}")
    print(f"  A) session extreme (causal) INSIDE the CISD window: {a_inside} "
          f"({a_inside/max(n_conf,1)*100:.1f}%)  <- what the guard tests")
    print(f"  B) delivery leg starts BEFORE the sweep           : {b_before_sweep} "
          f"({b_before_sweep/max(n_conf,1)*100:.1f}%)")
    print(f"  C) extreme SURVIVES to London close (hindsight)   : {c_still_extreme}/{c_checked} "
          f"({c_still_extreme/max(c_checked,1)*100:.1f}%)  <- NOT causally knowable")
    print()
    if a_inside == n_conf:
        print("  => The guard NEVER rejects: by construction the CISD always anchors to the")
        print("     running session extreme, so 'extreme inside the CISD window' is always true.")
        print("     Enforcing it causally changes ZERO trades.")
    print(f"  => Only {c_still_extreme/max(c_checked,1)*100:.0f}% of CISDs mark the session's FINAL high/low.")
    print("     Filtering on THAT would need to see the future — it is a lookahead filter.")


if __name__ == "__main__":
    main()
