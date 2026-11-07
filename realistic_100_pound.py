"""
Reality check: what happens if you deposit £100 and run Option A?

Applies:
  · Spread on entry (fill at ASK for buys, BID for sells)
  · Slippage on entry + stop-out (2 ticks each direction typical)
  · Broker minimum lot enforcement (volume_min per symbol)
  · Round-down lot sizing to volume_step
  · Trade SKIP if computed lot size < volume_min (can't afford minimum)
  · No FTMO limits (this is your own broker, not a prop firm)

Uses the actual trade log from combined_top3.py.
"""
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Read trade log
CSV = "combined_top3_3R_trades.csv"
df = pd.read_csv(CSV, parse_dates=["sweep_time", "entry_time", "exit_time"])
df = df[df["outcome"] != "open"].sort_values("entry_time").reset_index(drop=True)
print(f"Loaded {len(df)} trades from {df['entry_time'].min()} → {df['entry_time'].max()}\n")

# Symbol specs (from live MT5 broker; see smoke_test output)
# Slippage: assume a *decent* retail broker on major sessions
SPECS = {
    "US500.cash": {
        "tick_size":   0.01,
        "tick_value":  0.01,       # $ per 0.01 point per 1.0 lot
        "spread":      0.55,       # ~0.55 points typical
        "slip":        0.05,       # per side (entry & stop) — 0.05 point
        "volume_min":  0.1,
        "volume_step": 0.1,
    },
    "GBPJPY": {
        "tick_size":   0.001,
        "tick_value":  0.62,       # $ per 0.001 pip per 1.0 lot (rough)
        "spread":      0.021,      # 2.1 pips typical
        "slip":        0.003,      # 0.3 pip per side
        "volume_min":  0.01,
        "volume_step": 0.01,
    },
    "UKOIL.cash": {
        "tick_size":   0.001,
        "tick_value":  0.10,       # $ per 0.001 per 1.0 lot
        "spread":      0.063,      # 6.3 cents typical
        "slip":        0.010,      # 1 cent per side
        "volume_min":  0.1,
        "volume_step": 0.1,
    },
}

GBP_USD = 1.27       # rough spot rate; £100 → ~$127


def simulate(start_gbp: float, risk_pct: float, exclude_symbols=(), verbose=False):
    """
    Realistic cost model:
      · Entry fills at ASK (spread already accounted for) — no entry slip
      · TP is a limit order — fills at TP exactly (no adverse slip)
      · SL is a stop order — fills at SL - slip (adverse slip on losses only)

    Per trade:
      · Win  =  3 × close_sl_dist  -  spread
      · Loss = -(close_sl_dist  +  spread + slip)
    """
    balance = start_gbp * GBP_USD
    initial_balance = balance
    blown = False; blow_time = None
    executed = skipped = 0
    per_taken   = {s: 0 for s in SPECS}
    per_skipped = {s: 0 for s in SPECS}
    per_pnl     = {s: 0.0 for s in SPECS}
    eq = [balance]

    for _, tr in df.iterrows():
        sym = tr["symbol"]
        if sym in exclude_symbols or sym not in SPECS:
            continue
        s = SPECS[sym]
        close_sl = abs(tr["entry"] - tr["sl"])

        # Real risk on execution: sized to what we'd actually lose
        real_loss_dist = close_sl + s["spread"] + s["slip"]
        dollars_per_lot = real_loss_dist / s["tick_size"] * s["tick_value"]
        if dollars_per_lot <= 0:
            continue

        target_risk = balance * risk_pct / 100
        raw = target_risk / dollars_per_lot
        lots = math.floor(raw / s["volume_step"]) * s["volume_step"]
        lots = round(lots, 4)
        if lots < s["volume_min"]:
            skipped += 1; per_skipped[sym] += 1
            continue

        if tr["outcome"] == "win":
            win_dist = 3 * close_sl - s["spread"]
            pnl = lots * max(win_dist, 0) / s["tick_size"] * s["tick_value"]
        else:
            pnl = -lots * real_loss_dist / s["tick_size"] * s["tick_value"]

        balance += pnl
        eq.append(balance)
        executed += 1
        per_taken[sym] += 1
        per_pnl[sym]   += pnl
        if balance <= 5:
            blown = True; blow_time = tr["entry_time"]; break

    return {
        "start": initial_balance, "end": balance,
        "blown": blown, "blow_time": blow_time,
        "executed": executed, "skipped": skipped,
        "eq": np.array(eq),
        "per_sym_taken": per_taken, "per_sym_skipped": per_skipped,
        "per_sym_pnl": per_pnl,
    }


def report(label: str, r: dict):
    end_gbp = r["end"] / GBP_USD
    start_gbp = r["start"] / GBP_USD
    ret_pct = (r["end"] - r["start"]) / r["start"] * 100
    peak = np.maximum.accumulate(r["eq"])
    dd = ((r["eq"] - peak) / peak * 100).min() if len(r["eq"]) > 1 else 0
    print(f"\n── {label} ──")
    if r["blown"]:
        print(f"  ACCOUNT BLOWN at {r['blow_time']}  ·  balance ${r['end']:.2f}")
    print(f"  Start   :  £{start_gbp:>10,.2f}  (${r['start']:>10,.2f})")
    print(f"  End     :  £{end_gbp:>10,.2f}  (${r['end']:>10,.2f})   [{ret_pct:+.1f}%]")
    print(f"  Max DD  :  {dd:.1f}%")
    print(f"  Trades  :  taken {r['executed']}  ·  skipped {r['skipped']} "
          f"(couldn't afford minimum lot)")
    print(f"  Per-symbol:")
    for sym in SPECS:
        pnl_gbp = r["per_sym_pnl"][sym] / GBP_USD
        print(f"    {sym:12}  taken {r['per_sym_taken'][sym]:>4}  "
              f"skipped {r['per_sym_skipped'][sym]:>4}  pnl £{pnl_gbp:+,.2f}")


# ── Run at multiple risk levels ──
print("=" * 80)
print(f"REALISTIC £100 SIMULATION  ·  after spread + slippage + volume_min")
print(f"  GBP/USD assumed = {GBP_USD}  ·  effective start = ${100 * GBP_USD:.2f}")
print("=" * 80)

results = {}
for risk in [0.5, 1.0, 2.0, 3.0, 5.0]:
    r = simulate(100.0, risk)
    results[risk] = r
    report(f"Risk = {risk}%  ·  £100 start", r)

# ── SL-cost diagnostic: after-cost effective RR ──
print("\n" + "=" * 80)
print("SL-COST DIAGNOSTIC — effective RR after realistic costs (no entry slip, "
      "TP fills at target, SL slips by slip amount)")
print("=" * 80)
for sym in SPECS:
    s = SPECS[sym]
    sub = df[df["symbol"] == sym]
    close_sl = (sub["entry"] - sub["sl"]).abs()
    win_cost  = s["spread"]                # winners pay half-spread on entry approx
    loss_cost = s["spread"] + s["slip"]    # losses pay spread + slip
    eff_rr = (3 * close_sl - win_cost) / (close_sl + loss_cost)
    # Breakeven WR needed
    be_wr = 1 / (1 + eff_rr) * 100
    actual_wr = (sub["outcome"] == "win").mean() * 100
    edge = (actual_wr - be_wr)
    print(f"  {sym:12}  N={len(sub):>4}  "
          f"eff_RR median={eff_rr.median():.2f}  "
          f"BE_WR={be_wr.median():.1f}%  actual_WR={actual_wr:.1f}%  "
          f"edge={edge.median() if hasattr(edge,'median') else (actual_wr-be_wr.median()):+.1f}pp")

# ── Also run a "no UKOIL" variant of everything ──
print("\n" + "=" * 80)
print("RE-RUN WITH UKOIL DROPPED (only US500 + GBPJPY)")
print("=" * 80)
results_no_ukoil = {}
for risk in [0.5, 1.0, 2.0, 3.0]:
    r = simulate(100.0, risk, exclude_symbols=("UKOIL.cash",))
    results_no_ukoil[risk] = r
    report(f"NO-UKOIL · Risk = {risk}%  ·  £100 start", r)

print("\n── NO-UKOIL · same 1% risk, different starts ──")
for start in [100, 500, 1000, 5000, 10000, 50000]:
    r = simulate(start, 1.0, exclude_symbols=("UKOIL.cash",))
    end_gbp = r["end"] / GBP_USD
    status = "BLOWN" if r["blown"] else "survived"
    ret = (r["end"] - r["start"]) / r["start"] * 100
    print(f"  £{start:>6,} → £{end_gbp:>15,.0f}  ({ret:+8.1f}%)  "
          f"exec {r['executed']:>4}  skip {r['skipped']:>4}  {status}")

# ── US500 ONLY (the only symbol with true positive edge) ──
print("\n" + "=" * 80)
print("US500 ONLY (drop GBPJPY & UKOIL) — the only symbol with positive edge")
print("=" * 80)
for start in [100, 500, 1000, 5000, 10000]:
    for risk in [0.5, 1.0, 2.0]:
        r = simulate(start, risk, exclude_symbols=("UKOIL.cash", "GBPJPY"))
        end_gbp = r["end"] / GBP_USD
        status = "BLOWN" if r["blown"] else ""
        ret = (r["end"] - r["start"]) / r["start"] * 100
        print(f"  £{start:>6,} @ {risk}% → £{end_gbp:>12,.0f}  ({ret:+8.1f}%)  "
              f"exec {r['executed']:>4}  skip {r['skipped']:>4}  {status}")
    print()

# ── Filter: only take trades where SL distance is wide enough for edge to survive ──
# For US500: need close_sl > 3x spread (i.e. spread < 1/3 of SL distance) for eff RR > 2.5
print("\n" + "=" * 80)
print("US500 ONLY + WIDE-SL FILTER (skip trades where SL < 3× spread)")
print("=" * 80)


def simulate_filtered(start_gbp: float, risk_pct: float, min_sl_multiple: float,
                      only_symbols=("US500.cash",)):
    balance = start_gbp * GBP_USD
    initial_balance = balance
    blown = False; blow_time = None
    exec_n = skip_n = filt_n = 0
    eq = [balance]
    for _, tr in df.iterrows():
        sym = tr["symbol"]
        if sym not in only_symbols:
            continue
        s = SPECS[sym]
        close_sl = abs(tr["entry"] - tr["sl"])
        # SL-width filter: require enough room for edge to survive costs
        if close_sl < min_sl_multiple * s["spread"]:
            filt_n += 1; continue
        real_loss_dist = close_sl + s["spread"] + s["slip"]
        dpl = real_loss_dist / s["tick_size"] * s["tick_value"]
        target = balance * risk_pct / 100
        raw = target / dpl
        lots = math.floor(raw / s["volume_step"]) * s["volume_step"]
        lots = round(lots, 4)
        if lots < s["volume_min"]:
            skip_n += 1; continue
        if tr["outcome"] == "win":
            pnl = lots * max(3 * close_sl - s["spread"], 0) / s["tick_size"] * s["tick_value"]
        else:
            pnl = -lots * real_loss_dist / s["tick_size"] * s["tick_value"]
        balance += pnl
        eq.append(balance); exec_n += 1
        if balance <= 5:
            blown = True; blow_time = tr["entry_time"]; break
    return dict(start=initial_balance, end=balance, blown=blown, exec=exec_n,
                skip=skip_n, filtered=filt_n, eq=np.array(eq))


for start in [100, 500, 1000, 5000, 10000]:
    for risk in [0.5, 1.0, 2.0]:
        r = simulate_filtered(start, risk, min_sl_multiple=3.0)
        end_gbp = r["end"] / GBP_USD
        ret = (r["end"] - r["start"]) / r["start"] * 100
        status = "BLOWN" if r["blown"] else ""
        print(f"  £{start:>6,} @ {risk}% → £{end_gbp:>12,.0f}  ({ret:+8.1f}%)  "
              f"exec {r['exec']:>4}  filtered {r['filtered']:>4}  skipped {r['skip']:>4}  {status}")
    print()

# ── All 3 symbols with wide-SL filter (min SL > 3× spread) ──
print("\n" + "=" * 80)
print("ALL 3 SYMBOLS + WIDE-SL FILTER (skip trades where SL < 3× spread each)")
print("=" * 80)
for start in [100, 500, 1000, 5000, 10000]:
    for risk in [0.5, 1.0, 2.0]:
        r = simulate_filtered(start, risk, min_sl_multiple=3.0,
                              only_symbols=("US500.cash", "GBPJPY", "UKOIL.cash"))
        end_gbp = r["end"] / GBP_USD
        ret = (r["end"] - r["start"]) / r["start"] * 100
        status = "BLOWN" if r["blown"] else ""
        print(f"  £{start:>6,} @ {risk}% → £{end_gbp:>12,.0f}  ({ret:+8.1f}%)  "
              f"exec {r['exec']:>4}  filtered {r['filtered']:>4}  skipped {r['skip']:>4}  {status}")
    print()

# ── Per-symbol effect of the wide-SL filter ──
print("\n── HOW MANY TRADES SURVIVE THE 3× SPREAD FILTER? ──")
for sym in SPECS:
    s = SPECS[sym]
    sub = df[df["symbol"] == sym].copy()
    sub["close_sl"] = (sub["entry"] - sub["sl"]).abs()
    kept = sub[sub["close_sl"] >= 3 * s["spread"]]
    dropped = sub[sub["close_sl"] < 3 * s["spread"]]
    if len(kept):
        kept_wr = (kept["outcome"] == "win").mean() * 100
        kept_r = kept["pnl_r"].sum()
    else:
        kept_wr, kept_r = 0, 0
    if len(dropped):
        drop_wr = (dropped["outcome"] == "win").mean() * 100
        drop_r = dropped["pnl_r"].sum()
    else:
        drop_wr, drop_r = 0, 0
    print(f"  {sym:12}  total {len(sub):>4}  kept {len(kept):>4} (WR {kept_wr:.0f}% totR {kept_r:+.0f})  "
          f"dropped {len(dropped):>4} (WR {drop_wr:.0f}% totR {drop_r:+.0f})")

# ── Also test bigger deposits so you can see when things start working ──
print("\n" + "=" * 80)
print("SAME 1% RISK — DIFFERENT STARTING DEPOSITS")
print("=" * 80)
for start in [100, 250, 500, 1000, 2500, 5000, 10000]:
    r = simulate(start, 1.0)
    end_gbp = r["end"] / GBP_USD
    status = "BLOWN" if r["blown"] else "survived"
    print(f"  £{start:>6,} start → £{end_gbp:>15,.0f}   "
          f"executed {r['executed']:>4}  skipped {r['skipped']:>4}  {status}")

# ── Equity chart ──
BG, FG = "#0d1117", "#e6edf3"
COLORS = ["#3fb950", "#d29922", "#58a6ff", "#f85149", "#a371f7"]

fig, ax = plt.subplots(figsize=(15, 8), facecolor=BG)
ax.set_facecolor(BG)
for i, (risk, r) in enumerate(results.items()):
    if r["executed"] == 0:
        continue
    eq_gbp = r["eq"] / GBP_USD
    label = f"{risk}% risk  →  £{eq_gbp[-1]:,.0f}"
    if r["blown"]:
        label += "  (BLOWN)"
    ax.plot(range(len(eq_gbp)), eq_gbp, color=COLORS[i % len(COLORS)],
            linewidth=1.4, label=label)

ax.axhline(100, color="#444c56", linewidth=0.8, linestyle="--", label="£100 start")
ax.set_yscale("log")
ax.set_title("£100 deposit  ·  Option A after spread + slippage + volume_min",
             color=FG, fontsize=13, fontweight="bold", pad=10)
ax.set_ylabel("Balance (£)", color=FG)
ax.set_xlabel("Executed trade #", color=FG)
ax.legend(facecolor="#161b22", edgecolor="#444c56", labelcolor=FG, loc="lower right")
ax.tick_params(colors=FG); ax.grid(True, alpha=0.10, color=FG, which="both")
for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
plt.tight_layout()
plt.savefig("strategies/top3_optionA/realistic_100gbp.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("\n → chart saved: strategies/top3_optionA/realistic_100gbp.png")
