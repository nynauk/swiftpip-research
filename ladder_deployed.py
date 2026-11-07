"""
SCALING LADDER on the deployed bot's trade stream (curated 21 · M3+M5+M15 ·
1%/trade · 4/day cap · 2000-2026).

Model (single-track climb, conservative):
  start: Phase 1 of a $10k challenge (fee already paid).
  each rung: P1 needs +10%, P2 needs +5%, then FUNDED pays out at +10% cycles
  (80% split banked as cash). -5% rolling-24h daily / -10% overall blows the
  account (re-buy same rung, fee from cash).
  UPGRADE: at each payout, if cash covers the next rung's challenge fee, buy it
  and restart P1 at the bigger size (the smaller account is abandoned).
  At the top rung ($200k) payouts just accumulate.
Risk per trade = 1% of the CURRENT account size (the bot sizes off balance).
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUT = os.path.join("strategies", "all_pairs")
BG, FG = "#0d1117", "#e6edf3"
LADDER = [(10_000, 89.0), (25_000, 250.0), (50_000, 345.0),
          (100_000, 540.0), (200_000, 1_080.0)]
SPLIT = 0.80

df = pd.read_csv(os.path.join(OUT, "curated_trades.csv"))
df["entry_time"] = pd.to_datetime(df["entry_time"])
df["exit_time"] = pd.to_datetime(df["exit_time"])
df = df.sort_values("entry_time").reset_index(drop=True)
df["eday"] = df["entry_time"].dt.normalize()
capped = df.groupby("eday", group_keys=False).head(4)
cl = capped[capped.outcome != "open"].sort_values("exit_time")
stream = list(zip(cl["exit_time"], cl["pnl_r"].astype(float)))

rung, phase, bal = 0, "P1", 0.0
banked = 0.0; fees = LADDER[0][1]
danchor = None; dpnl = 0.0
payouts = {i: 0 for i in range(len(LADDER))}
blows = {i: 0 for i in range(len(LADDER))}
funded_at = {}; timeline = []
t0 = stream[0][0]

for t, r in stream:
    size = LADDER[rung][0]
    pnl = r * size * 0.01
    if danchor is not None and (t - danchor).total_seconds() > 86400:
        danchor = None; dpnl = 0.0
    bal += pnl
    if danchor is None:
        if pnl < 0: danchor = t; dpnl = pnl
    else:
        dpnl += pnl

    if dpnl <= -0.05 * size or bal <= -0.10 * size:           # blown
        blows[rung] += 1
        fee = LADDER[rung][1]
        banked -= fee; fees += fee                            # re-buy same rung
        phase, bal, danchor, dpnl = "P1", 0.0, None, 0.0
    elif phase == "P1" and bal >= 0.10 * size:
        phase, bal, danchor, dpnl = "P2", 0.0, None, 0.0
    elif phase == "P2" and bal >= 0.05 * size:
        phase, bal, danchor, dpnl = "FUNDED", 0.0, None, 0.0
        funded_at.setdefault(rung, t)
    elif phase == "FUNDED" and bal >= 0.10 * size:
        banked += SPLIT * 0.10 * size                         # payout (80% split)
        payouts[rung] += 1
        bal, danchor, dpnl = 0.0, None, 0.0
        if rung < len(LADDER) - 1 and banked >= LADDER[rung + 1][1]:
            fee = LADDER[rung + 1][1]
            banked -= fee; fees += fee
            rung += 1                                          # buy the next rung
            phase, bal = "P1", 0.0
    timeline.append((t, LADDER[rung][0], banked))

net = banked - 0.0                                             # fees already deducted from banked
print("=" * 92)
print("SCALING LADDER · deployed bot stream · $10k → $200k · P1+10% / P2+5% / funded +10% cycles")
print("=" * 92)
print(f"  {'rung':>8} {'funded on':>12} {'climb time':>11} {'payouts':>8} {'blown':>6}")
prev = t0
for i, (size, fee) in enumerate(LADDER):
    if i in funded_at:
        ft = funded_at[i]
        print(f"  ${size:>6,} {str(ft.date()):>12} {str((ft - prev).days) + 'd':>11} "
              f"{payouts[i]:>8} {blows[i]:>6}")
        prev = ft
    else:
        print(f"  ${size:>6,} {'never':>12} {'—':>11} {payouts[i]:>8} {blows[i]:>6}")
top_t = funded_at.get(len(LADDER) - 1)
if top_t:
    yrs_at_top = (stream[-1][0] - top_t).days / 365.25
    top_pay = payouts[len(LADDER) - 1]
    print(f"\n  climb $10k P1 → $200k funded: {(top_t - t0).days} days "
          f"({(top_t - t0).days / 30.44:.1f} months)")
    print(f"  at the top: {top_pay} × $16,000 payouts over {yrs_at_top:.1f} yrs "
          f"= {top_pay / yrs_at_top:.1f}/yr ≈ ${top_pay / yrs_at_top * 16000:,.0f}/yr")
print(f"\n  TOTAL banked (net of ALL fees ${fees:,.0f}): ${banked:,.0f}")
print(f"  total blowups: {sum(blows.values())}")
print(f"  NOTE: FTMO max allocation is $400k → running 2×$200k doubles the top-rung income.")

# chart
ts = [x[0] for x in timeline]; sz = [x[1] for x in timeline]; bk = [x[2] for x in timeline]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 8.5), facecolor=BG, sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.6]})
a1.step(ts, sz, color="#58a6ff", linewidth=1.4, where="post")
a1.set_ylabel("Account size ($)", color=FG)
a1.set_yscale("log"); a1.set_yticks([10e3, 25e3, 50e3, 100e3, 200e3])
a1.set_yticklabels(["$10k", "$25k", "$50k", "$100k", "$200k"])
a1.set_title("Scaling ladder · account size over time", color=FG, fontsize=11, fontweight="bold")
a2.plot(ts, bk, color="#3fb950", linewidth=1.4)
a2.axhline(0, color="#444c56", linewidth=0.8)
a2.set_ylabel("Banked cash, net of fees ($)", color=FG); a2.set_xlabel("Date", color=FG)
a2.set_title(f"Cumulative banked payouts · ${banked:,.0f} net · {sum(payouts.values())} payouts · "
             f"{sum(blows.values())} blowups", color=FG, fontsize=11, fontweight="bold")
a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M" if abs(x) >= 1e6 else f"${x/1e3:.0f}k"))
for ax in (a1, a2):
    ax.set_facecolor(BG); ax.tick_params(colors=FG); ax.grid(True, alpha=0.1, color=FG)
    ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
fig.patch.set_facecolor(BG)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "ladder_deployed.png"), dpi=140, facecolor=BG)
plt.close(fig)
print(f"\nSaved → {OUT}/ladder_deployed.png")
