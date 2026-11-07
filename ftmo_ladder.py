"""
FTMO account-ladder progression on the final EMA 7/18 cross-back strategy.

Start with a $10k Challenge -> pass Phase 1 (+10%) -> Phase 2 (+5%) -> Funded.
Take a payout (80% split), then use it to scale up to the next account size.
Ladder: $10k -> $25k -> $50k -> $100k -> $200k. EVERY size has its own Phase 1
and Phase 2. Rules per account: -5% daily loss, -10% overall loss = blown (retry).

The trade stream is the realistic cross-back backtest (per-pair average spreads
already applied to every trade). 1% risk/trade. Trades consumed in exit order.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import MetaTrader5 as mt5

import main as bt
import frequency_tune as fq
import test_cross_exit as tc
import test_gradient_exit as tg   # EMA-gradient exit (close before the cross)

EMA = (7, 18)
LADDER = [10_000, 25_000, 50_000, 100_000, 200_000]
FEES = {10_000: 89, 25_000: 178, 50_000: 289, 100_000: 540, 200_000: 1080}
RISK_PCT = 0.01
SPLIT = 0.80
P1, P2, DAILY, OVER = 0.10, 0.05, 0.05, 0.10
PAYOUTS_TO_SCALE = 1          # take 1 funded payout, then scale to the next size
OUT = os.path.join("strategies", "ema718_final")


def load(s):
    p = os.path.join("data_cache", f"{s}_M15.pkl")
    try:
        return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
    except Exception:
        if os.path.exists(p):
            os.remove(p)
        return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")


def run_ladder(trades):
    """trades: DataFrame sorted by exit_time with columns exit_time, pnl_r."""
    t = pd.to_datetime(trades["exit_time"]).tolist()
    R = trades["pnl_r"].tolist()

    si = 0; S = LADDER[si]; phase = "P1"; bal = S
    danchor = None; dpnl = 0.0
    banked = 0.0; fees = FEES[S]                 # first challenge bought
    funded_payouts = 0
    per = {s: dict(p1=None, p2=None, funded=None, payouts=0, blown=0, trades=0) for s in LADDER}
    events = []           # (time, type, size)
    trace_t = [t[0]]; trace_b = [bal]
    cash_t = [t[0]]; cash_v = [-fees]            # net cash starts negative (bought 1st account)

    for time, r in zip(t, R):
        risk = RISK_PCT * S
        pnl = r * risk
        if danchor is not None and (time - danchor).total_seconds() > 86400:
            danchor = None; dpnl = 0.0
        bal += pnl
        per[S]["trades"] += 1
        if danchor is None:
            if pnl < 0:
                danchor = time; dpnl = pnl
        else:
            dpnl += pnl

        # breach?
        if dpnl <= -DAILY * S or bal <= (1 - OVER) * S:
            per[S]["blown"] += 1
            events.append((time, "BLOWN", S))
            phase = "P1"; bal = S; danchor = None; dpnl = 0.0
            fees += FEES[S]                        # re-buy the same-size challenge
            funded_payouts = 0
            trace_t.append(time); trace_b.append(bal)
            continue

        if phase == "P1" and bal >= (1 + P1) * S:
            if per[S]["p1"] is None: per[S]["p1"] = time
            events.append((time, "PASS_P1", S))
            phase = "P2"; bal = S; danchor = None; dpnl = 0.0
        elif phase == "P2" and bal >= (1 + P2) * S:
            if per[S]["p2"] is None: per[S]["p2"] = time
            events.append((time, "PASS_P2", S))
            phase = "FUNDED"; bal = S; danchor = None; dpnl = 0.0
            if per[S]["funded"] is None: per[S]["funded"] = time
        elif phase == "FUNDED" and bal >= (1 + P1) * S:
            # PAYOUT only at exactly +10% profit; withdraw the 10% (overshoot carries)
            gross = P1 * S
            banked += SPLIT * gross               # trader keeps 80% of the 10%
            per[S]["payouts"] += 1; funded_payouts += 1
            bal -= gross
            danchor = None; dpnl = 0.0
            events.append((time, "PAYOUT", S))
            if funded_payouts >= PAYOUTS_TO_SCALE and si < len(LADDER) - 1:
                # use that payout money to BUY the next-size account
                si += 1; S = LADDER[si]; fees += FEES[S]
                phase = "P1"; bal = S; funded_payouts = 0
                events.append((time, "SCALE_UP", S))
            cash_t.append(time); cash_v.append(banked - fees)
        trace_t.append(time); trace_b.append(bal)

    cash_t.append(t[-1]); cash_v.append(banked - fees)
    return dict(per=per, events=events, banked=banked, fees=fees, net=banked - fees,
                final_size=S, final_phase=phase,
                trace_t=trace_t, trace_b=trace_b,
                cash_t=cash_t, cash_v=cash_v)


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    bases = {s: fq.precompute_base(s, load(s)) for s in fq.UNIVERSE}
    # gradient exit (lb=1, profit-only): close when the fast EMA slope turns against us
    trades = tc.combine([tg.run_gradient(s, bases[s], EMA, fq.BASEP, grad_lb=1, profit_only=True)
                         for s in fq.UNIVERSE])
    trades = trades[trades["outcome"] != "open"].copy()
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])
    trades = trades.sort_values("exit_time").reset_index(drop=True)

    # spread table (avg spread per pair actually used)
    print("=" * 92)
    print("AVG SPREAD PER INSTRUMENT USED IN THE BACKTEST")
    print("=" * 92)
    for grp, syms in [("FX (pips)", [s for s in fq.UNIVERSE if "." not in s]),
                      ("Index/energy (points)", [s for s in fq.UNIVERSE if "." in s])]:
        parts = []
        for s in syms:
            sp = fq.SPEC[s][0]
            pip = 0.01 if "JPY" in s else (1.0 if "." in s else 0.0001)
            parts.append(f"{s} {sp/pip:.1f}")
        print(f"  {grp}: " + " · ".join(parts))

    res = run_ladder(trades)

    print("\n" + "=" * 92)
    print("FTMO LADDER PROGRESSION · $10k -> $200k · Phase 1 (+10%) + Phase 2 (+5%) each · "
          "-5% daily / -10% overall")
    print(f"(scale up after {PAYOUTS_TO_SCALE} funded payout; 80% profit split; 1% risk/trade)")
    print("=" * 92)
    print(f"  {'Size':8} {'Phase1 pass':>12} {'Phase2 pass':>12} {'Funded':>12} "
          f"{'Payouts':>8} {'Blown':>6} {'Trades':>7}")
    for s in LADDER:
        p = res["per"][s]
        f = lambda d: d.strftime("%Y-%m-%d") if d is not None else "-"
        print(f"  ${s/1000:>5.0f}k {f(p['p1']):>12} {f(p['p2']):>12} {f(p['funded']):>12} "
              f"{p['payouts']:>8} {p['blown']:>6} {p['trades']:>7}")

    reached200 = next((e[0] for e in res["events"] if e[1] == "SCALE_UP" and e[2] == 200_000), None)
    t0 = res["trace_t"][0]
    print(f"\n  Reached $200k funded: {reached200.strftime('%Y-%m-%d') if reached200 else 'not reached'}"
          + (f"  ({(reached200 - t0).days/30.44:.1f} months from start)" if reached200 else ""))
    print(f"  Total challenge fees paid : ${res['fees']:,.0f}")
    print(f"  Total payouts banked (80%): ${res['banked']:,.0f}")
    print(f"  NET business profit       : ${res['net']:,.0f}")
    print(f"  Ended at: ${res['final_size']/1000:.0f}k account, phase {res['final_phase']}")
    blown = sum(res['per'][s]['blown'] for s in LADDER)
    print(f"  Accounts blown along the way: {blown}")

    # ── plot ──
    BG, FG = "#0d1117", "#e6edf3"
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(13, 9), facecolor=BG,
                                   gridspec_kw={"height_ratios": [1.4, 1]})
    axA.set_facecolor(BG); axB.set_facecolor(BG)

    axA.plot(res["trace_t"], res["trace_b"], color="#58a6ff", linewidth=1.0, zorder=3)
    for s in LADDER:
        axA.axhline(s, color="#30363d", linestyle="-", linewidth=0.6)
        axA.axhline(s * 1.10, color="#1f6b38", linestyle=":", linewidth=0.6)
        axA.text(res["trace_t"][0], s, f" ${s/1000:.0f}k", color="#7d8590", fontsize=8, va="bottom")
    ev = {"PASS_P1": ("^", "#d29922", "Phase 1 pass"), "PASS_P2": ("s", "#58a6ff", "Phase 2 pass"),
          "PAYOUT": ("*", "#3fb950", "payout"), "SCALE_UP": ("D", "#f2cc60", "scale up"),
          "BLOWN": ("x", "#f85149", "blown")}
    for key, (mk, col, lbl) in ev.items():
        xs = [t for t, ty, s in res["events"] if ty == key]
        ys = [s * (1.10 if key in ("PASS_P1", "PAYOUT") else (1.05 if key == "PASS_P2" else 1.0))
              for t, ty, s in res["events"] if ty == key]
        if xs:
            axA.scatter(xs, ys, marker=mk, s=60, color=col, zorder=5, label=f"{lbl} ({len(xs)})")
    axA.set_yscale("log")
    axA.set_yticks(LADDER); axA.set_yticklabels([f"${s/1000:.0f}k" for s in LADDER])
    axA.set_title("FTMO ladder · account balance climbing $10k → $200k "
                  "(Phase 1 + Phase 2 at each size)", color=FG, fontsize=12, fontweight="bold")
    axA.set_ylabel("Account balance (log)", color=FG)
    axA.tick_params(colors=FG); axA.grid(True, alpha=0.08, color=FG)
    axA.legend(facecolor=BG, edgecolor="#30363d", labelcolor=FG, fontsize=8, loc="lower right", ncol=2)
    for sp in axA.spines.values(): sp.set_edgecolor("#30363d")

    axB.plot(res["cash_t"], res["cash_v"], color="#3fb950", linewidth=1.8, drawstyle="steps-post")
    axB.fill_between(res["cash_t"], 0, res["cash_v"], step="post", color="#3fb950", alpha=0.12)
    axB.set_title(f"Net cash (payouts banked − challenge fees) · banked ${res['banked']:,.0f} · "
                  f"net ${res['net']:,.0f}", color=FG, fontsize=12, fontweight="bold")
    axB.set_ylabel("Net cash ($)", color=FG); axB.set_xlabel("Date", color=FG)
    axB.tick_params(colors=FG); axB.grid(True, alpha=0.1, color=FG)
    axB.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    for ax in (axA, axB):
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in axB.spines.values(): sp.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "ftmo_ladder.png"), dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nSaved -> {OUT}/ftmo_ladder.png")
    mt5.shutdown()


if __name__ == "__main__":
    main()
