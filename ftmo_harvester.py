"""
FTMO HARVESTER — is the CHALLENGE STRUCTURE harvestable without a trading edge?

Not a trading strategy — a Monte-Carlo of the full FTMO lifecycle to test whether
the fee/payout ASYMMETRY is +EV:
  Phase 1 (+10%) → Phase 2 (+5%) → Funded (+10% payout cycles, 80% split), with
  -5% daily / -10% overall = blow. Fee (€89≈$96 on $10k) is REFUNDED on the first
  funded payout. Funded losses cost the trader nothing beyond the account (free option).
We sweep per-trade RISK and the strategy's true edge (WR×RR, incl. break-even and
the slightly-negative-after-costs reality we measured) to find the EV per challenge
bought. If a config is +EV, harvesting = buy many cheap accounts and run it.
"""
import numpy as np

START = 10_000.0
FEE = 96.0
SPLIT = 0.80
DAILY = -0.05; OVERALL = -0.10
P1, P2, FUND = 0.10, 0.05, 0.10
TPD = 3                      # trades/day
MAX_TRADES = 2000            # ~2.5yr cap per phase
MAX_PAYOUTS = 30
N = 40_000


def run_phase(rng, target, win, rr, risk):
    """Simulate one phase from a fresh $START. Return 'pass' / 'blow' / 'timeout'."""
    bal = 0.0                # P&L in fraction of START (barriers measured from initial)
    for t in range(0, MAX_TRADES, TPD):
        day = 0.0
        for _ in range(TPD):
            r = rr * risk if rng.random() < win else -risk
            bal += r; day += r
            if bal <= OVERALL:
                return "blow"
            if bal >= target:
                return "pass"
        if day <= DAILY:
            return "blow"
    return "timeout"


def run_account(rng, win, rr, risk):
    """Full lifecycle. Return (net_$_to_trader, funded_bool)."""
    net = -FEE                                   # pay the fee
    if run_phase(rng, P1, win, rr, risk) != "pass":
        return net, False
    if run_phase(rng, P2, win, rr, risk) != "pass":
        return net, False
    # FUNDED — free option: payout cycles until blow
    first = True
    for _ in range(MAX_PAYOUTS):
        res = run_phase(rng, FUND, win, rr, risk)
        if res == "pass":
            net += SPLIT * FUND * START          # 80% of +10% = $800
            if first:
                net += FEE; first = False        # fee refunded on first payout
        else:
            break                                # blown (or timed out) — account dead
    return net, True


def sim(win, rr, risk, n=N, seed=0):
    rng = np.random.default_rng(seed)
    nets = np.empty(n); funded = 0
    for i in range(n):
        nets[i], f = run_account(rng, win, rr, risk)
        funded += f
    exp = win * rr - (1 - win)                    # per-trade expectancy in R
    return dict(ev=nets.mean(), funded=funded / n, exp_r=exp,
                pos=(nets > 0).mean(), p95=np.percentile(nets, 95))


def main():
    print("=" * 104)
    print(f"FTMO HARVESTER · Monte-Carlo lifecycle · ${START:,.0f} acct · fee ${FEE:.0f} (refunded on 1st payout) · "
          f"80% split · {N:,} accts/config")
    print("=" * 104)
    print("  strategy edge is (WR, RR) → per-trade expectancy.  EV = mean net $ per challenge bought.")
    print(f"  {'WR':>5} {'RR':>4} {'exp/trade':>10} {'risk':>6} {'funded%':>8} {'EV/challenge':>13} {'P(profit)':>10}")
    print("-" * 104)
    # break-even, slightly-negative (our real post-cost result ~PF 0.85), and slightly-positive
    strats = [(0.500, 1.0, "break-even 1R"), (0.334, 2.0, "break-even 2R"),
              (0.300, 2.0, "NEG (real, PF0.86)"), (0.320, 2.0, "NEG (PF0.94)"),
              (0.360, 2.0, "POS (PF1.13)"), (0.550, 1.0, "POS 1R")]
    for win, rr, label in strats:
        print(f"  ── {label} ──")
        for risk in (0.005, 0.01, 0.02, 0.03, 0.05):
            s = sim(win, rr, risk)
            flag = "  <<< +EV" if s["ev"] > 0 else ""
            print(f"  {win:>5.3f} {rr:>4.1f} {s['exp_r']:>+9.3f}R {risk:>5.1%} {s['funded']:>7.1%} "
                  f"${s['ev']:>+11.0f} {s['pos']:>9.1%}{flag}")
        print()


if __name__ == "__main__":
    main()
