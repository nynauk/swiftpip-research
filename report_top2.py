"""
Full report on the top-2 strategies:
  • EMA9/21 + ICT DOL (no-news)
  • H4-swing 24h + IFVG · RR=2

For each:
  - Generate full trade log if needed
  - Plot biggest win + most recent trade (with EMAs + ICT context)
  - Print every relevant stat
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MetaTrader5 as mt5

import main as bt
import news_filter as nf
import ema_dol_strategy as ema_dol


MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]


# ─────────────────────────────────────────────────────────────────────────────
# Generate / load trade logs
# ─────────────────────────────────────────────────────────────────────────────
def gen_ema_dol(news_block: bool):
    """Run EMA9/21 DOL across 7 majors and return combined trade df."""
    dfs = []
    for s in MAJORS:
        r = ema_dol.run_ema_dol(s, fast=9, slow=21, tp_mode="DOL",
                                min_rr=2.0, news_block=news_block)
        if r["n"]:
            dfs.append(r["df"])
    return pd.concat(dfs, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def gen_h4_swing_24h():
    """Run H4-swing 24h IFVG RR=2 across 7 majors and combine."""
    NO_RETRACE = 1e9
    per_symbol_min_w = {
        "EURUSD": 0.0002, "GBPUSD": 0.0002, "USDCAD": 0.0002, "USDCHF": 0.0002,
        "USDJPY": 0.02,   "AUDUSD": 0.0002, "NZDUSD": 0.0002,
    }
    dfs = []
    for s in MAJORS:
        df_h4  = bt.fetch_data(s, mt5.TIMEFRAME_H4,  bt.START_DATE, bt.END_DATE, tf_name="H4")
        df_15m = bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
        df_4h_sw = bt.find_swings(df_h4, bt.SWING_LOOKBACK)
        sweeps   = bt.find_sweeps(df_4h_sw)
        fvgs_15m = bt.find_all_fvgs(df_15m)
        fvgs_4h  = bt.find_all_fvgs(df_4h_sw)

        bt.DISRESPECT_MULT    = NO_RETRACE
        bt.MIN_FVG_WIDTH      = per_symbol_min_w.get(s, 0.0002)
        bt.REQUIRE_4H_TOUCH   = False
        bt.MAX_CLOSE_LOC      = 0.3
        bt.RR                 = 2.0
        bt.SESSION_START_HOUR = 0
        bt.SESSION_END_HOUR   = 24
        bt.HTF_BIAS_MODE      = "OFF"
        bt.SL_MODE            = "IFVG_EDGE"
        bt.NEWS_BLOCK_ENABLED = True
        if not bt.NEWS_BLOCK_SET:
            bt.NEWS_BLOCK_SET = nf.build_news_block_set(df_15m.index[0].date(), df_15m.index[-1].date())

        trades, _ = bt.run_backtest(df_4h_sw, df_15m, sweeps, fvgs_15m, fvgs_4h)
        if trades:
            df = pd.DataFrame(trades)
            df["symbol"] = s
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Plot a single trade with full ICT context + EMA overlay
# ─────────────────────────────────────────────────────────────────────────────
def plot_trade_full(df_15m, trade, out_path, title_prefix, show_ema=True,
                    ema_fast=9, ema_slow=21):
    BG, FG = "#0d1117", "#e6edf3"
    GRN, RED, YLW = "#3fb950", "#f85149", "#d29922"
    BLU, PURP = "#58a6ff", "#bc8cff"
    UP, DN = "#26a69a", "#ef5350"
    EMA_F_COL, EMA_S_COL = "#ff9f43", "#5b9eff"

    # Index lookup
    entry_idx = df_15m.index.get_indexer([pd.Timestamp(trade["entry_time"])], method="nearest")[0]
    exit_idx  = df_15m.index.get_indexer([pd.Timestamp(trade["exit_time"])],  method="nearest")[0]

    # Find a "context start": earlier of cross_time / sweep_time / 30 bars before entry
    ctx_idx = entry_idx - 30
    if "cross_time" in trade and pd.notna(trade.get("cross_time")):
        c_idx = df_15m.index.get_indexer([pd.Timestamp(trade["cross_time"])], method="nearest")[0]
        ctx_idx = min(ctx_idx, c_idx - 5)
    if "sweep_time" in trade and pd.notna(trade.get("sweep_time")):
        s_idx = df_15m.index.get_indexer([pd.Timestamp(trade["sweep_time"])], method="nearest")[0]
        ctx_idx = min(ctx_idx, s_idx - 5)

    lo = max(0, ctx_idx)
    hi = min(len(df_15m), exit_idx + 12)
    win = df_15m.iloc[lo:hi].copy().reset_index()
    win.rename(columns={win.columns[0]: "time"}, inplace=True)

    if show_ema:
        ema_f = df_15m["close"].ewm(span=ema_fast, adjust=False).mean().iloc[lo:hi].values
        ema_s = df_15m["close"].ewm(span=ema_slow, adjust=False).mean().iloc[lo:hi].values

    ex = entry_idx - lo
    xx = exit_idx - lo
    is_short = trade["direction"] == "short"
    entry, sl, tp = float(trade["entry"]), float(trade["sl"]), float(trade["tp"])

    fig, ax = plt.subplots(figsize=(18, 10), facecolor=BG)
    ax.set_facecolor(BG)

    # Candles
    for k, row in win.iterrows():
        col = UP if row["close"] >= row["open"] else DN
        ax.plot([k, k], [row["low"], row["high"]], color=col, linewidth=0.9, zorder=2)
        body_lo = min(row["open"], row["close"])
        h = abs(row["close"] - row["open"])
        if h == 0:
            h = (win["high"].max() - win["low"].min()) * 0.0005
        ax.add_patch(plt.Rectangle((k - 0.35, body_lo), 0.7, h,
                                   facecolor=col, edgecolor=col, zorder=3))

    if show_ema:
        xs = np.arange(len(win))
        ax.plot(xs, ema_f, color=EMA_F_COL, linewidth=1.6, label=f"EMA {ema_fast}", zorder=4)
        ax.plot(xs, ema_s, color=EMA_S_COL, linewidth=1.6, label=f"EMA {ema_slow}", zorder=4)

    # Cross marker (if EMA strategy)
    if "cross_time" in trade and pd.notna(trade.get("cross_time")):
        cx = df_15m.index.get_indexer([pd.Timestamp(trade["cross_time"])], method="nearest")[0] - lo
        if 0 <= cx < len(win):
            ax.axvline(cx, color=YLW, linestyle="--", alpha=0.5, linewidth=1.0, zorder=1)
            ax.text(cx, win["high"].max(), "  EMA cross",
                    color=YLW, fontsize=10, va="top", fontweight="bold", rotation=90)

    # Sweep marker (if H4-swing strategy)
    if "sweep_time" in trade and pd.notna(trade.get("sweep_time")):
        sx = df_15m.index.get_indexer([pd.Timestamp(trade["sweep_time"])], method="nearest")[0] - lo
        if 0 <= sx < len(win):
            ax.axvline(sx, color=BLU, linestyle="--", alpha=0.5, linewidth=1.0, zorder=1)
            ax.text(sx, win["high"].max(), "  Sweep",
                    color=BLU, fontsize=10, va="top", fontweight="bold", rotation=90)
        if "swept_level" in trade and pd.notna(trade.get("swept_level")):
            ax.axhline(float(trade["swept_level"]), color=BLU, linewidth=1.2,
                       linestyle="--", alpha=0.85)
            lbl = "Swept high" if is_short else "Swept low"
            ax.text(0.6, float(trade["swept_level"]), f"  {lbl}",
                    color=BLU, fontsize=9, va="bottom", fontweight="bold")

    # FVG zone
    if "fvg_top" in trade and pd.notna(trade.get("fvg_top")):
        ft, fb = float(trade["fvg_top"]), float(trade["fvg_bottom"])
        ax.axhspan(fb, ft, color=PURP, alpha=0.20, zorder=1)
        ax.axhline(ft, color=PURP, linewidth=0.8, linestyle=":", alpha=0.7)
        ax.axhline(fb, color=PURP, linewidth=0.8, linestyle=":", alpha=0.7)
        ax.text(0.6, ft, "  FVG / IFVG zone", color=PURP, fontsize=9, va="bottom", fontweight="bold")

    # Position boxes
    box_w = max(xx - ex, 8)
    tgt_lo, tgt_hi = sorted([entry, tp])
    ax.add_patch(plt.Rectangle((ex, tgt_lo), box_w, tgt_hi - tgt_lo,
                               facecolor=GRN, alpha=0.15, edgecolor=GRN, linewidth=1.0, zorder=4))
    stp_lo, stp_hi = sorted([entry, sl])
    ax.add_patch(plt.Rectangle((ex, stp_lo), box_w, stp_hi - stp_lo,
                               facecolor=RED, alpha=0.15, edgecolor=RED, linewidth=1.0, zorder=4))

    # Levels + staggered labels
    rr_str = f"+{trade.get('achieved_rr', trade['pnl_r']):.2f}R" if trade["outcome"] == "win" else f"({trade['pnl_r']:+.2f}R)"
    span = max(win["high"].max() - win["low"].min(), 1e-6)
    min_gap = span * 0.025
    lvls = sorted([
        (entry, FG,  f"Entry  {entry}"),
        (tp,    GRN, f"TP {rr_str}  {tp}"),
        (sl,    RED, f"SL  {sl}"),
    ], key=lambda x: -x[0])
    last_y = None
    for price, col, lbl in lvls:
        ax.axhline(price, color=col, linewidth=1.0, alpha=0.85, zorder=5)
        y = price
        if last_y is not None and (last_y - y) < min_gap:
            y = last_y - min_gap
        last_y = y
        ax.text(ex + box_w + 0.5, y, f"  {lbl}", color=col,
                fontsize=10, va="center", fontweight="bold")

    # Entry/exit markers
    ax.scatter([ex], [entry], marker="v" if is_short else "^",
               s=220, color=YLW, edgecolor="black", zorder=7, label="Entry")
    exit_col = GRN if trade["outcome"] == "win" else (RED if trade["outcome"] == "loss" else YLW)
    ax.scatter([xx], [float(trade["exit"])], marker="o", s=180, color=exit_col,
               edgecolor="black", zorder=7, label=f"Exit ({trade['outcome'].upper()})")

    n_ticks = 10
    step = max(1, len(win) // n_ticks)
    ticks = list(range(0, len(win), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([win["time"].iloc[t].strftime("%m-%d %H:%M") for t in ticks],
                       rotation=30, ha="right", fontsize=9, color=FG)

    sym = trade.get("symbol", "?")
    dur_hr = (pd.to_datetime(trade["exit_time"]) - pd.to_datetime(trade["entry_time"])).total_seconds() / 3600
    ax.set_title(
        f"{title_prefix}  |  {sym} M15  |  {trade['direction'].upper()}  |  "
        f"{trade['outcome'].upper()}  ({trade['pnl_r']:+.2f}R)  |  "
        f"Entry {pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d %H:%M')}  "
        f"|  Duration {dur_hr:.1f}h",
        color=FG, fontsize=12, fontweight="bold", pad=14
    )
    ax.set_ylabel("Price", color=FG)
    ax.tick_params(colors=FG)
    ax.grid(True, alpha=0.10, color=FG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.legend(facecolor="#161b22", edgecolor="#444c56", labelcolor=FG, loc="upper left", fontsize=10)

    # Annotation panel
    lines = [
        f"Symbol      : {sym}",
        f"Direction   : {trade['direction'].upper()}",
    ]
    if "cross_time" in trade and pd.notna(trade.get("cross_time")):
        lines.append(f"Cross time  : {pd.to_datetime(trade['cross_time']).strftime('%Y-%m-%d %H:%M')}")
    if "sweep_time" in trade and pd.notna(trade.get("sweep_time")):
        lines.append(f"Sweep time  : {pd.to_datetime(trade['sweep_time']).strftime('%Y-%m-%d %H:%M')}")
    if "swept_level" in trade and pd.notna(trade.get("swept_level")):
        lines.append(f"Swept level : {trade['swept_level']}")
    if "fvg_time" in trade and pd.notna(trade.get("fvg_time")):
        lines.append(f"FVG time    : {pd.to_datetime(trade['fvg_time']).strftime('%Y-%m-%d %H:%M')}")
    lines += [
        f"Entry time  : {pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d %H:%M')}",
        f"Exit time   : {pd.to_datetime(trade['exit_time']).strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Entry       : {entry}",
        f"Stop loss   : {sl}",
        f"Take profit : {tp}",
        f"Risk dist   : {abs(entry - sl):.5f}",
        f"Reward dist : {abs(tp - entry):.5f}",
    ]
    if "achieved_rr" in trade and pd.notna(trade.get("achieved_rr")):
        lines.append(f"Achieved RR : {trade['achieved_rr']:.2f}")
    lines += [
        f"P/L         : {trade['pnl_r']:+.2f}R",
        f"Duration    : {dur_hr:.1f} h",
    ]
    annot = "\n".join(lines)
    ax.text(0.985, 0.02, annot, transform=ax.transAxes,
            fontsize=9, color=FG, va="bottom", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#161b22",
                      edgecolor="#444c56", alpha=0.95))

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Full stats
# ─────────────────────────────────────────────────────────────────────────────
def full_stats(df: pd.DataFrame, label: str):
    df = df.sort_values("entry_time").reset_index(drop=True)
    closed = df[df["outcome"] != "open"].copy()
    if not len(closed):
        print(f"{label}: no closed trades")
        return

    period_start = pd.to_datetime(closed["entry_time"]).min()
    period_end   = pd.to_datetime(closed["entry_time"]).max()
    years = (period_end - period_start).days / 365.25

    n = len(closed)
    n_win = (closed["outcome"] == "win").sum()
    n_loss = (closed["outcome"] == "loss").sum()
    wr = n_win / n * 100
    total_r = closed["pnl_r"].sum()
    gross_w = closed.loc[closed["outcome"] == "win", "pnl_r"].sum()
    gross_l = -closed.loc[closed["outcome"] == "loss", "pnl_r"].sum()
    pf = gross_w / gross_l if gross_l else float("inf")
    avg_win = closed.loc[closed["outcome"] == "win", "pnl_r"].mean() if n_win else 0
    avg_loss = closed.loc[closed["outcome"] == "loss", "pnl_r"].mean() if n_loss else 0
    biggest_win = closed["pnl_r"].max()
    biggest_loss = closed["pnl_r"].min()
    expectancy = closed["pnl_r"].mean()

    # Streaks
    cur_w = cur_l = best_w = best_l = 0
    win_span = loss_span = (None, None)
    run_start = None
    for _, row in closed.iterrows():
        if row["outcome"] == "win":
            if cur_w == 0:
                run_start = row["entry_time"]
            cur_w += 1
            if cur_w > best_w:
                best_w = cur_w
                win_span = (run_start, row["entry_time"])
            cur_l = 0
        elif row["outcome"] == "loss":
            if cur_l == 0:
                run_start = row["entry_time"]
            cur_l += 1
            if cur_l > best_l:
                best_l = cur_l
                loss_span = (run_start, row["entry_time"])
            cur_w = 0

    # Compounded equity
    bal = 10_000
    eq = [bal]
    for r in closed["pnl_r"]:
        bal += bal * 0.01 * float(r)
        eq.append(bal)
    eq_arr = np.array(eq)
    dd = ((eq_arr - np.maximum.accumulate(eq_arr)) / np.maximum.accumulate(eq_arr) * 100).min()
    cagr = (bal / 10_000) ** (1 / years) - 1 if years > 0 else 0

    # Time in trade
    closed["dur"] = (pd.to_datetime(closed["exit_time"]) - pd.to_datetime(closed["entry_time"])).dt.total_seconds() / 3600

    # Monthly
    closed["month"] = pd.to_datetime(closed["entry_time"]).dt.to_period("M")
    monthly = closed.groupby("month")["pnl_r"].sum()
    pos_m = (monthly > 0).sum()
    neg_m = (monthly < 0).sum()

    # Long/short
    ln = closed[closed["direction"] == "long"]
    sh = closed[closed["direction"] == "short"]
    lwr = (ln["outcome"] == "win").sum() / len(ln) * 100 if len(ln) else 0
    swr = (sh["outcome"] == "win").sum() / len(sh) * 100 if len(sh) else 0

    print(f"\n{'=' * 90}")
    print(f"  {label}")
    print(f"{'=' * 90}")
    print(f"  Period                     : {period_start} → {period_end}  ({years:.2f} years)")
    print()
    print(f"  HEADLINE")
    print(f"    Total trades             : {n}")
    print(f"    Wins / Losses            : {n_win} / {n_loss}")
    print(f"    Win rate                 : {wr:.2f}%")
    print(f"    Expectancy per trade     : {expectancy:+.3f}R")
    print(f"    Total R                  : {total_r:+.1f}R")
    print(f"    Profit factor (gross W/L): {pf:.2f}")
    print()
    print(f"  TRADE QUALITY")
    print(f"    Avg win                  : +{avg_win:.2f}R")
    print(f"    Avg loss                 : {avg_loss:.2f}R")
    print(f"    Biggest win              : +{biggest_win:.2f}R")
    print(f"    Biggest loss             : {biggest_loss:.2f}R")
    if "achieved_rr" in closed.columns:
        print(f"    Avg achieved RR (wins)   : {closed[closed['outcome']=='win']['achieved_rr'].mean():.2f}")
    print()
    print(f"  STREAKS & DRAWDOWN")
    print(f"    Longest WIN streak       : {best_w}  ({win_span[0]} → {win_span[1]})")
    print(f"    Longest LOSS streak      : {best_l}  ({loss_span[0]} → {loss_span[1]})")
    print(f"    Max drawdown (1% risk)   : {dd:.2f}%")
    print(f"    Recovery factor          : {abs(total_r/dd):.2f}  (R per % DD)")
    print()
    print(f"  COMPOUNDED EQUITY (1% risk on $10k)")
    print(f"    Ending balance           : ${bal:,.0f}")
    print(f"    Net return               : {(bal/10_000 - 1)*100:+,.1f}%")
    print(f"    CAGR                     : {cagr*100:+.1f}%")
    print()
    print(f"  FREQUENCY")
    print(f"    Trades per year          : {n/years:.1f}")
    print(f"    Trades per month         : {n/years/12:.1f}")
    print(f"    Trades per week          : {n/years/52:.1f}")
    print()
    print(f"  TIME IN TRADE")
    print(f"    Average                  : {closed['dur'].mean():.1f} hours  ({closed['dur'].mean()/24:.2f} days)")
    print(f"    Median                   : {closed['dur'].median():.1f} hours")
    print(f"    Longest                  : {closed['dur'].max():.1f} hours")
    print()
    print(f"  MONTHLY CONSISTENCY")
    print(f"    Months in sample         : {len(monthly)}")
    print(f"    Profitable months        : {pos_m}  ({pos_m/len(monthly)*100:.1f}%)")
    print(f"    Losing months            : {neg_m}  ({neg_m/len(monthly)*100:.1f}%)")
    print(f"    Best month               : {monthly.max():+.1f}R")
    print(f"    Worst month              : {monthly.min():+.1f}R")
    print()
    print(f"  LONG vs SHORT")
    print(f"    LONG : N={len(ln):4d}  WR={lwr:5.1f}%  R={ln['pnl_r'].sum():+7.1f}")
    print(f"    SHORT: N={len(sh):4d}  WR={swr:5.1f}%  R={sh['pnl_r'].sum():+7.1f}")
    print()
    print(f"  PER-SYMBOL")
    for sym, g in closed.groupby("symbol"):
        wr_s = (g["outcome"] == "win").sum() / len(g) * 100
        print(f"    {sym:8}: N={len(g):4d}  WR={wr_s:5.1f}%  R={g['pnl_r'].sum():+7.1f}")


# ─────────────────────────────────────────────────────────────────────────────
def render_strategy(df: pd.DataFrame, label: str, out_dir: str, show_ema: bool):
    """Plot biggest win + most recent for one strategy + print full stats."""
    os.makedirs(out_dir, exist_ok=True)
    closed = df[df["outcome"] != "open"].sort_values("entry_time")
    if not len(closed):
        return

    full_stats(df, label)
    df.to_csv(os.path.join(out_dir, "trades.csv"), index=False)

    biggest = closed[closed["outcome"] == "win"].sort_values("pnl_r", ascending=False).iloc[0]
    recent = closed.iloc[-1]

    for tag, t in [("biggest_win", biggest), ("most_recent", recent)]:
        sym = t["symbol"]
        df_15m = bt.fetch_data(sym, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
        out = os.path.join(out_dir, f"{tag}.png")
        title = f"{label.upper()} — {tag.upper().replace('_', ' ')}"
        plot_trade_full(df_15m, t.to_dict(), out, title, show_ema=show_ema)
        print(f"  → {out}")


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")

    print("Generating EMA9/21 DOL trades...")
    ema_df = gen_ema_dol(news_block=True)
    print("Generating H4-swing 24h IFVG trades...")
    h4_df = gen_h4_swing_24h()

    render_strategy(ema_df, "EMA9/21 + ICT DOL (news-blocked)",
                    os.path.join("strategies", "report_ema_dol"),
                    show_ema=True)
    render_strategy(h4_df, "H4-Swing 24h + IFVG RR=2 (news-blocked)",
                    os.path.join("strategies", "report_h4_swing"),
                    show_ema=False)

    mt5.shutdown()


if __name__ == "__main__":
    main()
