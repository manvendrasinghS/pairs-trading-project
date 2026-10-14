"""The four figures, written to figures/ (Agg backend, so this runs headless)."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from signals import TradeRule


def plot_spread_with_trades(
    pair_bt: pd.DataFrame,
    pair_name: str,
    rule: TradeRule = TradeRule(),
    out_path: str = "figures/spread_with_trades.png",
) -> str:
    """z-score path with the bands and the trades marked."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pair_bt.index, pair_bt["zscore"], lw=1.0, color="#333", label="z-score")

    for level, style, label in [
        (rule.entry, "--", f"entry ±{rule.entry}"),
        (-rule.entry, "--", None),
        (rule.exit, ":", f"exit ±{rule.exit}"),
        (-rule.exit, ":", None),
        (rule.stop, "-", f"stop ±{rule.stop}"),
        (-rule.stop, "-", None),
    ]:
        ax.axhline(level, ls=style, lw=0.8, color="#c44", alpha=0.7, label=label)
    ax.axhline(0, lw=0.8, color="#888")

    entries = pair_bt[pair_bt["event"].str.startswith("entry", na=False)]
    exits = pair_bt[pair_bt["event"].str.startswith("exit", na=False)]
    ax.scatter(entries.index, entries["zscore"], marker="^", s=45,
               color="#2a7", zorder=5, label="entry")
    ax.scatter(exits.index, exits["zscore"], marker="v", s=45,
               color="#c33", zorder=5, label="exit")

    ax.set_title(f"{pair_name}: spread z-score and trades (out of sample)")
    ax.set_ylabel("z-score (formation-window sd)")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_equity_curve(
    oos_returns: pd.Series,
    benchmark: pd.Series | None = None,
    out_path: str = "figures/equity_curve.png",
) -> str:
    """Growth of $1 out of sample vs the benchmark, drawdown underneath."""
    equity = (1.0 + oos_returns.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    ax1.plot(equity.index, equity, lw=1.4, color="#1f4e79", label="pairs strategy (net of costs)")
    if benchmark is not None and len(benchmark):
        bench_eq = (1.0 + benchmark.reindex(oos_returns.index).fillna(0.0)).cumprod()
        ax1.plot(bench_eq.index, bench_eq, lw=1.2, color="#999", label="SPY buy-and-hold")
    ax1.axhline(1.0, lw=0.8, color="#000", alpha=0.3)
    ax1.set_ylabel("growth of $1")
    ax1.set_title("Out-of-sample equity curve (walk-forward, selection never sees the future)")
    ax1.legend(loc="upper left", fontsize=9)

    ax2.fill_between(drawdown.index, drawdown, 0, color="#c33", alpha=0.45)
    ax2.set_ylabel("drawdown")
    ax2.set_xlabel("date")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_is_vs_oos(pair_records: pd.DataFrame, out_path: str = "figures/is_vs_oos_sharpe.png") -> str:
    """In-sample vs out-of-sample Sharpe, one dot per selected pair.

    With real predictive content the dots would sit near the 45-degree line.
    """
    df = pair_records[["is_sharpe", "oos_sharpe"]].dropna()
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(df["is_sharpe"], df["oos_sharpe"], s=32, alpha=0.65, color="#1f4e79")

    lim = float(np.nanmax(np.abs(df.to_numpy()))) * 1.1 if len(df) else 3.0
    ax.plot([-lim, lim], [-lim, lim], ls="--", lw=1.0, color="#888",
            label="perfect persistence (45°)")
    ax.axhline(0, lw=0.8, color="#000", alpha=0.4)
    ax.axvline(0, lw=0.8, color="#000", alpha=0.4)
    if len(df) > 2:
        rho = df["is_sharpe"].corr(df["oos_sharpe"])
        ax.set_title(f"In-sample vs out-of-sample Sharpe (corr = {rho:+.2f})")
    ax.set_xlabel("in-sample Sharpe (formation window)")
    ax.set_ylabel("out-of-sample Sharpe (trading window)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_cost_sensitivity(sens: pd.DataFrame, out_path: str = "figures/cost_sensitivity.png") -> str:
    """Sharpe against assumed cost. The zero crossing is the break-even."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sens["one_way_bps"], sens["sharpe"], marker="o", lw=1.6, color="#1f4e79")
    ax.axhline(0, ls="--", lw=1.0, color="#c33", label="Sharpe = 0 (edge gone)")
    ax.set_xlabel("assumed one-way transaction cost (bps of gross notional)")
    ax.set_ylabel("out-of-sample Sharpe ratio")
    ax.set_title("Where does the edge break even?")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    from backtest import CostModel, backtest_pair, load_or_build_plan
    from data_pipeline import to_simple_returns

    prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
    pair_records = pd.read_csv("data/pair_records.csv")
    oos = pd.read_csv("data/oos_returns.csv", index_col=0, parse_dates=True)["net_return"]
    sens = pd.read_csv("data/cost_sensitivity.csv")

    made: list[str] = []

    # Example pair picked by most trades, not best Sharpe: a criterion that is
    # independent of whether the trades made money.
    if not pair_records.empty:
        plan = load_or_build_plan(prices, verbose=False)
        target = pair_records.sort_values("oos_n_trades", ascending=False).iloc[0]
        roll = plan[int(target["roll"])]
        sp = next(p for p in roll.pairs if p.name == target["pair"])

        returns = to_simple_returns(prices)
        trading_ret = returns.loc[returns.index.intersection(roll.trading_index)]
        bt = backtest_pair(trading_ret[sp.ticker_a], trading_ret[sp.ticker_b],
                           sp.z_oos, sp.stats, costs=CostModel())
        made.append(plot_spread_with_trades(
            bt, f"{sp.name} (roll {roll.roll_id}, {roll.trade_start} to {roll.trade_end})"))

        made.append(plot_is_vs_oos(pair_records))

    benchmark = None
    if "SPY" in prices.columns:
        benchmark = to_simple_returns(prices)["SPY"].reindex(oos.index).dropna()
    made.append(plot_equity_curve(oos, benchmark=benchmark))

    made.append(plot_cost_sensitivity(sens))

    print("Wrote figures:")
    for path in made:
        print(f"  {path}")
