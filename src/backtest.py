"""Block 3: walk-forward backtest, net of costs.

    net_return_t = position_{t-1} * (w_A*r_A,t + w_B*r_B,t) - costs_t

position_{t-1} because the position is decided at the previous close. Pair
selection runs inside the loop (build_walk_forward_plan), so a pair earns credit
only for the window after it was chosen.
"""

from __future__ import annotations

import pathlib
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from cointegration import engle_granger, screen_pairs
from data_pipeline import TRADING_DAYS_PER_YEAR, to_log_prices, to_simple_returns
from signals import (
    FormationStats,
    TradeRule,
    fit_formation,
    generate_positions,
    rolling_zscore,
    summarize_signal,
    zscore_out_of_sample,
)


@dataclass(frozen=True)
class CostModel:
    """Costs in bps.

    one_way_bps covers half the bid-ask crossed, commission and some slippage,
    charged on gross notional traded, so a round trip is ~10bp. It excludes
    market impact, which daily bars cannot calibrate: these results describe a
    small account. borrow_bps_annual is charged daily on the short leg; 50bp/yr
    is easy-to-borrow mega-cap, and hard-to-borrow names run 500bp+.
    """
    one_way_bps: float = 5.0
    borrow_bps_annual: float = 50.0

    @property
    def one_way(self) -> float:
        return self.one_way_bps / 10_000.0

    @property
    def borrow_daily(self) -> float:
        return (self.borrow_bps_annual / 10_000.0) / TRADING_DAYS_PER_YEAR


def spread_returns(ret_a: pd.Series, ret_b: pd.Series, stats: FormationStats) -> pd.Series:
    """Return of +1 unit of long spread at gross exposure 1."""
    w_a, w_b = stats.weights()
    aligned = pd.concat([ret_a.rename("a"), ret_b.rename("b")], axis=1).dropna()
    return (w_a * aligned["a"] + w_b * aligned["b"]).rename("spread_return")


def backtest_pair(
    ret_a: pd.Series,
    ret_b: pd.Series,
    zscore: pd.Series,
    stats: FormationStats,
    rule: TradeRule = TradeRule(),
    costs: CostModel = CostModel(),
) -> pd.DataFrame:
    """Daily P&L for one pair over one window.

    position is decided at day t's close; position_live is the shift(1) of it and
    is what earns day t's return. Drop that shift and the strategy buys the dip
    at the close that defined the dip.
    """
    spread_ret = spread_returns(ret_a, ret_b, stats)

    signal = generate_positions(zscore, rule=rule)
    idx = spread_ret.index.intersection(signal.index)
    signal = signal.loc[idx]
    spread_ret = spread_ret.loc[idx]

    position_live = signal["position"].shift(1).fillna(0.0)

    gross = position_live * spread_ret

    # |change| of 1.0 means 1.0 of gross notional traded, i.e. both legs.
    turnover = position_live.diff().abs().fillna(position_live.abs())
    txn_cost = turnover * costs.one_way

    # Exactly one leg is short either way, at a fixed share of gross exposure.
    w_a, w_b = stats.weights()
    short_leg_notional = abs(w_b) if w_b < 0 else abs(w_a)
    borrow = (position_live != 0).astype(float) * short_leg_notional * costs.borrow_daily

    cost = txn_cost + borrow
    net = gross - cost

    out = pd.DataFrame({
        "zscore": zscore.loc[idx],
        "position": signal["position"],
        "position_live": position_live,
        "days_held": signal["days_held"],
        "event": signal["event"],
        "spread_return": spread_ret,
        "gross_return": gross,
        "cost": cost,
        "net_return": net,
    })
    out["equity"] = (1.0 + out["net_return"]).cumprod()
    return out


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough fall, as a negative fraction."""
    equity = pd.Series(equity).dropna()
    if equity.empty:
        return float("nan")
    running_peak = equity.cummax()
    return float((equity / running_peak - 1.0).min())


def performance_metrics(net_returns: pd.Series, label: str = "") -> dict[str, float]:
    """Performance panel for a daily return series.

    Sharpe assumes a zero risk-free rate, so a true excess-return Sharpe would
    be lower. hit_rate is over active days only, since flat days are not bets.
    Pairs trading should show negative skew and fat tails; positive skew in a
    backtest is a reason to look for a bug.
    """
    r = pd.Series(net_returns).dropna()
    n = len(r)
    if n < 2 or r.std(ddof=1) == 0:
        return {"label": label, "n_days": n, "sharpe": float("nan")}

    equity = (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    years = n / TRADING_DAYS_PER_YEAR

    mean_d, sd_d = float(r.mean()), float(r.std(ddof=1))
    downside = r[r < 0].std(ddof=1)

    active = r[r != 0]
    mdd = max_drawdown(equity)
    ann_ret = float((1.0 + total_return) ** (1.0 / years) - 1.0) if total_return > -1 else -1.0

    return {
        "label": label,
        "n_days": n,
        "total_return": total_return,
        "ann_return": ann_ret,
        "ann_vol": sd_d * np.sqrt(TRADING_DAYS_PER_YEAR),
        "sharpe": mean_d / sd_d * np.sqrt(TRADING_DAYS_PER_YEAR),
        "sortino": (mean_d / downside * np.sqrt(TRADING_DAYS_PER_YEAR))
        if downside and downside > 0 else float("nan"),
        "max_drawdown": mdd,
        "calmar": ann_ret / abs(mdd) if mdd and mdd < 0 else float("nan"),
        "hit_rate": float((active > 0).mean()) if len(active) else float("nan"),
        "skew": float(scipy_stats.skew(r)),
        "kurtosis": float(scipy_stats.kurtosis(r, fisher=False)),  # 3.0 = normal
        "worst_day": float(r.min()),
        "best_day": float(r.max()),
    }


def extract_trades(pair_bt: pd.DataFrame) -> pd.DataFrame:
    """One row per trade. Daily stats hide the handful of large losses, and this
    gives the honest denominator for the Sharpe."""
    trades: list[dict] = []
    open_trade: dict | None = None

    for date, row in pair_bt.iterrows():
        event = row["event"]
        if isinstance(event, str) and event.startswith("entry"):
            open_trade = {
                "entry_date": date,
                "direction": "long_spread" if event == "entry_long" else "short_spread",
                "entry_z": row["zscore"],
                "pnl": 0.0,
                "days": 0,
            }
        if open_trade is not None:
            open_trade["pnl"] += float(row["net_return"])
            open_trade["days"] += 1
        if isinstance(event, str) and event.startswith("exit") and open_trade is not None:
            open_trade.update(exit_date=date, exit_z=row["zscore"], exit_reason=event)
            trades.append(open_trade)
            open_trade = None

    if open_trade is not None:      # still open at the window end
        open_trade.update(exit_date=pair_bt.index[-1], exit_z=pair_bt["zscore"].iloc[-1],
                          exit_reason="window_end")
        trades.append(open_trade)

    return pd.DataFrame(trades)


@dataclass
class WalkForwardConfig:
    """2y formation, 6m trading, 5 pairs held equal-weighted.

    step == trading_days, so out-of-sample windows do not overlap and the
    concatenated return series has no double-counted days. 12m/6m is the
    Gatev-Goetzmann-Rouwenhorst convention; 2y gives the ADF tests more power.
    """
    formation_days: int = 504
    trading_days: int = 126
    step: int = 126
    top_k: int = 5


@dataclass
class SelectedPair:
    """A pair one formation window chose, with its parameters frozen."""
    name: str
    ticker_a: str
    ticker_b: str
    same_sector: bool
    formation_pvalue: float
    half_life_days: float
    stats: FormationStats
    z_oos: pd.Series
    z_is: pd.Series
    oos_pvalue: float               # cointegration re-test on the trading window
    cointegration_survived: bool


@dataclass
class RollPlan:
    """What one roll decided, before any P&L."""
    roll_id: int
    formation_end: object
    trade_start: object
    trade_end: object
    n_tested: int
    n_significant: int
    pairs: list[SelectedPair]
    trading_index: pd.DatetimeIndex
    formation_index: pd.DatetimeIndex


def build_walk_forward_plan(
    prices: pd.DataFrame,
    config: WalkForwardConfig = WalkForwardConfig(),
    use_rolling_z: bool = False,
    verbose: bool = True,
) -> list[RollPlan]:
    """Screen, freeze and z-score every roll. The expensive half of the study.

    Split out from the P&L because the screen depends only on prices and the
    schedule, while costs and thresholds depend only on the z-score path. So the
    5 cost levels and 12 threshold cells share one screen instead of re-running
    it 17 times. It also means every lookahead risk lives in this function,
    which never sees a cost or a threshold.
    """
    prices = prices.sort_index()
    log_prices = to_log_prices(prices)

    n = len(prices)
    span = config.formation_days + config.trading_days
    if n < span:
        raise ValueError(f"Need at least {span} days, got {n}.")

    plans: list[RollPlan] = []
    starts = range(0, n - span + 1, config.step)

    for roll_id, i in enumerate(starts):
        f_start, f_end = i, i + config.formation_days
        t_start, t_end = f_end, min(f_end + config.trading_days, n)

        formation_log = log_prices.iloc[f_start:f_end]
        trading_log = log_prices.iloc[t_start:t_end]

        # Formation window only. This is the anti-lookahead line.
        screen = screen_pairs(formation_log)
        chosen = screen[screen["selected"]].head(config.top_k) if not screen.empty else screen

        selected: list[SelectedPair] = []
        for _, row in chosen.iterrows():
            a, b = row["ticker_a"], row["ticker_b"]

            stats = fit_formation(formation_log[a], formation_log[b])
            if stats.sd <= 0:
                continue

            if use_rolling_z:
                z_oos = rolling_zscore(trading_log[a], trading_log[b], stats)
            else:
                z_oos = zscore_out_of_sample(trading_log[a], trading_log[b], stats)
            # Same frozen rule replayed on the window it was fitted to: what a
            # careless backtest would report.
            z_is = zscore_out_of_sample(formation_log[a], formation_log[b], stats)

            # Diagnostic only, never fed back into selection.
            try:
                oos_p = engle_granger(trading_log[a], trading_log[b], a, b).eg_pvalue
            except Exception:
                oos_p = float("nan")

            selected.append(SelectedPair(
                name=f"{a}-{b}", ticker_a=a, ticker_b=b,
                same_sector=bool(row["same_sector"]),
                formation_pvalue=float(row["eg_pvalue"]),
                half_life_days=float(row["half_life_days"]),
                stats=stats, z_oos=z_oos, z_is=z_is,
                oos_pvalue=oos_p,
                cointegration_survived=bool(oos_p <= 0.05),
            ))

        plans.append(RollPlan(
            roll_id=roll_id,
            formation_end=formation_log.index[-1].date(),
            trade_start=trading_log.index[0].date(),
            trade_end=trading_log.index[-1].date(),
            n_tested=len(screen),
            n_significant=int((screen["eg_pvalue"] <= 0.05).sum()) if not screen.empty else 0,
            pairs=selected,
            trading_index=trading_log.index,
            formation_index=formation_log.index,
        ))
        if verbose:
            print(f"  screened roll {roll_id:>2} | {plans[-1].trade_start} -> "
                  f"{plans[-1].trade_end} | tested {plans[-1].n_tested:>3}, "
                  f"sig {plans[-1].n_significant:>3}, selected {len(selected)}")

    return plans


PLAN_CACHE = "data/walk_forward_plan.pkl"


def _plan_to_records(plan: list[RollPlan]) -> list[dict]:
    """Flatten to builtins and pandas objects for the cache.

    Pickle stores a class by import path. Run as a script, these dataclasses are
    `__main__.RollPlan`, so the cache would not load from multiple_testing.py.
    """
    return [{
        "roll_id": r.roll_id,
        "formation_end": r.formation_end,
        "trade_start": r.trade_start,
        "trade_end": r.trade_end,
        "n_tested": r.n_tested,
        "n_significant": r.n_significant,
        "trading_index": r.trading_index,
        "formation_index": r.formation_index,
        "pairs": [{
            "name": p.name, "ticker_a": p.ticker_a, "ticker_b": p.ticker_b,
            "same_sector": p.same_sector,
            "formation_pvalue": p.formation_pvalue,
            "half_life_days": p.half_life_days,
            "stats": (p.stats.alpha, p.stats.beta, p.stats.mean, p.stats.sd, p.stats.n_obs),
            "z_oos": p.z_oos, "z_is": p.z_is,
            "oos_pvalue": p.oos_pvalue,
            "cointegration_survived": p.cointegration_survived,
        } for p in r.pairs],
    } for r in plan]


def _plan_from_records(records: list[dict]) -> list[RollPlan]:
    return [RollPlan(
        roll_id=rec["roll_id"],
        formation_end=rec["formation_end"],
        trade_start=rec["trade_start"],
        trade_end=rec["trade_end"],
        n_tested=rec["n_tested"],
        n_significant=rec["n_significant"],
        trading_index=rec["trading_index"],
        formation_index=rec["formation_index"],
        pairs=[SelectedPair(
            name=p["name"], ticker_a=p["ticker_a"], ticker_b=p["ticker_b"],
            same_sector=p["same_sector"],
            formation_pvalue=p["formation_pvalue"],
            half_life_days=p["half_life_days"],
            stats=FormationStats(*p["stats"]),
            z_oos=p["z_oos"], z_is=p["z_is"],
            oos_pvalue=p["oos_pvalue"],
            cointegration_survived=p["cointegration_survived"],
        ) for p in rec["pairs"]],
    ) for rec in records]


def load_or_build_plan(
    prices: pd.DataFrame,
    config: WalkForwardConfig = WalkForwardConfig(),
    use_rolling_z: bool = False,
    cache_path: str = PLAN_CACHE,
    verbose: bool = True,
) -> list[RollPlan]:
    """Cached version of build_walk_forward_plan (~15 min to rebuild).

    The cache is keyed on nothing, so delete data/walk_forward_plan.pkl after
    changing the universe, the dates or the config. A half-clever key that misses
    one input would be worse: it would invite trust it has not earned.
    """
    path = pathlib.Path(cache_path)
    if path.exists():
        try:
            with path.open("rb") as fh:
                plan = _plan_from_records(pickle.load(fh))
            if verbose:
                print(f"Loaded cached walk-forward plan ({len(plan)} rolls) <- {path}")
            return plan
        except Exception as exc:
            print(f"  [plan cache] ignoring unreadable cache ({exc}); rebuilding.")

    plan = build_walk_forward_plan(prices, config, use_rolling_z, verbose=verbose)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(_plan_to_records(plan), fh)
    if verbose:
        print(f"Cached walk-forward plan ({len(plan)} rolls) -> {path}")
    return plan


def walk_forward_backtest(
    prices: pd.DataFrame,
    config: WalkForwardConfig = WalkForwardConfig(),
    rule: TradeRule = TradeRule(),
    costs: CostModel = CostModel(),
    use_rolling_z: bool = False,
    verbose: bool = True,
    plan: list[RollPlan] | None = None,
) -> dict:
    """Trade each roll's selected pairs through its out-of-sample window.

    Pass a pre-built `plan` to reuse one screen across cost or threshold runs.

    Returns oos_returns (daily portfolio net returns), pair_records (one row per
    roll-pair, with in-sample and out-of-sample stats), roll_records, trades, and
    the settings used.
    """
    returns = to_simple_returns(prices.sort_index())

    if plan is None:
        plan = build_walk_forward_plan(prices, config, use_rolling_z, verbose=verbose)

    portfolio_chunks: list[pd.Series] = []
    pair_records: list[dict] = []
    roll_records: list[dict] = []
    all_trades: list[pd.DataFrame] = []

    if verbose:
        print("=" * 78)
        print("BLOCK 3 — Walk-forward out-of-sample backtest")
        print("=" * 78)
        print(f"  formation={config.formation_days}d  trading={config.trading_days}d  "
              f"step={config.step}d  top_k={config.top_k}")
        print(f"  costs: {costs.one_way_bps}bp one-way, {costs.borrow_bps_annual}bp/yr borrow")
        print(f"  rule : entry={rule.entry} exit={rule.exit} stop={rule.stop} "
              f"max_hold={rule.max_hold}d   z-score={'rolling' if use_rolling_z else 'frozen'}")
        print()

    for roll in plan:
        trading_ret = returns.loc[returns.index.intersection(roll.trading_index)]
        formation_ret = returns.loc[returns.index.intersection(roll.formation_index)]

        if not roll.pairs:
            # Hold cash, and record the zero: dropping idle windows would
            # quietly inflate the strategy's returns.
            portfolio_chunks.append(pd.Series(0.0, index=trading_ret.index))
            roll_records.append({
                "roll": roll.roll_id, "formation_end": roll.formation_end,
                "trade_start": roll.trade_start, "trade_end": roll.trade_end,
                "n_tested": roll.n_tested, "n_significant": roll.n_significant,
                "n_selected": 0, "n_survived_oos": 0, "oos_sharpe": float("nan"),
            })
            if verbose:
                print(f"  roll {roll.roll_id:>2} | {roll.trade_start} -> "
                      f"{roll.trade_end} | no pair passed filters — flat")
            continue

        pair_returns: dict[str, pd.Series] = {}
        n_survived = 0

        for sp in roll.pairs:
            a, b = sp.ticker_a, sp.ticker_b

            bt_oos = backtest_pair(trading_ret[a], trading_ret[b], sp.z_oos,
                                   sp.stats, rule, costs)
            bt_is = backtest_pair(formation_ret[a], formation_ret[b], sp.z_is,
                                  sp.stats, rule, costs)

            n_survived += int(sp.cointegration_survived)

            m_is = performance_metrics(bt_is["net_return"], f"{sp.name} IS")
            m_oos = performance_metrics(bt_oos["net_return"], f"{sp.name} OOS")
            sig = summarize_signal(bt_oos)

            pair_records.append({
                "roll": roll.roll_id,
                "pair": sp.name,
                "same_sector": sp.same_sector,
                "formation_pvalue": sp.formation_pvalue,
                "oos_pvalue": sp.oos_pvalue,
                "cointegration_survived": sp.cointegration_survived,
                "beta": float(sp.stats.beta),
                "half_life_days": sp.half_life_days,
                "is_sharpe": m_is.get("sharpe", float("nan")),
                "oos_sharpe": m_oos.get("sharpe", float("nan")),
                "oos_total_return": m_oos.get("total_return", float("nan")),
                "oos_max_drawdown": m_oos.get("max_drawdown", float("nan")),
                "oos_n_trades": sig["n_trades"],
                "oos_stopped_out": sig["n_stopped_out"],
                "oos_time_in_market": sig["time_in_market"],
            })

            pair_returns[sp.name] = bt_oos["net_return"]
            tr = extract_trades(bt_oos)
            if not tr.empty:
                tr["roll"], tr["pair"] = roll.roll_id, sp.name
                all_trades.append(tr)

        if not pair_returns:
            portfolio_chunks.append(pd.Series(0.0, index=trading_ret.index))
            continue

        # Equal weights: 5 assets over 126 days is nowhere near enough data to
        # estimate a covariance matrix worth optimizing.
        book = pd.DataFrame(pair_returns).reindex(trading_ret.index).fillna(0.0)
        portfolio = book.mean(axis=1)
        portfolio_chunks.append(portfolio)

        roll_sharpe = performance_metrics(portfolio).get("sharpe", float("nan"))
        roll_records.append({
            "roll": roll.roll_id,
            "formation_end": roll.formation_end,
            "trade_start": roll.trade_start,
            "trade_end": roll.trade_end,
            "n_tested": roll.n_tested,
            "n_significant": roll.n_significant,
            "n_selected": len(pair_returns),
            "n_survived_oos": n_survived,
            "oos_sharpe": roll_sharpe,
        })

        if verbose:
            print(f"  roll {roll.roll_id:>2} | {roll.trade_start} -> "
                  f"{roll.trade_end} | tested {roll.n_tested:>3}, "
                  f"sig {roll.n_significant:>3}, traded {len(pair_returns)} | "
                  f"cointegration held OOS for {n_survived}/{len(pair_returns)} | "
                  f"OOS Sharpe {roll_sharpe:6.2f}")

    oos_returns = pd.concat(portfolio_chunks).sort_index()
    # Non-overlapping windows should never collide, but make sure.
    oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]

    return {
        "oos_returns": oos_returns,
        "pair_records": pd.DataFrame(pair_records),
        "roll_records": pd.DataFrame(roll_records),
        "trades": pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(),
        "config": config,
        "costs": costs,
        "rule": rule,
        "plan": plan,
    }


def cost_sensitivity(
    prices: pd.DataFrame,
    cost_levels_bps: tuple[float, ...] = (0.0, 2.5, 5.0, 10.0, 20.0),
    config: WalkForwardConfig = WalkForwardConfig(),
    rule: TradeRule = TradeRule(),
    plan: list[RollPlan] | None = None,
) -> pd.DataFrame:
    """Sharpe against assumed cost, so the reader can substitute their own.

    Any strategy is profitable at zero cost; the question is where break-even
    sits. One shared plan guarantees the rows differ only in cost.
    """
    if plan is None:
        plan = build_walk_forward_plan(prices, config, verbose=False)

    rows = []
    for bps in cost_levels_bps:
        res = walk_forward_backtest(
            prices, config=config, rule=rule,
            costs=CostModel(one_way_bps=bps), verbose=False, plan=plan,
        )
        m = performance_metrics(res["oos_returns"], f"{bps}bp")
        rows.append({
            "one_way_bps": bps,
            "ann_return": m.get("ann_return", float("nan")),
            "sharpe": m.get("sharpe", float("nan")),
            "max_drawdown": m.get("max_drawdown", float("nan")),
        })
    return pd.DataFrame(rows)


def print_metrics(metrics: dict, title: str) -> None:
    print(f"  {title}")
    print(f"    days={metrics.get('n_days', 0)}  "
          f"ann.return={metrics.get('ann_return', float('nan')):7.2%}  "
          f"ann.vol={metrics.get('ann_vol', float('nan')):6.2%}")
    print(f"    Sharpe={metrics.get('sharpe', float('nan')):6.2f}  "
          f"Sortino={metrics.get('sortino', float('nan')):6.2f}  "
          f"maxDD={metrics.get('max_drawdown', float('nan')):7.2%}  "
          f"Calmar={metrics.get('calmar', float('nan')):5.2f}")
    print(f"    hit rate={metrics.get('hit_rate', float('nan')):5.1%}  "
          f"skew={metrics.get('skew', float('nan')):5.2f}  "
          f"kurtosis={metrics.get('kurtosis', float('nan')):5.2f}  "
          f"worst day={metrics.get('worst_day', float('nan')):6.2%}")


def run_block3(prices: pd.DataFrame, plan: list[RollPlan] | None = None) -> dict:
    """Block 3 headline: out-of-sample result and the IS/OOS decay."""
    results = walk_forward_backtest(prices, plan=plan)
    oos = results["oos_returns"]

    print()
    m = performance_metrics(oos, "portfolio OOS")
    print_metrics(m, "OUT-OF-SAMPLE PORTFOLIO (net of costs)")

    pr = results["pair_records"]
    if not pr.empty:
        print()
        print("  IN-SAMPLE vs OUT-OF-SAMPLE (per selected pair, Sharpe)")
        print(f"    mean IS  Sharpe : {pr['is_sharpe'].mean():6.2f}")
        print(f"    mean OOS Sharpe : {pr['oos_sharpe'].mean():6.2f}")
        decay = pr["is_sharpe"].mean() - pr["oos_sharpe"].mean()
        print(f"    decay           : {decay:6.2f} Sharpe points")
        print()
        survived = pr["cointegration_survived"].mean()
        print(f"  COINTEGRATION SURVIVAL: {survived:.1%} of selected pairs were still")
        print(f"    cointegrated (p<=0.05) in the following trading window.")

    if "SPY" in prices.columns:
        spy = to_simple_returns(prices)["SPY"].loc[oos.index.intersection(prices.index)]
        spy = spy.reindex(oos.index).dropna()
        print()
        print_metrics(performance_metrics(spy, "SPY"), "BENCHMARK: SPY buy-and-hold, same period")
        corr = oos.reindex(spy.index).corr(spy)
        print(f"    correlation of strategy to SPY: {corr:+.3f}")
    print("=" * 78)
    return results


if __name__ == "__main__":
    px = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
    # Screen once, reuse for the headline run and the sensitivity table.
    wf_plan = load_or_build_plan(px)
    res = run_block3(px, plan=wf_plan)

    res["oos_returns"].to_frame("net_return").to_csv("data/oos_returns.csv")
    res["pair_records"].to_csv("data/pair_records.csv", index=False)
    res["roll_records"].to_csv("data/roll_records.csv", index=False)
    if not res["trades"].empty:
        res["trades"].to_csv("data/trades.csv", index=False)
    print("\nSaved OOS returns, pair records, roll records and trades -> data/")

    print()
    print("=" * 78)
    print("COST SENSITIVITY — where does the edge break even?")
    print("=" * 78)
    sens = cost_sensitivity(px, plan=wf_plan)
    print(sens.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    sens.to_csv("data/cost_sensitivity.csv", index=False)
    print("=" * 78)
