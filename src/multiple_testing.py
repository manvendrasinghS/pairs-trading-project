"""Block 4: multiple-testing audit.

Bonferroni and Benjamini-Hochberg on the screen's p-values, the Deflated Sharpe
Ratio on the strategy's return, and a threshold grid whose size feeds the trial
count. 435 tests at the 5% level produce ~22 hits with nothing going on, so what
matters is the excess over chance.

Benjamini & Hochberg (1995); Bailey & Lopez de Prado (2014).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from data_pipeline import TRADING_DAYS_PER_YEAR

EULER_MASCHERONI = 0.5772156649015329


def expected_false_positives(n_tests: int, alpha: float = 0.05) -> float:
    """n_tests * alpha: the hits chance alone hands you."""
    return n_tests * alpha


def bonferroni(pvalues: np.ndarray | pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """Require p <= alpha/n, controlling the chance of any false discovery.

    Assumes independent tests, which ours are not (XOM-CVX and XOM-COP share a
    leg), so it is conservative — a lower bound on real findings.
    """
    p = pd.Series(pvalues).dropna()
    n = len(p)
    threshold = alpha / n if n else np.nan
    return pd.DataFrame({
        "pvalue": p,
        "bonferroni_threshold": threshold,
        "survives": p <= threshold,
    })


def benjamini_hochberg(pvalues: np.ndarray | pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """FDR control: sort ascending, find the largest k with p_(k) <= (k/m)*alpha,
    reject every rank <= k. Also returns monotone q-values.

    Controls the expected share of discoveries that are false, which for a
    portfolio of pairs is the useful question. Note the rank cutoff: rejecting
    row by row instead is the standard bug.
    """
    p = pd.Series(pvalues).dropna().sort_values()
    m = len(p)
    if m == 0:
        return pd.DataFrame(columns=["pvalue", "rank", "bh_threshold", "qvalue", "survives"])

    ranks = np.arange(1, m + 1)
    thresholds = ranks / m * alpha

    below = p.to_numpy() <= thresholds
    k = int(ranks[below].max()) if below.any() else 0

    raw_q = p.to_numpy() * m / ranks
    qvalues = np.minimum.accumulate(raw_q[::-1])[::-1]
    qvalues = np.minimum(qvalues, 1.0)

    return pd.DataFrame({
        "pvalue": p.to_numpy(),
        "rank": ranks,
        "bh_threshold": thresholds,
        "qvalue": qvalues,
        "survives": ranks <= k,
    }, index=p.index)


def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """Expected Sharpe of the best of n_trials zero-edge strategies.

        sharpe_std * [ (1-g)*Z(1 - 1/N) + g*Z(1 - 1/(N*e)) ]

    Grows like sqrt(2 ln N): 435 trials buy about 2.9 cross-sectional sd of free
    Sharpe. This is the bar, not zero. Per-period units, like the PSR.
    """
    if n_trials < 2 or not np.isfinite(sharpe_std) or sharpe_std <= 0:
        return 0.0
    n = float(n_trials)
    term1 = (1.0 - EULER_MASCHERONI) * scipy_stats.norm.ppf(1.0 - 1.0 / n)
    term2 = EULER_MASCHERONI * scipy_stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(sharpe_std * (term1 + term2))


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    benchmark_sharpe_per_period: float = 0.0,
) -> float:
    """P(true Sharpe > benchmark), with skew and kurtosis corrections.

        Phi( (SR - SR*) * sqrt(T-1) / sqrt(1 - skew*SR + (kurt-1)/4 * SR^2) )

    The skew term is what bites here: negative skew lowers the PSR, and pairs
    trading earns many small wins against rare large losses. kurt is raw.
    """
    r = pd.Series(returns).dropna()
    T = len(r)
    if T < 10 or r.std(ddof=1) == 0:
        return float("nan")

    sr = float(r.mean() / r.std(ddof=1))
    skew = float(scipy_stats.skew(r))
    kurt = float(scipy_stats.kurtosis(r, fisher=False))

    denom = np.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2))
    z = (sr - benchmark_sharpe_per_period) * np.sqrt(T - 1) / denom
    return float(scipy_stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    trial_sharpes_annualized: np.ndarray | pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """PSR measured against expected_max_sharpe instead of zero.

    > 0.95 means unlikely to be search luck; ~0.50 means indistinguishable from
    the best of N coin flips. n_trials must count discarded strategies too —
    pairs screened times parameter combinations — so run_block4 passes an
    explicit number. Trial Sharpes, when given, set the cross-sectional spread;
    otherwise it falls back to 1/sqrt(T).
    """
    r = pd.Series(returns).dropna()
    T = len(r)
    if T < 10:
        return {"dsr": float("nan")}

    if trial_sharpes_annualized is not None and len(pd.Series(trial_sharpes_annualized).dropna()) > 2:
        trials = pd.Series(trial_sharpes_annualized).dropna() / np.sqrt(periods_per_year)
        sharpe_std = float(trials.std(ddof=1))
        source = "empirical spread of trial Sharpes"
    else:
        sharpe_std = 1.0 / np.sqrt(T)
        source = "theoretical 1/sqrt(T) under the null"

    bar = expected_max_sharpe(n_trials, sharpe_std)
    dsr = probabilistic_sharpe_ratio(r, benchmark_sharpe_per_period=bar)
    psr0 = probabilistic_sharpe_ratio(r, benchmark_sharpe_per_period=0.0)

    sr_daily = float(r.mean() / r.std(ddof=1))
    return {
        "n_trials": float(n_trials),
        "observed_sharpe_ann": sr_daily * np.sqrt(periods_per_year),
        "sharpe_std_daily": sharpe_std,
        "expected_max_sharpe_ann": bar * np.sqrt(periods_per_year),
        "psr_vs_zero": psr0,
        "dsr": dsr,
        "sharpe_std_source": source,
    }


def parameter_sensitivity(
    prices: pd.DataFrame,
    entries: tuple[float, ...] = (1.5, 2.0, 2.5),
    exits: tuple[float, ...] = (0.0, 0.5, 1.0),
    stops: tuple[float, ...] = (3.0, 4.0),
    plan: list | None = None,
) -> pd.DataFrame:
    """Walk-forward Sharpe across a grid of thresholds.

    Diagnostic, not a selection step: the headline run keeps the pre-committed
    defaults. A real finding degrades gracefully across nearby cells, and the
    grid size is an input to the deflated Sharpe.
    """
    # Imported here to avoid a circular import at module load.
    from backtest import (
        WalkForwardConfig,
        load_or_build_plan,
        performance_metrics,
        walk_forward_backtest,
    )
    from signals import TradeRule

    if plan is None:
        plan = load_or_build_plan(prices, WalkForwardConfig(), verbose=False)

    rows = []
    for entry in entries:
        for exit_ in exits:
            if exit_ >= entry:
                continue      # exit band has to sit inside the entry band
            for stop in stops:
                if stop <= entry:
                    continue  # a stop inside the entry band exits immediately
                rule = TradeRule(entry=entry, exit=exit_, stop=stop)
                res = walk_forward_backtest(
                    prices, config=WalkForwardConfig(), rule=rule,
                    verbose=False, plan=plan,
                )
                m = performance_metrics(res["oos_returns"])
                rows.append({
                    "entry": entry, "exit": exit_, "stop": stop,
                    "sharpe": m.get("sharpe", float("nan")),
                    "ann_return": m.get("ann_return", float("nan")),
                    "max_drawdown": m.get("max_drawdown", float("nan")),
                })
    return pd.DataFrame(rows)


def run_block4(
    oos_returns: pd.Series,
    pair_records: pd.DataFrame,
    roll_records: pd.DataFrame,
    screen_pvalues: pd.Series | None = None,
    n_parameter_combinations: int = 1,
) -> dict:
    """Put the result on trial for data mining and report the verdict."""
    print("=" * 78)
    print("BLOCK 4 — How much of this is luck?")
    print("=" * 78)

    tests_per_roll = int(roll_records["n_tested"].mean()) if not roll_records.empty else 0
    n_rolls = len(roll_records)
    total_pair_tests = tests_per_roll * n_rolls

    print("  (a) THE SIZE OF THE SEARCH")
    print(f"      pairs tested per formation window : {tests_per_roll}")
    print(f"      formation windows (rolls)         : {n_rolls}")
    print(f"      total cointegration tests run     : {total_pair_tests}")
    print(f"      parameter combinations explored   : {n_parameter_combinations}")
    print(f"      => effective strategy trials      : "
          f"{total_pair_tests * n_parameter_combinations}")
    print()

    if screen_pvalues is not None and len(screen_pvalues.dropna()):
        p = screen_pvalues.dropna()
        hits = int((p <= 0.05).sum())
        expected = expected_false_positives(len(p), 0.05)
        bonf = bonferroni(p)
        bh = benjamini_hochberg(p)
        print("  (b) DO THE 'SIGNIFICANT' PAIRS SURVIVE A MULTIPLICITY CORRECTION?")
        print(f"      tests                          : {len(p)}")
        print(f"      raw hits at p<=0.05            : {hits}")
        print(f"      expected by chance alone        : {expected:.1f}")
        print(f"      excess over chance              : {hits - expected:+.1f}")
        print(f"      survive Bonferroni (p<={bonf['bonferroni_threshold'].iloc[0]:.2e}) : "
              f"{int(bonf['survives'].sum())}")
        print(f"      survive Benjamini-Hochberg FDR : {int(bh['survives'].sum())}")
        print()

    trial_sharpes = pair_records["oos_sharpe"] if not pair_records.empty else None
    dsr = deflated_sharpe_ratio(
        oos_returns,
        n_trials=max(2, total_pair_tests * n_parameter_combinations),
        trial_sharpes_annualized=trial_sharpes,
    )

    print("  (c) DEFLATED SHARPE RATIO (Bailey & Lopez de Prado)")
    print(f"      observed OOS Sharpe (annualized)     : {dsr['observed_sharpe_ann']:6.2f}")
    print(f"      Sharpe scatter across trials         : "
          f"{dsr['sharpe_std_daily'] * np.sqrt(TRADING_DAYS_PER_YEAR):6.2f} annualized "
          f"({dsr['sharpe_std_source']})")
    print(f"      expected BEST Sharpe from luck alone : "
          f"{dsr['expected_max_sharpe_ann']:6.2f}   <- the bar to clear")
    print(f"      PSR vs zero  (ignores the search)    : {dsr['psr_vs_zero']:6.2%}")
    print(f"      DSR vs luck  (accounts for search)   : {dsr['dsr']:6.2%}")
    print()
    if np.isfinite(dsr["dsr"]):
        if dsr["dsr"] > 0.95:
            print("      VERDICT: survives the deflation.")
        elif dsr["dsr"] > 0.50:
            print("      VERDICT: suggestive, not conclusive once the search is counted.")
        else:
            print("      VERDICT: indistinguishable from luck; the edge is not established.")

    if not pair_records.empty:
        is_mean = pair_records["is_sharpe"].mean()
        oos_mean = pair_records["oos_sharpe"].mean()
        print()
        print("  (d) CROSS-CHECK: in-sample vs out-of-sample decay")
        print(f"      mean IS Sharpe {is_mean:.2f} -> mean OOS Sharpe {oos_mean:.2f}")
        if is_mean != 0:
            print(f"      retention: {oos_mean / is_mean:.1%} of the in-sample edge")

        # If the screen ranks pairs usefully, a lower p-value should mean a
        # better subsequent Sharpe. It comes out near zero.
        sub = pair_records[["formation_pvalue", "oos_sharpe"]].dropna()
        if len(sub) > 5:
            rho = sub["formation_pvalue"].corr(sub["oos_sharpe"], method="spearman")
            print()
            print(f"      corr(formation p-value, OOS Sharpe) = {rho:+.3f} (Spearman)")
            print("      Negative would mean stronger evidence buys a better OOS result;")
            print("      near zero means the screen ranks pairs no better than random.")
    print("=" * 78)
    return dsr


if __name__ == "__main__":
    oos = pd.read_csv("data/oos_returns.csv", index_col=0, parse_dates=True)["net_return"]
    pairs = pd.read_csv("data/pair_records.csv")
    rolls = pd.read_csv("data/roll_records.csv")
    try:
        screen = pd.read_csv("data/pair_screen_full_sample.csv")["eg_pvalue"]
    except FileNotFoundError:
        screen = None

    # Grid first: its size is an input to the deflated Sharpe.
    prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
    print("=" * 78)
    print("PARAMETER SENSITIVITY — does the result survive nearby thresholds?")
    print("=" * 78)
    grid = parameter_sensitivity(prices)
    print(grid.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    grid.to_csv("data/parameter_sensitivity.csv", index=False)
    print(f"\n  cells run: {len(grid)}   "
          f"Sharpe range: {grid['sharpe'].min():.2f} to {grid['sharpe'].max():.2f}")
    print("  A finding that only exists in one cell is not a finding.")
    print()

    run_block4(oos, pairs, rolls, screen_pvalues=screen,
               n_parameter_combinations=len(grid))
