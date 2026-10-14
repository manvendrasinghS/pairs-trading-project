"""Block 1: the cointegration screen.

Engle-Granger and Johansen tests, hedge ratios, spread half-lives, and the
screen that runs them over every candidate pair. Everything here is computed on
a formation window; a low p-value is a candidate, not a finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

from data_pipeline import TRADING_DAYS_PER_YEAR, sector_of


def adf_pvalue(series: pd.Series, regression: str = "c") -> float:
    """ADF p-value. H0 is a unit root, so small p = evidence of mean reversion.

    regression="c" allows a non-zero mean but no time trend: a trend term would
    call a steadily drifting spread stationary, which is not tradeable.
    """
    series = pd.Series(series).dropna()
    if len(series) < 20:
        return np.nan
    return float(adfuller(series, regression=regression, autolag="AIC")[1])


# 1% not 5%: the gate inherits the ADF test's own error rate, so at 5% roughly
# one pair in ten fails the precondition by chance. Measured on 60 random walks
# (test_leg_check_size_on_random_walks): 2/60 rejected at 5%, 0/60 at 1%.
LEG_UNIT_ROOT_ALPHA = 0.01


def is_unit_root(series: pd.Series, alpha: float = LEG_UNIT_ROOT_ALPHA) -> bool:
    """True if the series looks I(1). Checked per leg before testing a pair:
    an already-stationary "price" gives a tiny pair p-value that means nothing."""
    p = adf_pvalue(series)
    return bool(np.isnan(p) or p > alpha)


def estimate_hedge_ratio(log_a: pd.Series, log_b: pd.Series) -> tuple[float, float]:
    """OLS of log_a on log_b -> (alpha, beta).

    OLS is asymmetric, so (A,B) and (B,A) are different tests. Pairs are always
    ordered alphabetically and cross-checked against Johansen.
    """
    x = sm.add_constant(log_b.rename("b"))
    fit = sm.OLS(log_a, x).fit()
    return float(fit.params.iloc[0]), float(fit.params.iloc[1])


def spread_series(log_a: pd.Series, log_b: pd.Series, alpha: float, beta: float) -> pd.Series:
    """Spread from given alpha and beta. Never re-estimates: that separation is
    what stops an out-of-sample spread being built from out-of-sample prices."""
    return (log_a - alpha - beta * log_b).rename("spread")


def half_life(spread: pd.Series) -> float:
    """Half-life in days from an AR(1) fit: phi = 1 + b, hl = -ln2 / ln(phi).

    A tradability measure, not a stationarity test. OLS biases phi downward, so
    a true random walk still gives a large but finite half-life (283 to 13,000
    days across seeds) — hence the band in screen_pairs.
    """
    s = pd.Series(spread).dropna()
    if len(s) < 20:
        return float("nan")

    s_lag = s.shift(1)
    delta = (s - s_lag).dropna()
    s_lag = s_lag.loc[delta.index]

    fit = sm.OLS(delta, sm.add_constant(s_lag)).fit()
    b = float(fit.params.iloc[1])

    phi = 1.0 + b
    if phi <= 0 or phi >= 1:
        return float("inf")       # no decay, or a degenerate fit
    return float(-np.log(2.0) / np.log(phi))


@dataclass
class PairStats:
    """What one formation window says about one pair."""
    ticker_a: str
    ticker_b: str
    same_sector: bool
    alpha: float
    beta: float
    eg_pvalue: float
    spread_mean: float
    spread_std: float
    half_life_days: float
    legs_are_i1: bool
    n_obs: int

    @property
    def name(self) -> str:
        return f"{self.ticker_a}-{self.ticker_b}"

    def summary_line(self) -> str:
        stars = "***" if self.eg_pvalue < 0.01 else "**" if self.eg_pvalue < 0.05 else "*" if self.eg_pvalue < 0.10 else ""
        hl = f"{self.half_life_days:6.1f}" if np.isfinite(self.half_life_days) else "   inf"
        return (f"{self.name:<12} p={self.eg_pvalue:.4f}{stars:<3} beta={self.beta:6.3f}  "
                f"half-life={hl}d  sd={self.spread_std:.4f}  "
                f"{'same-sector' if self.same_sector else 'cross-sector'}")


def engle_granger(log_a: pd.Series, log_b: pd.Series, name_a: str, name_b: str) -> PairStats:
    """Engle-Granger two-step test. H0 is no cointegration.

    Uses statsmodels.coint rather than adfuller() on the residual: beta was
    fitted to minimize that residual's variance, so standard ADF critical values
    over-reject. coint applies the wider Engle-Granger values.
    """
    aligned = pd.concat([log_a.rename("a"), log_b.rename("b")], axis=1).dropna()
    a, b = aligned["a"], aligned["b"]

    alpha, beta = estimate_hedge_ratio(a, b)
    spread = spread_series(a, b, alpha, beta)

    # trend="c" matches the spread definition: constant mean, no time trend.
    _, eg_p, _ = coint(a, b, trend="c", autolag="AIC")

    return PairStats(
        ticker_a=name_a,
        ticker_b=name_b,
        same_sector=(sector_of(name_a) == sector_of(name_b)),
        alpha=alpha,
        beta=beta,
        eg_pvalue=float(eg_p),
        spread_mean=float(spread.mean()),
        spread_std=float(spread.std(ddof=1)),
        half_life_days=half_life(spread),
        legs_are_i1=bool(is_unit_root(a) and is_unit_root(b)),
        n_obs=int(len(aligned)),
    )


def diagnose_pair(log_a: pd.Series, log_b: pd.Series, name: str = "") -> dict:
    """Level vs trend stationarity, spread drift and half-life for one pair.

    Written to explain why the positive controls fail. Two funds on the same
    underlying drift apart by their fee differential, and a spread with a
    deterministic trend is not stationary around a constant, so the level test
    correctly fails to reject: GLD-IAU is p=0.15 on levels and 0.0000 allowing a
    trend. SPY-IVV is worse (0.74 and 0.70) because it also has a 2016 level
    shift, which unit-root tests are blind to (Perron 1989). The screen keeps
    trend="c" anyway, since a drifting mean is not something this z-score can
    trade.
    """
    from scipy import stats as _st

    aligned = pd.concat([log_a.rename("a"), log_b.rename("b")], axis=1).dropna()
    a, b = aligned["a"], aligned["b"]
    alpha, beta = estimate_hedge_ratio(a, b)
    spread = spread_series(a, b, alpha, beta)

    def _drift(series: pd.Series) -> tuple[float, float]:
        """Straight-line slope through the series in bps/year, plus R^2."""
        slope, _, rvalue, _, _ = _st.linregress(np.arange(len(series)), series.to_numpy())
        return float(slope * TRADING_DAYS_PER_YEAR * 10_000), float(rvalue ** 2)

    drift_ols, r2_ols = _drift(spread)

    # Also at beta=1, the economically right ratio for same-asset funds: a
    # fitted beta absorbs part of the drift and hides the fee mechanism
    # (GLD-IAU: -6.4bp/y, R2=0.42 at OLS beta vs -14.9bp/y, R2=0.99 at beta=1,
    # against a 15bp fee gap).
    unit_spread = (a - b).rename("unit_spread")
    drift_unit, r2_unit = _drift(unit_spread)

    return {
        "pair": name,
        "beta": beta,
        "p_level": float(coint(a, b, trend="c", autolag="AIC")[1]),   # the screen's test
        "p_trend": float(coint(a, b, trend="ct", autolag="AIC")[1]),  # allows a drift
        "drift_bps_per_year": drift_ols,
        "trend_r2": r2_ols,
        "drift_bps_per_year_unit_beta": drift_unit,
        "trend_r2_unit_beta": r2_unit,
        "spread_std": float(spread.std(ddof=1)),
        "half_life_days": half_life(spread),
    }


def johansen_cointegrated(
    log_a: pd.Series,
    log_b: pd.Series,
    significance: str = "95%",
) -> bool:
    """Johansen trace test on {A, B} as a robustness flag.

    Symmetric, unlike Engle-Granger, which depends on which leg is on the left.
    True means the r=0 statistic clears its critical value.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    data = pd.concat([log_a, log_b], axis=1).dropna()
    if len(data) < 50:
        return False
    try:
        # det_order=0 matches trend="c"; k_ar_diff=1 is the usual default.
        res = coint_johansen(data.values, det_order=0, k_ar_diff=1)
    except Exception:
        return False

    col = {"90%": 0, "95%": 1, "99%": 2}[significance]
    return bool(res.lr1[0] > res.cvt[0, col])


def screen_pairs(
    log_prices: pd.DataFrame,
    pvalue_max: float = 0.05,
    half_life_min: float = 5.0,
    half_life_max: float = 60.0,
    require_i1_legs: bool = True,
    require_positive_beta: bool = True,
) -> pd.DataFrame:
    """One row per pair with a `selected` flag. Call on a formation window only.

    Returns every pair, not just the winners, so later blocks can report how
    many tests were run. The half-life band is tradability; the positive-beta
    filter drops "pairs" that are long both legs, i.e. market bets.
    """
    tickers = sorted(log_prices.columns)
    rows: list[dict] = []

    for a, b in [(x, y) for i, x in enumerate(tickers) for y in tickers[i + 1:]]:
        try:
            stats = engle_granger(log_prices[a], log_prices[b], a, b)
        except Exception:
            continue  # degenerate window, skip the pair rather than die

        selected = (
            stats.eg_pvalue <= pvalue_max
            and np.isfinite(stats.half_life_days)
            and half_life_min <= stats.half_life_days <= half_life_max
            and (stats.legs_are_i1 or not require_i1_legs)
            and (stats.beta > 0 or not require_positive_beta)
            and stats.spread_std > 0
        )

        rows.append({
            "pair": stats.name,
            "ticker_a": stats.ticker_a,
            "ticker_b": stats.ticker_b,
            "same_sector": stats.same_sector,
            "alpha": stats.alpha,
            "beta": stats.beta,
            "eg_pvalue": stats.eg_pvalue,
            "spread_mean": stats.spread_mean,
            "spread_std": stats.spread_std,
            "half_life_days": stats.half_life_days,
            "legs_are_i1": stats.legs_are_i1,
            "n_obs": stats.n_obs,
            "selected": selected,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Strongest evidence first; ties to the faster reverter.
    return out.sort_values(["eg_pvalue", "half_life_days"]).reset_index(drop=True)


def run_block1(log_prices: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Full-sample screen, for exposition only. Block 3 re-screens per window."""
    print("=" * 78)
    print("BLOCK 1 — Cointegration screen across the whole universe")
    print("=" * 78)

    results = screen_pairs(log_prices)
    n_tested = len(results)
    n_hits = int((results["eg_pvalue"] <= 0.05).sum())
    n_selected = int(results["selected"].sum())
    expected_by_chance = 0.05 * n_tested

    print(f"  Pairs tested                     : {n_tested}")
    print(f"  p <= 0.05 (raw)                  : {n_hits}")
    print(f"  ...expected by CHANCE if nothing "
          f"were cointegrated: {expected_by_chance:.1f}")
    print(f"  Surviving all tradability filters: {n_selected}")
    print(f"  Excess of hits over chance       : {n_hits - expected_by_chance:+.1f}")
    print()

    print(f"  Strongest {top_n} candidates (by Engle-Granger p-value):")
    print("  " + "-" * 74)
    for _, r in results.head(top_n).iterrows():
        hl = f"{r.half_life_days:6.1f}" if np.isfinite(r.half_life_days) else "   inf"
        mark = "SELECTED" if r.selected else "        "
        print(f"  {mark} {r.pair:<12} p={r.eg_pvalue:.4f}  beta={r.beta:6.3f}  "
              f"half-life={hl}d  {'same-sector' if r.same_sector else 'cross-sector'}")

    print()
    print("  POSITIVE CONTROLS — funds tracking the same underlying asset.")
    print("  These should be detected trivially. They are not:")
    print()
    expense_ratios = {"SPY": 0.0945, "IVV": 0.03, "VOO": 0.03, "GLD": 0.40, "IAU": 0.25}

    print(f"    {'pair':<10} {'p(level)':>9} {'p(trend)':>9} {'drift(b=1)':>11} "
          f"{'R2':>6} {'fee gap':>9} {'sd':>8}")
    print("    " + "-" * 70)
    diagnostics = {}
    for a, b in [("IVV", "SPY"), ("SPY", "VOO"), ("GLD", "IAU")]:
        if a in log_prices.columns and b in log_prices.columns:
            d = diagnose_pair(log_prices[a], log_prices[b], f"{a}-{b}")
            diagnostics[d["pair"]] = d
            # log(A/B) should drift by A's fee less B's.
            fee_gap = (expense_ratios.get(b, np.nan) - expense_ratios.get(a, np.nan)) * 100
            print(f"    {d['pair']:<10} {d['p_level']:9.4f} {d['p_trend']:9.4f} "
                  f"{d['drift_bps_per_year_unit_beta']:+8.1f}bp/y {d['trend_r2_unit_beta']:6.3f} "
                  f"{fee_gap:+7.1f}bp {d['spread_std']:8.5f}")
    print()
    print("  p(level) is the screen's test and fails for every control. p(trend)")
    print("  allows a deterministic drift, and GLD-IAU then comes back at 0.0000:")
    gld = diagnostics.get("GLD-IAU")
    if gld:
        print(f"  its tracking difference falls "
              f"{abs(gld['drift_bps_per_year_unit_beta']):.1f}bp/year "
              f"(R^2={gld['trend_r2_unit_beta']:.3f}) against a 15bp")
        print("  fee gap, so the spread is stationary around a sloping line rather")
        print("  than a constant. SPY-IVV fails either way: it has a 2016 level")
    else:
        print("  SPY-IVV fails either way, because it has a 2016 level")
    print("  shift on top of its fee drift, and unit-root tests are blind to shifts")
    print("  (Perron 1989). Note the spread sd of ~15bp: the most reliably related")
    print("  pairs in existence are untradeable, because the spread never moves far")
    print("  enough to pay for two legs of costs.")
    print("=" * 78)
    return results


if __name__ == "__main__":
    from data_pipeline import to_log_prices

    prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
    logp = to_log_prices(prices)
    table = run_block1(logp)
    table.to_csv("data/pair_screen_full_sample.csv", index=False)
    print("\nSaved full-sample screen -> data/pair_screen_full_sample.csv")
