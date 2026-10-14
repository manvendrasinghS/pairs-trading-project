"""Tests for the pairs-trading machinery.

Backtest bugs don't crash, they just make the results better, so most of these
pin behaviour on synthetic data whose answer is known. The ones worth reading
first: test_cointegrated_pair_is_detected, test_independent_walks_not_cointegrated
and test_no_lookahead_future_cannot_matter.

    PYTHONPATH=src pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import (
    CostModel,
    backtest_pair,
    extract_trades,
    max_drawdown,
    performance_metrics,
    spread_returns,
)
from cointegration import (
    adf_pvalue,
    engle_granger,
    estimate_hedge_ratio,
    half_life,
    is_unit_root,
    screen_pairs,
)
from data_pipeline import candidate_pairs, to_log_prices, to_simple_returns
from signals import (
    FormationStats,
    TradeRule,
    fit_formation,
    generate_positions,
    zscore_out_of_sample,
)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2015-01-01", periods=n, freq="B")


def _random_walk(n: int, seed: int, sigma: float = 0.01, start: float = 4.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(start + np.cumsum(rng.normal(0, sigma, n)), index=_dates(n))


def _cointegrated_pair(
    n: int = 1200, seed: int = 7, beta: float = 1.3, noise_sd: float = 0.02,
) -> tuple[pd.Series, pd.Series]:
    """log_a = 0.5 + beta*log_b + AR(1) noise over a random-walk log_b.

    AR(1) rather than white noise, because white-noise spreads are
    unrealistically easy to detect. phi=0.94 is a half-life of ~11 days.
    """
    rng = np.random.default_rng(seed)
    log_b = _random_walk(n, seed=seed + 1)

    phi, eps = 0.94, rng.normal(0, noise_sd, n)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + eps[i]

    log_a = pd.Series(0.5 + beta * log_b.to_numpy() + noise, index=log_b.index)
    return log_a, log_b


def test_simple_returns_arithmetic():
    prices = pd.DataFrame({"X": [100.0, 110.0, 99.0]}, index=_dates(3))
    r = to_simple_returns(prices)["X"]
    assert np.isclose(r.iloc[0], 0.10)
    assert np.isclose(r.iloc[1], -0.10)


def test_log_prices_reject_non_positive():
    with pytest.raises(ValueError):
        to_log_prices(pd.DataFrame({"X": [1.0, 0.0]}, index=_dates(2)))


def test_candidate_pairs_count():
    for n in (2, 5, 25):
        tickers = [f"T{i}" for i in range(n)]
        assert len(candidate_pairs(tickers)) == n * (n - 1) // 2


def test_portfolio_return_is_weighted_sum_of_simple_returns():
    """r_portfolio = w_A*r_A + w_B*r_B at gross exposure 1."""
    stats = FormationStats(alpha=0.0, beta=1.0, mean=0.0, sd=1.0, n_obs=100)
    w_a, w_b = stats.weights()
    assert np.isclose(w_a, 0.5) and np.isclose(w_b, -0.5)
    assert np.isclose(abs(w_a) + abs(w_b), 1.0)

    ret_a = pd.Series([0.02, -0.01], index=_dates(2))
    ret_b = pd.Series([0.01, 0.01], index=_dates(2))
    sr = spread_returns(ret_a, ret_b, stats)
    assert np.isclose(sr.iloc[0], 0.5 * 0.02 - 0.5 * 0.01)
    assert np.isclose(sr.iloc[1], 0.5 * -0.01 - 0.5 * 0.01)


def test_random_walk_looks_like_unit_root():
    rw = _random_walk(800, seed=42)
    assert adf_pvalue(rw) > 0.05
    assert is_unit_root(rw) is True


def test_stationary_series_is_not_unit_root():
    rng = np.random.default_rng(3)
    noise = pd.Series(rng.normal(0, 1, 800), index=_dates(800))
    assert adf_pvalue(noise) < 0.05
    assert is_unit_root(noise) is False


def test_hedge_ratio_recovers_known_beta():
    log_a, log_b = _cointegrated_pair(beta=1.3, noise_sd=0.01)
    alpha, beta = estimate_hedge_ratio(log_a, log_b)
    assert beta == pytest.approx(1.3, abs=0.05)
    assert alpha == pytest.approx(0.5, abs=0.20)


def test_cointegrated_pair_is_detected():
    """Power: real tethers must come back with a small p-value, over 5 seeds."""
    for seed in (7, 11, 13, 17, 19):
        stats = engle_granger(*_cointegrated_pair(seed=seed), "A", "B")
        assert stats.eg_pvalue < 0.01, (
            f"seed {seed}: failed to detect real cointegration (p={stats.eg_pvalue})"
        )


def test_leg_check_size_on_random_walks():
    """Measures the leg gate's false-rejection rate: 2/60 at 5%, 0/60 at 1%.

    This is why LEG_UNIT_ROOT_ALPHA is 0.01. At 5% the gate was throwing away
    real pairs — one with cointegration p = 3.6e-14 — on ADF noise. Up to 2 is
    allowed at 1% so the test isn't itself a coin flip.
    """
    pvalues = [adf_pvalue(_random_walk(1200, seed=1000 + s)) for s in range(60)]
    rejected_at_1pct = sum(p <= 0.01 for p in pvalues)
    rejected_at_5pct = sum(p <= 0.05 for p in pvalues)

    assert rejected_at_1pct <= 2, (
        f"leg gate wrongly rejected {rejected_at_1pct}/60 genuine random walks — "
        "it is discarding real pairs"
    )
    assert rejected_at_1pct <= rejected_at_5pct, "1% gate should be no looser than 5%"

    rng = np.random.default_rng(99)
    stationary = pd.Series(rng.normal(0, 1, 1200), index=_dates(1200))
    assert is_unit_root(stationary) is False, "gate failed to catch a stationary leg"


def test_independent_walks_not_cointegrated():
    """Specificity, as a rate over 20 seed pairs rather than a single draw.

    A few seeds do print p < 0.05, which is the multiple-testing problem Block 4
    exists for, so this asserts a rate rather than zero.
    """
    false_positives = 0
    trials = 20
    for s in range(trials):
        a = _random_walk(600, seed=100 + 2 * s)
        b = _random_walk(600, seed=101 + 2 * s)
        if engle_granger(a, b, "A", "B").eg_pvalue < 0.05:
            false_positives += 1
    assert false_positives <= 4, (
        f"{false_positives}/{trials} false positives — far above the 5% nominal rate, "
        "which would mean the test is mis-specified"
    )


def test_half_life_of_known_ar1():
    """Against the analytic -ln(2)/ln(phi) = 6.58 days for phi = 0.9."""
    rng = np.random.default_rng(5)
    phi, n = 0.90, 4000
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = phi * s[i - 1] + rng.normal(0, 0.01)
    spread = pd.Series(s, index=_dates(n))

    expected = -np.log(2) / np.log(phi)
    assert half_life(spread) == pytest.approx(expected, rel=0.15)


def test_half_life_of_random_walk_is_untradeable():
    """Not "infinite" — the OLS bias on phi gives random walks a finite
    half-life (283 to 13,149 days here). The useful claim is that it lands well
    outside the 5-60 day band screen_pairs accepts."""
    for seed in (21, 22, 23, 24, 25):
        hl = half_life(_random_walk(1500, seed=seed))
        assert not np.isfinite(hl) or hl > 100, (
            f"seed {seed}: random walk produced a suspiciously tradeable "
            f"half-life of {hl:.1f} days"
        )


def test_screen_reports_every_pair_not_just_winners():
    """Block 4 needs the count of tests run, so the screen must report all pairs."""
    frame = pd.DataFrame({
        "AAA": _random_walk(400, seed=1),
        "BBB": _random_walk(400, seed=2),
        "CCC": _random_walk(400, seed=3),
    })
    out = screen_pairs(frame)
    assert len(out) == 3
    assert "selected" in out.columns
    assert "eg_pvalue" in out.columns


def test_zscore_uses_only_frozen_formation_stats():
    """Tamper with the tail: early z-scores must not move.

    If any normalization constant came from the series being scored, changing
    the future would change the past.
    """
    log_a, log_b = _cointegrated_pair(n=600, seed=13)
    stats = fit_formation(log_a.iloc[:400], log_b.iloc[:400])

    z_original = zscore_out_of_sample(log_a, log_b, stats)

    tampered = log_a.copy()
    tampered.iloc[500:] += 0.5
    z_tampered = zscore_out_of_sample(tampered, log_b, stats)

    pd.testing.assert_series_equal(z_original.iloc[:500], z_tampered.iloc[:500])


def test_position_sign_convention():
    """Cheap spread (z negative) is +1, rich spread is -1.

    A flipped sign inverts the strategy and often looks profitable, since
    inverting a losing rule gives a winning one.
    """
    z = pd.Series([0.0, -2.5, -2.0, -0.2, 0.0, 2.5, 0.1], index=_dates(7))
    sig = generate_positions(z, TradeRule(entry=2.0, exit=0.5, stop=4.0, max_hold=60))

    assert sig["position"].iloc[0] == 0     # z=0.0  -> flat
    assert sig["position"].iloc[1] == 1     # z=-2.5 -> cheap -> long
    assert sig["position"].iloc[2] == 1     # still open
    assert sig["position"].iloc[3] == 0     # inside the exit band -> close
    assert sig["position"].iloc[5] == -1    # z=+2.5 -> rich -> short
    assert sig["position"].iloc[6] == 0     # closed at target


def test_stop_loss_fires_before_target():
    """The stop is checked first, so a huge move exits instead of doubling down."""
    z = pd.Series([0.0, 2.5, 3.0, 4.5, 4.6], index=_dates(5))
    sig = generate_positions(z, TradeRule(entry=2.0, exit=0.5, stop=4.0))
    assert sig["position"].iloc[1] == -1
    assert sig["event"].iloc[3] == "exit_stop"
    assert sig["position"].iloc[3] == 0
    assert sig["position"].iloc[4] == 0     # no re-entry beyond the stop


def test_stop_out_disarms_until_spread_calms():
    """No new entry until |z| is back inside the entry band.

    The first version stopped out at z=4.5 and re-entered short at 4.6 on the
    next bar. Here: stop at index 3, flat while z stays extreme, re-armed at 7,
    trading again at 8.
    """
    z = pd.Series([0.0, 2.5, 3.0, 4.5, 4.6, 5.0, 4.2, 1.0, 2.5, 2.6],
                  index=_dates(10))
    sig = generate_positions(z, TradeRule(entry=2.0, exit=0.5, stop=4.0))

    assert sig["event"].iloc[3] == "exit_stop"
    assert (sig["position"].iloc[4:7] == 0).all(), "re-entered while still stopped out"
    assert sig["position"].iloc[7] == 0
    assert sig["position"].iloc[8] == -1


def test_time_stop_fires():
    z = pd.Series([2.5] * 12, index=_dates(12))   # past entry, never exits
    sig = generate_positions(z, TradeRule(entry=2.0, exit=0.5, stop=4.0, max_hold=5))
    assert "exit_time" in set(sig["event"])
    assert sig["position"].iloc[-1] == 0 or sig["position"].iloc[-1] == -1


def test_no_lookahead_future_cannot_matter():
    """Rewrite the last 50 days of both legs; earlier net returns must be identical.

    The most valuable test here, because a leak is invisible in the output.
    """
    log_a, log_b = _cointegrated_pair(n=700, seed=17)
    stats = fit_formation(log_a.iloc[:400], log_b.iloc[:400])

    prices_a = np.exp(log_a.iloc[400:])
    prices_b = np.exp(log_b.iloc[400:])
    ret_a, ret_b = prices_a.pct_change().dropna(), prices_b.pct_change().dropna()

    z = zscore_out_of_sample(log_a.iloc[400:], log_b.iloc[400:], stats)
    base = backtest_pair(ret_a, ret_b, z, stats)

    log_a2, log_b2 = log_a.copy(), log_b.copy()
    log_a2.iloc[-50:] += 0.30
    log_b2.iloc[-50:] -= 0.30
    pa2, pb2 = np.exp(log_a2.iloc[400:]), np.exp(log_b2.iloc[400:])
    z2 = zscore_out_of_sample(log_a2.iloc[400:], log_b2.iloc[400:], stats)
    tampered = backtest_pair(pa2.pct_change().dropna(), pb2.pct_change().dropna(), z2, stats)

    n_safe = len(base) - 51
    pd.testing.assert_series_equal(
        base["net_return"].iloc[:n_safe],
        tampered["net_return"].iloc[:n_safe],
        check_names=False,
    )


def test_position_is_shifted_before_earning_returns():
    """Day 1's move is large; without the shift the strategy would capture it."""
    stats = FormationStats(alpha=0.0, beta=1.0, mean=0.0, sd=1.0, n_obs=100)
    idx = _dates(4)
    ret_a = pd.Series([0.00, 0.10, 0.00, 0.00], index=idx)
    ret_b = pd.Series([0.00, 0.00, 0.00, 0.00], index=idx)
    z = pd.Series([-2.5, -2.5, -0.1, 0.0], index=idx)        # entry at day 0

    bt = backtest_pair(ret_a, ret_b, z, stats, costs=CostModel(0.0, 0.0))
    assert bt["position"].iloc[0] == 1
    assert bt["position_live"].iloc[0] == 0
    assert bt["position_live"].iloc[1] == 1
    assert np.isclose(bt["gross_return"].iloc[0], 0.0)
    assert np.isclose(bt["gross_return"].iloc[1], 0.05)


def test_costs_reduce_returns_monotonically():
    """Catches a sign error on costs, or charging them on |position| not turnover."""
    log_a, log_b = _cointegrated_pair(n=800, seed=19)
    stats = fit_formation(log_a.iloc[:400], log_b.iloc[:400])
    pa, pb = np.exp(log_a.iloc[400:]), np.exp(log_b.iloc[400:])
    z = zscore_out_of_sample(log_a.iloc[400:], log_b.iloc[400:], stats)

    totals = []
    for bps in (0.0, 5.0, 20.0, 100.0):
        bt = backtest_pair(pa.pct_change().dropna(), pb.pct_change().dropna(), z, stats,
                           costs=CostModel(one_way_bps=bps, borrow_bps_annual=0.0))
        totals.append(bt["net_return"].sum())
    assert totals == sorted(totals, reverse=True), f"costs not monotone: {totals}"


def test_no_cost_when_flat():
    """Otherwise every cash roll in the walk-forward bleeds for doing nothing."""
    stats = FormationStats(alpha=0.0, beta=1.0, mean=0.0, sd=1.0, n_obs=100)
    idx = _dates(5)
    zeros = pd.Series([0.0] * 5, index=idx)
    z = pd.Series([0.0, 0.1, -0.1, 0.0, 0.1], index=idx)   # never reaches entry
    bt = backtest_pair(zeros, zeros, z, stats, costs=CostModel(50.0, 500.0))
    assert (bt["position"] == 0).all()
    assert np.isclose(bt["cost"].sum(), 0.0)


def test_max_drawdown_hand_computed():
    equity = pd.Series([100.0, 120.0, 60.0, 90.0], index=_dates(4))
    assert max_drawdown(equity) == pytest.approx(-0.50)


def test_sharpe_of_constant_returns_is_undefined_not_infinite():
    """A pair that never traded has sd 0; inf would poison every aggregate."""
    flat = pd.Series([0.001] * 50, index=_dates(50))
    m = performance_metrics(flat)
    assert np.isnan(m["sharpe"])


def test_metrics_on_known_series():
    r = pd.Series([0.01, -0.02, 0.03], index=_dates(3))
    m = performance_metrics(r)
    expected = (1.01 * 0.98 * 1.03) - 1.0
    assert m["total_return"] == pytest.approx(expected)


def test_extract_trades_pairs_entries_with_exits():
    stats = FormationStats(alpha=0.0, beta=1.0, mean=0.0, sd=1.0, n_obs=100)
    idx = _dates(8)
    ret = pd.Series([0.0, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0], index=idx)
    z = pd.Series([-2.5, -2.0, -0.1, 0.0, 2.5, 2.0, 0.1, 0.0], index=idx)

    bt = backtest_pair(ret, pd.Series(0.0, index=idx), z, stats, costs=CostModel(0.0, 0.0))
    trades = extract_trades(bt)
    assert len(trades) == 2
    assert set(trades["direction"]) == {"long_spread", "short_spread"}
    assert trades["exit_reason"].str.startswith("exit").all()


def test_end_to_end_on_synthetic_cointegrated_pair():
    """Smoke test of fit -> z -> positions -> P&L on constructed mean reversion.

    Gross return, since whether real pairs survive real costs is the actual
    research question and no test should assume the answer.
    """
    log_a, log_b = _cointegrated_pair(n=1500, seed=23, noise_sd=0.03)
    stats = fit_formation(log_a.iloc[:800], log_b.iloc[:800])

    pa, pb = np.exp(log_a.iloc[800:]), np.exp(log_b.iloc[800:])
    z = zscore_out_of_sample(log_a.iloc[800:], log_b.iloc[800:], stats)
    bt = backtest_pair(pa.pct_change().dropna(), pb.pct_change().dropna(), z, stats,
                       costs=CostModel(0.0, 0.0))

    assert (bt["position"] != 0).sum() > 20, "strategy never traded a mean-reverting spread"
    assert bt["gross_return"].sum() > 0, "failed to profit from constructed mean reversion"
