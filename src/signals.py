"""Block 2: spread -> position.

Formation-window parameters are frozen in FormationStats, the z-score is built
from them, and generate_positions turns that path into -1 / 0 / +1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cointegration import estimate_hedge_ratio, spread_series


@dataclass(frozen=True)
class FormationStats:
    """Hedge ratio and spread moments from the formation window.

    frozen=True is the point: nothing downstream can update these with
    out-of-sample information.
    """
    alpha: float
    beta: float
    mean: float
    sd: float
    n_obs: int

    def weights(self) -> tuple[float, float]:
        """Weights on (A, B) for +1 unit of long spread, gross exposure 1.

        Normalizing gross to 1 keeps Sharpes and drawdowns comparable across
        pairs with different betas. Approximate: beta relates log changes, while
        these are applied to simple returns of dollar notionals.
        """
        denom = 1.0 + abs(self.beta)
        w_a = 1.0 / denom
        w_b = -self.beta / denom
        return w_a, w_b


def fit_formation(log_a: pd.Series, log_b: pd.Series) -> FormationStats:
    """The only place these parameters are estimated."""
    aligned = pd.concat([log_a.rename("a"), log_b.rename("b")], axis=1).dropna()
    a, b = aligned["a"], aligned["b"]

    alpha, beta = estimate_hedge_ratio(a, b)
    spread = spread_series(a, b, alpha, beta)

    return FormationStats(
        alpha=alpha,
        beta=beta,
        mean=float(spread.mean()),
        sd=float(spread.std(ddof=1)),
        n_obs=int(len(aligned)),
    )


def zscore_out_of_sample(
    log_a: pd.Series,
    log_b: pd.Series,
    stats: FormationStats,
) -> pd.Series:
    """z = (spread - formation mean) / formation sd.

    No .mean(), .std() or .rolling() over the data being scored — every constant
    comes from stats, which is what makes this out-of-sample by construction. If
    the relationship drifts, z will sit off zero for months; that is the real
    risk showing up, not a bug.
    """
    if stats.sd <= 0:
        raise ValueError("Formation spread has zero standard deviation.")
    aligned = pd.concat([log_a.rename("a"), log_b.rename("b")], axis=1).dropna()
    spread = spread_series(aligned["a"], aligned["b"], stats.alpha, stats.beta)
    return ((spread - stats.mean) / stats.sd).rename("zscore")


def rolling_zscore(
    log_a: pd.Series,
    log_b: pd.Series,
    stats: FormationStats,
    window: int = 60,
) -> pd.Series:
    """Robustness variant: trailing mean/sd, frozen hedge ratio.

    Still causal, but it asks whether the spread reverts to its recent average,
    which adapts to drift and can chase a broken pair down forever.
    """
    aligned = pd.concat([log_a.rename("a"), log_b.rename("b")], axis=1).dropna()
    spread = spread_series(aligned["a"], aligned["b"], stats.alpha, stats.beta)

    mu = spread.rolling(window, min_periods=window).mean()
    sd = spread.rolling(window, min_periods=window).std(ddof=1)
    return ((spread - mu) / sd).rename("zscore")


@dataclass
class TradeRule:
    """Thresholds in z-score units: enter at 2sd, take profit at 0.5sd, abandon
    at 4sd, and give up after 60 days (~2-3 half-lives).

    Textbook values, fixed before the first run. parameter_sensitivity reports
    the whole surface so the reader can see how much is threshold-picking.
    """
    entry: float = 2.0
    exit: float = 0.5
    stop: float = 4.0
    max_hold: int = 60


def generate_positions(z: pd.Series, rule: TradeRule = TradeRule()) -> pd.DataFrame:
    """z-score path -> position path, with entry/exit events.

    Sign convention: z very negative means the spread is cheap, so position +1.
    We bet against the deviation.

    The loop is explicit because the rule is path-dependent — a vectorized
    np.where would flip position every time z crossed a threshold.
    """
    positions: list[int] = []
    held: list[int] = []
    events: list[str] = []

    state = 0
    days = 0
    disarmed = False

    for z_t in z.to_numpy():
        event = ""

        if np.isnan(z_t):      # rolling warm-up: hold, don't guess
            positions.append(state)
            held.append(days)
            events.append(event)
            if state != 0:
                days += 1
            continue

        if state == 0:
            # After a stop-out the spread is still beyond the entry threshold,
            # so a naive machine re-enters next bar at a worse price, in the
            # trade it just declared broken. Entries stay disarmed until |z| is
            # back inside the band.
            if disarmed:
                if abs(z_t) < rule.entry:
                    disarmed = False
            elif z_t <= -rule.entry:
                state, days, event = 1, 0, "entry_long"
            elif z_t >= rule.entry:
                state, days, event = -1, 0, "entry_short"
        else:
            # Stop first: a broken relationship outranks the profit target.
            if abs(z_t) >= rule.stop:
                state, days, event = 0, 0, "exit_stop"
                disarmed = True
            elif abs(z_t) <= rule.exit:
                state, days, event = 0, 0, "exit_target"
            elif days >= rule.max_hold:
                state, days, event = 0, 0, "exit_time"

        positions.append(state)
        held.append(days)
        events.append(event)
        if state != 0:
            days += 1

    return pd.DataFrame(
        {"position": positions, "days_held": held, "event": events},
        index=z.index,
    )


def summarize_signal(signal: pd.DataFrame) -> dict[str, float]:
    """Trade counts, time in market and exit reasons.

    Worth reading before any Sharpe: two trades in three years is not a track
    record, and 95% time in market is a directional bet.
    """
    pos = signal["position"]
    entries = signal["event"].str.startswith("entry").sum()
    return {
        "n_trades": int(entries),
        "time_in_market": float((pos != 0).mean()),
        "n_stopped_out": int((signal["event"] == "exit_stop").sum()),
        "n_time_stopped": int((signal["event"] == "exit_time").sum()),
        "n_target_exits": int((signal["event"] == "exit_target").sum()),
        "avg_days_held": float(signal.loc[signal["event"].str.startswith("exit"), "days_held"].mean())
        if entries else float("nan"),
    }
