# Does Cointegration Survive Out of Sample? A Pairs Trading Study

**Research question.** If you search a universe of liquid US equities for
cointegrated pairs, how much of what you find is real?

Not "can I build a profitable pairs strategy" — that question invites you to keep
adjusting until the answer is yes. The question here is about the search itself:
what fraction of a cointegration screen's output is signal, what fraction is the
arithmetic of running hundreds of tests, and how much of a pairs backtest's
apparent edge disappears once the selection step is no longer allowed to see the
data it is judged on.

The answer, over 30 liquid US stocks and ETFs from January 2012 to August 2026
(3,675 trading days, 435 candidate pairs):

> A broad cointegration screen finds about as many "significant" pairs as pure
> chance would produce, none of them tradeable, and a properly walk-forwarded
> strategy built on them loses money **even before transaction costs**. The same
> pairs, with the same frozen parameters and the same trading rule, show a mean
> in-sample Sharpe of **+1.54** and an out-of-sample Sharpe of **−0.11**. That
> 1.65-point gap is the price of hindsight, measured.

---

## Headline findings

| # | Finding | Number |
|---|---|---|
| 1 | Candidate pairs screened (full sample) | 435 |
| 2 | Pairs "significant" at p ≤ 0.05 | 23 |
| 3 | False positives expected by chance alone (435 × 0.05) | **21.75** |
| 4 | Significance rate: same-sector vs cross-sector | **2.9% vs 5.5%** |
| 5 | Pairs passing the full tradability screen (full sample) | **0** |
| 6 | Out-of-sample Sharpe, net of costs, 3,150 days / 25 rolls | **−0.11** |
| 7 | Out-of-sample annualised return / volatility | −0.56% / 4.15% |
| 8 | Mean **in-sample** Sharpe, same pairs and same rule | **+1.54** |
| 9 | In-sample → out-of-sample decay | **1.65 Sharpe points** |
| 10 | Selected pairs still cointegrated in the next 6 months | **3.3%** |
| 11 | corr(formation p-value, out-of-sample Sharpe), Spearman | **−0.04** |
| 12 | Sharpe at **zero** transaction cost | −0.05 |
| 13 | Best of 18 threshold combinations tested | Sharpe **+0.06** |
| 14 | Pairs surviving Bonferroni / Benjamini–Hochberg | **0 / 0** |
| 15 | Deflated Sharpe Ratio (N = 195,750 trials) | **0.0%** |
| 16 | Correlation of strategy returns to SPY | +0.042 |
| 17 | SPY buy-and-hold over the same 3,150 days | Sharpe 0.84 |

Read rows 2 and 3 together: 23 hits against 21.75 expected by chance is an excess
of 1.25 pairs. Then read row 4: the pairs with an actual economic reason to
co-move were *less* likely to pass than random cross-sector combinations, and the
cross-sector rate of 5.5% is essentially the nominal size of the test, which is
what a pure null distribution looks like.

This is a negative result, and it is the point. The machinery that establishes it
— walk-forward with pair selection inside the loop, structurally frozen
parameters, adversarial lookahead tests, a cost-sensitivity curve and a full
multiple-testing audit — was built before the answer was known.

---

## Contents

- [Cointegration, not correlation](#cointegration-not-correlation)
- [The trade, step by step](#the-trade-step-by-step)
- [Method: the five blocks](#method-the-five-blocks)
- [Results](#results)
- [Figures](#figures)
- [What I got wrong](#what-i-got-wrong)
- [Repository layout](#repository-layout)
- [How to reproduce](#how-to-reproduce)
- [Limitations](#limitations)
- [Extensions](#extensions)
- [References](#references)

---

## Cointegration, not correlation

Two companies in the same business face the same forces, so although each share
price wanders unpredictably, the *relationship* between them may be stable. When
one gets expensive relative to the other, a pairs trader buys the cheap one,
shorts the expensive one in the amount that cancels their shared exposure, and
waits. The bet is not "Coca-Cola will go up"; it is "Coca-Cola and Pepsi have
drifted apart and will come back together", which can pay off whether the market
rises or falls. Hence *market-neutral*: the measured correlation of this
strategy's returns to SPY was +0.042.

The distinction the method rests on:

- **Correlation** is short-term co-movement of *returns*. It says nothing about
  where the levels end up. Two assets can be 99% correlated daily and drift
  arbitrarily far apart over years.
- **Cointegration** is a long-term relationship between *levels*: two I(1) series
  are cointegrated if some β makes `y_t − β·x_t` stationary. That stationary
  combination is the spread, and it is the thing you trade.

The usual picture is a drunk and their dog: both wander randomly, but the leash
keeps the distance stationary. Correlation measures whether they step in the same
direction; cointegration measures whether the leash exists. You need the leash.

Stationarity is what makes the trade meaningful at all. A stationary series *has*
a mean to revert to and its deviations decay. For a random walk, "the spread is
2 standard deviations below its mean" is a statement about the sample you happened
to look at, not about the future. So the screening step is one question asked 435
times: is this linear combination stationary, or does it only look stationary in
this sample?

This repo contains a clean demonstration in both directions. SPY and IVV, two ETFs
tracking the same index, have return correlation above 0.99 and **fail** the
cointegration test at p = 0.74. The textbook pair KO–PEP has p = 0.99 and an
estimated half-life of 5,895 days. Both have real explanations — see
[Results](#results).

### Where the profit is supposed to come from

Three candidate stories, and being honest about which one your evidence supports
is most of the work. (1) Liquidity provision, when a fund liquidates one name and
whoever takes the other side is paid — real, but the payment goes to whoever is
fastest and cheapest. (2) Slow information diffusion, which was true in 1990 and
is largely gone in liquid US large caps. (3) There is no edge left at this
frequency in this universe, where you compete with people who see the order book.

The results here are consistent with the third story.

---

## The trade, step by step

**1. Check both legs are I(1).** ADF test on each log price series, requiring each
to *fail* to reject the unit root. If one leg is already stationary, any
stationary "spread" is an artefact. (`cointegration.is_unit_root`)

**2. Estimate the tether.** `log A_t = α + β·log B_t + e_t`. β is the hedge ratio,
and on logs it is an elasticity. (`cointegration.estimate_hedge_ratio`)

**3. Test whether the residual is stationary.** The residual *is* the spread, and
the Engle–Granger two-step test asks whether it is stationary.
(`cointegration.engle_granger`)

> A trap worth knowing: you cannot run a standard ADF test on OLS residuals with
> standard critical values. OLS *chose* β to minimise residual variance, so it has
> already made the residual look as stationary as possible and standard critical
> values over-reject. `statsmodels.tsa.stattools.coint` applies values that
> account for the pre-estimation of β.

**4. Check the reversion is fast enough to trade.** Fit an AR(1) and compute the
half-life: `Δs_t = a + b·s_{t−1} + ε_t` → `φ = 1 + b` → `HL = −ln(2)/ln(φ)`.
Required band here is 5–60 days: below 5 it is microstructure noise untradeable on
daily closes, above 60 a position is held for months, which means few independent
trades, more borrow cost and more time for the relationship to end.
(`cointegration.half_life`)

**5. Freeze the parameters.** α, β and the spread's mean and standard deviation,
computed on the formation window only and stored in an immutable object.
(`signals.FormationStats`, frozen dataclass)

**6. Standardise and trade a fixed rule.** `z_t = (s_t − μ)/σ`, then:

```
z ≤ −2.0   →   long the spread   (buy A, short B)  — A is cheap relative to B
z ≥ +2.0   →   short the spread  (short A, buy B)  — A is expensive
|z| ≤ 0.5  →   flat   (take profit)
|z| ≥ 4.0  →   flat   (stop loss, then disarm — see below)
held 60d   →   flat   (time stop)
```

The sign is worth pausing on. If z is very negative the spread is unusually low,
so A is cheap relative to B, and betting on reversion means betting the spread
rises: buy A, short B. Get it backwards and the strategy is exactly wrong while
still running happily, with an equity curve that is merely upside down. There is a
unit test for the sign. (`signals.generate_positions`)

**7. Size the legs.** `w_A = 1/(1+|β|)`, `w_B = −β/(1+|β|)`, so
`|w_A| + |w_B| = 1`. Normalising gross exposure to 1 makes every return a return
on gross capital deployed, which keeps Sharpes comparable across pairs with very
different hedge ratios. (`signals.FormationStats.weights`)

**8. Compute P&L honestly.**

```
r_t = position_{t−1} · (w_A·r_{A,t} + w_B·r_{B,t}) − costs_t − borrow_t
```

`position_{t−1}`, not `position_t`: the position was decided at yesterday's close,
so it earns today's return, and that single `.shift(1)` is the difference between a
backtest and a fantasy. The returns are **simple**, not log, because
`r_p = w_A·r_A + w_B·r_B` holds for simple returns and is exactly false for log
returns. (`backtest.backtest_pair`)

---

## Method: the five blocks

Each module is a standalone, runnable "block". Its docstring states what it does
and the trap it exists to avoid; the reasoning lives here.

### Block 0 — Data (`src/data_pipeline.py`)

30 tickers, 2012-01-03 to 2026-08-14, 3,675 trading days of split- and
dividend-adjusted closes. 25 large caps and sector ETFs grouped by economic
driver, plus **5 positive controls**: SPY/IVV/VOO on the same index, GLD/IAU on
the same metal. The controls must come out cointegrated, since their ratios are
pinned by arbitrage, so if the test fails to flag them the test is broken.

Three decisions worth defending:

- **Cross-sector combinations are deliberately included.** 401 of the 435 pairs
  have no economic reason to be related, and that is the control group: without
  pairs that *should* fail you cannot show the screen separates signal from noise.
  The same-sector (2.9%) versus cross-sector (5.5%) comparison turned out to be
  the most informative statistic in the study.
- **`auto_adjust=True` is essential.** Un-adjusted prices jump on split dates, so
  a 4-for-1 split becomes a 4σ signal and the backtest trades an accounting
  event. `sanity_checks()` has a tripwire for single-day moves above 50%.
- **Incomplete days are dropped, never forward-filled.** If leg A has no print and
  leg B moves 2%, filling A invents a 2% spread move and the strategy trades the
  invention. Thin tickers are dropped first, then incomplete days; the other order
  lets one short history delete years of good data.

### Block 1 — The screen (`src/cointegration.py`)

All 435 pairs, five filters:

| Filter | Threshold | Why |
|---|---|---|
| Engle–Granger p-value | ≤ 0.05 | the spread is stationary |
| Both legs I(1) | ADF p > 0.01 each | cointegration is undefined otherwise |
| Half-life | 5–60 days | fast enough to trade, slow enough to be real |
| β > 0 | — | a negative hedge ratio is a directional bet in disguise |
| σ of spread | > 0 | degenerate pairs |

The leg test uses **α = 0.01, not 0.05**, because it is used backwards from its
normal purpose: we are trying to confirm the null, so a false rejection discards a
legitimate pair. Measured on 60 simulated genuine random walks, 2/60 were wrongly
rejected at 5% and **0/60 at 1%**.

`screen_pairs` returns all 435 rows with a `selected` boolean rather than only the
winners, because the number of tests run is itself a quantity that has to be
reported. A **Johansen** trace test is available as a cross-check: Engle–Granger
needs you to choose which variable goes on the left-hand side, and Johansen is
symmetric, so a pair the two disagree on is a pair whose apparent relationship
depends on an arbitrary normalisation.

### Block 2 — Signals (`src/signals.py`)

The z-score and the trade state machine, built around two lookahead traps.

**Trap 1 — the full-sample z-score.** Standardising with the full sample's mean
and standard deviation leaks the future into the signal: the spread is guaranteed
to look mean-reverting around a mean it is *defined* to average out to. The fix is
structural rather than a comment — `FormationStats` is frozen and can only come
from `fit_formation()`, and `zscore_out_of_sample()` contains no `.mean()`,
`.std()` or `.rolling()` call over the data it is scoring.

**Trap 2 — same-bar execution.** Trading at the close that generated the signal is
a free look at the last tick of the day, and it is worth most on exactly the days
the signal fires. Fixed by `.shift(1)`, pinned by a test that vandalises the last
50 days of prices and asserts every earlier net return is unchanged.

`generate_positions` uses an explicit loop rather than vectorised `np.where`
logic, because a vectorised threshold rule is stateless: it re-decides from
scratch every day and has no concept of "I am in a trade I entered 12 days ago".
Holding periods, time stops and the re-arm rule cannot be expressed without state.

### Block 3 — The walk-forward backtest (`src/backtest.py`)

```
formation window : 504 trading days (2 years)   ← screen and fit here
trading window   : 126 trading days (6 months)  ← trade here, never screened on
step             : 126 days  →  test windows never overlap
top_k            : 5 pairs per roll, equal-weighted
costs            : 5bp one-way on turnover + 50bp/yr borrow on the short leg
result           : 25 rolls, 3,150 out-of-sample days
```

The schedule follows Gatev–Goetzmann–Rouwenhorst's 12m/6m convention, lengthened
to 24 months of formation because cointegration tests need more observations than
a distance metric does.

**The one thing that makes this the honest version:** the screen runs inside the
loop. `build_walk_forward_plan` re-screens all 435 pairs on each formation window,
and only what that window chose is traded in the following six months. A pair that
turns out to be a star later gets no credit for the years before it was selected.

Three smaller choices that matter: test windows never overlap, so the concatenated
series contains each day exactly once and the Sharpe's standard error is not
inflated by double counting; rolls where no pair passes hold cash **and the zero
is recorded**, since dropping idle windows inflates both return and Sharpe; and
weights are equal, because 5 assets over 126 days is nowhere near enough data to
estimate a covariance matrix worth optimising.

Cointegration is re-tested on each trading window purely as a diagnostic that
never feeds back into a decision, and it produced finding #10, the 3.3% survival
rate. The rule: you may look at the future to *evaluate* a method, never to make a
decision inside it.

### Block 4 — The multiple-testing audit (`src/multiple_testing.py`)

- **Expected false positives** = n × α. 435 × 0.05 = 21.75, against 23 observed.
- **Bonferroni** (α/n, family-wise error rate), reported as a bound and noted as
  conservative here because pair tests are not independent: pairs sharing a leg
  (JPM–BAC, JPM–WFC, JPM–C) are strongly dependent, so the effective number of
  independent tests is well below 435.
- **Benjamini–Hochberg FDR**, the right criterion for a screening problem, since
  it controls the expected *proportion* of discoveries that are false. Implemented
  via the rank cutoff `k = max{i : p_(i) ≤ (i/n)·q}`; comparing each p-value to its
  own row's threshold instead is where naive implementations go wrong.
- **Probabilistic and Deflated Sharpe Ratios** (Bailey & López de Prado). The PSR
  is the probability the true Sharpe exceeds a benchmark, accounting for
  track-record length and for the skew and kurtosis that inflate its standard
  error. The DSR sets that benchmark to what luck alone delivers over N trials,
  `E[max SR] ≈ σ_SR·[(1−γ)·Z(1−1/N) + γ·Z(1−1/(N·e))]` with γ the
  Euler–Mascheroni constant. N here is 435 pairs × 25 rolls × 18 threshold
  cells = **195,750**.
- **Parameter sensitivity** over entry/exit/stop thresholds (18 valid cells),
  reported in full. A diagnostic, never a selection step: the headline result
  stays at the textbook defaults chosen before any backtest was run.

---

## Results

### 1. The screen found roughly what chance predicts

| | count | share |
|---|---|---|
| Significant at p ≤ 0.05 | 23 | 5.3% |
| Expected by chance (435 × 0.05) | 21.75 | 5.0% |
| **Excess over chance** | **+1.25** | |
| Passing the full tradability screen | **0** | 0% |

Split by whether the pair has an economic reason to exist:

| | pairs | significant | rate |
|---|---|---|---|
| Same sector | 34 | 1 | **2.9%** |
| Cross sector | 401 | 22 | **5.5%** |

The cross-sector rate is essentially the nominal size of the test, and the
economically motivated pairs were *less* likely to pass than random combinations.
Whatever the screen is measuring, it is not economic relatedness. The one
same-sector hit is BAC–JPM at p = 0.046 with a half-life of 148 days; KO–PEP comes
in at p = 0.99.

### 2. Zero pairs passed the full screen, and it isn't a bug

All 23 significant pairs failed the **half-life** filter. The minimum half-life
among them was 64 days against the 60-day cap, and the median was 102 days.

Over 14 years, the linear combinations that look stationary are the ones that
revert so slowly they are barely distinguishable from a persistent drift. The band
was set from the economics of the trading window before the screen was run, and it
did not prevent the study from happening: on two-year formation windows plenty of
pairs show half-lives inside the band (median 12.7 days among the 122 selected),
so the walk-forward selected 5 pairs in 24 of 25 rolls. The contrast is itself the
finding — short-window mean reversion is mostly noise, and it disappears when you
look at 14 years instead of 2.

### 3. The walk-forward result

| | Strategy (net) | SPY buy-and-hold |
|---|---|---|
| Days | 3,150 | 3,150 |
| Annualised return | **−0.56%** | +13.86% |
| Annualised volatility | 4.15% | 17.20% |
| **Sharpe** | **−0.11** | 0.84 |
| Sortino | −0.13 | 1.03 |
| Max drawdown | −17.94% | −33.72% |
| Calmar | −0.03 | 0.41 |
| Hit rate (daily) | 49.7% | 55.2% |
| Skew / kurtosis | +0.26 / 17.29 | −0.31 / 17.34 |
| Worst day | −2.27% | −10.94% |
| Correlation to SPY | **+0.042** | 1.00 |

Roll-level Sharpes ranged from −2.43 to +2.50 with no discernible pattern, 12 of
25 positive, which is what you would get from a coin.

The strategy *was* market-neutral: +0.042 correlation to SPY at a quarter of SPY's
volatility, exactly what the construction promises. It also lost comprehensively
to buying the index, which belongs in the table because leaving it out would be
dishonest. A market-neutral sleeve is not competing to beat the index, it is
competing to add uncorrelated return — a Sharpe of +0.6 at 0.04 correlation would
have been valuable despite being below SPY's 0.84. This wasn't that.

A −17.9% drawdown on a 4.15%-volatility book is about four annual volatilities,
so it was not a volatility event but the slow bleed of a strategy with no edge,
punctuated by stop-outs.

### 4. The trade-level anatomy of "no edge"

195 trades across the 25 rolls:

| Exit reason | count | mean P&L |
|---|---|---|
| Target (\|z\| ≤ 0.5) | 69 | **+4.01%** |
| Stop loss (\|z\| ≥ 4) | 62 | **−4.03%** |
| Time stop (60 days) | 22 | −0.78% |
| Window end (forced flat) | 42 | −0.30% |
| **All** | **195** | **−0.02%** |

Win rate 50.8%, mean winner +3.12%, mean loser −3.25%, mean holding period 28.5
days, average time in market 35%. A coin flip with symmetric payoffs, where the
small asymmetry against you is the cost. Note that stop-outs (62) nearly equal
target exits (69): a third of all trades ended with the spread diverging to 4σ
rather than converging, which for a mean-reverting signal is the whole story.

### 5. The number the project exists to produce

| | Sharpe |
|---|---|
| Mean **in-sample** (formation window) | **+1.54** |
| Mean **out-of-sample** (next 6 months) | **−0.11** |
| **Decay** | **1.65 points** |

The same pairs, the same frozen hedge ratios and spread statistics, and the same
trading rule. The only difference is whether the data was seen by the selection
step. So 1.65 Sharpe points is the size of the illusion available to anyone who
screens and backtests on one sample, and that number generalises well beyond pairs
trading.

### 6. Cointegration does not persist

**3.3%** of selected pairs were still cointegrated (p ≤ 0.05) during the six
months they were traded. Nineteen of the 25 rolls had zero survivors out of five
selected pairs.

Cointegration measured on two years of daily data is therefore not a persistent
property of a pair, it is largely a property of that sample. Any strategy that
conditions on a historical cointegration test is conditioning on something that
mostly does not survive the holding period.

The model-free version of the same point: the Spearman correlation between a
pair's formation p-value and its subsequent out-of-sample Sharpe is about zero. If
the screen had predictive content this would be clearly negative.

### 7. Cost sensitivity: no break-even, because no edge

| one-way bps | ann. return | Sharpe | max drawdown |
|---|---|---|---|
| **0.0** | **−0.28%** | **−0.05** | −16.5% |
| 2.5 | −0.42% | −0.08 | −16.9% |
| 5.0 | −0.56% | −0.11 | −17.9% |
| 10.0 | −0.84% | −0.18 | −20.0% |
| 20.0 | −1.40% | −0.32 | −23.8% |

The strategy loses money at zero transaction cost, which is a cleaner negative
result than "profitable until 6bp": no cost assumption anyone prefers can rescue
it, and the objection that the costs are too optimistic only strengthens the
conclusion. The curve is still the right thing to present — had the zero-cost
Sharpe been +0.8, this table would locate the exact cost at which the strategy
stops existing.

Costs matter disproportionately here because pairs trading is high-turnover and
small-edge: every trade is two legs, both crossing the spread, both paying
commission, one paying borrow throughout. And the signal fires precisely when
spreads are widest and liquidity is thinnest, so the days the model most wants to
trade are the days trading is most expensive.

### 8. The multiple-testing audit

How large was the search?

```
pairs tested per formation window : 435
formation windows (rolls)         : 25
total cointegration tests run     : 10,875
threshold combinations explored   : 18
effective strategy trials (N)     : 195,750
```

Do the "significant" pairs survive a multiplicity correction?

| | |
|---|---|
| Tests | 435 |
| Raw hits at p ≤ 0.05 | 23 |
| Expected by chance | 21.8 |
| Excess over chance | +1.2 |
| Survive Bonferroni (p ≤ 1.15 × 10⁻⁴) | **0** |
| Survive Benjamini–Hochberg FDR | **0** |

Not one pair survives either correction. Bonferroni is doubly conservative here,
but BH, the appropriate criterion for a screening problem, also returns zero.

| Deflated Sharpe Ratio | |
|---|---|
| Observed out-of-sample Sharpe (annualised) | −0.11 |
| Spread of Sharpe across trials (σ_SR, annualised) | 1.76 |
| Expected **best** Sharpe from luck alone, N = 195,750 | 7.97 |
| PSR vs zero (ignores the search) | 34.3% |
| **DSR vs luck (accounts for the search)** | **0.0%** |

One caveat I would rather state than have found: the 7.97 bar is implausibly high
as a real-world hurdle. σ_SR is estimated from the spread of individual pairs'
Sharpes over single 126-day windows, which are extremely noisy, so the bar
inherits the noise; a tighter implementation would estimate it from
strategy-level trials. The conclusion does not depend on it — the observed Sharpe
is negative, so it fails against a bar of zero, and the PSR of 34.3%, which
involves no deflation at all, already says there is a 66% chance the true Sharpe
is below zero.

**Parameter sensitivity.** Sharpe ranges from −0.78 to +0.06 across the 18
threshold combinations. The best cell is +0.06, 16 of 18 are negative, and the
headline configuration (entry 2.0 / exit 0.5 / stop 4.0) sits at −0.11 in the
middle. This is what a robustly absent edge looks like, and it is why the surface
is reported rather than a peak: had one cell shown +1.2 while its neighbours were
negative, that cell would be a knife-edge artefact of an arbitrary threshold, and
reporting it as the result is one of the most common ways pairs-trading studies
mislead.

**The model-free cross-check.** Spearman correlation between a pair's formation
p-value and its subsequent out-of-sample Sharpe: **−0.039**. The screen ranks
pairs no better than random, and this needs no distributional assumption at all,
which makes it the result I trust most here.

### 9. The positive controls failed, and that was the most useful result

SPY–IVV gives p = 0.74, GLD–IAU p = 0.15, and both must be cointegrated by
arbitrage. The first instinct is that the test is broken. It isn't — there are two
real mechanisms, and both are quantified.

**Expense-ratio drift.** GLD charges 40bp a year, IAU charges 25bp. That 15bp
differential is a deterministic downward drift in the log ratio, every year,
forever, so the spread is stationary around a *declining line* rather than a
level. The test uses `regression='c'`, which tests level-stationarity, so it
correctly fails. Imposing β = 1 (the right restriction for two funds holding the
same metal), the drift measures **−14.9bp/yr with R² = 0.99** against the 15bp fee
gap. Re-test with a trend term and it passes at p ≈ 0.0000.

**A structural break.** SPY–IVV fails even with a trend term (p = 0.70), because
the spread has a level shift: its yearly mean moves from −36.0bp in 2015 to
+1.2bp in 2016, coinciding with IVV's fee cut. Post-2016 the drift is −5.5bp/yr
with R² = 0.93 against a 6.45bp fee differential. Perron (1989) showed unit-root
tests have very low power against a stationary series containing a structural
break, which is exactly what we observe.

So the controls are cointegrated, the test is correctly implemented, and the
failures are informative. `regression='c'` stayed as the primary specification
because a spread with a drifting mean is untradeable, and `diagnose_pair()`
reports drift, trend R² and the same quantities under β = 1 so the mechanism is
visible rather than mysterious.

And the punchline: these are the most reliably cointegrated pairs in existence and
they are completely untradeable. The SPY–IVV spread has a standard deviation of
**15.5 basis points**, so a 2σ deviation is 31bp against a round trip of roughly
20bp. Statistical significance and tradability are nearly unrelated questions.

The general lesson is worth more than the finding: **a failing positive control is
data, not noise.** The temptation is to loosen the test until the control passes;
the right move is to find out why.

---

## Figures

All in `figures/`, generated by `python src/plots.py`.

| Figure | What it shows |
|---|---|
| `spread_with_trades.png` | The mechanism: z-score path, entry/exit/stop bands and trade markers for one pair. The example is the pair with the **most trades**, a criterion independent of whether those trades made money. It happens to show the re-arm rule working — stop-out at z ≈ −5, then a month between −5 and −6.7 with no re-entry — and the failure mode a table hides, a spread that leaves the band and does not come back. |
| `equity_curve.png` | The result: out-of-sample growth of $1 against SPY, with a drawdown panel. For a market-neutral book the drawdown profile determines fundability. |
| `is_vs_oos_sharpe.png` | The finding: in-sample vs out-of-sample Sharpe, one dot per selected pair. Under a screen with real predictive content the dots would cluster on the 45° line. Instead every dot sits in the right half (in-sample Sharpe is positive by construction — that is what selection means) and they spread almost uniformly from −4 to +3 vertically. |
| `cost_sensitivity.png` | The caveat: Sharpe against assumed cost, with the zero line marked. |

---

## What I got wrong

Kept deliberately, because the errors are more instructive than the code.

**A real bug: the stop loss that amplified losses.** The original logic was "if
|z| ≥ 4, exit" — but the next day the entry condition |z| ≥ 2 was still satisfied,
since z was 4.6, so the strategy re-entered the same losing position at a worse
price. Test case: z = [0, 2.5, 3.0, 4.5, 4.6] stops out at index 3 and re-enters
short at index 4. That turns a stop loss into a loss *amplifier*: instead of one
capped loss you get an unbounded chain of them as the spread keeps diverging,
which is precisely what the stop exists to prevent. The fix is a `disarmed` flag —
after a stop-out, no new entry in that pair until |z| comes back inside the entry
band.

Why this is the most important thing in the repo: a backtest would never have
flagged it. No exception, no NaN, no impossible number, just a worse equity curve
that would have been blamed on the strategy rather than the code. The only way to
find it was a test asserting what the rule was *supposed* to do.

**Not a bug: a finite half-life for a random walk.** `half_life()` returns finite
values (283, 359, 580, 617, 13,149 days across seeds) for genuine random walks,
where the true answer is infinite. This is the Dickey–Fuller downward bias: in
finite samples OLS estimates of an autoregressive coefficient near 1 are biased
downward, so φ̂ lands just below 1. The formula is right and the test assertion was
too strong. What actually rejects those artefacts is the 5–60 day band, which is a
second, independent reason for it.

**A real design flaw: the I(1) leg gate discarded real pairs.** A synthetic pair
with an Engle–Granger p-value of 3.6 × 10⁻¹⁴ was rejected because one leg's ADF
p-value landed at 0.04. The leg test is used backwards from its usual purpose, so
a false rejection throws away a good pair. Tightening to α = 0.01 helped but did
not fully fix it, which prompted measuring the gate: on 60 simulated random walks,
2/60 wrongly rejected at 5% and 0/60 at 1%. So the 1% gate is correctly sized, and
the residual failures came from my *synthetic generator* adding a large stationary
AR(1) component to leg A. The lesson: when a filter behaves unexpectedly, measure
its size on data whose truth you control before changing its threshold.

**Corporate TLS interception.** Every download failed and yfinance reported
"possibly delisted", which was its interpretation of the failure rather than the
failure. The real error was
`curl: (60) SSL certificate problem: self signed certificate in certificate chain`:
a TLS-inspecting proxy re-signs HTTPS with a private root CA in the macOS System
keychain, while yfinance's `curl_cffi` transport validates against its own bundled
list. `ensure_ssl_trust()` exports both keychains into a PEM bundle and points the
SSL environment variables at it, fail-soft, because a certificate workaround must
never be able to break the study.

**A 13-minute backtest that had to run 17 times.** The first version re-ran the
whole screen inside every backtest call, so `cost_sensitivity` (5 runs) and
`parameter_sensitivity` (18 runs) would have taken over an hour, and it did time
out. The fix came from noticing what depends on what: screening, hedge ratios,
spread statistics and the z-score path depend only on prices and the window
schedule, while costs and thresholds affect only the P&L computed from that path.
So the screen is built once, cached and reused. The research-hygiene benefit beats
the speed: one shared plan guarantees the rows of a sensitivity table differ only
in the parameter being varied.

That cache produced its own bug. Pickling the dataclasses directly recorded them
as `__main__.RollPlan`, since `backtest.py` had been run as a script, so loading
the cache from `multiple_testing.py` failed. It now stores plain dicts and pandas
objects. A cache file that depends on how the writer was invoked is not a cache,
it is a trap.

---

## Repository layout

```
pairs-trading-study/
├── README.md
├── requirements.txt
├── notebooks/
│   └── pairs_trading_study.ipynb   ← the narrative write-up (reads data/, computes nothing)
├── src/
│   ├── data_pipeline.py       Block 0  universe, download, cleaning, log px vs simple returns
│   ├── cointegration.py       Block 1  ADF, Engle-Granger, Johansen, half-life, the screen
│   ├── signals.py             Block 2  frozen formation stats, z-scores, trade state machine
│   ├── backtest.py            Block 3  cost model, P&L engine, walk-forward, cost sensitivity
│   ├── multiple_testing.py    Block 4  Bonferroni, BH-FDR, PSR, DSR, parameter sensitivity
│   └── plots.py               the four figures
├── tests/
│   └── test_pairs.py          27 tests — mostly encoding traps, not checking types
├── data/                      generated: prices, screen, OOS returns, trades, run logs
└── figures/                   generated: the four PNGs
```

Every module is runnable standalone, reads its inputs from `data/` and writes its
outputs there. Scripts rather than notebooks so that results are diffable:
`python src/backtest.py > log`, commit the log, and see exactly what changed when
the code changes. `data/` and `figures/` are committed so the results and the
notebook can be read without a 15-minute re-run; the screen cache
`data/walk_forward_plan.pkl` is not, since it is large, machine-local and rebuilt
automatically. The notebook is narrative only and holds no modelling logic of its
own, so it cannot drift out of agreement with the scripts.

### A note on the tests

The 27 tests are the part of this repo worth reading first, because they are not
type checks. Four kinds:

1. **Mathematical identities** — portfolio return equals the weighted sum of
   *simple* returns; max drawdown against a hand-computed example; the Sharpe of a
   constant series is undefined, not infinite.
2. **Statistical properties on synthetic data with known truth** — a genuinely
   cointegrated pair is detected at p < 0.01 across 5 seeds; 20 independent random
   walks produce ≤ 4 false hits; the leg gate's empirical size is measured on 60
   random walks.
3. **Adversarial tamper tests** — vandalise the last 50 days of prices and assert
   every earlier net return is identical; tamper with a series' tail and assert
   earlier z-scores don't move. These are the only tests that can catch lookahead
   bias.
4. **Rule semantics** — sign convention; stop fires before target; stop-out
   disarms; no cost charged while flat; costs reduce returns monotonically.

The principle: test the trap, not the plumbing. A test asserting
`len(result) == len(input)` catches nothing that matters.

---

## How to reproduce

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python src/data_pipeline.py      # download + clean          → data/prices.csv
python src/cointegration.py      # Block 1 screen (~2 min)   → data/pair_screen_full_sample.csv
python src/backtest.py           # Block 3 (~14 min first run, then cached)
python src/multiple_testing.py   # Block 4 + parameter grid
python src/plots.py              # figures/

PYTHONPATH=src pytest tests/ -q  # 27 tests, ~10 s

# optional: the narrative write-up (needs `pip install jupyter`)
jupyter lab notebooks/pairs_trading_study.ipynb
```

Run from the repository root — the modules use relative paths into `data/`. Add
`PYTHONPATH=src` if your environment does not put `src/` on the path.

Two notes:

- **Delete `data/walk_forward_plan.pkl`** if you change the universe, the date
  range or `WalkForwardConfig`. The cache is deliberately keyed on nothing,
  because a half-clever cache key that misses one input is worse than an obviously
  manual one.
- Behind a corporate TLS-inspecting proxy, `ensure_ssl_trust()` handles the
  certificate problem automatically on macOS. On other platforms, set
  `SSL_CERT_FILE` to your own CA bundle.

Yahoo Finance data is not audit-grade: it is free, occasionally revised, and has
no notion of point-in-time index membership. Numbers may shift slightly on
re-download. Every conclusion here is a first-order effect, not a
third-decimal-place one.

---

## Limitations

In order of how much they should change your reading of the results.

1. **Survivorship bias, unfixable here, and it biases results upward.** The
   universe is the liquid US large caps that exist *today*, so every company that
   went bankrupt, was acquired or fell out of the index is absent — exactly the
   pairs whose relationship broke catastrophically. Fixing it needs a
   point-in-time constituent database. Since the headline is already negative, the
   true result is somewhat worse than reported.
2. **No market-impact model.** Impact depends on order size relative to
   liquidity, participation rate and intraday timing, none of which is in daily
   closes. Rather than invent an uncalibrated coefficient, the cost-sensitivity
   curve lets a reader substitute their own assumption.
3. **Static hedge ratio.** β is re-estimated each roll but fixed within the
   trading window. The modern treatment is a Kalman filter, with the caveat that a
   more adaptive β makes spreads look more stationary, so its free parameters have
   to be counted in the trial budget.
4. **Universe selection is itself a choice.** These 30 tickers were chosen by me,
   so the conclusions are about this universe, not about "US equities".
5. **A single frequency and a single market.** Daily closes, US large caps,
   2012–2026 — and daily frequency is the leading explanation for the negative
   result.
6. **Zero risk-free rate assumed** in every Sharpe, over a sample spanning both a
   near-zero and a 5% rate era.
7. **Shorting is assumed frictionless beyond a flat 50bp** — no locate risk, no
   rate spikes, no recall, and recall risk is perfectly correlated with the
   position going wrong.
8. **My own iteration is uncounted.** This code was written and rewritten many
   times while looking at output. That is researcher degrees of freedom no formula
   corrects for, and it is why the trading parameters were pinned at pre-committed
   textbook values and behaviour was locked with tests rather than eyeballed on
   plots.

---

## Extensions

In the order I would actually do them.

1. **Point-in-time universe** including delisted names. The only remaining issue
   that would make a *positive* result untrustworthy rather than merely imprecise.
2. **Relationship health as a live state, not a selection gate.** Motivated
   directly by the 3.3% survival rate: trade a pair only while its tether is
   currently visible, and size by how strong it is, instead of admitting it for
   six months on the strength of one test.
3. **Intraday data**, to test whether the edge lives at a frequency where
   liquidity provision is still paid.
4. **Factor-residual formulation** on a wider universe: build a statistical risk
   model, extract each stock's residual, and trade residuals rather than pairs.
   This is what stat-arb desks do, and it scales where pair enumeration does not
   (1,000 names is ~500,000 pairs).
5. **Block bootstrap** on the out-of-sample return series, for a confidence
   interval on the Sharpe rather than a point estimate.

---

## References

- Engle, R. and Granger, C. (1987). "Co-integration and Error Correction:
  Representation, Estimation, and Testing." *Econometrica* 55(2).
- Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration Vectors
  in Gaussian Vector Autoregressive Models." *Econometrica* 59(6).
- Gatev, E., Goetzmann, W. and Rouwenhorst, K. (2006). "Pairs Trading:
  Performance of a Relative-Value Arbitrage Rule." *Review of Financial Studies*
  19(3). — the canonical study; source of the 12m/6m convention.
- Perron, P. (1989). "The Great Crash, the Oil Price Shock, and the Unit Root
  Hypothesis." *Econometrica* 57(6). — why unit-root tests are blind to
  structural breaks.
- Bailey, D. and López de Prado, M. (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality."
  *Journal of Portfolio Management* 40(5).
- Benjamini, Y. and Hochberg, Y. (1995). "Controlling the False Discovery Rate."
  *JRSS B* 57(1).
- Avellaneda, M. and Lee, J. (2010). "Statistical Arbitrage in the US Equities
  Market." *Quantitative Finance* 10(7). — the factor-residual formulation.

---

## Companion project

[`vol-forecasting-study`](../vol-forecasting-study) — *Does Implied Volatility
Predict Realized Volatility?* In-sample regression, out-of-sample R² against a
persistence benchmark, and the variance risk premium with HAC standard errors.

The two are complementary. That one asks whether a **signal exists**; this one
asks whether a signal **survives contact with a book** — search, selection bias,
position sizing, transaction costs, drawdowns and multiplicity.

---

*Research code on free daily data. A study of a method, not investment advice and
not a trading system.*
