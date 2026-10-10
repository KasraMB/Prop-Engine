# Uncertainty reporting

Outcome dispersion is not uncertainty in an estimated mean. Neither measures
whether the simulator's assumptions match future trading.

## What is reported

- One reporting contract across cash-risk, historical replay, recorded fills
  and target-model results.
- Cheap default reports: observed sample dispersion, conditional standard errors
  when independence is declared, and explicit reasons for withheld inference.
- Opt-in percentile bootstrap intervals for means; circular block resampling for
  an explicitly declared ordered historical scalar series.
- Paired differences retain scenario alignment. Scenario sensitivity is reported
  as an envelope, never as a confidence interval.
- Selected training results are descriptive. Holdout reports condition on a
  frozen policy and do not correct repeated holdout reuse or model selection.
- Tests use independent numerical calculations and check suppression as well as
  successful inference. Resampling stays outside optimizer candidate scoring.

The report does not automatically identify a true distribution family, quantify
model misspecification, or estimate parameter uncertainty without refitting.

## API

```python
from propfirm_engine import Engine, UncertaintyConfig

report = Engine().uncertainty(
    {"net_cash": [-200, 100, 50, 900, -100, 300],
     "cash_per_day": [-20, 10, 5, 90, -10, 30]},
    sample_kind="independent_model",
    options=UncertaintyConfig(resamples=2000, seed=19),
)
cash = report["metrics"]["net_cash"]
print(cash["sample_standard_deviation"])
print(cash["mean_standard_error"], cash["mean_interval"])
print(report["unmeasured"])
```

`uncertainty_report` is also a top-level function. Each metric contains observed
count, missing count, mean, sample standard deviation, mean standard error,
mean interval, method and status. Sample SD uses `ddof=1`, unlike the descriptive
population SD in existing `distributions`. The interval is for the mean of the
supplied observations, not the range of future outcomes.

All metric arrays must have equal lengths. `None` and NaN are missing; infinity
and non-scalar values are rejected. Incomplete metrics remain descriptive. An
all-equal sample has observed SD zero but receives no mean SE or interval: it
cannot establish that unobserved losses are impossible.

No resampling happens by default (`resamples=0`). Independent model and execution
samples receive `sample SD / sqrt(n)` as their conditional mean SE. Other designs
do not. Request resampling explicitly to obtain approximate percentile intervals.
Bootstrap endpoints themselves have simulation error; increase the resample
count and check stability, especially at high confidence levels.

## Sampling contracts

| `sample_kind` | Meaning | Mean inference |
| --- | --- | --- |
| `independent_model` | Independent paths from a fixed declared model | IID mean SE, optional IID bootstrap |
| `execution_model` | Externally supplied independent execution scenarios on the same history | Same calculation, execution uncertainty only |
| `historical_series` | Ordered, equally spaced scalar observations | Optional circular block bootstrap with explicit block length |
| `historical_windows` | Dependent/overlapping historical starts | Descriptive only |
| `single_history` | One observed replay | Descriptive only |
| `training_model` | Outcomes used to select a policy | Descriptive only |

These are assertions by the caller, not properties the engine can verify from a
list of numbers. A bootstrap of independent paths generated from an estimated
law quantifies simulation error conditional on that fitted law. It does not
quantify error in the law itself. More paths cannot eliminate calibration bias.

Finite variance is needed for the usual mean-SE interpretation. The engine does
not fit a normal distribution to payouts or infer a distribution family from a
small sample. Payout mixtures, tail losses and censoring make that especially
unsafe. Even a bootstrap cannot discover tail events absent from the sample.

## Historical estimation uncertainty

```python
from propfirm_engine import UncertaintyConfig, uncertainty_report

# One observed cash increment per complete session, including zero-cash sessions.
daily_cash = [-105.2, 0, 0, 0, 900, 0, -105.2, 0, 0, 0, 0, 0] * 10
report = uncertainty_report(
    {"cash_per_session": daily_cash},
    sample_kind="historical_series",
    options=UncertaintyConfig(resamples=2000, block_length=12, seed=19),
)
print(report["metrics"]["cash_per_session"])
```

The example is an interface demonstration, not evidence of profitability.
Circular blocks wrap at the end of the supplied scalar series. Blocks are
sampled with replacement and the last block is truncated to preserve sample
length. The reported SE is the sample SD of the bootstrap means.

Stationarity and sufficiently weak dependence must be defensible. Choose block
length using the dependence scale and test several plausible lengths; do not
choose it to get a favorable interval. At least two full blocks must fit, but
two blocks are not a recommendation for an adequate study. Abrupt regime changes,
long dependence, short samples and rare payouts can invalidate interpretation.

This calculation does not reorder trades, rerun accounts, or refit a policy.
It estimates uncertainty in the mean of the supplied scalar observations only.
For policy or bankroll uncertainty under alternative market histories, generate
full histories, rerun the exact engine, and keep the outer calibration/refit
experiment distinct from the inner execution simulations. Rolling reports do
not silently opt into this method. Multiple execution seeds at each historical
start must not be flattened into an equally spaced historical series.

The denominator matters: daily observations give dollars per observed day;
session observations give dollars per session. Do not call the latter calendar
EV/day. Zero-cash sessions must remain in the series. A nonlinear ratio, ruin
event, percentile, or capital requirement needs its own estimator, not this
mean interval.

## Paired policy comparison

```python
from propfirm_engine import UncertaintyConfig, paired_uncertainty

report = paired_uncertainty(
    {"net_cash": [-90, 12, 114, 216]},
    {"net_cash": [-100, 0, 100, 200]},
    sample_kind="independent_model",
    options=UncertaintyConfig(resamples=2000, seed=19),
)
print(report["metrics"]["net_cash"])
```

The difference is selected minus baseline. Rows must refer to the same scenarios
in the same order and at the same horizon. Pairing is an input contract; equal
array lengths alone cannot prove correct alignment. Missingness in either side
suppresses inference. Target fit reports orient `objective_gain` so positive
means improvement for either maximization or minimization.

Keep a genuinely untouched holdout. The selected policy's training report has
`selected_on_sample=True`, which suppresses inference even when its underlying
random tapes were independent. A frozen policy on independent OOS model tapes
can receive conditional inference. Neither that report nor repeated search seeds
measure historical selection bias. Reusing OOS to choose strategies turns it
into training data; use a fresh outer holdout or nested chronological evaluation.

## Model sensitivity

```python
from propfirm_engine import scenario_sensitivity

report = scenario_sensitivity({
    "base": {"cash_per_day": 15},
    "higher_costs": {"cash_per_day": 5},
    "lower_drift": {"cash_per_day": -8},
}, baseline="base")
print(report["deltas"], report["envelope"])
```

Supply actual rerun results, holding the evaluated policy fixed if the question
is robustness. The helper does not execute scenarios or assign probabilities.
The envelope covers only supplied scenarios; it is not a worst-case guarantee
or confidence interval. Refitting under each scenario answers a different
question and must be labelled separately.

## Result access

- `BacktestResult.uncertainty`: single chronological replay, no inference.
- `EventReplay.uncertainty`: delegates to its single chronological account replay.
- `RollingReplayResult.uncertainty`: window/objective dispersion, no IID claims.
- `HoldoutFit.uncertainty`: selected OOS result.
- `ResearchSummary.uncertainty`: objective, net cash and cash/day model mean SE.
- `TargetFit.uncertainty`: model holdout and paired objective gain.
- `risk_report(...)["uncertainty"]`: mean report for the existing cash distributions.
- Legacy `Results.uncertainty(sample_kind=...)`: requires an explicit classification
  of the supplied attempt outcomes; exposes payoff, payout and trading days.

These result properties are computed on access. Risk reports reuse already
computed moments. No bootstrap is added to candidate scoring. To request
intervals, pass the raw aligned observations to `uncertainty_report`; for cash
risk output they are available in `path_records`. Cached descriptive results and
legacy `score_standard_error` fields remain available; use the structured report
for the selection/dependence status before interpreting them as uncertainty.

The dashboard's JSON exports include structured cash-risk uncertainty; target
search also exports training and paired holdout reports. This change does not add
a new dashboard panel or bootstrap controls. Dataclass `asdict` does not include
properties; Python callers should export `.uncertainty` explicitly when needed.

Existing ruin reports retain their specialized Wilson probability intervals and
one-sided order-statistic bankroll bounds under their own sampling contracts.
The new mean intervals do not replace them or estimate infinite-horizon ruin.

## Method references

- [NIST bootstrap overview](https://www.itl.nist.gov/div898/handbook/eda/section3/eda334.htm):
  empirical resampling and percentile intervals; caution with tail statistics.
- [Circular block bootstrap](https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.CircularBlockBootstrap.html):
  fixed-size blocks with end-to-start wrap. No additional runtime dependency is used.
- [Cawley and Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html):
  model-selection overfitting and selection bias in performance evaluation.
