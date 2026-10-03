# Drift-aware analytical model and engine comparison

## Joint risk and target research

Added 2026-10-03. `target_research.py` reuses the chronological `_Replay`
lifecycle and its existing interpreter, feasibility projection and payout ledger.
There is no second production implementation of the account rules. This mode
does not remove consistency or qualifying-day requirements.

`TargetPolicy` pairs each ordered `RiskRegime` with a **net** dollar profit target.
The loss budget includes modeled trading costs. One theoretical contract is used;
bracket distances are freely adjustable. For requested loss budget S, remaining
buffer B, target T and total round-trip cost C:

```text
actual loss budget = min(S, B)
gross stop        = actual loss budget - C
gross target      = T + C
zero-drift p(win) = gross stop / (gross stop + gross target)
```

For nonzero gross dollar drift mu, the implementation uses the stable exponential
first-exit formula with nu = 2*mu/sigma^2. Costs are translated into the barriers
and charged once by the existing engine. At zero gross drift the expected net
trade P&L is -C, not zero. If a one-cent gross bracket cannot fit, canonical
projection distinguishes an untradeable buffer (`CAPPED_OUT`) from a deliberately
too-small policy budget (`policy_skip`). Historical `Engine.backtest` rejects a
`TargetPolicy`; it cannot silently reuse old wins with new profit targets.

This is an **eventual-hit Bernoulli model with an imposed session clock**. Each
available session receives one completed bracket, with bookkeeping timestamps
at 10:00 and 10:05 in the profile timezone. Those timestamps are not predicted
holding times. Brownian first-passage duration, session-end liquidation, ticks,
gaps, changing market regimes and real execution are not modeled. Drift and
volatility use a common arbitrary diffusion-time unit; they determine hitting
odds, not how many real trading days a bracket takes. Cash/calendar day is thus
conditional on the one-bracket-per-session assumption and must not be presented
as achievable market EV/day.

The account side retains the selected profile's rules, settlement delays,
wallet constraint, fees, surviving losses and repeated evaluation attempts,
including fresh evaluations after live handoff. Unsupported firm mechanics
remain subject to the normal chronological API's validation.

### Search and validation

`fit_targets` uses the existing bounded CMA-ES search over risk and target
parameters, rounding proposals to cents. Optional per-regime `risk_choices` and
`target_choices` snap proposals to declared levels. Exact thresholds such as
$150 winning days and $1,500 evaluation days can then be searched without relying
on a continuous optimizer landing on a single dollar value. These choices are
inputs fixed before selection, not values inferred from holdout outcomes.

The default criterion is the equal-weight mean of each path's received external
net cash/calendar day. Custom finite-scalar callables and minimization are
supported. The initial policy participates in selection, and IS-unvisited regimes
retain their initial parameters. Explicit candidate seeds are counted in output;
seeding an example must never be described as discovering it.

Seventy percent of independently generated uniform tapes are used for selection.
The remaining thirty percent use a separate RNG stream and are evaluated only
after the policy is frozen. Identical training tapes are reused across candidates;
this is common-random-number variance reduction, **not shared price paths**.
The result exposes held-out score standard error and paired improvement standard
error. These quantify model sampling error only, not uncertainty about the market
model, firm terms, holding times or strategy edge. Repeatedly tuning on these
holdout results invalidates that holdout.

`benchmarks/target_policy.py` searches from a constant $500-risk/$500-target
policy using declared dollar grids, then compares the frozen result with the
named LucidFlex example on the same held-out tapes. It exports inputs, policies,
summary distributions and a representative decision/event trace as JSON. The
named example is **not** included in its training candidates. The search is
heuristic; it does not prove a global optimum or guarantee that exact example.

Risk reports now accompany held-out and baseline results. Configure the benchmark
with `--bankroll 2000 --ruin-target .01 --confidence .95`; the bankroll here is a
funding-shortfall analysis threshold, not a change to its unlimited-wallet
performance scenario. Reports include variance, percentiles, tail losses,
cash drawdowns, bankroll curves and, where the independent sample supports it,
confidence-qualified capital. See the [risk-report contract](BRACKET_BACKTEST.md#cash-risk-and-bankroll-reporting).

The initial reproducible run (100 paths, 30 sessions, 20 generations, seed 42,
zero gross drift/cost, immediate settlement) evaluated 143 distinct policies.
Starting from $500/$500, it selected evaluation risk/target $2,000/$1,500,
funded-build $1,000/$1,500, and funded-day $1,500/$500. On the 30 held-out paths,
mean cash/calendar day was $9.93 (standard error $7.99), versus -$3.82 for the
flat baseline and $17.80 (standard error $7.37) for the named example. The paired
gain over the flat baseline was $13.75/day with standard error $6.98. This small
model experiment demonstrates representability and search, not reliable policy
superiority; the example outperformed the selected policy in this sample.

The independent test oracle in `tests/test_target_research.py` uses rational
arithmetic and no production accounting helpers. It checks all 128 seven-session
win/loss branches of the zero-cost, immediate-settlement example, including
retries and surviving funded losses. Branch probability masses sum to one. In
that restricted benchmark the first-attempt pass probability is (4/7)^2, and
the first receipt by session seven has probability (4/7)^2*(1/2)*(40/43)^4.
The fastest path produces $1,170 received cash before evaluation fees. These are
not full-lifecycle payout probabilities or a strategy-profitability claim.

### What the supplied paper contributes

The supplied 24-page Fernandez (2026) paper, `7260819.pdf`, provides the
drift-aware first-passage framework (section 2.3), a distinction between floor
updates and breach checks (2.4), and repeated-attempt accounting (3.2 and 3.4).
The underlying static-barrier derivation is also covered in
[MIT's Martingales II lecture](https://ocw.mit.edu/courses/15-070j-advanced-stochastic-processes-fall-2013/ce6414f24b7c22b5030d757713d3fc8f_MIT15_070JF13_Lec11.pdf).

The new research workflow uses the fixed-bracket exit probability only. It does
**not** apply the paper's approximate continuity correction, zero-drift duration
formula, two-state account valuation, or single payout-threshold reduction to
the full LucidFlex lifecycle. Consistency requires daily-maximum state; winning
days and payout cycles require more state still. Replacing them with a single
fixed barrier would discard the very behavior being optimized.

The paper's cautions about calibration and effective sample size motivate
separate model/historical labels, fixed-before-selection search choices, and
independent checks. They do not imply that equal mean and variance establish
equal path-dependent outcomes. Nor do they require repricing an actually quoted
evaluation fee merely because a researcher changes the assumed market drift.
Importance-sampling weights are needed when changing the sampling distribution
while estimating expectations under the original distribution; this workflow
instead reports each explicitly chosen model's own conditional expectations.

Firm parameters remain sourced to the official pages in the profile documentation,
not the paper's cross-firm comparison table. Ordered real price data is still
required to validate alternative targets against an actual strategy; no such
validation is supplied by this research mode.

## Earlier diffusion-only API

Implemented 2026-09-26. This is a generic analytical API, not a strategy API.
It implements Fernandez (2026), supplied 7260819.pdf, equations (3), (5) and the
per-evaluation-fee branch of (9). No driftless duration/state-value equations
were imported. The zero-drift limit is supported as a numerical continuity case.

## Contract

- barrier_pass_probability: infinite-horizon probability of a constant-drift,
  constant-volatility Brownian P&L hitting a target before a capped trailing floor.
- target, drawdown and freeze_level are dollar amounts relative to phase opening.
  freeze_level=-drawdown gives a static floor; freeze_level >= target-drawdown
  means no freeze before passing. Positive infinity is supported for never-freeze.
- mu is net USD per trading session; sigma is USD per square-root trading session.
  Both are already expressed at the chosen size. Do not multiply by size twice.
- floor_updates and breach_checks are independent observations per session.
  Infinity means continuous. Finite values select the paper's approximate
  two-clock correction, with its warnings preserved in the result.
- A monitoring-step/buffer ratio above 0.5 is explicitly flagged, following the
  paper's stated range. Being inside that range is NOT an independently verified
  accuracy guarantee, especially with drift, jumps or generated excursions.
- estimate_session_moments fits population moments (ddof=0) of the empirical
  net-dollar session totals used by IID whole-day resampling. Net trade P&L is
  size_base * return - trade_cost. Every source session enters; no survivor filter.
  The sigma estimate uses actual session totals, not an independence shortcut
  based on square-root trade count. This is not a serial-dependence correction.
- expected_funding_cost calculates entry_fee + (1/P-1)*reset_fee + activation_fee
  for IID unlimited affordable retries. It is not finite-wallet economics,
  elapsed-calendar value, subscription billing, or profit from a funded account.

The API accepts explicit model parameters, not a whole Account whose unsupported
rules could be silently discarded. It does not apply consistency, winning days,
daily loss limits, dynamically changing size, payout locks or settlement clocks.
Zero volatility is rejected rather than pretending a deterministic process is a
diffusion. Invalid/overflowing numeric inputs are rejected; ordinary strong-drift
probabilities can validly round to zero or one.

## Executable example

Synthetic cost and risk below are illustrative, not instrument specifications.

```python
from propfirm_engine import (
    IIDGenerator, preprocess, estimate_session_moments,
    barrier_pass_probability, expected_funding_cost,
)

stream = IIDGenerator(win_rate=0.52, rr=1.0, trades_per_day=20,
                      intraday_excursion=0.0).generate(n_days=1000, seed=260926)
data = preprocess(stream.rows)
moments = estimate_session_moments(data, size_base=100, trade_cost=1)
estimate = barrier_pass_probability(
    target=3000, drawdown=2000, freeze_level=100,
    mu=moments.mu, sigma=moments.sigma,
    floor_updates=1,  # EOD updates; continuous breach is the default
)
assert 0 <= estimate.probability <= 1
assert estimate.method == "paper_eq3_eq5"
cost = expected_funding_cost(estimate.probability, entry_fee=105.20, reset_fee=105)
assert cost >= 105.20
```

The fee values are the user's dated screenshot scenario, not a fresh checkout quote.

## Reproducing the comparison

```text
python benchmarks/analytical_comparison.py --output results/analytical-comparison.json --trades-dir results/analytical-trades
```

Default workload: three generated histories of 5,000 sessions x 20 trades,
20,000 resampled attempts per case, 600-session engine horizon, two fixed dollar
risk scales and one identical-returns/different-MAE sensitivity case. Each
baseline history is saved as a compressed NumPy archive with returns, lows, day
IDs and timestamps. Generated archives stay local in an already-ignored output
folder; generator seeds, parameters, canonical-array hashes and result summaries
are included in the generated JSON. Re-running replaces its chosen output files.

The experiment runs the existing Engine, not a new proxy simulator. It uses
explicit summary_approximation because the generator has no ordered MTM path.
The default strict guard is unchanged and covered by a regression test.

Separate comparisons:

1. Paper equation (3): continuous floor and continuous breach.
2. Paper (3)+(5): EOD floor and continuous breach interpretation.
3. Paper (3)+(5) sensitivity: breach clock set to trades/session, NOT a firm rule.
4. Engine barrier-only evaluation: consistency removed explicitly for comparison.
5. Engine evaluation with the selected conservative 50% consistency rule restored.

The main comparison uses (2) versus (4); (5) shows the additional eligibility
constraint. No fitted or optimized policy is selected. Sizes/cases are declared
in advance, and both engine variants reuse the same seed and source days.

Wilson intervals describe conditional Monte Carlo error given one empirical
history and policy; they are not bounds on diffusion error or uncertainty in the
estimated moments. Consistency effects use paired outcomes, not independent-row
error estimates. Unfinished attempts are counted as censored, with empirical
eventual-success bounds [passed/N, (passed+censored)/N], not called failures.
Those bounds describe censoring alone, not a confidence interval. The analytical
formula is infinite horizon, so substantial censoring would prevent a direct
point comparison. No attempt-year or cash-receipt time is invented.

Inspect the generated JSON for per-case results, seeds, hashes, scope and
conditional uncertainty. Generated results are not distributed as performance claims.
