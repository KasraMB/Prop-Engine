# Drift-aware analytical model and engine comparison

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
