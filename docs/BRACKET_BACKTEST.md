# Chronological bracket replay

Updated 2026-09-26. This is the current executable contract, not a certificate
that every legal, discretionary or platform-specific firm term is simulated.

## Scope and architecture

`Engine.backtest` adapts the existing `_ReferenceSim` rule interpreter for
incremental execution. It uses the existing `project_position` sizing arithmetic,
compiled Account/Phase rules, and `PayoutLedger`. `Engine.fit` uses the existing
CMA-ES search. There is no independent replacement rule engine.

The fast, resampled `Engine.run` API is unchanged except for reserving its
per-trade cost during feasibility projection. Its closed-summary strict guard
remains active. The dashboard homepage exposes chronological replay and holdout
fitting through a shared Python adapter in local HTTP and browser-worker modes.
The account trace uses the same adapter and displays canonical backtest events.
The Monte Carlo dashboard is removed. No acknowledgment checkbox is required;
the declared execution model and input validation remain unchanged.

## Trading inputs

- Sequential, non-overlapping, timezone-aware entries/exits and explicit session dates.
- Stop and target are positive **gross dollar amounts per one contract**.
- Every supplied trade exits at exactly its stop or target; ideal fills, no gaps,
  no stop slippage, no overlapping positions and no overnight session carry.
- Each trade may have a different stop/target ratio. Historical ratios/outcomes
  are immutable during fitting. Normalized returns or net P&L alone do not meet
  this input contract.
- Fixed trade opportunities per day are supported, but not required. Passing,
  failure, wallet limits, payout waiting and voluntary skips may prevent execution.
- One selected contract type per replay. Portfolio/mixed mini-micro exposure
  and automatic conversion between instruments are not implemented.
- Aware timestamps are normalized to UTC; session windows use the configured
  named timezone. Weekends are checked. The producer must remove exchange
  holidays and honor early closes; the library does not ship an exchange calendar.

This is a declared execution model, not proof that a real trade respected its
stop. A generic closed-trade/MAE dataset is not silently upgraded to this contract.

## Sizing

For each pre-trade state:

1. Select the first matching named regime.
2. Convert its total net dollar-loss budget to contracts using that trade's stop,
   per-contract round-trip cost and fixed per-trade cost.
3. Project onto the remaining floor buffer using the existing feasibility function.
4. Enforce integer quantities and the current contract/scaling limit.

The cap includes all supplied trading costs. Exact buffer exhaustion on a loss
is a real floor-touch breach. If even one contract cannot fit, `CAPPED_OUT`
terminates the attempt and counts in `failed_attempts`, without mislabeling it
as an actual firm drawdown violation. Retry pricing follows the current phase.

A zero or sub-contract policy budget is a voluntary skip if the account can
otherwise trade. It is not rounded up above the requested dollar budget. This
differs intentionally from the legacy projection's minimum-size policy floor.

The chronological adapter executes the projection's original Python arithmetic
with rational dollars. Its rule state and payout ledger also retain exact
decimal-rational amounts; event presentation uses floats. No undocumented cent
rounding is imposed on fractional payout amounts.

## Named regimes

`RiskRegime` supports phase, retained-profit sign, whether a prior payout was
approved, and exact remaining qualifying-day count. Conditions see only the
pre-trade state. The current session becomes a qualifying day only when finalized.
Both phases require unconditional fallback regimes. Specific regimes come first.

IS-unvisited regimes retain their supplied baseline instead of an unsupported
search-selected value. Choose regimes and bounds before opening OOS results.

## Lifecycle and cash

- One active account; retries occur no earlier than the next observed session.
- Failed evaluations use the supplied reset fee; failed funded accounts start
  a new purchase at the supplied evaluation fee. Actual reset eligibility after
  the configured reset-expiry period ends requires a new purchase.
- The Lucid profile expires inactive accounts after 30 elapsed days. A trade
  with absolute net P&L of at least $1 resets that clock. This per-trade activity
  interpretation and exact time boundary are explicit scenarios. Inactivity
  during an open trade or a pending payout is rejected as unsupported, rather
  than guessing liquidation or approval outcomes.
- Evaluation passing occurs at a realized close with every pass gate satisfied.
  Funding waits for the caller's elapsed activation delay.
- Request the maximum eligible payout at session close. The supplied policy
  pauses trading until approval. Requests force the configured floor lock.
- The scenario approves every request after `approval_delay`, deducts gross at
  approval, resets the cycle counters, and receives the trader share after
  `receipt_delay`, less the explicit `payment_fee`. These are elapsed-time scenarios, **not** business-day or
  guaranteed firm processing calendars. Manual denials are not predicted.
- Profit-driven scaling changes at session close. Payout-driven reductions apply
  immediately at approval as an explicit conservative interpretation.
- An optional finite external wallet pays fees only from existing cash/received
  payouts. Unreceived payouts cannot finance a retry.
- Five approved payouts in the Lucid profile emit a `live_handoff` event and end
  that simulated-funded account. User-selected lifecycle change (2026-09-26):
  queue a fresh evaluation, subject to the same next-session restriction,
  `retry_delay` and wallet checks as a breach. Charge a new evaluation fee, not a
  reset fee, only when the next attempt can start. Handoffs are not failed attempts.
  `RESTART_PENDING` means handoff occurred but no new attempt has started yet;
  an unaffordable retry reports `INSUFFICIENT_WALLET`.
  The live account's future value stays unvalued; previously approved receipts
  remain attached to the old attempt and are processed within the horizon, even
  while a new attempt is active. Unreceived money cannot finance the new fee.
  Repeated purchases after handoff are a research assumption, not verified firm
  permission to maintain live and simulated accounts concurrently.
- Observation runs from the first declared session's opening to the last
  declared session's close. Gaps/weekends count as elapsed time. Cash after that
  horizon is not included; `outstanding_payouts` is separately reported.

## Rolling historical starts

Added 2026-10-03. `RollingConfig(window_sessions, stride_sessions=1)` selects
fixed-length windows of complete **observed sessions**, not elapsed days. Actual
timestamps, gaps, brackets and trade ordering remain unchanged. Each start gets
a fresh account and the configured initial wallet. Its lifecycle still retries
after failures and live handoffs. This is start-date sensitivity, not resampling.

`Engine.rolling_backtest(..., rolling=rolling, objective=None)` returns compact
per-window records and an equal-weight mean objective. Defaults use each window's
net external cash divided by its own elapsed calendar days. This is a mean of
rates across starts, not a ratio of summed cash to summed days. A custom objective
receives each canonical `BacktestResult`, must return a finite real scalar and
is averaged the same way. No independent-sample standard errors are supplied.

Candidate starts are session indices 0, stride, 2*stride, etc. Incomplete tail
windows are excluded and counted; no complete windows is an error. Complete
windows can still end before an evaluation resolves or before a payout arrives:

- First-evaluation outcomes are passed, failed, unresolved, not started (wallet),
  or not applicable (direct funded). Later retries cannot overwrite the first
  evaluation outcome. Pass rate divides passes by started first evaluations,
  including unresolved ones. No started evaluations yields `None`, not zero.
- Time to pass is elapsed calendar days from the first evaluation's phase start,
  conditional on its passing inside the window. Other windows have no pass time.
- Payout frequency counts windows with a receipt event; approvals alone do not
  count. MLL frequency counts windows with an actual `FAIL_TRAILING_DD`, not
  `CAPPED_OUT` or inactivity. Both lifecycle frequencies include later attempts.
- Cash, fees, receipts, executed trades and failed attempts cover the whole
  window lifecycle. Outstanding payouts are separate. Maximum **external cash**
  drawdown measures peak-to-trough cumulative receipts minus fees, starting at
  zero; it is not trading equity drawdown and does not infer intratrade prices.
- Distributions report arithmetic mean, median, min/max and linear-interpolated
  5th/95th percentiles. These describe dependent window outcomes, not confidence
  bounds or a count of independent futures. Overlapping profits are never summed.

Passing `rolling` to `Engine.fit` splits whole sessions first, then optimizes the
mean IS-window objective. No window crosses the split. Search settings and window
length/spacing must be selected before inspecting OOS. The frozen policy is
reported on OOS windows; `fit.score` is their mean objective. The original single
chronological replays remain available alongside `in_sample_rolling` and
`out_of_sample_rolling`, so neither view is silently replaced. Compact window
records retain session bounds and fingerprints; replay those exact bounds with
`Engine.backtest` when a full event ledger is needed.

## Fitting and reporting

`Engine.fit` splits whole chronological sessions at floor(0.70 * session_count).
It selects a policy exclusively on IS and evaluates that frozen policy on OOS
with a fresh account and the same initial-wallet scenario.

Default criterion: **observed net external cash / elapsed calendar days**.
Maximization is the default; custom lambdas may use `direction="minimize"`.
All supplied objectives score the complete lifecycle, not separately optimized
evaluation/funded phases. `fit.score` always means OOS; the IS score is explicitly
named. No confidence interval or population-EV claim is inferred from one history.

Outstanding balances, unreceived payouts and live value are NOT silently called
losses or fair values. The objective is finite-horizon realized cash extraction;
inspect `status`, `outstanding_payouts` and the full event trace alongside it.
Search is heuristic, not a guarantee of a global optimum.

## Cash risk and bankroll reporting

Added 2026-10-03. The report concerns **external cash**, not the nominal $50,000
account balance. One observation is a complete fixed-horizon lifecycle path,
including retries, cash fees, actual receipts and live-handoff restarts.

`RiskConfig` accepts a cent-valued `bankroll` (or None),
`target_ruin_probability` in [0,1], a `confidence` in (0,1), a `tail_probability`
in (0,1], and a tuple of requested percentile fractions. The defaults are a
1% ruin target, 95% confidence, 5% worst tail, and P1 through P99 at nine levels.
No finite bankroll is silently inferred when the Python configuration omits it.
The dashboard uses the configured initial wallet.

Pass `risk=RiskConfig(...)` to `Engine.rolling_backtest` or `Engine.fit` to attach
a report to the final OOS `out_of_sample_rolling.risk`. Search candidates and
training windows do not receive confidence claims. The reporting configuration
does not change the optimization objective or select candidates.

The theoretical `evaluate_targets` and `fit_targets` API also accepts `risk`.
`fit.holdout.risk` and `fit.baseline_holdout.risk` refer only to independent held-out
model tapes. `fit.training.risk` stays None. Independent here means conditional
on a fixed model and already-selected policy; it is not evidence of real-market
independence or a correct market model. Pass `risk=None` to omit the report.

### Metrics and denominators

- Descriptive mean, variance (N denominator), standard deviation, min/max,
  median and configurable linearly interpolated percentiles. Sample variance
  (N-1 denominator) is a separate API field and is undefined at N=1.
- Profit, loss and break-even frequencies, all with the full path count as
  denominator. A stopped zero-wallet path is not discarded from the sample.
- Net cash/day uses each path's actual calendar duration, not an annualization.
- Loss is max(0, -net cash). VaR uses the empirical inverse CDF; expected
  shortfall averages the worst selected fraction, including fractional weight
  on the boundary observation. Worst-tail mean **net cash** is also reported.
- Maximum cash drawdown is the peak-to-trough decline in receipts minus fees.
  Longest underwater duration starts when cash falls below its running peak and
  includes unrecovered drawdowns through the observation horizon.
- Payout count, receipts, fees, outstanding payouts, attempts, failures and
  conditional time to first receipt. The time distribution reports its own N;
  paths without receipts are not given artificial zero waiting times.
- Net cash divided by the configured initial wallet is available only when
  that wallet is positive. It is not annualized and does not value the funded
  account, pending payments or live trading rights.

Performance distributions retain the original configured-wallet behavior.
Capital requirements are a separately labeled unrestricted-wallet counterfactual.
Changing the analysis bankroll does not turn unrestricted profits into finite-wallet
profits. Rerun the engine with the desired `initial_wallet` for that performance.

### Finite-horizon funding failure and required capital

Finite-horizon funding failure is the **first inability to pay a required evaluation/reset/activation fee
within the observed horizon**, even if a later delayed receipt would let trading
resume. Equality with the fee is enough to continue. A prop-account breach is
not investor ruin, and ending the horizon with no wallet wait is not perpetual
survival. A fee for an attempt outside the horizon is not invented.

For each unrestricted-wallet cash path C(t), starting at zero:

```text
required bankroll B* = ceil_to_cents(max(0, -min_t C(t)))
empirical ruin at B = count(B* > B) / N
```

The same ordered policy/trades or uniform tape are used for the counterpart.
Fees and receipts retain engine ordering, even at identical timestamps.
Approvals are not spendable receipts. Account commissions reduce trade P&L in
the engine; they are not charged a second time to the external wallet.

`cash_risk_path(result)` refuses a result containing `wallet_wait` unless given
`unrestricted=...` from the matching horizon, policy and cost/delay configuration
with `initial_wallet=None`. The rolling and model adapters do that replay
automatically only when needed. Paths that never waited already match their
unrestricted trajectories and do not need a second run.

For a requested empirical ruin fraction a, the minimum observed-sample bankroll
is an order statistic, not an interpolated percentile: with sorted requirements
b1,...,bN, select b[N-floor(a*N)], or zero when a=1. Ties and exact payment
boundaries are retained. The full step curve is exported.

### Confidence and insufficient samples

Historical starts are dependent; neither overlapping nor merely non-overlapping
windows are automatically treated as IID. Historical reports therefore provide
no probability confidence intervals or confidence-supported capital. Single
histories show actual path facts only, not a future distribution.

For independently generated **held-out** model paths, marginal two-sided Wilson
intervals describe profit, loss, payout and predeclared-bankroll ruin probabilities.
A distinct one-sided order-statistic tolerance bound supplies the confidence-supported
capital: choose the largest k with BinomialCDF(k; N, a) <= 1-confidence, and use
b[N-k]. If no such k exists, return None with `insufficient_independent_paths`.
This avoids applying a fixed-threshold confidence interval to a data-selected
bankroll. The bound is distribution-free for IID requirements, conservative for
ties, and conditional on the model. General distinctions between percentile
coverage and confidence are described by [NIST's tolerance-limit reference](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/tolelimi.htm).

The best possible zero-failure upper bound is
`1 - (1-confidence) ** (1/N)`. It is reported as sample-resolution information,
not as the current bankroll's risk when failures have occurred. At a 1% target
and 95% one-sided confidence, at least 299 independent holdout paths are needed
even when none fail. A 30-path holdout cannot certify that target by this method.
Zero observed failures and zero future risk are never equated.

No interval covers model misspecification, future firm changes, missing costs,
nonstationarity, repeatedly tuning against the holdout, or choosing a favorable
reporting horizon after inspecting outcomes. No Sharpe ratio, infinite-horizon
ruin estimate or annualized return is inferred from fee/payout cashflows.

## LucidFlex evidence and deliberate boundaries

Official sources rechecked 2026-09-26:

- [Evaluation](https://support.lucidtrading.com/en/articles/12945790-lucidflex-evaluation-account):
  50K target $3,000, loss distance $2,000, 50% consistency, 4 minis/40 micros.
- [Drawdown](https://support.lucidtrading.com/en/articles/12945815-lucidflex-drawdown):
  EOD ratchet, $50,100 lock, request-triggered locking, breach on touching floor.
- [Payouts](https://support.lucidtrading.com/en/articles/12945796-lucidflex-payouts):
  five $150 days, cycle profit, $500 minimum, 50%/$2,000 maximum, 90% trader share.
- [Scaling](https://support.lucidtrading.com/en/articles/12945808-lucidflex-scaling-plan):
  2/20, 3/30, 4/40 from profit thresholds $0, $1,000, $2,000; EOD updates.
- [Hours](https://support.lucidtrading.com/en/articles/11404729-allowed-trading-times):
  ordinary session window and holiday-close override requirements.
- [Commissions](https://support.lucidtrading.com/en/articles/11508978-approved-products-and-commissions):
  quoted per side; the replay input is round trip per executed contract.
- [Inactivity](https://support.lucidtrading.com/en/articles/11404632-inactivity-policy):
  $1 activity requirement within 30 calendar days, and a 30-day reset window
  before breached evaluations are deleted.

User decisions, not fresh official quotations: retained-profit payout base;
$1 new cycle profit; gross withdrawal versus net receipt; continuous open-equity
MLL checks; no trading while pending; omit the undocumented consistency cushion;
cap stop risk to buffer and terminate an insufficient-buffer attempt.

Interpretations: lowest funded tier continues below zero profit; tier thresholds
use continuous dollar bands; payout reductions update scaling immediately;
America/New_York handles Eastern daylight saving. Fee examples are the user's
dated screenshots, not refreshed checkout prices.

Still outside faithful simulation: discretionary live transfer/reviews, current
agreement restrictions, unprovided payment-provider charges, exchange calendars,
and [microscalping review](https://support.lucidtrading.com/en/articles/11404742-prohibited-microscalping).
Do not use this profile to certify review-triggering strategies.
Other accounts needing unsupported rules fail capability validation rather than
silently executing with those rules omitted.

## Multi-path and ultimate ruin

`Engine.ruin(spec, history, policy, config, simulation=RuinConfig(...), risk=RiskConfig(...))`
returns two distinct models. The dashboard runs this only after policy selection;
on a fit request it supplies the final 30% of sessions, never IS or mixed IS/OOS.
The seed, source fingerprint, source size, path configuration, per-path records
and extracted cash-cycle records are retained in the JSON export.

### Full dated-engine bootstrap

`RuinConfig` defaults to 100 independent paths of 250 sessions, stationary
whole-session blocks of mean length 5, and seed 1729. Mean block 1 is IID session
resampling. Blocks wrap around the source's end; trades inside each session keep
their original ordering, brackets and outcomes. Local trade clocks are moved to
the firm's open-weekday calendar. Original gaps and exchange holidays are not
retained. A nonexistent daylight-saving clock is rejected rather than repaired.
Synthetic histories are bootstrapped as realized data, not regenerated from their
original stochastic parameters. Vary block length and source period for sensitivity.

Every path uses the canonical dated lifecycle, not the legacy summary executor.
Within a path, resets, live handoffs, fee selection, payout approval/receipt queues
and risk regimes behave exactly as in bracket replay. All payouts are retained.
Unrestricted funding reveals the entire cash-deficit path; first inability to pay
at a chosen bankroll is then identified without early-stopping selection bias.
The exported cash distributions use the configured wallet, including the engine's
waiting/resumption behavior when late receipts arrive. Ruin still records the first
funding failure. Capital needs and cycle calibration use the unrestricted counterpart;
they are never inferred from prematurely stopped performance paths.

The horizon curve and bankroll curve concern finite time. Independent paths permit
conditional Monte Carlo intervals; they do not turn the source history into a
larger independent historical sample. `ultimate_full_engine.status` remains
`not_identified`: a finite-horizon frequency estimates a lower bound on ultimate
ruin, while the unmodelled continuation may still ruin any survivor.

### IID settled-cycle approximation

Each terminated account (failure or live handoff) whose requested receipts all
arrived contributes `(X,D)`: net external cash and the largest interim cash
deficit **allocated to that account**. Incomplete/unsettled accounts are excluded
and counted. Fractional-cent custom cash flows are rounded adversely. The empirical
law gives every retained cycle equal weight, independent of duration.

`ultimate_cycle_ruin(cycles, bankroll=..., target=.01, confidence=.95, paths=10000,
max_cycles=2000, tail_tolerance=.0001, seed=1729)` also accepts explicitly supplied
`CashCycle(net_cash, required_cash)` values in whole cents. Repeated identical
records express probability weights. At each independent draw:

```text
if bankroll < D: ruin
else: bankroll += X
```

This settled-cycle model deliberately ignores cross-account receipt overlap,
fee-state dependence and serial market dependence. Its initial cycle is drawn
from the same pooled law as later cycles. Calendar time is not simulated here.
Excluding censored accounts can bias the law toward shorter cycles; increasing
Monte Carlo paths does not repair that or reveal unobserved tail outcomes.
All ultimate claims and confidence intervals are conditional on this law.

For finite-support IID increments with possible negative values and mean <= 0,
ultimate ruin is 1 for every finite bankroll. All-nonnegative increments are
handled separately, including zero-net cycles with positive interim funding needs.
For positive mean and negative increments, find a positive adjustment rate r with
`mean(exp(-r*X)) <= 1`. Let `Dmax = max(D)`. The exponential-supermartingale bound is:

```text
ultimate ruin at B <= exp(-r * (B-Dmax)), for B >= Dmax
sufficient bankroll for target alpha = ceil_to_cents(Dmax + log(1/alpha)/r)
```

This is a **sufficient bound, not a minimum-capital estimate**. A strictly positive
rate is conservatively bracketed below the moment-generating-function root.
The exponential ruin-bound framework is described in
[Karl Sigman's random-walk ruin notes](https://www.columbia.edu/~ks20/4703-Sigman/4703-07-Notes-IS.pdf);
the additional Dmax shift covers the account cycle's interim cash excursion.

Monte Carlo stops a surviving path only when its remaining ruin bound is at most
`tail_tolerance`, or at `max_cycles`. Computational-limit survivors are unresolved,
not safe. The reported probability range spans the ruined fraction through the
ruined-plus-unresolved fraction plus escaped-path tail bounds. Confidence bounds
use conservative Bernoulli KL/Chernoff inversion and a union bound for the two
events, adding the continuation tail. They cover sampling and truncation uncertainty
conditional on the cycle law, **not** law-estimation uncertainty or model error.
Zero simulated failures is not silently reported as zero ultimate risk.

## Verification

Tests include hand-calculated complete evaluation/funded/payout traces, fee and
rounding boundaries, variable trade ratios, pending-payment clocks, scaling,
finite-wallet affordability, all 64 six-trade evaluation paths against an
independent full-consistency calculation, and OOS-mutation leakage tests.

Passing tests validates those cases and the declared model, not unknown firm terms.
