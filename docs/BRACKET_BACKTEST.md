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
remains active. New replay functionality is not exposed through the legacy
dashboard UI, although its Python modules are included in the browser source mirror.

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
- Five approved payouts in the Lucid profile produce `LIVE_HANDOFF`, stop new
  attempts, and leave future live value unvalued. Previously approved receipts
  are still processed if they fall inside the observed horizon.
- Observation runs from the first declared session's opening to the last
  declared session's close. Gaps/weekends count as elapsed time. Cash after that
  horizon is not included; `outstanding_payouts` is separately reported.

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

## Verification

Tests include hand-calculated complete evaluation/funded/payout traces, fee and
rounding boundaries, variable trade ratios, pending-payment clocks, scaling,
finite-wallet affordability, all 64 six-trade evaluation paths against an
independent full-consistency calculation, and OOS-mutation leakage tests.

Passing tests validates those cases and the declared model, not unknown firm terms.
