# Research correctness audit

Started: 2026-10-09. Baseline: v0.2.0, ee9a60a.

This audit separates implementation defects, weak tests and model limitations.
Passing tests or agreement between two implementations is not an independent
economic oracle. Completion means the findings below are resolved and release
gates pass, not that all possible strategies or future firm rules are certified.

## Findings before repairs

| ID | Area | Finding | Required evidence |
| --- | --- | --- | --- |
| A1 | Optimization | Price, bracket and rolling minimization tests do not distinguish winning candidates. Target-search fee tests can also have tied scores. | Competing candidates, independently ordered scores, both directions, and sign mutations rejected. |
| A2 | Tail statistics | Renewal CVaR tests accept the ordinary mean. Tail counts round upward; settings lack validation; undefined bootstrap rates are dropped and undefined objectives become finite penalties. | Exact fractional-tail arithmetic, validated inputs, undefined rates rejected, mean mutation rejected, batched/full-draw parity. |
| A3 | Payout tests | The named buffer-blocking parity fixture allows a $200 payout and checks no expected amount. | Assert actual blocking and boundary outcomes independently. |
| A3b | Timing tests | The named static-EOD parity fixture constructs a continuous-check rule. | Select EOD explicitly and include intraday breach/recovery that distinguishes the timings. |
| A4 | CI | No job installs research extras; DuckDB/calendar tests can disappear through importorskip. | Required research imports, synthetic-data tests and no dependency skips in that job. |
| A5 | Chronology | Bracket and price splits use binary float multiplication: 90 * .7 truncates to 62 rather than 63 sessions. | Exact decimal split arithmetic and 90-session regressions. |
| A6 | Provenance | MarketFeed overwrites source assumptions; merge drops per-event assumptions. | Wrapper/merge/prepared/streamed preservation and malformed-input rejection. |
| A7 | Capital reporting | Price evaluation calls cash_risk_path without the unrestricted counterpart when a wallet stops the replay. | Identical price/execution replay with unrestricted funding; stopped performance remains separate. |
| A8 | Capital inference | General strategies can inspect wallet. A counterfactual unlimited-wallet run can change their decisions, invalidating a universal capital threshold. | Explicit wallet-invariance contract; suppress unidentified capital/ruin claims by default, retain actual funded-path performance. |
| A9 | Payout parity | Dated ledger clips requests to buffer headroom; summary/reference fire gates block the whole request. Same schema can produce different payouts. | One documented maximum-request rule, independently calculated partial/blocking boundaries across executors. |
| A10 | Overnight risk | EOD balance-based ratchet can raise the floor above open equity without a failure. Reproduction ends with equity 52,500, floor 53,000 and zero failures. | Recheck current equity against the newly effective floor without applying it retroactively to earlier lows. |
| A11 | Preflight | Unsupported funded floor recomputation and absent winning-day rules pass preflight and fail only after evaluation succeeds. | Reject unsupported dated-ledger profiles before replay/search. |
| A12 | Counterfactual identity | Capital validation omits input identity; target research also hashes realized decisions into its input fingerprint, which changes with wallet truncation. | Hash model/policy/tape inputs independently of outcomes; reject counterparts from another history or tape. |
| A13 | Stochastic holdout | General fitting reuses execution seeds across partitions; a reset stochastic strategy can repeat training noise in OOS. | Disjoint default streams, explicit seed overrides, holdout-seed mutation cannot affect selection. |

## Review coverage

- Accounting: independent random cash reconciliation, signed symmetry, FIFO fee
  allocation, reversals, partial fills and atomic baskets. Existing portfolio and
  event fixtures remain gates; agreement alone is not the oracle.
- Rules and execution: observed breach before recovery, exact loss boundaries,
  overnight bases, session/inactivity/DST ties, stale marks, gap/collision scenarios,
  causality under changed future data and explicit unsupported-profile rejection.
- Optimization: distinct economic candidates across bracket, rolling, price and
  target fitting; general-strategy parameter fixtures; both scorer sign and final
  selection; OOS mutation isolation; fresh state, fixed seeds and resume parity.
- Statistics: independent binomial capital-bound calculation, fractional empirical
  tails, ratio-of-sums bootstrap, cash-deficit versus final-loss distinction,
  fee affordability equality, dependent-window inference suppression and censored
  cycle warnings. No model-based interval covers model misspecification.
- Release: complete suite, research extras, isolated artifacts, executable examples,
  synchronized browser sources and measured performance gates.

## Research boundaries

Historical windows are dependent observations, not independent future trials.
Model-based confidence is conditional on the generator and a frozen policy.
Repeatedly inspected holdouts are not untouched validation. Missing price paths,
unknown firm discretion and external callback look-ahead cannot be inferred away.
The Lucid inactivity cutoff is the approved scenario, not verified contract text.

## Repair log

A1-A13 are implemented, with focused regressions in test_research_audit.py and
the existing suites. The search tests inject deterministic candidate proposals
but run real account replays; both score ordering and the selected policy are
asserted. Always-maximize and mean-instead-of-tail mutations are performed only
in process memory and must fail the independent assertions.

The payout buffer is defined consistently as an absolute non-withdrawable
balance: maximum request is min(fraction cap, dollar cap, available headroom),
then tested against the minimum request. This changes old summary results for
custom buffered profiles, not the zero-buffer Lucid reference.

General callbacks can choose different actions at different wallet values.
Without wallet_invariant=True, no single-path capital threshold is claimed.
Evaluate wallet-sensitive policies separately at each bankroll on fixed scenarios;
their ruin probability need not be monotone in bankroll. The declaration also
covers execution/withdrawal callbacks and cannot detect hidden external state.

## Validation record

- Full local suite: 1,727 passed, 11 opt-in checks skipped, two expected legacy
  deprecation warnings. Dedicated jobs exercise the package and browser checks.
- Browser source bundle rebuilt from the repaired package; manifest parity passes.
- All eight single-process runtime and allocation budgets passed on the recorded
  Windows/Python 3.12 hardware, with three warm repetitions per case.
  Quote search used 185,370 peak Python bytes versus 313,934 for research recording;
  median fitting time was 0.573 seconds against the 0.70-second budget.
  Native/process memory is reported separately, not hidden in Python allocations.
- Final clean-install and remote CI evidence will be recorded after completion.

## Research use checklist

1. Freeze the rule version, execution scenario, fees, calendar and data revision.
2. Audit timestamps, missing sessions, roll construction and corporate/contract
   adjustments in the external data producer. The engine does not certify data.
3. Declare input fidelity and run execution-cost/gap/staleness sensitivity cases.
4. Define regimes and objectives on IS; use inner validation for selection.
   Reserve an untouched final holdout after research choices are frozen.
5. Inspect all OOS cash/risk metrics, failures, rejected orders and open/unsettled
   outcomes. Marked equity is not spendable payout cash.
6. Use dependent historical windows descriptively. Independent model scenarios
   support conditional sampling inference only; report model and calibration risk.
7. Save code/input revisions, seeds, settings and callback revision alongside
   results. Recheck actual platform fills and current official rules before use.
