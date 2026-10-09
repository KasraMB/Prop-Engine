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
| A2 | Tail statistics | Renewal CVaR tests accept the ordinary mean. The implementation rounds tail sample count up, rather than weighting the boundary observation; settings lack validation. | Controlled bootstrap draws, exact fractional-tail arithmetic, invalid-input tests, mean mutation rejected. |
| A3 | Payout tests | The named buffer-blocking parity fixture allows a $200 payout and checks no expected amount. | Assert actual blocking and boundary outcomes independently. |
| A4 | CI | No job installs research extras; DuckDB/calendar tests can disappear through importorskip. | Required research imports, synthetic-data tests and no dependency skips in that job. |
| A5 | Chronology | Bracket and price splits use binary float multiplication: 90 * .7 truncates to 62 rather than 63 sessions. | Exact decimal split arithmetic and 90-session regressions. |
| A6 | Provenance | MarketFeed overwrites source assumptions; merge drops per-event assumptions. | Wrapper/merge/prepared/streamed preservation and malformed-input rejection. |
| A7 | Capital reporting | Price evaluation calls cash_risk_path without the unrestricted counterpart when a wallet stops the replay. | Identical price/execution replay with unrestricted funding; stopped performance remains separate. |
| A8 | Capital inference | General strategies can inspect wallet. A counterfactual unlimited-wallet run can change their decisions, invalidating a universal capital threshold. | Explicit wallet-invariance contract; suppress unidentified capital/ruin claims by default, retain actual funded-path performance. |
| A9 | Payout parity | Dated ledger clips requests to buffer headroom; summary/reference fire gates block the whole request. Same schema can produce different payouts. | One documented maximum-request rule, independently calculated partial/blocking boundaries across executors. |
| A10 | Overnight risk | EOD balance-based ratchet can raise the floor above open equity without a failure. Reproduction ends with equity 52,500, floor 53,000 and zero failures. | Recheck current equity against the newly effective floor without applying it retroactively to earlier lows. |

## Review still in progress

- Portfolio cash conservation, fees, partials, reversals and atomic settlement.
- Breach timing, EOD floor updates with overnight holdings, calendar/expiry ties.
- Causal execution, stale marks, order priority and profile preflight rejection.
- Holdout isolation, warmup, resume identity and stochastic reproducibility.
- Finite-horizon cash risk, cycle censoring, dependency-aware confidence claims.
- Clean installation, public examples, browser synchronization and performance.

## Research boundaries

Historical windows are dependent observations, not independent future trials.
Model-based confidence is conditional on the generator and a frozen policy.
Repeatedly inspected holdouts are not untouched validation. Missing price paths,
unknown firm discretion and external callback look-ahead cannot be inferred away.
The Lucid inactivity cutoff is the approved scenario, not verified contract text.

## Repair log

No repairs recorded yet.
