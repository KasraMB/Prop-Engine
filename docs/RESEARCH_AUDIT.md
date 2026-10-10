# Research correctness audit

Updated: 2026-10-09. Scope: trade-log prop-firm simulation.

Passing tests or interpreter/kernel agreement is not an independent economic
oracle. The checks below address known defects and test weaknesses; they do not
certify every input, firm agreement or future outcome.

## Retained correctness checks

| Area | Repair and independent evidence |
| --- | --- |
| Optimization | Distinct candidate scores in bracket, rolling and synthetic target search; both directions checked. An always-maximize mutation must fail. |
| CVaR | Fractional empirical tail weights, paired reward/time resampling, invalid settings and undefined rates; a mean-instead-of-tail mutation must fail. |
| Payouts | Explicit blocked and partial requests. Buffer headroom clips maximum withdrawal before testing the minimum request. |
| Chronology | Exact decimal 70/30 boundaries, including 63/27 sessions in a 90-session history. |
| Open equity | A newly effective EOD floor is checked against current open equity without retroactively applying it to earlier marks. |
| Preflight | Unsupported funded payout mechanics fail before evaluation starts. |
| Capital | Unrestricted counterparts must match input identity; synthetic input fingerprints do not depend on wallet truncation. |
| Accounting | Signed long/short symmetry, FIFO fees, reversals, partial fills, multiple instruments and independent cash reconciliation. |
| Uncertainty | Dependence and training selection suppress unsupported inference; bootstrap calculations have independent numerical oracles and memory checks. |

Regressions live in `tests/test_research_audit.py` and the accounting, rule,
optimizer, risk and uncertainty suites. The targeted mutations exist only in
test-process memory. They must be detected by the independent assertions.

Market-data execution and strategy generation are outside this package.
Recorded fills retain their actual quantities and exits. Sizing search uses
the separate bracket-log contract and preserves recorded reward/risk ratios.

## Release checks

Separate-phase search additionally checks single-attempt stopping, delayed and
multiple funded payouts, unresolved-account reporting, distinct candidate
selection in both directions, common-window OOS isolation, archive bounds and
exact input-trade work accounting. Its phase metrics are horizon-specific
screening proxies; final selection always uses the complete IS lifecycle.

CI runs the unit suite on Python 3.11/3.12/3.13, minimum supported numerical
dependencies, isolated wheel/source installation, executable documentation,
browser integration and single-process benchmarks. The published Python bundle
must match the canonical source. Consult the commit's CI run for current counts,
rather than interpreting an old test total as current evidence.

## Research boundaries

Historical windows overlap and are not independent future trials. Model-based
confidence is conditional on the generator and a frozen policy. Closed P&L alone
does not establish intratrade drawdown behavior. Supplied marks establish only
observed equity, not an unobserved path between them.

The Lucid inactivity cutoff is a user-selected scenario, not verified contract
text. Unknown agreements, discretionary reviews and live-account value are not
inferred. Passing tests is not evidence of profitability.

## Research use checklist

1. Freeze the rule version, fees, calendar, code revision and trade-log revision.
2. Audit timestamps, missing sessions, fees and the external strategy backtest.
3. Supply the observation evidence required by the account's drawdown rules.
4. Define sizing regimes and objectives on IS; reserve an untouched final holdout.
5. Inspect all OOS cash/risk metrics, skipped fills and unsettled outcomes.
   Marked account equity is not spendable payout cash.
6. Treat overlapping historical windows descriptively. Report model and
   calibration risk separately from conditional simulation error.
7. Save seeds and configuration with results. Recheck current official rules.
