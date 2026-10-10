# Changelog

## 0.3.0 - 2026-10-09

- Scope the library to prop-firm simulation from externally generated trade logs.
- Remove market-data backtesting, strategy execution and their fitting APIs,
  bundled studies, guides and dependencies. This is a breaking API change.
- Keep recorded long/short fills, arbitrary exits, partials and portfolio marks.
- Keep trade-history sizing, account rules, payouts, risk and model research.
- Make recorded-fill accounting the default capability check.
- Update executable examples, package checks and streaming benchmarks.

## 0.2.3 - 2026-10-09

- Consistent mean-uncertainty reports across replay, fitting and cash-risk APIs.
- Explicit sampling contracts, conditional standard errors, opt-in IID and
  circular block bootstrap intervals, paired comparisons and scenario sensitivity.
- Training selection and dependent windows withhold unsupported inference.
- Bounded bootstrap batches and reused summary moments.

## 0.2.2 - 2026-10-09

- Configured-rule reports through `firms.lucidflex.info`, `Engine.info`
  and `account_info`.
- Unspecified fees, assumptions and source references remain explicit.

## 0.2.1 - 2026-10-09

- Exact decimal IS/OOS boundaries and fractional renewal CVaR tails.
- Undefined renewal rates fail explicitly; bootstrap temporary storage is bounded.
- Open equity checked against newly effective EOD drawdown floors.
- Consistent payout sizing across summary, reference and dated engines.
- Input identity checks and independent economic/mutation fixtures.

The payout buffer repair can change custom funded-profile results.
LucidFlex's zero buffer is unaffected. Rerun affected prior studies.

## 0.2.0 - 2026-10-09

- Recorded-fill portfolio accounting with arbitrary exits and concurrent assets.
- Multi-stage accounts, observed drawdown, suspension and payout calendars.
- Versioned LucidFlex evidence, MIT license and isolated installation checks.
- Distinct rolling replay result export.

Earlier features outside the current trade-log scope remain in Git history,
not in the supported API.
