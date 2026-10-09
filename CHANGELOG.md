# Changelog

## 0.2.3 - 2026-10-09

- Consistent mean-uncertainty reports across replay, fitting and cash-risk APIs.
- Explicit sampling contracts, conditional mean standard errors, opt-in IID and
  circular block bootstrap intervals, paired comparisons and scenario sensitivity.
- Training selection and dependent historical windows withhold inference;
  parameter, model, selection and numerical uncertainty remain explicitly unmeasured.
- Bounded bootstrap batches, reused summary moments and executable documentation.

## 0.2.2 - 2026-10-09

- Printable configured-rule reports through `firms.lucidflex.info`, `Engine.info`
  and `account_info`, including all phases, payout schemas and lifecycle settings.
- Explicit unspecified-fee labels in the reference preview; custom configuration,
  runtime scenarios, profile assumptions and source references remain separate.
- Tested quickstart examples and clean-install coverage for the reporting API.

## 0.2.1 - 2026-10-09

- Exact decimal IS/OOS boundaries, including 63/27 sessions for a 90-session history.
- Disjoint execution seeds for training, inner validation and holdout strategy paths.
- Fractionally weighted renewal CVaR tails and validated objective settings.
- Undefined renewal rates fail explicitly instead of receiving a finite score;
  bootstrap batches bound temporary index storage without changing seeded draws.
- Overnight equity checked against newly effective EOD drawdown floors.
- Consistent maximum payout sizing across summary, reference and dated engines:
  buffer headroom clips the request before minimum-request eligibility is tested.
- Unrestricted price counterparts for wallet-stopped capital reports.
- General strategy capital/ruin estimates require wallet_invariant=True; otherwise
  those estimates are unidentified, while actual performance remains available.
- Source assumptions retained through feed wrapping and merging; capital
  counterparts must match input fingerprints; unsupported profiles fail preflight.
- Independent economic fixtures, targeted in-memory mutation tests and research CI.
- Explicit timezone conversion dependency for clean research-extra installations.

Rerun affected prior studies. The buffer change can alter custom funded profiles;
LucidFlex's zero buffer is unaffected. The wallet declaration covers strategy,
execution and scenario callbacks and is not automatically proven by the engine.

## 0.2.0 - 2026-10-09

- General portfolio replay with arbitrary fills, partial exits and concurrent assets.
- Causal strategies, external opportunities, orders, trailing exits and atomic baskets.
- Explicit quote, last-trade and OHLC-path execution scenarios.
- Multi-stage accounts, observed drawdown, suspension, payout decisions and calendars.
- Shared market tapes, typed parameter search, chronological holdouts and walk-forward fits.
- Complete performance/risk reports, separate cycle-ruin estimates, search diagnostics and resume.
- Compact recording, bounded caches, keyed random draws and measured performance budgets.
- Versioned LucidFlex evidence, MIT licensing and clean artifact installation checks.
- Distinct rolling replay result export; existing optimizer result imports remain compatible.

The prior bracket and summary APIs remain available with their narrower input
contracts. Missing market paths, firm discretion and unknown agreements are not
inferred. This release is not a live execution or contractual certification.
