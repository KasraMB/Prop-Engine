# Separate-phase search

`Engine.fit_phases` is an experimental alternative to joint sizing search.
It reuses the account engine, payout ledger and CMA-ES. There are no worker
threads or processes. `Engine.fit` remains the default joint optimizer.

Input is the existing bracket trade log, not market data. This does not add
target retuning or arbitrary-exit sizing.

## Example

```python
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo
from propfirm_engine import BacktestConfig, BracketHistory, BracketTrade, DollarPolicy, Engine, PhaseSearch, cash_risk_path
from propfirm_engine.firms.lucidflex import replay_50k

day, rows = date(2026, 9, 1), []
while len(rows) < 30:
    if day.weekday() < 5:
        at = datetime.combine(day, time(10), ZoneInfo("America/New_York"))
        rows.append(BracketTrade(at, at + timedelta(minutes=5), day,
                                 100, 800, len(rows) % 5 != 4))
    day += timedelta(days=1)
history = BracketHistory(tuple(rows))
spec = replay_50k(eval_fee=105.20, reset_fee=105, contract_type="micro")
config = BacktestConfig(0, timedelta(0), timedelta(0), timedelta(0))
policy = DollarPolicy.constant(100)
search = PhaseSearch(trade_budget=1000, horizon_sessions=3, stride_sessions=3,
                     archive_size=4, population=4)
result = Engine().fit_phases(
    spec, history, config, policy=policy,
    risk_bounds={"evaluation": (50, 300), "funded": (50, 300)},
    search=search, train_fraction=.7, seed=42,
    objective=lambda replay: replay.net_cash_per_day,
    constraint=lambda replay: cash_risk_path(replay).required_bankroll <= 2000,
)
print(result.policy, result.score)  # frozen policy and OOS score
print(result.out_of_sample_risk["distributions"])
print(result.evaluation_candidates, result.funded_candidates)
print(result.work)
```

These synthetic rows demonstrate the interface, not profitability. Supply your
own strategy's completed trades and predeclare the research settings.

## Screening and censoring

Complete chronological sessions are split before searching. Every phase
candidate sees the same complete IS windows. `horizon_sessions` counts observed
trade-log sessions, not calendar days or days without recorded trades.
`stride_sessions` sets start spacing. Overlapping starts are dependent.

Each screen starts one fresh phase with an unlimited wallet. It stops trading
at pass, failure or live handoff but collects payable receipts through the
window end. Funded screens include activation fees and all payouts. Evaluation
screens charge the initial evaluation fee, without purchasing retries.

Evaluation objectives are observed passes/calendar day, pass-by-horizon
frequency, and fees per observed pass. Funded objectives are net payouts,
total observed duration, payout variance and failure-by-horizon frequency.
Lower duration, variance, cost and failure frequency are preferred in their
respective searches. Higher pass rates and payouts are preferred.

Nondominated candidates are retained up to `archive_size`, using normalized
crowding distance. Equal metric vectors retain the earlier candidate. The
complete baseline always participates in lifecycle selection. Final regimes
never visited on IS retain their baseline risk.

Each candidate contains compact `samples` in window order and a `metrics` dict:

- `unresolved_fraction` is not counted as failure. Durations are censored at the
  window end. Do not interpret them as expected lifetime duration.
- `pass_probability` means passed within the window, with unresolved accounts
  included in the denominator. `passes_per_day` uses all observed exposure time.
- `cost_per_observed_pass` includes fees of failed and unresolved screens and is
  `None` without passes. It is a proxy, not expected acquisition cost with resets.
  For constant entry fees it ranks like pass probability; that redundancy is explicit.
- `mean_duration_days` covers the whole funded phase, not time to first payout.
  Terminal duration stops at failure/handoff; receipts may arrive later.
- `failure_probability` covers any account failure within the declared window,
  including buffer exhaustion and inactivity. It is not eventual ruin or
  specifically breach-before-next-payout probability.
- `mean_net_payout` and `payout_variance` describe receipts less phase fees.
  Outstanding payouts, retained equity and live value are not received cash.

These are historical screening statistics, not lifetime-value estimates.
Use horizon sensitivity checks. Candidate retention does not eliminate sampling
error or search bias. Phase statistics never replace full-lifecycle selection.

## Lifecycle selection and API results

Pairs combine evaluation and funded regimes from shortlisted policies. Pair
order uses an IS-only seed, without favoring a phase objective. Each pair runs
the actual IS chronology with fees, resets, wallet constraints, delays and
payouts. Its funded entry date is determined by its own evaluation performance.
The best eligible lifecycle score wins; ties retain the earlier incumbent.

`direction="minimize"` is supported. Optional `constraint(replay)` returns a
boolean IS-only eligibility decision, such as a cash-drawdown limit. No feasible
candidate raises an error. Callables must be pure, must not inspect OOS and
should not run extra simulations outside the work counter.

The example uses an unlimited simulation wallet and an IS capital constraint.
Do not infer required capital from a wallet-truncated run inside a custom
constraint; `cash_risk_path` rejects that inference without a matching unrestricted
counterpart. Final built-in risk reports supply that counterpart automatically.

The selected policy is frozen before OOS, which starts a fresh account and wallet.
`score`, `policy`, `in_sample`, `out_of_sample`, session partitions and score
properties follow ordinary fitting semantics; `fit` exposes the `HoldoutFit`.
Both `in_sample_risk` and `out_of_sample_risk` retain all cash metrics regardless
of the objective. Wallet-stopped results use an unrestricted counterpart for
capital reporting. The report is `single_history`, not independent-path inference.
Use the existing rolling/bootstrap APIs for further frozen-policy risk studies.

## Work budget and comparison

`Engine.compare_searches(...)` takes the same arguments and returns `joint` and
`separate` results. It does not select an architecture using OOS. The joint arm
uses existing joint CMA-ES with a hard work cap instead of a generation count.

The baseline lifecycle is charged first. Each phase receives up to a quarter
of the remaining work, split among its objectives. Remaining capacity evaluates
pairs. Inadequate budgets or insufficient complete IS windows raise errors.

`trade_budget` caps input trade visits across all candidate runs, including
phase screens and pair selection. A complete supplied window counts even if
the account terminates early: the coordinator still traverses the input.
No trial-ledger cache is kept. Each score cache holds at most 256 entries.
Candidate sample storage scales with archive size times the number of windows.
At most 64 materialized windows are cached; they share immutable trade objects.

The architectures receive equal maximum budgets, not necessarily equal consumed
work. An archive can exhaust its pairs first. Compare `search_trade_visits`,
`unused_budget`, phase/lifecycle evaluation counts and `search_seconds` with
performance. Trade visits are a reproducible work proxy, not identical CPU time.
Final IS/OOS reports and required unrestricted-wallet runs are outside the search
cap, explicitly counted in `report_trade_visits` and `report_seconds`.

Run `python benchmarks/chronological.py` for public paired workloads. Freeze
architecture, bounds, objectives, horizons and archive rules before final OOS.
If OOS chooses the architecture, it becomes validation data; reserve a new holdout.
