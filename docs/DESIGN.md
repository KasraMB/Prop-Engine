# Design overview

The package separates strategy inputs, account mechanics, execution assumptions,
cashflows and policy optimization. Strategies are external input producers.
The initial operating scope is one active account with repeated attempts.

The [research engine plan](RESEARCH_ENGINE_PLAN.md) tracks the migration to
general event replay, portfolio execution and measured performance budgets.

## Components

| Component | Responsibility |
| --- | --- |
| `model.py`, `rules.py`, `schema.py` | Firm-neutral account, phase, rule and payout definitions |
| `validate.py`, `compiler.py` | Reject unsupported combinations; compile rule state |
| `execution.py` | Sequential bracket-history, dollar-policy and lifecycle input contracts |
| `reference.py`, `kernels.py` | Readable rule interpreter and compiled batch execution |
| `feasibility.py` | Project requested size onto executable risk limits |
| `payouts.py`, `cashflows.py` | Payout events and external cash accounting |
| `backtest.py` | Chronological lifecycle orchestration using existing components |
| `optimizer.py`, `fitting.py` | Search and chronological held-out policy evaluation |
| `rolling.py` | Whole-session historical windows and compact outcomes over the same backtest engine |
| `analytical.py` | Explicitly scoped diffusion approximations |
| `firms/` | Account configurations; no firm-name branches in execution logic |

## Public workflows

`Fill`, `Marks` and `Book` are the general portfolio accounting foundation.
Fills carry signed integer contracts, actual execution prices and per-fill fees.
The book uses FIFO lots, supports negative futures prices, and values all open
positions. A `Marks` event applies simultaneous instrument prices atomically.
Event keys are UTC time plus an explicit sequence number; ties are not guessed.
`merge_events` combines ordered feeds lazily and rejects duplicate event keys.

`Book(mark_fills=True)` treats each execution price as the latest mark for that
instrument. With `mark_fills=False`, separate marks are required before fills.
Snapshots expose each mark's timestamp; the accounting layer does not invent a
staleness policy. Snapshots are optional and immutable. Closed lots are discarded;
there is no retained execution history unless a caller records it.

`Engine.replay_events` applies the shared account lifecycle to recorded fills.
`Engine.replay_strategy` adds causal callbacks and quote-based order execution.
Both keep explicit observation and liquidation assumptions. `Engine.backtest`
remains the sequential bracket adapter. Broader rule and optimization work is
tracked in the research engine plan.

`Engine.backtest` processes an immutable sequential stop/target history.
`Engine.fit` selects named dollar-risk regimes on IS and reports the frozen
policy on the chronological OOS partition. Per-trade historical reward/risk
ratios are preserved. Custom objectives receive complete lifecycle results.

Optional rolling evaluation changes the sampling of historical starts, not the
rule engine. Windows retain timestamps and ordering, reset account/wallet state,
and never cross IS/OOS boundaries. Optimizer candidates use mean IS-window scores;
the reported score uses untouched OOS windows. Single chronological replay results
remain available. Overlapping outcomes are not independent observations.

`Engine.run` is the separate resampled Monte Carlo research path. Its strict
input-capability checks remain active; closed summaries do not establish ordered
intratrade paths. Both dashboard pages use the chronological APIs through
one shared JSON adapter. The account trace renders canonical backtest events;
it has no separate accounting implementation. The Monte Carlo dashboard is removed.
The Pages replay runs in a worker with a hash-verified canonical Python bundle.

## Accounting invariants

- Realized account balance, open equity and external wallet cash are different.
- EOD trailing updates do not imply EOD-only breach detection.
- A hard breach cannot be undone by later profit, passing or payout eligibility.
- Position caps include modeled costs and integer quantities.
- Payout request, approval, account deduction and cash receipt are separate events.
- Withdrawals are not trading losses. Retained profit and fresh cycle profit differ.
- Buffer exhaustion, inability to trade, inactivity, observation end and live
  handoff have distinct diagnostic meanings.
- OOS data cannot select policies, regime definitions or search settings.

## Extension boundary

Define new accounts using the existing rule tree and lifecycle configuration.
Add executor support and independently calculated boundary tests before enabling
new rule mechanics. Unsupported rules must fail validation, not disappear.
Keep strategy-specific logic outside the engine package.

The [bracket execution contract](BRACKET_BACKTEST.md) documents implemented
behavior and official-source interpretations. Unknown discretionary enforcement,
execution effects and live-account value must not be described as verified.

## Validation

Use independent hand calculations and exhaustive small paths as well as
reference/kernel parity. Exercise costs, equality boundaries, resets, calendars,
cash affordability and IS/OOS isolation. Keep the browser source mirror aligned.
Passing tests establishes tested behavior, not complete contractual fidelity.

## Serial replay performance

`python benchmarks/chronological.py` measures fixed-seed normal, rolling and
joint-target dashboard workloads. It reports median wall time across three runs,
peak traced Python allocations in a separate warmed run, and a complete-output
hash. It does not use multiprocessing, omit metrics or shorten histories.

The shared replay converts fixed costs and regime budgets to exact rational
dollars once per runner. Net cash is cached on the immutable result; replacing
its ledger creates a fresh cache. Model calendars use a four-entry bounded cache
keyed by timezone, weekdays, start date and session count; account state and
random outcomes are never cached there. Percentiles share one quantile call,
and JSON conversion visits dataclass fields without deep-copying the ledger.
Rolling fixed-policy reports skip an otherwise discarded single-path risk report.

On the October 3, 2026 local run, against `9eadf23`, normal/rolling/target median
times changed from 0.896/2.292/2.145 seconds to 0.535/1.671/1.561 seconds.
Peak traced allocations changed from 722,279/619,247/364,875 bytes to
549,222/610,986/298,929 bytes. All three complete-output hashes matched before
and after. These are workload-specific measurements, not performance guarantees;
traced Python allocations do not include all native or browser memory.

Objective selection does not limit the normal replay's returned performance
metrics or policy. Rolling reports always retain cash and cash/day distributions;
single paths retain observed values, without pretending to estimate variance
from independent paths. Historical reward/risk ratios remain fixed by input.
