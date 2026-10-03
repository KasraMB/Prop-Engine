# Design overview

The package separates strategy inputs, account mechanics, execution assumptions,
cashflows and policy optimization. Strategies are external input producers.
The initial operating scope is one active account with repeated attempts.

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
