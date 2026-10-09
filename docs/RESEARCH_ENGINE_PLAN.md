# Research engine plan

Updated: 2026-10-08

## Goal

Publish a reusable futures prop-firm research engine. Strategies and market data
remain outside the package. Support general trading behavior without forcing
strategies into sequential stop-or-target bets. Preserve one account with repeated
attempts as the default lifecycle. Other account arrangements are later extensions.

This is the implementation checklist and handoff record. Update it in the same
commit as completed work. A checked item needs tests or benchmark evidence.
Existing bracket and price APIs remain supported during migration.

## Design

- One chronological execution and account core, not another unrelated simulator.
- Reuse rule definitions, compiler, payout ledger, cash-risk calculations, search,
  caches and current regression fixtures.
- Keep orders, fills, positions, marks, account balance and external cash separate.
- Process market, execution, calendar and lifecycle events on one ordered clock.
- Strategies receive causal state and submit decisions through a small interface.
- No stop or target is required merely to represent a position. Dollar-risk sizing
  needs an explicit risk basis; quantity sizing does not need an invented stop.
- A recorded execution, a resizable opportunity and a strategy callback are
  different input contracts. Changing exits can change later opportunities.
- Ordered marks establish equity only at their supplied resolution. Closed P&L,
  MAE and OHLC do not establish an exact unobserved intrabar path.
- Reject unsupported capabilities before search. Approximation requires an
  explicit choice and travels with every result.
- Firm rules, strategy decisions, execution scenarios and research lifecycle
  choices are independent. Never report research abandonment as a firm breach.

## Scope and phases

### 0. Baseline and version control

- [x] Back up existing history and uncommitted work.
- [x] Run the current tests and commit the existing price-replay work separately.
- [x] Commit this roadmap and link it from the design guide.
- [x] Verify author and committer, push main, and verify the remote commit.

Baseline: `1895652` passed 1,457 tests, with 10 optional browser tests skipped.
The history bundle and working-file archive are local ignored results. Private
papers and notes were not staged. Later commits must repeat identity/push checks.

Do not commit user notes, papers, market data, generated results or local archives.
Use KasraMB's configured Git identity. Use short ASCII commit messages and concise
names. New code and comments use ASCII; comments explain only non-obvious intent.
Do not rewrite history, fabricate authorship metadata or add attribution trailers.

### 1. Event and portfolio contracts

- [x] Typed instruments, signed fills, ordered marks and deterministic event keys.
- [x] Position accounting for long, short, partials, scale-ins and reversals.
- [x] Concurrent instruments and account-wide mark-to-market equity.
- [x] Explicit fees, price precision, contract units and gross/net definitions.
- [x] Validate chronology, identifiers and finite values.
- [ ] Validate input capabilities against the requested execution and firm rules.
- [x] Keep portfolio snapshots separate from retained event history.
- [ ] Integrate search, research and trace recording modes with lifecycle results.

Acceptance: hand-calculated partial-exit and reversal fixtures, same-time ordering,
multi-instrument equity, fees exactly once, no future marks, and invalid-input
rejection. No claim of prop-firm fidelity until integrated with the lifecycle.

Implemented: `events.py`, `instruments.py` and `portfolio.py`. `Book` is a FIFO
accounting component; it does not enforce firm rules or replace existing replay.
Snapshots retain mark timestamps. Thirty new tests include independent random
cash reconciliation and bounded retention of closed lots.

Verification: 1,487 tests passed, 10 optional browser tests skipped. The browser
source bundle contains 45 verified modules. A wheel build succeeded and its
contents were checked: engine and package metadata only, no market data or local
inputs. The broader release gates and lifecycle integration remain unfinished.

Baseline benchmark: Python 3.12.7 on Windows, three warm runs with synthetic event
creation included. 20,000 events took 0.713 seconds (median); 200,000 took 6.966
seconds. Separate peak traced Python allocations were 3,524 and 3,484 bytes. This
alternating flat-position workload retains no trace; it is not a native-memory,
open-lot, full-account or optimizer benchmark. Run `benchmarks/portfolio.py` to
reproduce it. Compiled execution and the broader performance gates remain open.

### 2. Incremental account replay

- [x] Adapt the current lifecycle coordinator to fills and marks without jumping
  from entry directly to exit.
- [x] Reuse existing closed-P&L gates, payout ledger and retry/wallet accounting.
- [ ] Detect breach at the first supplied observation, cancel outstanding orders,
  apply explicit liquidation, and ignore later hypothetical recovery.
- [x] Track closed balance separately from open equity and peak/floor state.
- [x] Support partial exits, overlapping positions and multi-instrument exposure.
- [ ] Define account transitions while positions or orders remain open.
- [ ] Make skip, order rejection and research abandonment explicit policies.
- [ ] Correct inactivity clocks with separate session/expiry cutoffs, timezone,
  non-trading days and exact-time ordering.
- [ ] Route the bracket adapter through the shared semantics with parity tests.

Acceptance: early-exit import through fees and retries, unrealized breach before a
profitable exit, simultaneous portfolio moves, and trace/summary metric parity.

Recorded-fill milestone: `Engine.replay_events` adapts the existing `_Replay`
coordinator and `_ReferenceSim` rules to streaming fills and atomic portfolio
marks. Arbitrary exits, FIFO partials/reversals, EOD floors, continuous observed
breaches, contract units, payouts, wallet constraints and retries are covered.
General order and calendar events remain open. See [event replay](EVENT_REPLAY.md).

Explicit `observed_marks` fidelity is required. Stale marks fail validation.
Forced closes use fresh last marks and a selected per-contract fee. Positions
are closed at session cutoff; evaluation transitions wait for flat. A skipped
source portfolio must return to flat before recording resumes. These are explicit
recorded-history scenarios, not general order rejection or callback semantics.
The account cannot recover after breach. Approved receipts survive termination.
Order cancellation and the separate inactivity expiry calendar remain open.

Acceptance evidence includes a hand-calculated full LucidFlex payout path,
100-session randomized bracket parity, partial exits and allocated fees,
simultaneous offsetting marks, breach before a later winning exit, pending-payout
skips, live handoff, wallet exhaustion, exact inactivity ties, trace parity and
future-data prefix invariance. The adapter uses the existing support guards;
other firm-rule combinations and optimizer input contracts are not expanded.

Verification: 1,530 tests passed, 10 optional browser tests skipped and two known
deprecation warnings. The executable event API example passes. The browser
bundle contains 46 verified Python modules. The wheel builds and contains only
the engine and package metadata. Private papers, notes and market data remain
outside version control.

Mark-heavy benchmark: Python 3.12.7 on Windows, three warm medians including event
creation and replay setup. With trace off, 2,000/20,000 events took 0.161/1.664
seconds, with 19,302/18,792 peak traced Python bytes. With trace on, they took
0.192/1.800 seconds and 387,434/3,774,910 bytes. One held contract and two fills
isolate mark-stream scaling; execution/lifecycle logs still grow with fills.
This is not a compiled, multi-asset, native-memory or optimization benchmark.

### 3. Orders and strategy adapters

- [ ] Market, limit, stop-market and stop-limit order state machines.
- [ ] Amend/cancel, partial fills, linked exits and OCO behavior.
- [ ] Fixed, trailing, time, signal and multi-target exits.
- [ ] Long/short, pyramiding, reversals and explicit multi-leg execution.
- [ ] Recorded-fill, opportunity and strategy-driven replay adapters.
- [ ] External causal strategy callbacks, warm-up and fill feedback.
- [ ] Instrument-specific costs, ticks, multipliers and mini/micro equivalence.
- [ ] Replaceable bar, trade and quote execution models and slippage models.
- [ ] Overnight positions and forced exits governed by the selected profile.

Acceptance: the same strategy can enter and exit without changing account code;
changed exits regenerate dependent signals; ambiguous bars are labeled. No order
book or live brokerage integration is required for the first research release.

### 4. General firm profiles

- [ ] Static, EOD trailing and intraday trailing drawdown on explicit bases.
- [ ] Daily loss failure versus daily suspension.
- [ ] Evaluation and funded consistency, profit and day-count gates.
- [ ] Account-wide order/exposure limits and contract scaling.
- [ ] Multiple evaluation stages and configurable phase transitions.
- [ ] Payout request, approval/denial, deduction, receipt and processing calendars.
- [ ] Configurable withdrawal policy, retained buffer and restart policy.
- [ ] Versioned official-source evidence and per-profile acceptance fixtures.

Start with LucidFlex 50K DLL off. Do not equate a registered rule with executor
support. Unknown discretionary decisions remain explicit scenarios.

### 5. Policy search and validation

- [ ] Use the general replay core for every candidate and reported baseline.
- [ ] Named regimes and custom causal policies using account and strategy state.
- [ ] Fixed or optimized risk, quantity and targets as separate permissions.
- [ ] Continuous, integer and categorical parameters and portfolio constraints.
- [ ] Custom per-path and distribution objectives with all metrics preserved.
- [ ] Distinguish infeasible candidates from simulator errors.
- [ ] Chronological 70/30 default, rolling starts and walk-forward evaluation.
- [ ] Inner validation, repeated seeds, search diagnostics and policy stability.
- [ ] Explicit open-position/warm-up treatment at fold boundaries.
- [ ] Preserve cross-asset and strategy-state dependencies during resampling.
- [ ] Report historical, execution and market uncertainty separately.
- [ ] Finite-horizon funding ruin and separately labeled ultimate approximations.

OOS never chooses candidates, search settings, regime definitions or stopping.
Repeatedly inspected holdouts are research evidence, not untouched final tests.

### 6. Performance and release

- [ ] Prepared immutable columnar inputs and compact numeric event/state storage.
- [ ] Incremental equity/rule updates and efficient merging of ordered feeds.
- [ ] Reusable buffers and compiled hot paths where profiling supports them.
- [ ] Search, research and full-trace recording modes with identical economics.
- [ ] Bounded caches, batched scenarios and optional streamed traces.
- [ ] Avoid candidate-by-scenario copies of full histories and event ledgers.
- [ ] Stable random streams when different policies execute different trades.
- [ ] Checkpoint/resume, cancellation and workload/memory estimates.
- [ ] Synthetic benchmarks for sequential/concurrent, multi-asset, bar/tick,
  Monte Carlo and complete optimization workloads.
- [ ] Record preparation, cold compilation, warm throughput and peak memory.
- [ ] Set hardware-specific budgets from measured baselines; gate regressions.
- [ ] Reference/accelerated and recording-mode parity tests.
- [ ] Build/install checks, public API examples, license and versioned releases.
- [ ] Synchronize the browser package without creating alternate accounting.

Performance is a gate in every phase, not a final rewrite. Improve single-core
time and memory first. Arbitrary Python callbacks remain supported even when
they cannot use compiled execution. Never drop metrics or change arithmetic to
obtain a favorable benchmark. Monetary scaling and overflow need explicit tests.

## Required acceptance cases

- Partial exits, scale-ins, reversals and fees reconcile independently.
- Long/short symmetry under symmetric execution assumptions.
- Portfolio marks use concurrent prices, not sums of unrelated trade extrema.
- Breach followed by recovery cannot revive an account or credit later P&L.
- Intraday peak ratchets before a subsequent loss; EOD peaks do not ratchet early.
- Holidays, DST, session boundaries and payout/inactivity event ties are explicit.
- Future-data changes cannot alter earlier decisions.
- Limits include pending exposure according to profile semantics.
- Equivalent adapters agree under equivalent execution assumptions.
- Batch size, trace mode and resume boundaries do not change economics.
- The engine imports and builds without private data or research dependencies.

## Open decisions

- The user states inactivity expiry is at 16:15. Confirm the exact date/calendar
  convention from official evidence before replacing the elapsed-day scenario.
- Existing sub-contract policies skip while insufficient account buffer ends an
  attempt. The general API must expose the chosen behavior, not silently change
  existing saved runs. The user's research convention is abandonment when unable
  to trade; voluntary zero allocation remains a distinct action.
- Specify mark basis, staleness, liquidation and same-time feed ordering per
  execution profile. No missing quote or intrabar path may be guessed silently.
- Stop/target outcomes, forced closes and post-handoff restarts from earlier
  experiments are adapters/scenarios, not restrictions of the general core.
- The public license needs an explicit owner choice before a package release.
  GitHub pushes are authorized; publishing to a package registry is not implied.

## Completion

A strategy with different entry, exit and position-management logic can be added
without changing account code. Supported profiles, input fidelity, tests and
performance budgets are visible. Required data and unsupported features are
reported before execution. Publish the engine and synthetic fixtures only, not
market data or a hosted multi-user service.
