# Research engine plan

Updated: 2026-10-09

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
- [x] Validate input capabilities against the requested execution and firm rules.
- [x] Keep portfolio snapshots separate from retained event history.
- [x] Integrate search, research and trace recording modes with lifecycle results.

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
- [x] Detect breach at the first supplied observation, cancel outstanding orders,
  apply explicit liquidation, and ignore later hypothetical recovery.
- [x] Track closed balance separately from open equity and peak/floor state.
- [x] Support partial exits, overlapping positions and multi-instrument exposure.
- [x] Define account transitions while positions or orders remain open.
- [x] Make skip, order rejection and research abandonment explicit policies.
- [x] Correct inactivity clocks with separate session/expiry cutoffs, timezone,
  non-trading days and exact-time ordering.
- [x] Route the bracket adapter through the shared semantics with parity tests.

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

- [x] Market, limit, stop-market and stop-limit order state machines.
- [x] Amend/cancel, partial fills, linked exits and OCO behavior.
- [x] Fixed, trailing, time, signal and multi-target exits.
- [x] Long/short, pyramiding, reversals and explicit multi-leg execution.
- [x] Recorded-fill, opportunity and strategy-driven replay adapters.
- [x] External causal strategy callbacks, warm-up and fill feedback.
- [x] Instrument-specific costs, ticks, multipliers and mini/micro equivalence.
- [x] Replaceable bar, trade and quote execution models and slippage models.
- [x] Overnight positions and forced exits governed by the selected profile.

Acceptance: the same strategy can enter and exit without changing account code;
changed exits regenerate dependent signals; ambiguous bars are labeled. No order
book or live brokerage integration is required for the first research release.

Quote strategy milestone: `Engine.replay_strategy` now drives `Order`/`Amend`/
`Cancel` actions through the same event accounting and account coordinator.
Explicit quote marks, shared per-side liquidity, adverse execution models,
partial fills, stop gaps, persistent stop-limit triggers, submission-anchored
trailing exits, OCO, linked exits and pending exposure are tested. Immutable
callbacks receive actual account and fill outcomes. Warm-up cannot buy or trade
an account. Multi-leg orders remain sequential, not atomic spread execution.
See [strategy replay](STRATEGY_REPLAY.md). General bar execution, overnight
holding, broader rules and search are still open.

Calendar decision: the owner approved local activity date plus 30 calendar days
at 16:15 New York, including weekends. `LifecycleSpec.inactivity_close` selects
the local cutoff; None preserves the prior elapsed-time scenario. LucidFlex
defaults to the chosen cutoff; `elapsed_inactivity=True` explicitly restores
old research behavior. Tests cover DST, weekend expiry and exact closing ties.
This is a selected scenario, not a newly verified official cutoff.

License decision: the owner selected MIT. LICENSE and package metadata now carry
that license; registry publication is still not authorized.

Quote milestone verification: 1,560 tests passed with 10 optional browser tests
skipped; the subsequently added executable strategy example passed separately.
The browser bundle contains 48 verified Python modules. The synthetic callback
benchmark (Python 3.12.7, three warm runs, trace off) took 0.364/3.192 seconds for
2,000/20,000 quotes. Peak traced Python allocations were 21,763/20,763 bytes.
This holds one position with two fills, includes quote creation and immutable
callback snapshots, and does not measure many-order storage or native memory.

### 4. General firm profiles

- [x] Static, EOD trailing and intraday trailing drawdown on explicit bases.
- [x] Daily loss failure versus daily suspension.
- [x] Evaluation and funded consistency, profit and day-count gates.
- [x] Account-wide order/exposure limits and contract scaling.
- [x] Multiple evaluation stages and configurable phase transitions.
- [x] Payout request, approval/denial, deduction, receipt and processing calendars.
- [x] Configurable withdrawal policy, retained buffer and restart policy.
- [x] Versioned official-source evidence and per-profile acceptance fixtures.

Start with LucidFlex 50K DLL off. Do not equate a registered rule with executor
support. Unknown discretionary decisions remain explicit scenarios.

Observed-rule extension: static and EOD/intraday trailing rules use the existing
reference predicates with observed equity. Continuous peaks ratchet before later
losses; EOD floors do not ratchet on open intraday profits. Daily loss can fail
or suspend, with hard breach preceding soft suspension. Funded consistency gates
and declared largest-day cycle resets reuse the compiled rules. The executor owns
portfolio breach decisions: a closed loss offset by open profit must not cause
the payout ledger to invent a breach. Independent fixtures cover these cases.

Payout callbacks can retain cash, skip requests or return approve/deny decisions.
An explicit ProcessingCalendar rolls elapsed delays through selected processing
hours, weekdays and supplied holidays. These are scenario policies. No holiday
list, discretionary approval probability or account agreement is invented.
The bracket and summary APIs keep their existing narrower capability checks.

Observed-rule verification: 1,577 tests passed, with 10 optional browser tests
skipped and two known deprecation warnings. Nine additional static floor-lock
and processing-calendar boundary cases passed in a subsequent focused run.
The browser bundle contains 49 verified Python modules.

Phase/holding extension: observation replay accepts named evaluation sequences
followed by at most one funded phase. Each stage starts its declared balance and
rule state; activation is charged only at funding. Named phase caps and transition
delays are explicit. Failures restart at the first stage unless retry is disabled.
Handoff can restart or stop. Strategy Abandon is a research action, not a firm
breach, and declares whether another paid attempt should follow.

`LifecycleSpec.flatten_at_close` defaults to True, including LucidFlex. Profiles
that explicitly permit holding can retain positions and GTC orders overnight.
Daily loss can reset from closed balance or prior closing open equity; drawdown
peaks can use a declared balance/equity basis. Open horizon positions remain
marked, not external cash. Tests cover overnight DLL, EOD ratchets, GTC, stage
failures, fees, activation delays and eval-only sequences.

Bracket settlement now calls the same incremental rule observation method as
portfolio replay. Its declared execution assumptions and capability guards remain;
three per-trade NumPy allocations were removed. Focused parity checks cover 138
bracket, price, event and dashboard cases after this consolidation.

Lifecycle milestone verification: the complete suite passed 1,600 tests after
settlement consolidation, with 10 optional browser tests skipped and two known
deprecation warnings. The browser bundle remains synchronized at 49 modules.

### 5. Policy search and validation

- [x] Use the general replay core for every candidate and reported baseline.
- [x] Named regimes and custom causal policies using account and strategy state.
- [x] Fixed or optimized risk, quantity and targets as separate permissions.
- [x] Continuous, integer and categorical parameters and portfolio constraints.
- [x] Custom per-path and distribution objectives with all metrics preserved.
- [x] Distinguish infeasible candidates from simulator errors.
- [x] Chronological 70/30 default, rolling starts and walk-forward evaluation.
- [x] Inner validation, repeated seeds, search diagnostics and policy stability.
- [x] Explicit open-position/warm-up treatment at fold boundaries.
- [x] Preserve cross-asset and strategy-state dependencies during resampling.
- [x] Report historical, execution and market uncertainty separately.
- [x] Finite-horizon funding ruin and separately labeled ultimate approximations.

OOS never chooses candidates, search settings, regime definitions or stopping.
Repeatedly inspected holdouts are research evidence, not untouched final tests.

General fitting milestone: `Engine.fit_strategy` reuses quote strategy replay and
the existing CMA-ES and cash-risk reports. Small discrete spaces are enumerated
within budget. Only named declared parameters change; an external policy can use
phase names, balance, floor, payout days and its causal signal state. Unsupported
execution remains an error, not an infeasible score. The factory and execution
setup must produce fresh state for every window and seed.

MarketTape validates once and stores immutable integer-tick quote columns with
UTC microsecond clocks. Zero-copy session views are shared across candidates.
Rolling starts and walk-forward refits preserve complete ordered market streams.
Externally generated market scenarios are consumed one at a time; the engine
does not shuffle quotes or break cross-asset/state dependencies. Independence
requires an explicit model declaration. Finite-horizon risk uses unrestricted
counterparts when a wallet stops trading; ultimate-cycle integration is separate.

Optional inner validation, repeated seeds and compact trial diagnostics are
available. Generic unvisited parameters and selection stability still need care:
their identification cannot be inferred from an opaque strategy factory.
Resume reconstructs deterministic search steps using completed scores. It checks
input/profile/settings fingerprints and requires a caller revision for opaque
callbacks. Cancellation never returns partial performance. Keyed RandomStream
draws are independent of the number of previously executed opportunities.
See [strategy fitting](STRATEGY_FITTING.md) for contracts and executable examples.

Fitting milestone verification: 1,640 tests passed, 10 optional browser tests
skipped, and two known deprecation warnings. The synchronized browser bundle
contains 52 verified Python files. New acceptance cases include OOS mutation
isolation, mixed-domain resume parity, explicit infeasibility, wallet counterparts,
warmup boundaries, prepared/stream parity and immutable input precision.

### 6. Performance and release

- [x] Prepared immutable columnar inputs and compact numeric event/state storage.
- [x] Incremental equity/rule updates and efficient merging of ordered feeds.
- [x] Reusable buffers and compiled hot paths where profiling supports them.
- [x] Search, research and full-trace recording modes with identical economics.
- [x] Bounded caches, batched scenarios and optional streamed traces.
- [x] Avoid candidate-by-scenario copies of full histories and event ledgers.
- [x] Stable random streams when different policies execute different trades.
- [x] Checkpoint/resume, cancellation and workload/memory estimates.
- [x] Synthetic benchmarks for sequential/concurrent, multi-asset, bar/tick,
  Monte Carlo and complete optimization workloads.
- [x] Record preparation, cold compilation, warm throughput and peak memory.
- [x] Set hardware-specific budgets from measured baselines; gate regressions.
- [x] Reference/accelerated and recording-mode parity tests.
- [ ] Build/install checks, public API examples, license and versioned releases.
- [ ] Synchronize the browser package without creating alternate accounting.

Performance is a gate in every phase, not a final rewrite. Improve single-core
time and memory first. Arbitrary Python callbacks remain supported even when
they cannot use compiled execution. Never drop metrics or change arithmetic to
obtain a favorable benchmark. Monetary scaling and overflow need explicit tests.

Recording milestone: search retains cash records and complete event/order counts,
research retains lifecycle/order logs, and trace also retains observed states.
Optional sinks stream all three event types. Tests compare callback chains,
terminal parent links, duplicate IDs, final positions, full funded payouts and
cash-risk metrics across modes. Compact order identities remain O(unique IDs).
Cash events and account attempts also consume space; no constant-memory claim.

Profiling identified repeated Fraction/tick conversion and flat-book arithmetic.
Tick fractions are now prepared per instrument; flat marks avoid redundant P&L
updates. Replay validation/compilation uses a 128-entry immutable LRU; legacy
trade/account/rule caches have configurable entry bounds. Payout lookup no longer
copies all requests. Exact-time strategy closes now apply the same inactivity
tie rule as recorded fills. Processing calendars compare UTC clocks across DST
folds and reject nonexistent local cutoffs rather than inventing an offset.

Measurements on Python 3.12.7 / Windows, three warm medians, no JIT in this path:
20,000 mark-heavy streaming strategy observations took 2.393 seconds (8,358/s),
versus the earlier 3.192-second baseline. Prepared replay took 2.151 seconds
(9,300/s), plus 0.635 seconds preparation and 1,520,016 input bytes. Separate peak
Python replay allocations were 15,954/16,308 bytes, excluding the prepared tape.
These small held-position workloads do not represent every strategy.

For 2,000 order-heavy prepared observations, search used 390,399 peak Python
bytes versus research's 1,945,175; warm times were 0.534/0.517 seconds. The measured
benefit here is memory, not speed. Input storage was 152,016 bytes in both cases.
Run `benchmarks/portfolio.py --mode orders --events 200 --recording search` and
repeat with research. Preparation, native/process memory and broader multi-asset,
bar and optimizer budgets still require the remaining release benchmarks.

Retention milestone verification: 1,655 tests passed, 10 optional browser tests
skipped, and two known deprecation warnings. The prior fitting checkpoint also
passed GitHub's Python 3.11/3.12/3.13 and real-browser jobs; Pages deployed it.
The local browser bundle at this checkpoint contains 52 verified Python files.

Input extension: external Opportunity policies use the same order callbacks and
skip pre-horizon signals instead of replaying them as fresh OOS entries. TradeTick
and Bar adapters require explicit last_trade/ohlc_path fidelity, synthetic spread
and liquidity. Provenance travels on individual Market events and prepared tapes;
iterators cannot silently strip the approximation guard. Ordered feed merging
requires explicit atomic/rejected timestamp ties. See [market inputs](MARKET_INPUTS.md).

Basket selects all-or-none market/limit legs for distinct instruments on one
observation. Portfolio rules see combined settlement, not artificial intermediate
leg loss. Legged orders remain available. Nested OCO/parent baskets and atomic
partial fills are rejected, not guessed. Pending exposure remains conservative.
Acceptance cases include liquidity waits, IOC, cancellation, fees, loss/recovery
and offsetting exits that would falsely breach under sequential settlement.

The Lucid reference now carries rule_version lucidflex-50k-2026-10-08 with eight
official help URLs. Evaluation, drawdown, payout, scaling, hours, inactivity,
commissions and review pages were rechecked. Numeric, interpreted and unmodelled
terms are separated in the execution guide. No additional firm preset is claimed
verified. Preflight reports the supported adapter/profile capabilities.

General strategy evaluations retain complete cash-cycle summaries and expose the
existing IID ultimate approximation separately from finite-horizon risk. Denied
requests no longer count as missing receipts, and voluntary closed abandonment
completes a cash cycle without becoming a firm breach. Censored cycles are counted
and excluded with an explicit selection-bias warning.

Adapter milestone verification: 1,682 tests passed, 10 optional browser tests
skipped, and two known deprecation warnings. Public market-input examples execute
without external data. The synchronized browser bundle contains 55 verified
Python files. Remaining work is search stability, workload/budget reporting and
the final build, performance and release gates below.

## Required acceptance cases

Release candidate: optimizer starts now use separate search seeds, with per-run
winners, parameter ranges and selection frequency. OOS-mutation and resume tests
cover repeated searches. Pre-run workload estimates include warmup, validation,
final reporting and potential wallet counterparts; arbitrary callback memory is
not bounded. Provenance from every observation survives preparation and replay.

The synthetic matrix and hardware-specific budgets are documented in
[performance](PERFORMANCE.md). The calibrated local budget passed separately.
Existing fused summary kernels remain available with reference parity; general
callbacks retain exact Python execution and shared prepared arrays. No second
compiled account implementation was added merely to claim acceleration.

The 0.2.0 source distribution and wheel both built, passed content allowlists,
installed into separate targets and ran the documented fitting example outside
the repository. Fifty-three focused packaging/stream tests passed. Final full
suite passed 1,691 tests with 11 opt-in skips and two known deprecation warnings.
The package check passed separately. A final export audit found two result types
sharing a public name; RollingReplayResult now exposes historical replay without
changing the existing optimizer RollingResult import. Remote CI, browser
deployment and version tag are still pending.

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

- The official inactivity page still does not specify a cutoff. The selected
  16:15 calendar-date convention above is explicitly user-approved, not official.
- Existing sub-contract policies skip while insufficient account buffer ends an
  attempt. The general API must expose the chosen behavior, not silently change
  existing saved runs. The user's research convention is abandonment when unable
  to trade; voluntary zero allocation remains a distinct action.
- Specify mark basis, staleness, liquidation and same-time feed ordering per
  execution profile. No missing quote or intrabar path may be guessed silently.
- Stop/target outcomes, forced closes and post-handoff restarts from earlier
  experiments are adapters/scenarios, not restrictions of the general core.
- MIT was selected by the owner. GitHub pushes are authorized; publishing to a
  package registry is not implied.

## Completion

A strategy with different entry, exit and position-management logic can be added
without changing account code. Supported profiles, input fidelity, tests and
performance budgets are visible. Required data and unsupported features are
reported before execution. Publish the engine and synthetic fixtures only, not
market data or a hosted multi-user service.
