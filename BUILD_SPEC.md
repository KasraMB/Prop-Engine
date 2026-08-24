# propfirm_engine — Build Specification

Companion to `ARCHITECTURE.md` and `MODEL_RISKS.md`. This document defines **what to build at each step and what each step must pass** — not how to build it. Each step is a contract: a component, its observable behavior, and a set of test cases that prove it. Implementation is entirely yours; you may optimize, restructure, or rewrite internals freely, and the tests remain the definition of "correct."

**How to read a step.** Each step has:
- **Build** — the component and its *observable contract* (what it takes and returns, and the behavior tests can see). Internal representation, names, and algorithms are unconstrained.
- **Must pass** — described test cases. You construct the concrete inputs and expected outputs; the description fixes *what* each case checks unambiguously. A step is done when every case passes.
- **Freedom** — an explicit note on what you may change without breaking the contract, where useful.

**Rules that hold across all steps:**
- Terminology (`ExitCode`, `StateField`, `Action`, `Severity`, `Timing`, `PayoutSchema`, etc.) matches `ARCHITECTURE.md`. Cross-references point there.
- Later steps depend only on the *observable contracts* of earlier steps, never their internals — so improving an internal never breaks a downstream test.
- Two trust-hierarchy gates from `MODEL_RISKS.md` are build-blocking and called out where they land: **Level 1 oracle parity (§G6)** at Step 6, and the **Level 0 frozen-strategy discipline (§G7)** as a precondition on Steps 9–11's *inputs* (not a component you build).

---

## Step 1 — Integer vocabulary and DSL model

**Build.** The enum vocabulary (`ExitCode`, `StateField`, `Action`, `Severity`, `Timing`, `Stage`) and the frozen DSL tree (`Firm → Program → Variant → Account → Phase`, with `Rule` as an abstract base). See `ARCHITECTURE.md` §3–§5. Observable contract: the enums exist with the documented members and integer semantics (alive = 0, failure ≥ 10); the DSL objects are immutable, hashable by value, and expose the documented accessors (`firm.program(x).variant(y).account(z)`).

**Must pass.**
- An `ExitCode` where `ALIVE == 0`, `PASSED` is a distinct success code, every failure code is `≥ 10`, and all failure codes are mutually distinct.
- A DSL object (e.g. an `Account`) rejects mutation after construction.
- Two DSL objects built from identical values are equal and hash-equal; two differing in any field are not.
- A `Program` constructed with a single account type still exposes a `default` variant, so the `program → variant → account` traversal has one shape whether or not variants were explicitly given.
- An `Account` with one phase and an `Account` with two phases are both valid and distinguishable by their phase tuple.

**Freedom.** Enum integer *values* are yours as long as the ordering semantics hold (alive = 0, failures ≥ 10, failures distinct). Internal storage of the tree is unconstrained.

---

## Step 2 — Rules and the rule registry

**Build.** The concrete rule types (at least: profit target, min trading days, trailing drawdown, daily loss, **consistency-gate** (a payout/pass eligibility conjunct, *not* a failure — `ARCHITECTURE.md` §5), min-winning-days, consistency-raises-target) as subclasses of `Rule`, each declaring its state requirements and compiling to a numeric record carrying its action, severity, timing, and parameters. Plus the `RULE_REGISTRY` mapping rule kinds to implementations and a hard-fail check for unregistered kinds. See `ARCHITECTURE.md` §5, §6, §6a. Observable contract: each rule reports the `StateField`s it requires; each rule compiles to a record whose action/severity/timing/parameters match its definition; the registry check raises on an unknown kind.

**Must pass.**
- A rule that differs from another only in a numeric parameter is the *same type*, unequal by value — a new number is a new instance, not a new type.
- Each rule reports requirements consistent with its behavior (e.g. a trailing-drawdown rule requires the equity and drawdown-floor state; the consistency-gate requires max-day-pnl and cycle-profit state (`EQUITY` + `CYCLE_START_EQUITY`)).
- The consistency-gate compiles to a **payout or pass** action (whichever event it gates), **not** a fail action — it is an eligibility conjunct, and no `FAIL_CONSISTENCY` is emitted (`ARCHITECTURE.md` §5, `MODEL_RISKS.md` §C8).
- Compiling a fail-rule yields a record with a fail action and the correct severity default (e.g. daily-loss soft, trailing-drawdown hard) and the right fail code.
- Compiling a pass-rule yields a pass action; a payout-rule yields a payout action; the consistency-raises-target rule yields an adjust action naming the target field.
- A rule carries its two timing fields; a rule for which timing is irrelevant defaults both to continuous — the consistency-gate is such a rule (its `check_timing` is inert; it is only read inside the payout/pass conjunction, `MODEL_RISKS.md` §C8).
- The registry check passes for every registered kind and raises for an unregistered kind.

**Freedom.** How rules store parameters, and how the compiled record is laid out, are unconstrained — only the *reported* requirements/action/severity/timing/params are contracted.

---

## Step 3 — Trade dataset and preprocessing

**Build.** The dataset contract of `ARCHITECTURE.md` §11: validate the required input columns, normalize to per-unit return, derive the per-trade holding-interval-clipped adverse-excursion low (`trade_low`), assign each trade a session-day index by the canonical session calendar, build the per-day side table, and derive the calendar cadence (`trading_days_per_week`). Observable contract: given raw rows, produce a dataset exposing per-trade returns, day indices, per-trade `trade_low`, the per-day index table, day count, and the cadence figure; reject malformed input.

**Must pass.**
- Input missing all of `return` and (`pnl` + `size`) is rejected; input with exactly one of them is accepted, and `pnl`+`size` normalizes to the same per-unit return that a direct `return` column would give.
- Trades are ordered by timestamp regardless of input order.
- Day assignment uses the session boundary, not midnight: a trade after the session reset belongs to the next trading day even if it is the same calendar date, and vice versa. Day indices are monotonic non-decreasing.
- `trade_low` reflects only excursion **within** the position's holding interval: a case where a spanned bar's low occurred before entry (or after exit) must **not** lower `trade_low` (`MODEL_RISKS.md` §D1). A case fully inside the holding interval does lower it.
- When no adverse-excursion input is present, `trade_low` falls back to the realized down-move of the trade (documented lower-fidelity behavior), not to zero or an error.
- The per-day table lets any day's trades be located; `trading_days_per_week` equals distinct trading days divided by the calendar-week span of the data.
- Multi-asset input: a day is the set of all assets' trades on that session day; an asset with no trades that day is legitimately absent, not filled. Day identity is fixed before any resampling (`MODEL_RISKS.md` §G1).

**Freedom.** Array layout, dtypes, and how the per-day table is represented are yours. Only the *retrievable* quantities and the validation behavior are contracted.

---

## Step 3b — Synthetic trade-stream generators (testing + future research)

Buildable any time after Step 3, off the core critical path. It produces the **same raw input rows** the real backtest produces, so synthetic data drives the Step 6–11 tests through the identical `preprocess()` pipeline.

**Build.** A common generator interface and three implementations (`IIDGenerator`, `RegimeSwitchingGenerator`, `StochasticVolGenerator`) that manufacture raw trade rows from statistical parameters. See `ARCHITECTURE.md` §11.7. Observable contract: each generator, given parameters + seed, emits a raw-row table in the **exact §11.1 schema** (`timestamp`, `return` or `pnl`+`size`, `mae`, `symbol`) that `preprocess()` accepts unchanged; each attaches provenance (type, parameters, seed, derived edge); generation is deterministic under a fixed seed.

**Definitions fixed here (must be used exactly):**
- Risk is the unit: each trade is `+RR` (win) or `−1` (loss), `RR > 0`. Free parameters are `win_rate` and `RR` **only**.
- Derived (reported, never input): `edge = win_rate*(RR+1) − 1`; `breakeven win_rate = 1/(RR+1)`.

**Must pass.**
- Every generator's output validates and preprocesses through Step 3 with no special-casing — a synthetic `TradeDataset` is indistinguishable in shape/behavior from a real one.
- Accepting an explicit `edge` parameter is **rejected** (or absent from the API): the only strategy inputs are `win_rate` and `RR`; edge is derived and reported. A generator whose realized long-run edge over a large sample does not match `win_rate*(RR+1) − 1` (within sampling error) fails.
- `IIDGenerator`: trade outcomes are independent; over a large sample the realized win frequency matches `win_rate`, and there is no systematic autocorrelation in outcomes.
- `RegimeSwitchingGenerator`: outcomes show **persistence** — measurable positive autocorrelation / longer losing (and winning) runs than i.i.d. at the same overall `win_rate`. The stationary mix of regime win-rates reproduces the target overall `win_rate`.
- `StochasticVolGenerator`: trade-magnitude shows volatility clustering (autocorrelated absolute returns) and fatter realized tails than the i.i.d. fixed-size case, while preserving the target `win_rate`.
- Synthesized `mae` is consistent with each trade's outcome (a winning trade's adverse excursion does not exceed its risk; the excursion is deep enough, per the intraday parameter, to exercise `check_timing=CONTINUOUS` rules) and survives Step 3's holding-interval clipping.
- A fixed seed reproduces an identical row table; provenance records type, parameters, seed, and derived edge.
- `trades_per_day` produces the intended session/day structure so `preprocess()` derives the expected `trading_days_per_week`.

**Freedom.** The statistical internals (how regimes or the vol process are implemented) are entirely yours; only the emitted schema, the derived-edge consistency, the per-type dependence properties, and determinism are contracted. **Not built now:** the breakeven-mapping sweep over parameter grids (`ARCHITECTURE.md` §11.7.4) — a research layer on top of this and the finished engine, specified later. **Interpretation guard (`MODEL_RISKS.md` §I1):** synthetic results are model-conditional and must be read across the full synthetic ladder, never from one generator.

---

## Step 4 — Firm config and the validator

**Build.** The three-layer config format (raw rule-object tables; the `scaled`/`build_accounts` sugar; and the safety net) plus the role-aware `validate`. See `ARCHITECTURE.md` §7, §9. Observable contract: a firm can be expressed as pure config and assembled into DSL objects; `validate` accepts sane-but-irregular accounts and rejects broken ones.

**Must pass.**
- A hand-written cell of arbitrary rule structure builds into the intended account.
- Two sizes with *different rule structure* coexist in one table (e.g. one has a daily-loss rule, the other does not).
- The `scaled`/builder sugar produces accounts value-identical to the equivalent hand-written cells.
- `validate` accepts a normal account and an irregular-but-sane account (unusual rule counts across phases).
- `validate` rejects: an empty phase; a negative rule parameter; a rule requiring state the kernel does not produce; an **eval** phase with no pass condition (unwinnable); and **more than one `TrailingDrawdownRule` in a single phase** (the kernel supports one trailing reference per phase, `ARCHITECTURE.md` §8 — this must fail loudly, not silently collide). A **funded** phase with only survival + payout rules and no pass condition is accepted (role-aware).
- A config naming a rule whose kind is unregistered fails at validation, not at simulation.

**Freedom.** The sugar is optional convenience; the raw-table floor must remain able to express anything. How the builder is written is unconstrained.

---

## Step 5 — Compiler

**Build.** The requirements resolver (union of the live `StateField`s per phase, with the "only compute what is required" property) and the struct-of-arrays emission (per-phase rule arrays: kind, params, action, severity, timing, adjust-field, fail-code; plus the per-payout reset set and the compiled payout schema). See `ARCHITECTURE.md` §6b, §8. Observable contract: compiling an account yields a numeric representation the kernel can consume, and the live-state set for a phase is exactly the union its rules require (plus the always-on driving fields).

**Must pass.**
- The resolved state set for a phase is the union of its rules' requirements plus the always-needed driving fields; a phase with no consistency rule does not include consistency-only state; a phase with no winning-days rule does not include the qualifying-day counter.
- Two rules requiring the same semantic `StateField` produce one entry for it (no double-allocation), and the union is order-independent.
- The emitted rule arrays have one aligned entry per rule, preserving rule order (so downstream precedence is well-defined).
- The payout schema compiles with its dollar-cap tuple, cap-fraction, min-request, buffer, split (including tiered split), max-payouts, and post-payout transition flags intact (`ARCHITECTURE.md` §6b).

**Freedom.** The compiled layout is entirely yours — this is a prime optimization surface. The only contract is that the resolved requirements are correct and the schema round-trips faithfully.

---

## Step 6 — Single-path kernel + reference oracle  ·  **LEVEL-1 GATE (§G6)**

**Build.** Two implementations of one-attempt simulation over a fixed trade path: a fast kernel and a slow, obviously-correct pure-Python reference (`reference.py`). Both consume a compiled account + a trade path + a **sizing policy-parameter array** (`policy_params`) and return the terminal `ExitCode`, the payouts taken (amount + day index each), and enough per-step state for the reference to be inspectable. Size is computed per trade as a function of the current stage mask and `policy_params` (`ARCHITECTURE.md` §16.1), not held constant in the loop; today `policy_params` is length-1 and reproduces a fixed size. See `ARCHITECTURE.md` §12, §16.1. Observable contract: on any given path *and* policy-parameter array, both produce identical terminal outcomes and payout sequences; the reference additionally exposes per-trade state for debugging.

**This step is a build-blocking gate.** No later step's numbers are trusted until it passes (`MODEL_RISKS.md` §G6, Level 1). The equivalence bar is **bitwise on the per-sim path**: each simulation is an independent sequential accumulation with no within-sim reduction (`prange` parallelizes across sims, not within one), so with `fastmath` off the reference and kernel must agree bit-for-bit — any non-bitwise difference is a real bug, not benign reassociation. Contract boundaries are exact by construction of that bitwise bar. Tolerance is reserved only for a genuine cross-lane reduction if one is ever introduced (`MODEL_RISKS.md` §G6/§C1).

**Must pass — behavior.**
- Hitting the profit target with all other pass-conditions satisfied returns PASSED; hitting it while a min-days pass-condition is unmet does **not** pass (the account keeps running).
- A hard breach terminates with the correct fail code. A soft breach truncates the day (remaining same-day trades skipped), the account survives, the partial-day loss stands, the day still counts as a trading day but not as a winning day, and simulation resumes next day.
- A payout fires only when its full qualifying conjunction holds; on firing it records amount + day index, applies the configured post-payout transition (counter resets; optional equity reduction; optional floor recompute), and continues. After max-payouts the attempt reaches its terminal state.
- Timing: with EOD-update/continuous-check drawdown, the floor advances off the day's closing equity while a breach is detected intraday against the day's low-water mark. With continuous/continuous it updates and breaches live. The four update×check combinations behave per `ARCHITECTURE.md` §6a.
- *(Deferred — `MODEL_RISKS.md` §C2/§A2):* the capped-out outcome is **not** tested here. It is only well-defined once a minimum-position-size config and a dynamic sizing policy exist; under a constant policy an account never caps out, so `CAPPED_OUT` is inert and has no boundary to test. Re-add its test when the sizing policy is built.
- An adjust rule mutates its target (e.g. raises the profit target) without failing or passing, and a later pass-predicate reads the mutated value. An **EOD-timed** adjust (the consistency-raises-target default) fires at *day close*, not intraday: a case where the consistency condition is violated mid-day but the target only changes at the day boundary — kernel and reference agree on the timing (`ARCHITECTURE.md` §12).
- **Sizing hook:** a length-1 `policy_params` reproduces the constant-size result exactly (parity with the fixed-size case — note this is a *uniform-size* baseline, one scalar across all instruments, **not** a replay of the historical backtest's variable per-instrument sizing; `ARCHITECTURE.md` §16.1). A `policy_params` that assigns different sizes to different stage-mask values produces the correspondingly different equity path — and the kernel and reference agree on it. (The policy's *functional form* is not fixed here; only that size is read per trade from `policy_params` + stage and that both implementations agree.)

**Must pass — boundary cases (exact; these are the §G6 list).**
- Equity exactly equal to the drawdown floor; one tick above; one tick below — each resolves to the correct side of the breach.
- Profit target reached exactly (equity exactly at target).
- A breach and a target satisfied on the **same trade** — resolves by the documented precedence (fail wins), deterministically.
- A breach on the first trade; a breach on the final trade of a day.
- A soft breach on a day that also carries an EOD-timed rule: the truncated day's **closing equity is the equity after its last executed trade** (the breach trade), the EOD rule evaluates against that true closing equity (not the intraday low, and not a special breach value), and both implementations agree (`MODEL_RISKS.md` §C5). Breach *detection* used the intraday low-water mark; the EOD *close* uses closing equity — the two are distinct and both are tested.
- A payout requested at exactly `min_request`; at exactly a dollar cap; at exactly the `cap_fraction` bound.
- **A payout that qualifies on winning-days but whose cycle profit is below `min_request`, or whose release would breach `buffer_floor`, does NOT fire** — no `$0` payout is recorded, no `max_payouts` slot is consumed, the winning-day counter is not reset, and the account keeps accumulating (the fire/amount boundary of §6b). This directly protects the payout-count distribution.
- **An EOD-timed breach (or an EOD-completed payout) on an *intermediate* day terminates the attempt at that day's close, not only on the last day** — the `_close_day` terminal return is honored at every rollover (§12). A pure end-of-day (EOD×EOD) drawdown that would breach at the close of day 3 of a 10-day path ends the attempt on day 3.
- **On a single day-close where an EOD breach and an EOD-completed 5th qualifying day coincide, the breach wins** (`_close_day` uses the same fail→adjust→pass→payout precedence as the trade loop, §C4/§12).
- **The closing day's own pnl is included in its own EOD predicate** (`_close_day` folds the day into counters *before* evaluating EOD rules, `MODEL_RISKS.md` §C9): a day that completes the 5th qualifying win pays out on that same close; a day whose own pnl tips the consistency ratio above the gate blocks the payout on that same close (the account continues, it does not fail).
- **Consistency is an eligibility gate, not a failure** (`MODEL_RISKS.md` §C8): a funded payout does **not** fire while `max_day_pnl > threshold × cycle_profit` (where `cycle_profit = equity − cycle_start_equity` — profit since the last payout reset, *not* lifetime `total_pnl`) — the payout is withheld and the account keeps trading (never terminated); once other days bring the ratio to/below threshold, the payout can fire. On eval, the same ratio gates the PASS (there `cycle_profit` equals phase profit, no payout having occurred). The gate is only evaluated inside a payout/pass conjunction (which already requires meaningful positive profit), so it never divides by zero/negative profit and never fires spuriously. A `ConsistencyGate` never produces a `FAIL_CONSISTENCY`.
- **Reaching `max_payouts` returns `MAXED_OUT`, not `PASSED`** — a distinct funded-success terminal code (§6b.2).
- A withdrawal followed by drawdown-floor recomputation (post-payout transition).
- A payout immediately followed by a breach.
- The trailing-floor lock transition (floor reaches `lock_at`, stops trailing thereafter).
- A `trade_low` whose adverse excursion lies outside the holding interval does not cause a breach it shouldn't (ties Step 3's clipping into the kernel).
- *(Deferred — not in this step's exact-boundary set:* the minimum-size / capped-out boundary, which is undefined without minimum-position-size config and a dynamic sizing policy; `MODEL_RISKS.md` §C2/§A2.)

**Freedom.** The kernel's internals — state layout, loop structure, vectorization, compilation — are completely open, *provided* it matches the reference on every case above. The reference is the fixed oracle; optimize the kernel against it without limit.

---

## Step 7 — Fingerprint and caches

**Build.** A structural fingerprint of a compiled account (stable, value-derived, version string inside the hash) and the cache layer keyed on it (compiled accounts, compiled rules, preprocessed trades). See `ARCHITECTURE.md` §10. Observable contract: identical configs produce identical fingerprints; any value change (including a version-string change or a per-size quirk) changes the fingerprint; a cache lookup by fingerprint returns the previously compiled artifact.

**Must pass.**
- The same account fingerprints identically across repeated construction.
- Changing any rule parameter, a severity, a timing field, the program version string, **an `eval_fee` or `activation_fee` (the fee is the entire downside, §0), or any `PayoutSchema` field** changes the fingerprint. Two accounts differing only in fee, or only in split/cap/buffer, must not share a cache key.
- Two structurally identical accounts share a fingerprint (and therefore a cache entry); a size-specific quirk (e.g. a differing min-days on one size) yields its own fingerprint.
- A trade-preprocessing cache keyed on the raw input + session parameter returns the same dataset without reprocessing.

**Freedom.** Hash function, key format, and cache backend are yours. The contract is stability + sensitivity + correct hit/miss behavior.

---

## Step 8 — Resampling

**Build.** Index/path generators over the dataset: at minimum an i.i.d. day bootstrap and a stationary (geometric-block) day bootstrap, operating at **whole-day** granularity and, for multi-asset data, resampling days **jointly across all assets**. Each generated path has an explicit **path length `L`** (number of resampled days per attempt), a first-class modeling parameter that must be set deliberately, not left implicit (`MODEL_RISKS.md` §C7). See `ARCHITECTURE.md` §11.4 and `MODEL_RISKS.md` §G1, §C7. Observable contract: given a dataset, a length `L`, and parameters, produce resampled paths that (a) are exactly `L` days long, (b) only ever reference real days, (c) keep each day's trades intact and in order, (d) keep all assets of a day together, and (e) are deterministic under a fixed seed.

**Must pass.**
- A generated path is exactly `L` days long; `L` is an explicit input, not a value inferred from the source length. Two runs at different `L` are both valid and differ only in path length. **Eval and funded phases may take different `L`** (they draw independent paths in the engine loop, `ARCHITECTURE.md` §17); the generator accepts `L` per call, so `L_eval` and `L_funded` can differ.
- Every day in a resampled path is a real day from the source; no day is split across the boundary and no trade is orphaned from its day.
- The i.i.d. generator produces essentially no systematic runs of consecutive source-days; the stationary generator with a long mean block produces many consecutive-day runs (block structure is present and scales with the block-length parameter). The i.i.d. case is recoverable as the short-block limit of the stationary generator.
- A fixed seed reproduces an identical path; different seeds differ.
- Multi-asset: a resampled day carries *all* assets' trades for that source day together; assets are never drawn independently (a resampled day cannot mix asset-A's trades from one source day with asset-B's from another).
- Day identity used for resampling is the canonical session day fixed in Step 3, so the generator cannot manufacture cross-asset alignment that did not exist.

**Freedom.** Generation algorithm, index representation, and how paths are materialized are yours. The contract is fixed-`L` length, day-integrity, joint-asset-integrity, block behavior, and determinism. **Note (`MODEL_RISKS.md` §C7):** `L` shapes both the payout-count distribution (Step 10) and the renewal cycle-time (Step 11), so it must be reported alongside any result and its sensitivity examined — it is not a free implementation choice.

---

## Step 9 — Batch Monte Carlo and the engine

**Build.** The batch driver (many resampled attempts through the kernel, in memory-bounded batches, retaining rich per-attempt raw outcomes) and the `Engine.run` orchestration (validate → preprocess/cache → fingerprint → compile/cache → per-phase survivors-only simulation → aggregate raw outcomes → return a results object). `Engine.run` accepts a `policy_params` argument and threads it to the kernel, defaulting to the length-1 constant-size case (`ARCHITECTURE.md` §16.1). See `ARCHITECTURE.md` §12, §17. Observable contract: `Engine.run(account, dataset, config, policy_params=...)` returns per-attempt raw outcomes (exit code, payouts with day indices, total trading days, size) for the whole batch, agrees with the single-path kernel on shared paths *and the same policy_params*, and reproduces the fixed-size behavior when `policy_params` is the default.

**Level-0 precondition (`MODEL_RISKS.md` §G7).** The dataset handed to `Engine.run` for any result you intend to *trust* must come from a frozen strategy on a held-out period — not the sample the strategy was selected on. This is a discipline on inputs, not a component; tests here use synthetic/fixed data and need not enforce it, but the engine should carry the dataset's provenance through to the results so it can be reported.

**Must pass.**
- On a batch of identical straight-through paths, every attempt's outcome equals the single-path kernel's outcome for that path (batch ⇄ oracle agreement).
- Multi-phase: the funded phase is simulated only for attempts that passed the eval phase (survivors-only); an attempt that fails eval never produces funded-phase outcomes.
- Batching is transparent: the same seed and config produce the same aggregate outcomes regardless of batch size (batch size is a memory knob, not a result-changing parameter).
- The raw outcomes retain the time axis (payout day indices, total trading days) and per-attempt fields needed downstream — nothing is pre-aggregated away.
- Running the same account twice hits the compiled-account and trade caches (no recompute), producing identical results.
- The default `policy_params` reproduces fixed-size outcomes; a non-trivial `policy_params` threads unchanged to the kernel and produces the same result the single-path kernel gives for that policy on each shared path (the hook is a passthrough, adding no engine-level behavior of its own).

**Freedom.** Batch size, parallelism, memory strategy, and aggregation internals are entirely yours, constrained only by determinism-under-seed and batch ⇄ oracle agreement.

---

## Step 10 — Single-attempt decision statistics

**Build.** The decision-statistics layer over raw outcomes: the **distribution axis** (probability an attempt is profitable, payout-count distribution, payoff quantiles, return-on-fee) and the **time axis** (payout velocity, time-to-first-payout, return-on-fee per unit calendar time), with calendar time derived from cadence, never from a dataset fraction. See `ARCHITECTURE.md` §14. Observable contract: given a batch of raw outcomes, a fee, and the dataset cadence, produce each statistic; time-based statistics are in calendar units.

**Definitions fixed here (must be used exactly):**
- **Profitable attempt** = cumulative net payouts of the attempt − all fees attributable to that attempt (evaluation + activation) > 0.
- Calendar time of an attempt = its trading-day count ÷ `trading_days_per_week`.

**Must pass.**
- `P(profitable)` counts attempts whose net payouts exceed attributable fees, by the definition above — and is demonstrably *not* recoverable from the mean payout alone (two batches with equal mean payout but different profitable-fraction give different `P(profitable)`).
- The payout-count distribution sums to 1 and matches the raw per-attempt payout counts (P(0), P(1), …, P(max)).
- Return-on-fee and its quantiles are computed per attempt then summarized; the full distribution is available, not just the mean.
- Payout velocity and return-on-fee-per-year use calendar time from cadence; doubling `trading_days_per_week` (same trading-day counts) halves the implied calendar duration and correspondingly changes the rate.
- A "fraction of the source dataset" is **never** used as a duration; a path longer than the source history still yields a well-defined calendar duration.
- The mean payout remains available but is one statistic among these, not the reported summary.
- `pass_rate` (fraction with `code == PASSED`) is treated as an **eval-phase** metric only: a funded attempt that succeeds returns `TIMED_OUT` or `MAXED_OUT`, never `PASSED`, so funded economic performance is read from `net_payout`/`payouts_taken`/the payout-count distribution — a funded batch's success is never measured by `pass_rate` (`MODEL_RISKS.md` §H3).

**Freedom.** How statistics are computed and summarized is yours; the contracts are the fixed definitions and the calendar-time (not dataset-fraction) rule.

---

## Step 11 — Renewal economics

**Build.** The renewal layer above `Engine.run`, consuming completed attempt outcomes and composing them into a repeated-attempt (renewal-reward) process: the closed-form reward rate `R_renewal = E[R_cycle]/E[T_cycle]`, the empirical long-run rate `R_path` from simulated attempt-sequences, fee-bankroll efficiency, and the finite-horizon cumulative-cashflow distribution. See `ARCHITECTURE.md` §15. Observable contract: given completed attempt outcomes (reward and calendar time per attempt), produce both rate definitions, the bankroll-efficiency figure, and the horizon cashflow distribution — all in Python above the engine, touching no kernel state.

**Definitions fixed here (must be used exactly):**
- **Profitable renewal sequence** = cumulative net payouts across the sequence − cumulative fees across all attempts in the sequence > 0 (distinct from the single-attempt definition in Step 10).
- Per-attempt reward = net payouts − attributable fees; per-attempt time = attempt calendar duration.

**Must pass.**
- `R_renewal` equals the ratio of mean per-attempt reward to mean per-attempt time.
- `R_path` estimated by simulating long attempt-sequences converges toward `R_renewal` when attempts are drawn i.i.d. with light-tailed cycle times (the clean ergodic case), confirming the two definitions agree under their stated assumption.
- The two rates are reported *separately*, and their divergence is surfaced as a diagnostic (the layer must not silently report only the closed form). The divergence `r_path` can show is **ratio-estimator / finite-horizon (Jensen) bias**: a constructed heavy-tailed / skewed cycle-time set makes `R_path` depart from `R_renewal`; an i.i.d. light-tailed set does not. **`r_path` cannot show cross-cycle correlation** — i.i.d. draws remove it — so the test must *not* claim a "correlated set shows divergence" (it won't, because the resampler destroys the correlation first; `MODEL_RISKS.md` §H5). Diagnosing correlation is deferred to order-preserving sequence simulation.
- Attempt cycle-time `T_i` is the **whole attempt** (eval + funded) calendar duration, not funded-only (`MODEL_RISKS.md` §H4): a two-phase attempt's `T_i` includes its eval days. A test with a long eval and short funded phase must bill the eval time into the rate.
- Fee-bankroll efficiency scales as expected: doubling the fee (all else equal) halves income-per-unit-bankroll; the figure is expressed per unit calendar time per unit bankroll.
- Finite-horizon cumulative cashflow returns a *distribution* over sequences (not just a mean), so convexity is visible at the renewal level as it is at the attempt level.
- The layer consumes only completed attempt outcomes; it introduces no retry/bankroll state into the kernel or compiled account (sequential-renewal-in-scope, portfolio-deferred boundary, `MODEL_RISKS.md` §0).

**Freedom.** Sequence-simulation method, horizon handling, and summary form are yours; the contracts are the fixed definitions, the report-both-rates rule, and the analysis-layer boundary (no kernel involvement).

---

## Steps 12–14 — deferred layers (spec deliberately light)

These are past the single-account engine + renewal analysis you're building now. Their contracts are stated at low resolution on purpose; pin them down when you reach them.

- **Step 12 — additional rules and firms.** Adding a firm should be config + existing rules; a genuinely new mechanic is a new rule kind that the registry (Step 2) forces you to implement and re-run Step 6's oracle parity for. *Must pass:* each new firm's accounts pass `validate`; each new rule kind gets its own Step-6-style boundary tests before use.
- **Step 13 — generator ladder.** Evaluate the same frozen strategy/contract/objective across i.i.d. → block → stationary → regime-conditioned → stochastic-volatility generators; report every headline number as a **model-sensitivity band** across generators and block-length range (`MODEL_RISKS.md` §G1). *Must pass:* a result is only reported with its band across the ladder, never as a single-generator point.
- **Step 14 — optimizer.** Out of scope for this document. When specified, its binding precondition is **nested / walk-forward out-of-sample** so it never optimizes on the strategy-selection sample (`MODEL_RISKS.md` §G7), and its objective is a renewal-and-survival objective (Step 11), not single-attempt expected payout.

---

## The one invariant across every step

A step is "done" only when its **Must pass** cases hold, and a later step may assume only the **observable contract** of earlier steps — never their internals. That is what lets you improve any implementation freely: as long as the contract still holds and the tests still pass, the change is safe by construction. The two hard gates — Step 6 oracle parity (Level 1) and the Step 9–11 frozen-strategy input discipline (Level 0) — are the only places where "passing the tests" is not sufficient on its own; there, trust in the *inputs and the oracle* precedes trust in any number the engine produces.
