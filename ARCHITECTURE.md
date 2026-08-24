# propfirm_engine — Architecture

## Design invariants

- **Phases are independent gates.** Passing the eval phase unlocks the funded phase; nothing carries over (no balance, no drawdown reference, no high-water mark). Each phase simulates fresh. The only thing a phase transition carries is a boolean "prior phase passed."
- **The kernel is a predicate/action evaluator.** Every rule is a predicate over per-simulation state paired with an action that fires when the predicate triggers. Failures, multi-condition passes, payouts, soft/hard breaches, target adjustments, and stages are all instances of this one mechanism (§6). Nothing about pass/fail is hardcoded in the loop.
- **Timing is carried by the rule, not the loop.** *When* a reference point advances and *when* a predicate is checked are two independent per-rule fields (§6a), so a firm with EOD-trailing-but-intraday-detected drawdown is config, not a kernel fork. Intraday detection reads a per-day low-water mark; the engine does not simulate a tick path within a trade.
- **Rules are independent within a phase.** Predicates read shared kernel state but never depend on each other; sequencing is expressed through tracked counters, never through predicate-to-predicate coupling.
- **Failure is an integer code, not a boolean.** `0` = alive, nonzero = exit reason. Plain pass/fail reads `!= 0`; per-rule failure attribution assigns each rule its own code. The two modes share one integer field, so attribution can be added without a type change anywhere downstream.
- **The variant level is always present.** A program with a single account type carries one implicit `default` variant, so every consumer sees one tree shape.
- **The fingerprint is one authoritative value:** a structural hash of the compiled config. The human-readable version string is metadata *inside* what gets hashed, not a second source of truth.
- **Raw per-simulation outcomes are retained in full, including the time axis.** Batching is a memory device, not a statistics device. The account is a convex structured product, so the engine reports the payoff *distribution* and its *time-normalization* (§14), not just a mean — `P(profitable)`, payout-count histogram, payout velocity, return-on-fee per year — all as post-processing over the raw outcomes, never something the kernel must be re-entered to compute. Calendar time is derived from trade cadence (§11.5), never from a fraction of the source dataset.

---

## 1. The two-system model

```
   PROP-FIRM DSL              COMPILE            NUMERICAL ENGINE           STATISTICS
   (expressive,      ───►   (once, cached)  ───►  (NumPy + Numba,    ───►  (separate,
    object-oriented)                               zero Python objs)        swappable)
```

The DSL is expressive and Pythonic. The hot loop never sees a Python object. Everything expensive is reduced to primitive arrays and integer codes. Rules describe *what* must hold; the kernel, keyed by integer rule type, decides *how* it is checked (§6, §12).

---

## 2. File structure

The project stays flat; a file is split into a package only when it grows unwieldy.

```
propfirm_engine/
├── pyproject.toml
├── README.md
├── src/propfirm_engine/
│   ├── __init__.py
│   │
│   ├── model.py            # Firm, Program, Variant, Account, Phase — frozen dataclasses
│   ├── rules.py            # Rule base + all rule dataclasses + RULE_REGISTRY
│   ├── enums.py            # ExitCode, StateField, Action, Severity, Stage (IntEnum)
│   │
│   ├── config.py           # firm config: rule-object tables + `scaled` helper + build_accounts
│   ├── validate.py         # sanity checks on assembled accounts (permissive, not a schema)
│   │
│   ├── compiler.py         # DSL objects -> CompiledAccount (arrays + codes), + requirements
│   ├── fingerprint.py      # structural hash of a compiled account
│   │
│   ├── data.py             # TradeDataset + preprocessing (raw -> simulation-ready arrays)
│   ├── synthetic.py        # synthetic raw-row generators (iid / regime / stoch-vol) — §11.7; test + research
│   │
│   ├── kernels.py          # @njit hot paths: single-path + batch. NO Python objects here.
│   ├── reference.py        # slow pure-Python step() oracle for validating the kernel
│   ├── resampling.py       # index generators: iid bootstrap, stationary bootstrap
│   ├── simulate.py         # orchestration: batches -> kernels -> raw outcome arrays
│   │
│   ├── statistics.py       # metrics + confidence intervals over raw outcomes
│   ├── objectives.py       # single-attempt scalar objectives (income, ROI, P(profitable)) — §14
│   ├── renewal.py          # renewal-economics analysis layer over completed attempts — §15
│   ├── results.py          # Results object wrapping raw outcomes + lazy stats
│   │
│   ├── engine.py           # Engine.run(): the orchestrator tying it all together
│   ├── registry.py         # FIRMS registry + convenience access
│   ├── cache.py            # 3 caches: preprocessed trades, compiled accounts, compiled rules
│   │
│   └── firms/              # one file per firm — PURE config, no logic
│       ├── __init__.py     # assembles FIRMS registry
│       ├── lucid.py        # LucidFlexDLL, LucidFlexNoDLL, LucidDirect, LucidPro, LucidDaily
│       ├── apex.py
│       └── topstep.py
│
└── tests/
    ├── test_model.py
    ├── test_rules.py
    ├── test_config.py
    ├── test_validate.py
    ├── test_compiler.py
    ├── test_fingerprint.py
    ├── test_data.py
    ├── test_kernels.py
    ├── test_reference.py
    ├── test_resampling.py
    ├── test_statistics.py
    └── test_engine.py
```

The `firms/` folder is pure configuration — every firm is data, no firm has its own code path. Adding a firm never touches the engine. When `rules.py` exceeds ~15 rules, it becomes a package.

---

## 3. The integer vocabulary (`enums.py`)

This is the contract between the DSL and the kernel, and is defined first. It has four parts: exit codes (how a simulation ends), state fields (what the kernel tracks), and the action/severity vocabulary (what a rule does when it triggers), plus the stage bitmask.

```python
from enum import IntEnum

class ExitCode(IntEnum):
    ALIVE          = 0     # still running
    PASSED         = 1     # cleared the phase (all pass-predicates satisfied)
    # --- failure codes start at 10; each rule owns one for attribution mode ---
    FAIL_GENERIC   = 10    # pass/fail mode uses only this for any failure
    FAIL_TRAILING_DD = 11
    FAIL_STATIC_DD   = 12
    FAIL_DAILY_LOSS  = 13
    FAIL_MIN_DAYS    = 14   # RESERVED — currently unreachable. MinimumTradingDaysRule is a PASS gate
                            #   (it withholds PASSED until enough days), so a shortfall surfaces as
                            #   TIMED_OUT, not this code. Kept for firms that actively FAIL on min-days.
    FAIL_CONSISTENCY = 15   # RESERVED / unused — consistency is an eligibility GATE, not a failure
                            #   (real firms withhold payout/pass, never terminate). Kept only for a
                            #   hypothetical future firm that genuinely FAILS on consistency. (§5, C8)
    TIMED_OUT      = 20    # ran out of trades without passing
    CAPPED_OUT     = 21    # DEFERRED with the sizing policy (MODEL_RISKS C2/A2): only well-defined once a
                           # minimum-position-size config exists. Reserved but never emitted under a constant
                           # size policy; not on the Step 6 boundary-test list until sizing is built.
    MAXED_OUT      = 22    # funded: reached max_payouts — an economic SUCCESS, distinct from eval PASSED.
                           # Separate code so filtering PASSED on a funded batch can't silently conflate
                           # "cleared eval" with "hit the payout ceiling" (MODEL_RISKS H3). Funded economic
                           # stats read net_payout/payouts_taken; PASSED/MAXED_OUT are eval/funded terminals.

class StateField(IntEnum):
    """Index into the per-sim state vector. The requirements resolver decides which are live."""
    EQUITY            = 0   # equity at trade close (realized)
    PEAK_EQUITY       = 1   # running max of closing equity (reference for trailing DD)
    DD_FLOOR          = 2   # the trailing-DD line itself (peak - amount, or locked value)
    DD_LOCKED         = 3   # 0/1: has the floor locked and stopped trailing?
    DAY_LOW           = 14  # worst floating equity reached so far this day (for intraday checks)
    DAY_PNL           = 4
    TOTAL_PNL         = 5
    DAY_INDEX         = 6
    N_TRADING_DAYS    = 7
    MAX_DAY_PNL       = 8    # for consistency
    N_QUALIFYING_DAYS = 9    # winning days meeting a profit threshold (e.g. 5 days >= $150)
    PAYOUTS_TAKEN     = 10   # increments on each payout; lets "post-payout" predicates stay pure
    N_SOFT_BREACHES   = 11   # for soft-breach escalation ("N soft breaches => hard breach")
    STAGE_MASK        = 12   # bitmask of active stage predicates (see Stage)
    PROFIT_TARGET     = 13   # live, mutable target (seeded from rule p0; raised by Action.ADJUST)
    CYCLE_START_EQUITY = 15  # equity at the start of the current payout cycle; cycle_profit = equity - this.
                             #   reset to current equity on each payout, so cap_fraction*cycle_profit is well-defined
    CUMULATIVE_PAID   = 16   # total gross dollars paid out so far; drives the tiered legacy split (split_first_tier)

class Action(IntEnum):
    """What a predicate does when it triggers."""
    FAIL   = 0    # end the simulation in failure (see Severity for hard vs soft)
    PASS   = 1    # terminal success — the phase is cleared
    PAYOUT = 2    # repeatable success — record a payout, reset configured counters, continue
    ADJUST = 3    # mutate a target StateField (e.g. raise PROFIT_TARGET) instead of failing/passing

class Severity(IntEnum):
    """Only meaningful for Action.FAIL predicates."""
    HARD = 0    # account terminated
    SOFT = 1    # current day truncated; remaining trades skipped; resume next day

class Timing(IntEnum):
    """Two independent per-rule axes (see §6a): when the reference point advances, and
    when the predicate is checked. Orthogonal — 'trails EOD, detected intraday on floating
    balance' is update_timing=EOD, check_timing=CONTINUOUS."""
    CONTINUOUS = 0    # every trade; checks read the day's intraday low-water mark
    EOD        = 1    # only at day close; updates read the day's closing equity

class Stage(IntEnum):
    """Independent bit positions in STATE.STAGE_MASK. Multiple may be set at once."""
    IN_PROFIT             = 0
    CONSISTENCY_SATISFIED = 1
    PRE_FIRST_PAYOUT      = 2
    PAYOUT_ELIGIBLE       = 3
    # add bits freely; each is an independent predicate, never a phase of a state machine
```

A failure is any `code >= 10`. Pass/fail mode writes only `FAIL_GENERIC`; attribution mode writes the specific code. Every array downstream is `int8`/`int16`; nothing widens when attribution is enabled. Precedence — which rule wins when two trigger on the same trade — is decided solely by the order the kernel checks rules, a local decision inside one function.

`STAGE_MASK` is reserved now even though nothing sets bits yet; the cost of the reservation is one integer field, and it is what lets stages be added later without a kernel restructure (§6). `PROFIT_TARGET` is a *live* state field rather than a static rule constant so that `Action.ADJUST` rules (a consistency rule that raises the target instead of failing) can mutate it while predicates stay pure — the same read-state / write-via-action discipline as the counters.

---

## 4. DSL model (`model.py`)

All classes are frozen and hashable by value. No dicts or lists appear anywhere reachable from an `Account`; tuples are used throughout so the structural fingerprint is well-defined.

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Phase:
    name: str
    role: str                     # "eval" | "funded" — drives role-aware validation & payouts
    rules: tuple["Rule", ...]
    payout_schema: "PayoutSchema | None" = None   # funded phases carry one; eval phases leave it None (§6b)

@dataclass(frozen=True)
class Account:
    name: str
    size: int
    phases: tuple[Phase, ...]     # (eval, funded) or just (funded,)
    eval_fee: float = 0.0         # cost to start an attempt — the entire downside (§0). Part of identity.
    activation_fee: float = 0.0   # additional cost on reaching funded; some account types have it, some don't
    currency: str = "USD"

@dataclass(frozen=True)
class Variant:
    name: str
    accounts: tuple[Account, ...]           # tuple, not dict (hashability)
    def account(self, name: str) -> Account:
        for a in self.accounts:
            if a.name == name: return a
        raise KeyError(name)

@dataclass(frozen=True)
class Program:
    name: str
    variants: tuple[Variant, ...]
    version: str = "v1"           # effective-date/version string, part of the hash
    def variant(self, name: str = "default") -> Variant:
        for v in self.variants:
            if v.name == name: return v
        raise KeyError(name)

@dataclass(frozen=True)
class Firm:
    name: str
    programs: tuple[Program, ...]
    def program(self, name: str) -> Program:
        for p in self.programs:
            if p.name == name: return p
        raise KeyError(name)
```

A program with a single account type gets `Variant("default", accounts=(...))`. Every consumer traverses `firm.program(x).variant(y).account(z)` — one shape, with no `if has_variants` branching anywhere.

An account with two phases holds `phases=(eval, funded)`; a direct-funded account holds `phases=(funded,)`. No special-case simulator code distinguishes them. `Phase.role` distinguishes an eval phase (a `PASS` predicate ends it) from a funded phase (which may issue repeatable `PAYOUT`s and need not be terminable at all); the validator and payout logic key on this role rather than on rule contents.

---

## 5. Rules (`rules.py`)

A rule is a predicate plus an action. The base declares three things: what state the rule needs (`requirements`), how it compiles to a numeric record (`compile`), and — carried in that record — what the predicate *does* when it triggers. There is no `process_trade` method in Python; the predicate arithmetic and the action live once, in the kernel, keyed by rule type (§12).

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from .enums import StateField, ExitCode, Action, Severity

@dataclass(frozen=True)
class Rule(ABC):
    @abstractmethod
    def requirements(self) -> tuple[StateField, ...]: ...
    @abstractmethod
    def compile(self) -> "CompiledRule": ...
```

Every concrete rule is a direct child of `Rule` — one level deep, no intermediate abstract classes (a `DrawdownRule` parent buys nothing, since trailing and static drawdown share no code and compile to different kinds). The base defines the *interface*, never shared behavior; rule behavior lives in the kernel, not the class hierarchy. `@abstractmethod` makes a half-finished rule fail at instantiation rather than at compile time.

### Fail-predicates (disjunctive)

Any one triggering ends the phase; `Severity` decides how. A hard breach terminates the account; a soft breach truncates the current day and resumes on the next.

```python
@dataclass(frozen=True)
class TrailingDrawdownRule(Rule):
    amount: float
    update_timing: Timing = Timing.CONTINUOUS   # when the floor trails (EOD => off closing equity)
    check_timing:  Timing = Timing.CONTINUOUS   # when the breach is tested (CONTINUOUS => intraday low)
    lock_at: float | None = None                # floor stops trailing once it reaches this level
    def requirements(self):
        need = [StateField.EQUITY, StateField.PEAK_EQUITY, StateField.DD_FLOOR]
        if self.check_timing == Timing.CONTINUOUS: need.append(StateField.DAY_LOW)  # intraday low
        if self.lock_at is not None:              need.append(StateField.DD_LOCKED)
        return tuple(need)
    def compile(self): return CompiledRule(
        kind=RULE_TRAILING_DD, p0=self.amount,
        p1=(self.lock_at if self.lock_at is not None else INF),
        update_timing=self.update_timing, check_timing=self.check_timing,
        action=Action.FAIL, severity=Severity.HARD, fail_code=ExitCode.FAIL_TRAILING_DD)

@dataclass(frozen=True)
class DailyLossRule(Rule):
    amount: float
    severity: Severity = Severity.SOFT      # daily-loss is soft at most firms; overridable per account
    def requirements(self): return (StateField.DAY_PNL,)
    def compile(self): return CompiledRule(kind=RULE_DAILY_LOSS, p0=self.amount,
                                           action=Action.FAIL, severity=self.severity,
                                           fail_code=ExitCode.FAIL_DAILY_LOSS)
```

### Pass-predicates (conjunctive)

The phase is cleared only when *all* pass-predicates hold simultaneously. The profit target is one pass-predicate among possibly several — never a hardcoded special case — so "eval pass = profit target AND ≥3 trading days AND consistency" is pure config.

```python
@dataclass(frozen=True)
class ProfitTargetRule(Rule):
    target: float
    def requirements(self): return (StateField.EQUITY,)
    def compile(self): return CompiledRule(kind=RULE_PROFIT_TARGET, p0=self.target,
                                           action=Action.PASS)

@dataclass(frozen=True)
class MinimumTradingDaysRule(Rule):
    minimum_days: int
    def requirements(self): return (StateField.N_TRADING_DAYS,)
    def compile(self): return CompiledRule(kind=RULE_MIN_DAYS, p0=self.minimum_days,
                                           action=Action.PASS)   # a gate on passing, not a failure
```

`MinimumTradingDaysRule` shows why pass must be conjunctive: an account that hits the profit target on day 1 with `minimum_days=3` must keep running. Reaching the target satisfies one pass-predicate; the phase clears only when the min-days predicate is also satisfied.

### Payout-predicates and counters (conjunctive, repeatable)

A funded payout fires when its conjunction holds, then *resets configured counters and continues* — the repeatable analogue of a pass. Counters that feed a payout (qualifying-day counts, consistency windows) are ordinary `StateField`s; the predicate reads them, and the payout's fire-action resets them.

```python
@dataclass(frozen=True)
class MinimumWinningDaysRule(Rule):
    count: int          # e.g. 5 winning days
    threshold: float    # each day must clear this profit (e.g. 150.0)
    def requirements(self): return (StateField.N_QUALIFYING_DAYS,)
    def compile(self): return CompiledRule(kind=RULE_MIN_WINNING_DAYS,
                                           p0=self.count, p1=self.threshold,
                                           action=Action.PAYOUT)

@dataclass(frozen=True)
class ConsistencyGate(Rule):
    """Consistency as an ELIGIBILITY GATE, not a failure. Real firms (Lucid Pro/Flex,
    Topstep, others) never *fail* an account for consistency — they *withhold a positive
    event* until the ratio is satisfied:
      - funded: `largest_day_profit / account_profit <= threshold` is required to REQUEST A
        PAYOUT; if violated the payout simply doesn't fire and the trader keeps earning on
        other days until the ratio falls (it resets after each payout).
      - eval: the same ratio is required to UPGRADE (PASS) to funded.
    So consistency is a CONJUNCT in the PAYOUT predicate (funded) or the PASS predicate (eval),
    with `action` selecting which. It is NOT a FAIL rule (see MODEL_RISKS §C8 for why the
    earlier fail-model was wrong and why this framing removes the degeneracy entirely).

    The predicate `max_day_pnl <= threshold * cycle_profit` (cycle_profit = equity -
    cycle_start_equity) is only ever evaluated as part of a payout/pass conjunction, which
    already requires meaningful positive profit (min_request, profit target) — so the
    'profit <= 0' and 'fires on trade 1' degeneracies cannot arise. The denominator is CYCLE
    profit, not lifetime TOTAL_PNL, because the funded gate RESETS AFTER EACH PAYOUT: "account
    profit" means profit earned THIS cycle, which also makes the gate agree with the §6b
    payout-amount arithmetic (cap_fraction * cycle_profit) it sits beside. On eval this is
    identical to phase profit, since the cycle spans the whole phase when no payout has occurred.
    No activation gate and no new mechanic: it reads MAX_DAY_PNL and cycle profit (EQUITY -
    CYCLE_START_EQUITY), all already tracked. `check_timing` is INERT for this rule — the gate is
    only ever read inside the payout/pass conjunction (evaluated fold-then-evaluate at day close,
    §C9), so EOD vs CONTINUOUS are behaviorally identical and nothing may branch on it."""

    threshold: float                         # e.g. 0.40 (Pro funded), 0.50 (Flex eval)
    action: Action = Action.PAYOUT           # PAYOUT on funded, PASS on eval — which event it gates
    cushion: float = 0.0                     # Flex's small allowance: effective bound is slightly looser
                                             #   than nominal, computed on actual daily profit (per-firm).
    def requirements(self): return (StateField.MAX_DAY_PNL, StateField.EQUITY,
                                    StateField.CYCLE_START_EQUITY)
    def compile(self): return CompiledRule(kind=RULE_CONSISTENCY_GATE, p0=self.threshold,
                                           p1=self.cushion, action=self.action)
```

Consistency thus reuses the payout fire-gate machinery (§6b): a payout for which the consistency conjunct is *not satisfied* simply does not fire, exactly like one blocked by `min_request` — the account keeps accumulating, and the reset-after-payout already in the model matches the firms' "resets after each payout." Thresholds (35%/40% Pro, 50% Flex) and the pre/post-2025-11-28 legacy split are config, on the same legacy-date pattern as the payout split tiers (§6b). The Flex **cushion** — a small allowance making the effective bound looser than the nominal percentage, computed on actual daily profit — is a per-firm parameter to calibrate, not a new mechanic.

Some firms enforce consistency instead by *raising the profit target* (never failing, never gating a payout, but making the bar harder). That is the same predicate with a different action — `Action.ADJUST` writing `PROFIT_TARGET`:

```python
@dataclass(frozen=True)
class ConsistencyRaisesTargetRule(Rule):
    threshold: float
    raise_to: float     # new PROFIT_TARGET when the consistency condition is violated
    check_timing: Timing = Timing.EOD   # consistency is a whole-day property; evaluated at day close
    def requirements(self): return (StateField.DAY_PNL, StateField.TOTAL_PNL,
                                     StateField.MAX_DAY_PNL, StateField.PROFIT_TARGET)
    def compile(self): return CompiledRule(kind=RULE_CONSISTENCY_ADJUST, p0=self.threshold,
                                           p1=self.raise_to, action=Action.ADJUST,
                                           check_timing=self.check_timing,
                                           adjust_field=StateField.PROFIT_TARGET)
```

`ConsistencyGate(0.4)` and `ConsistencyGate(0.5)` are the same implementation with different data — new *numbers* are new instances, authored in `firms/`; only new *behavior* (gate-vs-adjust) is a new class here.

### The rule registry (bottom of `rules.py`)

Because the config floor accepts any rule object, every rule a config can name must be implemented in the kernel. The registry provides that guarantee: it maps each rule's `kind` code to the kernel behavior, and the compiler hard-fails if asked to compile a rule whose `kind` is not registered.

```python
# every rule kind the kernel implements. A rule class not in here CANNOT be simulated.
RULE_REGISTRY = {
    RULE_PROFIT_TARGET:     ProfitTargetRule,
    RULE_TRAILING_DD:       TrailingDrawdownRule,
    RULE_STATIC_DD:         StaticDrawdownRule,
    RULE_DAILY_LOSS:        DailyLossRule,
    RULE_MIN_DAYS:          MinimumTradingDaysRule,
    RULE_MIN_WINNING_DAYS:  MinimumWinningDaysRule,
    RULE_CONSISTENCY_GATE:  ConsistencyGate,
    RULE_CONSISTENCY_ADJUST: ConsistencyRaisesTargetRule,
}

class UnknownRuleError(Exception): ...

def assert_kernel_supports(kind: int):
    if kind not in RULE_REGISTRY:
        raise UnknownRuleError(
            f"rule kind {kind} has no kernel implementation. "
            f"Add: (1) a Rule subclass, (2) a kind code, (3) a kernel branch, "
            f"(4) a RULE_REGISTRY entry."
        )
```

Adding a rule is always the same four steps. A new condition that needs *memory* (a running count) is that plus a new `StateField` counter and its update at the day boundary — the counter/predicate pattern that covers winning-day counts, soft-breach escalation, and stage predicates alike.

---

## 6. The predicate/action layer

This section states the single mechanism the previous sections instantiate. Every rule is a **predicate over pure kernel state** paired with an **action on trigger**. There are exactly two predicate combinators and three actions.

**Combinators.**
- **Fail-predicates are disjunctive:** the kernel checks them each trade; the *first* to trigger fires its action. Ordering defines attribution precedence.
- **Pass/payout-predicates are conjunctive:** the kernel fires the shared action only when *all* predicates of that action are simultaneously satisfied.

**Actions.**
- **`FAIL` + `Severity.HARD`** — terminate the simulation, return the fail code.
- **`FAIL` + `Severity.SOFT`** — truncate the current trading day: skip the remaining trades of this day, then resume at the next day. State accumulated *up to* the breach stands (the partial loss is real, the day still counts as a trading day, it does not count as a winning day); only subsequent same-day trades are prevented.
- **`PASS`** — terminal success; the phase is cleared. Used by eval phases.
- **`PAYOUT`** — repeatable success; record a payout event (amount and day-index), reset the counters this account's config says reset per payout, and continue. Used by funded phases.
- **`ADJUST`** — mutate a target state field instead of ending anything. Used by rules that *change the account* rather than fail or pass it — e.g. a consistency rule that raises `PROFIT_TARGET` rather than failing the trader. The target is a live `StateField` seeded from the rule's parameter; the fire-action rewrites it. Because the target lives in state (not as a static constant read by another rule), this does not violate the purity discipline: the profit-target pass-predicate still just *reads* `PROFIT_TARGET`, unaware anything adjusted it.

**Purity discipline.** Predicates only *read* state; actions *write* state. Ordering and history ("post-payout", "after N soft breaches"), and cross-rule effects ("consistency raises the target"), are never expressed as one predicate depending on another — that would smuggle a state machine back in. Instead the underlying fact is tracked state (`PAYOUTS_TAKEN`, `N_SOFT_BREACHES`, `PROFIT_TARGET`), which a fire-action writes and a plain predicate reads. This keeps every predicate independent and composable.

**Stages are the same mechanism, surfaced as a bitmask.** A stage is an independent boolean predicate over kernel state (`IN_PROFIT`, `CONSISTENCY_SATISFIED`, `PRE_FIRST_PAYOUT`, …). Because several are true at once, stage membership is a bitmask (`STATE.STAGE_MASK`), not a single current-stage integer. Each trade, the kernel sets each stage bit to its predicate's truth. An account that lacks a stage never sets that bit — the same "only compute what is required" principle as the requirements resolver. A payout condition such as "in profit AND 5 qualifying winning days AND consistency satisfied" is then literally "these required bits are all set", and a future sizing policy conditions on the mask directly. This is why stages must be a bitmask from the start: combinations of simultaneously-active stages are expressible; a single-stage enum could never represent them.

Everything the kernel does — eval passing, multi-condition payouts, hard and soft breaches, target adjustment, winning-day counters, stages — is one of these predicate/action combinations. The loop in §12 is a direct transcription of this section.

### 6a. Timing: two orthogonal axes per rule

*When* a rule acts is as configurable as *what* it does, and it is carried by the rule, never baked into loop order — otherwise every firm with a different timing model would need a kernel fork. Two independent `Timing` fields per rule capture the full space:

- **`update_timing`** — when the rule's *reference point* advances. For trailing drawdown this is when `PEAK_EQUITY`/`DD_FLOOR` ratchet up: `CONTINUOUS` = on every trade's new high; `EOD` = only off the day's closing equity.
- **`check_timing`** — when the *predicate is evaluated*. `CONTINUOUS` = tested intraday against the day's **intraday low-water mark** (catches a breach that floated through the line before close, including gap-through); `EOD` = tested only against the day's **closing equity**.

The two are orthogonal, and the common real configurations are cells of their product:

| update × check | Example |
|---|---|
| CONTINUOUS × CONTINUOUS | Intraday trailing drawdown (updates and breaches live) |
| EOD × EOD | Pure end-of-day drawdown (floor and breach both at close) |
| **EOD × CONTINUOUS** | **Floor trails at EOD, breach detected intraday on floating balance** |
| CONTINUOUS × EOD | Rare, but expressible for free |

The `EOD × CONTINUOUS` row is the case that a single timing flag cannot represent, and it is why timing must be two fields. It also depends on the day carrying two equity facts — its closing equity and its intraday low — which is the data-model addition in §11.

Timing fields live on *every* rule, not just drawdown, because the axis is general (a consistency target may be EOD while a drawdown is intraday). The cost is two `int8`s per rule in the struct-of-arrays; rules for which timing is irrelevant simply default to `CONTINUOUS`.

**Trailing-DD locking.** A near-universal companion to trailing drawdown: once the floor reaches a threshold (commonly the starting balance, or start + a fixed offset), it *stops trailing and locks*. This is two things on the rule — a `lock_at` parameter — and one bit of state (`DD_LOCKED`). When `DD_FLOOR` crosses `lock_at`, the fire-action sets `DD_LOCKED`, and the update step thereafter leaves the floor fixed regardless of `update_timing`. Because it is near-universal it belongs in the general mechanism, not as a per-firm exception.

### 6b. Payout schema — the product's upside leg

A `PAYOUT` predicate decides *whether* a payout can be requested; the **payout schema** decides *how much* is actually released and *what it does to account state*. Because the entire economic value of a funded account is the sum of released payouts, this schema is not an add-on — it is the product definition, and it is per-account-type config. Real firms vary every field independently, so each is explicit.

**The fire/amount boundary (stated once, for the whole model).** A payout must never fire for a zero or blocked amount — doing so would record a $0 payout, consume one of `max_payouts`, and reset the cycle counters, corrupting the payout-count distribution (§14.1) that is the product's headline. Therefore the conditions that can make a release *impossible* are **fire gates**, evaluated as part of whether the payout fires at all, not after-the-fact amount reducers:

- **`min_request`** (cycle profit below the minimum requestable) → the payout does not fire; the account keeps accumulating toward a future cycle where a request is possible.
- **`buffer_floor`** (releasing would drop balance below the non-withdrawable floor) → the payout does not fire this cycle. Note this gate depends on *balance*, not cycle profit, so it is a genuinely separate condition from the qualifying-day/consistency conjunction.

The remaining conditions merely *size* an amount that will definitely be released (a fire is guaranteed positive):

```
requestable = cycle_profit                                   # profit earned this cycle
# --- FIRE GATES (payout does NOT fire if either fails; nothing is recorded, no slot consumed) ---
can_fire = qualifying_conjunction                            # winning-days / consistency / etc.
       and requestable >= min_request
       and (balance - min(dollar_cap(i), cap_fraction*requestable)) >= buffer_floor
# --- AMOUNT (only computed when can_fire; always > 0) ---
gross = min( dollar_cap(payout_index),                       # per-request ceiling, may step with index
             cap_fraction * requestable )                    # AND a fraction-of-profit ceiling
net   = split(cumulative_paid, gross)                        # trader's share, possibly tiered
```

So `_all_payout_satisfied` in the kernel (§12) evaluates the full `can_fire` gate — qualifying conjunction **and** `min_request` **and** `buffer_floor` — and the fire block runs only then. The old "`gross = 0 if …`" formulation is replaced by these gates; there is no path that records a zero payout.

The schema fields, each drawn from a real Lucid mechanic:

```python
@dataclass(frozen=True)
class PayoutSchema:
    # --- how much can be requested ---
    dollar_cap: tuple[float, ...]     # per-payout-index ceiling; last value repeats.
                                      #   Flex 50K: (2000,) → flat; Pro 50K: (2000, 2500) → steps after payout 1
    cap_fraction: float = 1.0         # fraction-of-cycle-profit ceiling; Flex = 0.5 ("50% of profit up to cap")
    min_request: float = 0.0          # minimum cycle profit to request at all (Flex 50K: 500)
    # --- what must stay in the account ---
    buffer_floor: float = 0.0         # ABSOLUTE non-withdrawable balance level (Pro 50K: 52_100 =
                                      #   50_000 funded start + 2_000 max-loss + 100). Flex: 0.
                                      #   Coheres only relative to the funded start balance = account.size;
                                      #   the validator (§9) checks it is sane vs that start (B3).
    # --- trader's share ---
    split: float = 0.90               # flat trader share
    split_first_tier: float | None = None   # legacy: 100% up to a cumulative threshold, then `split`
    split_tier_cap: float = 0.0             # cumulative-paid threshold at which the tier changes
    # --- lifecycle ---
    max_payouts: int = 5              # after this many, the account reaches its terminal transition
    # --- what a taken payout does to state (the post-payout transition, §6b.1) ---
    reset_fields: tuple[StateField, ...] = ()   # counters zeroed each payout (qualifying days, consistency window)
    withdraw_reduces_equity: bool = True        # does the paid amount leave the account balance?
    recompute_floor_on_payout: bool = False     # does DD_FLOOR re-derive off the post-withdrawal balance?
```

The dollar cap is a **tuple indexed by payout number** because caps step: Flex is `(2000,)` (flat — the last element repeats for all later payouts), Pro is `(2000, 2500)` (base, then +500 from payout 2), Direct is `(2000, 2000, 2000, 2500)` (step at payout 4). The `cap_fraction` captures Flex's "50% of profit up to $2,000", meaning a $3,500 cycle releases $1,750, not $2,000 — reaching the dollar ceiling needs twice the ceiling in cycle profit. The `buffer_floor` captures the Pro/Daily non-withdrawable balance that blocks a request until the account sits above it; Flex has none. `split_first_tier`/`split_tier_cap` capture the grandfathered "100% on the first $10k, then 90%" legacy accounts.

#### 6b.1 The post-payout transition (closes a critical hole)

Taking a payout is a **state transition**, not just a recorded number, and its details dominate any multi-payout estimate. Three things happen, each config-controlled because firms differ:

1. **Counters reset** (`reset_fields`) — qualifying-day count and consistency window zero out; the next cycle starts fresh. This is the existing reset-mask mechanism.
2. **Equity may drop** (`withdraw_reduces_equity`) — if the paid amount leaves the balance, `equity -= gross`. If instead profit above the cap *stays in the account as buffer* (the Lucid behavior: excess is not forfeited), only the withdrawn `gross` reduces equity while the retained overflow remains. **Disambiguation ("rolls forward" reads two ways):** the retained excess becomes *drawdown cushion*, not a re-withdrawable balance — the kernel folds it into the new cycle baseline via `cycle_start_equity = equity` (after any withdrawal), so it raises the distance to the floor but is *not* counted as cycle profit toward the next payout. It protects the account; it does not pre-fund the next withdrawal. This is the consequential post-payout choice A2 flags; it is stated here as the default and is per-firm overridable.
3. **The drawdown floor may re-derive** (`recompute_floor_on_payout`) — whether the trailing floor recomputes off the post-withdrawal balance or persists. This decides whether the *next* payout is easier or harder, and is the single most consequential post-payout choice. For a locked floor, locking persists through the payout.

Because retained excess profit stays as buffer, a taken payout typically makes the account *safer* on the next cycle (more distance to the floor) even as it extracts cash — a structural feature the schema must preserve, since it directly affects survival to later payouts.

#### 6b.2 Terminal transition

After `max_payouts`, the sim account reaches its terminal state, returning `ExitCode.MAXED_OUT` — a funded economic *success*, deliberately distinct from eval `PASSED` so downstream filters can't conflate "cleared the eval" with "hit the payout ceiling" (`MODEL_RISKS.md` §H3). The accumulated payouts are the recorded outcome. (The real firms route this into a live-account review; modeling the live phase is out of scope — the sim account's economic output is fully captured by the payouts it released before the terminal transition.)

The payout schema lives on the funded `Phase` (the `payout_schema` field, §4) and compiles alongside its rules; it is part of the account fingerprint (§10). The qualifying `PAYOUT` predicates gate *whether* a payout fires, and the schema computes *how much* and *what changes*.

---

## 7. Firm config (`config.py`, `firms/`) — the three-layer format

The taxonomy is `Firm → Program → (default) Variant → Account(size) → Phase(eval|funded) → Rule`. Separate account types (for example DLL and NoDLL variants of a product) are modeled as separate `Program`s, each with a single implicit `default` variant; the variant level is structural padding set once and never authored by hand.

The format is built for arbitrary irregularity first, so regular accounts are the cheap special case. Three layers, each strictly on top of the last.

### Layer 1 — Unconditional floor: tables of real rule objects

A cell holds a fully-formed eval/funded rule structure. Anything expressible as rules — including per-account choices like a rule's `Severity` or which counters a payout resets — is expressible here, and this floor never narrows.

```python
# firms/lucid.py — pure data, no logic
LUCID_FLEX_DLL = {
    "50K": dict(
        eval=(ProfitTargetRule(3000.0), TrailingDrawdownRule(2500.0),
              DailyLossRule(1000.0), MinimumTradingDaysRule(3)),      # >=3 days before pass
        funded=(TrailingDrawdownRule(2500.0),
                DailyLossRule(1000.0, severity=Severity.SOFT),        # soft: truncates the day
                MinimumWinningDaysRule(count=5, threshold=150.0)),    # payout condition
    ),
    # a size with genuinely different STRUCTURE just writes different rules:
    "150K": dict(
        eval=(ProfitTargetRule(9000.0), TrailingDrawdownRule(4500.0),
              DailyLossRule(3000.0), MinimumTradingDaysRule(7)),      # 7 not 3
        funded=(TrailingDrawdownRule(4500.0), DailyLossRule(3000.0),
                MinimumWinningDaysRule(count=5, threshold=150.0)),
    ),
}

# counters a payout resets, per account type (kernel provides the reset capability;
# config declares which counters actually reset — a firm-specific fact):
LUCID_FLEX_DLL_PAYOUT_RESETS = (StateField.N_QUALIFYING_DAYS,)
```

### Layer 2 — Opt-in sugar: `scaled()` + `build_accounts()`

For account types whose sizes share structure and only scale in value, these helpers avoid repeating the skeleton per size. They *produce* Layer-1 tables and accounts; they never restrict what a table may contain. An account type can use them for regular sizes and hand-write a quirky one.

```python
def scaled(rule_cls, per_size: dict[str, float], **fixed):
    """{size: rule_cls(value, **fixed)} — sugar for the regular case only."""
    return {sz: rule_cls(v, **fixed) for sz, v in per_size.items()}

def build_accounts(name_prefix, sizes: dict[str, int], cells: dict[str, dict]) -> tuple:
    """Turn a {size: {'eval':(...), 'funded':(...)}} table into Account objects.
    `cells` is exactly a Layer-1 table, however it was produced (by hand or by scaled())."""
    accounts = []
    for size_name, size_val in sizes.items():
        cell = cells[size_name]
        phases = tuple(
            Phase(name=role, role=role, rules=tuple(rules))
            for role, rules in cell.items()
        )
        accounts.append(Account(name=size_name, size=size_val, phases=phases))
    return tuple(accounts)
```

A mostly-regular account type reads as a small table with structure implied by a per-type cell builder, and quirks written explicitly:

```python
SIZES = {"25K": 25_000, "50K": 50_000, "100K": 100_000, "150K": 150_000}

def _flex_dll_cell(target, dd, dll, min_days):
    return dict(
        eval=(ProfitTargetRule(target), TrailingDrawdownRule(dd),
              DailyLossRule(dll), MinimumTradingDaysRule(min_days)),
        funded=(TrailingDrawdownRule(dd), DailyLossRule(dll, severity=Severity.SOFT),
                MinimumWinningDaysRule(count=5, threshold=150.0)),
    )

LUCID_FLEX_DLL = {
    "25K":  _flex_dll_cell(1500, 1000, 500,  3),
    "50K":  _flex_dll_cell(3000, 2500, 1000, 3),
    "100K": _flex_dll_cell(6000, 3000, 2000, 3),
    "150K": _flex_dll_cell(9000, 4500, 3000, 7),   # quirk visible in one line
}
LucidFlexDLL = Program("LucidFlexDLL",
    variants=(Variant("default", build_accounts("LucidFlexDLL", SIZES, LUCID_FLEX_DLL)),),
    version="v2026_08")
```

The cell builder is per-account-type sugar used when it helps. An account type whose every size differs skips it and writes raw cells; nothing forces the helper.

### Layer 3 — Safety net: `validate()` + `RULE_REGISTRY`

Because Layer 1 accepts anything, two guards catch what a permissive format would pass silently. The registry (§5) rejects unimplemented rules. The validator (§9) rejects broken accounts without rejecting irregular ones.

The division of labor: **`validate` rejects broken; `RULE_REGISTRY` rejects unimplemented; the table format permits every irregularity that is neither.** Expressiveness is unconditional; every convenience is opt-in on top; the safety net catches what the permissive floor lets through.

---

## 8. Requirements resolver (`compiler.py`)

Unioning state-field *names* is unsafe if two rules mean different things by the same field. `StateField` is therefore semantic, not nominal: `PEAK_EQUITY` means one specific thing (the running max of equity across the phase). A rule needing "intraday peak" uses a *different* `StateField` member rather than reinterpreting an existing one, which keeps the union always safe.

```python
def resolve_requirements(phase) -> frozenset[StateField]:
    needed = set()
    for rule in phase.rules:
        needed.update(rule.requirements())
    # EQUITY and DAY_INDEX are always needed to drive the loop
    needed.update({StateField.EQUITY, StateField.DAY_INDEX})
    return frozenset(needed)
```

The compiler emits, per phase: the live `StateField` set, a struct-of-arrays rule representation (`kind[]`, `p0[]`, `p1[]`, `action[]`, `severity[]`, `fail_code[]`), the per-payout reset mask, and grouping of predicates by action so the kernel can iterate fail-predicates and conjoin pass/payout-predicates without per-trade branching on action type. An account with no consistency rule never allocates or updates `MAX_DAY_PNL`; an account with no winning-days rule never maintains `N_QUALIFYING_DAYS`.

**Known state-layout limitation — at most one *trailing* drawdown reference per phase.** The kernel carries single scalars for the trailing floor (`peak`, `dd_floor`, `dd_locked`, `dd_amount`, `lock_at`), and `StateField` has one `PEAK_EQUITY`/`DD_FLOOR`/`DD_LOCKED`. Two *independent* `TrailingDrawdownRule`s in one phase would therefore collide on that single reference — a real, acknowledged exception to the "arbitrary rule structure is the floor" principle (§7). It is deliberately not solved now because no target firm needs two trailing floors in one phase; the neighbouring cases are fine (a *static* drawdown stores no reference — it recomputes inline as `start − amount`; and the EOD-vs-intraday case is *one* trailing rule with two timing fields, §6a). If a firm ever needs two independent trailing references, the fix is to index DD state by rule rather than keep phase-level scalars. Until then the validator (§9) rejects >1 trailing-DD rule per phase, so the limitation fails loudly rather than colliding silently.

`CompiledRule` is a plain record; in the kernel it is parallel NumPy arrays, never a Python object:

```python
# struct-of-arrays passed to the kernel
rule_kind:        int8[R]
rule_p0:          float64[R]   # primary parameter (amount / target / threshold / count)
rule_p1:          float64[R]   # second parameter (winning-day threshold / lock_at / raise_to); 0/INF when unused
rule_action:      int8[R]      # Action (FAIL / PASS / PAYOUT / ADJUST)
rule_severity:    int8[R]      # Severity (meaningful only for FAIL)
rule_update_tim:  int8[R]      # Timing — when the reference point advances
rule_check_tim:   int8[R]      # Timing — when the predicate is evaluated
rule_adjust_field:int8[R]      # StateField to mutate (meaningful only for ADJUST)
rule_fail_code:   int8[R]
```

---

## 9. Validator (`validate.py`) — permissive, not a schema

A rigid schema would reject legitimate irregularity, defeating an all-firms engine. The validator instead asserts sanity invariants that must hold regardless of firm structure, and runs inside `Engine.run()` before compilation.

```python
from .enums import StateField, Action
from .rules import assert_kernel_supports

# state the kernel actually produces; any rule needing something else is a bug
KERNEL_PRODUCED_STATE = {
    StateField.EQUITY, StateField.PEAK_EQUITY, StateField.DD_FLOOR, StateField.DD_LOCKED,
    StateField.DAY_LOW, StateField.DAY_PNL, StateField.TOTAL_PNL, StateField.DAY_INDEX,
    StateField.N_TRADING_DAYS, StateField.MAX_DAY_PNL, StateField.N_QUALIFYING_DAYS,
    StateField.PAYOUTS_TAKEN, StateField.N_SOFT_BREACHES, StateField.STAGE_MASK,
    StateField.PROFIT_TARGET, StateField.CYCLE_START_EQUITY, StateField.CUMULATIVE_PAID,
}

class InvalidAccountError(Exception): ...

def validate(account) -> None:
    if not account.phases:
        raise InvalidAccountError(f"{account.name}: no phases")
    for ph in account.phases:
        if not ph.rules:
            raise InvalidAccountError(f"{account.name}/{ph.name}: no rules")
        for r in ph.rules:
            compiled = r.compile()
            assert_kernel_supports(compiled.kind)              # kernel can run it
            for sf in r.requirements():                        # needs only producible state
                if sf not in KERNEL_PRODUCED_STATE:
                    raise InvalidAccountError(
                        f"{account.name}/{ph.name}: {type(r).__name__} needs "
                        f"{sf!r}, which the kernel does not produce")
            for field_name, val in vars(r).items():            # sane parameters
                # NOTE: this blanket "reject any negative numeric field" is a coarse FLOOR, not a
                # schema — it will wrongly reject the first legitimately-signed parameter a future
                # rule needs (e.g. a rule whose threshold can be negative). Tighten to per-field
                # bounds when such a rule appears; adequate for the current rule set.
                if isinstance(val, (int, float)) and val < 0:
                    raise InvalidAccountError(
                        f"{account.name}/{ph.name}: {type(r).__name__}."
                        f"{field_name} is negative ({val})")
        # state-layout limit (§8): at most one trailing-DD reference per phase, since the kernel
        # carries a single trailing floor. Fail loudly rather than collide silently.
        if sum(isinstance(r, TrailingDrawdownRule) for r in ph.rules) > 1:
            raise InvalidAccountError(
                f"{account.name}/{ph.name}: >1 TrailingDrawdownRule — the kernel supports one "
                f"trailing reference per phase (§8). Index DD state by rule if a firm needs two.")
        # buffer_floor sanity (B3): it is an ABSOLUTE balance level; a value far above the funded
        # start silently blocks EVERY payout (funded phase yields zero income yet passes otherwise),
        # and a value below start never gates. Require it within a sane band of the funded start.
        if ph.role == "funded" and ph.payout_schema is not None:
            bf = ph.payout_schema.buffer_floor
            start = account.size    # funded phase starts at the account size
            if bf and not (start <= bf <= start * 1.5):   # floor at/above start, not absurdly high
                raise InvalidAccountError(
                    f"{account.name}/{ph.name}: buffer_floor {bf} is not sane vs funded start "
                    f"{start} — below start never gates; far above start blocks all payouts.")
        # role-aware terminability:
        _assert_terminable(account, ph)
```

`_assert_terminable` encodes the one universal structural rule, keyed on `Phase.role`. An **eval** phase must have at least one `Action.PASS` predicate, or it can never be cleared — the most common config typo (a dropped profit target). A **funded** phase need not be passable at all; it may consist only of survival (`FAIL`) rules and repeatable `PAYOUT`s. The check therefore requires a terminal action *appropriate to the role* rather than a fixed rule list, which admits every legitimate irregularity while still catching the unwinnable-eval mistake.

---

## 10. Fingerprint (`fingerprint.py`) — one authoritative hash

Because everything is frozen tuples of primitives, a structural hash is well-defined and stable. The version string lives inside the hashed content, so it cannot disagree with the hash.

```python
import hashlib, json

def fingerprint(account, program_version: str) -> str:
    payload = {
        "version": program_version,
        "size": account.size,
        "currency": account.currency,
        "eval_fee": account.eval_fee,            # fee is the entire downside (§0) — part of identity;
        "activation_fee": account.activation_fee,#   two accounts differing only in fee are different products
        "phases": [
            {"name": ph.name, "role": ph.role,
             "rules": sorted(
                 [ [type(r).__name__] + [f"{k}={v}" for k,v in sorted(vars(r).items())]
                   for r in ph.rules ]),
             # payout schema is part of the account's identity — omitting it would collide
             # accounts differing only in split/cap/etc. to one cache key (see MODEL_RISKS A1)
             "payout": (sorted(f"{k}={v}" for k,v in vars(ph.payout_schema).items())
                        if ph.payout_schema is not None else None)}
            for ph in account.phases
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",",":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
```

Because per-account resolved values are hashed, a size-specific quirk (such as `min_days=7` on the 150K, or a rule's `severity`) produces its own fingerprint, and correcting a table value changes the fingerprint so stale cached results are not silently reused. **The payout schema is part of the hashed identity** — two funded accounts differing only in `split`, `dollar_cap`, `cap_fraction`, or a post-payout transition flag must not collide to one cache key. This is both the compiled-account cache key and the reproducibility record — a result is tied to config `a3f9…`, e.g. LucidFlexDLL 50K v2026_08.

---

## 11. Trade data (`data.py`) — required input and derived representation

This is the dataset the engine consumes. There are two layers: the **columns the caller must supply** (the raw input contract) and the **arrays preprocessing derives** from them (what the kernel actually reads).

### 11.1 Required input columns

The raw input is one row per closed trade, in any tabular form (CSV, Parquet, DataFrame). The required and optional columns:

| Column | Type | Req? | Meaning |
|---|---|---|---|
| `timestamp` | datetime | **required** | Close time of the trade. Used to order trades and to derive the trading-day index. Must be timezone-consistent; the session boundary that defines a "day" (see 11.3) is applied to this. |
| `return` | float | **required*** | Per-unit return of the trade — P&L per 1 unit of size, *after* commissions/fees. The kernel computes realized P&L as `size × return`. Supplying return (not raw dollar P&L) is what lets a future sizing policy scale it. |
| `pnl` | float | alt. | Dollar P&L per trade at the size actually traded historically. Accepted as an alternative to `return`; preprocessing converts it to a per-unit return by dividing by the historical size (next column), which is then required. |
| `size` | float | cond. | The size the historical trade was taken at. Required only when the input provides `pnl` rather than `return`, to normalize to per-unit. |
| `mae` | float | optional | Maximum Adverse Excursion: the worst *unrealized* loss reached during the trade **while the position was actually open**, in the same per-unit units as `return`. Must be computed from bar excursions *clipped to the position's `[entry_time, exit_time]` holding interval* — a bar low before entry or after exit is not exposure and must not be included (`MODEL_RISKS.md` §D1). This is what makes intraday/floating-balance breach detection real rather than realized-only (see 11.4). Strongly recommended for any firm with an intraday drawdown or daily-loss check. |
| `mfe` | float | optional | Maximum Favorable Excursion. Not needed for current rules; accept and ignore unless a future rule requires it. |
| `symbol` | str | cond. | Instrument. **Required when one account trades multiple assets**, because days are resampled jointly across assets (§11.4) — the loader groups trades into calendar days across all symbols. Optional (single constant) for a one-instrument account. |

\* Exactly one of `return` or (`pnl` + `size`) must be present. Everything else is optional.

The single most consequential optional column is **`mae`**. Without it, "intraday low" degrades to the running low of *realized* equity between closed trades, and true within-trade gap-through cannot be seen (11.4). With it, the intraday low-water mark reflects the real floating drawdown. The engine runs either way; `mae` sets the fidelity ceiling of every `check_timing=CONTINUOUS` rule.

### 11.2 Derived representation (`TradeDataset`)

Preprocessing converts the raw rows into contiguous NumPy arrays — one standardized representation, preprocessed once and reused across every firm and account tested. It carries **per-trade** arrays and a small **per-day** side table, because intraday-timing rules (§6a) evaluate against day-level facts.

```python
@dataclass(frozen=True)
class TradeDataset:
    # --- per-trade arrays, length N (trade order) ---
    ret:        "np.ndarray"   # float64[N] — per-unit return, after fees
    day:        "np.ndarray"   # int32[N]   — trading-day index, 0-based, monotonic non-decreasing
    trade_low:  "np.ndarray"   # float64[N] — per-unit floating low of THIS trade (from mae; = min(ret,0) if absent)
    # --- per-day side table, length D (day order); index by TradeDataset.day ---
    day_first:  "np.ndarray"   # int32[D]   — index of first trade of each day
    day_count:  "np.ndarray"   # int32[D]   — number of trades in each day
    n_days:     int
    # --- calendar cadence, for converting simulated trading-days to wall-clock time (§14) ---
    trading_days_per_week: float   # e.g. 5.0 — how many trading days the strategy is active per calendar week

    @property
    def n_trades(self) -> int: return self.ret.shape[0]
```

`trade_low` is the per-unit worst floating point of each trade: from `mae` when supplied (`-mae`), otherwise the realized `min(ret, 0)` as a lower-fidelity fallback. The kernel combines a running equity with `size × trade_low` to maintain the current day's intraday low-water mark (`StateField.DAY_LOW`), which `check_timing=CONTINUOUS` rules test against. The per-day `day_first`/`day_count` table lets the kernel find day boundaries in O(1) and lets resampling operate at day granularity (11.3).

### 11.3 Trading-day identification and the session boundary

A "trading day" is defined by a **session boundary** applied to `timestamp` — typically the firm's daily reset time (many futures firms reset at 5:00 pm US/Eastern, not midnight). This is a preprocessing parameter, not a hardcoded midnight split, because daily-loss and EOD-drawdown rules key off the firm's session, and getting it wrong silently miscounts every day-scoped rule. Trades are sorted by `timestamp`, then each is assigned a 0-based `day` index by the chosen session boundary; `day` is monotonic non-decreasing, which is what the kernel's day-rollover logic and the per-day table rely on.

### 11.4 Resampling granularity is coupled to the day model

Because the intraday low and every day-scoped counter belong to a *day*, resampling must preserve day integrity: the bootstrap resamples **whole trading days** — each day's trades in their original intra-day order, carried with that day's facts — rather than shuffling individual trades across day boundaries. Trade-level resampling would scramble which trades compose a day and make `DAY_LOW`, `DAY_PNL`, and winning-day counts meaningless. Day-level block resampling is also the more faithful model for these accounts, since the rules are overwhelmingly day-structured. The stationary bootstrap's block unit is therefore the day; `day_first`/`day_count` make assembling a resampled path a gather over day blocks.

**Multi-asset accounts resample days jointly across all assets.** When one account trades several instruments (e.g. correlated futures such as MES/MNQ/M2K plus MGC/SIL/MCL/6E), the account's equity is the aggregate across them, so a "day" is *every asset's trades on that calendar day, kept together*. Assets must never be resampled independently — doing so destroys the cross-asset correlation that produces the worst aggregate drawdown days (equity indices dumping together, metals together), which is exactly the tail that knocks accounts out. The block unit is the joint calendar day across the whole instrument set, and the mean block length should be estimated from the *aggregate* daily series' dependence structure, not guessed (see `MODEL_RISKS.md` §G1).

### 11.5 Pipeline and caching

`raw rows → validate (required columns, one of return/pnl) → sort by timestamp → normalize to per-unit return → derive trade_low → assign trading-day index by session boundary → build per-day table → derive trading_days_per_week from the calendar span → TradeDataset`. Done once and cached keyed on a hash of the raw input plus the session-boundary parameter, so testing many firms never re-preprocesses the same trades.

`trading_days_per_week` is derived once here (distinct trading days ÷ calendar weeks the data spans) and is the *only* bridge between the engine's internal clock and wall-clock time. The engine itself is clockless — it processes trades in sequence and has no notion of how much calendar time a resampled path represents — so this cadence is what makes every time-based statistic in §14 well-defined. A simulated path of `k` trading days corresponds to `k / trading_days_per_week` calendar weeks; **simulated duration is measured in resampled trading days, never as a fraction of the source dataset** (a bootstrap can generate a path longer or shorter than the history it was drawn from, so "fraction of data consumed" is a meaningless quantity and must not be used as a duration proxy).

### 11.7 Synthetic trade-stream generators (`synthetic.py`)

A synthetic generator **manufactures** raw trade rows from statistical parameters, as an alternative source to the real backtest. It exists for two purposes: (1) **testing** — deterministic, known-property streams to drive the Step 6–11 test suite without needing real data; and (2) **future research** — mapping which regions of strategy-space survive a given firm (the breakeven-line work, §11.7.4, deferred until the engine is complete).

**Critical contract: a generator emits the exact raw-row schema of §11.1** — `timestamp`, per-unit `return` (or `pnl`+`size`), `mae`, `symbol` — *not* the derived `TradeDataset`. Synthetic rows therefore flow through the *same* `preprocess()` pipeline as real data (validation, holding-interval `trade_low` derivation, session-day assignment, per-day table, cadence). The engine cannot tell synthetic from real input, which is exactly what makes synthetic data valid for testing the whole stack including Step 3.

#### 11.7.1 Strategy parameterization — `win_rate` and `RR` only

Risk is the unit: every trade risks exactly **−1R**. Reward is `RR` units on a win (`RR > 0`, risk-to-reward). A trade outcome is therefore `+RR` with probability `win_rate`, `−1` with probability `1 − win_rate`. **Edge is derived, never input** — accepting it as a third parameter would over-determine the system and admit user-inconsistent triples (an edge that disagrees with `win_rate`+`RR`). The generator *reports* the derived per-trade expectancy as output metadata:

```
edge (expectancy per trade, in R) = win_rate * RR - (1 - win_rate)         # = win_rate*(RR+1) - 1
breakeven win rate                = 1 / (RR + 1)                            # where edge = 0
```

Edge is a derived function of `win_rate` and `RR`, so the two-parameter grid spans the full space of first-order strategy statistics, and the breakeven contour (`edge = 0`, i.e. `win_rate = 1/(RR+1)`) is a known reference line within it. Different `RR` values have different breakeven win rates — a higher `RR` has a lower one — which is arithmetic, not a result. What a system's stats must be to clear a given firm, and how far from the breakeven line that requirement sits, is **not** determined by this parameterization; it is precisely what the breakeven-mapping research (§11.7.4) would measure by running swept parameters through the engine. Because edge is derived from the same `win_rate`/`RR` the stream is built from, the grid is internally consistent by construction.

Full parameter set: `win_rate`, `RR`, `trades_per_day` (drives the calendar/session structure and cadence), an **intraday-excursion** parameter (how deep a trade's adverse excursion typically runs relative to its outcome, so the synthesized `mae` is realistic enough to exercise `check_timing=CONTINUOUS` rules), plus each generator type's own dependence parameters (below). Win/loss *dispersion* (variance around the nominal `+RR`/`−1`) is an optional parameter; the base case is fixed-size outcomes.

#### 11.7.2 The generator ladder (mirrors the resampling ladder, but is distinct)

There are two deliberately-parallel ladders that share a dependence vocabulary but must not be confused (`MODEL_RISKS.md` §I1):

- The **resampling ladder** (§G1) takes *real* trades and varies only their *dependence structure* (i.i.d. day-bootstrap → stationary → regime-conditioned). It preserves the empirical return distribution.
- The **synthetic ladder** (here) *invents* trades and varies *both* distribution and dependence, using the same dependence names so results can be cross-checked ("does the breakeven line shift the same way when regime-dependence is added synthetically as when it's added via resampling?").

The generator types, all emitting the identical §11.1 raw-row contract behind one common interface:

1. **`IIDGenerator`** — independent draws: each trade is `+RR`/`−1` by an independent `win_rate` coin. The naive baseline; optimistic on drawdown for the same reason i.i.d. resampling is (no clustered losing streaks).
2. **`RegimeSwitchingGenerator`** — a Markov chain over regimes (e.g. good/neutral/bad), each with its own `win_rate` (and optionally `RR`), driven by a transition matrix. Produces the *persistent* winning and losing periods that actually knock accounts out — the streakiness i.i.d. can't make.
3. **`StochasticVolGenerator`** — trade magnitude scales with a slow-moving volatility process, producing volatility clustering and fatter realized tails than the nominal fixed-size outcomes.

#### 11.7.3 Common interface and provenance

All generators expose one interface: parameters in → a raw-row table out (§11.1 schema) → `preprocess()` → `TradeDataset`. Every synthetic dataset carries **provenance**: generator type, all parameters, seed, and the derived edge/breakeven. This rides through to `Results` so a synthetic-derived number is never mistaken for a real-data result (the synthetic analogue of the §G7/§G1 selection-bias discipline; see `MODEL_RISKS.md` §I1). Generation is deterministic under a fixed seed.

#### 11.7.4 Breakeven mapping (future research — not built until the engine is complete)

Once the engine is finished, a research layer sweeps generator parameters (`win_rate` × `RR` grids, per generator type and per firm/account), runs each synthetic strategy through the engine, and finds the contour where the pass/payout/renewal outcome crosses a chosen threshold — the *minimum-requirement surface* for a firm. This measures what statistics a system needs to clear firm X as a function of `RR`, and where that requirement sits relative to the breakeven line — whatever the answer turns out to be. It is a layer *on top of* the generator and engine, specified later; noted here only so the generator's parameterization is chosen to support it.

---

## 12. Kernels (`kernels.py`) — the hot path

The kernel is a direct transcription of §6 and §6a: a per-trade loop that maintains equity and the day's intraday low, advances timing-gated reference points, checks disjunctive fail-predicates against the timing-appropriate equity (intraday low for `CONTINUOUS`, closing equity for `EOD`), conjoins pass/payout-predicates, applies `ADJUST` mutations, updates stage bits, and records payout events. Performance techniques applied: `@njit(cache=True)`, `parallel=True` with `prange` over the batch, no allocation inside the inner loop, primitive arguments only, pre-generated RNG. **`fastmath` is deliberately *not* used** (`MODEL_RISKS.md` §C1/§G6): it is a function-level flag that permits floating-point reassociation, which would make the exact breach/target/payout boundary comparisons non-deterministic and defeat the Level-1 oracle-parity gate. This loop is integer- and comparison-heavy rather than reduction-heavy, so `fastmath` buys little here; correctness at the contract boundary is worth far more than the marginal vectorization.

Two equity facts drive the timing axes: `equity` (running realized equity at each trade close) and `day_low` (the worst floating equity seen so far this day, formed from `equity + size × trade_low[t]`). `CONTINUOUS` checks read `day_low`; `EOD` checks read the day's closing `equity` at the day boundary. Reference points (`peak`, `dd_floor`) advance under `update_timing`, and a floor that reaches `lock_at` sets `DD_LOCKED` and stops.

The single-path kernel is the correctness oracle, built and tested before Monte Carlo. Structure (elided arithmetic shown as comments to keep the control flow legible):

```python
@njit(cache=True)   # NO fastmath (§G6/C1): breach/target/payout tests are exact boundary comparisons
def simulate_one_phase(ret, day, trade_low, size_base, policy_params, start_equity, profit_target0,
                       rule_kind, rule_p0, rule_p1, rule_action, rule_severity,
                       rule_update_tim, rule_check_tim, rule_adjust_field, rule_fail_code,
                       payout_reset_mask, out_payout_amt, out_payout_day):
    equity = start_equity; peak = start_equity
    dd_floor = start_equity - dd_amount; dd_locked = 0   # LIVE from trade 1 (§C3): a trailing-DD
                                                         # account can breach on its first trade;
                                                         # no -INF sentinel / "seeded later" delay.
    day_pnl = 0.0; total_pnl = 0.0; max_day_pnl = 0.0; day_low = equity
    profit_target = profit_target0                       # live; ADJUST rules may raise it
    cycle_start_equity = start_equity                    # cycle_profit = equity - cycle_start_equity
    cumulative_paid = 0.0                                # gross dollars paid; drives tiered split
    n_days = 0; n_qual_days = 0; payouts_taken = 0; n_soft = 0
    stage_mask = 0; cur_day = -1
    t = 0; N = ret.shape[0]

    while t < N:
        d = day[t]
        if d != cur_day:                      # ----- day boundary -----
            if cur_day != -1:
                # _close_day may itself terminate the attempt (an EOD-timed breach, or an
                # EOD winning-day increment that completes a payout / hits max_payouts). Its
                # terminal return MUST be honored here, not just at end-of-path — otherwise the
                # entire EOD×EOD "pure end-of-day" timing row (§6a) would only ever fire on the
                # last day. (MODEL_RISKS §B1 fixed end-of-path; this is the same bug at every
                # intermediate rollover.)
                code = _close_day(...)        # winning-day count; EOD breach vs closing equity;
                                              # EOD ADJUSTs; EOD floor update + lock; EOD payout
                if code != ExitCode.ALIVE:
                    return code, payouts_taken
            day_pnl = 0.0; day_low = equity; cur_day = d; n_days += 1

        # sizing hook (§16.1): size is a compiled function of stage + policy params, not a constant.
        # Today policy_params has length 1 and this reproduces the fixed-size case; the optimizer
        # later widens policy_params without changing this call site or any downstream shape.
        # DELIBERATE SEMANTIC (§16.1): stage_mask here is last trade's, since stages are recomputed
        # AFTER the checks below. So size reacts to stage with a one-trade lag. This is the intended
        # rule (you size for trade t using the account state entering t); the optimizer inherits it.
        size = _size_policy(stage_mask, policy_params, size_base)
        # (CAPPED_OUT is deferred with the sizing policy — see MODEL_RISKS C2/A2. Not tested until a
        #  minimum-position-size config exists; a constant policy never caps out, so it is inert here.)

        p = size * ret[t]
        equity += p; day_pnl += p; total_pnl += p
        trade_floor = equity + size * trade_low[t]        # this trade's floating low (holding-interval clipped, §11)
        if trade_floor < day_low: day_low = trade_floor   # maintain intraday low-water mark

        # reference-point update — GUARDED by update_timing (§6a). CONTINUOUS ratchets intraday;
        # EOD updates happen only in _close_day off the closing equity. Without this guard an
        # EOD-update floor would wrongly trail intraday (MODEL_RISKS B3).
        if (not dd_locked) and dd_update_timing == CONTINUOUS:
            if equity > peak: peak = equity
            dd_floor = peak - dd_amount
            if dd_floor >= lock_at: dd_floor = lock_at; dd_locked = 1

        # --- FAIL predicates: disjunctive, first trigger wins (precedence = order) ---
        breached = _first_fail(rule_kind, rule_p0, rule_action, rule_severity, rule_check_tim,
                               equity, day_low, peak, dd_floor, day_pnl, start_equity)
        if breached.hit:
            if breached.severity == HARD:
                return breached.fail_code, payouts_taken
            else:                             # SOFT: truncate day, resume next day
                n_soft += 1
                # closing equity = equity after the last EXECUTED trade (here, the breach trade,
                # since later same-day trades are skipped) — the true EOD equity, not a special
                # breach value (§C5). Breach *detection* used the intraday low; the close uses equity.
                code = _close_day(..., closing_equity=equity)  # partial day counts; not a winning day
                if code != ExitCode.ALIVE:    # a soft-breached day can still terminate at its
                    return code, payouts_taken#   close (e.g. an EOD rule) — honor it here too.
                t = _advance_to_next_day(day, t); cur_day = -1
                continue

        # --- ADJUST predicates: mutate a target field (e.g. raise profit_target) ---
        # TIMING-AWARE (like fail checks): only CONTINUOUS adjusts fire here, intraday. EOD-timed
        # adjusts (the ConsistencyRaisesTargetRule default, §5) are deferred to _close_day and run
        # against the day's closing state — without this filter an EOD adjust would fire intraday
        # at the wrong time (same class of bug as the B3 floor guard).
        profit_target = _apply_adjusts(rule_kind, rule_p0, rule_p1, rule_action, rule_check_tim,
                                       CONTINUOUS,   # phase = intraday; _close_day passes EOD
                                       rule_adjust_field, profit_target, day_pnl, total_pnl, max_day_pnl)

        # --- stage bits: independent predicates over current state ---
        stage_mask = _recompute_stages(equity, peak, total_pnl, n_qual_days, payouts_taken)

        # --- PASS predicates: conjunctive; all must hold (reads live profit_target) ---
        if _all_pass_satisfied(rule_kind, rule_p0, rule_action, equity, start_equity,
                               profit_target, n_days):
            return ExitCode.PASSED, payouts_taken

        # --- PAYOUT predicates: conjunctive; fire only for a guaranteed-positive release (§6b) ---
        cycle_profit = equity - cycle_start_equity
        # _all_payout_satisfied evaluates the FULL fire gate: qualifying conjunction AND
        # cycle_profit >= min_request AND (balance - capped_gross) >= buffer_floor. It never
        # fires for a zero/blocked amount, so no $0 payout is ever recorded and no max_payouts
        # slot is burned when a request isn't actually possible (§6b, MODEL_RISKS payout gate).
        if _all_payout_satisfied(schema, payouts_taken, n_qual_days, cycle_profit, equity):
            gross = _payout_amount(schema, payouts_taken, cycle_profit)   # guaranteed > 0 here
            net   = _net(schema, cumulative_paid, gross)                  # tiered split reads cumulative_paid
            out_payout_amt[payouts_taken] = net
            out_payout_day[payouts_taken] = cur_day
            payouts_taken += 1
            cumulative_paid += gross
            _apply_post_payout(schema, ...)    # reset counters; maybe reduce equity; maybe recompute floor (§6b.1)
            cycle_start_equity = equity        # begin a new cycle (after any equity reduction the transition applied)
            if payouts_taken >= schema_max_payouts:
                return ExitCode.MAXED_OUT, payouts_taken   # funded success, distinct from PASSED (§6b.2, H3)
        t += 1

    # END-OF-PATH CLOSE (MODEL_RISKS B1): the final day never triggers a rollover, so its EOD checks,
    # EOD floor update, and winning-day finalization must be run explicitly here before returning.
    # This can still produce a terminal outcome (an EOD breach on the last day, or a payout whose
    # 5th qualifying day IS the last day), so its result — not an unconditional TIMED_OUT — is returned.
    if cur_day != -1:
        code = _close_day(...)                 # finalize last day; may breach, may complete a payout
        if code != ExitCode.ALIVE:
            return code, payouts_taken
    return ExitCode.TIMED_OUT, payouts_taken
```

`_close_day` is the single home for everything that happens at a day's end, reached at three points — a natural day boundary, a soft-breach truncation, and the end-of-path close after the final trade — and **all three callers honor its return** (`MODEL_RISKS.md` §B1): if it returns a terminal `ExitCode` the attempt ends there. Each caller supplies the day's **closing equity = the equity after the day's last *executed* trade** (`MODEL_RISKS.md` §C5); on a truncated day that is the trade at which the soft breach fired, since later same-day trades are skipped. This is distinct from the intraday low-water mark that *breach detection* reads — closing equity is the true EOD equity, always, never a breach-point substitute. `_close_day` does not distinguish why it was called — one code path. Its **internal order is fixed and load-bearing** (`MODEL_RISKS.md` §B2): (1) *first* fold the just-closed day into the day-scoped counters — update `max_day_pnl` from this day's `day_pnl`, and increment `N_QUALIFYING_DAYS` if the day met the winning-day threshold — *then* (2) evaluate the EOD predicates (breach, adjust, pass, payout) against the now-updated state and the supplied closing equity. So the closing day's own pnl **does** count toward its own EOD consistency test and its own winning-day payout gate; the day that completes the 5th qualifying win can pay out on that same close. Within step (2) the precedence is the **same as the trade loop (§C4): fail → adjust → pass → payout** — an EOD breach coinciding with an EOD-completed 5th qualifying day is a *failure*, not a payout. `_close_day` returns the resulting terminal `ExitCode`, or `ALIVE` if the day ends unresolved. Routing all three triggers through one function with one fold-then-evaluate order and one precedence is what keeps natural, truncated, and final day-ends from diverging.

The batch kernel is what Monte Carlo calls. `prange` gives one core per simulation; each simulation is independent, so it parallelizes cleanly:

```python
@njit(parallel=True, cache=True)   # NO fastmath (§G6/C1): keep boundary comparisons exact
def simulate_batch(all_idx, ret, day, trade_low, size_base, policy_params, start_equity, profit_target0,
                   rule_kind, rule_p0, rule_p1, rule_action, rule_severity,
                   rule_update_tim, rule_check_tim, rule_adjust_field, rule_fail_code,
                   payout_reset_mask,
                   out_code, out_payouts_taken, out_payout_amt, out_payout_day):  # written in place
    B = all_idx.shape[0]
    for b in prange(B):                       # parallel over simulations
        idx = all_idx[b]                      # idx gathers whole days (§11.4), not loose trades
        code, k = _simulate_indexed(idx, ret, day, trade_low, size_base, start_equity, ...,
                                    out_payout_amt[b], out_payout_day[b])
        out_code[b] = code
        out_payouts_taken[b] = k
```

Multi-phase composition happens outside the kernel, since phases are independent: phase 0 runs for all simulations, and phase 1 runs only for those that passed. Because a failed eval never reaches funded, the funded phase simulates only survivors — a speedup that follows directly from phase independence.

The loop never dispatches to Python rule objects. Everything is `int8`/`float64` and integer `kind`/`action`/`severity` codes; removing per-trade Python dispatch is the primary performance mechanism.

### Reference oracle (`reference.py`)

A slow, readable, pure-Python `step(trade)` implementation of the same predicate/action semantics exists solely for validation and debugging — it exposes full state after every trade ("why did this account fail on trade 4,217"). It is never used in the Monte Carlo loop; a per-trade Python interface is exactly the dispatch overhead the compiled kernel exists to avoid. Tests assert the kernel and the oracle agree on hand-built sequences (see `test_reference.py`).

---

## 13. Raw outcomes and statistics (`statistics.py`)

Exact quantiles require O(n) memory (Munro–Paterson), so a true single-pass streaming quantile is necessarily approximate. Raw outcome arrays are therefore retained — and retained *richly*, including the time axis — which makes batching purely a memory-throughput device and keeps exact quantiles and BCa bootstrap CIs available. Beyond ~10⁹ simulations, the P² streaming quantile estimator (five markers, O(1) memory) can be substituted.

The kernel records, per simulation: the exit code, the number of payouts taken, and for each payout its **amount and day-index**, plus the account size and total trading days. This is deliberately richer than any single metric consumes, so the distribution and time-normalized decision statistics (§14) — `P(profitable)`, the payout-count histogram, payout velocity, time-to-first-payout, return-on-fee per year — are pure post-processing and never require re-entering the kernel. From the per-payout day-indices the aggregator derives `first_payout_day` and `total_trading_days` for the time axis.

```python
def pass_rate(codes):                     # codes: int8[B]
    return np.mean(codes == ExitCode.PASSED)

def wilson_ci(k, n, z=1.96):              # binomial CI for a pass rate
    p = k / n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-h)/d, (c+h)/d)

def bootstrap_ci(values, stat=np.mean, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.empty(B)
    for i in range(B):
        boot[i] = stat(values[rng.integers(0, n, n)])
    return np.percentile(boot, [2.5, 97.5])
```

Because statistics never touch the simulation engine, methods such as BCa or regime-aware CIs can be added later without recompiling anything.

---

## 14. Decision statistics and objectives (`objectives.py`, `results.py`)

The account is a convex structured product (see `MODEL_RISKS.md` §0): bounded downside (the fee), capped path-dependent upside (the payouts). For a convex payoff the **shape of the outcome distribution carries the value, not its mean** — two accounts with identical `E[payout]` can be entirely different products (a reliable 2-payout grinder vs a mostly-lose-the-fee/occasionally-hit-5 lottery). The engine therefore reports the **payoff distribution and the decision statistics derived from it**, along **two axes the mean discards**: the distribution of outcomes, and their normalization by time. All of this is post-processing over the rich raw outcomes of §13 — no kernel or data-model change.

An **objective** is any function `raw_outcomes → scalar` intended for the deferred optimizer to maximize; the statistics below are the vocabulary those objectives are built from.

### 14.1 Distribution axis — because convexity lives in the shape

```python
# Fees are PATH-DEPENDENT (MODEL_RISKS H1): an attempt that fails the eval phase never pays the
# activation fee, so a single scalar fee mis-bills failed attempts. The engine records, per attempt,
# the fees actually attributable to it (always the eval fee; the activation fee only if it reached
# the funded phase). Statistics take that per-attempt `attributable_fee` array, not one number.
def attributable_fee(o, eval_fee, activation_fee):
    # eval_fee always; activation only for attempts that reached funded (o.reached_funded is bool[B]).
    # Direct-funded accounts (phases=(funded,)) set eval_fee to the entry cost and activation_fee=0
    # (reached_funded is trivially true from t=0), so the single entry cost is billed exactly once.
    return eval_fee + activation_fee * o.reached_funded

def prob_profitable(o, eval_fee, activation_fee):
    # THE key single-attempt decision number: P(this attempt's net payouts exceed the fees
    # attributable to it). Definition (fixed):
    #   profitable attempt == cumulative realized net payouts − fees attributable to THIS attempt > 0
    #   where attributable fee = eval_fee + (activation_fee if the attempt reached funded else 0).
    # This is an ATTEMPT-level statistic; the renewal-sequence analogue is defined separately in §15.
    return np.mean(o.net_payout > attributable_fee(o, eval_fee, activation_fee))

def payout_count_dist(o, max_payouts):
    # P(0 payouts), P(1), ... P(max) — the natural unit of the product, since caps make
    # each payout a discrete step and the account dies path-dependently.
    return np.bincount(o.payouts_taken, minlength=max_payouts + 1) / len(o.payouts_taken)

def return_on_fee(o, eval_fee, activation_fee):
    # distribution of net_payout / attributable_fee — the instrument's true yield, comparable across
    # sizes and prices on the one axis that matters (capital in vs capital out).
    return o.net_payout / attributable_fee(o, eval_fee, activation_fee)   # array; summarize with mean AND quantiles

def payoff_quantiles(o, eval_fee, activation_fee, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    net = o.net_payout - attributable_fee(o, eval_fee, activation_fee)
    return np.quantile(net, qs)   # full profile, incl. the low quantile the optimizer may target
```

`prob_profitable` is the single most decision-relevant scalar and is *provably not* recoverable from `E[payout]`: positive expected value at 20%-profitable is a different bet from positive expected value at 70%-profitable, selected and sized differently. The payout-count histogram is the product's actual payoff profile and falls straight out of the `payouts_taken` the kernel already records.

### 14.2 Time axis — because the product is a rate, not a lump sum

The fee is capital tied up for the account's duration and cannot be recycled until the account resolves; the whole stack-and-rotate model depends on *velocity*. `$9,000 over 7 weeks` and `$9,000 over 2 years` are the same number to a mean and radically different instruments. Time comes from the cadence in §11.5 — resampled trading-days converted to calendar time — **never from any fraction of the source dataset**.

```python
def _calendar_weeks(trading_days, trading_days_per_week):
    return trading_days / trading_days_per_week

def payout_velocity(o, ds, weeks_per_month=4.345):
    # expected NET payout per calendar month — the honest "is this a good income stream" number,
    # and what makes a fast Pro account and a slow Flex account comparable.
    months = _calendar_weeks(o.total_trading_days, ds.trading_days_per_week) / weeks_per_month
    return np.mean(o.net_payout / np.maximum(months, 1e-9))

def time_to_first_payout(o, ds):
    # distribution (report median + tail): when the account stops being pure risk and starts
    # returning capital. A long or fat-tailed time-to-first is a real defect even at good total EV.
    reached = o.first_payout_day[o.payouts_taken > 0]
    return _calendar_weeks(reached, ds.trading_days_per_week)   # array of weeks; summarize with quantiles

def return_on_fee_per_year(o, ds, eval_fee, activation_fee):
    # capital efficiency: annualized yield on the attributable fee. The closest thing to the
    # instrument's true rate of return, and a likely optimizer objective (see §15).
    years = _calendar_weeks(o.total_trading_days, ds.trading_days_per_week) / 52.0
    fee = attributable_fee(o, eval_fee, activation_fee)          # path-dependent (H1)
    return np.mean((o.net_payout / fee) / np.maximum(years, 1e-9))
```

Time-normalization is what stops any "maximize total payout" objective from silently preferring a slow account that ties the fee up forever: a per-year or per-month rate penalizes duration the mean ignores.

### 14.3 Composite / constrained objectives

Because both axes are available, the optimizer's objective can be well-posed for a convex, fee-per-attempt instrument — which a bare mean cannot express:

```python
def expected_payout_st_profitable(o, eval_fee, activation_fee, floor=0.5):
    # maximize expected net payout SUBJECT TO P(profitable) >= floor
    if prob_profitable(o, eval_fee, activation_fee) < floor: return -np.inf
    return np.mean(o.net_payout - attributable_fee(o, eval_fee, activation_fee))   # path-dependent (H1)
```

### 14.4 Raw outcome fields these consume

The above require, per simulation, that the kernel/aggregator expose: `net_payout` (sum of split-adjusted payouts, §6b), `payouts_taken`, `first_payout_day`, `total_trading_days`, `reached_funded` (bool — did the attempt clear the eval phase; drives path-dependent fees, H1), and `code`. All are already recorded (§13) or trivially derived from recorded per-payout day-indices; none needs a kernel change. **`max_payouts` is schema config, not a dataset field** (H2): it comes from the funded phase's `PayoutSchema` (§6b), not from `TradeDataset`.

**`total_trading_days` spans the *whole attempt*, eval + funded (H4).** Because phases run sequentially and the fee is tied up from the moment the eval starts until the attempt terminates, `total_trading_days` is the sum of the eval phase's days and the funded phase's days — not funded-only. An attempt that spends 12 days clearing the eval and 40 in the funded phase has `total_trading_days = 52`. Using funded-only here would understate the capital tie-up and overstate every rate in §15 (velocity, `R_renewal`, fee-efficiency), which is a first-order error for a product §0 defines as a *rate*. The aggregator therefore accumulates day-counts across every phase the attempt actually ran.

**`pass_rate` must not be used as an economic success metric for funded accounts (H3).** A funded attempt that banks several payouts and then runs out of path returns `TIMED_OUT`, not `PASSED` — it is an economic *success* with a non-`PASSED` code. `pass_rate` (which keys on `code == PASSED`) is meaningful only for the *eval* phase (did it clear?). Funded economic performance is read from `net_payout` / `payouts_taken` / the payout distribution, never from `pass_rate`. Conflating them silently undercounts funded success.

### 14.5 Results object

`Results` wraps the raw outcomes and exposes both axes lazily:

```python
class Results:
    def __init__(self, outcomes, dataset, eval_fee, activation_fee, max_payouts, fingerprint):
        # eval_fee/activation_fee come from the Account (§4), not passed loosely — they are part of
        # the account's identity and its fingerprint (§10). max_payouts comes from the funded
        # PayoutSchema (§6b), not the dataset (H2).
        self._o = outcomes; self._ds = dataset
        self._eval_fee = eval_fee; self._activation_fee = activation_fee   # fee is path-dependent (H1)
        self._max_payouts = max_payouts    # from the funded phase's PayoutSchema, NOT the dataset (H2)
        self.fingerprint = fingerprint
    # distribution axis
    @property
    def prob_profitable(self):   return prob_profitable(self._o, self._eval_fee, self._activation_fee)
    @property
    def payout_count_dist(self): return payout_count_dist(self._o, self._max_payouts)
    @property
    def payoff_quantiles(self):  return payoff_quantiles(self._o, self._eval_fee, self._activation_fee)
    # time axis
    @property
    def payout_velocity(self):   return payout_velocity(self._o, self._ds)
    @property
    def roi_per_year(self):      return return_on_fee_per_year(self._o, self._ds,
                                                               self._eval_fee, self._activation_fee)
    # optimizer entry point
    def objective(self, fn, **kw): return fn(self._o, **kw)
```

The mean (`E[payout]`) remains available, but it is one number among these, not the summary — reporting it alone would hide exactly the convexity and velocity the engine exists to measure.

---

## 15. Renewal economics (`renewal.py`) — analysis layer above the engine

The single account is one **renewal cycle**, not the terminal unit of analysis. The real process is sequential attempts: pay a fee → run an account attempt → it terminates (breach, timeout, or all payouts taken) → pay another fee → retry → repeat. The economically meaningful quantity is the **long-run cashflow rate** of this renewal-reward process, which is what makes otherwise-incomparable accounts comparable (a 20%-chance-of-$5k-in-5-days attempt vs a 60%-chance-of-$1.5k-in-30-days attempt are only rankable as repeated capital-recycling machines, not as single outcomes).

This is strictly an **analysis layer**. It sits above `Engine.run()` and consumes completed attempt outcomes; it never touches the kernel. The pipeline is:

```
trade process → account attempt (Engine.run) → outcome → renewal process → cashflow/time statistics
```

No retry logic, bankroll, or sequential-attempt state enters the kernel or the compiled account. The kernel still simulates exactly one attempt; the renewal layer composes many completed attempt outcomes in Python.

### 15.1 Three metric tiers, kept distinct

- **Single-attempt metrics (§14):** payout distribution, `P(profitable)`, time-to-payout, payoff quantiles, return-on-fee. These describe *one cycle*.
- **Renewal metrics (this section):** long-run reward rate, fee-bankroll efficiency (expected income per month per $1,000 of fee bankroll), expected cumulative cashflow over a finite horizon. These describe *the repeated process*.
- **Portfolio layer (deferred, not built):** simultaneous accounts and their correlation. Sequential renewal is in scope; simultaneous/correlated accounts remain explicitly deferred (see `MODEL_RISKS.md` §0). The line is deliberate — sequential attempts compose one-at-a-time completed outcomes and stay single-account; simultaneous accounts are the portfolio problem and are not modeled.

**"Profitable" is defined at two distinct levels — do not conflate them:**
- **Profitable *attempt*** (§14) = cumulative realized net payouts of one attempt − all fees attributable to that attempt (evaluation + activation) > 0.
- **Profitable *renewal sequence*** (this section) = cumulative net payouts across the sequence of attempts − cumulative fees paid across all attempts in the sequence > 0.

These are different statistics: a sequence can be profitable while most of its attempts were not (one funded run pays for many failed evals), and a high per-attempt profit rate can still lose over a sequence if fees accumulate faster than payouts. Renewal metrics use the sequence-level definition; single-attempt metrics use the attempt-level one.

### 15.2 The reward rate — two definitions, stated honestly

Each attempt `i` yields a net reward `R_i` (payouts − fee) and consumes a cycle time `T_i` (calendar time to termination, from §11.5 cadence). `T_i` is the **whole attempt's** duration — eval plus funded (§14.4/H4) — because the fee is tied up from the eval's first day until the attempt ends; charging only funded time would understate tie-up and inflate the rate. (Note each phase draws its own `[B, L]` path in the engine loop of §17, so an attempt's day-count is the eval path length plus the funded path length, not a single `L`; `L` is a *per-phase* horizon — see `MODEL_RISKS.md` §C7.) Two rates:

```
                E[R_cycle]                          Σ R_i
  R_renewal =  ------------          R_path = lim  ---------
                E[T_cycle]                   H→∞    Σ T_i
```

`R_path` — the realized long-run cashflow per unit time — is the quantity ultimately cared about. `R_renewal` (ratio of means) equals `R_path` **only under i.i.d. cycles and the renewal-reward theorem's ergodicity assumptions**. Two distinct things can break that equality here, and they must not be conflated:

1. **Ratio-estimator / finite-horizon bias (Jensen).** Even with perfectly i.i.d. cycles, `E[ΣR/ΣT]` over a finite horizon is not `E[R]/E[T]` — the ratio of sums is a biased estimate of the ratio of means, worsened by the skewed cycle-length a hard drawdown barrier produces (most attempts die fast, a few run long).
2. **Cycle correlation.** Attempts generated from the same strategy and the same resampled data can be *correlated* across cycles, which the renewal-reward theorem forbids.

The `r_path` below draws completed attempts **i.i.d.**, so it exposes only source (1) — it *cannot* diagnose source (2), because independent draws destroy exactly the cross-cycle correlation whose detection would be the point. This is the same limitation as the i.i.d. day bootstrap the risk doc warns against (`MODEL_RISKS.md` §B1), one level up. Diagnosing cycle correlation requires generating attempt *sequences that preserve order* — i.e. running the whole renewal chain on a single continuous resampled day-path rather than stitching independently-drawn attempts — which is deferred with the generator ladder (`MODEL_RISKS.md` §H5). So the layer honestly reports: `R_renewal` (closed form), `R_path` (i.i.d., isolating Jensen/finite-horizon bias), and a stated *gap* — cross-cycle correlation — that neither currently captures.

```python
def r_renewal(attempts):                 # ratio of means — valid under iid/ergodic cycles
    return attempts.net_reward.mean() / attempts.cycle_time.mean()

def r_path(attempts, horizon, n_sequences, seed):
    # Empirical finite-horizon rate under I.I.D. attempt draws. This isolates ratio-estimator /
    # finite-horizon (Jensen) bias vs r_renewal. It does NOT — and cannot — reveal cross-cycle
    # correlation, because independent draws remove it (MODEL_RISKS §H5). A divergence here means
    # Jensen/horizon bias is material; it does NOT certify the cycles are uncorrelated.
    rng = np.random.default_rng(seed)
    rates = np.empty(n_sequences)
    for s in range(n_sequences):
        R = T = 0.0
        while T < horizon:
            i = rng.integers(len(attempts.net_reward))   # I.I.D. draw of a completed attempt
            R += attempts.net_reward[i]; T += attempts.cycle_time[i]
        rates[s] = R / T
    return rates                        # distribution; compare its mean to r_renewal

def fee_bankroll_efficiency(attempts, fee, bankroll=1000.0, weeks_per_month=4.345):
    # `fee` here is the eval-fee-per-cycle (the deterministic per-attempt entry cost), NOT the
    # path-dependent attributable fee of §14/H1 — at the renewal level every cycle pays the eval
    # fee to start, so a scalar is correct here. (Activation, when present, is folded into the
    # per-attempt reward via net_payout upstream.)
    # expected income per month per $bankroll of fees — the headline renewal number.
    rate_per_week = r_renewal(attempts)  # net cashflow per calendar week
    return rate_per_week * weeks_per_month * (bankroll / fee)
```

The divergence between `R_renewal` and the distribution of `R_path` is a result, but a *specific* one: it measures **ratio-estimator / finite-horizon (Jensen) bias** — the only thing i.i.d. attempt draws can reveal. A material gap means the closed-form `R_renewal` should not be trusted as the finite-horizon rate. It does **not** speak to cross-cycle correlation, which i.i.d. resampling removes by construction; that source is a separately-stated, currently-uncaptured gap (`MODEL_RISKS.md` §H5) closed only by order-preserving sequence simulation. Reporting both rates plus the named correlation gap is the honest position — not "the two agree, therefore cycles are independent."

### 15.3 Finite-horizon cumulative cashflow

For practical bankroll planning the infinite-horizon rate is less useful than the distribution of **cumulative net cashflow over a finite horizon** (e.g. "over 6 months, on a $5,000 fee bankroll, what is the distribution of total take-home?"). This is `R_path`'s numerator sampled at a fixed `horizon`, retaining the full distribution rather than the rate — carrying convexity through to the renewal layer, exactly as §14 does for the single attempt.

---

## 16. Forward-compatibility: the bet-sizing optimizer (design note, not built)

The engine is structured so the optimizer drops in without restructuring. The pieces already reserved for it:

- **Sizing feeds back into the path, and the hook exists now.** Trade data is per-unit returns (§11); the kernel computes `size × ret` where `size = _size_policy(stage_mask, policy_params, size_base)` per trade (§16.1). Today `policy_params` is length-1 and reproduces a constant size; a policy widens it. The hook is already threaded through the engine so the optimizer needs no kernel change.
- **The policy is compiled, parameterized, and searched from outside.** Its *shape* is fixed in the kernel (e.g. a multiplier per stage — `size = size_base × size_mult[stage_bits]`), and its *parameters* are passed as an array exactly like rule `p0`. The kernel is compiled once; the optimizer proposes new parameter arrays, never new code. A policy conditioned on stages reads `STATE.STAGE_MASK` (§6) directly, and combinations of active stages are addressable because the mask is a bitmask.
- **Objectives are already post-processing.** The optimizer maximizes any function from §14 (single-attempt) or §15 (renewal) over the outcomes; swapping objective never touches the kernel. The right target is a renewal quantity: **maximize payout velocity / long-run reward rate subject to survival constraints**, not single-attempt expected payout. Formally the eventual objective is closer to

  ```
  max_π   E_π[net cashflow] / E_π[time]        (the renewal reward rate, §15)
  s.t.    P(catastrophic loss) < ε
          P(profitable attempt) > p_min
          P(drawdown breach)    < d_max
  ```

  Maximizing bare `E[payout]` is the wrong target for a convex, fee-per-attempt, repeated instrument: it is blind to convexity (accepts a lottery-ticket account), to velocity (accepts an account that ties the fee up for years), and to the renewal structure (ignores that fast failures recycle capital faster than slow ones). A distribution-, time-, and renewal-aware objective is what makes the optimizer's search well-posed.

The nesting is three layers: the kernel (one simulation), a policy evaluation (Monte Carlo over resampled sequences with a fixed policy → a distribution → a scalar objective — essentially today's `Engine.run`), and the optimizer (proposes parameters, reads the objective, iterates), which lives entirely in Python because it runs once per candidate, not once per trade. Two design consequences worth recording now: the objective is a *noisy* Monte Carlo estimate, so the optimizer should assume stochastic objectives (Bayesian optimization / CMA-ES / SPSA rather than exact-gradient methods); and comparing candidates should use **Common Random Numbers** — the same resample seeds across candidates *within a generator* — so the objective difference reflects the policy, not RNG noise. CRN is a paired-variance-reduction tool for comparing policies under one return-generating process; it should **not** be forced across fundamentally different generators (§ generator ladder in `MODEL_RISKS.md` §G1), where the goal is robustness across processes, not paired comparison. The hard drawdown barrier makes this a constrained problem (maximize the reward rate subject to bounded bust probability), not plain expected-value maximization.

### 16.1 The sizing hook — reserved now, policy and optimizer deferred

The one piece built *now* so the optimizer is a later drop-in rather than a kernel rewrite: **`size` is a compiled function of state and a policy-parameter array, not a constant.** The kernel takes a `policy_params` array (threaded through `Engine.run` → `simulate_batch` → `simulate_one_phase`) and computes `size = _size_policy(stage_mask, policy_params, size_base)` per trade. Today `policy_params` has length 1 and `_size_policy` reproduces the fixed-size case exactly; the optimizer later widens the array and the call site, downstream shapes, and outcome plumbing are all unchanged.

Why this specific shape (a **compiled, parameterized policy**) and not the alternatives:
- **Compiled parameterized policy (chosen).** Policy *shape* fixed in the kernel, *parameters* passed as data — the same pattern rules already use (fixed kernel behavior + `p0` data). Fast, handles discrete or continuous state, optimizer proposes parameter arrays without recompiling.
- **Precomputed size schedule (permitted internal optimization).** If the policy turns out to depend only on *discrete* stage bits, `_size_policy` may be implemented as an O(1) lookup into a per-stage size table precomputed from `policy_params` outside the loop. This is an implementation detail *behind the same interface* — allowed, not required.
- **Python callback `size = policy(state)` per trade (rejected).** Reintroduces per-trade Python dispatch in the hot loop — the exact overhead the compile-to-kernel design exists to remove, and catastrophic under an optimizer that re-runs the whole Monte Carlo thousands of times.
- **Post-hoc rescaling (rejected).** Simulating at unit size then multiplying outcomes by size is invalid: size feeds back into the *path* (it changes *when* the account breaches, not just the final number), and the path-dependent barriers make that feedback first-order.

Deliberately deferred (do **not** build now): the policy's actual functional form (how many stages, multiplier-per-stage vs continuous-in-state) — a modeling choice that interacts with over/under-fitting and belongs with the optimizer; the optimizer itself; and the market-impact model. **Dependency to remember:** the moment `size` varies, the size-invariance limit (`MODEL_RISKS.md` §G3) goes live — the kernel will faithfully simulate a policy whose larger sizes have fictional (impact-free) fills, so no optimized policy that materially scales size is trustworthy until the impact model exists. The hook may exist now; trusting its outputs waits on G3.

**What the length-1 baseline is — and is not.** The constant-size case applies *one scalar* `size_base` to the per-unit `ret` across all trades and all seven instruments. It is a **uniform-size baseline, not a replay of the historical backtest**: the real backtest had variable per-trade, per-instrument sizing, and a single scalar cannot size SIL differently from MES. So the Step 6 / Step 9 "reproduces the constant-size result exactly" parity tests check the kernel against *itself at uniform size*, which is the correct deterministic reference — they do **not**, and are not meant to, reproduce the backtest's actual P&L. Per-instrument or historical sizing is a policy (a wider `policy_params`), not the baseline.

---

## 17. Engine (`engine.py`)

`Engine.run()` is pure orchestration:

```
run(account, trades, config, policy_params=DEFAULT_CONST_SIZE)   # policy_params length-1 today (§16.1)
  ├─ validate(account)                            # §9, role-aware, before any compile
  ├─ trades = cache.preprocess(trades)            # once, cached by raw hash
  ├─ fp = fingerprint(account, program.version)   # §10
  ├─ compiled = cache.compile(account, fp)         # cached by fingerprint
  ├─ for phase in compiled.phases (sequential, survivors-only):
  │     idx = resampling.generate(config, trades)     # [B, L] day-block gather, §11.4
  │     simulate_batch(..., policy_params, ...) -> rich raw outcomes   # §12, batched for memory
  ├─ aggregate raw outcomes (codes, payouts, day-indices, size)
  └─ return Results(outcomes, fingerprint=fp)      # §13–14, lazy stats & objectives
```

---

## 18. Performance summary

| Technique | Where | Effect |
|---|---|---|
| Compile objects → int codes + arrays | `compiler.py` → `kernels.py` | Removes per-trade Python dispatch — the largest single gain |
| Predicate/action loop, no Python objects | `kernels.py` | Pass/fail/payout/breach all handled by integer codes in one compiled loop |
| `@njit(cache=True)` (no fastmath) | all kernels | LLVM native code; `cache` skips recompile between runs. `fastmath` is omitted so boundary comparisons stay exact for the §G6 oracle gate |
| `parallel=True` + `prange` over sims | `simulate_batch` | Independent sims → near-linear core scaling |
| Struct-of-arrays for rules | compiler output | Contiguous memory, SIMD-friendly, no object headers |
| Requirements-driven state | `compiler.py` | Counters (consistency, winning-days, intraday low) computed only when a rule needs them |
| Timing as int8 fields, not loop forks | `kernels.py` | Every firm's update/check timing runs in one compiled loop; no per-firm kernel |
| Day-block resampling | `resampling.py` | Preserves intraday-low and day-scoped counters; gather over day blocks |
| Survivors-only phase 2 | `engine.py` | Funded phase runs only on sims that passed eval |
| Preprocess-once + 3 caches | `cache.py` | Trade preprocessing, compiled accounts, compiled rules reused across the test matrix |
| Pre-generate resample indices | `resampling.py` | RNG index arrays built upstream; kernel only gathers |
| Keep rich raw outcomes, batch for memory | `simulate.py` | Exact quantiles, BCa, and any time-based objective stay available; memory bounded |
| Fingerprint = cache key | `fingerprint.py` | Identical configs are never recompiled; also gives reproducibility |
| Common Random Numbers (future) | optimizer | Same seeds across candidates → low-variance objective comparison |

Deferred without disturbing the model or compiler: regime-aware resampling (`resampling.py`), P² streaming quantiles (`statistics.py`), the bet-sizing optimizer (§15), GPU kernels.

---

## 19. Build order

This is the *implementation* sequencing. `BUILD_SPEC.md` gives the *test* sequencing (what each step must pass); the two cover the same work but group it slightly differently — notably this list splits batch-MC (step 9) from the statistics/engine wiring (step 10), whereas `BUILD_SPEC` Step 9 bundles the engine with batch-MC and puts statistics in Step 10. When they differ on grouping, `BUILD_SPEC`'s step boundaries are authoritative for "what must pass before proceeding," since the tests define done (§ the one invariant). The ordering and the gates (Level 0 frozen strategy, Level 1 oracle parity) are identical in both.

1. **`enums.py` + `model.py`** — the integer vocabulary (exit codes incl. `CAPPED_OUT`, state fields incl. the drawdown floor/lock and `DAY_LOW`, actions incl. `ADJUST`, severity, timing, stages) and the DSL tree. Test hashability and immutability.
2. **`rules.py`** — the starter rules across all four actions (ProfitTarget/MinDays as `PASS`, TrailingDD/DailyLoss as `FAIL`, MinWinningDays and ConsistencyGate as `PAYOUT`/`PASS` eligibility conjuncts, ConsistencyRaisesTarget as `ADJUST`), each carrying its timing fields, plus `RULE_REGISTRY`.
3. **`data.py`** — the dataset contract (§11): input-column validation, per-unit-return normalization, `trade_low` derivation from `mae`, session-boundary day indexing, and the per-day table. Get one real trade file in and a `TradeDataset` out before simulating, since every rule reads this.
4. **`config.py` + `validate.py`** — the three-layer table format, `scaled`/`build_accounts`, and the role-aware validator. One full firm expressed as pure config and passing `validate` before any simulation exists, forcing the model to hold the real taxonomy.
5. **`compiler.py`** — requirements resolver and struct-of-arrays emission (action, severity, timing, adjust-field, reset-mask).
6. **Single-path kernel + `reference.py`** — `simulate_one_phase` and the pure-Python oracle together, so the predicate/action semantics (timing-gated update/check, pass conjunction, hard/soft breach, DD lock, end-of-path close, payout+reset, target adjust, stage bits) are tested against each other exhaustively before parallelization. **Oracle equivalence is a hard gate (`MODEL_RISKS.md` §G6, Level 1 of the trust hierarchy):** **bitwise on the per-sim path** (each sim is an independent sequential accumulation with no within-sim reduction, so with `fastmath` off the kernel and reference agree bit-for-bit; a non-bitwise difference is a real bug), exact at the enumerated contract boundaries (equity exactly on the floor, breach-and-target same trade, payout exactly at cap/`cap_fraction`/`min_request`, floor-lock transition, soft-breach-then-EOD, first/last-trade breach, end-of-path close, closing-day pnl in its own EOD predicate, consistency gate withholds a payout/pass without failing, holding-interval-clipped `trade_low`). Boundary-exactness is why **`fastmath` is off** (C1). Capped-out is **not** on this list — deferred with the sizing policy (§16.1, `MODEL_RISKS.md` §C2/§A2). This gate precedes any Monte Carlo at scale.
7. **`fingerprint.py` + `cache.py`** — reproducibility locked early.
8. **`resampling.py`** — IID first, then the stationary bootstrap, resampling **whole days** (§11.4).
9. **`simulate_batch` (parallel) + `simulate.py`** — batching and Monte Carlo; assert the batch kernel matches the single-path oracle.
10. **`statistics.py` + `objectives.py` + `results.py` + `engine.py`** — full wiring. The decision-statistics layer (§14) turns rich raw outcomes into the distribution axis (`P(profitable)`, payout-count histogram, quantiles) and the time axis (payout velocity, time-to-first-payout, return-on-fee per year), with the mean as one number among them rather than the summary.
11. **`renewal.py`** — the renewal-economics analysis layer (§15) over completed attempts: `R_renewal`, `R_path`, fee-bankroll efficiency, finite-horizon cumulative cashflow. Pure Python over `Engine.run` outcomes; no kernel change.
12. **Additional rules and firms** — adding a firm is writing config and selecting existing rules; a genuinely new rule trips `RULE_REGISTRY`, which names the four places to add it.
13. **(Later) the generator ladder** — evaluate the same *frozen* strategy/contract/objective across IID → block → stationary → regime-conditioned → stochastic-vol resamplers; report every headline number as a **model-sensitivity band** across the ladder and the block-length range, and keep only edges that survive all of them (`MODEL_RISKS.md` §G1).
14. **(Later) the optimizer (§16)** — compiled parameterized sizing policy + an outer stochastic optimizer maximizing a renewal-aware objective (§15) under survival constraints, reusing everything above unchanged. **Mandatory: nested / walk-forward out-of-sample** — the optimizer must never see the sample used to establish the strategy's viability, or CRN just makes it more efficient at exploiting that sample's noise (`MODEL_RISKS.md` §G7).

Correctness precedes speed precedes scale, and **trust precedes compute**: the strategy must be frozen and non-overfit (Level 0, §G7) and the kernel must pass the oracle (Level 1, §G6) before any large Monte Carlo — a million simulations of an unproven kernel, or an optimizer exploiting one overfit sample, only produce a more precise wrong answer. The single-path kernel and its reference oracle existing before Monte Carlo prevent debugging a parallel kernel and a rule bug simultaneously.
