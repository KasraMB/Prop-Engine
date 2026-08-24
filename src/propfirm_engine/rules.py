"""Rules and the rule registry (ARCHITECTURE §5, §6, §6a; BUILD_SPEC Step 2).

A rule is a *predicate plus an action*. The base (:class:`Rule`) declares two
things and no shared behaviour: what per-simulation state the rule reads
(:meth:`Rule.requirements`) and how it lowers to a numeric record the kernel
consumes (:meth:`Rule.compile`). All actual rule behaviour lives in the kernel,
keyed by :class:`RuleKind` — never in the class hierarchy — so every concrete
rule is exactly one level deep (§5).

The compiled record (:class:`CompiledRule`) carries everything the kernel needs
to run a rule *without* seeing a Python object: its kind, up to two numeric
parameters, its action, its severity, its two orthogonal timing axes (§6a), the
state field an ``ADJUST`` mutates, and the failure code a ``FAIL`` emits. Fields
that are irrelevant to a given rule take documented inert defaults (timing
defaults to ``CONTINUOUS``; ``fail_code``/``adjust_field`` stay ``None``).

The :data:`RULE_REGISTRY` maps every rule *kind* the kernel implements to its
:class:`Rule` subclass. Because the config floor (§7) accepts any rule object,
:func:`assert_kernel_supports` is the guarantee that a config cannot name a rule
the kernel cannot run: it hard-fails on an unregistered kind.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

from .enums import Action, ExitCode, Severity, StateField, Timing

#: Sentinel stored in ``CompiledRule.p1`` when a trailing floor never locks
#: (``lock_at is None``): an infinite lock level is one the floor can never
#: reach, so the "stop trailing" branch is inert without a special-case flag.
NEVER_LOCK = math.inf


class RuleKind(IntEnum):
    """The integer tag the kernel switches on to run a rule (§5, §12).

    One member per implemented mechanic. Adding a rule is always four steps
    (§5): a :class:`Rule` subclass, a kind here, a kernel branch, and a
    :data:`RULE_REGISTRY` entry — and :func:`assert_kernel_supports` forces the
    last one to exist before a config naming the rule can be simulated."""

    PROFIT_TARGET = 0
    TRAILING_DD = 1
    STATIC_DD = 2
    DAILY_LOSS = 3
    MIN_DAYS = 4
    MIN_WINNING_DAYS = 5
    # 6 retired: consistency never fails, so there is no plain FAIL consistency kind
    CONSISTENCY_ADJUST = 7
    CONSISTENCY_GATE = 8


@dataclass(frozen=True)
class CompiledRule:
    """The numeric record a rule lowers to (§8).

    Frozen and hashable by value so it feeds the structural fingerprint (§10)
    and so two rules that compile identically compare equal. In the kernel this
    record becomes one aligned column across parallel arrays (``kind[]``,
    ``p0[]``, ...); here it is a single object for testability. Every field the
    struct-of-arrays carries is present, with inert defaults where a field does
    not apply to a rule:

    - ``p0``/``p1`` — up to two numeric parameters, meaning is per-kind.
    - ``severity`` — meaningful only for ``Action.FAIL`` (§Severity).
    - ``update_timing``/``check_timing`` — the two orthogonal axes of §6a;
      both default to ``CONTINUOUS`` for rules where timing is irrelevant.
    - ``adjust_field`` — the ``StateField`` an ``Action.ADJUST`` mutates; ``None``
      for every other action.
    - ``fail_code`` — the :class:`ExitCode` an ``Action.FAIL`` emits; ``None``
      for every other action.
    """

    kind: RuleKind
    action: Action
    p0: float = 0.0
    p1: float = 0.0
    severity: Severity = Severity.HARD
    update_timing: Timing = Timing.CONTINUOUS
    check_timing: Timing = Timing.CONTINUOUS
    adjust_field: StateField | None = None
    fail_code: ExitCode | None = None


@dataclass(frozen=True)
class Rule(ABC):
    """Abstract base for every rule. Frozen and hashable by value, so any DSL
    object containing rules is itself hashable (required for the fingerprint,
    §10). ``@abstractmethod`` makes a half-finished rule fail at instantiation."""

    @abstractmethod
    def requirements(self) -> tuple[StateField, ...]:
        """The ``StateField``s this rule's predicate reads. The compiler unions
        these across a phase to decide which state is live (§8)."""
        raise NotImplementedError

    @abstractmethod
    def compile(self) -> CompiledRule:
        """Lower this rule to its numeric :class:`CompiledRule` record."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Fail-predicates (disjunctive): any one triggering ends the phase; Severity   #
# decides hard (terminate) vs soft (truncate the day, resume next). (§6)       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrailingDrawdownRule(Rule):
    """A floor that trails the peak of equity and terminates on breach (§6a).

    ``update_timing`` decides when the floor ratchets up (``EOD`` = off the day's
    closing equity); ``check_timing`` decides when a breach is tested
    (``CONTINUOUS`` = intraday against the day's low-water mark). The two are
    orthogonal, which is why they are separate fields (§6a). ``lock_at`` freezes
    the floor once it reaches that level (a near-universal companion mechanic);
    ``None`` means it never locks."""

    amount: float
    update_timing: Timing = Timing.CONTINUOUS
    check_timing: Timing = Timing.CONTINUOUS
    lock_at: float | None = None

    def requirements(self) -> tuple[StateField, ...]:
        need = [StateField.EQUITY, StateField.PEAK_EQUITY, StateField.DD_FLOOR]
        if self.check_timing == Timing.CONTINUOUS:
            need.append(StateField.DAY_LOW)  # intraday low-water mark
        if self.lock_at is not None:
            need.append(StateField.DD_LOCKED)
        return tuple(need)

    def compile(self) -> CompiledRule:
        return CompiledRule(
            kind=RuleKind.TRAILING_DD,
            p0=self.amount,
            p1=(self.lock_at if self.lock_at is not None else NEVER_LOCK),
            action=Action.FAIL,
            severity=Severity.HARD,
            update_timing=self.update_timing,
            check_timing=self.check_timing,
            fail_code=ExitCode.FAIL_TRAILING_DD,
        )


@dataclass(frozen=True)
class StaticDrawdownRule(Rule):
    """A fixed floor at ``start_equity − amount`` that never trails (§8).

    Unlike trailing drawdown it stores no reference point — the kernel recomputes
    the floor inline from the account's starting equity — so it needs no
    ``PEAK_EQUITY``/``DD_FLOOR``/``DD_LOCKED`` state and carries no
    ``update_timing`` (there is nothing to advance). ``check_timing`` still
    applies: ``CONTINUOUS`` tests the breach intraday, ``EOD`` only at close."""

    amount: float
    severity: Severity = Severity.HARD
    check_timing: Timing = Timing.CONTINUOUS

    def requirements(self) -> tuple[StateField, ...]:
        need = [StateField.EQUITY]
        if self.check_timing == Timing.CONTINUOUS:
            need.append(StateField.DAY_LOW)
        return tuple(need)

    def compile(self) -> CompiledRule:
        return CompiledRule(
            kind=RuleKind.STATIC_DD,
            p0=self.amount,
            action=Action.FAIL,
            severity=self.severity,
            check_timing=self.check_timing,
            fail_code=ExitCode.FAIL_STATIC_DD,
        )


@dataclass(frozen=True)
class DailyLossRule(Rule):
    """A cap on the loss taken within a single trading day (§5).

    Soft at most firms — a breach truncates the day and the account resumes next
    day — but the severity is overridable per account. Timing is irrelevant to
    its definition (it reads the day's running P&L), so both axes default to
    ``CONTINUOUS``."""

    amount: float
    severity: Severity = Severity.SOFT

    def requirements(self) -> tuple[StateField, ...]:
        return (StateField.DAY_PNL,)

    def compile(self) -> CompiledRule:
        return CompiledRule(
            kind=RuleKind.DAILY_LOSS,
            p0=self.amount,
            action=Action.FAIL,
            severity=self.severity,
            fail_code=ExitCode.FAIL_DAILY_LOSS,
        )


# --------------------------------------------------------------------------- #
# Pass-predicates (conjunctive): the phase clears only when ALL hold. (§6)     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProfitTargetRule(Rule):
    """A pass-predicate satisfied when equity reaches the target (§5).

    It reads the *live* ``PROFIT_TARGET`` state field rather than a static
    constant so an ``ADJUST`` rule can raise the target while this predicate
    stays a pure read (§6). It is one pass-predicate among possibly several —
    never a hardcoded special case — so "target AND ≥N days" is pure config."""

    target: float

    def requirements(self) -> tuple[StateField, ...]:
        return (StateField.EQUITY,)

    def compile(self) -> CompiledRule:
        return CompiledRule(
            kind=RuleKind.PROFIT_TARGET, p0=self.target, action=Action.PASS
        )


@dataclass(frozen=True)
class MinimumTradingDaysRule(Rule):
    """A pass-*gate*: the phase cannot clear until enough trading days elapse (§5).

    This is why pass must be conjunctive — an account that hits the profit target
    on day 1 with ``minimum_days=3`` must keep running; the phase clears only when
    this predicate is *also* satisfied. A shortfall surfaces as ``TIMED_OUT``, not
    a failure code (it withholds ``PASSED`` rather than failing)."""

    minimum_days: int

    def requirements(self) -> tuple[StateField, ...]:
        return (StateField.N_TRADING_DAYS,)

    def compile(self) -> CompiledRule:
        return CompiledRule(
            kind=RuleKind.MIN_DAYS, p0=self.minimum_days, action=Action.PASS
        )


# --------------------------------------------------------------------------- #
# Payout-predicates (conjunctive, repeatable): fire, reset counters, continue. #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MinimumWinningDaysRule(Rule):
    """A payout-predicate: N winning days each clearing a profit threshold (§5).

    The repeatable analogue of a pass — when the conjunction holds a payout fires,
    the configured counters reset, and the account continues. ``count`` and
    ``threshold`` are the two parameters (e.g. 5 days each ≥ \\$150)."""

    count: int
    threshold: float

    def requirements(self) -> tuple[StateField, ...]:
        return (StateField.N_QUALIFYING_DAYS,)

    def compile(self) -> CompiledRule:
        return CompiledRule(
            kind=RuleKind.MIN_WINNING_DAYS,
            p0=self.count,
            p1=self.threshold,
            action=Action.PAYOUT,
        )


# --------------------------------------------------------------------------- #
# Consistency: one whole-day predicate (max_day / profit <= threshold) worn as #
# two actions — ADJUST (ConsistencyRaisesTargetRule) and the PASS/PAYOUT gate  #
# (ConsistencyGateRule) our firms actually use. Consistency never *fails* an    #
# account, so there is no FAIL variant; ExitCode.FAIL_CONSISTENCY is reserved   #
# but unused, for a hypothetical fail-on-consistency firm (MODEL_RISKS §C8).    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConsistencyRaisesTargetRule(Rule):
    """The consistency predicate with an ``ADJUST`` action: instead of failing the
    trader it *raises the profit target* (§5).

    Same whole-day property, hence the same EOD default. This is a distinct rule
    kind (not a parameter of :class:`ConsistencyGateRule`) so the config picks the
    behaviour — gate vs adjust — explicitly. The fire-action rewrites
    ``PROFIT_TARGET`` (named in ``adjust_field``); the target pass-predicate
    stays a pure read of that live field, so purity is preserved (§6)."""

    threshold: float
    raise_to: float
    check_timing: Timing = Timing.EOD

    def requirements(self) -> tuple[StateField, ...]:
        return (
            StateField.DAY_PNL,
            StateField.TOTAL_PNL,
            StateField.MAX_DAY_PNL,
            StateField.PROFIT_TARGET,
        )

    def compile(self) -> CompiledRule:
        return CompiledRule(
            kind=RuleKind.CONSISTENCY_ADJUST,
            p0=self.threshold,
            p1=self.raise_to,
            action=Action.ADJUST,
            check_timing=self.check_timing,
            adjust_field=StateField.PROFIT_TARGET,
        )


@dataclass(frozen=True)
class ConsistencyGateRule(Rule):
    """Consistency as a pure *gate* — the mechanic real firms (e.g. Lucid) use.

    The same whole-day consistency predicate as :class:`ConsistencyRaisesTargetRule`,
    but it neither fails nor adjusts: it simply **withholds** a success until the ratio clears.
    ``gate`` picks *which* success it gates — ``Action.PASS`` (eval: the phase
    will not clear) or ``Action.PAYOUT`` (funded: the payout will not fire) —
    and that is the *only* thing that varies between the eval and funded uses.
    Because it only ever withholds, it carries **no severity and no fail code**,
    and it needs no ``activate_above`` gate: a spuriously high ratio early on
    just means "not yet — keep trading" (exactly the firms' own wording), never
    a spurious kill.

    **Cycle-scoped denominator.** Profit is ``EQUITY − CYCLE_START_EQUITY``, which
    generalizes for free: in eval there are no payouts so the cycle spans the whole
    phase (ratio over total phase profit); in funded ``CYCLE_START_EQUITY`` and
    ``MAX_DAY_PNL`` reset on each payout (§6b.1 ``reset_fields``), so the ratio is
    automatically "since the last payout" — the firms' "resets after each payout".

    **This gate subsumes the ADJUST variant.** Clearing the gate requires
    ``total_profit ≥ max_day / threshold``, i.e. the gate *implicitly* raises the
    effective target to ``max_day / threshold`` — a value that floats with the
    offending day, so it is a strictly more general (dynamic) version of what
    :class:`ConsistencyRaisesTargetRule`'s fixed ``raise_to`` does.

    **Timing is inert for this rule** (MODEL_RISKS §C8). The gate is never
    evaluated on its own timing axis — it is only ever *read inside the payout/
    pass conjunction*, at whatever point that conjunction is evaluated (which
    §C9's fold-then-evaluate already fixes to the day close). So its
    ``check_timing`` carries no semantic choice: EOD and CONTINUOUS must produce
    identical behavior because the gate is only consulted when the payout/pass
    predicate is. It therefore exposes **no timing knob** and compiles to the
    default ``CONTINUOUS`` — the field is present on the record only because the
    struct-of-arrays is uniform, and nothing may branch on its value here.

    ``cushion`` is Lucid Flex's small allowance (ARCHITECTURE §5): the effective
    bound is slightly looser than the nominal ``threshold``, computed on actual
    daily profit, and is a per-firm calibration — not a new mechanic. It rides in
    the compiled record's ``p1``; ``0.0`` means no cushion."""

    threshold: float
    gate: Action = Action.PAYOUT  # PAYOUT (funded) or PASS (eval) — which event it gates
    cushion: float = 0.0  # Flex allowance; effective bound looser than threshold (rides in p1)

    def __post_init__(self) -> None:
        if self.gate not in (Action.PASS, Action.PAYOUT):
            raise ValueError(
                f"ConsistencyGateRule.gate must be Action.PASS or Action.PAYOUT, "
                f"not {self.gate!r} — a consistency gate only ever withholds a "
                f"success, it never fails or adjusts."
            )

    def requirements(self) -> tuple[StateField, ...]:
        # Numerator MAX_DAY_PNL; denominator is cycle profit = EQUITY - CYCLE_START.
        return (
            StateField.MAX_DAY_PNL,
            StateField.EQUITY,
            StateField.CYCLE_START_EQUITY,
        )

    def compile(self) -> CompiledRule:
        # No check_timing passed: timing is inert here (see docstring), so the
        # record takes the default CONTINUOUS and nothing branches on it. The
        # cushion rides in p1 (ARCHITECTURE §5).
        return CompiledRule(
            kind=RuleKind.CONSISTENCY_GATE,
            p0=self.threshold,
            p1=self.cushion,
            action=self.gate,
        )


# --------------------------------------------------------------------------- #
# The rule registry (§5): every kind the kernel implements. A rule class not   #
# in here CANNOT be simulated, and the compiler hard-fails on its kind.        #
# --------------------------------------------------------------------------- #

RULE_REGISTRY: dict[RuleKind, type[Rule]] = {
    RuleKind.PROFIT_TARGET: ProfitTargetRule,
    RuleKind.TRAILING_DD: TrailingDrawdownRule,
    RuleKind.STATIC_DD: StaticDrawdownRule,
    RuleKind.DAILY_LOSS: DailyLossRule,
    RuleKind.MIN_DAYS: MinimumTradingDaysRule,
    RuleKind.MIN_WINNING_DAYS: MinimumWinningDaysRule,
    RuleKind.CONSISTENCY_ADJUST: ConsistencyRaisesTargetRule,
    RuleKind.CONSISTENCY_GATE: ConsistencyGateRule,
}


class UnknownRuleError(Exception):
    """Raised when a config names a rule whose kind has no kernel implementation."""


def assert_kernel_supports(kind: int) -> None:
    """Guarantee the kernel can run a rule of ``kind``; raise otherwise (§5).

    This is the safety net for the permissive config floor (§7): because a config
    can name any rule object, this check is what makes an unimplemented rule fail
    loudly at validation rather than silently at simulation."""
    if kind not in RULE_REGISTRY:
        raise UnknownRuleError(
            f"rule kind {kind!r} has no kernel implementation. "
            f"Add: (1) a Rule subclass, (2) a RuleKind code, (3) a kernel branch, "
            f"(4) a RULE_REGISTRY entry."
        )


__all__ = [
    "NEVER_LOCK",
    "RuleKind",
    "CompiledRule",
    "Rule",
    "TrailingDrawdownRule",
    "StaticDrawdownRule",
    "DailyLossRule",
    "ProfitTargetRule",
    "MinimumTradingDaysRule",
    "MinimumWinningDaysRule",
    "ConsistencyRaisesTargetRule",
    "ConsistencyGateRule",
    "RULE_REGISTRY",
    "UnknownRuleError",
    "assert_kernel_supports",
]
