"""The compiler — DSL objects → numeric arrays the kernel consumes (ARCHITECTURE
§6b, §8; BUILD_SPEC Step 5).

Two jobs, both pure data transforms:

* **The requirements resolver** (:func:`resolve_requirements`): the union of the
  ``StateField``s every rule in a phase reads, plus the always-on driving fields
  (``EQUITY``, ``DAY_INDEX``). ``StateField`` is *semantic*, not nominal, so the
  union is always safe — two rules naming the same field mean the same thing, and
  a rule that never appears never widens the live set ("only compute what is
  required": a phase with no consistency rule allocates no ``MAX_DAY_PNL``).
* **Struct-of-arrays emission** (:func:`compile_phase`): each rule lowers to one
  aligned column across parallel NumPy arrays (``kind``, ``p0``, ``p1``,
  ``action``, ``severity``, the two timing axes, ``adjust_field``, ``fail_code``),
  **preserving rule order** so the kernel's fail-precedence (§6) is well-defined.
  Per-action index arrays are provided so the kernel can iterate fail-predicates
  and conjoin pass/payout-predicates without re-scanning for the action each trade.
* The funded **payout schema** lowers to :class:`CompiledPayoutSchema` — the
  ``dollar_cap`` tuple as an array, the fire-gate scalars, a *uniform* split
  representation (a tier that is inert when the firm has none), the post-payout
  transition flags, and the per-payout reset set.

The compiled layout is a free optimization surface (BUILD_SPEC Step 5 Freedom);
the only contract is that the resolved requirements are correct and the schema
round-trips faithfully. ``int8`` holds every code (``StateField`` ≤ 16,
``ExitCode`` ≤ 22, ``RuleKind`` ≤ 8, actions/severity/timing tiny); ``-1`` is the
inert sentinel for a missing ``adjust_field``/``fail_code``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .enums import Action, StateField
from .rules import NEVER_LOCK, RuleKind
from .schema import PayoutSchema

#: Fields the loop always needs, independent of the rules present (§8).
_DRIVING_FIELDS = frozenset({StateField.EQUITY, StateField.DAY_INDEX})

#: int8 sentinel for an inapplicable adjust_field / fail_code column.
NONE_CODE = -1

#: Scalar seed for a phase that has no profit-target / no trailing-DD rule: an
#: infinite target/amount is one the account can never reach, so the corresponding
#: kernel check is inert without a special-case flag (mirrors NEVER_LOCK).
_UNREACHABLE = float("inf")


def resolve_requirements(phase) -> frozenset[StateField]:
    """The live ``StateField`` set for a phase: the union of its rules'
    requirements plus the always-on driving fields (§8). Order-independent and
    de-duplicated by construction (it is a set)."""
    needed: set[StateField] = set(_DRIVING_FIELDS)
    for rule in phase.rules:
        needed.update(rule.requirements())
    return frozenset(needed)


@dataclass(frozen=True)
class CompiledPayoutSchema:
    """A funded phase's :class:`PayoutSchema` in kernel-ready form (§6b).

    ``split`` is represented *uniformly* as a two-tier rule so the kernel's net
    computation never branches on "does this firm have a legacy tier": while
    ``cumulative_paid < tier_cap`` the trader keeps ``first_tier_split``, else
    ``split``. A firm with no tier compiles to ``tier_cap = 0`` (so the first tier
    never applies) and ``first_tier_split = split`` (so it would not matter if it
    did) — the arithmetic is identical to a flat split.
    """

    dollar_cap: np.ndarray  # float64[k]; last element repeats for later payouts
    cap_fraction: float
    min_request: float
    buffer_floor: float
    split: float
    first_tier_split: float
    tier_cap: float
    max_payouts: int
    reset_fields: np.ndarray  # int8[*] — StateField codes zeroed each payout (§6b.1)
    withdraw_reduces_equity: bool
    recompute_floor_on_payout: bool
    resets_qualifying_days: bool  # precomputed: is N_QUALIFYING_DAYS in reset_fields?
    #                               (hoisted out of the per-attempt hot loop)

    def dollar_cap_at(self, payout_index: int) -> float:
        """Per-request ceiling for the ``payout_index``-th payout; last repeats."""
        if payout_index < 0:
            raise IndexError("payout_index must be >= 0")
        i = min(payout_index, self.dollar_cap.shape[0] - 1)
        return float(self.dollar_cap[i])


@dataclass(frozen=True)
class CompiledPhase:
    """A phase lowered to the arrays and live-state set the kernel reads."""

    role: str
    live_state: frozenset[StateField]
    n_rules: int
    # pre-extracted scalar seeds the §12 kernel takes directly, so it need not
    # re-scan the arrays each attempt. Each is _UNREACHABLE when the phase lacks
    # the rule (the corresponding check is then inert). The one-trailing-DD-per-
    # phase limit (§8, enforced by validate) makes dd_amount/lock_at unambiguous.
    profit_target0: float  # seed for PROFIT_TARGET (ProfitTargetRule.p0)
    dd_amount: float  # trailing-DD amount (TrailingDrawdownRule.p0)
    lock_at: float  # trailing-DD lock level (TrailingDrawdownRule.p1; inf = never)
    dd_update_timing: int  # when the trailing floor ratchets (Timing; CONTINUOUS if none)
    winning_day_threshold: float  # a day counts as "winning" at/above this (inf if none)
    # struct-of-arrays, one aligned entry per rule, in rule order (precedence!)
    kind: np.ndarray  # int8[R]
    p0: np.ndarray  # float64[R]
    p1: np.ndarray  # float64[R]
    action: np.ndarray  # int8[R]
    severity: np.ndarray  # int8[R]
    update_timing: np.ndarray  # int8[R]
    check_timing: np.ndarray  # int8[R]
    adjust_field: np.ndarray  # int8[R], NONE_CODE where inapplicable
    fail_code: np.ndarray  # int8[R], NONE_CODE where inapplicable
    # per-action index groupings into the arrays above, order-preserving
    fail_idx: np.ndarray  # int32[*]
    pass_idx: np.ndarray  # int32[*]
    payout_idx: np.ndarray  # int32[*]
    adjust_idx: np.ndarray  # int32[*]
    payout: CompiledPayoutSchema | None  # funded phases only


@dataclass(frozen=True)
class CompiledAccount:
    """An account lowered for simulation: metadata plus its compiled phases."""

    name: str
    size: int
    eval_fee: float
    activation_fee: float
    currency: str
    phases: tuple[CompiledPhase, ...]


def _compile_payout_schema(schema: PayoutSchema) -> CompiledPayoutSchema:
    has_tier = schema.split_first_tier is not None
    return CompiledPayoutSchema(
        dollar_cap=np.asarray(schema.dollar_cap, dtype=np.float64),
        cap_fraction=float(schema.cap_fraction),
        min_request=float(schema.min_request),
        buffer_floor=float(schema.buffer_floor),
        split=float(schema.split),
        # uniform tier: inert (tier_cap=0, first_tier_split=split) when no legacy tier
        first_tier_split=float(schema.split_first_tier if has_tier else schema.split),
        tier_cap=float(schema.split_tier_cap if has_tier else 0.0),
        max_payouts=int(schema.max_payouts),
        reset_fields=np.asarray([int(f) for f in schema.reset_fields], dtype=np.int8),
        withdraw_reduces_equity=bool(schema.withdraw_reduces_equity),
        recompute_floor_on_payout=bool(schema.recompute_floor_on_payout),
        resets_qualifying_days=int(StateField.N_QUALIFYING_DAYS)
        in [int(f) for f in schema.reset_fields],
    )


def compile_phase(phase) -> CompiledPhase:
    """Lower one phase to its :class:`CompiledPhase` (§8)."""
    compiled = [r.compile() for r in phase.rules]
    r = len(compiled)

    kind = np.empty(r, dtype=np.int8)
    p0 = np.empty(r, dtype=np.float64)
    p1 = np.empty(r, dtype=np.float64)
    action = np.empty(r, dtype=np.int8)
    severity = np.empty(r, dtype=np.int8)
    update_timing = np.empty(r, dtype=np.int8)
    check_timing = np.empty(r, dtype=np.int8)
    adjust_field = np.empty(r, dtype=np.int8)
    fail_code = np.empty(r, dtype=np.int8)

    # scalar seeds the kernel takes directly (§12); inert-by-default when absent
    profit_target0 = _UNREACHABLE
    dd_amount = _UNREACHABLE
    lock_at = NEVER_LOCK
    dd_update_timing = 0  # Timing.CONTINUOUS; irrelevant when no trailing DD
    winning_day_threshold = _UNREACHABLE  # no day qualifies when there is no rule

    for i, c in enumerate(compiled):
        kind[i] = int(c.kind)
        p0[i] = float(c.p0)
        p1[i] = float(c.p1)
        action[i] = int(c.action)
        severity[i] = int(c.severity)
        update_timing[i] = int(c.update_timing)
        check_timing[i] = int(c.check_timing)
        adjust_field[i] = NONE_CODE if c.adjust_field is None else int(c.adjust_field)
        fail_code[i] = NONE_CODE if c.fail_code is None else int(c.fail_code)
        if c.kind == RuleKind.PROFIT_TARGET:
            profit_target0 = float(c.p0)
        elif c.kind == RuleKind.TRAILING_DD:
            dd_amount = float(c.p0)
            lock_at = float(c.p1)
            dd_update_timing = int(c.update_timing)
        elif c.kind == RuleKind.MIN_WINNING_DAYS:
            winning_day_threshold = float(c.p1)

    # per-action index groupings (order-preserving), from the ordered `action` array
    idx = {a: [] for a in ("fail", "pass", "payout", "adjust")}
    label = {
        int(Action.FAIL): "fail",
        int(Action.PASS): "pass",
        int(Action.PAYOUT): "payout",
        int(Action.ADJUST): "adjust",
    }
    for i in range(r):
        idx[label[int(action[i])]].append(i)

    payout = (
        _compile_payout_schema(phase.payout_schema)
        if phase.payout_schema is not None
        else None
    )

    return CompiledPhase(
        role=phase.role,
        live_state=resolve_requirements(phase),
        n_rules=r,
        profit_target0=profit_target0,
        dd_amount=dd_amount,
        lock_at=lock_at,
        dd_update_timing=dd_update_timing,
        winning_day_threshold=winning_day_threshold,
        kind=kind,
        p0=p0,
        p1=p1,
        action=action,
        severity=severity,
        update_timing=update_timing,
        check_timing=check_timing,
        adjust_field=adjust_field,
        fail_code=fail_code,
        fail_idx=np.asarray(idx["fail"], dtype=np.int32),
        pass_idx=np.asarray(idx["pass"], dtype=np.int32),
        payout_idx=np.asarray(idx["payout"], dtype=np.int32),
        adjust_idx=np.asarray(idx["adjust"], dtype=np.int32),
        payout=payout,
    )


def compile_account(account) -> CompiledAccount:
    """Lower a whole account to a :class:`CompiledAccount` (§8)."""
    return CompiledAccount(
        name=account.name,
        size=account.size,
        eval_fee=float(account.eval_fee),
        activation_fee=float(account.activation_fee),
        currency=account.currency,
        phases=tuple(compile_phase(ph) for ph in account.phases),
    )


__all__ = [
    "resolve_requirements",
    "CompiledPayoutSchema",
    "CompiledPhase",
    "CompiledAccount",
    "compile_phase",
    "compile_account",
    "NONE_CODE",
]
