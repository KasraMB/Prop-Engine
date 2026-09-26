"""Sanity and execution-capability checks before account compilation.

Reject invalid configuration AND structures the current engine would silently
discard or collapse. These limits describe this implementation, not universal
firm rules: currently one eval, one funded, or eval followed by funded. Relax a
capability guard only alongside implementation and independent acceptance tests.

Passing validation does not certify firm fidelity, input-path sufficiency, or
all rule semantics. See docs/BRACKET_BACKTEST.md for remaining limitations.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

from .enums import Action, ExitCode, Severity, StateField, Timing
from .rules import RuleKind, assert_kernel_supports


# compile_phase hoists these kinds to ONE state seed/threshold per phase.
# Other kinds retain their own parameters; do not ban repetition generally.
_SINGLETON_STATE_KINDS = frozenset(
    {RuleKind.PROFIT_TARGET, RuleKind.TRAILING_DD, RuleKind.MIN_WINNING_DAYS}
)

#: The state the kernel actually produces (§9). Any rule requiring something
#: outside this set is a bug — a rule cannot read state nothing computes. This is
#: the full :class:`StateField` set today; it is spelled out (not ``set(StateField)``)
#: so that adding a *reserved-but-not-yet-produced* field later does not silently
#: satisfy the guard.
KERNEL_PRODUCED_STATE: frozenset[StateField] = frozenset(
    {
        StateField.EQUITY,
        StateField.PEAK_EQUITY,
        StateField.DD_FLOOR,
        StateField.DD_LOCKED,
        StateField.DAY_LOW,
        StateField.DAY_PNL,
        StateField.TOTAL_PNL,
        StateField.DAY_INDEX,
        StateField.N_TRADING_DAYS,
        StateField.MAX_DAY_PNL,
        StateField.N_QUALIFYING_DAYS,
        StateField.PAYOUTS_TAKEN,
        StateField.N_SOFT_BREACHES,
        StateField.STAGE_MASK,
        StateField.PROFIT_TARGET,
        StateField.CYCLE_START_EQUITY,
        StateField.CUMULATIVE_PAID,
    }
)


class InvalidAccountError(Exception):
    """Raised for an invalid account or an unsupported execution structure."""


def validate(account) -> None:
    """Assert the sanity invariants of §9 on an assembled account; raise on any.

    Accepts supported irregularity (including direct-funded and eval-only
    accounts); rejects broken or unrepresentable execution structures.
    """
    if not account.phases:
        raise InvalidAccountError(f"{account.name}: no phases")
    if isinstance(account.size, bool) or not isinstance(account.size, Integral) or account.size <= 0:
        raise InvalidAccountError(f"{account.name}: size must be a positive integer")
    for name in ("eval_fee", "activation_fee"):
        _number(getattr(account, name), f"{account.name}.{name}")

    _assert_supported_lifecycle(account)

    for ph in account.phases:
        opening = ph.start_equity
        if opening is not None and (isinstance(opening, bool)
                                    or not isinstance(opening, Real)
                                    or not math.isfinite(opening)):
            raise InvalidAccountError(f"{account.name}/{ph.name}.start_equity must be finite")
        if not ph.rules:
            raise InvalidAccountError(f"{account.name}/{ph.name}: no rules")

        scalar_kinds_seen = set()
        for r in ph.rules:
            compiled = r.compile()
            if isinstance(compiled.kind, bool) or not isinstance(compiled.kind, Integral):
                raise InvalidAccountError(f"{account.name}/{ph.name}: kind must be an integer RuleKind code")
            assert_kernel_supports(compiled.kind)  # the kernel can run it (§5)
            prefix = f"{account.name}/{ph.name}: {RuleKind(compiled.kind).name}"
            _number(compiled.p0, f"{prefix}.p0")
            _number(compiled.p1, f"{prefix}.p1", allow_inf=compiled.kind == RuleKind.TRAILING_DD)
            _validate_compiled_rule(compiled, prefix)
            for sf in r.requirements():  # it needs only producible state
                if sf not in KERNEL_PRODUCED_STATE:
                    raise InvalidAccountError(
                        f"{account.name}/{ph.name}: {type(r).__name__} needs {sf!r}, "
                        f"which the kernel does not produce"
                    )
            _assert_non_negative_params(account, ph, r)
            if compiled.kind in _SINGLETON_STATE_KINDS:
                if compiled.kind in scalar_kinds_seen:
                    kind_name = RuleKind(compiled.kind).name
                    raise InvalidAccountError(
                        f"{account.name}/{ph.name}: multiple {kind_name} rules are "
                        "unsupported: the compiler/kernel shares one state scalar "
                        "for this kind per phase"
                    )
                scalar_kinds_seen.add(compiled.kind)
            if ph.role == "funded" and compiled.action == Action.PASS:
                raise InvalidAccountError(
                    f"{account.name}/{ph.name}: funded phase has unsupported PASS "
                    "action; funded phases support payouts, not pass transitions"
                )
            if ph.role == "eval" and compiled.action == Action.PAYOUT:
                raise InvalidAccountError(
                    f"{account.name}/{ph.name}: eval phase has unsupported PAYOUT "
                    "action; payout processing is supported only in funded phases"
                )

        # The payout schema belongs on the funded phase (§6b). A funded phase with
        # a PAYOUT-action rule needs a schema to size that payout, or the compiler
        # (Step 5) has a predicate with nothing to compile against; a non-funded
        # phase must not carry one at all (a swallowed authoring mistake otherwise).
        if ph.role == "funded":
            _validate_funded_payouts(account, ph)
        elif ph.payout_schema is not None:
            raise InvalidAccountError(
                f"{account.name}/{ph.name}: a non-funded phase carries a "
                f"payout_schema — it belongs on the funded phase (§6b)"
            )

        _assert_terminable(account, ph)


def _assert_supported_lifecycle(account) -> None:
    """Engine.prepare selects one phase per role and always runs eval first."""
    roles = tuple(ph.role for ph in account.phases)
    for ph in account.phases:
        if ph.role not in ("eval", "funded"):
            raise InvalidAccountError(
                f"{account.name}/{ph.name}: unsupported phase role {ph.role!r}; "
                "the current engine supports only 'eval' and 'funded'"
            )
    for role in ("eval", "funded"):
        if roles.count(role) > 1:
            raise InvalidAccountError(
                f"{account.name}: multiple {role} phases are unsupported; "
                "the current engine executes at most one phase per role"
            )
    if roles == ("funded", "eval"):
        raise InvalidAccountError(
            f"{account.name}: unsupported phase order {roles!r}; "
            "evaluation must precede funding"
        )


def _assert_non_negative_params(account, ph, rule) -> None:
    """Coarse floor: reject any negative numeric rule parameter (§9).

    Deliberately blunt — it will wrongly reject the first legitimately-signed
    parameter a future rule needs, at which point it is tightened to per-field
    bounds. Adequate for the current rule set, all of whose parameters are
    non-negative. Enum-valued fields (timing/severity/gate) are ``IntEnum`` with
    non-negative values, so they pass.
    """
    for field_name, val in vars(rule).items():
        # bool is an int subclass but no rule carries a bool parameter; guard by
        # excluding bool so a future flag field is not misread as a number.
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)) and val < 0:
            raise InvalidAccountError(
                f"{account.name}/{ph.name}: {type(rule).__name__}.{field_name} "
                f"is negative ({val})"
            )


def _validate_funded_payouts(account, ph) -> None:
    """A funded phase with a ``PAYOUT`` rule must carry a schema to size it; if it
    carries a schema, sanity-check it (§6b, §9)."""
    has_payout_rule = any(r.compile().action == Action.PAYOUT for r in ph.rules)
    if has_payout_rule and ph.payout_schema is None:
        raise InvalidAccountError(
            f"{account.name}/{ph.name}: funded phase has a PAYOUT rule but no "
            f"payout_schema to size it (§6b) — the compiler has nothing to compile "
            f"the payout against."
        )
    if ph.payout_schema is not None:
        _validate_payout_schema(account, ph)


def _validate_payout_schema(account, ph) -> None:
    """Sanity-check a funded phase's :class:`PayoutSchema` (§6b, §9 B3)."""
    schema = ph.payout_schema
    prefix = f"{account.name}/{ph.name}"
    if (not isinstance(schema.fraction_basis, str)
            or schema.fraction_basis not in ("cycle_profit", "retained_profit")):
        raise InvalidAccountError(f"{prefix}: unsupported fraction_basis")
    _number(schema.min_cycle_profit, f"{prefix}.min_cycle_profit")
    if schema.profit_reference_balance is not None:
        value = schema.profit_reference_balance
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise InvalidAccountError(f"{prefix}.profit_reference_balance must be finite real")
        if schema.fraction_basis != "retained_profit":
            raise InvalidAccountError(f"{prefix}: profit_reference_balance requires retained_profit")
    for name in ("withdraw_reduces_equity", "recompute_floor_on_payout"):
        if not isinstance(getattr(schema, name), bool):
            raise InvalidAccountError(f"{prefix}.{name} must be a boolean")
    for value in schema.reset_fields:
        _enum_code(value, StateField, f"{prefix}.reset_fields")
    for name in ("split", "cap_fraction", "min_request", "buffer_floor", "split_tier_cap"):
        _number(getattr(schema, name), f"{prefix}.{name}")
    if schema.split_first_tier is not None:
        _number(schema.split_first_tier, f"{prefix}.split_first_tier")
        if not 0 < schema.split_first_tier <= 1:
            raise InvalidAccountError(f"{prefix}: split_first_tier must be in (0, 1]")
    if isinstance(schema.max_payouts, bool) or not isinstance(schema.max_payouts, Integral):
        raise InvalidAccountError(f"{prefix}: max_payouts must be an integer")
    if any(f not in (StateField.N_QUALIFYING_DAYS, StateField.MAX_DAY_PNL)
           for f in schema.reset_fields):
        raise InvalidAccountError(
            f"{prefix}: unsupported reset_fields; only N_QUALIFYING_DAYS and MAX_DAY_PNL are executable"
        )
    if not schema.dollar_cap:
        raise InvalidAccountError(
            f"{account.name}/{ph.name}: payout schema has an empty dollar_cap tuple"
        )
    for cap in schema.dollar_cap:
        _number(cap, f"{prefix}.dollar_cap", allow_inf=True)
        if cap == 0:
            raise InvalidAccountError(f"{prefix}: dollar_cap entries must be positive")
    if not 0.0 < schema.split <= 1.0:
        raise InvalidAccountError(
            f"{account.name}/{ph.name}: payout split {schema.split} must be in (0, 1]"
        )
    if schema.max_payouts < 1:
        raise InvalidAccountError(
            f"{account.name}/{ph.name}: max_payouts {schema.max_payouts} must be >= 1"
        )
    if not 0.0 < schema.cap_fraction <= 1.0:
        raise InvalidAccountError(
            f"{account.name}/{ph.name}: cap_fraction {schema.cap_fraction} must be in (0, 1]"
        )
    if schema.min_request < 0.0:
        # A negative min_request inverts the §6b fire gate (`cycle_profit >=
        # min_request` becomes always-true), silently mis-firing payouts.
        raise InvalidAccountError(
            f"{account.name}/{ph.name}: min_request {schema.min_request} must be >= 0"
        )
    # buffer_floor is an absolute protected balance, not a fraction of nominal
    # account size. A distant floor can intentionally defer withdrawals; finite
    # and nonnegative checks above suffice without inventing a 1.5x limit.


def _assert_terminable(account, ph) -> None:
    """Role-aware terminability (§9): an eval phase must have a ``PASS`` predicate;
    a funded phase need not be passable."""
    if ph.role == "eval":
        actions = {r.compile().action for r in ph.rules}
        if Action.PASS not in actions:
            raise InvalidAccountError(
                f"{account.name}/{ph.name}: eval phase has no PASS predicate — it "
                f"can never be cleared (a dropped profit target?)"
            )


def _enum_code(value, enum_type, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise InvalidAccountError(f"{name} must be an integer {enum_type.__name__} code")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise InvalidAccountError(f"{name} is not a supported {enum_type.__name__} code") from exc


def _validate_compiled_rule(rule, prefix):
    """Only combinations with an actual predicate/action kernel branch."""
    for name, enum_type in (("action", Action), ("severity", Severity),
                            ("update_timing", Timing), ("check_timing", Timing)):
        _enum_code(getattr(rule, name), enum_type, f"{prefix}.{name}")
    supported_actions = {
        RuleKind.PROFIT_TARGET: (Action.PASS,),
        RuleKind.TRAILING_DD: (Action.FAIL,),
        RuleKind.STATIC_DD: (Action.FAIL,),
        RuleKind.DAILY_LOSS: (Action.FAIL,),
        RuleKind.MIN_DAYS: (Action.PASS,),
        RuleKind.MIN_WINNING_DAYS: (Action.PAYOUT,),
        RuleKind.CONSISTENCY_ADJUST: (Action.ADJUST,),
        RuleKind.CONSISTENCY_GATE: (Action.PASS, Action.PAYOUT),
    }
    if rule.action not in supported_actions[rule.kind]:
        raise InvalidAccountError(f"{prefix}: unsupported action for this rule kind")
    if rule.action == Action.FAIL:
        code = _enum_code(rule.fail_code, ExitCode, f"{prefix}.fail_code")
        if not code.is_failure:
            raise InvalidAccountError(f"{prefix}.fail_code must identify a failure")
    elif rule.fail_code is not None:
        raise InvalidAccountError(f"{prefix}.fail_code is ignored for non-FAIL actions")
    if rule.action == Action.ADJUST:
        field = _enum_code(rule.adjust_field, StateField, f"{prefix}.adjust_field")
        if field != StateField.PROFIT_TARGET:
            raise InvalidAccountError(f"{prefix}.adjust_field only supports PROFIT_TARGET")
    elif rule.adjust_field is not None:
        raise InvalidAccountError(f"{prefix}.adjust_field is ignored for non-ADJUST actions")
    if rule.kind in (RuleKind.MIN_DAYS, RuleKind.MIN_WINNING_DAYS):
        if rule.p0 != math.floor(rule.p0):
            raise InvalidAccountError(f"{prefix}.p0 must be a whole day count")
    if rule.kind in (RuleKind.CONSISTENCY_GATE, RuleKind.CONSISTENCY_ADJUST):
        if rule.p0 > 1:
            raise InvalidAccountError(f"{prefix}: consistency threshold must be in [0, 1]")


def _number(value, name, *, allow_inf=False):
    if (isinstance(value, bool) or not isinstance(value, Real) or value < 0
            or math.isnan(value) or (not allow_inf and not math.isfinite(value))):
        raise InvalidAccountError(f"{name} must be a nonnegative finite number"
                                  + (" or positive infinity" if allow_inf else ""))


__all__ = ["validate", "InvalidAccountError", "KERNEL_PRODUCED_STATE"]
