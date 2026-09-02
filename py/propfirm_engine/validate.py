"""The validator — permissive, not a schema (ARCHITECTURE §9; BUILD_SPEC Step 4).

A rigid schema would reject legitimate irregularity, defeating an all-firms
engine. The validator instead asserts *sanity invariants that must hold
regardless of firm structure*, and is meant to run inside ``Engine.run`` before
compilation. Its division of labour with the rest of the config floor (§7):

    ``validate`` rejects **broken**; the rule registry (§5) rejects
    **unimplemented**; the table format permits every irregularity that is neither.

The one universal structural rule is *role-aware terminability* (:func:`_assert_terminable`):
an **eval** phase must be clearable by at least one ``PASS`` predicate (or it can
never be passed — the most common config typo, a dropped profit target), while a
**funded** phase need not be passable at all (it may be only survival ``FAIL``
rules plus repeatable ``PAYOUT``s). Keying on ``Phase.role`` rather than a fixed
rule list admits every legitimate irregularity while still catching the
unwinnable-eval mistake.
"""

from __future__ import annotations

from .enums import Action, StateField
from .rules import TrailingDrawdownRule, assert_kernel_supports

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
    """Raised when an assembled account violates a sanity invariant (§9)."""


def validate(account) -> None:
    """Assert the sanity invariants of §9 on an assembled account; raise on any.

    Accepts irregular-but-sane accounts (unusual rule counts, direct-funded
    accounts, quirky per-size structure); rejects genuinely broken ones.
    """
    if not account.phases:
        raise InvalidAccountError(f"{account.name}: no phases")

    for ph in account.phases:
        if not ph.rules:
            raise InvalidAccountError(f"{account.name}/{ph.name}: no rules")

        for r in ph.rules:
            compiled = r.compile()
            assert_kernel_supports(compiled.kind)  # the kernel can run it (§5)
            for sf in r.requirements():  # it needs only producible state
                if sf not in KERNEL_PRODUCED_STATE:
                    raise InvalidAccountError(
                        f"{account.name}/{ph.name}: {type(r).__name__} needs {sf!r}, "
                        f"which the kernel does not produce"
                    )
            _assert_non_negative_params(account, ph, r)

        # State-layout limit (§8): the kernel carries a single trailing-DD reference
        # per phase, so two independent trailing rules would collide silently.
        if sum(isinstance(r, TrailingDrawdownRule) for r in ph.rules) > 1:
            raise InvalidAccountError(
                f"{account.name}/{ph.name}: >1 TrailingDrawdownRule — the kernel "
                f"supports one trailing reference per phase (§8). Index DD state by "
                f"rule if a firm ever needs two."
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
    if not schema.dollar_cap:
        raise InvalidAccountError(
            f"{account.name}/{ph.name}: payout schema has an empty dollar_cap tuple"
        )
    if any(c < 0 for c in schema.dollar_cap):
        raise InvalidAccountError(
            f"{account.name}/{ph.name}: payout schema dollar_cap has a negative entry"
        )
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
    # buffer_floor (§9 B3): it is an ABSOLUTE balance level. A value far above the
    # funded start silently blocks EVERY payout (funded yields zero income yet
    # otherwise validates); a value below start never gates. Require it within a
    # sane band of the funded start (= account.size).
    bf = schema.buffer_floor
    start = account.size
    if bf and not (start <= bf <= start * 1.5):
        raise InvalidAccountError(
            f"{account.name}/{ph.name}: buffer_floor {bf} is not sane vs funded "
            f"start {start} — below start never gates; far above blocks all payouts."
        )


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


__all__ = ["validate", "InvalidAccountError", "KERNEL_PRODUCED_STATE"]
