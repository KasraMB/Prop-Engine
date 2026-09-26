"""Step 4 — the role-aware validator (BUILD_SPEC Step 4, ARCHITECTURE §9)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from propfirm_engine.enums import Action, Severity, StateField
from propfirm_engine.model import Account, Phase
from propfirm_engine.rules import (
    CompiledRule,
    ConsistencyGateRule,
    DailyLossRule,
    MinimumWinningDaysRule,
    ProfitTargetRule,
    Rule,
    RuleKind,
    TrailingDrawdownRule,
    UnknownRuleError,
)
from propfirm_engine.schema import PayoutSchema
from propfirm_engine.validate import InvalidAccountError, validate

_DEFAULT_SCHEMA = PayoutSchema(dollar_cap=(2000.0,), split=0.9, max_payouts=5)


def _eval_phase(*extra):
    return Phase(
        "eval",
        "eval",
        (ProfitTargetRule(3000.0), TrailingDrawdownRule(2500.0), *extra),
    )


def _funded_phase(schema=_DEFAULT_SCHEMA, *extra):
    # A funded phase carrying a PAYOUT rule needs a schema to size it, so the
    # helper attaches a valid default unless a test overrides it.
    return Phase(
        "funded",
        "funded",
        (TrailingDrawdownRule(2500.0), MinimumWinningDaysRule(5, 150.0), *extra),
        payout_schema=schema,
    )


def _account(*phases, size=50_000):
    return Account("50K", size, phases=phases)


# --- accepts sane accounts (regular and irregular) -------------------------- #


def test_accepts_a_normal_two_phase_account():
    validate(_account(_eval_phase(), _funded_phase()))


def test_accepts_an_irregular_but_sane_account():
    # Wildly different rule counts across phases, an extra daily-loss rule, and a
    # direct-funded shape — all sane, none broken.
    eval_ph = _eval_phase(DailyLossRule(1000.0))
    validate(_account(eval_ph, _funded_phase()))


def test_accepts_a_funded_only_account_with_no_pass_condition():
    # A funded phase may be only survival + payout rules (role-aware terminability).
    validate(_account(_funded_phase()))


# --- rejects broken accounts ------------------------------------------------ #


def test_rejects_account_with_no_phases():
    with pytest.raises(InvalidAccountError):
        validate(Account("X", 50_000, phases=()))


def test_rejects_empty_phase():
    with pytest.raises(InvalidAccountError):
        validate(_account(Phase("eval", "eval", ())))


def test_rejects_negative_rule_parameter():
    bad = Phase("eval", "eval", (ProfitTargetRule(3000.0), DailyLossRule(-500.0)))
    with pytest.raises(InvalidAccountError):
        validate(_account(bad))


def test_rejects_unwinnable_eval_phase_with_no_pass_condition():
    # eval with only FAIL rules can never be cleared.
    bad = Phase("eval", "eval", (TrailingDrawdownRule(2500.0), DailyLossRule(1000.0)))
    with pytest.raises(InvalidAccountError):
        validate(_account(bad))


def test_rejects_more_than_one_trailing_dd_in_a_phase():
    bad = Phase(
        "eval",
        "eval",
        (ProfitTargetRule(3000.0), TrailingDrawdownRule(2500.0), TrailingDrawdownRule(2000.0)),
    )
    with pytest.raises(InvalidAccountError):
        validate(_account(bad))


# --- payout schema sanity (§9 B3) ------------------------------------------- #


def test_accepts_explicit_absolute_buffer_far_above_funded_start():
    # A distant protected balance can intentionally defer withdrawals. It does
    # not block them forever if enough profit accrues; no arbitrary nominal cap.
    schema = PayoutSchema(
        dollar_cap=(2000.0,), split=0.9, max_payouts=5, buffer_floor=500_000.0
    )
    validate(_account(_funded_phase(schema)))


def test_accepts_sane_buffer_floor_just_above_start():
    schema = PayoutSchema(
        dollar_cap=(2000.0,), split=0.9, max_payouts=5, buffer_floor=52_100.0
    )
    validate(_account(_funded_phase(schema)))


def test_rejects_out_of_range_split():
    schema = PayoutSchema(dollar_cap=(2000.0,), split=1.5, max_payouts=5)
    with pytest.raises(InvalidAccountError):
        validate(_account(_funded_phase(schema)))


def test_rejects_empty_dollar_cap():
    schema = PayoutSchema(dollar_cap=(), split=0.9, max_payouts=5)
    with pytest.raises(InvalidAccountError):
        validate(_account(_funded_phase(schema)))


def test_rejects_negative_min_request():
    schema = PayoutSchema(
        dollar_cap=(2000.0,), split=0.9, max_payouts=5, min_request=-100.0
    )
    with pytest.raises(InvalidAccountError):
        validate(_account(_funded_phase(schema)))


def test_rejects_funded_payout_rule_without_a_schema():
    # A PAYOUT-action rule needs a schema to size it (§6b) — the compiler would
    # otherwise have a predicate with nothing to compile against.
    funded = Phase(
        "funded",
        "funded",
        (TrailingDrawdownRule(2500.0), MinimumWinningDaysRule(5, 150.0)),
        payout_schema=None,
    )
    with pytest.raises(InvalidAccountError):
        validate(_account(funded))


def test_rejects_non_funded_phase_carrying_a_payout_schema():
    # A payout schema belongs on the funded phase; one on eval is a swallowed
    # authoring mistake otherwise.
    eval_ph = Phase(
        "eval",
        "eval",
        (ProfitTargetRule(3000.0), TrailingDrawdownRule(2500.0)),
        payout_schema=_DEFAULT_SCHEMA,
    )
    with pytest.raises(InvalidAccountError):
        validate(_account(eval_ph, _funded_phase()))


# --- registry / produced-state guards --------------------------------------- #


@dataclass(frozen=True)
class _UnregisteredKindRule(Rule):
    """A rule whose kind has no kernel implementation (registered nowhere)."""

    def requirements(self):
        return (StateField.EQUITY,)

    def compile(self):
        return CompiledRule(kind=9999, action=Action.PASS)  # not in RULE_REGISTRY


@dataclass(frozen=True)
class _UnproducedStateRule(Rule):
    """A (registered-kind) rule that asks for state the kernel does not produce."""

    def requirements(self):
        return (123456,)  # not a produced StateField

    def compile(self):
        return CompiledRule(kind=RuleKind.PROFIT_TARGET, action=Action.PASS)


def test_rejects_unregistered_rule_kind_at_validation():
    bad = Phase("eval", "eval", (ProfitTargetRule(3000.0), _UnregisteredKindRule()))
    with pytest.raises(UnknownRuleError):
        validate(_account(bad))


def test_rejects_rule_requiring_unproduced_state():
    bad = Phase("eval", "eval", (ProfitTargetRule(3000.0), _UnproducedStateRule()))
    with pytest.raises(InvalidAccountError):
        validate(_account(bad))


# --- the terminable check is role-aware, not rule-list-based ----------------- #


def test_soft_severity_daily_loss_does_not_count_as_a_pass():
    # An eval phase whose only non-fail rule is a soft-fail daily loss is still
    # unwinnable — soft is still FAIL, not PASS.
    bad = Phase(
        "eval",
        "eval",
        (TrailingDrawdownRule(2500.0), DailyLossRule(1000.0, severity=Severity.SOFT)),
    )
    with pytest.raises(InvalidAccountError):
        validate(_account(bad))


def test_consistency_gate_pass_satisfies_the_eval_terminability_requirement():
    # A ConsistencyGateRule(gate=PASS) compiles to action=PASS, so it alone makes
    # an eval phase clearable — the terminability check is action-based, not a
    # hardcoded profit-target check.
    ok = Phase(
        "eval",
        "eval",
        (TrailingDrawdownRule(2500.0), ConsistencyGateRule(0.5, gate=Action.PASS)),
    )
    validate(_account(ok))


def test_consistency_gate_payout_does_not_satisfy_eval_terminability():
    # gate=PAYOUT compiles to action=PAYOUT (a repeatable success), which does NOT
    # clear an eval phase — so an eval whose only non-fail rule is a payout gate is
    # unwinnable and must be rejected. This is the exact case that would slip
    # through if compile() dropped `action=self.gate`.
    bad = Phase(
        "eval",
        "eval",
        (TrailingDrawdownRule(2500.0), ConsistencyGateRule(0.5, gate=Action.PAYOUT)),
    )
    with pytest.raises(InvalidAccountError):
        validate(_account(bad))
