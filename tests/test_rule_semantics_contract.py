"""Public validation rejects rule records the kernel cannot faithfully execute."""
from dataclasses import dataclass, replace

import pytest

from propfirm_engine.engine import Engine, RunConfig
from propfirm_engine.enums import Action, ExitCode, StateField
from propfirm_engine.model import Account, Phase
from propfirm_engine.rules import CompiledRule, Rule, RuleKind, ProfitTargetRule
from propfirm_engine.schema import PayoutSchema
from propfirm_engine.validate import InvalidAccountError, validate


@dataclass(frozen=True)
class RecordRule(Rule):
    record: CompiledRule

    def compile(self):
        return self.record

    def requirements(self):
        return ()


BASE = CompiledRule(RuleKind.TRAILING_DD, Action.FAIL, p0=100,
                    p1=float("inf"), fail_code=ExitCode.FAIL_TRAILING_DD)


def account(record=BASE, schema=None):
    role = "funded" if schema is not None else "eval"
    rules = (RecordRule(record),) if schema is not None else (RecordRule(record), ProfitTargetRule(200))
    return Account("synthetic", 1000, (Phase("phase", role, rules, schema),))


@pytest.mark.parametrize("field", ["action", "severity", "check_timing", "update_timing"])
@pytest.mark.parametrize("value", [999, 0.5, True, "0", None])
@pytest.mark.parametrize("prepare", [False, True])
def test_invalid_compiled_enums_fail_before_lowering(field, value, prepare):
    a = account(replace(BASE, **{field: value}))
    with pytest.raises(InvalidAccountError, match=field):
        if prepare:
            Engine().prepare(a, RunConfig())
        else:
            validate(a)


@pytest.mark.parametrize("value", [True, 1.0, "1", None])
def test_kind_cannot_alias_an_enum_through_coercion(value):
    with pytest.raises(InvalidAccountError, match="kind"):
        validate(account(replace(BASE, kind=value)))


@pytest.mark.parametrize("kind,action", [
    (RuleKind.TRAILING_DD, Action.PASS),
    (RuleKind.STATIC_DD, Action.PAYOUT),
    (RuleKind.DAILY_LOSS, Action.ADJUST),
    (RuleKind.MIN_DAYS, Action.FAIL),
    (RuleKind.MIN_WINNING_DAYS, Action.PASS),
    (RuleKind.CONSISTENCY_ADJUST, Action.PASS),
    (RuleKind.CONSISTENCY_GATE, Action.FAIL),
])
def test_kernel_action_kind_matrix_is_enforced(kind, action):
    with pytest.raises(InvalidAccountError, match="action"):
        validate(account(replace(BASE, kind=kind, action=action, p1=0)))


@pytest.mark.parametrize("code", [None, ExitCode.PASSED, ExitCode.TIMED_OUT, 999, True, 11.0])
def test_fail_action_requires_real_failure_code(code):
    with pytest.raises(InvalidAccountError, match="fail_code"):
        validate(account(replace(BASE, fail_code=code)))


@pytest.mark.parametrize("field", [None, StateField.EQUITY, 999, True])
def test_adjust_cannot_name_an_ignored_target(field):
    r = CompiledRule(RuleKind.CONSISTENCY_ADJUST, Action.ADJUST,
                     p0=0.5, p1=200, adjust_field=field)
    with pytest.raises(InvalidAccountError, match="adjust_field"):
        validate(account(r))


@pytest.mark.parametrize("kind", [RuleKind.MIN_DAYS, RuleKind.MIN_WINNING_DAYS])
def test_fractional_day_requirements_are_rejected(kind):
    action = Action.PASS if kind == RuleKind.MIN_DAYS else Action.PAYOUT
    r = CompiledRule(kind, action, p0=1.5, p1=10 if action == Action.PAYOUT else 0)
    schema = PayoutSchema((100,), 1, 2) if action == Action.PAYOUT else None
    with pytest.raises(InvalidAccountError, match="whole"):
        validate(account(r, schema))


@pytest.mark.parametrize("kind", [RuleKind.CONSISTENCY_GATE, RuleKind.CONSISTENCY_ADJUST])
def test_consistency_threshold_is_a_fraction(kind):
    adjust = kind == RuleKind.CONSISTENCY_ADJUST
    r = CompiledRule(kind, Action.ADJUST if adjust else Action.PASS,
                     p0=1.1, adjust_field=StateField.PROFIT_TARGET if adjust else None)
    with pytest.raises(InvalidAccountError, match="threshold"):
        validate(account(r))


@pytest.mark.parametrize("field", ["withdraw_reduces_equity", "recompute_floor_on_payout"])
@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_payout_flags_must_be_actual_booleans(field, value):
    schema = replace(PayoutSchema((100,), 1, 2), **{field: value})
    with pytest.raises(InvalidAccountError, match=field):
        validate(account(BASE, schema))


def test_integral_enum_codes_and_neutral_count_are_supported():
    validate(account(replace(BASE, kind=1, action=0, severity=1,
                             update_timing=1, check_timing=0, fail_code=11)))
    validate(account(CompiledRule(RuleKind.MIN_DAYS, Action.PASS, p0=0)))


@pytest.mark.parametrize("field,value", [
    ("fail_code", ExitCode.FAIL_GENERIC),
    ("adjust_field", StateField.PROFIT_TARGET),
])
def test_inapplicable_compiled_fields_are_not_silently_ignored(field, value):
    rule = CompiledRule(RuleKind.MIN_DAYS, Action.PASS, p0=1)
    with pytest.raises(InvalidAccountError, match=field):
        validate(account(replace(rule, **{field: value})))


@pytest.mark.parametrize("value", [True, 9.0, "9"])
def test_reset_field_must_be_a_real_integer_code(value):
    schema = PayoutSchema((100,), 1, 2, reset_fields=(value,))
    with pytest.raises(InvalidAccountError, match="reset_fields"):
        validate(account(BASE, schema))


def test_failure_codes_and_supported_adjust_remain_legal():
    validate(account(replace(BASE, fail_code=ExitCode.FAIL_GENERIC)))
    validate(account(CompiledRule(RuleKind.CONSISTENCY_ADJUST, Action.ADJUST,
                                 p0=0.5, p1=300, adjust_field=StateField.PROFIT_TARGET)))
