"""Numeric configuration must not silently disable rules or truncate counts."""
from dataclasses import replace

import pytest

from propfirm_engine.enums import StateField
from propfirm_engine.model import Account, Phase
from propfirm_engine.rules import MinimumWinningDaysRule, TrailingDrawdownRule
from propfirm_engine.schema import PayoutSchema
from propfirm_engine.validate import InvalidAccountError, validate


SCHEMA = PayoutSchema((100.0,), 0.9, 2)


def account(schema=SCHEMA, rule=None):
    return Account("fixture", 1000, (Phase("funded", "funded",
                   (rule or TrailingDrawdownRule(100.0), MinimumWinningDaysRule(1, 10.0)), schema),))


@pytest.mark.parametrize("field", ["eval_fee", "activation_fee"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_external_fees(field, value):
    with pytest.raises(InvalidAccountError, match=field):
        validate(replace(account(), **{field: value}))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1, 1.5, True])
def test_invalid_nominal_size(value):
    with pytest.raises(InvalidAccountError, match="size"):
        validate(replace(account(), size=value))


@pytest.mark.parametrize("field,value", [
    ("split_first_tier", -0.1), ("split_first_tier", 1.1),
    ("split_first_tier", float("nan")), ("split_tier_cap", float("nan")),
    ("split_tier_cap", -1.0), ("min_request", float("nan")),
    ("min_request", float("inf")), ("max_payouts", 1.5),
    ("max_payouts", float("nan")), ("max_payouts", True),
    ("dollar_cap", (0.0,)), ("dollar_cap", (float("nan"),)),
    ("cap_fraction", float("nan")), ("split", float("nan")),
])
def test_invalid_payout_numbers(field, value):
    with pytest.raises(InvalidAccountError, match=field):
        validate(account(replace(SCHEMA, **{field: value})))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_rule_distance(value):
    with pytest.raises(InvalidAccountError, match="TRAILING_DD.*p0"):
        validate(account(rule=TrailingDrawdownRule(value)))


@pytest.mark.parametrize("field", [StateField.EQUITY, 127])
def test_unimplemented_reset_is_rejected(field):
    with pytest.raises(InvalidAccountError, match="reset_fields"):
        validate(account(replace(SCHEMA, reset_fields=(field,))))


def test_valid_neutral_and_unlimited_sentinels():
    validate(account(PayoutSchema((float("inf"),), 1.0, 2),
                     TrailingDrawdownRule(100.0, lock_at=None)))
    validate(account(replace(SCHEMA, split_first_tier=1.0, split_tier_cap=100.0,
                            reset_fields=(StateField.N_QUALIFYING_DAYS, StateField.MAX_DAY_PNL))))
