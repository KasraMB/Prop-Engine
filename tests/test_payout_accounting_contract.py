"""Hand-calculated generic accounting cases, independently checked in both engines.

These fixtures do not certify any firm's current rules.
"""
from dataclasses import replace

import numpy as np
import pytest

from propfirm_engine.compiler import compile_phase
from propfirm_engine.enums import Action, StateField
from propfirm_engine.kernels import simulate_one_phase
from propfirm_engine.model import Phase
from propfirm_engine.reference import simulate_reference
from propfirm_engine.rules import ConsistencyGateRule, MinimumWinningDaysRule
from propfirm_engine.schema import PayoutSchema


@pytest.fixture(params=["kernel", "reference"])
def run(request):
    def simulate(schema, returns, *, policy=(1.0,), consistency=False):
        rules = (MinimumWinningDaysRule(1, 1.0),)
        if consistency:
            rules += (ConsistencyGateRule(0.5, gate=Action.PAYOUT),)
        cp = compile_phase(Phase("fixture", "funded", rules, schema))
        ret = np.array(returns, dtype=float)
        args = (cp, ret, np.arange(len(ret)), np.minimum(ret, 0.0),
                1.0, np.array(policy), 1000.0)
        if request.param == "kernel":
            result = simulate_one_phase(*args)
            return result[1], result[2]
        result = simulate_reference(*args)
        return result.payout_amounts, result.payout_days
    return simulate


BASE = PayoutSchema((10000.0,), 1.0, 10,
                    reset_fields=(StateField.N_QUALIFYING_DAYS,))


@pytest.mark.parametrize("profit,cap,fraction,expected", [
    (750.0, 10000.0, 0.5, []),
    (1000.0, 499.0, 1.0, []),
    (1000.0, 500.0, 0.5, [450.0]),
])
def test_minimum_is_gross_request_after_fraction_and_cap(run, profit, cap, fraction, expected):
    schema = replace(BASE, dollar_cap=(cap,), cap_fraction=fraction,
                     min_request=500.0, split=0.9)
    assert run(schema, [profit])[0] == expected


def test_below_minimum_request_does_not_reset_accumulated_profit(run):
    schema = replace(BASE, cap_fraction=0.5, min_request=500.0)
    assert run(schema, [750.0, 250.0]) == ([500.0], [1])


@pytest.mark.parametrize("returns,expected", [
    ([2000.0], [1900.0]),
    ([600.0, 600.0, 600.0], [600.0, 580.0, 540.0]),
    ([1000.0, 1000.0], [1000.0, 900.0]),
])
def test_split_threshold_apportions_each_gross_dollar(run, returns, expected):
    schema = replace(BASE, split=0.9, split_first_tier=1.0, split_tier_cap=1000.0)
    assert run(schema, returns)[0] == expected


def test_next_trade_sizing_sees_payout_and_reduced_balance(run):
    # Day 0 pays all $100. Next trade enters post-payout, flat (stage 3),
    # so its $10/unit return uses multiplier 10: another $100, not $10/$20/$200.
    assert run(BASE, [100.0, 10.0], policy=(1.0, 1.0, 2.0, 10.0, 20.0)) == (
        [100.0, 100.0], [0, 1])


@pytest.mark.parametrize("reset,expected", [(True, [200.0, 20.0]), (False, [200.0])])
def test_best_day_scope_resets_only_when_requested(run, reset, expected):
    schema = replace(BASE, reset_fields=BASE.reset_fields + (
        (StateField.MAX_DAY_PNL,) if reset else ()))
    assert run(schema, [100.0, 100.0, 10.0, 10.0], consistency=True)[0] == expected
