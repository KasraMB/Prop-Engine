"""Fraction bases and cycle eligibility are different quantities."""
from dataclasses import replace

import numpy as np
import pytest

from propfirm_engine import (
    Account, Engine, Phase, PayoutSchema, RunConfig, StateField,
    MinimumWinningDaysRule, InvalidAccountError, compile_phase, fingerprint, preprocess, validate,
)
from propfirm_engine.kernels import simulate_one_phase
from propfirm_engine.reference import simulate_reference


def schema(**changes):
    return PayoutSchema((2000.0,), 0.9, 5, cap_fraction=0.5,
                        min_request=500.0, fraction_basis="retained_profit",
                        profit_reference_balance=50_000.0, min_cycle_profit=1.0,
                        reset_fields=(StateField.N_QUALIFYING_DAYS,), **changes)


def phase(sc):
    return Phase("funded", "funded", (MinimumWinningDaysRule(1, 0.0),), sc)


@pytest.fixture(params=["reference", "kernel", "batch"])
def run(request):
    def execute(sc, returns):
        ph = phase(sc)
        values = np.asarray(returns, dtype=float)
        if request.param == "batch":
            class OrderedDays:
                def generate(self, n_days, L, n_paths, seed):
                    return np.tile(np.arange(n_days), (n_paths, 1))
            ds = preprocess([{"timestamp": f"2026-01-{i+1:02d}T12:00:00", "return": x}
                             for i, x in enumerate(values)])
            out = Engine().run(Account("test", 50_000, (ph,)), ds,
                               RunConfig(n_paths=1, L_funded=len(values),
                                         resampler=OrderedDays()))
            return out.payouts_taken[0], out.net_payout[0]
        args = (compile_phase(ph), values, np.arange(len(values)),
                np.minimum(values, 0.0), 1.0, np.ones(1), 50_000.0)
        if request.param == "kernel":
            _, amounts, _, _ = simulate_one_phase(*args)
        else:
            amounts = simulate_reference(*args).payout_amounts
        return len(amounts), sum(amounts)
    return execute


def test_retained_profit_carries_into_next_request(run):
    # 53,000 -> withdraw 1,500 -> 51,500; earn 100 -> 51,600.
    # Second gross request is 800, not 50. New cycle profit is only 100.
    assert run(schema(), [3000, 100]) == (2, 2070.0)


def test_cycle_basis_remains_available_for_other_programs(run):
    assert run(replace(schema(), fraction_basis="cycle_profit",
                       profit_reference_balance=None), [3000, 100]) == (1, 1350.0)


@pytest.mark.parametrize("new_profit,count,total", [
    (0, 1, 1800), (0.5, 1, 1800), (1, 2, 3150.45), (-1, 1, 1800),
])
def test_retained_balance_does_not_replace_cycle_gate(run, new_profit, count, total):
    actual_count, actual_total = run(schema(), [5000, new_profit])
    assert actual_count == count
    assert actual_total == pytest.approx(total)


def test_cycle_losses_are_included_before_new_eligibility(run):
    # Retained 3,000 after first request. -200 then +150 leaves negative cycle.
    assert run(schema(), [5000, -200, 150]) == (1, 1800.0)


def test_profit_reference_is_not_the_phase_opening_balance(run):
    # Reference can deliberately differ from opening: retained profit starts at 500.
    sc = replace(schema(), profit_reference_balance=49_500.0)
    assert run(sc, [500]) == (1, 450.0)


def test_unspecified_retained_reference_uses_phase_opening(run):
    assert run(replace(schema(), profit_reference_balance=None), [3000, 100]) == (2, 2070.0)


def test_retained_profit_below_minimum_does_not_pay(run):
    assert run(schema(), [750.0]) == (0, 0)


@pytest.mark.parametrize("field,value", [
    ("fraction_basis", "unknown"), ("fraction_basis", None),
    ("profit_reference_balance", float("nan")),
    ("profit_reference_balance", True),
    ("min_cycle_profit", -1), ("min_cycle_profit", float("inf")),
])
def test_new_schema_terms_are_validated(field, value):
    sc = replace(schema(), **{field: value})
    with pytest.raises(InvalidAccountError):
        validate(Account("test", 50_000, (phase(sc),)))


def test_cache_identity_includes_all_new_terms():
    base = schema()
    configs = [base, replace(base, min_cycle_profit=2),
               replace(base, profit_reference_balance=49_000),
               replace(base, fraction_basis="cycle_profit", profit_reference_balance=None)]
    assert len({fingerprint(Account("test", 50_000, (phase(sc),))) for sc in configs}) == 4
