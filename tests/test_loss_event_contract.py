"""Independent checks of closed-trade event ordering, not full MTM fidelity."""
import numpy as np
import pytest

from propfirm_engine.compiler import compile_phase
from propfirm_engine.enums import ExitCode, Timing
from propfirm_engine.kernels import simulate_one_phase
from propfirm_engine.model import Phase
from propfirm_engine.reference import simulate_reference
from propfirm_engine.rules import DailyLossRule, ProfitTargetRule, TrailingDrawdownRule


@pytest.fixture(params=["kernel", "reference"])
def run(request):
    def simulate(rules, ret, low, *, cost=0.0, days=None):
        cp = compile_phase(Phase("fixture", "eval", (ProfitTargetRule(10000.0),) + tuple(rules)))
        args = (cp, np.array(ret), np.zeros(len(ret), dtype=int) if days is None else np.array(days),
                np.array(low), 1.0, np.array([1.0]), 1000.0)
        if request.param == "kernel":
            return simulate_one_phase(*args, trade_cost=cost)[0]
        return simulate_reference(*args, trade_cost=cost).code
    return simulate


def test_close_fee_can_cross_established_floor(run):
    # Gross close $901 survives $900 floor; $1 settlement fee reaches it.
    assert run([TrailingDrawdownRule(100.0)], [-99.0], [-99.0], cost=1.0) == ExitCode.FAIL_TRAILING_DD


def test_fee_only_breach_not_deferred_to_next_trade(run):
    assert run([TrailingDrawdownRule(1.0)], [0.0], [0.0], cost=1.0) == ExitCode.FAIL_TRAILING_DD


def test_current_trade_low_precedes_its_closing_ratchet(run):
    # $950 excursion is above the old $900 floor, then close $1200 raises floor
    # to $1100. The earlier $950 must not be compared with that later floor.
    rule = TrailingDrawdownRule(100.0, update_timing=Timing.CONTINUOUS)
    assert run([rule], [200.0], [-50.0]) == ExitCode.TIMED_OUT


def test_previous_trade_low_not_rechecked_after_later_ratchet(run):
    rule = TrailingDrawdownRule(100.0, update_timing=Timing.CONTINUOUS)
    assert run([rule], [0.0, 200.0], [-50.0, 0.0]) == ExitCode.TIMED_OUT


def test_next_trade_is_checked_against_new_floor(run):
    rule = TrailingDrawdownRule(100.0, update_timing=Timing.CONTINUOUS)
    assert run([rule], [200.0, 0.0], [-50.0, -100.0]) == ExitCode.FAIL_TRAILING_DD


@pytest.mark.parametrize("reverse", [False, True])
def test_same_observed_loss_cannot_hide_hard_breach_behind_soft_rule(run, reverse):
    rules = [DailyLossRule(100.0), TrailingDrawdownRule(100.0)]
    if reverse:
        rules.reverse()
    assert run(rules, [-100.0], [-100.0]) == ExitCode.FAIL_TRAILING_DD


def test_eod_update_does_not_recheck_old_low_next_session(run):
    rule = TrailingDrawdownRule(100.0, update_timing=Timing.EOD)
    assert run([rule], [200.0, 0.0], [-50.0, 0.0], days=[0, 1]) == ExitCode.TIMED_OUT
