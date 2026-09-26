"""Direct tests for the scalar objectives (refactor-safety; §14.3).

``objectives.py`` had no direct test — it was only exercised transitively. These
pin each objective's definition against a hand-computed value on a small crafted
batch, so a refactor of the statistics layer they build on cannot silently change
what an optimizer maximizes.
"""

from __future__ import annotations

import numpy as np

from propfirm_engine.engine import Outcomes
from propfirm_engine.enums import ExitCode
from propfirm_engine.objectives import (
    annualized_return_on_fee,
    expected_net_payoff,
    expected_payout_st_profitable,
)


def _outcomes(net_payout, reached_funded, *, eval_fee=100.0, activation_fee=50.0,
              total_days=None, tdpw=5.0):
    net_payout = np.asarray(net_payout, np.float64)
    b = net_payout.shape[0]
    rf = np.asarray(reached_funded, dtype=bool)
    td = np.full(b, 52, np.int32) if total_days is None else np.asarray(total_days, np.int32)
    return Outcomes(
        code=np.full(b, int(ExitCode.MAXED_OUT), np.int32),
        reached_funded=rf,
        net_payout=net_payout,
        payouts_taken=np.ones(b, np.int32),
        first_payout_day=np.zeros(b, np.int32),
        total_trading_days=td,
        size=50_000, size_base=1.0, max_payouts=3,
        eval_fee=eval_fee, activation_fee=activation_fee,
        trading_days_per_week=tdpw, fingerprint="x",
    )


def test_expected_net_payoff_is_mean_net_of_attributable_fee():
    # attributable fee = eval_fee + activation_fee*reached_funded.
    # attempt A: net 1000, reached funded -> fee 150 -> net payoff 850
    # attempt B: net 0,   not funded     -> fee 100 -> net payoff -100
    o = _outcomes([1000.0, 0.0], [True, False])
    assert expected_net_payoff(o) == (850.0 + -100.0) / 2


def test_expected_payout_subject_to_profitable_floor():
    # both attempts profitable (net payoff > 0): constraint P(profitable)=1 >= 0.5
    o = _outcomes([1000.0, 500.0], [True, True])  # fees 150 each -> 850, 350 both > 0
    assert expected_payout_st_profitable(o, floor=0.5) == expected_net_payoff(o)
    # now only 1 of 2 profitable -> P=0.5 < 0.9 -> constraint violated -> -inf
    o2 = _outcomes([1000.0, 0.0], [True, True])  # 850>0, -150<0 -> P=0.5
    assert expected_payout_st_profitable(o2, floor=0.9) == float("-inf")


def test_annualized_return_on_fee_scales_with_time():
    # same payouts/fees, but half the calendar duration -> ~double the annualized
    # return-on-fee (time in the denominator, from cadence).
    slow = _outcomes([1000.0, 1000.0], [True, True], total_days=[52, 52])
    fast = _outcomes([1000.0, 1000.0], [True, True], total_days=[26, 26])
    assert annualized_return_on_fee(fast) > annualized_return_on_fee(slow)
    assert np.isclose(annualized_return_on_fee(fast),
                      2 * annualized_return_on_fee(slow), rtol=1e-9)
