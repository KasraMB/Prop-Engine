"""Direct hand-computed assertions on the reference oracle (refactor-safety).

The reference is the fixed point the kernel is proven against, so it must be
pinned by something *independent of the kernel*. The Step-6 tests assert
hand-computed values through ``_both`` (which also runs the reference), but a
dedicated set here guards the reference on its own — so if a refactor ever factors
a shared helper out of both kernel and reference, a shared bug is still caught by
an oracle assertion that never touches the kernel.

Convention: ``start_equity=0``, unit size, so equity == cumulative P&L.
"""

from __future__ import annotations

import numpy as np

from propfirm_engine.compiler import compile_phase
from propfirm_engine.enums import Action, ExitCode, Timing
from propfirm_engine.model import Phase
from propfirm_engine.reference import simulate_reference
from propfirm_engine.rules import (
    ConsistencyGateRule,
    MinimumWinningDaysRule,
    ProfitTargetRule,
    TrailingDrawdownRule,
)
from propfirm_engine.schema import PayoutSchema


def _ref(cp, ret, day, low, start=0.0):
    return simulate_reference(cp, np.asarray(ret, np.float64),
                              np.asarray(day, np.int32), np.asarray(low, np.float64),
                              1.0, np.array([1.0]), start)


def test_reference_passes_on_exact_target():
    cp = compile_phase(Phase("eval", "eval", (ProfitTargetRule(3000.0),
                                              TrailingDrawdownRule(10_000.0))))
    r = _ref(cp, [1500.0, 1500.0], [0, 1], [0.0, 0.0])
    assert r.code == int(ExitCode.PASSED)
    assert r.total_trading_days == 2


def test_reference_breaches_trailing_dd_at_the_floor():
    cp = compile_phase(Phase("eval", "eval", (ProfitTargetRule(1e9),
                                              TrailingDrawdownRule(1000.0))))
    # +500 -> floor trails to -500; then -1000 -> equity -500 <= -500 -> breach
    r = _ref(cp, [500.0, -1000.0], [0, 1], [0.0, 0.0])
    assert r.code == int(ExitCode.FAIL_TRAILING_DD)


def test_reference_fires_a_funded_payout_with_exact_amount():
    schema = PayoutSchema(dollar_cap=(2500.0,), split=0.9, max_payouts=1,
                          cap_fraction=0.5, min_request=1.0)
    cp = compile_phase(Phase("funded", "funded",
        (TrailingDrawdownRule(1e9), MinimumWinningDaysRule(2, 100.0)),
        payout_schema=schema))
    # two qualifying days of +250 each -> cycle profit 500; gross = min(2500,
    # 0.5*500=250) = 250; net = 0.9*250 = 225; reaching max_payouts -> MAXED_OUT.
    r = _ref(cp, [250.0, 250.0], [0, 1], [0.0, 0.0])
    assert r.payouts_taken == 1
    assert r.payout_amounts[0] == 0.9 * 250.0
    assert r.code == int(ExitCode.MAXED_OUT)


def test_reference_consistency_gate_withholds_pass_without_failing():
    # eval PASS gated by 50% consistency: one day is 80% of profit -> gate blocks
    # the PASS, the account keeps running (never FAIL_CONSISTENCY), then times out.
    cp = compile_phase(Phase("eval", "eval",
        (ProfitTargetRule(1000.0), TrailingDrawdownRule(1e9),
         ConsistencyGateRule(0.5, gate=Action.PASS))))
    r = _ref(cp, [200.0, 800.0], [0, 1], [0.0, 0.0])
    assert r.code == int(ExitCode.TIMED_OUT)  # target hit but consistency blocks PASS
