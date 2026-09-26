"""Extra kernel↔reference parity coverage for refactor safety (Level-1 gate, §G6).

The Step-6 fuzz in ``test_kernels.py`` runs short paths, no feasibility, and only a
few rule shapes; the feasibility fuzz in ``test_feasibility.py`` never combines the
projection with payouts/consistency/soft-breach. Those are exactly the seams a
speed rewrite (``@njit``, fastmath, float reassociation, vectorization) is most
likely to disturb without tripping a test. This module widens the differential
oracle to cover them:

* **long paths** (thousands of trades) — where accumulation-order/reassociation
  bugs surface that short paths hide;
* **feasibility active over a full funded phase** — projection × close_day ×
  payout fire-gate × consistency × soft-breach, together;
* **the rule combinations absent from the Step-6 randomized net** — static DD at
  EOD, buffer_floor gating, tiered split, multi-element dollar_cap with
  recompute+lock, and a CONTINUOUS-check consistency-raises-target.

Every test asserts the fast kernel and the pure-Python reference agree bit-for-bit
(exact float on payouts, exact int on codes/days).
"""

from __future__ import annotations

import numpy as np

from propfirm_engine.compiler import compile_phase
from propfirm_engine.enums import Action, ExitCode, Severity, StateField, Timing
from propfirm_engine.feasibility import FeasibilitySpec
from propfirm_engine.kernels import simulate_one_phase
from propfirm_engine.model import Phase
from propfirm_engine.reference import simulate_reference
from propfirm_engine.rules import (
    ConsistencyGateRule,
    ConsistencyRaisesTargetRule,
    DailyLossRule,
    MinimumWinningDaysRule,
    ProfitTargetRule,
    StaticDrawdownRule,
    TrailingDrawdownRule,
)
from propfirm_engine.schema import PayoutSchema

_N_QUAL = StateField.N_QUALIFYING_DAYS


def _assert_parity(cp, ret, day, low, *, size_base=1.0, policy=(1.0,), start=0.0,
                   feas=None):
    ret = np.asarray(ret, np.float64); day = np.asarray(day, np.int32)
    low = np.asarray(low, np.float64); pol = np.asarray(policy, np.float64)
    kc, ka, kd, ktd = simulate_one_phase(cp, ret, day, low, size_base, pol, start,
                                         feasibility=feas)
    r = simulate_reference(cp, ret, day, low, size_base, pol, start, feasibility=feas)
    assert kc == r.code, (kc, r.code)
    assert ka == r.payout_amounts, (ka, r.payout_amounts)
    assert kd == r.payout_days, (kd, r.payout_days)
    assert ktd == r.total_trading_days, (ktd, r.total_trading_days)
    return kc, ka


def _rich_funded(update, check):
    """A funded phase exercising most of the machinery at once: trailing DD (locks),
    a soft daily-loss, a tiered/multi-cap payout schema with buffer_floor +
    recompute, min-winning-days, and a consistency payout gate."""
    mll = TrailingDrawdownRule(2000.0, update_timing=update, check_timing=check,
                               lock_at=52_100.0)
    daily = DailyLossRule(800.0, severity=Severity.SOFT)
    schema = PayoutSchema(
        dollar_cap=(1000.0, 1500.0), split=0.9, split_first_tier=0.8,
        split_tier_cap=2000.0, max_payouts=4, cap_fraction=0.5, min_request=1.0,
        buffer_floor=49_000.0, reset_fields=(_N_QUAL,),
        recompute_floor_on_payout=True,
    )
    return compile_phase(Phase(
        "funded", "funded",
        (mll, daily, MinimumWinningDaysRule(2, 100.0),
         ConsistencyGateRule(0.7, gate=Action.PAYOUT)),
        payout_schema=schema,
    ))


# --------------------------------------------------------------------------- #
# Long paths                                                                   #
# --------------------------------------------------------------------------- #


def test_long_path_parity_all_timings():
    rng = np.random.default_rng(2026)
    for update in (Timing.CONTINUOUS, Timing.EOD):
        for check in (Timing.CONTINUOUS, Timing.EOD):
            cp = compile_phase(Phase("eval", "eval",
                (ProfitTargetRule(1e9),  # unreachable -> runs the whole path
                 TrailingDrawdownRule(4000.0, update_timing=update, check_timing=check))))
            for _ in range(4):
                ndays = 250
                tpd = 20
                n = ndays * tpd  # 5000 trades
                ret = rng.normal(0.05, 1.2, size=n)
                day = np.repeat(np.arange(ndays), tpd)
                low = -np.abs(rng.normal(0.5, 0.4, size=n))
                _assert_parity(cp, ret, day, low, size_base=100.0, start=50_000.0)


# --------------------------------------------------------------------------- #
# Feasibility active over a full funded phase                                  #
# --------------------------------------------------------------------------- #


def test_feasibility_parity_over_funded_phase_with_payouts():
    feas = FeasibilitySpec(q_min=1.0, unit_loss=2.0, alpha=0.8)
    rng = np.random.default_rng(77)
    seen = set()
    total_payouts = 0
    for update in (Timing.CONTINUOUS, Timing.EOD):
        for check in (Timing.CONTINUOUS, Timing.EOD):
            cp = _rich_funded(update, check)
            for _ in range(50):
                ndays = int(rng.integers(4, 20))
                tpd = int(rng.integers(1, 6))
                n = ndays * tpd
                ret = rng.normal(0.15, 1.3, size=n)
                day = np.repeat(np.arange(ndays), tpd)[:n]
                low = -np.abs(rng.normal(0.6, 0.5, size=n))
                policy = np.array([rng.uniform(0.5, 40.0)])
                code, payouts = _assert_parity(cp, ret, day, low, size_base=100.0,
                                               policy=policy, start=50_000.0, feas=feas)
                seen.add(code)
                total_payouts += len(payouts)
    # non-vacuity: the funded+feasibility fuzz actually FIRED payouts (projection ×
    # payout gate), WITHERED (projection × close), and BREACHED (projection ×
    # intraday check) -- so the parity is exercised on those seams, not just survival.
    assert total_payouts > 0
    assert int(ExitCode.CAPPED_OUT) in seen
    assert int(ExitCode.FAIL_TRAILING_DD) in seen


# --------------------------------------------------------------------------- #
# Rule combinations absent from the Step-6 randomized net                      #
# --------------------------------------------------------------------------- #


def test_static_dd_eod_parity():
    cp = compile_phase(Phase("eval", "eval",
        (ProfitTargetRule(1e9),
         StaticDrawdownRule(1000.0, severity=Severity.HARD)),))
    # a slow bleed that crosses the static floor only at an EOD close
    ret = [-90.0] * 40
    day = list(range(40))
    low = [0.0] * 40
    code, _ = _assert_parity(cp, ret, day, low, size_base=1.0, start=50_000.0)
    assert code == int(ExitCode.FAIL_STATIC_DD)


def test_buffer_floor_blocks_payout_parity():
    # A payout that would drop equity below buffer_floor must not fire, in BOTH.
    schema = PayoutSchema(dollar_cap=(5000.0,), split=1.0, max_payouts=2,
                          cap_fraction=1.0, min_request=1.0, buffer_floor=49_900.0)
    cp = compile_phase(Phase("funded", "funded",
        (TrailingDrawdownRule(3000.0), MinimumWinningDaysRule(1, 50.0)),
        payout_schema=schema))
    # one qualifying day of +200 cycle profit; cap 5000 > profit, but releasing it
    # would breach the 49_900 buffer (equity 50_200 - 200 = 50_000 >= 49_900 ok?).
    ret = [200.0]
    day = [0]
    low = [0.0]
    _assert_parity(cp, ret, day, low, size_base=1.0, start=50_000.0)


def test_tiered_split_and_multi_cap_with_recompute_and_lock_parity():
    schema = PayoutSchema(
        dollar_cap=(500.0, 800.0, 1200.0), split=0.9, split_first_tier=0.75,
        split_tier_cap=1000.0, max_payouts=3, cap_fraction=1.0, min_request=1.0,
        reset_fields=(_N_QUAL,), recompute_floor_on_payout=True,
    )
    cp = compile_phase(Phase("funded", "funded",
        (TrailingDrawdownRule(2000.0, lock_at=52_100.0),
         MinimumWinningDaysRule(1, 100.0)),
        payout_schema=schema))
    # several qualifying days -> multiple payouts crossing the tier-split boundary
    # and the floor lock; parity must hold across all of it.
    ret = [600.0] * 8
    day = list(range(8))
    low = [0.0] * 8
    _assert_parity(cp, ret, day, low, size_base=1.0, start=50_000.0)


def test_consistency_raises_target_continuous_parity():
    # ConsistencyRaisesTargetRule is only EOD-fuzzed in Step 6; pin the CONTINUOUS
    # check too. A dominant day raises the target intraday in both implementations.
    cp = compile_phase(Phase("eval", "eval",
        (ProfitTargetRule(1000.0),
         ConsistencyRaisesTargetRule(0.5, 3000.0, check_timing=Timing.CONTINUOUS),
         TrailingDrawdownRule(10_000.0))))
    # day0 +900 (dominant), then small days; target should raise to 3000 so the
    # +900 no longer passes -> keeps running.
    ret = [900.0, 100.0, 100.0]
    day = [0, 1, 2]
    low = [0.0, 0.0, 0.0]
    _assert_parity(cp, ret, day, low, size_base=1.0, start=50_000.0)
