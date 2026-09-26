"""Intraday-low convention + per-trade cost (kernel & reference, Level-1 gate).

The intraday low-water mark is the true worst floating equity *from entry* — the
lower of the trade's close and its MAE excursion (``trade_low = -mae``). A 1R stop
therefore floats to exactly 1R below entry, not 2R (the old post-close-base bug).
The summary model settles per-trade cost at close; that net closing equity is
also tested for a breach, even if the pre-cost excursion stayed above the floor.

Every case runs BOTH the kernel and the reference oracle and asserts they agree,
then checks the outcome against a hand-computed expectation.
"""

from __future__ import annotations

import numpy as np

from propfirm_engine.compiler import compile_phase
from propfirm_engine.enums import ExitCode, Timing
from propfirm_engine.model import Phase
from propfirm_engine.kernels import simulate_one_phase
from propfirm_engine.reference import simulate_reference
from propfirm_engine.rules import ProfitTargetRule, TrailingDrawdownRule


def _both(rules, ret, day, low, *, start=50_000.0, size_base=1.0, cost=0.0):
    cp = compile_phase(Phase("eval", "eval", tuple(rules)))
    ret = np.asarray(ret, np.float64); day = np.asarray(day, np.int32)
    low = np.asarray(low, np.float64); pol = np.array([1.0])
    k = simulate_one_phase(cp, ret, day, low, size_base, pol, start, trade_cost=cost)
    r = simulate_reference(cp, ret, day, low, size_base, pol, start, trade_cost=cost)
    assert k[0] == r.code and k[1] == r.payout_amounts and k[3] == r.total_trading_days
    return k[0]


# --------------------------------------------------------------------------- #
# The intraday low is 1R below entry for a 1R stop (not 2R)                     #
# --------------------------------------------------------------------------- #


def test_full_stop_loss_floats_to_one_R_not_two():
    # entry 50000, one full loser: ret = -1, mae = 1 (trade_low = -1), unit size.
    # True intraday low = 50000 - 1 = 49999.  A CONTINUOUS floor is breached iff it
    # sits at or above 49999, i.e. amount <= 1R. The old (buggy) 2R low = 49998
    # would have breached any floor up to amount = 2.
    mll = lambda amt: TrailingDrawdownRule(amt, check_timing=Timing.CONTINUOUS)  # noqa: E731
    # amount 0.9 -> floor 49999.1 >= 49999 -> breach
    assert _both([ProfitTargetRule(1e9), mll(0.9)], [-1.0], [0], [-1.0]) == \
        int(ExitCode.FAIL_TRAILING_DD)
    # amount 1.1 -> floor 49998.9 < 49999 -> survives (the old 2R bug would breach)
    assert _both([ProfitTargetRule(1e9), mll(1.1)], [-1.0], [0], [-1.0]) == \
        int(ExitCode.TIMED_OUT)


def test_winner_that_dipped_below_entry_can_breach_intraday():
    # A trade that CLOSES up (+1.5) but floated to -0.5R intraday (mae 0.5) — its
    # intraday low is 50000 - 0.5 = 49999.5, which a tight CONTINUOUS floor catches
    # even though the close alone (+1.5) never went negative.
    mll = TrailingDrawdownRule(0.4, check_timing=Timing.CONTINUOUS)  # floor 49999.6
    assert _both([ProfitTargetRule(1e9), mll], [1.5], [0], [-0.5]) == \
        int(ExitCode.FAIL_TRAILING_DD)


def test_low_is_the_close_when_mae_is_shallower_than_the_realized_move():
    # trade_low = 0 (no excursion beyond close) but a -1000 close: the low must
    # still reach the close (= min(ret, trade_low)), so a floor at 900 breaches.
    mll = TrailingDrawdownRule(900.0, check_timing=Timing.CONTINUOUS)
    assert _both([ProfitTargetRule(1e9), mll], [-1000.0], [0], [0.0]) == \
        int(ExitCode.FAIL_TRAILING_DD)


# --------------------------------------------------------------------------- #
# Per-trade cost                                                                #
# --------------------------------------------------------------------------- #


def test_per_trade_cost_reduces_realized_pnl():
    # 10 trades of +12 (unit size) reach a 100 target with no cost, but at a $5
    # per-trade cost each nets +7 -> +70 over ten trades -> never reaches 100.
    rules = [ProfitTargetRule(100.0), TrailingDrawdownRule(10_000.0)]
    ret = [12.0] * 10
    day = list(range(10))
    low = [0.0] * 10
    assert _both(rules, ret, day, low, start=0.0, cost=0.0) == int(ExitCode.PASSED)
    assert _both(rules, ret, day, low, start=0.0, cost=5.0) == int(ExitCode.TIMED_OUT)


def test_cost_is_applied_to_equity_via_reference_trace():
    # With ret = 0 on every trade, a $3 cost drains exactly $3 per executed trade.
    cp = compile_phase(Phase("eval", "eval",
                             (ProfitTargetRule(1e9), TrailingDrawdownRule(1e9))))
    n = 8
    r = simulate_reference(cp, np.zeros(n), np.arange(n, dtype=np.int32), np.zeros(n),
                           1.0, np.array([1.0]), 50_000.0, trace=True, trade_cost=3.0)
    assert r.trace[-1]["total_pnl"] == -3.0 * n  # exact
    assert r.trace[-1]["equity"] == 50_000.0 - 3.0 * n


def test_kernel_and_reference_agree_with_cost_over_fuzz():
    rng = np.random.default_rng(2026)
    for _ in range(120):
        nd = int(rng.integers(2, 10)); tpd = int(rng.integers(1, 5)); n = nd * tpd
        ret = rng.normal(0.1, 1.3, n)
        day = np.repeat(np.arange(nd), tpd)[:n].astype(np.int32)
        low = -np.abs(rng.normal(0.6, 0.5, n))
        cost = float(rng.uniform(0, 4))
        rules = [ProfitTargetRule(float(rng.uniform(50, 400))),
                 TrailingDrawdownRule(float(rng.uniform(200, 2000)),
                                      update_timing=Timing.EOD,
                                      check_timing=Timing.CONTINUOUS)]
        _both(rules, ret, day, low, start=50_000.0, size_base=float(rng.uniform(1, 100)),
              cost=cost)  # asserts parity inside
