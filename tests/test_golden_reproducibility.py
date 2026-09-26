"""Golden-value reproducibility guards (refactor-safety net).

The rest of the suite checks that RNG output is *deterministic under a seed* and
*correctly distributed*, but not that a given seed yields a *specific* stream. A
speed refactor that vectorizes the resampler's sequential loop, or changes how the
engine derives per-phase seeds, could produce a different-but-still-valid,
still-deterministic, still-correctly-distributed stream — changing every
simulation result while every distributional/determinism test stays green.

These golden tests pin the exact seed→output mapping so any such change is caught
as a conscious, visible break (update the golden only when the change is
intended). They lock: (1) each resampler's exact integer output, (2) the engine's
phase-seed derivation, and (3) the whole seed→result pipeline via a hash.
"""

from __future__ import annotations

import hashlib

import numpy as np

from propfirm_engine.data import preprocess
from propfirm_engine.engine import Engine, RunConfig
from propfirm_engine.enums import Action, Timing
from propfirm_engine.model import Account, Phase
from propfirm_engine.resampling import IIDDayBootstrap, StationaryDayBootstrap
from propfirm_engine.rules import (
    ConsistencyGateRule,
    MinimumWinningDaysRule,
    ProfitTargetRule,
    TrailingDrawdownRule,
)
from propfirm_engine.schema import PayoutSchema
from propfirm_engine.synthetic import IIDGenerator


def test_iid_day_bootstrap_exact_stream():
    out = IIDDayBootstrap().generate(5, 6, 3, 42)
    assert out.dtype == np.int32
    assert out.tolist() == [[0, 3, 3, 2, 2, 4], [0, 3, 1, 0, 2, 4], [3, 3, 3, 3, 2, 0]]


def test_stationary_day_bootstrap_exact_stream():
    # Locks the sequential block-threading output; a vectorized rewrite must
    # reproduce this exactly (or the golden is updated with intent).
    out = StationaryDayBootstrap(2.5).generate(5, 6, 3, 42)
    assert out.dtype == np.int32
    assert out.tolist() == [[0, 3, 4, 0, 1, 2], [0, 1, 1, 2, 3, 4], [3, 4, 0, 1, 2, 3]]


def test_engine_phase_seed_derivation_is_pinned():
    # Engine.run derives (eval_seed, funded_seed) via SeedSequence(seed).
    # generate_state(2); pin the mapping so a change to that derivation is visible.
    assert [int(x) for x in np.random.SeedSequence(0).generate_state(2)] == \
        [2968811710, 3677149159]
    assert [int(x) for x in np.random.SeedSequence(7).generate_state(2)] == \
        [2083679832, 3939563265]


def _golden_account() -> Account:
    mll = TrailingDrawdownRule(2000.0, update_timing=Timing.EOD,
                               check_timing=Timing.CONTINUOUS)
    ev = Phase("eval", "eval", (ProfitTargetRule(3000.0), mll,
                                ConsistencyGateRule(0.6, gate=Action.PASS)))
    sc = PayoutSchema(dollar_cap=(2000.0,), split=0.9, max_payouts=3,
                      cap_fraction=0.5, min_request=1.0)
    fu = Phase("funded", "funded", (mll, MinimumWinningDaysRule(3, 150.0)),
               payout_schema=sc)
    return Account("g", 50_000, phases=(ev, fu), eval_fee=150.0, activation_fee=100.0)


def test_engine_full_pipeline_golden_hash():
    # The strongest guard: a fixed account + fixed dataset + fixed seed must
    # reproduce the exact per-attempt result arrays. Any change to resampler RNG
    # order, phase-seed derivation, gather, or the kernel changes this hash.
    ds = preprocess(IIDGenerator(win_rate=0.5, rr=1.5).generate(120, seed=3).rows)
    cfg = RunConfig(intraday_mode="summary_approximation", n_paths=200, L_eval=25, L_funded=45, seed=1, size_base=100.0,
                    resampler=IIDDayBootstrap())
    o = Engine().run(_golden_account(), ds, cfg)
    h = hashlib.sha256()
    for a in (o.code, o.payouts_taken, np.round(o.net_payout, 9),
              o.total_trading_days, o.first_payout_day):
        h.update(np.ascontiguousarray(a).tobytes())
    # Rebaselined when the intraday-low convention was corrected (the floating low
    # is now measured from ENTRY as min(close, MAE excursion), not off post-close
    # equity — so a 1R stop floats to 1R, not 2R).
    assert h.hexdigest() == \
        "8ed075af903be65fce5cdedc5c87579c1dd5b85a5c6bb7e6f367aafc8a880f76"
