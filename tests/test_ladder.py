"""Generator ladder — model-sensitivity bands (BUILD_SPEC Step 13; §G1/§I1).

The step's one hard contract: **a result is only reported with its band across the
ladder, never as a single-generator point.** These tests pin (1) that every
headline is returned as a :class:`Band` spanning the whole ladder including a
block-length range, (2) that the band actually *moves* between the optimistic
i.i.d. floor and the pessimistic dependence models (so it is not collapsing to one
model), (3) determinism under seed, and (4) that there is no "the answer"
single-point accessor on the result.
"""

from __future__ import annotations

import numpy as np

from propfirm_engine.engine import RunConfig
from propfirm_engine.enums import Action, ExitCode, Severity, Timing
from propfirm_engine.ladder import (
    Band,
    LadderResult,
    Rung,
    default_ladder,
    run_ladder,
)
from propfirm_engine.model import Account, Phase
from propfirm_engine.resampling import IIDDayBootstrap, StationaryDayBootstrap
from propfirm_engine.rules import (
    ConsistencyGateRule,
    MinimumWinningDaysRule,
    ProfitTargetRule,
    TrailingDrawdownRule,
)
from propfirm_engine.schema import PayoutSchema
from propfirm_engine.synthetic import IIDGenerator, RegimeSwitchingGenerator


def _eval_account() -> Account:
    """A tight eval: hit a $600 target before a $500 trailing DD breaches. Losing
    streaks (regime persistence) make the DD breach more often, so the pass rate is
    genuinely model-sensitive — ideal for exercising the band."""
    mll = TrailingDrawdownRule(
        500.0, update_timing=Timing.EOD, check_timing=Timing.CONTINUOUS
    )
    eval_phase = Phase("eval", "eval", (ProfitTargetRule(600.0), mll))
    return Account("test", 50_000, phases=(eval_phase,), eval_fee=100.0)


def _survival_account() -> Account:
    """Pure survival: an unreachable target so the only terminals are TIMED_OUT
    (survived the L days) or FAIL_TRAILING_DD (breached the $800 floor). With a
    *losing* edge, clustered losses (regime/block dependence) unambiguously raise
    the breach rate, so 'survival' is a clean, correctly-directional model signal —
    unlike a target race, where clustered wins can cut the other way."""
    mll = TrailingDrawdownRule(
        800.0, update_timing=Timing.EOD, check_timing=Timing.CONTINUOUS
    )
    eval_phase = Phase("eval", "eval", (ProfitTargetRule(1e9), mll))  # target never hit
    return Account("surv", 50_000, phases=(eval_phase,), eval_fee=100.0)


#: fraction of attempts that survived (did not breach) — the survival contract's
#: headline. Losing-edge dependence lowers it monotonically, unlike pass_rate.
_SURVIVAL = {"survival": lambda o: float(np.mean(o.code == int(ExitCode.TIMED_OUT)))}


def _funded_account() -> Account:
    """Eval + funded with a real payout schema and fees, so the fee/payout/renewal
    metrics are all defined (not nan)."""
    mll = TrailingDrawdownRule(
        1_000.0, update_timing=Timing.EOD, check_timing=Timing.CONTINUOUS
    )
    eval_phase = Phase(
        "eval", "eval",
        (ProfitTargetRule(500.0), mll,
         ConsistencyGateRule(0.6, gate=Action.PASS)),
    )
    schema = PayoutSchema(dollar_cap=(1_000.0,), split=0.9, max_payouts=3,
                          cap_fraction=0.5, min_request=1.0)
    funded_phase = Phase(
        "funded", "funded",
        (mll, MinimumWinningDaysRule(3, 100.0)),
        payout_schema=schema,
    )
    return Account("test-f", 50_000, phases=(eval_phase, funded_phase),
                   eval_fee=150.0, activation_fee=100.0)


def _cfg(**kw) -> RunConfig:
    # size_base=100 -> per-trade P&L ~ +$150 / -$100, so the $500-$600 targets and
    # drawdowns are reachable within L days (not a $1-per-trade toy).
    base = dict(n_paths=400, L_eval=20, L_funded=40, seed=1, size_base=100.0)
    base["intraday_mode"] = "summary_approximation"  # legacy summary-model scenarios
    base.update(kw)
    return RunConfig(**base)


# --------------------------------------------------------------------------- #
# The band-reporting discipline                                               #
# --------------------------------------------------------------------------- #


def test_every_headline_is_a_band_spanning_the_whole_ladder():
    rungs = default_ladder(win_rate=0.45, rr=1.5)
    res = run_ladder(_eval_account(), rungs, n_days=120, config=_cfg())
    assert isinstance(res, LadderResult)
    # every rung appears in every metric's band; nothing is dropped
    names = tuple(r.name for r in rungs)
    for metric, band in res.bands.items():
        assert isinstance(band, Band)
        assert tuple(band.per_rung.keys()) == names


def test_ladder_includes_a_block_length_range():
    rungs = default_ladder(win_rate=0.45, rr=1.5, block_lengths=(2, 5, 10))
    names = [r.name for r in rungs]
    assert "iid" in names and "regime" in names and "stochvol" in names
    block = [n for n in names if n.startswith("block-")]
    assert block == ["block-2", "block-5", "block-10"]  # the block-length range


def test_no_single_point_accessor_the_band_is_the_result():
    # the honest contract: you can get a Band, but not a collapsed single number.
    res = run_ladder(_eval_account(), default_ladder(0.45, 1.5), n_days=120,
                     config=_cfg())
    assert not hasattr(res, "value")
    assert not hasattr(res, "point")
    assert not hasattr(res, "estimate")
    band = res.band("pass_rate")
    # the Band exposes the range and the per-rung detail, never a lone scalar summary
    assert set(band.per_rung) == set(res.rungs)
    assert np.isfinite(band.lo) and np.isfinite(band.hi)


# --------------------------------------------------------------------------- #
# The band has to actually MOVE — otherwise it hides no model risk            #
# --------------------------------------------------------------------------- #


def test_dependence_models_move_survival_beyond_sampling_noise():
    # Survival contract, mild losing edge (0.45 win, 1.0 RR -> edge -0.10) so
    # survival is mid-range and differences are large. The claim is MAGNITUDE, not
    # direction (the audit showed the sign is contract+edge dependent — clustering
    # can help OR hurt): a dependence rung must differ from the i.i.d. floor by far
    # more than the ~±0.015 sampling noise at 1500 paths, so the band is a real
    # model effect and a single-generator point would hide it.
    res = run_ladder(_survival_account(),
                     default_ladder(0.45, 1.0, block_lengths=(2, 5, 10, 20)),
                     n_days=200, config=_cfg(n_paths=1500), metrics=_SURVIVAL)
    band = res.band("survival")
    assert band.spread > 0.05  # a real band, not noise (robust across seeds)
    # at least one model departs from the optimistic i.i.d. point materially (either
    # direction) — that departure is the model risk a single-point report hides.
    maxdev = max(abs(v - band.iid) for v in band.per_rung.values())
    assert maxdev > 0.03


def test_block_length_is_the_cause_under_common_random_numbers():
    # Block length is a load-bearing modeling parameter (§C7). Under Common Random
    # Numbers (rungs sharing a crn_group draw the SAME dataset + resample seed), any
    # difference between block rungs is PURE block-length signal, with dataset and
    # resampling noise removed -- which is exactly what the old noisy magnitude test
    # could NOT establish. Proven with a null contrast:
    #   * a second block-2 rung in the group reproduces block-2 EXACTLY (the null:
    #     no change when block length does not change), and
    #   * block-20 differs (block length changes the number).
    # No magnitude threshold is needed; the effect is isolated by construction.
    def gen():
        return RegimeSwitchingGenerator(win_rate=0.45, rr=1.0, persistence=0.9,
                                        spread=0.2)
    rungs = [
        Rung("block-2", gen(), StationaryDayBootstrap(2.0), crn_group="g"),
        Rung("block-2-copy", gen(), StationaryDayBootstrap(2.0), crn_group="g"),
        Rung("block-20", gen(), StationaryDayBootstrap(20.0), crn_group="g"),
    ]
    res = run_ladder(_survival_account(), rungs, n_days=200,
                     config=_cfg(n_paths=1500), metrics=_SURVIVAL)
    sv = res.band("survival").per_rung
    assert sv["block-2"] == sv["block-2-copy"]  # CRN null: same block length -> identical
    assert sv["block-2"] != sv["block-20"]      # block length changes the number


# --------------------------------------------------------------------------- #
# Determinism + funded metrics                                                #
# --------------------------------------------------------------------------- #


def test_deterministic_under_seed():
    rungs = default_ladder(0.45, 1.5)
    a = run_ladder(_eval_account(), rungs, n_days=120, config=_cfg(seed=7))
    b = run_ladder(_eval_account(), rungs, n_days=120, config=_cfg(seed=7))
    assert a.as_table() == b.as_table()
    c = run_ladder(_eval_account(), rungs, n_days=120, config=_cfg(seed=8))
    # a different seed changes at least one number (not degenerate)
    assert a.as_table() != c.as_table()


def test_funded_metrics_are_defined_and_banded_across_the_ladder():
    res = run_ladder(_funded_account(), default_ladder(0.5, 1.2), n_days=200,
                     config=_cfg(n_paths=600))
    # fee_bankroll_efficiency needs a fee (present here) so it is finite on EVERY
    # rung; the point of the ladder is that its value still VARIES across the
    # world-models (a real band, not a constant).
    fbe = res.band("fee_bankroll_efficiency")
    assert all(np.isfinite(v) for v in fbe.per_rung.values())
    assert fbe.spread > 0.0  # the funded economics are genuinely model-sensitive
    # and some attempts actually reached funded + paid out (otherwise the funded
    # metrics would be measuring nothing).
    pv = res.band("payout_velocity")
    assert any(v != 0.0 and np.isfinite(v) for v in pv.per_rung.values())


def test_strategy_metadata_is_recorded_when_supplied():
    strat = {"win_rate": 0.45, "rr": 1.5}
    res = run_ladder(_eval_account(), default_ladder(**strat), n_days=120,
                     config=_cfg(), strategy=strat)
    assert res.strategy == strat  # caller-supplied frozen strategy rides through


def test_custom_rungs_are_honoured():
    rungs = [
        Rung("a", IIDGenerator(win_rate=0.5, rr=1.0), IIDDayBootstrap()),
        Rung("b", RegimeSwitchingGenerator(win_rate=0.5, rr=1.0, persistence=0.95),
             StationaryDayBootstrap(8.0)),
    ]
    res = run_ladder(_eval_account(), rungs, n_days=120, config=_cfg())
    assert res.rungs == ("a", "b")
    for band in res.bands.values():
        assert set(band.per_rung) == {"a", "b"}
