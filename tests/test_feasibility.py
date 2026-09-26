"""Feasibility projection (BUILD_SPEC Step 14; ARCHITECTURE §16.4b; MODEL_RISKS §C2).

Pins the §16.4b contract and, critically, that activating the projection keeps the
**Level-1 kernel↔reference bitwise gate** (§G6) intact — the executed path is
bit-identical between the fast kernel and the reference oracle because both call
the same :func:`project_position`. Also pins the two outcomes that must never be
conflated: **clipping** (normal, account trades on) vs **non-tradability**
(``CAPPED_OUT``, distinct from a drawdown breach), and that the ``r → 0`` exploit
is closed (a microscopic-risk policy withers rather than surviving forever).
"""

from __future__ import annotations

import numpy as np

from propfirm_engine.compiler import compile_phase
from propfirm_engine.enums import ExitCode, Timing
from propfirm_engine.feasibility import (
    FeasibilityDiag,
    FeasibilitySpec,
    project_position,
)
from propfirm_engine.kernels import simulate_one_phase
from propfirm_engine.model import Phase
from propfirm_engine.reference import simulate_reference
from propfirm_engine.rules import ProfitTargetRule, TrailingDrawdownRule

_CAPPED = int(ExitCode.CAPPED_OUT)
_FAIL_DD = int(ExitCode.FAIL_TRAILING_DD)


# --------------------------------------------------------------------------- #
# project_position — the pure projection arithmetic                            #
# --------------------------------------------------------------------------- #


def test_nontradable_when_budget_below_one_min_loss():
    # alpha*buffer < L_min (= q_min*unit_loss = 1*100 = 100) -> capped_out.
    q, capped, reduced, at_cap = project_position(5.0, buffer=90.0, q_min=1.0,
                                                  unit_loss=100.0, alpha=1.0)
    assert capped and q == 0.0 and not reduced and not at_cap


def test_feasible_request_executes_unclipped():
    # buffer 2000, alpha 1 -> budget 2000, cap = floor(2000/100)=20 units. desired
    # 10 fits -> executes 10, no clip, buffer not binding.
    q, capped, reduced, at_cap = project_position(10.0, buffer=2000.0, q_min=1.0,
                                                  unit_loss=100.0, alpha=1.0)
    assert not capped and q == 10.0 and not reduced and not at_cap


def test_oversized_request_is_clipped_not_failed():
    # desired 30 > cap 20 -> clip to 20, reduced True, buffer binding, NOT capped.
    q, capped, reduced, at_cap = project_position(30.0, buffer=2000.0, q_min=1.0,
                                                  unit_loss=100.0, alpha=1.0)
    assert not capped and q == 20.0 and reduced and at_cap


def test_subminimum_request_floors_up_to_one_contract():
    # A policy that wants < q_min still trades one contract (this is what keeps
    # r -> 0 from surviving). desired 0.3, q_min 1 -> executes 1.0, not a clip-down.
    q, capped, reduced, at_cap = project_position(0.3, buffer=2000.0, q_min=1.0,
                                                  unit_loss=100.0, alpha=1.0)
    assert not capped and q == 1.0 and not reduced and not at_cap


def test_granularity_rounds_down_to_contract_grid():
    # desired 7.9 with q_min 2 -> rounds down to 6 (integer multiples of q_min).
    # The grid (not the buffer) bound it: reduced, but NOT at_cap.
    q, capped, reduced, at_cap = project_position(7.9, buffer=10_000.0, q_min=2.0,
                                                  unit_loss=100.0, alpha=1.0)
    assert q == 6.0 and reduced and not at_cap  # 6 < 7.9, buffer had room


def test_at_cap_is_only_flagged_when_the_buffer_strictly_reduces_size():
    # mll_constrained (at_cap) means the buffer ACTIVELY cut the size below what the
    # policy wanted. At an exact tie (desired == cap) the policy got exactly what it
    # wanted, so at_cap is NOT flagged -- the buffer did not force a smaller size.
    _q, _c, reduced, at_cap = project_position(20.0, buffer=2000.0, q_min=1.0,
                                               unit_loss=100.0, alpha=1.0)
    assert not at_cap and not reduced  # got exactly the cap; buffer not binding


def test_subminimum_that_affords_exactly_one_contract_is_not_mll_constrained():
    # The §16.9 false-positive guard: a policy wanting < q_min is floored UP to q_min
    # (a regulatory-minimum decision, not a buffer decision). Even when the budget
    # affords exactly one contract, the buffer did not cut the *request*, so at_cap
    # must be False -- otherwise fraction_constrained over-counts.
    # budget = 1*150 = 150 -> cap_mult = int(150/100) = 1 -> cap_units = 1 == q_min.
    q, _c, reduced, at_cap = project_position(0.3, buffer=150.0, q_min=1.0,
                                              unit_loss=100.0, alpha=1.0)
    assert q == 1.0 and not reduced and not at_cap


def test_alpha_shrinks_the_executable_cap():
    # alpha 0.5 halves the budget: cap = floor(0.5*2000/100)=10 units.
    q, _c, reduced, at_cap = project_position(30.0, buffer=2000.0, q_min=1.0,
                                              unit_loss=100.0, alpha=0.5)
    assert q == 10.0 and reduced and at_cap


def test_capped_out_boundary_scales_with_alpha():
    # The wither threshold is measured against the safety-scaled budget: alpha*B <
    # L_min (= 100 here). At buffer=150 the account is non-tradable under alpha=0.5
    # (75 < 100) but tradable under alpha=1.0 (150 >= 100) -- the alpha<1 boundary
    # the earlier tests never pinned.
    strict = project_position(5.0, buffer=150.0, q_min=1.0, unit_loss=100.0, alpha=0.5)
    loose = project_position(5.0, buffer=150.0, q_min=1.0, unit_loss=100.0, alpha=1.0)
    assert strict[1] is True   # capped_out under alpha=0.5
    assert loose[1] is False   # tradable under alpha=1.0


def test_min_buffer_hard_floor_caps_out_independent_of_contract_size():
    # A hard drawdown floor: cap out when the raw buffer drops below min_buffer,
    # with fine (q_min=1) sizing preserved above it.
    # buffer 150 > 100 floor -> tradable, fine size
    q, capped, *_ = project_position(500.0, 150.0, q_min=1.0, unit_loss=1.0,
                                     alpha=1.0, min_buffer=100.0)
    assert not capped and q == 150.0
    # buffer 99 < 100 floor -> CAPPED_OUT even though one contract (L_min=1) would fit
    q, capped, *_ = project_position(500.0, 99.0, q_min=1.0, unit_loss=1.0,
                                     alpha=1.0, min_buffer=100.0)
    assert capped and q == 0.0
    # min_buffer defaults to 0 -> unchanged (only the can't-fit-one-contract cap)
    _q, capped, *_ = project_position(500.0, 5.0, q_min=1.0, unit_loss=1.0, alpha=1.0)
    assert not capped  # buffer 5 still fits L_min=1


def test_spec_rejects_bad_params():
    for bad in (dict(q_min=0.0, unit_loss=1.0), dict(q_min=1.0, unit_loss=-1.0),
                dict(q_min=1.0, unit_loss=1.0, alpha=0.0),
                dict(q_min=1.0, unit_loss=1.0, alpha=1.5)):
        try:
            FeasibilitySpec(**bad)
            assert False, f"expected rejection for {bad}"
        except ValueError:
            pass


# --------------------------------------------------------------------------- #
# Level-1 parity WITH the projection active                                    #
# --------------------------------------------------------------------------- #


def _phase(amount=2000.0, target=3000.0, update=Timing.CONTINUOUS,
           check=Timing.CONTINUOUS):
    mll = TrailingDrawdownRule(amount, update_timing=update, check_timing=check)
    return compile_phase(Phase("eval", "eval", (ProfitTargetRule(target), mll)))


def _run_both(cp, ret, day, low, size_base, policy, start, feas):
    ret = np.asarray(ret, np.float64); day = np.asarray(day, np.int32)
    low = np.asarray(low, np.float64); policy = np.asarray(policy, np.float64)
    k = simulate_one_phase(cp, ret, day, low, size_base, policy, start, feasibility=feas)
    r = simulate_reference(cp, ret, day, low, size_base, policy, start, feasibility=feas)
    return k, r


# Scale note: after projection, trade P&L is ``q * ret`` with ``q`` the executed
# size (units) and ``ret`` in R, so one unit's worst realized loss ~ 1R. The
# feasibility model is therefore consistent at ``unit_loss = 1`` (a ~1R stop);
# clipping/capping only bite once equity is driven near the trailing floor, so a
# small ``amount`` (buffer) is used to reach the barrier in a bounded path.


def test_kernel_and_reference_agree_bitwise_with_feasibility_active():
    # Random paths through both implementations under an active projection; the
    # executed path (hence code, payouts, days) must match bit-for-bit (§G6). The
    # small buffer + wide return spread drive clips, withers (CAPPED_OUT) and real
    # breaches (ret < -1R -> realized loss > planned) across the trials.
    feas = FeasibilitySpec(q_min=1.0, unit_loss=1.0, alpha=0.8)
    rng = np.random.default_rng(20260830)
    seen = set()
    for update in (Timing.CONTINUOUS, Timing.EOD):
        for check in (Timing.CONTINUOUS, Timing.EOD):
            cp = _phase(amount=40.0, update=update, check=check)
            for trial in range(60):
                ndays = int(rng.integers(3, 12))
                tpd = int(rng.integers(1, 5))
                n = ndays * tpd
                ret = rng.normal(0.1, 1.4, size=n)  # R units; sometimes < -1R
                day = np.repeat(np.arange(ndays), tpd)[:n]
                low = -np.abs(rng.normal(0.6, 0.5, size=n))  # adverse excursion (<=0)
                policy = np.array([rng.uniform(1.0, 60.0)])  # desired 1..60 units
                k, r = _run_both(cp, ret, day, low, 1.0, policy, 50_000.0, feas)
                assert k[0] == r.code, (update, check, trial, k[0], r.code)
                assert k[1] == r.payout_amounts  # exact float equality
                assert k[2] == r.payout_days
                assert k[3] == r.total_trading_days
                seen.add(k[0])
    # the random suite actually exercised the withered + breach terminals, not just
    # survival — otherwise "parity" would be vacuous on those paths.
    assert _CAPPED in seen and _FAIL_DD in seen


def test_capped_out_path_matches_between_implementations():
    # A steadily-losing path with a big requested size drives the buffer under
    # L_min -> CAPPED_OUT, and both implementations agree on the exact terminal.
    feas = FeasibilitySpec(q_min=1.0, unit_loss=1.0, alpha=1.0)
    cp = _phase(amount=30.0, target=1e9, update=Timing.CONTINUOUS,
                check=Timing.CONTINUOUS)
    ret = [-0.5] * 40
    day = list(range(40))
    low = [-0.5] * 40
    k, r = _run_both(cp, ret, day, low, 1.0, [50.0], 50_000.0, feas)  # desired 50 units
    assert k[0] == r.code
    assert k[0] in (_CAPPED, _FAIL_DD)  # a terminal, agreed by both


# --------------------------------------------------------------------------- #
# Clipping is NOT failure; non-tradability IS a distinct terminal              #
# --------------------------------------------------------------------------- #


def test_clipping_is_not_failure_the_account_trades_on():
    # A large desired size against a healthy buffer is clipped every trade, but on
    # a winning drift the account survives and the clips are just recorded.
    feas = FeasibilitySpec(q_min=1.0, unit_loss=1.0, alpha=0.5)
    cp = _phase(amount=2000.0, target=1e9)  # unreachable target -> runs to TIMED_OUT
    ret = [0.3] * 30  # mild positive drift: buffer grows, never withers
    day = list(range(30))
    low = [-0.2] * 30
    diag = FeasibilityDiag()
    code, amounts, days, nd = simulate_one_phase(
        cp, np.array(ret), np.array(day, np.int32), np.array(low),
        1.0, np.array([5000.0]), 50_000.0, feasibility=feas, diag_out=diag
    )
    assert code == int(ExitCode.TIMED_OUT)  # survived; clipping did not end it
    assert diag.reduced > 0  # desired 5000 >> cap (~1000), so every trade was clipped
    assert not diag.capped_out and not diag.breached
    assert diag.fraction_size_reduced > 0.0


def test_capped_out_is_distinct_from_a_drawdown_breach():
    # Slow bleed with feasibility ON: the projection shrinks each trade so the
    # account WITHERS (CAPPED_OUT) strictly before it could breach the floor.
    feas = FeasibilitySpec(q_min=1.0, unit_loss=1.0, alpha=1.0)
    cp = _phase(amount=30.0, target=1e9, update=Timing.EOD, check=Timing.CONTINUOUS)
    ret = [-0.4] * 60
    day = list(range(60))
    low = [-0.4] * 60  # realized loss (0.4R) < planned (1R) -> no gap breach, pure wither
    diag = FeasibilityDiag()
    code, *_ = simulate_one_phase(
        cp, np.array(ret), np.array(day, np.int32), np.array(low),
        1.0, np.array([30.0]), 50_000.0, feasibility=feas, diag_out=diag  # desired 30 units
    )
    assert code == _CAPPED  # withered, not breached
    assert diag.capped_out and not diag.breached
    assert diag.time_to_nontradable >= 0
    # the withering trade's sub-threshold buffer is folded into the min stats (§16.9)
    # -- a withered attempt must NOT report a min tradability ratio >= 1 (backwards).
    assert diag.min_tradability_ratio < 1.0
    assert diag.min_buffer < float("inf")


def test_r_to_zero_policy_still_withers_exploit_closed():
    # The anti-exploit core (§16.4b): even a microscopic desired size is floored up
    # to one contract, which risks L_min each trade, so on a losing drift the
    # account still reaches CAPPED_OUT rather than surviving indefinitely.
    feas = FeasibilitySpec(q_min=1.0, unit_loss=1.0, alpha=1.0)
    cp = _phase(amount=20.0, target=1e9, update=Timing.CONTINUOUS,
                check=Timing.CONTINUOUS)
    ret = [-0.6] * 100
    day = list(range(100))
    low = [-0.6] * 100
    code, _a, _d, nd = simulate_one_phase(
        cp, np.array(ret), np.array(day, np.int32), np.array(low),
        1.0, np.array([1e-9]), 50_000.0, feasibility=feas  # ~zero desired risk
    )
    assert code == _CAPPED       # it did NOT survive on microscopic risk (floored to q_min)
    assert nd < 100              # and it terminated before the path ran out


def test_engine_threads_feasibility_and_changes_the_distribution():
    # End-to-end: Engine.run with a feasibility spec drives the projection through
    # the batch and changes the outcome distribution (sizes are clipped near the
    # floor, some attempts wither to CAPPED_OUT); without the spec, no CAPPED_OUT.
    from propfirm_engine.data import preprocess
    from propfirm_engine.engine import Engine, RunConfig
    from propfirm_engine.model import Account
    from propfirm_engine.resampling import IIDDayBootstrap
    from propfirm_engine.synthetic import IIDGenerator

    mll = TrailingDrawdownRule(300.0, update_timing=Timing.EOD,
                               check_timing=Timing.CONTINUOUS)
    acct = Account("f", 50_000, phases=(Phase("eval", "eval",
                   (ProfitTargetRule(1e9), mll)),), eval_fee=100.0)
    ds = preprocess(IIDGenerator(win_rate=0.42, rr=1.0).generate(120, seed=3).rows)
    cfg = RunConfig(intraday_mode="summary_approximation", n_paths=400, L_eval=40, seed=1, size_base=100.0,
                    resampler=IIDDayBootstrap())
    # A full loser floats to ~1R below entry (the corrected intraday-low convention).
    # unit_loss=2 is a deliberately CONSERVATIVE 2R stop here (clips earlier than the
    # real 1R), and alpha 0.8 leaves a cushion so a capped loss lands ABOVE the floor
    # -> attempts WITHER rather than breach.
    feas = FeasibilitySpec(q_min=1.0, unit_loss=2.0, alpha=0.8)
    eng = Engine()
    with_feas = eng.run(acct, ds, cfg, feasibility=feas, policy_params=[5.0])
    without = eng.run(acct, ds, cfg, policy_params=[5.0])
    assert not np.array_equal(with_feas.code, without.code)  # it changed the paths
    assert int(np.sum(with_feas.code == _CAPPED)) > 0  # some withered
    assert int(np.sum(without.code == _CAPPED)) == 0    # never without the spec
    # the EVAL-phase §16.9 aggregate is surfaced (eval carries the tightest buffer),
    # not just the funded one; it counts the withered attempts.
    assert with_feas.eval_feas_agg is not None
    assert with_feas.eval_feas_agg.nontradable_failures > 0
    assert without.eval_feas_agg is None  # no aggregate without a spec


def test_batch_aggregates_feasibility_diagnostics():
    # A direct batch call fills the §16.9 aggregate (nontradable vs breach counts,
    # mean clip fraction); without a spec the aggregate is absent.
    from propfirm_engine.data import preprocess
    from propfirm_engine.resampling import IIDDayBootstrap
    from propfirm_engine.simulate import simulate_phase_batch
    from propfirm_engine.synthetic import IIDGenerator

    cp = _phase(amount=300.0, target=1e9, update=Timing.EOD,
                check=Timing.CONTINUOUS)
    ds = preprocess(IIDGenerator(win_rate=0.42, rr=1.0).generate(120, seed=3).rows)
    paths = IIDDayBootstrap().generate(ds.n_days, 40, 300, seed=2)
    feas = FeasibilitySpec(q_min=1.0, unit_loss=2.0, alpha=0.8)
    res = simulate_phase_batch(cp, ds, paths, 100.0, np.array([5.0]), 50_000.0,
                               feasibility=feas)
    d = res.feas_agg.as_dict()
    assert d["attempts"] == 300
    assert d["nontradable_failures"] + d["actual_breach_failures"] <= 300
    assert 0.0 <= d["mean_fraction_size_reduced"] <= 1.0
    # no spec -> no aggregate
    res2 = simulate_phase_batch(cp, ds, paths, 100.0, np.array([5.0]), 50_000.0)
    assert res2.feas_agg is None


def test_feasibility_none_is_untouched_when_no_trailing_floor():
    # A phase with no trailing rule: even with a spec, the projection is inactive
    # (firm-agnostic), so behavior equals the no-feasibility case.
    cp = compile_phase(Phase("eval", "eval", (ProfitTargetRule(300.0),)))
    feas = FeasibilitySpec(q_min=1.0, unit_loss=100.0, alpha=0.5)
    args = (np.array([1.0, 1.0, 1.0]), np.array([0, 1, 2], np.int32),
            np.array([0.0, 0.0, 0.0]), 100.0, np.array([1.0]), 50_000.0)
    with_feas = simulate_one_phase(cp, *args, feasibility=feas)
    without = simulate_one_phase(cp, *args, feasibility=None)
    assert with_feas == without  # projection never binds without a trailing floor
def test_fixed_cost_is_reserved_before_rounding():
    q, capped, *_ = project_position(3, 200, 1, 100, 1, fixed_cost=1)
    assert q == 1 and not capped
    q, capped, *_ = project_position(1, 100, 1, 100, 1, fixed_cost=1)
    assert q == 0 and capped


def test_cost_aware_projection_is_used_by_both_executors():
    cp = compile_phase(Phase("eval", "eval", (
        ProfitTargetRule(1000),
        TrailingDrawdownRule(200, update_timing=Timing.EOD),
    )))
    feasibility = FeasibilitySpec(q_min=1, unit_loss=100)
    args = (cp, np.array([-100.0]), np.array([0], dtype=np.int32),
            np.array([-100.0]), 3.0, np.array([1.0]), 0.0)
    ref = simulate_reference(*args, feasibility=feasibility, trade_cost=1)
    fast = simulate_one_phase(*args, feasibility=feasibility, trade_cost=1)
    assert ref.code == int(ExitCode.TIMED_OUT)
    assert fast[0] == ref.code
