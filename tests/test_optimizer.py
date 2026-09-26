"""The bet-sizing optimizer (BUILD_SPEC Step 14; ARCHITECTURE §16).

Pins the Step-14 optimizer contract at Tier 1: the policy searches over the sizing
hook only (never the strategy); the objective is the renewal reward rate under
survival constraints; the ``r → 0`` exploit is closed by feasibility; Common Random
Numbers make candidate comparisons honest; and the ONLY reported number is
out-of-sample (nested/walk-forward), never the data the search was fitted on.
"""

from __future__ import annotations

import numpy as np
import pytest

from propfirm_engine.data import preprocess
from propfirm_engine.engine import Engine, RunConfig
from propfirm_engine.enums import Action, ExitCode, Timing
from propfirm_engine.feasibility import FeasibilitySpec
from propfirm_engine.model import Account, Phase
from propfirm_engine.optimizer import (
    CMAES,
    OptConfig,
    PolicySpace,
    RenewalObjective,
    RollingResult,
    evaluate_policy,
    factorized_optimize,
    optimize,
    rolling_walk_forward,
    walk_forward,
)
from propfirm_engine.resampling import IIDDayBootstrap
from propfirm_engine.rules import (
    ConsistencyGateRule,
    MinimumWinningDaysRule,
    ProfitTargetRule,
    TrailingDrawdownRule,
)
from propfirm_engine.schema import PayoutSchema
from propfirm_engine.synthetic import IIDGenerator


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


def _funded_account() -> Account:
    """Eval + funded with a trailing DD, a payout schema and fees, so sizing has a
    real trade-off: too big -> breach/wither, too small -> misses the target/payouts.
    """
    mll = TrailingDrawdownRule(2_000.0, update_timing=Timing.EOD,
                               check_timing=Timing.CONTINUOUS)
    eval_phase = Phase("eval", "eval",
                       (ProfitTargetRule(3_000.0), mll,
                        ConsistencyGateRule(0.6, gate=Action.PASS)))
    schema = PayoutSchema(dollar_cap=(2_000.0,), split=0.9, max_payouts=3,
                          cap_fraction=0.5, min_request=1.0)
    funded_phase = Phase("funded", "funded",
                         (mll, MinimumWinningDaysRule(3, 150.0)),
                         payout_schema=schema)
    return Account("opt", 50_000, phases=(eval_phase, funded_phase),
                   eval_fee=150.0, activation_fee=100.0)


def _dataset(seed):
    return preprocess(IIDGenerator(win_rate=0.5, rr=1.5).generate(160, seed=seed).rows)


def _cfg(**kw):
    base = dict(n_paths=300, L_eval=25, L_funded=45, seed=1, size_base=100.0,
                resampler=IIDDayBootstrap())
    base["intraday_mode"] = "summary_approximation"  # legacy summary-model scenarios
    base.update(kw)
    return RunConfig(**base)


def _fast_opt(**kw):
    base = dict(sigma0=0.5, max_gen=6, popsize=6, seed=0,
                screen_paths=150, select_paths=400)
    base.update(kw)
    return OptConfig(**base)


# --------------------------------------------------------------------------- #
# Tier-1 policy space                                                          #
# --------------------------------------------------------------------------- #


def test_policy_space_has_one_multiplier_per_regime():
    space = PolicySpace(lo=0.0, hi=5.0)
    assert space.n_params == 5  # eval + 4 funded regimes
    pol = space.to_policy([0.5, 1.5, 2.5, 3.5, 4.5])
    assert pol.shape[0] == 5  # length the kernel indexes by regime index
    assert list(pol) == [0.5, 1.5, 2.5, 3.5, 4.5]  # contiguous, no gaps
    # bounds are mechanical, enforced on mapping
    assert np.all(space.to_policy([-9, 9, 9, 9, 9]) <= 5.0)
    assert np.all(space.to_policy([-9, 9, 9, 9, 9]) >= 0.0)
    # a short (length-1 baseline) theta pads with neutral 1.0
    assert list(space.to_policy([3.0])) == [3.0, 1.0, 1.0, 1.0, 1.0]


def test_length_one_neutral_policy_reproduces_constant_size():
    # to_policy(x0) is all-ones -> multiplies size_base by 1 in every stage, i.e. the
    # constant-size baseline the optimizer starts from (§16.5).
    space = PolicySpace()
    assert np.all(space.to_policy(space.x0()) == 1.0)


# --------------------------------------------------------------------------- #
# CMA-ES core                                                                  #
# --------------------------------------------------------------------------- #


def test_cmaes_maximizes_a_concave_function():
    target = np.array([1.5, -2.0, 0.7])

    def f(x):  # concave, unique max at `target`
        return -float(np.sum((x - target) ** 2))

    es = CMAES(np.zeros(3), sigma0=1.0, seed=0, max_gen=60)
    res = es.optimize(f)
    assert np.allclose(res.x, target, atol=0.1)
    assert res.score > -0.05
    # the search improved over its generations
    assert res.history[-1] > res.history[0]


def test_cmaes_respects_bounds():
    def f(x):
        return float(np.sum(x))  # wants to run to +inf, must be capped by bounds

    es = CMAES(np.zeros(2), sigma0=0.5, bounds=(-1.0, 1.0), seed=1, max_gen=40)
    res = es.optimize(f)
    assert np.all(res.x <= 1.0 + 1e-9) and np.all(res.x >= -1.0 - 1e-9)


# --------------------------------------------------------------------------- #
# Common Random Numbers / determinism                                          #
# --------------------------------------------------------------------------- #


def test_evaluate_policy_is_deterministic_common_random_numbers():
    acct, ds, cfg = _funded_account(), _dataset(11), _cfg()
    space, obj, eng = PolicySpace(), RenewalObjective(), Engine()
    a = evaluate_policy(eng, acct, ds, cfg, [1.0, 1.0, 1.0, 1.0], space, obj)
    b = evaluate_policy(eng, acct, ds, cfg, [1.0, 1.0, 1.0, 1.0], space, obj)
    assert a == b  # same policy + same seed -> identical score (CRN)
    # a different policy on the SAME config is evaluated on the SAME paths
    c = evaluate_policy(eng, acct, ds, cfg, [0.5, 0.5, 0.5, 0.5], space, obj)
    assert c != a  # sizing changed the outcome (paths held fixed)


# --------------------------------------------------------------------------- #
# optimize() — search on train only                                            #
# --------------------------------------------------------------------------- #


def test_optimize_selects_at_full_fidelity_including_baseline():
    acct, train = _funded_account(), _dataset(3)
    out = optimize(acct, train, _cfg(), opt_config=_fast_opt(),
                   objective=RenewalObjective())
    assert out.policy.shape[0] == 5
    assert np.isfinite(out.train_score) and np.isfinite(out.baseline_score)
    assert out.train_score >= out.baseline_score
    assert out.train_score == max(score for _, score in out.selection_scores)
    assert np.all(out.theta >= out.space.lo - 1e-9)
    assert np.all(out.theta <= out.space.hi + 1e-9)


def test_account_aware_bound_makes_reachable_size_independent_of_size_base():
    # The multiplier ceiling is the size that risks the whole MLL in one trade, so
    # the reachable EFFECTIVE size (size_base * hi) is an account property (MLL/unit),
    # independent of the size_base knob -- which is why the optimizer's result stops
    # tracking size_base.
    from propfirm_engine.optimizer import policy_space_for
    acct = _funded_account()  # trailing MLL = 2000
    reach = []
    for sb in (25.0, 100.0, 400.0):
        sp = policy_space_for(acct, size_base=sb)
        reach.append(sb * sp.hi)  # max effective size
    assert all(abs(x - reach[0]) < 1e-6 for x in reach)  # identical across size_base
    assert abs(reach[0] - 2000.0) < 1e-6  # = MLL / unit_loss(=1)
    # feasibility unit_loss (the stop) scales the ceiling down
    from propfirm_engine.feasibility import FeasibilitySpec
    sp2 = policy_space_for(acct, size_base=100.0,
                           feasibility=FeasibilitySpec(q_min=1.0, unit_loss=2.0))
    assert abs(100.0 * sp2.hi - 1000.0) < 1e-6  # MLL / 2


def test_searched_candidates_are_invariant_to_size_base():
    # Search is a reparameterization of a fixed effective-size range. The neutral
    # baseline is NOT: its effective size is size_base, clipped to the range.
    # Final selection may legitimately change when that additional candidate wins.
    # Do not enforce invariance by deliberately returning a worse policy.
    from propfirm_engine.optimizer import policy_space_for
    acct, train = _funded_account(), _dataset(3)  # MLL = 2000
    scores, effs = [], []
    for sb in (50.0, 400.0, 3000.0):  # last one is > MLL
        out = optimize(acct, train, _cfg(size_base=sb), opt_config=_fast_opt(),
                       objective=RenewalObjective())
        candidates = out.selection_scores[1:]  # exclude size_base-dependent baseline
        theta, score = max(candidates, key=lambda item: item[1])
        scores.append(score)
        effs.append(sb * theta)
        assert out.train_score >= out.baseline_score
        assert out.train_score >= score
    assert all(abs(s - scores[0]) < 1e-6 for s in scores)
    for e in effs[1:]:
        assert np.allclose(e, effs[0], rtol=1e-3, atol=1.0)  # identical effective sizing


def test_no_trailing_floor_falls_back_to_default_bound():
    from propfirm_engine.optimizer import policy_space_for
    acct = Account("nf", 50_000, phases=(Phase("eval", "eval",
                   (ProfitTargetRule(500.0),)),), eval_fee=100.0)
    assert policy_space_for(acct, size_base=100.0).hi == PolicySpace().hi


def test_cmaes_improves_the_real_objective_on_a_shaped_problem():
    # A search-QUALITY check on the real evaluation path (not a synthetic function):
    # start CMA-ES from a deliberately bad multiplier and confirm its best-scored
    # policy improves on that starting point over the generations.
    acct, train, cfg = _funded_account(), _dataset(3), _cfg(n_paths=200)
    space, obj, eng = PolicySpace(), RenewalObjective(), Engine()
    bad = evaluate_policy(eng, acct, train, cfg, [4.5] * 5, space, obj)  # over-sized
    from propfirm_engine.optimizer import CMAES

    def f(theta):
        return evaluate_policy(eng, acct, train, cfg, theta, space, obj)

    es = CMAES(np.array([4.5] * 5), sigma0=0.6, bounds=(0.0, 5.0), seed=0, max_gen=8)
    res = es.optimize(f)
    assert res.score > bad  # the search moved off the bad start toward something better


# --------------------------------------------------------------------------- #
# The r -> 0 exploit is closed by feasibility (§16.4b/§16.2)                    #
# --------------------------------------------------------------------------- #


def _survival_account(dd=12.0) -> Account:
    """Losing-edge survival account: an unreachable target, a small trailing DD, so
    the only terminals are TIMED_OUT (survived) or, under feasibility, CAPPED_OUT
    (withered). Small DD + losing edge = the account grinds toward its floor."""
    mll = TrailingDrawdownRule(dd, update_timing=Timing.EOD,
                               check_timing=Timing.CONTINUOUS)
    return Account("surv", 50_000,
                   phases=(Phase("eval", "eval", (ProfitTargetRule(1e9), mll)),),
                   eval_fee=100.0)


def test_feasibility_is_what_closes_the_r_to_zero_exploit():
    # The §16.4b exploit: without a minimum executable size, a vanishing-risk policy
    # achieves *near-perfect survival that is economically useless* — it barely
    # trades, so it never breaches. This test shows feasibility is precisely what
    # removes that hiding place, using a no-feasibility CONTROL (the missing control
    # the earlier version lacked).
    acct = _survival_account(dd=12.0)
    ds = preprocess(IIDGenerator(win_rate=0.4, rr=1.0).generate(160, seed=3).rows)
    cfg = _cfg(n_paths=400, L_eval=50)
    feas = FeasibilitySpec(q_min=1.0, unit_loss=2.0, alpha=0.8)
    eng = Engine()
    near0 = [1e-6] * 5

    with_feas = eng.run(acct, ds, cfg, policy_params=near0, feasibility=feas)
    without = eng.run(acct, ds, cfg, policy_params=near0)  # no min-size floor
    capped = int(ExitCode.CAPPED_OUT)
    survived = int(ExitCode.TIMED_OUT)

    # WITHOUT feasibility: the ~0 policy trades ~nothing and SURVIVES everything —
    # the useless-survival the optimizer would exploit. Zero withering.
    assert int(np.sum(without.code == capped)) == 0
    assert int(np.sum(without.code == survived)) == without.n_attempts
    # WITH feasibility: the ~0 desire is floored up to q_min, risks L_min each trade,
    # and on the losing drift the account WITHERS -> CAPPED_OUT. The exploit is gone.
    assert int(np.sum(with_feas.code == capped)) > 0.8 * with_feas.n_attempts


def test_r_to_zero_is_ranked_below_a_sensible_policy():
    # And under the renewal objective a sensible interior policy outscores the ~0
    # policy (which now withers and earns nothing) — so the search won't select it.
    acct, ds, cfg = _funded_account(), _dataset(7), _cfg()
    feas = FeasibilitySpec(q_min=1.0, unit_loss=2.0, alpha=0.8)
    space, obj, eng = PolicySpace(), RenewalObjective(), Engine()
    near_zero = evaluate_policy(eng, acct, ds, cfg, [1e-6] * 5, space, obj, feas)
    interior = evaluate_policy(eng, acct, ds, cfg, [1.0] * 5, space, obj, feas)
    assert interior > near_zero


# --------------------------------------------------------------------------- #
# Nested / walk-forward OOS — the only reported number (§16.7)                  #
# --------------------------------------------------------------------------- #


def test_walk_forward_reports_out_of_sample_only():
    acct = _funded_account()
    train, test = _dataset(3), _dataset(9999)  # independent partitions
    res = walk_forward(acct, train, test, _cfg(), opt_config=_fast_opt())
    # a real OOS number is produced, distinct from the training score object
    assert np.isfinite(res.oos_score)
    assert np.isfinite(res.baseline_oos_score)
    # the fitted policy is exactly what optimize() found on TRAIN (the search never
    # touched `test`); walk_forward only *scores* on test.
    fit = optimize(acct, train, _cfg(), opt_config=_fast_opt())
    assert np.allclose(res.theta, fit.theta)
    # oos_improvement is the honest, non-overfit headline
    assert res.oos_improvement == res.oos_score - res.baseline_oos_score


def test_walk_forward_oos_ladder_band():
    acct = _funded_account()
    train = _dataset(3)
    test = _dataset(9999)
    ladder = {"iid": _dataset(5), "alt": _dataset(6)}
    res = walk_forward(acct, train, test, _cfg(), opt_config=_fast_opt(),
                       ladder_datasets=ladder)
    assert set(res.oos_ladder) == {"iid", "alt"}  # a band across held-out rungs
    assert all(np.isfinite(v) for v in res.oos_ladder.values())


# --------------------------------------------------------------------------- #
# Rolling time-separated OOS + block bootstrap (fix #1: honest harness)         #
# --------------------------------------------------------------------------- #


def _long_dataset(days, seed=1):
    return preprocess(IIDGenerator(win_rate=0.55, rr=1.2).generate(days, seed=seed).rows)


def test_rolling_walk_forward_folds_are_expanding_and_time_separated():
    acct, ds = _funded_account(), _long_dataset(600, seed=3)
    res = rolling_walk_forward(acct, ds, _cfg(), n_folds=3, warmup_frac=0.4,
                               mean_block=5.0, seeds=(0, 1), min_test_days=50,
                               opt_config=_fast_opt(), objective=RenewalObjective())
    assert isinstance(res, RollingResult)
    assert len(res.folds) == 3
    assert res.mean_block == 5.0 and res.seeds == (0, 1)
    prev_end = None
    for i, f in enumerate(res.folds):
        assert f.train_days == (0, f.test_days[0])       # train is expanding [0, t)
        assert f.test_days[0] < f.test_days[1]           # non-empty, strictly-later test
        if prev_end is not None:
            assert f.test_days[0] == prev_end            # contiguous, non-overlapping
        prev_end = f.test_days[1]
        assert len(f.oos) == 2                           # one OOS score per seed
    assert res.folds[-1].test_days[1] == ds.n_days       # folds cover through the end
    # the harness reports a DISTRIBUTION, not one number
    assert len(res.oos_all) == 3 * 2
    assert np.isfinite(res.oos_median) and res.oos_spread >= 0.0
    assert len(res.fold_medians) == 3


def test_rolling_walk_forward_block_bootstrap_is_plumbed():
    # mean_block > 1 must actually change the resampling (block vs IID single days),
    # so the OOS distribution differs from the IID-day run on the same folds/seed.
    acct, ds = _funded_account(), _long_dataset(600, seed=4)
    common = dict(n_folds=2, warmup_frac=0.4, seeds=(0,), min_test_days=50,
                  opt_config=_fast_opt(), objective=RenewalObjective())
    iid = rolling_walk_forward(acct, ds, _cfg(), mean_block=1.0, **common)
    block = rolling_walk_forward(acct, ds, _cfg(), mean_block=10.0, **common)
    assert iid.mean_block == 1.0 and block.mean_block == 10.0
    assert iid.oos_all != block.oos_all  # the block bootstrap genuinely resamples differently


def test_rolling_walk_forward_rejects_too_many_folds():
    acct, ds = _funded_account(), _long_dataset(300, seed=5)
    with pytest.raises(ValueError):
        rolling_walk_forward(acct, ds, _cfg(), n_folds=50, min_test_days=60,
                             opt_config=_fast_opt())


# --------------------------------------------------------------------------- #
# Factorized (funded-first) fit + fee-less funded objective (§16.10)            #
# --------------------------------------------------------------------------- #


def test_renewal_objective_include_fees_flag_drops_both_fees():
    # include_fees=False -> pure E[net_payout]/E[T] (no eval/activation fee); the funded
    # leg's V_funded must never carry the real-money boundary fees.
    acct, ds, cfg = _funded_account(), _dataset(7), _cfg()
    o = Engine().run(acct, ds, cfg, policy_params=PolicySpace().to_policy([2.0] * 5),
                     feasibility=None)
    from propfirm_engine.statistics import attributable_fee
    twk = float(np.mean(o.total_trading_days)) / o.trading_days_per_week
    pure = float(np.mean(o.net_payout)) / twk
    withfee = float(np.mean(o.net_payout - attributable_fee(o))) / twk
    assert RenewalObjective(include_fees=False).value(o) == pytest.approx(pure, rel=1e-9)
    assert RenewalObjective(include_fees=True).value(o) == pytest.approx(withfee, rel=1e-9)


def test_legacy_factorized_entry_reports_joint_renewal():
    # Compatibility entry uses the joint engine; its reported training objective
    # should agree with a fresh Monte Carlo evaluation within sampling noise.
    acct, ds = _funded_account(), _long_dataset(600, seed=3)
    oc = _fast_opt(select_paths=1500)
    out = factorized_optimize(acct, ds, _cfg(), space=PolicySpace(lo=0.0, hi=20.0),
                              opt_config=oc, feasibility=None)
    assert out.policy.shape[0] == 5
    assert np.all(out.theta >= out.space.lo - 1e-9) and np.all(out.theta <= out.space.hi + 1e-9)
    joint = Engine().run(acct, ds, _cfg(n_paths=4000, seed=99), policy_params=out.policy)
    r_joint = RenewalObjective().value(joint)
    # both estimate the same whole-cycle rate; agree up to Monte-Carlo noise
    assert out.train_score == pytest.approx(r_joint, rel=0.20)


def test_factorized_optimize_plugs_into_walk_forward_as_a_fitter():
    acct = _funded_account()
    train, test = _long_dataset(500, seed=3), _long_dataset(220, seed=8)
    res = walk_forward(acct, train, test, _cfg(), opt_config=_fast_opt(select_paths=1200),
                       objective=RenewalObjective(), fitter=factorized_optimize)
    assert res.policy.shape[0] == 5
    assert np.isfinite(res.oos_score) and np.isfinite(res.baseline_oos_score)


# --------------------------------------------------------------------------- #
# CVaR (tail-scored) renewal objective (§16.11)                                #
# --------------------------------------------------------------------------- #


def test_cvar_q_ge_1_is_exactly_the_nominal_rate():
    # q>=1 must reproduce E[R]/E[T] bit-for-bit (the mean anchor of the q sweep).
    acct, ds, cfg = _funded_account(), _dataset(7), _cfg()
    o = Engine().run(acct, ds, cfg, policy_params=PolicySpace().to_policy([2.0] * 5))
    assert RenewalObjective(cvar_q=1.0).value(o) == RenewalObjective().value(o)


def test_cvar_left_tail_is_ratio_correct_and_not_above_the_mean():
    obj = RenewalObjective(cvar_q=0.25, n_boot=300, boot_seed=1)
    reward = np.array([10.0, 12.0, -5.0, 8.0, 40.0, -3.0, 9.0, 11.0])
    time = np.array([2.0, 3.0, 1.0, 0.0, 4.0, 1.0, 2.0, 3.0])  # a T_i == 0 (per-path R/T would blow up)
    nominal = float(np.sum(reward)) / float(np.sum(time))  # ratio-of-sums, ratio-correct
    cvar = obj._rate(reward, time)
    assert np.isfinite(cvar)                      # the T_i==0 attempt did NOT blow it up
    assert cvar <= nominal + 1e-9                 # CVaR of the LEFT tail sits below the mean
    assert RenewalObjective(cvar_q=1.0)._rate(reward, time) == pytest.approx(
        float(np.mean(reward)) / float(np.mean(time)))


def test_cvar_objective_is_crn_deterministic():
    # fixed boot_seed -> identical resample indices -> identical score across calls
    acct, ds, cfg = _funded_account(), _dataset(4), _cfg()
    o = Engine().run(acct, ds, cfg, policy_params=PolicySpace().to_policy([3.0] * 5))
    obj = RenewalObjective(cvar_q=0.25, n_boot=200)
    assert obj.value(o) == obj.value(o)


def test_factorized_optimize_rejects_separate_funded_objective():
    acct, ds = _funded_account(), _long_dataset(500, seed=3)
    with pytest.raises(ValueError, match="whole-account objective"):
        factorized_optimize(acct, ds, _cfg(), opt_config=_fast_opt(),
                            funded_objective=RenewalObjective(include_fees=False, cvar_q=0.25))
