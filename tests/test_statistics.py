"""Step 10 — single-attempt decision statistics (BUILD_SPEC Step 10, ARCHITECTURE §14)."""

from __future__ import annotations

import numpy as np
import pytest

from propfirm_engine.engine import Outcomes
from propfirm_engine.enums import ExitCode
from propfirm_engine.objectives import expected_payout_st_profitable
from propfirm_engine.results import Results
from propfirm_engine.statistics import (
    attributable_fee,
    bootstrap_ci,
    mean_payout,
    pass_rate,
    payoff_quantiles,
    payout_count_dist,
    payout_velocity,
    prob_profitable,
    return_on_fee,
    return_on_fee_per_year,
    time_to_first_payout,
    wilson_ci,
)


def _outcomes(*, net_payout, payouts_taken=None, first_payout_day=None,
              total_trading_days=None, reached_funded=None, code=None,
              eval_fee=100.0, activation_fee=0.0, max_payouts=3,
              trading_days_per_week=5.0):
    net_payout = np.asarray(net_payout, dtype=np.float64)
    b = len(net_payout)
    if payouts_taken is None:
        payouts_taken = (net_payout > 0).astype(np.int32)
    if first_payout_day is None:
        first_payout_day = np.where(np.asarray(payouts_taken) > 0, 5, -1).astype(np.int32)
    if total_trading_days is None:
        total_trading_days = np.full(b, 20, dtype=np.int32)
    if reached_funded is None:
        reached_funded = np.ones(b, dtype=bool)
    if code is None:
        code = np.full(b, int(ExitCode.TIMED_OUT), dtype=np.int32)
    return Outcomes(
        code=np.asarray(code, np.int32),
        reached_funded=np.asarray(reached_funded, bool),
        net_payout=net_payout,
        payouts_taken=np.asarray(payouts_taken, np.int32),
        first_payout_day=np.asarray(first_payout_day, np.int32),
        total_trading_days=np.asarray(total_trading_days, np.int32),
        size=50_000, size_base=100.0, max_payouts=max_payouts,
        eval_fee=eval_fee, activation_fee=activation_fee,
        trading_days_per_week=trading_days_per_week, fingerprint="abc123",
    )


# --- path-dependent fees (§H1) ---------------------------------------------- #


def test_attributable_fee_is_path_dependent():
    o = _outcomes(net_payout=[0.0, 0.0], reached_funded=[True, False],
                  eval_fee=150.0, activation_fee=50.0)
    fee = attributable_fee(o)
    np.testing.assert_allclose(fee, [200.0, 150.0])  # activation only when reached funded


# --- P(profitable) is not recoverable from the mean ------------------------- #


def test_prob_profitable_counts_net_above_attributable_fee():
    o = _outcomes(net_payout=[0.0, 200.0], eval_fee=100.0, activation_fee=0.0)
    assert prob_profitable(o) == 0.5  # only the 200 beats the 100 fee


def test_prob_profitable_not_recoverable_from_mean_payout():
    # Two batches with EQUAL mean payout but different profitable fractions.
    a = _outcomes(net_payout=[0.0, 200.0], eval_fee=100.0)  # mean 100, P=0.5
    b = _outcomes(net_payout=[100.0, 100.0], eval_fee=100.0)  # mean 100, P=0.0
    assert mean_payout(a) == mean_payout(b)
    assert prob_profitable(a) != prob_profitable(b)


# --- payout-count distribution (§14.1) -------------------------------------- #


def test_payout_count_distribution_sums_to_one_and_matches_counts():
    o = _outcomes(net_payout=[0, 10, 10, 10, 10], payouts_taken=[0, 1, 1, 2, 3],
                  max_payouts=3)
    dist = payout_count_dist(o)
    assert len(dist) == 4  # P(0)..P(3)
    np.testing.assert_allclose(dist, [1 / 5, 2 / 5, 1 / 5, 1 / 5])
    assert abs(dist.sum() - 1.0) < 1e-12


# --- return on fee ---------------------------------------------------------- #


def test_return_on_fee_is_per_attempt():
    o = _outcomes(net_payout=[0.0, 300.0], eval_fee=100.0, activation_fee=0.0)
    np.testing.assert_allclose(return_on_fee(o), [0.0, 3.0])


# --- time axis uses cadence, never a dataset fraction (§14.2) ---------------- #


def test_doubling_cadence_halves_calendar_duration_and_doubles_the_rate():
    slow = _outcomes(net_payout=[500.0], total_trading_days=[20],
                     trading_days_per_week=5.0)
    fast = _outcomes(net_payout=[500.0], total_trading_days=[20],
                     trading_days_per_week=10.0)  # same days, twice the cadence
    # twice the trading-days-per-week -> half the calendar duration -> twice the rate
    assert payout_velocity(fast) == np.float64(2.0) * payout_velocity(slow)
    assert return_on_fee_per_year(fast) == np.float64(2.0) * return_on_fee_per_year(slow)


def test_a_path_longer_than_the_source_still_has_a_finite_duration():
    # total_trading_days can exceed the source day count; duration comes from cadence,
    # not a fraction of the dataset, so it stays finite and well-defined.
    o = _outcomes(net_payout=[1000.0], total_trading_days=[9999],
                  trading_days_per_week=5.0)
    weeks = time_to_first_payout(o)
    assert np.all(np.isfinite(weeks))
    assert np.isfinite(payout_velocity(o))


def test_time_to_first_payout_only_counts_attempts_with_a_payout():
    o = _outcomes(net_payout=[0.0, 250.0], payouts_taken=[0, 1],
                  first_payout_day=[-1, 10], trading_days_per_week=5.0)
    weeks = time_to_first_payout(o)
    assert len(weeks) == 1  # only the attempt that took a payout
    assert weeks[0] == 11 / 5.0


# --- pass_rate is eval-only, never a funded success measure (§H3) ------------ #


def test_pass_rate_is_not_a_funded_success_metric():
    # A funded batch: economically successful (payouts banked) but codes are
    # MAXED_OUT/TIMED_OUT, never PASSED -> pass_rate is 0 despite real income.
    o = _outcomes(net_payout=[500.0, 800.0], payouts_taken=[2, 3],
                  code=[int(ExitCode.MAXED_OUT), int(ExitCode.TIMED_OUT)])
    assert pass_rate(o.code) == 0.0
    assert mean_payout(o) > 0  # funded success read from net_payout, not pass_rate


def test_pass_rate_counts_only_passed_codes():
    codes = [int(ExitCode.PASSED), int(ExitCode.PASSED), int(ExitCode.FAIL_TRAILING_DD),
             int(ExitCode.TIMED_OUT)]
    assert pass_rate(codes) == 0.5


# --- objectives (§14.3) ----------------------------------------------------- #


def test_expected_payout_subject_to_profitable_floor():
    # Below the P(profitable) floor -> -inf; above -> the expected net payoff.
    lousy = _outcomes(net_payout=[0.0, 0.0, 0.0, 300.0], eval_fee=100.0)  # P=0.25
    assert expected_payout_st_profitable(lousy, floor=0.5) == float("-inf")
    good = _outcomes(net_payout=[300.0, 300.0, 300.0, 0.0], eval_fee=100.0)  # P=0.75
    assert expected_payout_st_profitable(good, floor=0.5) == np.mean([200, 200, 200, -100])


# --- Results wrapper (§14.5) ------------------------------------------------ #


# --- edge cases (guards) ---------------------------------------------------- #


def test_zero_fee_has_undefined_return_on_fee():
    # No invented denominator: use a dollar objective for a free account.
    o = _outcomes(net_payout=[0.0, 200.0], eval_fee=0.0, activation_fee=0.0)
    assert np.all(np.isnan(return_on_fee(o)))
    assert np.isnan(return_on_fee_per_year(o))


def test_first_day_payout_counts_one_completed_day():
    o = _outcomes(net_payout=[200.0], first_payout_day=[0])
    np.testing.assert_allclose(time_to_first_payout(o), [1 / 5])


def test_paid_attempt_requires_nonnegative_first_payout_index():
    o = _outcomes(net_payout=[200.0], first_payout_day=[-1])
    with pytest.raises(ValueError, match="first_payout_day"):
        time_to_first_payout(o)


def test_mixed_zero_fee_paths_are_not_silently_dropped():
    o = _outcomes(net_payout=[0.0, 200.0], reached_funded=[False, True],
                  eval_fee=0.0, activation_fee=100.0)
    np.testing.assert_allclose(return_on_fee(o), [np.nan, 2.0])
    assert np.isnan(return_on_fee_per_year(o))


def test_tiny_positive_fee_is_not_replaced_by_epsilon():
    o = _outcomes(net_payout=[1.0], eval_fee=1e-12)
    np.testing.assert_allclose(return_on_fee(o), [1e12])


def test_zero_duration_has_undefined_time_normalized_statistics():
    o = _outcomes(net_payout=[0.0, 200.0], total_trading_days=[0, 0])
    assert np.isnan(payout_velocity(o))
    assert np.isnan(return_on_fee_per_year(o))


def test_payout_count_dist_folds_overflow_and_still_sums_to_one():
    # A (should-not-happen) count above max_payouts must not silently drop mass.
    o = _outcomes(net_payout=[10, 10, 10], payouts_taken=[2, 3, 5], max_payouts=3)
    dist = payout_count_dist(o)
    assert abs(dist.sum() - 1.0) < 1e-12  # the 5 is folded into the last bin
    assert dist[3] == 2 / 3  # the count-3 and the folded count-5


def test_empty_batch_does_not_raise_or_warn():
    o = _outcomes(net_payout=[])
    assert np.isnan(prob_profitable(o))
    assert np.isnan(mean_payout(o))
    assert np.isnan(payout_velocity(o))
    d = payout_count_dist(o)
    assert len(d) == o.max_payouts + 1 and np.all(np.isnan(d))


# --- direct coverage of the remaining §14/§13 functions --------------------- #


def test_payoff_quantiles_are_net_of_the_attributable_fee():
    o = _outcomes(net_payout=[0.0, 100.0, 200.0, 300.0, 400.0], eval_fee=100.0)
    q = payoff_quantiles(o, qs=(0.0, 0.5, 1.0))
    # net payoff = net_payout - 100 = [-100, 0, 100, 200, 300]
    np.testing.assert_allclose(q, [-100.0, 100.0, 300.0])


def test_wilson_ci_brackets_the_point_estimate_and_guards_zero_n():
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi
    assert np.isnan(wilson_ci(0, 0)).all()


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    vals = np.arange(100.0)
    lo, hi = bootstrap_ci(vals, seed=0)
    lo2, hi2 = bootstrap_ci(vals, seed=0)
    assert (lo, hi) == (lo2, hi2)  # deterministic under seed
    assert lo < vals.mean() < hi


def test_objective_at_exactly_the_profitable_floor_passes():
    # P(profitable) == floor should satisfy the constraint (>= floor), not fail it.
    o = _outcomes(net_payout=[300.0, 0.0], eval_fee=100.0)  # P = 0.5 exactly
    assert expected_payout_st_profitable(o, floor=0.5) == np.mean([200.0, -100.0])


def test_results_exposes_both_axes_lazily():
    o = _outcomes(net_payout=[0.0, 500.0], payouts_taken=[0, 2],
                  code=[int(ExitCode.FAIL_TRAILING_DD), int(ExitCode.MAXED_OUT)])
    r = Results(o)
    assert r.prob_profitable == 0.5
    assert abs(r.payout_count_dist.sum() - 1.0) < 1e-12
    assert r.mean_payout == 250.0
    assert np.isfinite(r.payout_velocity)
    assert np.isfinite(r.roi_per_year)
    assert r.fingerprint == "abc123"
    # the optimizer entry point evaluates any objective
    assert r.objective(expected_payout_st_profitable, floor=0.0) == np.mean([-100.0, 400.0])
