"""Step 11 — renewal economics (BUILD_SPEC Step 11, ARCHITECTURE §15)."""

from __future__ import annotations

import numpy as np
from datetime import timedelta

from propfirm_engine.cashflows import AttemptCashflows, Cashflow

from propfirm_engine.engine import Outcomes
from propfirm_engine.enums import ExitCode
from propfirm_engine.renewal import (
    fee_bankroll_efficiency,
    finite_horizon_cashflow,
    per_attempt_reward,
    per_attempt_time,
    prob_profitable_sequence,
    r_path,
    r_renewal,
    renewal_report,
)
from propfirm_engine.statistics import prob_profitable


def _explicit_fixture_schedules(o):
    # Synthetic test contract only: fee at start, receipt at end, durations
    # explicitly defined as these elapsed weeks. Production Outcomes cannot
    # infer these dates; do not use this helper as a real-data adapter.
    return tuple(AttemptCashflows(timedelta(weeks=float(weeks)), (
        Cashflow(timedelta(0), -float(fee), "fee"),
        Cashflow(timedelta(weeks=float(weeks)), float(payout), "receipt")))
        for weeks, fee, payout in zip(per_attempt_time(o),
                                     o.eval_fee + o.activation_fee * o.reached_funded,
                                     o.net_payout))


def _outcomes(*, net_payout, total_trading_days, trading_days_per_week=1.0,
              eval_fee=0.0, activation_fee=0.0, reached_funded=None):
    net_payout = np.asarray(net_payout, dtype=np.float64)
    b = len(net_payout)
    if reached_funded is None:
        reached_funded = np.ones(b, dtype=bool)
    return Outcomes(
        code=np.full(b, int(ExitCode.TIMED_OUT), np.int32),
        reached_funded=np.asarray(reached_funded, bool),
        net_payout=net_payout,
        payouts_taken=(net_payout > 0).astype(np.int32),
        first_payout_day=np.full(b, -1, np.int32),
        total_trading_days=np.asarray(total_trading_days, np.int32),
        size=50_000, size_base=100.0, max_payouts=3,
        eval_fee=eval_fee, activation_fee=activation_fee,
        trading_days_per_week=trading_days_per_week, fingerprint="x",
    )


# --- per-attempt reward / time definitions (§15) ---------------------------- #


def test_per_attempt_reward_is_net_payout_minus_attributable_fee():
    o = _outcomes(net_payout=[500.0, 0.0], total_trading_days=[10, 10],
                  eval_fee=150.0, activation_fee=50.0)
    np.testing.assert_allclose(per_attempt_reward(o), [300.0, -200.0])


def test_per_attempt_time_is_the_whole_attempt_calendar_duration():
    # T_i uses total_trading_days (eval + funded, §H4), converted by cadence.
    o = _outcomes(net_payout=[0.0], total_trading_days=[52], trading_days_per_week=5.0)
    np.testing.assert_allclose(per_attempt_time(o), [52 / 5.0])  # 10.4 weeks


# --- r_renewal (§15.2) ------------------------------------------------------ #


def test_r_renewal_is_ratio_of_mean_reward_to_mean_time():
    o = _outcomes(net_payout=[120.0, 80.0], total_trading_days=[5, 3],
                  trading_days_per_week=1.0)
    R = per_attempt_reward(o)
    T = per_attempt_time(o)
    assert r_renewal(o) == np.mean(R) / np.mean(T)


# --- r_path convergence vs Jensen bias (§15.2, §H5) ------------------------- #


def test_r_path_converges_to_r_renewal_for_iid_light_tailed_cycles():
    rng = np.random.default_rng(0)
    net = rng.choice([80.0, 120.0], 4000)
    days = rng.choice([3, 5], 4000)
    o = _outcomes(net_payout=net, total_trading_days=days, trading_days_per_week=1.0)
    rp = r_path(o, horizon_weeks=30, n_sequences=3000, seed=1)
    # light-tailed, long-enough horizon -> the two definitions agree (the clean
    # ergodic case): mean(r_path) ~ r_renewal.
    assert abs(rp.mean() - r_renewal(o)) < 0.5


def test_r_path_departs_from_r_renewal_for_heavy_tailed_cycle_times():
    # A skewed/heavy-tailed cycle-time set makes the finite-horizon ratio estimator
    # (r_path) depart from the ratio-of-means (r_renewal) — ratio-estimator /
    # finite-horizon (Jensen) bias. This is NOT a correlation effect: r_path draws
    # attempts i.i.d., so it *cannot* show cross-cycle correlation (§H5) — the gap
    # here is Jensen bias alone.
    rng = np.random.default_rng(0)
    big = rng.random(4000) < 0.05
    net = np.where(big, 50.0, 10.0)
    days = np.where(big, 80, 1)  # heavy tail in cycle time
    o = _outcomes(net_payout=net, total_trading_days=days, trading_days_per_week=1.0)
    rp = r_path(o, horizon_weeks=5, n_sequences=3000, seed=1)
    assert abs(rp.mean() - r_renewal(o)) > 2.0  # a large, unambiguous departure


def test_both_rates_are_available_separately():
    o = _outcomes(net_payout=[100.0] * 50, total_trading_days=[4] * 50,
                  trading_days_per_week=1.0)
    closed = r_renewal(o)
    empirical = r_path(o, horizon_weeks=20, n_sequences=100, seed=0)
    assert np.isscalar(closed) or isinstance(closed, float)
    assert empirical.shape == (100,)  # a distribution, reported separately


# --- fee-bankroll efficiency (§15.2) ---------------------------------------- #


def test_fee_bankroll_efficiency_halves_when_the_fee_doubles():
    o = _outcomes(net_payout=[300.0] * 100, total_trading_days=[10] * 100,
                  trading_days_per_week=5.0, eval_fee=150.0)
    e1 = fee_bankroll_efficiency(o, fee=150.0)
    e2 = fee_bankroll_efficiency(o, fee=300.0)  # double the fee, all else equal
    assert e2 == e1 / 2.0  # income-per-bankroll halves (fewer accounts per bankroll)


def test_fee_bankroll_efficiency_defaults_to_the_eval_fee():
    o = _outcomes(net_payout=[300.0] * 20, total_trading_days=[10] * 20,
                  trading_days_per_week=5.0, eval_fee=200.0)
    assert fee_bankroll_efficiency(o) == fee_bankroll_efficiency(o, fee=200.0)


# --- finite-horizon cumulative cashflow distribution (§15.3) ---------------- #


def test_finite_horizon_cashflow_returns_a_distribution_not_a_mean():
    rng = np.random.default_rng(0)
    net = rng.choice([0.0, 600.0], 2000)  # convex: mostly lose the fee, sometimes win
    o = _outcomes(net_payout=net, total_trading_days=np.full(2000, 4),
                  trading_days_per_week=1.0, eval_fee=100.0)
    cash = finite_horizon_cashflow(_explicit_fixture_schedules(o), horizon_weeks=40, n_sequences=500, seed=2)
    assert cash.shape == (500,)
    assert cash.std() > 0  # a real distribution, convexity visible at the renewal level


def test_prob_profitable_sequence_is_the_sequence_level_definition():
    # A profitable sequence has cumulative reward > 0. With a positive-edge stream
    # most long sequences profit; with a negative one, few do.
    good = _outcomes(net_payout=[300.0] * 200, total_trading_days=[4] * 200,
                     trading_days_per_week=1.0, eval_fee=100.0)
    bad = _outcomes(net_payout=[0.0] * 200, total_trading_days=[4] * 200,
                    trading_days_per_week=1.0, eval_fee=100.0)
    assert prob_profitable_sequence(_explicit_fixture_schedules(good), horizon_weeks=40, n_sequences=200, seed=0) == 1.0
    assert prob_profitable_sequence(_explicit_fixture_schedules(bad), horizon_weeks=40, n_sequences=200, seed=0) == 0.0


def test_sequence_can_be_profitable_while_most_attempts_are_not():
    # The case that motivates the sequence-level definition (§15.1): a stream where
    # MOST attempts lose the fee but rare big funded payouts more than cover them.
    # Over a long horizon the sequence almost always profits, even though the
    # single-attempt P(profitable) is low — so the sequence number exceeds it.
    rng = np.random.default_rng(0)
    big = rng.random(4000) < 0.1  # 10% of attempts hit a large payout
    net = np.where(big, 2000.0, 0.0)
    o = _outcomes(net_payout=net, total_trading_days=np.full(4000, 4),
                  trading_days_per_week=1.0, eval_fee=100.0)
    attempt_level = prob_profitable(o)  # ~0.1 (only the winners beat the fee)
    seq_level = prob_profitable_sequence(_explicit_fixture_schedules(o), horizon_weeks=100, n_sequences=500, seed=1)
    assert attempt_level < 0.2
    assert seq_level > attempt_level  # sequence pools winners over the fees


# --- degenerate-input guards (no hang, no crash) ---------------------------- #


def test_zero_cycle_time_returns_nan_not_an_infinite_loop():
    # If every attempt has zero calendar time the accumulated time can never reach
    # the horizon; r_path must return nan (like r_renewal), never hang.
    o = _outcomes(net_payout=[100.0, 50.0], total_trading_days=[0, 0],
                  trading_days_per_week=1.0)
    assert np.all(np.isnan(r_path(o, horizon_weeks=5, n_sequences=3, seed=0)))
    assert np.isnan(r_renewal(o))


def test_horizon_must_be_positive():
    import pytest
    o = _outcomes(net_payout=[100.0], total_trading_days=[4], trading_days_per_week=1.0)
    with pytest.raises(ValueError):
        r_path(o, horizon_weeks=0, n_sequences=1, seed=0)


# --- the divergence diagnostic (§15.2) -------------------------------------- #


def test_renewal_report_surfaces_both_rates_and_the_correlation_gap():
    o = _outcomes(net_payout=[100.0] * 100, total_trading_days=[4] * 100,
                  trading_days_per_week=1.0)
    rep = renewal_report(o, horizon_weeks=20, n_sequences=200, seed=0)
    assert "r_renewal" in rep and "r_path_mean" in rep
    assert rep["jensen_divergence"] == abs(rep["r_renewal"] - rep["r_path_mean"])
    # the honest position: the cross-cycle correlation gap is named as uncaptured
    assert "uncaptured" in rep["correlation_gap"]


# --- determinism ------------------------------------------------------------ #


def test_renewal_simulation_is_deterministic_under_seed():
    o = _outcomes(net_payout=[100.0, 200.0, 0.0], total_trading_days=[4, 6, 2],
                  trading_days_per_week=1.0)
    a = r_path(o, horizon_weeks=20, n_sequences=200, seed=7)
    b = r_path(o, horizon_weeks=20, n_sequences=200, seed=7)
    np.testing.assert_array_equal(a, b)
