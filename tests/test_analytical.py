import math
import numpy as np
import pytest

from propfirm_engine.analytical import (
    barrier_pass_probability, estimate_session_moments, expected_funding_cost,
)
from propfirm_engine import IIDGenerator, preprocess


def model(**kwargs):
    args = dict(target=3000, drawdown=2000, freeze_level=100, mu=0, sigma=850)
    return barrier_pass_probability(**(args | kwargs))


@pytest.mark.parametrize("freeze,expected", [
    (-2000, 0.4), (0, math.exp(-1) * 2 / 3),
    (100, math.exp(-1.05) * 2000 / 2900),
    (1000, math.exp(-1.5)), (math.inf, math.exp(-1.5)),
])
def test_continuous_zero_drift_limits(freeze, expected):
    assert model(freeze_level=freeze).probability == pytest.approx(expected)


@pytest.mark.parametrize("mu", [-100, -1e-10, 1e-10, 100])
def test_fixed_barrier_matches_independent_scale_formula(mu):
    nu = 2 * mu / 850 ** 2
    expected = math.expm1(-nu * 2000) / math.expm1(-nu * 5000)
    assert model(freeze_level=-2000, mu=mu).probability == pytest.approx(expected)


@pytest.mark.parametrize("mu", [-100, 100])
def test_drifted_frozen_formula_direct_arithmetic(mu):
    nu = 2 * mu / 850 ** 2
    expected = math.exp(-nu * 2100 / math.expm1(nu * 2000))
    expected *= math.expm1(nu * 2000) / (math.exp(nu * 2000) - math.exp(-nu * 900))
    assert model(mu=mu).probability == pytest.approx(expected)


def test_drift_monotonicity_extremes_and_small_drift_continuity():
    values = [model(mu=mu).probability for mu in [-1e6, -100, 0, 100, 1e6]]
    assert values == sorted(values) and values[0] == 0 and values[-1] == 1
    assert model(mu=1e-12).probability == pytest.approx(model().probability, abs=1e-12)
    assert model(mu=-1e-12).probability == pytest.approx(model().probability, abs=1e-12)


@pytest.mark.parametrize("fm,fb,expected", [
    (1, (850 / 96) ** 2, .3018),
    ((850 / 96) ** 2, (850 / 96) ** 2, .2543),
])
def test_paper_monitoring_examples(fm, fb, expected):
    result = model(freeze_level=0, floor_updates=fm, breach_checks=fb)
    assert result.probability == pytest.approx(expected, abs=5e-5)
    assert result.method == "paper_eq3_eq5" and result.warnings


def test_static_case_has_no_floor_clock_effect():
    coarse, continuous = model(freeze_level=-2000, floor_updates=1), model(freeze_level=-2000)
    assert coarse.probability == continuous.probability
    assert coarse.method == continuous.method and coarse.warnings == continuous.warnings


def test_clock_refinement_converges_and_reports_large_step():
    base = model().probability
    assert model(floor_updates=1e18, breach_checks=1e18).probability == pytest.approx(base, abs=1e-9)
    assert len(model(sigma=1500, floor_updates=1).warnings) == 2


@pytest.mark.parametrize("changes", [
    {"sigma": 0}, {"sigma": -1}, {"sigma": True}, {"mu": float("nan")},
    {"mu": float("inf")}, {"target": 0}, {"drawdown": -1}, {"freeze_level": -2001},
    {"floor_updates": 0}, {"breach_checks": -1}, {"target": "3000"},
    {"sigma": 10**500}, {"freeze_level": float("-inf")},
])
def test_invalid_parameters(changes):
    with pytest.raises(ValueError):
        model(**changes)


def test_session_moments_match_net_daily_sums_not_sqrt_trade_count_shortcut():
    rows = dict(timestamp=["2024-01-01 10:00", "2024-01-01 11:00",
                           "2024-01-02 10:00", "2024-01-02 11:00"],
                **{"return": [1, 1, -1, -1]})
    ds = preprocess(rows)
    fit = estimate_session_moments(ds, size_base=100, trade_cost=2)
    assert fit.mu == -4 and fit.sigma == 200 and fit.trade_sigma == 100
    assert fit.n_days == 2 and fit.n_trades == 4
    assert fit.mean_trades_per_day == 2


def test_moments_use_generated_sample_not_generator_expectation():
    stream = IIDGenerator(.52, 1, trades_per_day=20).generate(100, seed=123)
    ds = preprocess(stream.rows)
    fit = estimate_session_moments(ds, size_base=100, trade_cost=1)
    totals = (np.asarray(stream.rows["return"]).reshape(100, 20) * 100 - 1).sum(axis=1)
    assert fit.mu == pytest.approx(totals.mean())
    assert fit.sigma == pytest.approx(totals.std(ddof=0))


def test_single_session_rejected():
    ds = preprocess(IIDGenerator(.5, 1).generate(1, 1).rows)
    with pytest.raises(ValueError, match="two sessions"):
        estimate_session_moments(ds, size_base=100)


def test_retry_cost_is_not_attempt_fee_or_net_value():
    assert expected_funding_cost(.25, entry_fee=105.2, reset_fee=105) == pytest.approx(420.2)
    assert expected_funding_cost(1, entry_fee=100, reset_fee=99, activation_fee=10) == 110
    assert math.isinf(expected_funding_cost(0, entry_fee=100, reset_fee=99))
    with pytest.raises(ValueError):
        expected_funding_cost(1.1, entry_fee=100, reset_fee=99)
