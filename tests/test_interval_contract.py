"""Intervals must not turn missing data or malformed parameters into certainty."""
import numpy as np
import pytest

from propfirm_engine.statistics import bootstrap_ci, wilson_ci


def test_no_trials_has_undefined_wilson_interval_not_certain_zero():
    assert np.isnan(wilson_ci(0, 0)).all()


def test_wilson_boundary_matches_hand_calculated_case():
    np.testing.assert_allclose(wilson_ci(0, 10, z=2), [0, 2 / 7], atol=1e-15)
    np.testing.assert_allclose(wilson_ci(10, 10, z=2), [5 / 7, 1], atol=1e-15)


@pytest.mark.parametrize("k,n", [(-1, 10), (11, 10), (0, -1), (True, 10), (1.5, 10), (1, 10.5)])
def test_invalid_trial_counts(k, n):
    with pytest.raises(ValueError, match="counts"):
        wilson_ci(k, n)


@pytest.mark.parametrize("z", [0, -1, True, float("nan"), float("inf")])
def test_invalid_wilson_multiplier(z):
    with pytest.raises(ValueError, match="z"):
        wilson_ci(1, 10, z=z)


def test_extreme_finite_wilson_multiplier_does_not_overflow():
    assert wilson_ci(1, 10, z=1e308) == (0.0, 1.0)


def test_empty_bootstrap_is_undefined():
    assert np.isnan(bootstrap_ci([])).all()


@pytest.mark.parametrize("field,value", [
    ("n_boot", 0), ("n_boot", 1.5), ("n_boot", True),
    ("seed", -1), ("seed", 1.5), ("seed", True),
    ("alpha", 0), ("alpha", 1), ("alpha", True),
    ("alpha", float("nan")), ("alpha", float("inf")), ("stat", None),
])
def test_invalid_bootstrap_parameters(field, value):
    with pytest.raises(ValueError, match=field):
        bootstrap_ci([1, 2], **{field: value})


@pytest.mark.parametrize("values", [[[1, 2]], [True, False], ["1", "2"],
                                   [float("nan")], [float("inf")]])
def test_invalid_bootstrap_observations(values):
    with pytest.raises(ValueError, match="values"):
        bootstrap_ci(values)


@pytest.mark.parametrize("stat", [lambda x: np.nan, lambda x: [1, 2]])
def test_statistic_must_return_one_finite_value(stat):
    with pytest.raises(ValueError, match="stat"):
        bootstrap_ci([1, 2], stat=stat, n_boot=2)


def test_constant_observations_have_constant_interval():
    assert bootstrap_ci([3.0, 3.0, 3.0], n_boot=10) == (3.0, 3.0)
