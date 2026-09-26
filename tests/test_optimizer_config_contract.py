"""Sizing search controls must be effective, finite and executable."""
from types import SimpleNamespace

import numpy as np
import pytest

import propfirm_engine.optimizer as opt
from propfirm_engine.engine import RunConfig


@pytest.mark.parametrize("field", ["lo", "hi"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_policy_bounds(field, value):
    with pytest.raises(ValueError, match=field):
        opt.PolicySpace(**{field: value})


@pytest.mark.parametrize("bounds", [(1, 1), (2, 1)])
def test_policy_bounds_need_a_nonempty_range(bounds):
    with pytest.raises(ValueError, match="hi"):
        opt.PolicySpace(*bounds)


@pytest.mark.parametrize("theta", [[], [[1]], [1] * 6, [float("nan")], [float("inf")], [True], ["1"]])
def test_invalid_policy_vectors_are_not_truncated_or_coerced(theta):
    with pytest.raises(ValueError, match="theta"):
        opt.PolicySpace().to_policy(theta)


def test_short_policy_padding_and_baseline_obey_the_declared_bounds():
    space = opt.PolicySpace(lo=0.1, hi=0.5)
    np.testing.assert_allclose(space.to_policy([0.2]), [0.2, 0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(space.x0(), [0.5] * 5)


@pytest.mark.parametrize("field", ["screen_paths", "select_paths", "finalists", "popsize"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, float("nan")])
def test_invalid_search_counts(field, value):
    with pytest.raises(ValueError, match=field):
        opt.OptConfig(**{field: value})


@pytest.mark.parametrize("field", ["seed", "max_gen"])
@pytest.mark.parametrize("value", [-1, 1.5, True, float("nan")])
def test_invalid_nonnegative_search_counts(field, value):
    with pytest.raises(ValueError, match=field):
        opt.OptConfig(**{field: value})


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_invalid_initial_search_spread(value):
    with pytest.raises(ValueError, match="sigma0"):
        opt.OptConfig(sigma0=value)


@pytest.mark.parametrize("fraction", [0.1, 0.5])
def test_configured_search_spread_is_honored_as_fraction_of_range(monkeypatch, fraction):
    received = []

    class Search:
        def __init__(self, x0, sigma0, **kwargs):
            received.append(sigma0)

        def optimize(self, score):
            return opt.CMAResult(np.ones(5), 1.0)

    monkeypatch.setattr(opt, "CMAES", Search)
    monkeypatch.setattr(opt, "evaluate_policy", lambda *args, **kwargs: 1.0)
    opt.optimize(None, None, RunConfig(), space=opt.PolicySpace(0, 20),
                 opt_config=opt.OptConfig(sigma0=fraction), prepared=object())
    assert received == [20 * fraction]


def test_mutated_search_config_is_checked_before_engine_preparation():
    cfg = opt.OptConfig()
    cfg.max_gen = -1
    engine = SimpleNamespace(prepare=lambda *args: pytest.fail("invalid config reached prepare"))
    with pytest.raises(ValueError, match="max_gen"):
        opt.optimize(None, None, RunConfig(), space=opt.PolicySpace(), opt_config=cfg, engine=engine)


def test_zero_generations_and_default_population_are_valid():
    opt.OptConfig(max_gen=0, popsize=None)
