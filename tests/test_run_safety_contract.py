"""Reject invalid run/quantity/random-path inputs before compiled execution."""
from dataclasses import replace

import numpy as np
import pytest

from propfirm_engine.data import preprocess
from propfirm_engine.engine import Engine, RunConfig
from propfirm_engine.feasibility import FeasibilitySpec
from propfirm_engine.model import Account, Phase
from propfirm_engine.resampling import StationaryDayBootstrap
from propfirm_engine.rules import ProfitTargetRule


def setup():
    cfg = RunConfig(intraday_mode="summary_approximation", n_paths=2, L_eval=2)
    account = Account("fixture", 1000, (Phase("eval", "eval", (ProfitTargetRule(100.0),)),))
    ds = preprocess([{"timestamp": "2026-01-05T12:00", "return": 1.0}])
    engine = Engine()
    return engine, engine.prepare(account, cfg), ds, cfg


@pytest.mark.parametrize("field,value", [
    ("n_paths", 0), ("n_paths", 1.5), ("n_paths", True),
    ("L_eval", 0), ("L_funded", -1), ("batch_size", 0),
    ("seed", -1), ("seed", 0.5), ("seed", True),
    ("size_base", float("nan")), ("size_base", float("inf")), ("size_base", -1.0),
    ("size_base", True), ("trade_cost", -1.0), ("trade_cost", float("nan")),
    ("start_equity", float("inf")), ("start_equity", float("nan")),
])
def test_invalid_run_config_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        RunConfig(**{field: value})


def test_mutated_run_config_rechecked():
    engine, prep, ds, cfg = setup()
    cfg.size_base = float("nan")
    with pytest.raises(ValueError, match="size_base"):
        engine.run_prepared(prep, ds, cfg)


@pytest.mark.parametrize("policy", [[], [[1.0]], [float("nan")], [float("inf")], [-1.0], [True]])
def test_invalid_policy_rejected(policy):
    engine, prep, ds, cfg = setup()
    with pytest.raises(ValueError, match="policy"):
        engine.run_prepared(prep, ds, cfg, policy_params=policy)


@pytest.mark.parametrize("field", ["q_min", "unit_loss", "alpha", "min_buffer"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_feasibility_requires_finite_real_parameters(field, value):
    kwargs = dict(q_min=1.0, unit_loss=1.0)
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        FeasibilitySpec(**kwargs)


def test_minimum_loss_overflow_is_rejected():
    with pytest.raises(ValueError, match="q_min.*unit_loss"):
        FeasibilitySpec(q_min=1e300, unit_loss=1e300)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_stationary_parameter_validation(value):
    with pytest.raises(ValueError, match="mean_block"):
        StationaryDayBootstrap(value)


@pytest.mark.parametrize("paths", [np.array([0, 0]), np.array([[0.0, 0.0], [0.0, 0.0]]),
                                   np.array([[0, 1], [0, 0]]), np.array([[0, -1], [0, 0]])])
def test_bad_custom_paths_rejected_before_kernel(paths):
    class Custom:
        def generate(self, *args):
            return paths

    engine, prep, ds, cfg = setup()
    cfg.resampler = Custom()
    with pytest.raises(ValueError, match="resampler"):
        engine.run_prepared(prep, ds, cfg)
