"""Selection must use the requested objective and final-fidelity candidate scores."""
from types import SimpleNamespace

import numpy as np
import pytest

import propfirm_engine.optimizer as opt
from propfirm_engine.engine import RunConfig


@pytest.mark.parametrize("winner", [1.0, 3.0])
def test_final_fidelity_can_select_baseline_or_screening_runner_up(monkeypatch, winner):
    calls = []

    def score(engine, account, dataset, cfg, theta, space, objective, *args, **kwargs):
        value = float(theta[0])
        calls.append((cfg.n_paths, value))
        if cfg.n_paths == 10:
            return {2.0: 100.0, 3.0: 90.0}[value]
        return 100.0 if value == winner else 0.0

    class Search:
        def __init__(self, *args, **kwargs):
            pass

        def optimize(self, f):
            f(np.full(5, 2.0))
            f(np.full(5, 3.0))
            return opt.CMAResult(np.full(5, 2.0), 100.0)

    monkeypatch.setattr(opt, "evaluate_policy", score)
    monkeypatch.setattr(opt, "CMAES", Search)
    result = opt.optimize(object(), object(), RunConfig(), space=opt.PolicySpace(),
                          prepared=object(), opt_config=opt.OptConfig(screen_paths=10, select_paths=100))
    assert result.theta.tolist() == [winner] * 5
    assert result.train_score == 100.0
    assert (100, 3.0) in calls


def test_callable_objective_receives_actual_outcomes():
    outcome = object()
    engine = SimpleNamespace(run=lambda *args, **kwargs: outcome)
    seen = []

    def objective(value):
        seen.append(value)
        return 123.0

    assert opt.evaluate_policy(engine, None, None, None, [1.0] * 5,
                               opt.PolicySpace(), objective) == 123.0
    assert seen == [outcome]


def test_legacy_factorized_entry_forwards_whole_account_and_objective(monkeypatch):
    captured = []
    account, dataset, objective, prepared = object(), object(), object(), object()
    sentinel = object()

    def joint(a, ds, cfg, **kwargs):
        captured.append((a, ds, kwargs))
        return sentinel

    monkeypatch.setattr(opt, "optimize", joint)
    with pytest.warns(FutureWarning, match="factorized"):
        result = opt.factorized_optimize(account, dataset, RunConfig(),
                                         objective=objective, prepared=prepared)
    assert result is sentinel
    assert captured[0][0] is account and captured[0][1] is dataset
    assert captured[0][2]["objective"] is objective
    assert captured[0][2]["prepared"] is prepared


def test_legacy_funded_only_objective_is_not_silently_substituted():
    with pytest.raises(ValueError, match="whole.*objective"):
        opt.factorized_optimize(None, None, RunConfig(), funded_objective=opt.RenewalObjective())
