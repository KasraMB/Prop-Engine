"""Held-out records must retain raw evidence and every seed-specific policy."""
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from propfirm_engine import Account, Phase, MinimumWinningDaysRule, PayoutSchema, preprocess
from propfirm_engine.engine import Engine, RunConfig
from propfirm_engine.optimizer import (
    OptConfig, PolicySpace, evaluate_policy, rolling_walk_forward, walk_forward,
)


def account():
    return Account("synthetic", 1000, (
        Phase("funded", "funded", (MinimumWinningDaysRule(1, 0),),
              PayoutSchema((100,), 1.0, 2)),
    ), eval_fee=10)


def dataset(n=20):
    return preprocess([{"timestamp": datetime(2026, 1, 1, 12) + timedelta(days=i),
                        "return": 1.0 + i / 10} for i in range(n)])


def fixed_fit(account, train, cfg, **kwargs):
    theta = np.full(5, 1 + kwargs["opt_config"].seed / 100)
    return SimpleNamespace(theta=theta, policy=kwargs["space"].to_policy(theta),
                           train_score=0.0, baseline_score=0.0, history=[])


def objective(outcomes):
    return float(np.mean(outcomes.net_payout))


def test_detailed_evaluation_preserves_score_arrays_and_input_policy_snapshot():
    theta = np.ones(5)
    cfg = RunConfig(n_paths=12, L_funded=3, seed=6)
    result = evaluate_policy(Engine(), account(), dataset(), cfg, theta, PolicySpace(),
                             objective, return_details=True)
    assert result.score == objective(result.outcomes)
    assert result.seed == 6 and result.n_paths == 12
    assert result.outcomes.n_attempts == 12
    theta[0] = 9
    assert result.theta[0] == 1
    assert not result.theta.flags.writeable
    assert not result.policy.flags.writeable
    assert not result.outcomes.net_payout.flags.writeable
    assert dict(result.run_parameters)["L_funded"] == 3


def test_walk_forward_retains_both_comparison_sides_and_ladder_separately():
    result = walk_forward(account(), dataset(), dataset(), RunConfig(n_paths=5),
                          opt_config=OptConfig(select_paths=12), objective=objective,
                          fitter=fixed_fit, ladder_datasets={"scenario": dataset(10)})
    assert result.oos_score == result.oos_evaluation.score
    assert result.baseline_oos_score == result.baseline_evaluation.score
    assert result.oos_evaluation.seed == result.baseline_evaluation.seed
    assert result.oos_evaluation.n_paths == result.baseline_evaluation.n_paths == 12
    assert result.oos_ladder["scenario"] == result.ladder_evaluations["scenario"].score
    assert result.ladder_evaluations["scenario"].outcomes.n_attempts == 12


def test_rolling_folds_retain_every_seed_fit_not_only_the_last():
    result = rolling_walk_forward(account(), dataset(20), RunConfig(n_paths=5, L_funded=3),
                                  n_folds=2, warmup_frac=0.4, min_test_days=2,
                                  seeds=(2, 4), opt_config=OptConfig(select_paths=8),
                                  objective=objective, fitter=fixed_fit)
    for fold in result.folds:
        assert len(fold.evaluations) == 2
        np.testing.assert_allclose(fold.evaluations[0].theta, 1.02)
        np.testing.assert_allclose(fold.evaluations[1].theta, 1.04)
        assert fold.oos == [evaluation.oos_score for evaluation in fold.evaluations]
        assert fold.baseline_oos == [evaluation.baseline_oos_score for evaluation in fold.evaluations]
        assert [evaluation.oos_evaluation.seed for evaluation in fold.evaluations] == [993, 995]
        np.testing.assert_array_equal(fold.theta, fold.evaluations[-1].theta)  # legacy field


def test_retaining_details_does_not_resimulate_or_alias_raw_outcomes():
    calls = []

    class CountingEngine(Engine):
        def run(self, *args, **kwargs):
            outcomes = super().run(*args, **kwargs)
            calls.append(outcomes)
            return outcomes

    result = evaluate_policy(CountingEngine(), account(), dataset(), RunConfig(n_paths=4),
                             np.ones(5), PolicySpace(), objective, return_details=True)
    assert len(calls) == 1
    saved = result.outcomes.net_payout.copy()
    calls[0].net_payout[:] = -999
    np.testing.assert_array_equal(result.outcomes.net_payout, saved)


@pytest.mark.parametrize("filename", ["dashboard/montecarlo.py", "docs/py/mc_engine.py"])
def test_dashboard_charts_reuse_the_scored_outcomes(filename):
    # Static consumer guard; not a browser/Pyodide integration claim.
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / filename).read_text(encoding="utf-8")
    assert "o_base = wf.baseline_evaluation.outcomes" in source
    assert "o_opt = wf.oos_evaluation.outcomes" in source
