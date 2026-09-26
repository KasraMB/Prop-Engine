"""The experiment must keep scope, net-dollar inputs and censoring explicit."""
import numpy as np

from benchmarks.analytical_comparison import evaluate_case, outcome_summary
from propfirm_engine import IIDGenerator, preprocess, Engine, RunConfig, UnsupportedInputCapabilityError
from propfirm_engine.firms.lucidflex import build_account
from dataclasses import replace
import pytest


def test_short_horizon_is_censoring_not_failure_and_has_no_retry_estimate():
    ds = preprocess(IIDGenerator(.5, 1, trades_per_day=20, intraday_excursion=0).generate(20, 9).rows)
    result = evaluate_case(ds, risk=1, cost=0, n_paths=20, horizon=1, seed=19)
    for name in ("engine_barrier_only", "engine_with_consistency"):
        out = result[name]
        assert out["censored"] == 20 and out["breached"] == out["passed"] == 0
        assert out["eventual_probability_identification_bounds"] == [0, 1]
        assert out["iid_retry_cost_scenario"] is None
        assert out["intraday_mode"] == "summary_approximation"
    assert result["paired_consistency_effect_percentage_points"] == 0


def test_comparison_is_deterministic_and_does_not_mutate_dataset():
    ds = preprocess(IIDGenerator(.52, 1, trades_per_day=10).generate(50, 8).rows)
    saved = ds.ret.copy(), ds.trade_low.copy()
    args = dict(risk=200, cost=1, n_paths=60, horizon=30, seed=9)
    one, two = evaluate_case(ds, **args), evaluate_case(ds, **args)
    assert one == two
    assert np.array_equal(ds.ret, saved[0]) and np.array_equal(ds.trade_low, saved[1])
    for name in ("engine_barrier_only", "engine_with_consistency"):
        out = one[name]
        assert out["passed"] + out["breached"] + out["censored"] == 60
    assert one["engine_with_consistency"]["passed"] <= one["engine_barrier_only"]["passed"]


def test_same_returns_different_excursions_have_identical_analytical_inputs():
    base = IIDGenerator(.5, 1, trades_per_day=10, intraday_excursion=0)
    args = dict(risk=100, cost=1, n_paths=20, horizon=10, seed=9)
    results = [evaluate_case(preprocess(gen.generate(50, 7).rows), **args)
               for gen in (base, replace(base, intraday_excursion=.5))]
    assert results[0]["moments"] == results[1]["moments"]
    assert results[0]["paper_eod_continuous_breach"] == results[1]["paper_eod_continuous_breach"]


def test_analytical_addition_does_not_bypass_strict_engine_guard():
    ds = preprocess(IIDGenerator(.52, 1).generate(10, 5).rows)
    account = build_account(50_000)
    with pytest.raises(UnsupportedInputCapabilityError):
        Engine().run(account, ds, RunConfig(n_paths=10))
