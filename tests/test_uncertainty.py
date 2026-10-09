"""Independent mean/paired/block calculations and deliberately withheld inference."""
from dataclasses import replace
from math import sqrt

import numpy as np
import pytest

from propfirm_engine import (
    Engine, UncertaintyConfig, paired_uncertainty, scenario_sensitivity, uncertainty_report,
)
from propfirm_engine.uncertainty import _bootstrap_means


def report(values, **kwargs):
    return uncertainty_report({"cash": values}, sample_kind=kwargs.pop("sample_kind", "independent_model"), **kwargs)


def test_mean_error_is_not_outcome_dispersion():
    metric = report([-100, 0, 100, 400])["metrics"]["cash"]
    assert metric["mean"] == 100
    assert metric["sample_standard_deviation"] == pytest.approx(sqrt(140000/3))
    assert metric["mean_standard_error"] == pytest.approx(sqrt(140000/12))
    assert metric["mean_interval"] is None
    assert metric["status"] == "bootstrap_not_requested"
    duplicated = report([-100, 0, 100, 400]*4)["metrics"]["cash"]
    assert duplicated["mean_standard_error"] == pytest.approx(sqrt(560000/15)/4)


@pytest.mark.parametrize("block", [1, 2, 3, 4])
def test_bootstrap_matches_explicit_circular_index_oracle(block):
    values = np.array([-100., 60., 300., 2., -8., 80., 20., 90., -14.])
    n, repeats, seed = len(values), 400, 91
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(repeats, (n+block-1)//block))
    expected = []
    for row in starts:
        indices = [(start+i) % n for start in row for i in range(block)][:n]
        expected.append(sum(values[i] for i in indices)/n)
    actual = _bootstrap_means(values, block, repeats, np.random.default_rng(seed))
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    kind = "independent_model" if block == 1 else "historical_series"
    options = UncertaintyConfig(resamples=repeats, seed=seed, block_length=None if block == 1 else block)
    metric = report(values, sample_kind=kind, options=options)["metrics"]["cash"]
    assert metric["mean_interval"] == pytest.approx(np.quantile(expected, [.025, .975]))
    if block > 1:
        assert metric["mean_standard_error"] == pytest.approx(np.std(expected, ddof=1))
    assert metric["method"] == ("iid_percentile" if block == 1 else "circular_block_percentile")


def test_bootstrap_batches_preserve_seeded_draws_and_translation():
    a = np.arange(501, dtype=float)
    repeats, seed = 2200, 43
    expected = a[np.random.default_rng(seed).integers(0, len(a), (repeats, len(a)))].mean(axis=1)
    actual = _bootstrap_means(a, 1, repeats, np.random.default_rng(seed))
    np.testing.assert_allclose(actual, expected)
    shifted = _bootstrap_means(a+1e9, 1, repeats, np.random.default_rng(seed))
    np.testing.assert_allclose(shifted-1e9, actual, rtol=0, atol=1e-7)


def test_block_bootstrap_captures_persistence_instead_of_iid_resampling():
    values = [-10.]*100+[10.]*100
    options = UncertaintyConfig(resamples=3000, seed=71)
    iid = report(values, options=options)["metrics"]["cash"]
    blocks = report(values, sample_kind="historical_series",
                    options=replace(options, block_length=20))["metrics"]["cash"]
    assert blocks["mean_standard_error"] > 3*iid["mean_standard_error"]
    assert blocks["mean_interval"][1] > 3*iid["mean_interval"][1]


@pytest.mark.parametrize("kind", ["historical_windows", "single_history", "training_model"])
def test_no_intervals_for_unjustified_sampling_design(kind):
    metric = report([1, 2, 7], sample_kind=kind, options=UncertaintyConfig(resamples=200))["metrics"]["cash"]
    assert metric["mean_standard_error"] is None
    assert metric["mean_interval"] is None
    assert metric["method"] is None


def test_selection_and_missing_observations_suppress_inference():
    for values, kwargs, reason in [([1, 2, 7], {"selected_on_sample": True}, "selected_on_sample"),
                                 ([1, None, 7], {}, "missing_observations"),
                                 ([1, float("nan"), 7], {}, "missing_observations"),
                                 ([2, 2, 2], {}, "no_observed_variation"),
                                 ([2], {}, "insufficient_observations"),
                                 ([], {}, "insufficient_observations")]:
        metric = report(values, options=UncertaintyConfig(resamples=200), **kwargs)["metrics"]["cash"]
        assert metric["status"] == reason
        assert metric["mean_standard_error"] is None
        assert metric["mean_interval"] is None
    metric = report([1, None, 7], sample_kind="historical_series",
                    options=UncertaintyConfig(resamples=200, block_length=1))["metrics"]["cash"]
    assert metric["count"] == 2 and metric["missing"] == 1
    assert metric["mean_interval"] is None


def test_block_length_must_be_explicit_and_leave_multiple_blocks():
    metric = report([1, 2, 7, 8], sample_kind="historical_series",
                    options=UncertaintyConfig(resamples=200))["metrics"]["cash"]
    assert metric["status"] == "block_bootstrap_required"
    metric = report([1, 2, 7, 8], sample_kind="historical_series",
                    options=UncertaintyConfig(resamples=200, block_length=3))["metrics"]["cash"]
    assert metric["status"] == "insufficient_blocks"
    assert metric["mean_interval"] is None


def test_paired_gain_uses_covariance_not_independent_group_errors():
    base = {"cash": [-1000, 0, 1000, 2000]}
    selected = {"cash": [-990, 12, 1014, 2016]}
    metric = paired_uncertainty(selected, base, sample_kind="independent_model")["metrics"]["cash"]
    assert metric["mean"] == 13
    assert metric["mean_standard_error"] == pytest.approx(sqrt(20/12))
    absent = paired_uncertainty({"cash": [1, None, 3]}, {"cash": [None, 2, 1]},
                               sample_kind="independent_model")["metrics"]["cash"]
    assert absent["count"] == 1 and absent["missing"] == 2
    assert absent["mean_interval"] is None
    with pytest.raises(ValueError, match="matching"):
        paired_uncertainty({"cash": [1, 2]}, {"cash": [1]}, sample_kind="independent_model")
    with pytest.raises(ValueError, match="matching"):
        paired_uncertainty({"x": [1]}, {"y": [1]}, sample_kind="independent_model")


def test_metric_order_and_additions_do_not_change_bootstrap():
    options = UncertaintyConfig(resamples=1000, seed=67)
    a = report([0, 10, 100, 20], options=options)
    b = uncertainty_report({"extra": [1, 2, 4, 8], "cash": [0, 10, 100, 20]},
                           sample_kind="independent_model", options=options)
    assert a["metrics"]["cash"] == b["metrics"]["cash"]
    assert Engine().uncertainty({"cash": [0, 10, 100, 20]}, sample_kind="independent_model", options=options) == a


def test_sensitivity_is_an_unweighted_envelope_not_a_confidence_interval():
    result = scenario_sensitivity({"base": {"cash": 20, "rate": 2},
        "costs": {"cash": -40, "rate": -4}, "drift": {"cash": 50, "rate": 5}}, baseline="base")
    assert result["envelope"] == {"cash": [-40, 50], "rate": [-4, 5]}
    assert result["deltas"]["costs"] == {"cash": -60, "rate": -6}
    assert "not a confidence interval" in result["scope"]
    with pytest.raises(ValueError):
        scenario_sensitivity({"base": {"cash": 2}, "other": {"rate": 2}}, baseline="base")
    with pytest.raises(ValueError):
        scenario_sensitivity({"base": {"cash": float("nan")}}, baseline="base")


@pytest.mark.parametrize("kwargs", [{"confidence": 0}, {"confidence": 1}, {"confidence": float("nan")},
    {"confidence": True}, {"resamples": -1}, {"resamples": 1}, {"resamples": True},
    {"seed": -1}, {"seed": True}, {"block_length": 0}, {"block_length": 1.5}])
def test_invalid_options_rejected(kwargs):
    with pytest.raises(ValueError):
        UncertaintyConfig(**kwargs)


@pytest.mark.parametrize("values", [[float("inf")], [[1], [2]], [-1e308, 1e308]])
def test_bad_numeric_samples_rejected(values):
    with pytest.raises(ValueError):
        report(values)


def test_invalid_sample_contracts_rejected():
    with pytest.raises(ValueError):
        report([1, 2], sample_kind="guess")
    with pytest.raises(TypeError):
        report([1, 2], options={})
    with pytest.raises(TypeError):
        report([1, 2], selected_on_sample=1)
    with pytest.raises(ValueError):
        uncertainty_report({}, sample_kind="independent_model")
    with pytest.raises(ValueError):
        uncertainty_report({"x": [1], "y": [2, 3]}, sample_kind="independent_model")
    with pytest.raises(ValueError):
        report([1, 2], options=UncertaintyConfig(block_length=1))
    with pytest.raises(ValueError):
        report([1, 2], sample_kind="historical_series", options=UncertaintyConfig(block_length=3))


def test_risk_report_reuses_moments_and_does_not_claim_history_is_iid():
    from test_risk import records
    from propfirm_engine import risk_report
    paths = records([-100, 0, 100, 400], [100]*4)
    model = risk_report(paths, sample_kind="independent_model")["uncertainty"]
    assert model["metrics"]["net_cash"]["mean_standard_error"] == pytest.approx(sqrt(140000/12))
    assert model["metrics"]["net_cash_per_day"]["mean_standard_error"] == pytest.approx(sqrt(140000/12)/10)
    assert model["metrics"]["days_to_first_receipt"]["missing"] == 2
    for kind in ("historical_windows", "single_history", "training_model"):
        historical = risk_report(paths, sample_kind=kind)["uncertainty"]
        assert historical["metrics"]["net_cash"]["mean_standard_error"] is None


def test_target_training_is_descriptive_and_holdout_se_matches_raw_values():
    from test_target_research import SPEC, CONFIG
    from propfirm_engine.target_research import BracketModel, fit_targets, lucidflex_example
    policy = lucidflex_example()
    names = [r.name for r in policy.sizing.regimes]
    result = fit_targets(SPEC, BracketModel(sessions=10), CONFIG, policy=policy,
        risk_bounds={n: (100, 2000) for n in names}, target_bounds={n: (100, 3000) for n in names},
        paths=30, generations=0)
    assert result.training.uncertainty["metrics"]["objective"]["status"] == "selected_on_sample"
    metric = result.holdout.uncertainty["metrics"]["objective"]
    assert metric["mean_standard_error"] == pytest.approx(np.std(result.holdout.objective_values, ddof=1)/3)
    assert result.uncertainty["holdout"] == result.holdout.uncertainty
    assert result.uncertainty["paired_gain"]["metrics"]["objective_gain"]["mean"] == result.paired_holdout_gain
    changed = replace(result, direction="minimize",
        holdout=replace(result.holdout, objective_values=(1., 4.)),
        baseline_holdout=replace(result.baseline_holdout, objective_values=(3., 8.)))
    gain = changed.uncertainty["paired_gain"]["metrics"]["objective_gain"]
    assert gain["mean"] == 3 and gain["mean_standard_error"] == 1


def test_price_fit_labels_training_and_conditions_on_execution():
    from test_price_fitting import fit, MODEL
    result = fit(generations=0, paths=3, slippage=MODEL)
    assert result.in_sample.uncertainty["selected_on_sample"]
    assert result.out_of_sample.uncertainty["sample_kind"] == "execution_model"
    assert result.uncertainty["holdout"] == result.out_of_sample.uncertainty
    assert "same historical market tape" in result.uncertainty["holdout"]["scope"]
    assert result.out_of_sample.paths[0].replay.uncertainty["sample_kind"] == "single_history"
    assert result.out_of_sample.paths[0].uncertainty == result.out_of_sample.paths[0].replay.uncertainty
    changed = replace(result, direction="minimize",
        out_of_sample=replace(result.out_of_sample, objective_values=(1., 4.)),
        baseline_out_of_sample=replace(result.baseline_out_of_sample, objective_values=(3., 8.)))
    gain = changed.uncertainty["paired_gain"]["metrics"]["objective_gain"]
    assert gain["mean"] == 3 and gain["mean_standard_error"] == 1


def test_strategy_reports_retain_cash_and_portfolio_metrics():
    from test_strategy_fitting import fit
    result = fit()
    assert result.training.uncertainty["selected_on_sample"]
    assert result.uncertainty == result.selected.uncertainty
    assert {"balance", "net_cash", "net_cash_per_day"} <= result.uncertainty["metrics"].keys()
    assert all(m["mean_standard_error"] is None for m in result.uncertainty["metrics"].values())


def test_rolling_and_holdout_do_not_invent_independent_observations():
    from test_rolling_backtest import inputs
    from propfirm_engine import RollingConfig
    spec, history, policy, config = inputs()
    rolling = Engine().rolling_backtest(spec, history, policy, config, rolling=RollingConfig(4, 1))
    assert rolling.uncertainty["metrics"]["net_cash"]["count"] == 9
    assert rolling.uncertainty["metrics"]["net_cash"]["mean_standard_error"] is None
    fit = Engine().fit(spec, history, config, policy=policy,
        risk_bounds={r.name: (50, 1000) for r in policy.regimes}, generations=0)
    assert fit.uncertainty == fit.out_of_sample.uncertainty
    assert fit.uncertainty["sample_kind"] == "single_history"


def test_legacy_results_require_an_explicit_sampling_design():
    from test_statistics import _outcomes
    from propfirm_engine import Results
    results = Results(_outcomes(net_payout=[0, 0, 400], reached_funded=[False, True, True],
                               eval_fee=100, activation_fee=50))
    with pytest.raises(TypeError):
        results.uncertainty()
    metric = results.uncertainty(sample_kind="independent_model")["metrics"]["net_payoff"]
    expected = np.array([-100., -150., 250.])
    assert metric["mean"] == pytest.approx(0)
    assert metric["mean_standard_error"] == pytest.approx(expected.std(ddof=1)/sqrt(3))


def test_replay_wrappers_forward_single_history_report():
    from test_rolling_backtest import inputs
    from propfirm_engine import EventReplay, StrategyReplay
    replay = Engine().backtest(*inputs())
    events = EventReplay(replay, None, (), 0, 0, 0)
    strategy = StrategyReplay(events, ())
    assert events.uncertainty is replay.uncertainty
    assert strategy.uncertainty is replay.uncertainty


def test_reports_are_json_serializable_with_no_nonfinite_numbers():
    import json
    from dashboard.replay import _summary, _rolling_summary
    from test_rolling_backtest import inputs
    from propfirm_engine import RollingConfig
    replay = Engine().backtest(*inputs())
    assert _summary(replay)["uncertainty"]["sample_kind"] == "single_history"
    rolling = Engine().rolling_backtest(*inputs(), rolling=RollingConfig(4, 1))
    assert _rolling_summary(rolling)["uncertainty"]["sample_kind"] == "historical_windows"
    for output in (_summary(replay), _rolling_summary(rolling), report([None, None]),
                   report([1, 2, 3], options=UncertaintyConfig(resamples=100))):
        json.dumps(output, allow_nan=False)


def test_default_reports_do_not_run_bootstrap(monkeypatch):
    from propfirm_engine import uncertainty
    def unexpected(*args, **kwargs):
        raise AssertionError("resampling must be explicitly requested")
    monkeypatch.setattr(uncertainty, "_bootstrap_means", unexpected)
    metric = report([1, 4, 9])["metrics"]["cash"]
    assert metric["mean_standard_error"] > 0
    assert metric["mean_interval"] is None


def test_bootstrap_temporary_storage_is_batched():
    import tracemalloc
    values = np.arange(20_000, dtype=float)
    tracemalloc.start()
    try:
        means = _bootstrap_means(values, 1, 2000, np.random.default_rng(91))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert len(means) == 2000 and np.isfinite(means).all()
    assert peak < 64_000_000
