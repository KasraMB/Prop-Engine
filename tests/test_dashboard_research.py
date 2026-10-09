"""Dashboard target discovery, held-out independence and no seeded answer."""
from copy import deepcopy
from datetime import timedelta
import json

import numpy as np
import pytest

from dashboard import replay, research
from propfirm_engine import BacktestConfig, DollarPolicy, RiskRegime
from propfirm_engine.firms.lucidflex import replay_50k
from propfirm_engine.target_research import BracketModel, TargetPolicy, evaluate_targets


def request_fixture():
    return {"mode":"model_search", "profile":"lucidflex_50k_dll_off", "objective":"net_cash_per_day",
        "model":{"mu":0,"sigma":1000,"sessions":7,"start_date":"2026-01-05"},
        "account":{"eval_fee":105.2,"reset_fee":105,"contract_type":"micro"},
        "config":{"initial_wallet":2000,"cost_per_contract":0,"cost_per_trade":0,"payment_fee":0,
                  "approval_delay_hours":0,"receipt_delay_hours":0,"activation_delay_hours":0,"retry_delay_hours":0},
        "search":{"paths":10,"generations":2,"population":4,"seed":42,"holdout_seed":43},
        "initial":{"risk":500,"target":500}, "regime_set":"compact",
        "bounds":{"risk":[500,2000],"target":[150,3000]},
        "choices":{"risk":[500,1000,1500,2000],"target":[150,500,1000,1500,2000,3000]},
        "risk":{"target_ruin_probability":.01}}


def test_search_without_history_uses_flat_start_and_no_example_seed(monkeypatch):
    actual_fit = research.fit_targets
    calls = []
    def fit(*args, **kwargs):
        assert not kwargs.get("candidates")
        assert {r.risk_dollars for r in kwargs["policy"].sizing.regimes} == {500}
        assert set(kwargs["policy"].targets) == {500}
        calls.append(True)
        return actual_fit(*args, **kwargs)
    monkeypatch.setattr(research, "fit_targets", fit)
    req = request_fixture()
    original = deepcopy(req)
    output = replay.run(req)
    assert req == original and calls == [True]
    assert output["fit"]["candidate_seeds"] == 0
    assert output["reference_used_for_selection"] is False
    assert output["fit"]["training"]["paths"] == 7
    assert output["fit"]["holdout"]["paths"] == output["risk"]["paths"] == 3
    assert output["fit"]["evaluations"] > 1
    assert output["scope"].endswith("not historical strategy IS/OOS")
    assert output["training_uncertainty"]["selected_on_sample"]
    assert output["uncertainty"]["holdout"]["sample_kind"] == "independent_model"
    assert "objective_gain" in output["uncertainty"]["paired_gain"]["metrics"]
    json.dumps(output, allow_nan=False)


def test_holdout_cannot_select_policy_and_objective_reconciles():
    request = request_fixture()
    first = replay.run(request)
    request["search"]["holdout_seed"] = 89
    second = replay.run(request)
    assert first["fit"]["policy"] == second["fit"]["policy"]
    assert first["fit"]["training"] == second["fit"]["training"]
    fit = first["fit"]
    assert fit["holdout"]["score"] == pytest.approx(first["risk"]["distributions"]["net_cash_per_day"]["mean"])
    assert fit["paired_holdout_gain"] == pytest.approx(fit["holdout"]["score"]-fit["baseline_holdout"]["score"])
    baseline = first["initial_policy"]
    policy = TargetPolicy(DollarPolicy(tuple(RiskRegime(**r) for r in baseline["sizing"]["regimes"])), tuple(baseline["targets"]))
    training = np.random.default_rng(np.random.SeedSequence([42,0])).random((7,7))
    result = evaluate_targets(replay_50k(**request["account"]), BracketModel(sessions=7), policy,
        BacktestConfig(0,timedelta(0),timedelta(0),timedelta(0),initial_wallet=2000), training, risk=None)
    assert fit["training"]["score"] >= result.score


def test_continuous_detailed_search_and_alternate_objective():
    request = request_fixture()
    request.update(choices=None, regime_set="detailed", objective="net_cash")
    output = replay.run(request)
    assert len(output["fit"]["policy"]["targets"]) == 10
    assert output["fit"]["holdout"]["score"] == pytest.approx(output["fit"]["holdout"]["mean_net_cash"])
    assert all(v >= 150 for v in output["fit"]["policy"]["targets"])
    rate = output["fit"]["holdout"]["distributions"]["net_cash_per_day"]
    assert rate == output["risk"]["distributions"]["net_cash_per_day"]
    for summary in (output["fit"]["training"], output["fit"]["baseline_holdout"], output["reference_holdout"]):
        assert "net_cash_per_day" in summary["distributions"]


def test_dashboard_rejects_candidate_injection_and_invalid_choices():
    request = request_fixture()
    request["search"]["candidates"] = []
    with pytest.raises(ValueError, match="candidate seeds"):
        replay.run(request)
    del request["search"]["candidates"]
    request["choices"]["target"] = [0]
    with pytest.raises(ValueError, match="choices"):
        replay.run(request)
