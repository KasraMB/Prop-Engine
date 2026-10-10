from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from propfirm_engine import BracketHistory, DollarPolicy, Engine, PhaseSearch
from propfirm_engine.phase_search import (
    PhaseCandidate, PhaseSample, _Attempt, _axes, _metrics, _phase_sample, _retain,
)
from test_chronological_backtest import config, session_day, spec, trade


def history(n=30):
    return BracketHistory(tuple(trade(session_day(i), won=i % 5 != 4, target=800) for i in range(n)))


def settings(**kwargs):
    return PhaseSearch(**(dict(trade_budget=1000, horizon_sessions=3,
                              stride_sessions=3, archive_size=4, population=4) | kwargs))


def fit(**kwargs):
    return Engine().fit_phases(spec(), kwargs.pop("history", history()), kwargs.pop("config", config()),
        policy=DollarPolicy.constant(100), risk_bounds={"evaluation": (50, 300), "funded": (50, 300)},
        search=kwargs.pop("search", settings()), **kwargs)


def test_single_eval_stops_at_failure_and_does_not_restart():
    profile = spec()
    profile = replace(profile, account=replace(profile.account, phases=profile.account.phases[:1]))
    log = BracketHistory(tuple(trade(session_day(i), won=False, stop=2000) for i in range(3)))
    result = _Attempt(profile, log, DollarPolicy.constant(2000), config()).run()
    assert result.attempts == result.failed_attempts == 1
    assert result.fees == 105.2
    assert len([e for e in result.events if e.kind == "trade"]) == 1
    sample = _phase_sample(profile, log, DollarPolicy.constant(2000), config())
    assert sample.outcome == "failed" and sample.duration_days < result.calendar_days


def test_phase_censoring_is_not_a_failure_or_lifetime_duration():
    profile = spec()
    profile = replace(profile, account=replace(profile.account, phases=profile.account.phases[:1]))
    result = _phase_sample(profile, history(3), DollarPolicy.constant(0), config())
    metrics = _metrics((result,))
    assert result.outcome == "unresolved"
    assert metrics["failure_probability"] == 0 and metrics["unresolved_fraction"] == 1
    assert metrics["cost_per_observed_pass"] is None


def test_phase_metrics_have_independent_arithmetic():
    metrics = _metrics((PhaseSample("passed", 2, 0, 100, 0),
                        PhaseSample("failed", 1, 0, 100, 0),
                        PhaseSample("unresolved", 3, 0, 100, 0)))
    assert metrics["pass_probability"] == 1/3
    assert metrics["passes_per_day"] == 1/6
    assert metrics["cost_per_observed_pass"] == 300
    assert metrics["mean_duration_days"] == 2
    assert metrics["payout_variance"] == 0


def test_funded_screen_keeps_multiple_payouts_and_delayed_receipts():
    profile = spec()
    phase = profile.account.phases[1]
    phase = replace(phase, payout_schema=replace(phase.payout_schema, max_payouts=2))
    profile = replace(profile, account=replace(profile.account, phases=(phase,), eval_fee=0))
    log = BracketHistory(tuple(trade(session_day(i), target=300) for i in range(18)))
    result = _Attempt(profile, log, DollarPolicy.constant(100), config(receipt_delay=timedelta(days=1))).run()
    assert result.attempts == 1
    assert len([e for e in result.events if e.kind == "receipt"]) == 2
    assert result.receipts == 675 + 1012.5
    assert result.outstanding_payouts == 0
    sample = _phase_sample(profile, log, DollarPolicy.constant(100), config(receipt_delay=timedelta(days=1)))
    assert sample.outcome == "handoff"
    first = next(e for e in result.events if e.kind == "receipt")
    assert sample.duration_days > (first.at-result.start).total_seconds()/86400


@pytest.mark.parametrize("architecture", ["separate", "joint"])
def test_budget_and_reports_preserve_full_lifecycle(architecture):
    result = fit(architecture=architecture)
    assert result.work.search_trade_visits <= result.work.budget
    assert result.work.lifecycle_evaluations >= 1
    assert result.work.report_trade_visits == 30
    assert result.score == result.out_of_sample.net_cash_per_day
    assert {"net_cash", "net_cash_per_day", "required_bankroll"} <= result.out_of_sample_risk["distributions"].keys()
    direct = Engine().backtest(spec(), history().split(.7)[1], result.policy, config())
    assert result.out_of_sample == direct
    assert len(result.evaluation_candidates) <= 4 and len(result.funded_candidates) <= 4


@pytest.mark.parametrize("architecture", ["separate", "joint"])
def test_oos_cannot_change_candidates_or_selection(architecture):
    original = history()
    changed = BracketHistory(tuple(t if i < 21 else replace(t, won=not t.won, take_profit=10000)
                                  for i, t in enumerate(original.trades)))
    a, b = fit(architecture=architecture), fit(architecture=architecture, history=changed)
    assert a.policy == b.policy and a.fit.in_sample_score == b.fit.in_sample_score
    assert a.evaluation_candidates == b.evaluation_candidates
    assert a.funded_candidates == b.funded_candidates
    assert a.work.search_trade_visits == b.work.search_trade_visits


def test_pareto_archive_preserves_tradeoffs_and_rejects_dominated_points():
    def candidate(p, days, cost):
        return PhaseCandidate(DollarPolicy.constant(cost), dict(pass_probability=p,
            passes_per_day=p/days, cost_per_observed_pass=cost, mean_duration_days=days), ())
    a, b, c = candidate(.2, 1, 500), candidate(.5, 3, 200), candidate(.1, 4, 600)
    archive = _retain([], a, "eval", 4)
    archive = _retain(archive, b, "eval", 4)
    assert _retain(archive, c, "eval", 4) == [a, b]
    for i in range(30):
        archive = _retain(archive, candidate(.3+i/100, 2+i/10, 400-i), "eval", 4)
    assert len(archive) <= 4
    for i, x in enumerate(archive):
        assert not any(all(a >= b for a, b in zip(_axes(y, "eval"), _axes(x, "eval")))
                       for j, y in enumerate(archive) if i != j)


@pytest.mark.parametrize("direction, expected", [("maximize", 300), ("minimize", 50)])
def test_joint_direction_selects_competing_candidates(monkeypatch, direction, expected):
    import propfirm_engine.phase_search as module
    class Search:
        def __init__(self, x, *args, **kwargs):
            self.x = x
        def optimize(self, evaluate):
            evaluate(np.zeros(len(self.x)))
            evaluate(np.ones(len(self.x)))
    monkeypatch.setattr(module, "CMAES", Search)
    result = fit(architecture="joint", direction=direction,
                 objective=lambda r: r.policy.regimes[0].risk_dollars)
    assert result.policy.regimes[0].risk_dollars == expected


def test_constraint_is_training_only_and_bad_outputs_fail():
    seen = []
    def constraint(result):
        seen.append(result.end)
        return result.policy.regimes[0].risk_dollars <= 100
    result = fit(architecture="joint", constraint=constraint)
    assert max(seen) == result.in_sample.end < result.out_of_sample.start
    assert result.policy.regimes[0].risk_dollars <= 100
    with pytest.raises(ValueError, match="no searched"):
        fit(constraint=lambda r: False)
    with pytest.raises(ValueError, match="bool"):
        fit(constraint=lambda r: 1)
    with pytest.raises(ValueError, match="finite"):
        fit(objective=lambda r: float("nan"))


def test_compare_uses_same_budget_and_does_not_select_an_oos_winner():
    results = Engine().compare_searches(spec(), history(), config(), policy=DollarPolicy.constant(100),
        risk_bounds={"evaluation": (50, 300), "funded": (50, 300)}, search=settings())
    assert set(results) == {"joint", "separate"}
    assert results["joint"].work.budget == results["separate"].work.budget


@pytest.mark.parametrize("kwargs", [{"trade_budget": 0}, {"archive_size": 3},
    {"population": 1}, {"horizon_sessions": True}, {"stride_sessions": 1.5}])
def test_bad_settings_fail(kwargs):
    with pytest.raises(ValueError):
        settings(**kwargs)


def test_insufficient_budget_or_history_is_explicit():
    with pytest.raises(ValueError, match="baseline"):
        fit(search=settings(trade_budget=1))
    with pytest.raises(ValueError, match="four candidate"):
        fit(search=settings(trade_budget=100))
    with pytest.raises(ValueError, match="available sessions"):
        fit(search=settings(horizon_sessions=100))


@pytest.mark.parametrize("architecture", ["separate", "joint"])
def test_reported_work_matches_all_engine_inputs(monkeypatch, architecture):
    import propfirm_engine.phase_search as module
    original, phase = module.backtest, module._phase_sample
    visits = []
    def replay(spec, log, *args, **kwargs):
        visits.append(len(log.trades))
        return original(spec, log, *args, **kwargs)
    def screen(spec, log, *args, **kwargs):
        visits.append(len(log.trades))
        return phase(spec, log, *args, **kwargs)
    monkeypatch.setattr(module, "backtest", replay)
    monkeypatch.setattr(module, "_phase_sample", screen)
    result = fit(architecture=architecture)
    assert sum(visits) == result.work.search_trade_visits + result.work.report_trade_visits
    assert result.work.unused_budget >= 0


def test_wallet_truncation_does_not_understate_required_capital():
    result = fit(config=config(initial_wallet=0))
    assert result.out_of_sample.attempts == 0
    assert result.work.report_trade_visits == 60
    assert result.out_of_sample_risk["distributions"]["required_bankroll"]["minimum"] >= 105.2


def test_separate_search_explores_both_directions_and_all_phase_objectives(monkeypatch):
    import propfirm_engine.phase_search as module
    started = []
    class Search:
        def __init__(self, x, *args, **kwargs):
            self.x = x
            started.append(kwargs["seed"])
        def optimize(self, evaluate):
            evaluate(np.zeros(len(self.x)))
            evaluate(np.ones(len(self.x)))
    monkeypatch.setattr(module, "CMAES", Search)
    a = fit(search=settings(trade_budget=3000), objective=lambda r: r.policy.regimes[0].risk_dollars)
    b = fit(search=settings(trade_budget=3000), direction="minimize",
            objective=lambda r: r.policy.regimes[0].risk_dollars)
    assert a.policy.regimes[0].risk_dollars > b.policy.regimes[0].risk_dollars
    assert len(started) == 14 and len(set(started)) == 7


def test_seeded_candidates_and_sample_summaries_are_repeatable():
    a, b = fit(seed=13), fit(seed=13)
    assert a.fit == b.fit
    assert a.evaluation_candidates == b.evaluation_candidates
    assert a.funded_candidates == b.funded_candidates
    assert a.work.search_trade_visits == b.work.search_trade_visits


def test_compact_search_does_not_retain_candidate_ledgers():
    import tracemalloc
    fit()
    tracemalloc.start()
    try:
        result = fit(search=settings(trade_budget=2000))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 2_000_000
    assert all(not hasattr(c, "events") for c in result.evaluation_candidates + result.funded_candidates)


def test_unvisited_funded_regime_keeps_baseline():
    log = BracketHistory(tuple(trade(session_day(i), target=1) for i in range(30)))
    result = fit(history=log)
    assert result.policy.regimes[1].risk_dollars == 100


def test_eval_screen_stops_after_pass_without_entering_funded():
    profile = spec()
    profile = replace(profile, account=replace(profile.account, phases=profile.account.phases[:1]))
    log = BracketHistory(tuple(trade(session_day(i), target=1500, won=i < 2) for i in range(5)))
    result = _Attempt(profile, log, DollarPolicy.constant(100), config()).run()
    assert result.status == "EVALUATION_PASSED"
    assert result.final_balance == 53000 and result.failed_attempts == 0
    assert len([e for e in result.events if e.kind == "trade"]) == 2
    assert result.fees == 105.2 and result.receipts == 0
