"""Independent exact-dollar oracle and controlled model-search contracts."""
from dataclasses import replace
from datetime import timedelta
from fractions import Fraction as F
from itertools import product
from math import exp, prod

import numpy as np
import pytest

from propfirm_engine import BacktestConfig, DollarPolicy, Engine
from propfirm_engine.firms.lucidflex import replay_50k
from propfirm_engine.target_research import (
    BracketModel, TargetPolicy, bracket_probability, evaluate_targets,
    fit_targets, lucidflex_example, research_path,
)


SPEC = replay_50k(eval_fee=105.2, reset_fee=105, contract_type="micro")
CONFIG = BacktestConfig(0, timedelta(0), timedelta(0), timedelta(0))


def _seven_day_oracle(outcomes):
    """Independent small oracle, ONLY the zero-cost, instant-payment 7-day case.

    No production engine classes, rule evaluators, ledger or projection helpers.
    All amounts are offsets from 50,000. This intentionally duplicates the small
    benchmark's numeric rules so a production regression cannot fix both sides.
    """
    phase, balance, floor, best, qualifying = "eval", F(0), F(-2000), F(0), 0
    pending, fee, cash, weight = None, F("105.2"), F(0), F(1)
    trades, failures, passes, receipts = [], [], [], F(0)
    for day, won in enumerate(outcomes):
        if fee:
            cash -= fee
            fee = F(0)
        if pending is not None:
            phase, balance, floor, best, qualifying = pending, F(0), F(-2000), F(0), 0
            pending = None
        stop = min(F(2000), balance - floor)
        target = F(1500 if phase == "eval" else 2000 if qualifying == 0 else 150)
        probability = stop / (stop + target)
        weight *= probability if won else 1 - probability
        pnl = target if won else -stop
        balance += pnl
        trades.append((phase, balance + 50000, floor + 50000))
        if balance <= floor:
            failures.append(day)
            pending = "eval"
            fee = F(105) if phase == "eval" else F("105.2")
            continue
        best = max(best, pnl)
        if phase == "eval" and balance >= 3000 and best * 2 <= balance:
            passes.append(day)
            pending = "funded"
            continue
        floor = min(F(100), max(floor, balance - 2000))
        if phase == "funded":
            qualifying += pnl >= 150
            if qualifying >= 5 and balance >= 1000:
                gross = min(F(2000), balance / 2)
                balance -= gross
                floor = F(100)
                qualifying = 0
                receipt = gross * F(9, 10)
                cash += receipt
                receipts += receipt
    return weight, cash, receipts, trades, failures, passes


def test_every_seven_session_branch_against_independent_fraction_oracle():
    total, oracle_cash, engine_cash, payout_probability = F(0), F(0), 0.0, F(0)
    for outcomes in product((False, True), repeat=7):
        weight, cash, receipts, trades, failures, passes = _seven_day_oracle(outcomes)
        path = research_path(SPEC, BracketModel(sessions=7), lucidflex_example(), CONFIG,
                             [0.0 if won else np.nextafter(1.0, 0.0) for won in outcomes])
        result = path.replay
        actual = [(e.phase, F(str(e.balance)), F(str(e.floor))) for e in result.events if e.kind == "trade"]
        assert actual == trades, outcomes
        assert result.net_cash == float(cash), outcomes
        assert result.receipts == float(receipts), outcomes
        assert result.failed_attempts == len(failures)
        assert sum(e.kind == "evaluation_pass" for e in result.events) == len(passes)
        model_weight = prod(d.probability if d.won else 1 - d.probability for d in path.decisions)
        assert model_weight == pytest.approx(float(weight))
        total += weight
        oracle_cash += weight * cash
        engine_cash += model_weight * result.net_cash
        if receipts:
            payout_probability += weight
    assert total == 1
    assert payout_probability == F(16, 49) * F(1, 2) * F(40, 43)**4
    assert engine_cash == pytest.approx(float(oracle_cash))


def test_example_survives_later_full_loss_and_keeps_qualifying_days():
    path = research_path(SPEC, BracketModel(sessions=5), lucidflex_example(), CONFIG, [0, 0, 0, 0, 0.999])
    trades = [e for e in path.replay.events if e.kind == "trade"]
    assert trades[-1].balance == 50150
    assert trades[-1].floor == 50100
    assert trades[-1].qualifying_days == 2
    assert path.replay.failed_attempts == 0


def test_probability_limits_drift_and_cost_adjustment():
    assert bracket_probability(2000, 1500) == pytest.approx(4 / 7)
    for mu in (-200, -1e-9, 0, 1e-9, 200):
        p = bracket_probability(2000, 1500, mu=mu)
        assert 0 < p < 1
        assert p + bracket_probability(1500, 2000, mu=-mu) == pytest.approx(1)
    nu = 2 * 200 / 1000**2
    assert bracket_probability(2000, 1500, mu=200) == pytest.approx((1-exp(-nu*2000))/(1-exp(-nu*3500)))
    assert bracket_probability(2000, 1500, mu=1e8) == 1
    assert bracket_probability(2000, 1500, mu=-1e8) == 0
    config = replace(CONFIG, cost_per_contract=1, cost_per_trade=2)
    path = research_path(SPEC, BracketModel(sessions=7), lucidflex_example(), config, [0]*7)
    assert path.decisions[0].probability == pytest.approx(1997 / 3500)
    assert path.replay.receipts == 1170
    assert all(e.quantity == 1 for e in path.replay.events if e.kind == "trade")


@pytest.mark.parametrize("kwargs", [{"sigma": 0}, {"sigma": True}, {"mu": float("nan")}])
def test_probability_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        bracket_probability(2000, 1500, **kwargs)


def test_loss_at_non_integer_floor_breaches_exactly():
    policy = TargetPolicy(DollarPolicy.constant(2000), (1500, 2000.001))
    # One funded gain locks at 50,000.001, then loss leaves the floor exactly.
    path = research_path(SPEC, BracketModel(sessions=4), policy, CONFIG, [0, 0, 0, 0.999])
    assert path.replay.failed_attempts == 1
    assert any(e.code == "FAIL_TRAILING_DD" for e in path.replay.events)


def test_risk_clamps_to_remaining_buffer_and_costs_are_not_double_counted():
    path = research_path(SPEC, BracketModel(sessions=6), lucidflex_example(), CONFIG,
                         [0, 0, 0, 0, .999, .999])
    assert path.decisions[-1].net_risk == 50
    assert path.decisions[-1].probability == pytest.approx(50 / 200)
    assert path.replay.failed_attempts == 1
    config = replace(CONFIG, cost_per_contract=1.2, cost_per_trade=0.8)
    path = research_path(SPEC, BracketModel(sessions=1), lucidflex_example(), config, [.999])
    assert path.replay.failed_attempts == 1
    assert next(e.balance for e in path.replay.events if e.kind == "trade") == 48000


def test_costs_too_large_distinguish_policy_skip_from_untradeable_account():
    model = BracketModel(sessions=2)
    policy = TargetPolicy(DollarPolicy.constant(100), (100, 100))
    path = research_path(SPEC, model, policy, replace(CONFIG, cost_per_contract=100), [0, 0])
    assert not path.decisions
    assert any(e.kind == "policy_skip" for e in path.replay.events)
    assert path.replay.failed_attempts == 0
    path = research_path(SPEC, model, policy, replace(CONFIG, cost_per_contract=2000), [0, 0])
    assert any(e.code == "CAPPED_OUT" for e in path.replay.events)


def test_holdout_seed_does_not_select_policy_and_baseline_is_in_search():
    model = BracketModel(sessions=8)
    policy = lucidflex_example()
    names = [r.name for r in policy.sizing.regimes]
    kwargs = dict(policy=policy, risk_bounds={n: (100, 2000) for n in names},
                  target_bounds={n: (150, 3000) for n in names}, paths=10,
                  seed=12, generations=1, population=2)
    first = fit_targets(SPEC, model, CONFIG, **kwargs, holdout_seed=20)
    second = fit_targets(SPEC, model, CONFIG, **kwargs, holdout_seed=30)
    assert first.policy == second.policy
    assert first.training == second.training
    assert first.training.paths == 7 and first.holdout.paths == 3
    tapes = np.random.default_rng(np.random.SeedSequence([12, 0])).random((7, 8))
    baseline = evaluate_targets(SPEC, model, policy, CONFIG, tapes)
    assert first.training.score >= baseline.score
    assert "NOT historical" in first.scope
    assert first.paired_holdout_gain == pytest.approx(first.holdout.score - first.baseline_holdout.score)


def test_research_trace_is_reproducible_and_historical_api_rejects_target_policy():
    model, policy = BracketModel(sessions=7), lucidflex_example()
    first = research_path(SPEC, model, policy, CONFIG, [0]*7)
    assert first == research_path(SPEC, model, policy, CONFIG, [0]*7)
    assert first.replay.receipts == 1170
    assert any("passage times" in a for a in first.replay.assumptions)
    with pytest.raises(TypeError, match="DollarPolicy"):
        from propfirm_engine.target_research import _calendar
        Engine().backtest(SPEC, _calendar(SPEC, model), policy, CONFIG)


def test_gridded_search_custom_objective_and_unvisited_regimes():
    policy = lucidflex_example()
    names = [r.name for r in policy.sizing.regimes]
    baseline = TargetPolicy(DollarPolicy(tuple(replace(r, risk_dollars=500)
                            for r in policy.sizing.regimes)), (500, 500, 500))
    fitted = fit_targets(SPEC, BracketModel(sessions=1), CONFIG, policy=baseline,
        risk_bounds={n: (500, 2000) for n in names}, target_bounds={n: (150, 3000) for n in names},
        risk_choices={n: (500, 2000) for n in names}, target_choices={n: (150, 500, 1500) for n in names},
        paths=7, generations=2, population=4, objective=lambda r: r.fees, direction="minimize")
    assert fitted.holdout.score == pytest.approx(105.2)
    assert fitted.policy.sizing.regimes[1:] == baseline.sizing.regimes[1:]
    assert fitted.policy.targets[1:] == (500, 500)


@pytest.mark.parametrize("tape", [[0]*6, [1]*7, [True]*7, [float("nan")]*7])
def test_invalid_uniform_tapes(tape):
    with pytest.raises(ValueError):
        research_path(SPEC, BracketModel(sessions=7), lucidflex_example(), CONFIG, tape)


def test_benchmark_exports_strict_json_and_does_not_seed_example():
    import json
    from benchmarks.target_policy import exportable, run_experiment
    result = run_experiment(paths=7, sessions=7, generations=0)
    restored = json.loads(json.dumps(exportable(result), allow_nan=False))
    assert restored["fit"]["candidate_seeds"] == 0
    assert restored["example_used_for_selection"] is False
    assert restored["fit"]["training"]["paths"] == 4
    assert restored["fit"]["holdout"]["paths"] == 3
