"""Price fitting isolates chronological holdout and never optimizes execution assumptions."""
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from propfirm_engine import BacktestConfig, DollarPolicy, Engine, Instrument, PriceSession, SlippageModel, TickDistribution
from propfirm_engine.firms.lucidflex import replay_50k
from propfirm_engine.price_fitting import evaluate_prices
from propfirm_engine.target_research import TargetPolicy


SPEC = replay_50k(eval_fee=105.2, reset_fee=105, contract_type="mini")
CONFIG = BacktestConfig(3.5, timedelta(0), timedelta(0), timedelta(0))
POLICY = TargetPolicy(DollarPolicy.constant(500), (500, 500))
INSTRUMENT = Instrument("ES", 50, .25)
ZERO = TickDistribution((1.,))
MODEL = SlippageModel(ZERO, TickDistribution((.5,.5)), ZERO, TickDistribution((0.,0.,1.)), 4)


def history():
    result = []
    for week in range(2):
        for day in range(6, 11):
            d = date(2025, 1, day)+timedelta(days=7*week)
            at = datetime.combine(d, datetime.min.time(), timezone.utc).replace(hour=15)
            result.append(PriceSession(d, at+timedelta(minutes=1),
                                      [int(at.timestamp()*1e9)], [[100,112,99,110]]))
    return result


def fit(sessions=None, **kwargs):
    return Engine().fit_prices(SPEC, sessions or history(), INSTRUMENT, CONFIG, policy=POLICY,
                              risk_bounds={r.name:(100,2000) for r in POLICY.sizing.regimes},
                              target_bounds={r.name:(100,3000) for r in POLICY.sizing.regimes},
                              collision_policy="stop_first", **kwargs)


def test_split_and_objective_independent_distributions():
    out = fit(generations=0, paths=2, slippage=MODEL, objective=lambda r:r.net_cash)
    assert len(out.train_sessions) == 7 and len(out.test_sessions) == 3
    assert out.train_sessions[-1] < out.test_sessions[0]
    assert out.score == out.out_of_sample.score
    assert "net_cash_per_day" in out.out_of_sample.distributions
    assert "required_bankroll" in out.out_of_sample.distributions
    assert len(out.out_of_sample.paths) == 2
    assert out.out_of_sample.paths[0].replay.attempts >= 1
    assert out.evaluations == 1


def test_changing_oos_does_not_change_selected_policy_or_is_score():
    sessions = history()
    altered = sessions[:7]+[replace(s, ohlc=np.array([[100,101,10,20]])) for s in sessions[7:]]
    a = fit(sessions, generations=1, population=4, seed=17, paths=2, slippage=MODEL)
    b = fit(altered, generations=1, population=4, seed=17, paths=2, slippage=MODEL)
    assert a.policy == b.policy
    assert a.in_sample.score == b.in_sample.score
    assert a.evaluations == b.evaluations
    assert a.out_of_sample.paths[0].decisions != b.out_of_sample.paths[0].decisions


def test_custom_minimization_and_nonfinite_objective_rejected():
    out = fit(generations=1, population=4, direction="minimize", objective=lambda r:r.fees)
    assert out.direction == "minimize"
    assert out.in_sample.score >= 0
    with pytest.raises(ValueError, match="finite real"):
        fit(generations=0, objective=lambda r:float("nan"))


def test_no_slippage_does_not_duplicate_identical_paths():
    out = evaluate_prices(SPEC, history(), POLICY, INSTRUMENT, CONFIG, paths=10, collision_policy="stop_first")
    assert len(out.paths) == 1
    assert out.distributions["net_cash"]["distribution"]["count"] == 1


@pytest.mark.parametrize("kwargs", [{"paths":0}, {"train_fraction":1}, {"population":1}, {"direction":"bad"}])
def test_invalid_search_settings_rejected(kwargs):
    with pytest.raises(ValueError): fit(**kwargs)
