from dataclasses import replace
from fractions import Fraction

import pytest

from propfirm_engine import BracketHistory, DollarPolicy, Engine, RollingConfig
from propfirm_engine.backtest import net_cash, net_cash_per_day
from propfirm_engine.rolling import _prepare_rolling, _score_cash, rolling_backtest
from propfirm_engine.money import decimal_cash
from test_rolling_backtest import inputs


@pytest.mark.parametrize("wallet", [None, 100, 2000])
@pytest.mark.parametrize("objective", [net_cash, net_cash_per_day])
def test_compact_search_matches_all_window_scores_and_regimes(wallet, objective):
    spec, history, policy, config = inputs(65)
    config = replace(config, initial_wallet=wallet)
    rolling = RollingConfig(40, 1)
    prepared = _prepare_rolling(history, rolling)
    compact = _score_cash(spec, prepared, policy, config, objective)
    detailed = rolling_backtest(spec, history, policy, config, rolling=rolling, objective=objective)
    assert compact.score == detailed.score
    assert compact.visited_regimes == detailed.visited_regimes


def test_fitter_uses_compact_search_only_for_known_cash_objectives():
    spec, history, policy, config = inputs(40)
    events = []
    common = dict(policy=policy, risk_bounds={r.name: (50, 500) for r in policy.regimes},
                  rolling=RollingConfig(8, 1), generations=2, population=4, seed=42)
    compact = Engine().fit(spec, history, config, objective=net_cash, progress=events.append, **common)
    generic = Engine().fit(spec, history, config, objective=lambda r: r.net_cash, **common)
    assert compact == generic
    inner = [e for e in events if e["stage"] == "search_windows"]
    assert inner and inner[0]["completed_windows"] == 0
    assert inner[-1]["completed_windows"] == inner[-1]["windows"]


@pytest.mark.parametrize("cost", [0, .1, 1, .3333333333333333])
def test_integer_compaction_matches_all_fraction_event_accounting(monkeypatch, cost):
    from propfirm_engine import backtest, reference
    spec, history, _, config = inputs(50)
    history = BracketHistory(tuple(replace(t, stop_loss=100.1,
        take_profit=1500.2 if i % 8 < 2 else 200.3, won=i % 8 != 7)
        for i, t in enumerate(history.trades)))
    config = replace(config, cost_per_contract=cost, cost_per_trade=.05, initial_wallet=2000)
    policy = DollarPolicy.constant(200.123456789)
    compact = Engine().backtest(spec, history, policy, config)
    exact = lambda value: value if isinstance(value, Fraction) else Fraction(str(value))
    monkeypatch.setattr(backtest, "decimal_cash", exact)
    monkeypatch.setattr(reference, "decimal_cash", exact)
    rational = Engine().backtest(spec, history, policy, config)
    assert compact == rational


def test_decimal_cache_never_rounds_fractional_money_or_large_integers():
    for value in [0, 10**30+1, .1, .10000000000000002, Fraction(1, 3), Fraction(4, 2)]:
        assert decimal_cash(value) == (value if isinstance(value, Fraction) else Fraction(str(value)))
    assert type(decimal_cash(100.0)) is int
    assert isinstance(decimal_cash(.1), Fraction)
    assert decimal_cash.cache_info().maxsize == 4096


def test_custom_callable_equality_does_not_enable_cash_only_search():
    spec, history, policy, config = inputs(30)
    class Objective:
        def __eq__(self, other): return True
        def __call__(self, result): return len(result.events)
    options = dict(policy=policy, risk_bounds={r.name: (50, 500) for r in policy.regimes},
                   rolling=RollingConfig(8), generations=0)
    custom = Engine().fit(spec, history, config, objective=Objective(), **options)
    expected = Engine().fit(spec, history, config, objective=lambda r: len(r.events), **options)
    assert custom == expected


@pytest.mark.parametrize("risk", [0, .01, 99.99, 100, 100.1, 2000, 2000.01])
@pytest.mark.parametrize("cost,fixed", [(0, 0), (.1, .05), (1, 2)])
def test_integer_projection_matches_reference_projection(risk, cost, fixed):
    from propfirm_engine.feasibility import project_position
    spec, history, _, config = inputs(1)
    config = replace(config, cost_per_contract=cost, cost_per_trade=fixed)
    result = Engine().backtest(spec, history, DollarPolicy.constant(risk), config)
    f = lambda x: Fraction(str(x))
    unit = f(100)+f(cost)
    desired = (f(risk)-f(fixed))/unit
    project = getattr(project_position, "py_func", project_position)
    quantity, capped, _, _ = project(max(f(0), desired), f(2000), f(1), unit, f(1), fixed_cost=f(fixed))
    trades = [e for e in result.events if e.kind == "trade"]
    if risk == 0 or desired < 1:
        assert not trades
    else:
        assert not capped
        assert trades[0].quantity == min(int(quantity), spec.eval_contract_limit)
