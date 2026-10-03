"""Independent gambler's-ruin oracles and canonical lifecycle bootstrap parity."""
from dataclasses import replace
from datetime import date, datetime, timedelta
import json
from math import exp, log

import numpy as np
import pytest

from propfirm_engine import BacktestConfig, BracketHistory, BracketTrade, CashCycle, DollarPolicy, Engine, RiskConfig, RuinConfig, ultimate_cycle_ruin
from propfirm_engine.firms.lucidflex import replay_50k
from propfirm_engine.ruin import _adjustment_rate, _bernoulli_bounds, _resampled_history, _settled_cycles
from propfirm_engine.resampling import StationaryDayBootstrap
from propfirm_engine.risk import cash_risk_path


SPEC = replay_50k(eval_fee=105.20, reset_fee=105, contract_type="micro")
CONFIG = BacktestConfig(0, timedelta(0), timedelta(days=2), timedelta(0))


def history():
    trades, day = [], date(2026, 3, 2)
    for i in range(12):
        while day.weekday() > 4:
            day += timedelta(days=1)
        start = datetime.fromisoformat(f"{day}T10:00:00-05:00")
        trades.append(BracketTrade(start, start+timedelta(minutes=2), day, 2000, 1500, i % 3 != 2))
        day += timedelta(days=1)
    return BracketHistory(tuple(trades))


def test_ultimate_monte_carlo_matches_classical_gamblers_ruin():
    # +$1 with p=.6, -$1 with q=.4; ruin at inability to stake $1.
    law = [CashCycle(1, 1)]*3 + [CashCycle(-1, 1)]*2
    actual = ultimate_cycle_ruin(law, bankroll=5, paths=20000, max_cycles=2000, seed=8)
    exact = (2/3)**5
    assert actual["confidence_bounds"][0] <= exact <= actual["confidence_bounds"][1]
    assert actual["probability_bounds"][0] == pytest.approx(exact, abs=.01)
    assert actual["unresolved_paths"] == 0
    assert actual["probability_bounds"][1]-actual["probability_bounds"][0] <= .0001
    assert actual["adjustment_rate_per_dollar"] == pytest.approx(log(1.5))
    sufficient = actual["sufficient_bankroll"]
    assert exp(-log(1.5)*(sufficient-1)) <= .01


@pytest.mark.parametrize("law", [[CashCycle(-1,1)], [CashCycle(1,1),CashCycle(-1,1)]])
def test_nonpositive_cycle_drift_is_ultimate_ruin_not_finite_horizon(law):
    r = ultimate_cycle_ruin(law, bankroll=100000, max_cycles=1)
    assert r["probability_bounds"] == [1,1]
    assert r["sufficient_bankroll"] is None
    assert r["simulated_cycles"] == 0


def test_zero_net_cycles_still_have_within_cycle_funding_risk():
    assert ultimate_cycle_ruin([CashCycle(0,100)], bankroll=100)["probability_bounds"] == [0,0]
    assert ultimate_cycle_ruin([CashCycle(0,100)], bankroll=99)["probability_bounds"] == [1,1]
    assert ultimate_cycle_ruin([CashCycle(0,0),CashCycle(0,100)], bankroll=99)["probability_bounds"] == [1,1]


def test_positive_cycles_with_different_cash_needs_can_ruin_before_growth():
    # At $1, drawing cycle A first survives forever; B first ruins. Exact psi=.5.
    r = ultimate_cycle_ruin([CashCycle(1,1), CashCycle(1,2)], bankroll=1, paths=10000, seed=7)
    assert r["probability_bounds"][0] == pytest.approx(.5, abs=.02)
    assert r["confidence_bounds"][0] <= .5 <= r["confidence_bounds"][1]
    assert r["sufficient_bankroll"] == 2


def test_budget_reached_survivors_are_unresolved_not_safe():
    law = [CashCycle(1,1)]*3 + [CashCycle(-1,1)]*2
    r = ultimate_cycle_ruin(law, bankroll=5, paths=100, max_cycles=1)
    assert r["status"] == "simulation_budget_reached"
    assert r["unresolved_paths"] == 100
    assert r["probability_bounds"] == [0,1]
    assert r["confidence_bounds"] == [0,1]


def test_fee_equality_and_cent_arithmetic():
    assert ultimate_cycle_ruin([CashCycle(.1,.3)], bankroll=.3)["probability_bounds"] == [0,0]
    assert ultimate_cycle_ruin([CashCycle(.1,.3)], bankroll=.29)["probability_bounds"] == [1,1]
    with pytest.raises(ValueError, match="whole cents"):
        ultimate_cycle_ruin([CashCycle(.001,1)], bankroll=1)


def test_chernoff_zero_failure_bound_is_not_zero():
    lo, hi = _bernoulli_bounds(0,100,.05)
    assert lo == 0
    assert hi == pytest.approx(1-exp(-log(40)/100))
    assert _bernoulli_bounds(100,100,.05)[1] == 1


def test_extremely_small_target_does_not_overflow_reciprocal():
    law = [CashCycle(1,1)]*3 + [CashCycle(-1,1)]*2
    r = ultimate_cycle_ruin(law, bankroll=1, target=5e-324, paths=2, max_cycles=1)
    assert r["sufficient_bankroll"] > 1000


def test_adjustment_rate_satisfies_supermartingale_inequality():
    increments = np.array([-100,50,300,500], dtype=float)
    rate = _adjustment_rate(increments)
    assert rate > 0
    assert np.mean(np.exp(-rate*increments)) <= 1+1e-14


def test_full_engine_bootstrap_reproducible_and_matches_independent_replay():
    source, policy = history(), DollarPolicy.constant(2000)
    settings = RuinConfig(paths=4, sessions=16, mean_block=3, seed=12, cycle_paths=100, cycle_steps=10)
    risk = RiskConfig(bankroll=210.2)
    actual = Engine().ruin(SPEC, source, policy, CONFIG, simulation=settings, risk=risk)
    assert actual == Engine().ruin(SPEC, source, policy, CONFIG, simulation=settings, risk=risk)
    seeds = np.random.SeedSequence(12).spawn(4)
    for i, seed in enumerate(seeds):
        indices = StationaryDayBootstrap(3).generate(12,16,1,int(seed.generate_state(1)[0]))[0]
        generated = _resampled_history(SPEC, source, indices)
        expected = Engine().backtest(SPEC, generated, policy, CONFIG)
        assert actual["risk"]["path_records"][i]["required_bankroll"] == cash_risk_path(expected).required_bankroll
        finite = Engine().backtest(SPEC, generated, policy, replace(CONFIG, initial_wallet=210.2))
        assert any(e.kind == "wallet_wait" for e in finite.events) == (cash_risk_path(expected).required_bankroll > 210.2)
    assert actual["horizon_curve"][-1]["probability"] == actual["risk"]["ruin_probability"]
    assert [p["ruined_paths"] for p in actual["horizon_curve"]] == sorted(p["ruined_paths"] for p in actual["horizon_curve"])
    assert actual["ultimate_full_engine"]["status"] == "not_identified"
    json.dumps(actual, allow_nan=False)


def test_resampling_preserves_whole_session_order_and_brackets():
    source = history()
    generated = _resampled_history(SPEC, source, [2,0,1,2])
    for trade, index in zip(generated.trades, [2,0,1,2]):
        original = source.trades[index]
        assert (trade.stop_loss,trade.take_profit,trade.won) == (original.stop_loss,original.take_profit,original.won)
        assert trade.exit_at-trade.entry_at == original.exit_at-original.entry_at
    assert len(generated.sessions) == 4


def test_cycle_extraction_excludes_open_and_delayed_receipts():
    from types import SimpleNamespace
    def event(kind, cash=0):
        return SimpleNamespace(kind=kind, cash=cash, attempt=1)
    events = [event("fee",-100), event("request"), event("live_handoff")]
    assert _settled_cycles(SimpleNamespace(events=events)) == ([],1)
    events.append(event("receipt",200))
    assert _settled_cycles(SimpleNamespace(events=events)) == ([CashCycle(100,100)],0)
    assert _settled_cycles(SimpleNamespace(events=[event("fee",-100)])) == ([],1)


@pytest.mark.parametrize("kwargs", [{"paths":1}, {"sessions":0}, {"mean_block":.5},
    {"seed":True}, {"cycle_paths":1}, {"cycle_steps":0}, {"tail_tolerance":0}])
def test_invalid_ruin_configuration(kwargs):
    with pytest.raises(ValueError):
        RuinConfig(**kwargs)


def test_invalid_cash_cycles():
    with pytest.raises(ValueError):
        CashCycle(-2,1)
    with pytest.raises(ValueError):
        CashCycle(float("inf"),0)
    with pytest.raises(ValueError, match="money range"):
        ultimate_cycle_ruin([CashCycle(1e30,0)], bankroll=1)


def test_zero_wallet_performance_does_not_hide_required_capital():
    r = Engine().ruin(SPEC, history(), DollarPolicy.constant(2000),
        replace(CONFIG, initial_wallet=0), simulation=RuinConfig(paths=2, sessions=4))
    assert r["risk"]["distributions"]["net_cash"]["mean"] == 0
    assert r["risk"]["required_bankroll"] >= 105.2
    assert r["risk"]["ruin_probability"] == 1
    assert r["risk"]["performance_wallets"] == [0]
