"""Direction contracts from price fills through policy search and account cash."""
from dataclasses import replace
from datetime import timedelta
from fractions import Fraction as F

import numpy as np
import pytest

from propfirm_engine import Engine, Fill, Instrument, Opportunity, Order
from propfirm_engine.market_replay import _resolve
from propfirm_engine.slippage import prepare_execution
from test_market_replay import CONFIG, POLICY, SPEC, TARGETS, session
from test_slippage import model


def mirror(s):
    return replace(s, side=-s.side, ohlc=200-s.ohlc[:, [0, 2, 1, 3]])


@pytest.mark.parametrize("rows", [
    [[100, 111, 89, 105]],
    [[100, 101, 99, 100], [112, 113, 80, 90]],
    [[100, 101, 99, 100], [112, 113, 111, 112]],
    [[100, 101, 99, 100], [85, 110, 84, 100]],
    [[100, 103, 98, 102]],
    [[100, 103, 98, 99]],
    [[100, 111, 99, 110], [110, 112, 0, 1]],
    [[100, 103, 97, 101], [101, 112, 100, 111]],
])
@pytest.mark.parametrize("scenario", [None, model(market=1, stop=2), model(target_trade_through_ticks=2)])
def test_long_short_mirror_preserves_all_economics(rows, scenario):
    long = session(rows)
    short = mirror(long)
    instrument = Instrument("X", 100, .25)
    results = []
    for s in (long, short):
        tape = prepare_execution([s], instrument, scenario)[0] if scenario else None
        results.append(_resolve(s, instrument, 1, F(1000), F(1000), F(4),
                                slippage=scenario, tape=tape))
    a, b = results
    assert a[0] == b[0]
    assert a[5:] == b[5:]
    assert np.array(a[1:5])+np.array(b[1:5]) == pytest.approx([200]*4)


@pytest.mark.parametrize("rows,reason,exit_price,pnl", [
    ([[100, 102, 89, 95]], "target", 90, 1000),
    ([[100, 111, 89, 100]], "stop", 110, -1000),
    ([[100, 101, 99, 100], [115.13, 116, 114, 115]], "stop_gap", 115.25, -1525),
    ([[100, 103, 98, 99]], "session_close", 99, 100),
    ([[100, 103, 98, 101]], "session_close", 101, -100),
])
def test_short_fills_match_hand_calculated_prices(rows, reason, exit_price, pnl):
    s = replace(session(rows), side=-1)
    result = _resolve(s, Instrument("X", 100, .25), 1, F(1000), F(1000), F(0))
    assert result[1:4] == (100, 110, 90)
    assert result[4] == exit_price
    assert result[0].pnl == pnl
    assert result[-2] == reason


def test_short_entry_exit_rounding_costs_and_adverse_slippage():
    s = replace(session([[100.13, 100.25, 98.5, 99]]), side=-1)
    result = _resolve(s, Instrument("X", 100, .25), 1, F(100), F(100), F(4))
    assert result[1:5] == (100, 100.75, 98.75, 98.75)
    assert result[7] == 121
    scenario = model(market=1, stop=2)
    s = replace(session([[100, 120, 95, 110]]), side=-1)
    instrument = Instrument("X", 10, 1)
    result = _resolve(s, instrument, 1, F(100), F(50), F(4), slippage=scenario,
                      tape=prepare_execution([s], instrument, scenario)[0])
    assert result[1:5] == (99, 106, 93, 108)
    assert result[7] == -94


def test_short_does_not_include_post_exit_highs_in_drawdown():
    s = replace(session([[100, 109, 89, 90], [90, 500, 80, 200]]), side=-1)
    result = _resolve(s, Instrument("X", 100, .25), 1, F(1000), F(1000), F(0))
    assert result[0].low == 0
    assert result[0].pnl == 1000
    assert result[0].exit_at == s.entry_at + timedelta(minutes=1)


def test_mixed_direction_history_matches_long_account_and_payouts():
    from datetime import date
    from propfirm_engine import DollarPolicy, RiskRegime
    policy = DollarPolicy((RiskRegime("eval", "eval", 2000),
        RiskRegime("build", "funded", 1000, days_to_payout=5, after_payout=False),
        RiskRegime("fallback", "funded", 1500)))
    prices = [session([[100, 150, 100, 120]], day=date(2025, 1, d)) for d in (6, 7, 8, 9, 10, 13, 14)]
    mixed = [mirror(s) if i % 2 else s for i, s in enumerate(prices)]
    def replay(sessions):
        return Engine().backtest_prices(SPEC, sessions, policy, {"eval": 1500, "build": 3000, "fallback": 500},
            Instrument("X", 100, .01), CONFIG, collision_policy="stop_first")
    a, b = replay(prices), replay(mixed)
    assert a.replay.events == b.replay.events
    assert a.replay.net_cash == b.replay.net_cash == 1694.8
    assert a.replay.receipts == b.replay.receipts == 1800
    assert [d.side for d in b.decisions] == [1, -1, 1, -1, 1, -1, 1]
    assert [d.signed_quantity for d in b.decisions] == [1, -1, 1, -1, 1, -1, 1]


def test_direction_changes_fingerprint_even_on_flat_prices():
    s = session([[100, 100, 100, 100]])
    def replay(s):
        return Engine().backtest_prices(SPEC, [s], POLICY, TARGETS, Instrument("X", 100, .25),
                                       CONFIG, collision_policy="stop_first")
    assert replay(s).replay.history_fingerprint != replay(replace(s, side=-1)).replay.history_fingerprint


@pytest.mark.parametrize("side", [0, 2, -2, True, 1., "short", None])
def test_invalid_side_rejected(side):
    with pytest.raises(ValueError, match="side"):
        replace(session([[100, 101, 99, 100]]), side=side)


def test_fitter_and_fixed_policy_evaluation_retain_each_session_direction():
    from test_price_fitting import history, fit, MODEL, SPEC as spec, CONFIG as config, POLICY as policy, INSTRUMENT
    prices = history()
    mixed = [mirror(s) if i % 2 else s for i, s in enumerate(prices)]
    kwargs = dict(generations=1, population=4, paths=2, slippage=MODEL, seed=17)
    a, b = fit(prices, **kwargs), fit(mixed, **kwargs)
    assert a.policy == b.policy
    assert a.in_sample.score == b.in_sample.score
    assert a.out_of_sample.score == b.out_of_sample.score
    evaluated = Engine().evaluate_prices(spec, mixed, policy, INSTRUMENT, config,
        slippage=MODEL, paths=2, collision_policy="stop_first")
    assert any(d.side == -1 for p in evaluated.paths for d in p.decisions)
    assert any(d.side == -1 for p in b.out_of_sample.paths for d in p.decisions)
    assert b.out_of_sample.distributions == a.out_of_sample.distributions


@pytest.mark.parametrize("kwargs", [{"quantity": 0}, {"quantity": -1}, {"execution_seed": -1},
                                   {"compensate_slippage": 1}])
def test_price_entry_points_share_validation_without_slippage(kwargs):
    from test_price_fitting import fit, INSTRUMENT, POLICY as policy
    prices = [session([[100, 101, 99, 100]])]
    with pytest.raises(ValueError):
        Engine().backtest_prices(SPEC, prices, POLICY, TARGETS, INSTRUMENT, CONFIG,
                                collision_policy="stop_first", **kwargs)
    with pytest.raises(ValueError):
        Engine().evaluate_prices(SPEC, prices, policy, INSTRUMENT, CONFIG,
                                collision_policy="stop_first", **kwargs)
    with pytest.raises(ValueError):
        fit(generations=0, **kwargs)


@pytest.mark.parametrize("side", [-1, 1])
def test_orders_signals_and_recorded_fills_have_identical_directional_economics(side):
    from datetime import timedelta
    from test_strategy import AT, DAY, X, Script, market, run
    from test_opportunities import Policy, SPEC as spec, CONFIG as config, setup
    markets = [market(0), market(1), market(2, 10000+side*10)]
    strategy = run(markets, Script({0: [Order("in", "X", side)],
                                   1: [Order("out", "X", -side)]}), spec=spec, config=config)
    signals = Engine().replay_opportunities(spec, markets, [X], config,
        [Opportunity(AT, "X", side, tag="in"), Opportunity(AT+timedelta(minutes=1), "X", -side, tag="out")],
        Policy(), sessions=(DAY,), **setup(0))
    recorded = Engine().replay_events(spec,
        [Fill(AT+timedelta(minutes=1), "X", side, 10000),
         Fill(AT+timedelta(minutes=2), "X", -side, 10000+side*10)],
        [X], config, sessions=(DAY,), fidelity="observed_marks", mark_fills=True,
        max_mark_age=timedelta(minutes=10), liquidation_fee=0)
    assert strategy.result.book.balance == signals.replay.result.book.balance == recorded.book.balance == 50010
    assert signals.uncertainty == signals.replay.uncertainty


@pytest.mark.parametrize("adapter", ["prices", "brackets", "recorded", "strategy", "opportunities"])
def test_every_replay_adapter_supports_both_directions(adapter):
    report = Engine().check_replay(SPEC, adapter=adapter, features=("long", "short", "mixed_directions"))
    assert {"long", "short", "mixed_directions"} <= set(report.features)


def test_price_preflight_does_not_promise_historical_order_or_variable_quantity():
    from propfirm_engine import UnsupportedInputCapabilityError
    for feature in ("exact_intrabar", "sizing", "partial_exits"):
        with pytest.raises(UnsupportedInputCapabilityError):
            Engine().check_replay(SPEC, adapter="prices", features=(feature,))
    with pytest.raises(UnsupportedInputCapabilityError):
        Engine().check_replay(SPEC, adapter="prices", fidelity="observed_marks")


@pytest.mark.parametrize("side", [-1, 1])
def test_price_and_recorded_fill_accounting_agree(side):
    s = replace(session([[100, 103, 98, 102]]), side=side)
    instrument = Instrument("X", 100, .25)
    priced = Engine().backtest_prices(SPEC, [s], POLICY, TARGETS, instrument,
        replace(CONFIG, cost_per_contract=4), collision_policy="stop_first")
    decision, = priced.decisions
    recorded = Engine().replay_events(SPEC,
        [Fill(decision.entry_at, "X", side, decision.entry_price, 2),
         Fill(decision.exit_at, "X", -side, decision.exit_price, 2)], [instrument], CONFIG,
        sessions=(s.session,), fidelity="observed_marks", mark_fills=True,
        max_mark_age=timedelta(minutes=10), liquidation_fee=0)
    assert priced.replay.final_balance == recorded.replay.final_balance == 50000+side*200-4
    assert priced.replay.net_cash == recorded.replay.net_cash == -105.2
