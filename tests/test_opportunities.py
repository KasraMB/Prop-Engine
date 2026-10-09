from datetime import timedelta

import pytest

from propfirm_engine import Engine, Opportunity, OpportunityStrategy, Order
from test_strategy import AT, DAY, X, market
from test_strategy_fitting import SPEC, CONFIG, setup


class Policy:
    def __init__(self, quantity=1):
        self.quantity = quantity
        self.seen = []
        self.updates = []

    def on_opportunity(self, context, opportunity, market):
        self.seen.append((context, opportunity, market))
        if not context.warmup and context.available:
            return [Order(opportunity.tag, opportunity.symbol, self.quantity*opportunity.side)]

    def on_order(self, context, event):
        self.updates.append(event)


def test_none_opportunity_is_not_silent_end_of_data():
    with pytest.raises(TypeError, match="expected Opportunity"):
        OpportunityStrategy([None], Policy()).take()


def test_opportunities_are_released_causally_and_sizing_is_a_policy_choice():
    signals = [Opportunity(AT, "X", 1, tag="entry"),
               Opportunity(AT+timedelta(minutes=1), "X", -1, tag="exit")]
    policy = Policy(quantity=2)
    result = Engine().replay_opportunities(SPEC, [market(0), market(1), market(2, 10010)], [X],
        CONFIG, signals, policy, sessions=(DAY,), **setup(0))
    assert result.offered == 2 and result.expired == 0
    assert result.replay.result.book.balance == 50020
    assert [s[2].at for s in policy.seen] == [AT, AT+timedelta(minutes=1)]
    assert any(e.status == "filled" for e in policy.updates)


def test_expired_signals_are_not_traded_after_a_data_gap():
    signals = [Opportunity(AT, "X", 1, expires=AT+timedelta(minutes=1))]
    result = Engine().replay_opportunities(SPEC, [market(2)], [X], CONFIG, signals, Policy(),
        sessions=(DAY,), **setup(0))
    assert result.offered == 0 and result.expired == 1
    assert result.replay.result.replay.attempts == 0


def test_opportunity_adapter_rejects_broken_order_without_sorting_it():
    signals = [Opportunity(AT, "X", 1, tag="a"), Opportunity(AT, "X", -1, tag="b")]
    with pytest.raises(ValueError, match="strictly increase"):
        Engine().replay_opportunities(SPEC, [market(0)], [X], CONFIG, signals, Policy(),
            sessions=(DAY,), **setup(0))


def test_opportunities_can_be_used_by_the_same_strategy_fitter_factory():
    from propfirm_engine import Parameter
    from test_strategy_fitting import tape, ending_balance
    data = tape()
    signals = [Opportunity(m.at, "X", 1 if m.at.minute == 0 else -1, tag=str(m.at))
               for m in data if m.at.minute in (0, 1)]

    def factory(params, seed):
        return OpportunityStrategy(signals, Policy(params["quantity"]))

    result = Engine().fit_strategy(SPEC, data, CONFIG, factory,
        baseline={"quantity": 1}, space={"quantity": Parameter("integer", 1, 3)},
        setup=setup, generations=1, population=4, objective=ending_balance)
    assert result.params["quantity"] == 3
    assert result.score == 50090


def test_pre_horizon_signals_are_skipped_but_current_session_gap_signals_are_delivered():
    signals = [Opportunity(AT-timedelta(days=1), "X", 1, tag="old"),
               Opportunity(AT-timedelta(minutes=1), "X", 1, tag="current")]
    policy = Policy()
    result = Engine().replay_opportunities(SPEC, [market(0), market(1)], [X], CONFIG, signals, policy,
        sessions=(DAY,), **setup(0))
    assert result.before_start == 1 and result.offered == 1
    assert policy.seen[0][1].tag == "current"
