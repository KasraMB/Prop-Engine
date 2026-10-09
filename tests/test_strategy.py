from datetime import date, datetime, timedelta, timezone
from fractions import Fraction

import pytest

from propfirm_engine import (
    Amend, BacktestConfig, Cancel, Engine, Instrument, Market, Order, Quote, QuoteModel,
)
from propfirm_engine.firms.lucidflex import replay_50k


AT = datetime(2026, 9, 1, 14, tzinfo=timezone.utc)
DAY = date(2026, 9, 1)
X = Instrument("X", 1, 1)


def market(i, price=10_000, *, bid=None, ask=None, size=None):
    return Market(AT + timedelta(minutes=i),
                  (Quote("X", price if bid is None else bid, price if ask is None else ask,
                         price, size, size),))


class Script:
    def __init__(self, actions):
        self.actions = actions
        self.contexts, self.updates = [], []

    def on_market(self, context, market):
        self.contexts.append(context)
        return self.actions.get(len(self.contexts)-1, ())

    def on_order(self, context, event):
        self.updates.append(event)


def run(markets, strategy, **kwargs):
    settings = dict(sessions=(DAY,), fidelity="observed_marks", max_mark_age=timedelta(minutes=10),
                    liquidation_fee=0, models={"X": QuoteModel()}, trace=True)
    settings.update(kwargs)
    profile = settings.pop("spec", replay_50k(eval_fee=105.2, reset_fee=105, contract_type="mini"))
    config = settings.pop("config", BacktestConfig(0, timedelta(0), timedelta(0), timedelta(0)))
    return Engine().replay_strategy(profile, iter(markets), [X], config, strategy, **settings)


def fills(result):
    return [e.fill for e in result.orders if e.fill is not None]


def test_market_orders_execute_on_next_observation():
    strategy = Script({0: [Order("a", "X", 1)], 1: [Order("b", "X", -1, reduce_only=True)]})
    result = run([market(0), market(1, 10_005), market(2, 10_028)], strategy)
    entry, exit = fills(result)
    assert entry.at == market(1).at and entry.price == 10_005
    assert exit.price == 10_028 and result.result.book.balance == 50_023
    assert strategy.contexts[0].book is None
    assert strategy.contexts[1].book.positions[0].quantity == 1


def test_limit_partial_fills_share_liquidity_and_fee_once_per_order():
    strategy = Script({0: [Order("a", "X", 2, "limit", limit=10_000)],
                       3: [Order("b", "X", -2, reduce_only=True)]})
    result = run([market(0), market(1, size=1), market(2, size=1), market(3), market(4, 10_100)],
                 strategy, models={"X": QuoteModel(fee=1, fixed_fee=2)})
    assert [f.quantity for f in fills(result)] == [1, 1, -2]
    assert [f.fee for f in fills(result)] == [3, 1, 4]
    assert result.result.book.balance == 50_192


def test_stop_limit_stays_triggered_after_gap_until_executable():
    strategy = Script({0: [Order("a", "X", 1, "stop_limit", stop=10_010, limit=10_015)],
                       3: [Order("b", "X", -1, reduce_only=True)]})
    result = run([market(0), market(1, 10_030), market(2, 10_025),
                  market(3, 10_012), market(4, 10_020)], strategy)
    assert fills(result)[0].at == market(3).at
    assert result.result.book.balance == 50_008


def test_stop_market_gaps_and_slippage_are_not_stop_price_fills():
    strategy = Script({0: [Order("a", "X", 1, "stop", stop=10_010)],
                       1: [Order("b", "X", -1, reduce_only=True)]})
    result = run([market(0), market(1, 10_030), market(2, 10_040)], strategy,
                 models={"X": QuoteModel(slip_ticks=2)})
    assert [f.price for f in fills(result)] == [10_032, 10_038]
    assert result.result.book.balance == 50_006


def test_linked_exits_and_oco_cannot_reverse_after_target():
    strategy = Script({0: [Order("a", "X", 1),
        Order("tp", "X", -1, "limit", limit=10_100, reduce_only=True, parent="a", oco="exit"),
        Order("sl", "X", -1, "stop", stop=9900, reduce_only=True, parent="a", oco="exit")]})
    result = run([market(0), market(1), market(2, 10_100), market(3, 9800)], strategy)
    assert len(fills(result)) == 2 and result.result.book.balance == 50_100
    assert any(e.id == "sl" and e.reason == "oco" for e in result.orders)


def test_amend_and_cancel_use_only_later_quotes():
    strategy = Script({0: [Order("a", "X", 1, "limit", limit=9900)],
                       1: [Amend(Order("a", "X", 1, "limit", limit=10_000))],
                       2: [Order("b", "X", -1), Order("c", "X", 1, "limit", limit=9900)],
                       3: [Cancel("c")]})
    result = run([market(0), market(1), market(2), market(3, 10_020)], strategy)
    assert len(fills(result)) == 2 and fills(result)[0].at == market(2).at
    assert result.result.book.balance == 50_020


def test_trailing_exit_tracks_only_observed_favorable_marks():
    strategy = Script({0: [Order("a", "X", 1)],
                       1: [Order("trail", "X", -1, "trailing", trail=20, reduce_only=True)]})
    result = run([market(0), market(1), market(2, 10_040), market(3, 10_030),
                  market(4, 10_010)], strategy)
    assert fills(result)[-1].price == 10_010
    assert result.result.book.balance == 50_010


def test_trailing_exit_anchors_at_submission_before_the_next_gap():
    strategy = Script({0: [Order("a", "X", 1)],
                       1: [Order("trail", "X", -1, "trailing", trail=20, reduce_only=True)]})
    result = run([market(0), market(1), market(2, 9970)], strategy)
    assert fills(result)[-1].price == 9970
    assert result.result.book.balance == 49_970


def test_pending_exposure_is_reserved_and_rejected_order_is_reported():
    strategy = Script({0: [Order("a", "X", 3, "limit", limit=9900),
                           Order("b", "X", 2, "limit", limit=9900)]})
    result = run([market(0), market(1)], strategy)
    assert not fills(result)
    assert any(e.id == "b" and e.reason == "contract_limit" for e in strategy.updates)


def test_breach_precedes_protective_order_matching_and_cancels_orders():
    strategy = Script({0: [Order("a", "X", 1)],
                       1: [Order("sl", "X", -1, "stop", stop=8000, reduce_only=True),
                           Order("add", "X", 1, "limit", limit=8000)]})
    result = run([market(0), market(1), market(2, 8000), market(3, 11_000)], strategy)
    assert len(fills(result)) == 1
    assert result.result.replay.failed_attempts == 1
    assert {e.id for e in result.orders if e.reason == "account_ended"} == {"sl", "add"}


def test_ioc_cancels_unfilled_remainder():
    strategy = Script({0: [Order("a", "X", 2, tif="ioc")],
                       1: [Order("b", "X", -1, reduce_only=True)]})
    result = run([market(0), market(1, size=1), market(2)], strategy)
    assert [f.quantity for f in fills(result)] == [1, -1]
    assert any(e.id == "a" and e.reason == "ioc_remainder" for e in result.orders)


def test_callback_can_react_to_rejection_without_future_data():
    class React(Script):
        def on_order(self, context, event):
            super().on_order(context, event)
            if event.status == "rejected":
                return [Order("small", "X", 1)]
            if event.id == "small" and event.fill:
                return [Order("exit", "X", -1, reduce_only=True)]
    strategy = React({0: [Order("large", "X", 5)]})
    result = run([market(0), market(1), market(2, 10_010), market(3, 10_030)], strategy)
    assert [f.price for f in fills(result)] == [10_010, 10_030]
    assert result.result.book.balance == 50_020


def test_warmup_cannot_purchase_an_account_or_fill_orders():
    strategy = Script({0: [Order("warm", "X", 1)]})
    result = run([market(0)], strategy, warmup=[market(-24*60)])
    assert strategy.contexts[0].warmup and not strategy.contexts[0].available
    assert result.result.replay.fees == 0 and not fills(result)


def test_partial_exit_cannot_reverse_even_with_excess_order_quantity():
    strategy = Script({0: [Order("a", "X", 1)],
                       1: [Order("b", "X", -3, reduce_only=True)]})
    result = run([market(0), market(1), market(2, 10_123)], strategy)
    assert [f.quantity for f in fills(result)] == [1, -1]
    assert result.result.book.balance == 50_123 and not result.result.book.positions


def test_cancelled_unfilled_parent_cancels_linked_child():
    strategy = Script({0: [Order("a", "X", 1, "limit", limit=9900),
                           Order("b", "X", -1, "limit", limit=10_100, parent="a", reduce_only=True)],
                       1: [Cancel("a")]})
    result = run([market(0), market(1), market(2, 10_100)], strategy)
    assert not fills(result)
    assert any(e.id == "b" and e.reason == "parent_cancelled" for e in result.orders)


def test_shared_quote_liquidity_cannot_be_reused_by_two_orders():
    strategy = Script({0: [Order("a", "X", 1), Order("b", "X", 1)],
                       2: [Order("exit", "X", -2, reduce_only=True)]})
    result = run([market(0), market(1, size=1), market(2, size=1), market(3)], strategy)
    assert [f.at for f in fills(result)[:2]] == [market(1).at, market(2).at]


def test_order_expiry_precedes_fill_at_identical_timestamp():
    strategy = Script({0: [Order("a", "X", 1, expires=market(1).at)]})
    result = run([market(0), market(1)], strategy)
    assert not fills(result)
    assert any(e.reason == "expired" for e in result.orders)


def test_trace_mode_and_future_quotes_do_not_change_past_execution():
    actions = {0: [Order("a", "X", 1)], 1: [Order("b", "X", -1)]}
    short = run([market(0), market(1), market(2, 10_030)], Script(actions))
    quiet = run([market(0), market(1), market(2, 10_030)], Script(actions), trace=False)
    longer = run([market(0), market(1), market(2, 10_030), market(3, 15_000)], Script(actions))
    assert short.result.replay == quiet.result.replay
    assert short.orders == quiet.orders == longer.orders


@pytest.mark.parametrize("kwargs", [dict(quantity=0), dict(kind="bad"), dict(kind="limit"),
    dict(kind="market", stop=1), dict(kind="trailing", trail=0), dict(parent="a"),
    dict(reduce_only=1), dict(tif="bad"), dict(expires=AT.replace(tzinfo=None))])
def test_order_validation(kwargs):
    with pytest.raises(ValueError):
        Order(**(dict(id="a", symbol="X", quantity=1) | kwargs))
