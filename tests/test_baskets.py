from datetime import timedelta

import pytest

from propfirm_engine import Basket, Cancel, Engine, Instrument, Market, Order, Quote, QuoteModel
from test_strategy import AT, DAY, Script
from test_strategy_fitting import SPEC, CONFIG, setup


X, Y = Instrument("X", 1, 1), Instrument("Y", 1, 1)


def market(index, x=10000, y=10000, y_size=None):
    return Market(AT+timedelta(minutes=index), (Quote("X", x, x, x), Quote("Y", y, y, y, y_size, y_size)))


def basket(id, side=1, tif="day"):
    return Basket(id, (Order(id+"x", "X", side, reduce_only=side < 0, tif=tif),
                       Order(id+"y", "Y", side, reduce_only=side < 0, tif=tif)))


def run(markets, script, **kwargs):
    options = setup(0) | dict(models={"X": QuoteModel(), "Y": QuoteModel()}, sessions=(DAY,)) | kwargs
    return Engine().replay_strategy(SPEC, markets, [X, Y], CONFIG, Script(script), **options)


def test_atomic_basket_waits_for_every_leg_without_partial_fills():
    result = run([market(0), market(1, y_size=0), market(2), market(3, 10010, 10020)],
                  {0: [basket("entry")], 2: [basket("exit", -1)]})
    fills = [e.fill for e in result.orders if e.fill is not None]
    assert len(fills) == 4
    assert all(f.at == AT+timedelta(minutes=2) for f in fills[:2])
    assert result.result.book.balance == 50030


def test_atomic_net_execution_avoids_a_false_intermediate_leg_breach():
    last = Market(AT+timedelta(minutes=2),
                   (Quote("X", 7000, 10000, 10000), Quote("Y", 13000, 13000, 10000)))
    result = run([market(0), market(1), last], {0: [basket("entry")], 1: [basket("exit", -1)]},
                  models={"X": QuoteModel(fee=1), "Y": QuoteModel(fee=1)})
    assert result.result.replay.failed_attempts == 0
    assert result.result.book.balance == 49996
    assert result.result.fills == 4


def test_basket_net_loss_breaches_once_and_cannot_revive_on_later_quotes():
    result = run([market(0), market(1), market(2, 9000, 9000), market(3, 12000, 12000)],
                 {0: [basket("entry")]})
    assert result.result.replay.failed_attempts == 1
    assert result.result.book is None


def test_cancelling_one_leg_cancels_the_whole_basket():
    result = run([market(0), market(1, y_size=0), market(2)],
                 {0: [basket("entry")], 1: [Cancel("entryx")]})
    assert not any(e.fill for e in result.orders)
    assert {e.id for e in result.orders if e.status == "cancelled"} == {"entry", "entryx", "entryy"}
    assert not any(e.status == "rejected" for e in result.orders)


def test_ioc_basket_does_not_partially_execute():
    result = run([market(0), market(1, y_size=0)], {0: [basket("entry", tif="ioc")]})
    assert result.result.fills == 0
    assert dict(result.order_counts)["cancelled"] == 3


def test_invalid_basket_leg_does_not_buy_an_account():
    oversized = Basket("entry", (Order("x", "X", 3), Order("y", "Y", 3)))
    result = run([market(0), market(1)], {0: [oversized]})
    assert result.result.replay.attempts == 0
    assert result.result.fills == 0


def test_basket_recording_parity_keeps_economics_and_callbacks():
    first = run([market(0), market(1), market(2, 10010, 10020)],
                {0: [basket("entry")], 1: [basket("exit", -1)]})
    compact = run([market(0), market(1), market(2, 10010, 10020)],
                  {0: [basket("entry")], 1: [basket("exit", -1)]}, recording="search")
    assert compact.result.book == first.result.book
    assert compact.order_counts == first.order_counts
    assert compact.result.event_counts == first.result.event_counts


def test_unsupported_atomic_relationships_are_explicit_errors():
    with pytest.raises(ValueError, match="market/limit"):
        Basket("a", (Order("x", "X", 1, "stop", stop=10010), Order("y", "Y", 1)))
    with pytest.raises(ValueError, match="distinct"):
        Basket("a", (Order("x", "X", 1), Order("y", "X", 1)))
