from datetime import timedelta

import pytest

from propfirm_engine import Bar, TradeTick, Market, Quote, MarketTape, Order, bar_quotes, trade_quotes, merge_markets
from test_strategy import AT, DAY, X, Script, run


def bar(**kwargs):
    return Bar(**(dict(at=AT, end=AT+timedelta(minutes=1), symbol="X", open=10000,
                      high=10020, low=9980, close=10010) | kwargs))


@pytest.mark.parametrize("path,prices", [("ohlc", [10000, 10020, 9980, 10010]),
                                        ("olhc", [10000, 9980, 10020, 10010])])
def test_bar_paths_are_explicit_and_release_only_one_point_per_callback(path, prices):
    feed = bar_quotes([bar()], X, path=path, half_spread_ticks=1, liquidity=2)
    points = list(feed)
    assert [m.quotes[0].mark for m in points] == prices
    assert [m.at for m in points] == [AT+timedelta(seconds=20*i) for i in range(4)]
    assert all(m.quotes[0].ask-m.quotes[0].bid == 2 for m in points)
    assert feed.fidelity == "ohlc_path"


def test_approximation_requires_explicit_opt_in_and_survives_preparation():
    feed = bar_quotes([bar()], X, path="ohlc", half_spread_ticks=0, liquidity=None)
    tape = MarketTape(feed, [X], sessions=(DAY,))
    assert tape.view().fidelity == "ohlc_path"
    with pytest.raises(ValueError, match="ohlc_path"):
        run(tape, Script({}))
    from propfirm_engine import Engine
    from test_strategy_fitting import CONFIG, SPEC, setup
    strategy = Script({0: [Order("a", "X", 1)], 2: [Order("b", "X", -1, reduce_only=True)]})
    result = Engine().replay_strategy(SPEC, tape, [X], CONFIG, strategy,
        sessions=(DAY,), **(setup(0) | {"fidelity": "ohlc_path"}))
    assert result.result.book.balance == 49990
    assert any("OHLC approximation" in a for a in result.result.replay.assumptions)


def test_trade_prints_use_explicit_synthetic_spread_and_liquidity():
    feed = trade_quotes([TradeTick(AT, "X", 100)], [X], half_spread_ticks=2, liquidity=None)
    point, = feed
    assert point.quotes == (Quote("X", 98, 102, 100),)
    assert feed.fidelity == "last_trade"


def test_merge_keeps_asset_quotes_atomic_at_declared_ties():
    a = [Market(AT, (Quote("X", 100, 100, 100),))]
    b = [Market(AT, (Quote("Y", 200, 200, 200),))]
    point, = merge_markets(a, b, ties="atomic")
    assert {q.symbol for q in point.quotes} == {"X", "Y"}
    with pytest.raises(ValueError, match="collide"):
        list(merge_markets(a, b, ties="reject"))
    with pytest.raises(ValueError, match="distinct"):
        list(merge_markets(a, a, ties="atomic"))


def test_merge_never_sorts_a_broken_individual_feed():
    a = [Market(AT+timedelta(seconds=i), (Quote("X", 100, 100, 100),)) for i in (1, 0)]
    with pytest.raises(ValueError, match="strictly increase"):
        list(merge_markets(a, ties="reject"))


def test_adjoining_bars_require_ordered_boundary_sequences():
    second = bar(at=AT+timedelta(minutes=1), end=AT+timedelta(minutes=2))
    with pytest.raises(ValueError, match="seq"):
        list(bar_quotes([bar(), second], X, path="ohlc", half_spread_ticks=0, liquidity=None))
    second = bar(at=second.at, end=second.end, seq=1)
    assert len(list(bar_quotes([bar(), second], X, path="ohlc", half_spread_ticks=0, liquidity=None))) == 8


def test_feed_fidelity_and_path_change_input_fingerprints():
    tapes = [MarketTape(bar_quotes([bar()], X, path=path, half_spread_ticks=0, liquidity=None),
                         [X], sessions=(DAY,)) for path in ("ohlc", "olhc")]
    assert tapes[0].fingerprint != tapes[1].fingerprint
