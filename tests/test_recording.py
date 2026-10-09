from collections import Counter

import pytest

from propfirm_engine import Engine, EventState, OrderEvent, cash_risk_path
from test_strategy import DAY, X, market, Script, run
from test_strategy_fitting import CONFIG, SPEC, Daily, setup, tape


def test_search_research_and_trace_keep_identical_economics_and_counts():
    data = tape()
    results = [Engine().replay_strategy(SPEC, data, [X], CONFIG, Daily({"quantity": 1}, 0),
        sessions=data.sessions, recording=mode, **setup(0)) for mode in ("search", "research", "trace")]
    first = results[0]
    for result in results[1:]:
        assert result.result.book == first.result.book
        assert cash_risk_path(result.result.replay) == cash_risk_path(first.result.replay)
        assert result.result.event_counts == first.result.event_counts
        assert result.order_counts == first.order_counts
    assert not first.orders and not first.result.trace
    assert results[1].orders and not results[1].result.trace
    assert results[2].orders and results[2].result.trace
    assert len(first.result.replay.events) == 1
    assert len(results[1].result.replay.events) > 20


def test_streamed_order_state_and_lifecycle_events_match_retained_trace():
    data, streamed = tape(), []
    result = Engine().replay_strategy(SPEC, data, [X], CONFIG, Daily({"quantity": 1}, 0),
        sessions=data.sessions, recording="trace", sink=streamed.append, **setup(0))
    assert tuple(e for e in streamed if isinstance(e, EventState)) == result.result.trace
    assert tuple(e for e in streamed if isinstance(e, OrderEvent)) == result.orders
    assert Counter(e.status for e in result.orders) == dict(result.order_counts)


@pytest.mark.parametrize("mode", ["search", "research", "trace"])
def test_compaction_keeps_duplicate_checks_and_late_linked_exits(mode):
    from propfirm_engine import Order
    strategy = Script({0: [Order("a", "X", 1)],
        1: [Order("a", "X", 1), Order("b", "X", -1, reduce_only=True, parent="a")]})
    result = run([market(0), market(1), market(2, 10010)], strategy, trace=False, recording=mode)
    assert result.result.book.balance == 50010
    assert dict(result.order_counts)["rejected"] == 1
    assert any(e.reason == "duplicate_id" for e in strategy.updates)


def test_callback_chains_are_not_lost_when_delivered_events_are_discarded():
    from propfirm_engine import Order

    class Feedback(Script):
        def on_order(self, context, event):
            self.updates.append(event)
            if event.id == "a" and event.status == "filled":
                return [Order("b", "X", -1, reduce_only=True)]

    def execute(mode):
        strategy = Feedback({0: [Order("a", "X", 1)]})
        result = run([market(0), market(1), market(2, 10010)], strategy,
                     recording=mode, trace=False)
        return result, strategy.updates

    full, compact = execute("research"), execute("search")
    assert full[1] == compact[1]
    assert full[0].result.book == compact[0].result.book


def test_invalid_recording_combinations_rejected():
    with pytest.raises(ValueError, match="recording"):
        run([market(0)], Script({}), trace=True, recording="search")


def test_cached_replay_plan_is_bounded_and_cannot_be_mutated():
    from propfirm_engine.backtest import _check_support
    _check_support.cache_clear()
    first = _check_support(SPEC, observations=True)
    assert _check_support(SPEC, observations=True) is first
    assert _check_support.cache_info().maxsize == 128
    with pytest.raises(ValueError):
        first.phases[0].p0.flags.writeable = True


def test_search_cash_risk_matches_a_complete_funded_payout_path():
    from test_event_replay import run as event_run, winning_path, dates
    full = event_run(winning_path(), sessions=dates(7), recording="trace")
    compact = event_run(winning_path(), sessions=dates(7), recording="search", trace=False)
    assert cash_risk_path(full.replay) == cash_risk_path(compact.replay)
    assert compact.replay.receipts == 450
    assert compact.event_counts == full.event_counts
    assert compact.book == full.book


def test_legacy_caches_are_bounded_lru():
    from propfirm_engine import CompiledRuleCache
    from propfirm_engine.rules import ProfitTargetRule
    cache = CompiledRuleCache(max_entries=2)
    a, b, c = [ProfitTargetRule(n) for n in (100, 200, 300)]
    cache.get(a)
    cache.get(b)
    cache.get(a)
    cache.get(c)
    assert len(cache._store) == 2 and b not in cache._store
    before = cache.misses
    cache.get(b)
    assert cache.misses == before+1
