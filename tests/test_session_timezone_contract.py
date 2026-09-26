"""Session boundaries are local; chronological ordering is by actual instant."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from propfirm_engine.cache import TradeCache
from propfirm_engine.data import InvalidTradeDataError, preprocess


def rows(times):
    return [{"timestamp": time, "return": float(i + 1)} for i, time in enumerate(times)]


@pytest.mark.parametrize("value", ["2026-01-05T22:30:00Z", datetime(2026, 1, 5, 22, 30, tzinfo=timezone.utc)])
def test_aware_input_requires_explicit_session_timezone(value):
    with pytest.raises(InvalidTradeDataError, match="session_timezone"):
        preprocess(rows([value]))


@pytest.mark.parametrize("times", [
    ["2026-01-05T22:59:00Z", "2026-01-05T23:00:00Z"],  # CST
    ["2026-07-06T21:59:00Z", "2026-07-06T22:00:00Z"],  # CDT
])
def test_chicago_session_cutoff_follows_dst(times):
    ds = preprocess(rows(times), session_timezone="America/Chicago")
    assert ds.day.tolist() == [0, 1]
    assert ds.session_timezone == "America/Chicago"


def test_timezone_changes_local_session_partition():
    data = rows(["2026-01-05T16:59:00Z", "2026-01-05T17:00:00Z"])
    assert preprocess(data, session_timezone="UTC").n_days == 2
    assert preprocess(data, session_timezone="America/Chicago").n_days == 1


def test_fall_back_keeps_utc_order_not_wall_clock_order():
    data = rows(["2026-11-01T01:15:00-06:00", "2026-11-01T01:45:00-05:00"])
    ds = preprocess(data, session_timezone="America/Chicago")
    assert ds.ret.tolist() == [2.0, 1.0]
    assert ds.day.tolist() == [0, 0]


def test_mixed_aware_naive_input_rejected():
    with pytest.raises(InvalidTradeDataError, match="aware|naive"):
        preprocess(rows(["2026-01-05T12:00Z", "2026-01-05T13:00"]), session_timezone="UTC")


def test_naive_times_not_silently_localized_through_dst_gap():
    with pytest.raises(InvalidTradeDataError, match="aware"):
        preprocess(rows(["2026-03-08T02:30"]), session_timezone="America/Chicago")


def test_invalid_timezone_is_actionable():
    with pytest.raises(InvalidTradeDataError, match="session_timezone"):
        preprocess(rows(["2026-01-05T12:00Z"]), session_timezone="Unknown/Zone")


@pytest.mark.parametrize("value", ["1500-01-01", "3000-01-01", np.datetime64("3000-01-01")])
def test_datetime64_range_overflow_rejected(value):
    with pytest.raises(InvalidTradeDataError, match="range"):
        preprocess(rows([value]))


def test_cache_key_includes_session_timezone():
    cache = TradeCache()
    data = rows(["2026-01-05T16:59:00Z", "2026-01-05T17:00:00Z"])
    utc = cache.get(data, session_timezone="UTC")
    chicago = cache.get(data, session_timezone="America/Chicago")
    assert utc.n_days == 2 and chicago.n_days == 1
    assert cache.misses == 2
    assert cache.get(data, session_timezone="UTC") is utc


def test_aware_nanosecond_strings_keep_exact_chronological_order():
    ds = preprocess(rows(["2026-01-05T12:00:00.000000002Z", "2026-01-05T12:00:00.000000001Z"]),
                    session_timezone="UTC")
    assert ds.ret.tolist() == [2.0, 1.0]


def test_dst_clipping_compares_absolute_instants():
    data = [{"timestamp": "2026-11-01T01:15:00-06:00", "return": 1.0,
             "entry_time": "2026-11-01T01:45:00-05:00",
             "mae_bars": [("2026-11-01T01:50:00-05:00", 2.0),
                          ("2026-11-01T01:30:00-06:00", 999.0)]}]
    assert preprocess(data, session_timezone="America/Chicago").trade_low.tolist() == [-2.0]


def test_ambiguous_session_reset_rejected_instead_of_nonmonotonic_days():
    with pytest.raises(InvalidTradeDataError, match="ambiguous DST"):
        preprocess(rows(["2026-11-01T01:45:00-05:00", "2026-11-01T01:15:00-06:00"]),
                   session_timezone="America/Chicago", session_reset="01:30")


def test_nonexistent_zoned_wall_time_rejected():
    value = datetime(2026, 3, 8, 2, 30, tzinfo=ZoneInfo("America/Chicago"))
    with pytest.raises(InvalidTradeDataError, match="nonexistent"):
        preprocess(rows([value]), session_timezone="America/Chicago")


def test_engine_threads_timezone_for_raw_input():
    from propfirm_engine.engine import Engine, RunConfig
    from propfirm_engine.model import Account, Phase
    from propfirm_engine.rules import ProfitTargetRule
    account = Account("fixture", 1000, (Phase("eval", "eval", (ProfitTargetRule(1.0),)),))
    result = Engine().run(account, rows(["2026-01-05T12:00Z"]),
                          RunConfig(n_paths=2, L_eval=2, session_timezone="UTC"))
    assert result.n_attempts == 2
