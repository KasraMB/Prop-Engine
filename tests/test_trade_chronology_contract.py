"""Retain observed clocks and reject known overlap in a sequential summary engine."""
from dataclasses import replace

import numpy as np
import pytest

from propfirm_engine import InvalidTradeDataError, preprocess, slice_days
from propfirm_engine.data import validate_trade_dataset_numeric


def row(entry, exit_, symbol="ES"):
    return {"timestamp": exit_, "entry_time": entry, "return": 1.0, "symbol": symbol}


def test_preprocessing_retains_aligned_sorted_absolute_clocks():
    ds = preprocess([
        row("2026-01-02T11:00:00-05:00", "2026-01-02T12:00:00-05:00"),
        row("2026-01-01T10:00:00-05:00", "2026-01-01T11:00:00-05:00"),
    ], session_timezone="America/New_York")
    np.testing.assert_array_equal(ds.exit_timestamps, np.array(
        ["2026-01-01T16:00:00", "2026-01-02T17:00:00"], dtype="datetime64[ns]"))
    np.testing.assert_array_equal(ds.entry_timestamps, np.array(
        ["2026-01-01T15:00:00", "2026-01-02T16:00:00"], dtype="datetime64[ns]"))


@pytest.mark.parametrize("symbol", ["ES", "NQ"])
@pytest.mark.parametrize("reverse", [False, True])
def test_known_overlap_rejected_even_across_symbols_and_input_order(symbol, reverse):
    rows = [
        row("2026-01-01T10:00:00", "2026-01-01T12:00:00"),
        row("2026-01-01T11:00:00", "2026-01-01T13:00:00", symbol),
    ]
    with pytest.raises(InvalidTradeDataError, match="overlap"):
        preprocess(rows[::-1] if reverse else rows)


def test_nested_intervals_rejected():
    with pytest.raises(InvalidTradeDataError, match="overlap"):
        preprocess([
            row("2026-01-01T09:00:00", "2026-01-01T13:00:00"),
            row("2026-01-01T10:00:00", "2026-01-01T11:00:00"),
        ])


def test_adjacent_intervals_are_allowed():
    ds = preprocess([
        row("2026-01-01T10:00:00", "2026-01-01T11:00:00"),
        row("2026-01-01T11:00:00", "2026-01-01T12:00:00"),
    ])
    validate_trade_dataset_numeric(ds)


def test_absent_entry_is_unknown_not_invented_as_exit():
    ds = preprocess([row(None, "2026-01-01T12:00:00")])
    assert np.isnat(ds.entry_timestamps).all()
    assert not np.isnat(ds.exit_timestamps).any()


def test_day_slice_keeps_corresponding_clocks():
    ds = preprocess([
        row("2026-01-01T10:00:00", "2026-01-01T11:00:00"),
        row("2026-01-02T10:00:00", "2026-01-02T11:00:00"),
        row("2026-01-03T10:00:00", "2026-01-03T11:00:00"),
    ])
    part = slice_days(ds, 1, 3)
    np.testing.assert_array_equal(part.exit_timestamps, ds.exit_timestamps[1:])
    np.testing.assert_array_equal(part.entry_timestamps, ds.entry_timestamps[1:])
    validate_trade_dataset_numeric(part)


@pytest.mark.parametrize("field", ["entry_timestamps", "exit_timestamps"])
@pytest.mark.parametrize("values", [
    np.array([1.0]), np.array(["2026-01-01"], dtype="datetime64[D]"),
    np.array([], dtype="datetime64[ns]"), np.array([["2026-01-01"]], dtype="datetime64[ns]"),
])
def test_manual_clock_layout_is_checked(field, values):
    ds = preprocess([row(None, "2026-01-01T12:00:00")])
    with pytest.raises(InvalidTradeDataError, match=field):
        validate_trade_dataset_numeric(replace(ds, **{field: values}))


def test_modified_entry_cannot_bypass_overlap_guard():
    ds = preprocess([
        row("2026-01-01T10:00:00", "2026-01-01T11:00:00"),
        row("2026-01-01T12:00:00", "2026-01-01T13:00:00"),
    ])
    ds.entry_timestamps[1] = np.datetime64("2026-01-01T10:30:00")
    with pytest.raises(InvalidTradeDataError, match="overlap"):
        validate_trade_dataset_numeric(ds)


def test_missing_exit_clock_cannot_validate_supplied_entries():
    ds = preprocess([row("2026-01-01T10:00:00", "2026-01-01T11:00:00")])
    with pytest.raises(InvalidTradeDataError, match="exit_timestamps"):
        validate_trade_dataset_numeric(replace(ds, exit_timestamps=None))


def test_legacy_manual_dataset_without_clocks_remains_explicitly_unobserved():
    ds = preprocess([row(None, "2026-01-01T12:00:00")])
    validate_trade_dataset_numeric(replace(ds, exit_timestamps=None, entry_timestamps=None,
                                          cadence_source="supplied"))


@pytest.mark.parametrize("exits", [
    ["NaT", "2026-01-01T13:00:00"],
    ["2026-01-01T13:00:00", "2026-01-01T11:00:00"],
])
def test_mutated_exit_clock_must_be_present_and_ordered(exits):
    ds = preprocess([row(None, "2026-01-01T11:00:00"), row(None, "2026-01-01T13:00:00")])
    with pytest.raises(InvalidTradeDataError, match="exit_timestamps"):
        validate_trade_dataset_numeric(replace(ds, exit_timestamps=np.array(exits, dtype="datetime64[ns]")))


def test_engine_rejects_mutated_overlap_before_resampling():
    from propfirm_engine import Account, Phase, ProfitTargetRule, Engine, RunConfig

    class MustNotSample:
        def generate(self, *args):
            pytest.fail("invalid chronology reached resampling")

    ds = preprocess([
        row("2026-01-01T10:00:00", "2026-01-01T11:00:00"),
        row("2026-01-01T12:00:00", "2026-01-01T13:00:00"),
    ])
    ds.entry_timestamps[1] = np.datetime64("2026-01-01T10:30:00")
    account = Account("synthetic", 1000, (Phase("eval", "eval", (ProfitTargetRule(10),)),))
    with pytest.raises(InvalidTradeDataError, match="overlap"):
        Engine().run(account, ds, RunConfig(n_paths=1, resampler=MustNotSample()))
