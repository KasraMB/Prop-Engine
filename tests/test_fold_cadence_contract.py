"""A derived training-fold cadence must not depend on held-out timestamps."""
from dataclasses import replace

import pytest

from propfirm_engine import InvalidTradeDataError, preprocess, slice_days
from propfirm_engine.data import validate_trade_dataset_numeric


def rows(*days):
    return [{"timestamp": day + "T12:00:00", "return": 1.0} for day in days]


def test_future_dates_cannot_change_training_cadence():
    a = preprocess(rows("2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"))
    b = preprocess(rows("2026-01-01", "2026-01-02", "2026-06-03", "2026-06-04"))
    train_a, train_b = slice_days(a, 0, 2), slice_days(b, 0, 2)
    assert train_a.trading_days_per_week == train_b.trading_days_per_week == 2
    assert train_a.cadence_source == train_b.cadence_source == "derived"


def test_explicit_cadence_stays_a_caller_assumption_in_each_fold():
    ds = preprocess(rows("2026-01-01", "2026-01-02", "2026-06-03"), trading_days_per_week=5)
    assert slice_days(ds, 0, 2).trading_days_per_week == 5
    assert slice_days(ds, 2, 3).trading_days_per_week == 5
    assert ds.cadence_source == "supplied"


def test_nested_slice_matches_direct_slice():
    ds = preprocess(rows("2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02"))
    nested = slice_days(slice_days(ds, 1, 4), 1, 3)
    direct = slice_days(ds, 2, 4)
    assert nested.trading_days_per_week == direct.trading_days_per_week == 2


def test_derived_cadence_requires_observed_exit_clocks():
    ds = preprocess(rows("2026-01-01"))
    with pytest.raises(InvalidTradeDataError, match="derived.*exit_timestamps"):
        validate_trade_dataset_numeric(replace(ds, exit_timestamps=None, entry_timestamps=None))


def test_invalid_cadence_source_rejected():
    ds = preprocess(rows("2026-01-01"))
    with pytest.raises(InvalidTradeDataError, match="cadence_source"):
        validate_trade_dataset_numeric(replace(ds, cadence_source="guessed"))
