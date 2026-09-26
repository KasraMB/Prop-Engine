"""Reject malformed indexing before passing arrays to unchecked compiled loops."""
from dataclasses import replace

import numpy as np
import pytest

from propfirm_engine.data import InvalidTradeDataError, preprocess, validate_trade_dataset_numeric


def dataset():
    return preprocess([
        {"timestamp": "2026-01-05T12:00", "return": 1.0},
        {"timestamp": "2026-01-05T13:00", "return": -1.0},
        {"timestamp": "2026-01-06T12:00", "return": 2.0},
    ])


@pytest.mark.parametrize("field,value", [
    ("n_days", 0), ("n_days", 1.5), ("n_days", True), ("n_days", 3),
    ("day", np.array([0, 1, 0])), ("day", np.array([0, 0, 3])),
    ("day", np.array([0.0, 0.0, 1.0])), ("day", np.array([0, 1])),
    ("day_first", np.array([1, 2])), ("day_first", np.array([0, 1])),
    ("day_first", np.array([0, 9])), ("day_first", np.array([0.0, 2.0])),
    ("day_count", np.array([2, 0])), ("day_count", np.array([2, -1])),
    ("day_count", np.array([2, 20])), ("day_count", np.array([[2, 1]])),
    ("day_count", np.array([2, np.iinfo(np.uint64).max], dtype=np.uint64)),
    ("symbol", np.array([0, -1, 0])), ("symbol", np.array([0, 1, 0])),
    ("symbol", np.array([0, 0])), ("symbol", np.array([False, False, False])),
    ("symbol_names", ()), ("trading_days_per_week", True),
])
def test_malformed_dataset_is_rejected_without_executing_kernel(field, value):
    with pytest.raises(InvalidTradeDataError):
        validate_trade_dataset_numeric(replace(dataset(), **{field: value}))


def test_valid_layout_and_integer_dtypes_remain_accepted():
    ds = dataset()
    validate_trade_dataset_numeric(ds)
    validate_trade_dataset_numeric(replace(ds, day=ds.day.astype(np.int64)))
