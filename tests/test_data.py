"""Step 3 — trade dataset and preprocessing (BUILD_SPEC Step 3, ARCHITECTURE §11)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from propfirm_engine.data import (
    InvalidTradeDataError,
    TradeDataset,
    clip_mae_to_holding_interval,
    preprocess,
    slice_days,
)


def _rows(specs):
    """specs: list of (timestamp, kwargs) → list of row dicts."""
    out = []
    for ts, kw in specs:
        row = {"timestamp": ts}
        row.update(kw)
        out.append(row)
    return out


# --- input contract: exactly one of `return` or (`pnl`+`size`) (§11.1) ------- #


def test_missing_both_return_and_pnlsize_is_rejected():
    rows = _rows([(datetime(2024, 1, 1, 10), {}), (datetime(2024, 1, 1, 11), {})])
    with pytest.raises(InvalidTradeDataError):
        preprocess(rows)


def test_return_column_is_accepted():
    rows = _rows(
        [
            (datetime(2024, 1, 1, 10), {"return": 1.0}),
            (datetime(2024, 1, 1, 11), {"return": -0.5}),
        ]
    )
    ds = preprocess(rows)
    assert isinstance(ds, TradeDataset)
    assert ds.n_trades == 2
    np.testing.assert_allclose(ds.ret, [1.0, -0.5])


def test_pnl_and_size_normalize_to_the_same_per_unit_return():
    # pnl=200 at size=2 must give the same per-unit return (100) a `return` column would.
    via_pnl = preprocess(
        _rows(
            [
                (datetime(2024, 1, 1, 10), {"pnl": 200.0, "size": 2.0}),
                (datetime(2024, 1, 1, 11), {"pnl": -50.0, "size": 5.0}),
            ]
        )
    )
    via_return = preprocess(
        _rows(
            [
                (datetime(2024, 1, 1, 10), {"return": 100.0}),
                (datetime(2024, 1, 1, 11), {"return": -10.0}),
            ]
        )
    )
    np.testing.assert_allclose(via_pnl.ret, via_return.ret)


def test_zero_size_is_rejected():
    rows = _rows([(datetime(2024, 1, 1, 10), {"pnl": 100.0, "size": 0.0})])
    with pytest.raises(InvalidTradeDataError):
        preprocess(rows)


def test_missing_timestamp_is_rejected():
    with pytest.raises(InvalidTradeDataError):
        preprocess({"return": [1.0, 2.0]})


# --- ordering (§11.1) -------------------------------------------------------- #


def test_trades_are_ordered_by_timestamp_regardless_of_input_order():
    rows = _rows(
        [
            (datetime(2024, 1, 1, 12), {"return": 3.0}),
            (datetime(2024, 1, 1, 10), {"return": 1.0}),
            (datetime(2024, 1, 1, 11), {"return": 2.0}),
        ]
    )
    ds = preprocess(rows)
    np.testing.assert_allclose(ds.ret, [1.0, 2.0, 3.0])


# --- session-day assignment (§11.3) ----------------------------------------- #


def test_session_boundary_not_midnight_splits_the_same_calendar_date():
    # reset 17:00: a 16:00 trade and an 18:00 trade on the SAME calendar date fall
    # in DIFFERENT sessions; a 16:00 trade the NEXT calendar date rejoins the 18:00
    # trade's session (the "and vice versa" case).
    rows = _rows(
        [
            (datetime(2024, 1, 1, 16), {"return": 1.0}),  # session A
            (datetime(2024, 1, 1, 18), {"return": 1.0}),  # session B (next day, same date)
            (datetime(2024, 1, 2, 16), {"return": 1.0}),  # session B (next date, same session)
        ]
    )
    ds = preprocess(rows, session_reset="17:00")
    np.testing.assert_array_equal(ds.day, [0, 1, 1])
    assert ds.n_days == 2


def test_day_indices_are_monotonic_non_decreasing():
    rows = _rows(
        [
            (datetime(2024, 1, 3, 12), {"return": 1.0}),
            (datetime(2024, 1, 1, 12), {"return": 1.0}),
            (datetime(2024, 1, 2, 12), {"return": 1.0}),
            (datetime(2024, 1, 2, 13), {"return": 1.0}),
        ]
    )
    ds = preprocess(rows, session_reset="17:00")
    assert list(ds.day) == sorted(ds.day)
    assert np.all(np.diff(ds.day) >= 0)


def test_reset_boundary_trade_rolls_to_next_session():
    # Exactly at the reset time belongs to the NEXT session (at/after semantics).
    rows = _rows(
        [
            (datetime(2024, 1, 1, 16, 59), {"return": 1.0}),  # before reset
            (datetime(2024, 1, 1, 17, 0), {"return": 1.0}),  # exactly reset -> next
        ]
    )
    ds = preprocess(rows, session_reset="17:00")
    np.testing.assert_array_equal(ds.day, [0, 1])


# --- trade_low: holding-interval clipping + fallback (§11.2, §D1) ------------ #


def test_clip_excludes_excursions_outside_the_holding_interval():
    entry = datetime(2024, 1, 1, 10, 0)
    exit_ = datetime(2024, 1, 1, 10, 5)
    bars = [
        (datetime(2024, 1, 1, 9, 59), 5.0),  # BEFORE entry — must not count
        (datetime(2024, 1, 1, 10, 2), 3.0),  # inside
        (datetime(2024, 1, 1, 10, 7), 9.0),  # AFTER exit — must not count
    ]
    assert clip_mae_to_holding_interval(entry, exit_, bars) == 3.0


def test_clip_deeper_excursion_fully_inside_does_lower_it():
    entry = datetime(2024, 1, 1, 10, 0)
    exit_ = datetime(2024, 1, 1, 10, 10)
    bars = [
        (datetime(2024, 1, 1, 10, 2), 3.0),
        (datetime(2024, 1, 1, 10, 6), 7.0),  # deeper, still inside — this wins
    ]
    assert clip_mae_to_holding_interval(entry, exit_, bars) == 7.0


def test_trade_low_uses_clipped_mae_when_supplied():
    entry = datetime(2024, 1, 1, 10, 0)
    exit_ = datetime(2024, 1, 1, 10, 5)
    bars = [
        (datetime(2024, 1, 1, 9, 59), 5.0),  # excluded
        (datetime(2024, 1, 1, 10, 2), 3.0),  # inside
    ]
    mae = clip_mae_to_holding_interval(entry, exit_, bars)  # -> 3.0
    ds = preprocess(_rows([(exit_, {"return": 1.0, "mae": mae})]))
    # trade_low is the per-unit floating low = -mae, and reflects only the inside
    # excursion (3.0), not the pre-entry 5.0.
    np.testing.assert_allclose(ds.trade_low, [-3.0])


def test_trade_low_falls_back_to_realized_down_move_without_mae():
    rows = _rows(
        [
            (datetime(2024, 1, 1, 10), {"return": 1.5}),  # win -> floating low 0
            (datetime(2024, 1, 1, 11), {"return": -0.8}),  # loss -> floating low = ret
        ]
    )
    ds = preprocess(rows)
    np.testing.assert_allclose(ds.trade_low, [0.0, -0.8])


def test_negative_mae_is_rejected():
    rows = _rows([(datetime(2024, 1, 1, 10), {"return": 1.0, "mae": -2.0})])
    with pytest.raises(InvalidTradeDataError):
        preprocess(rows)


def test_partial_mae_column_uses_per_row_fallback_not_zero():
    # A row WITHOUT mae in a dataset where OTHER rows have it must fall back to
    # min(ret,0), NOT silently become 0 — a per-column choice would zero the
    # un-annotated loss and hide an intraday breach (§D1 direction).
    rows = _rows(
        [
            (datetime(2024, 1, 1, 10), {"return": 1.0, "mae": 2.0}),  # annotated
            (datetime(2024, 1, 1, 11), {"return": -0.8}),  # NO mae -> fallback
        ]
    )
    ds = preprocess(rows)
    np.testing.assert_allclose(ds.trade_low, [-2.0, -0.8])


def test_in_pipeline_clip_derives_trade_low_from_bar_excursions():
    # End-to-end: preprocess itself clips bar excursions to [entry_time, timestamp]
    # when a row carries entry_time + mae_bars. A pre-entry bar low must NOT lower
    # trade_low — proving the clip is a real, tested part of the derived field, not
    # an off-pipeline value the test trusts.
    entry = datetime(2024, 1, 1, 10, 0)
    exit_ = datetime(2024, 1, 1, 10, 5)
    rows = _rows(
        [
            (
                exit_,
                {
                    "return": 1.0,
                    "entry_time": entry,
                    "mae_bars": [
                        (datetime(2024, 1, 1, 9, 59), 5.0),  # before entry — excluded
                        (datetime(2024, 1, 1, 10, 2), 3.0),  # inside
                        (datetime(2024, 1, 1, 10, 7), 9.0),  # after exit — excluded
                    ],
                },
            )
        ]
    )
    ds = preprocess(rows)
    np.testing.assert_allclose(ds.trade_low, [-3.0])


def test_mae_bars_without_entry_time_is_rejected():
    rows = _rows(
        [
            (
                datetime(2024, 1, 1, 10, 5),
                {"return": 1.0, "mae_bars": [(datetime(2024, 1, 1, 10, 2), 3.0)]},
            )
        ]
    )
    with pytest.raises(InvalidTradeDataError):
        preprocess(rows)


def test_both_return_and_pnlsize_is_rejected():
    # Exactly one source of per-unit return (§11.1) — providing both is ambiguous.
    rows = _rows(
        [(datetime(2024, 1, 1, 10), {"return": 5.0, "pnl": 999.0, "size": 3.0})]
    )
    with pytest.raises(InvalidTradeDataError):
        preprocess(rows)


def test_missing_return_cell_is_rejected():
    rows = _rows(
        [
            (datetime(2024, 1, 1, 10), {"return": 1.0}),
            (datetime(2024, 1, 1, 11), {}),  # no return on this row
        ]
    )
    with pytest.raises(InvalidTradeDataError):
        preprocess(rows)


def test_missing_size_cell_is_rejected():
    rows = _rows(
        [
            (datetime(2024, 1, 1, 10), {"pnl": 100.0, "size": 2.0}),
            (datetime(2024, 1, 1, 11), {"pnl": 50.0}),  # no size -> nan -> reject
        ]
    )
    with pytest.raises(InvalidTradeDataError):
        preprocess(rows)


# --- per-day table and cadence (§11.2, §11.5) ------------------------------- #


def test_per_day_table_locates_each_days_trades():
    rows = _rows(
        [
            (datetime(2024, 1, 1, 10), {"return": 1.0}),
            (datetime(2024, 1, 1, 11), {"return": 2.0}),
            (datetime(2024, 1, 2, 10), {"return": 3.0}),
        ]
    )
    ds = preprocess(rows, session_reset="17:00")
    assert ds.n_days == 2
    np.testing.assert_array_equal(ds.day_count, [2, 1])
    # day 0 holds the first two trades; day 1 the third
    np.testing.assert_allclose(ds.ret[ds.day_slice(0)], [1.0, 2.0])
    np.testing.assert_allclose(ds.ret[ds.day_slice(1)], [3.0])


def test_trading_days_per_week_is_distinct_days_over_calendar_week_span():
    # Trade on six distinct session days spanning Jan 1..Jan 12 (span 11 days).
    days = [1, 3, 5, 8, 10, 12]
    rows = _rows([(datetime(2024, 1, d, 10), {"return": 1.0}) for d in days])
    ds = preprocess(rows, session_reset="17:00")
    assert ds.n_days == 6
    span_days = 12 - 1  # last - first calendar date
    expected = 6 / (span_days / 7.0)
    assert ds.trading_days_per_week == pytest.approx(expected)


def test_trading_days_per_week_override_is_honored():
    rows = _rows([(datetime(2024, 1, 1, 10), {"return": 1.0})])
    ds = preprocess(rows, trading_days_per_week=5.0)
    assert ds.trading_days_per_week == 5.0


# --- multi-asset joint days (§11.4, §G1) ------------------------------------ #


def test_multi_asset_day_is_all_symbols_trades_on_that_session_day():
    # Two symbols; on session day 0 both trade, on day 1 only ES trades. A day is
    # the union across assets; the absent asset on day 1 is legitimately missing.
    rows = _rows(
        [
            (datetime(2024, 1, 1, 10), {"return": 1.0, "symbol": "ES"}),
            (datetime(2024, 1, 1, 10), {"return": 2.0, "symbol": "NQ"}),
            (datetime(2024, 1, 2, 10), {"return": 3.0, "symbol": "ES"}),
        ]
    )
    ds = preprocess(rows, session_reset="17:00")
    assert ds.n_days == 2
    # day 0 carries both symbols' trades; day 1 only one
    day0_syms = set(ds.symbol[ds.day_slice(0)].tolist())
    day1_syms = set(ds.symbol[ds.day_slice(1)].tolist())
    assert len(day0_syms) == 2
    assert len(day1_syms) == 1
    assert ds.symbol_names == ("ES", "NQ")


def test_day_identity_is_fixed_by_the_session_calendar_not_by_symbol():
    # The same session day groups both symbols even when their timestamps differ
    # within the session — day identity is the canonical session date (§G1), so the
    # bootstrap cannot manufacture cross-asset alignment later.
    rows = _rows(
        [
            (datetime(2024, 1, 1, 18), {"return": 1.0, "symbol": "ES"}),  # session Jan 2
            (datetime(2024, 1, 2, 9), {"return": 2.0, "symbol": "NQ"}),  # session Jan 2
        ]
    )
    ds = preprocess(rows, session_reset="17:00")
    assert ds.n_days == 1
    np.testing.assert_array_equal(ds.day, [0, 0])


# --- column-oriented input also works --------------------------------------- #


def test_column_oriented_input_is_accepted():
    cols = {
        "timestamp": [datetime(2024, 1, 1, 11), datetime(2024, 1, 1, 10)],
        "return": [2.0, 1.0],
    }
    ds = preprocess(cols)
    np.testing.assert_allclose(ds.ret, [1.0, 2.0])


# --- slice_days: the time-slice primitive for rolling folds ----------------- #


def _multiday(n_days, per_day=2):
    """A dataset of `n_days` sessions, `per_day` trades each, return == day index."""
    rows = []
    for d in range(n_days):
        for k in range(per_day):
            rows.append((datetime(2024, 1, 1) + timedelta(days=d, hours=9 + k),
                         {"return": float(d)}))
    return preprocess(_rows(rows), session_reset="00:00")


def test_slice_days_rebases_and_stays_dense():
    ds = _multiday(10, per_day=3)
    lo, hi = slice_days(ds, 0, 7), slice_days(ds, 7, 10)
    assert (lo.n_days, hi.n_days) == (7, 3)
    assert lo.n_days + hi.n_days == ds.n_days
    assert int(lo.day.min()) == 0 and int(hi.day.min()) == 0  # re-based to 0
    assert lo.day_count.min() > 0 and hi.day_count.min() > 0  # side table dense
    assert lo.ret.size + hi.ret.size == ds.ret.size  # no trades lost
    # the later slice carries the later days' returns (== original day index)
    np.testing.assert_array_equal(np.unique(hi.ret), [7.0, 8.0, 9.0])
    # Derived cadence is estimated from the slice, not future/other periods.
    assert hi.trading_days_per_week == 3.0
    assert hi.session_reset == ds.session_reset


def test_slice_days_rejects_bad_range():
    ds = _multiday(5)
    for bad in [(-1, 3), (2, 2), (3, 2), (0, 6)]:
        with pytest.raises(ValueError):
            slice_days(ds, *bad)
