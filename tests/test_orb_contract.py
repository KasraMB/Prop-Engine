"""Causal ORB fixtures: prices are synthetic and imply no strategy performance."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from Test_Strategies.orb_reference import ORBConfig, PriceBar, opening_range_breakout

NY = ZoneInfo("America/New_York")


def bar(at, o=100, h=101, l=99, c=100, minutes=1):
    return PriceBar(at, at + timedelta(minutes=minutes), o, h, l, c)


def opening(day="2026-01-05"):
    start = datetime.fromisoformat(day + "T09:30:00").replace(tzinfo=NY)
    return [bar(start + timedelta(minutes=i)) for i in range(15)]


def long_setup(day="2026-01-05"):
    bars = opening(day)
    bars += [bar(bars[-1].end, h=103, c=102)]
    bars += [bar(bars[-1].end, o=102, h=105, l=100, c=104)]
    return bars


def test_long_entry_follows_completed_signal_and_anchors_actual_stop():
    bars = long_setup()
    entry, = opening_range_breakout(bars).entries
    assert entry.side == 1
    assert entry.signal_at == bars[15].end == entry.entry_at == bars[16].start
    assert entry.reference_entry_price == 102
    assert (entry.range_high, entry.range_low, entry.stop_price) == (101, 99, 99)
    assert entry.risk_points == 3  # not range width 2


def test_short_entry_and_stop():
    bars = opening()
    bars += [bar(bars[-1].end, l=97, c=98)]
    bars += [bar(bars[-1].end, o=98, h=100, l=97, c=99)]
    entry, = opening_range_breakout(bars).entries
    assert entry.side == -1 and entry.stop_price == 101 and entry.risk_points == 3


@pytest.mark.parametrize("close", [99, 100, 101])
def test_touching_boundary_does_not_trigger(close):
    bars = opening()
    bars += [bar(bars[-1].end, c=close), bar(bars[-1].end + timedelta(minutes=1))]
    assert opening_range_breakout(bars).entries == ()


def test_next_open_is_not_signal_close_or_future_entry_bar_close():
    bars = long_setup()
    bars[-1] = replace(bars[-1], open=104, high=200, close=180)
    entry, = opening_range_breakout(bars).entries
    assert entry.reference_entry_price == 104 and entry.risk_points == 5
    alternate = bars[:-1] + [replace(bars[-1], high=104, close=100)]
    assert opening_range_breakout(alternate).entries == (entry,)


def test_future_bars_do_not_change_existing_entry():
    bars = long_setup()
    before = opening_range_breakout(bars).entries
    bars += [bar(bars[-1].end, o=100, h=1000, l=-100, c=0)]
    assert opening_range_breakout(bars).entries == before


def test_one_entry_per_day_and_reset_next_day():
    bars = long_setup()
    bars += [bar(bars[-1].end, l=90, c=95), bar(bars[-1].end + timedelta(minutes=1))]
    bars += long_setup("2026-01-06")
    assert [e.session for e in opening_range_breakout(bars).entries] == ["2026-01-05", "2026-01-06"]


def test_no_fill_without_next_bar_and_no_next_day_carry():
    bars = long_setup()[:-1]
    result = opening_range_breakout(bars + opening("2026-01-06"))
    assert not result.entries
    assert ("2026-01-05", "missing_entry_bar") in result.skipped


def test_entry_requires_contiguous_next_bar():
    bars = long_setup()
    bars[-1] = replace(bars[-1], start=bars[-1].start + timedelta(minutes=1),
                       end=bars[-1].end + timedelta(minutes=1))
    result = opening_range_breakout(bars)
    assert not result.entries
    assert result.skipped == (("2026-01-05", "no_contiguous_entry_before_cutoff"),)


def test_gap_through_stop_skips_entry_without_retry():
    bars = long_setup()
    bars[-1] = bar(bars[-1].start, o=98, h=103, l=97, c=102)
    bars += [bar(bars[-1].end, o=102, h=103, l=100, c=102)]
    result = opening_range_breakout(bars)
    assert not result.entries
    assert result.skipped == (("2026-01-05", "entry_gapped_beyond_stop"),)


@pytest.mark.parametrize("missing", [0, 7, 14])
def test_incomplete_range_cannot_generate_entry(missing):
    bars = long_setup()
    del bars[missing]
    result = opening_range_breakout(bars)
    assert not result.entries and result.skipped


def test_incomplete_final_session_is_reported():
    result = opening_range_breakout(opening()[:7])
    assert result.skipped == (("2026-01-05", "incomplete_or_zero_width_opening_range"),)


def test_missing_bars_before_signal_cannot_hide_an_earlier_breakout():
    bars = long_setup()
    bars[15:] = [replace(b, start=b.start + timedelta(minutes=2),
                          end=b.end + timedelta(minutes=2)) for b in bars[15:]]
    result = opening_range_breakout(bars)
    assert not result.entries
    assert result.skipped == (("2026-01-05", "missing_signal_bar"),)


def test_zero_width_range_is_rejected():
    bars = long_setup()
    bars[:15] = [bar(b.start, h=100, l=100) for b in bars[:15]]
    result = opening_range_breakout(bars)
    assert not result.entries and result.skipped


def test_bar_crossing_range_boundary_is_not_used():
    bars = opening()[:14]
    bars += [bar(bars[-1].end, h=103, c=102, minutes=2)]
    bars += [bar(bars[-1].end, o=102, h=104, c=103)]
    result = opening_range_breakout(bars)
    assert not result.entries and result.skipped


def test_entry_at_cutoff_is_excluded():
    bars = long_setup()
    config = ORBConfig(entry_cutoff=datetime.strptime("09:46", "%H:%M").time())
    result = opening_range_breakout(bars, config)
    assert not result.entries
    assert result.skipped == (("2026-01-05", "no_contiguous_entry_before_cutoff"),)


@pytest.mark.parametrize("day,hour", [("2026-01-05", 14), ("2026-07-06", 13)])
def test_new_york_clock_tracks_dst_with_utc_input(day, hour):
    bars = [replace(b, start=b.start.astimezone(timezone.utc), end=b.end.astimezone(timezone.utc))
            for b in long_setup(day)]
    entry, = opening_range_breakout(bars).entries
    assert entry.entry_at.hour == hour and entry.entry_at.minute == 46


@pytest.mark.parametrize("mutate", [
    lambda b: [b[1], b[0]],
    lambda b: [b[0], b[0]],
    lambda b: [b[0], replace(b[1], start=b[0].start)],
])
def test_out_of_order_duplicate_or_overlapping_bars_rejected(mutate):
    with pytest.raises(ValueError, match="chronological"):
        opening_range_breakout(mutate(opening()))


@pytest.mark.parametrize("change", [
    {"open": float("nan")}, {"close": float("inf")}, {"low": True},
    {"high": 98}, {"low": 102}, {"open": "100"}, {"high": 10**500},
    {"start": datetime(2026, 1, 5)}, {"end": datetime(2026, 1, 5)},
])
def test_invalid_bar_values_rejected(change):
    with pytest.raises(ValueError):
        replace(opening()[0], **change)


@pytest.mark.parametrize("minutes", [0, -1, True, 1.5])
def test_invalid_range_length_rejected(minutes):
    with pytest.raises(ValueError):
        ORBConfig(range_minutes=minutes)


def test_empty_input_and_scope():
    result = opening_range_breakout([])
    assert result.entries == result.skipped == ()
    assert result.execution_model == "signals_only_next_bar_open_reference"
    assert any("not ordered MTM" in a for a in result.assumptions)


def test_strategy_is_not_an_engine_api_export():
    import propfirm_engine

    for name in ("PriceBar", "ORBConfig", "ORBEntry", "ORBResult", "opening_range_breakout"):
        assert name not in propfirm_engine.__all__
        assert not hasattr(propfirm_engine, name)


def test_strategy_is_not_packaged_with_engine_or_browser():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert not (root / "src/propfirm_engine/orb.py").exists()
    assert not (root / "docs/py/propfirm_engine/orb.py").exists()
    manifest = json.loads((root / "docs/py/manifest_replay.json").read_text(encoding="utf-8"))
    assert "propfirm_engine/orb.py" not in {entry["path"] for entry in manifest["files"]}
