from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from propfirm_engine import ProcessingCalendar


NY = ZoneInfo("America/New_York")


def test_elapsed_delay_then_roll_forward_across_dst_weekend_and_holiday():
    calendar = ProcessingCalendar(holidays=(date(2026, 11, 2),))
    friday = datetime(2026, 10, 30, 16, 30, tzinfo=NY)
    assert calendar.after(friday, timedelta(hours=1)) == datetime(2026, 11, 3, 9, tzinfo=NY)


def test_processing_close_is_exclusive_and_open_is_inclusive():
    calendar = ProcessingCalendar()
    opening = datetime(2026, 9, 1, 9, tzinfo=NY)
    closing = datetime(2026, 9, 1, 17, tzinfo=NY)
    assert calendar.after(opening, timedelta(0)) == opening
    assert calendar.after(closing, timedelta(0)) == opening + timedelta(days=1)


@pytest.mark.parametrize("kwargs", [dict(weekdays=()), dict(weekdays=(7,)),
    dict(holidays=("2026-01-01",)), dict(opening=time(18)), dict(closing=time(9)),
    dict(opening=time(9, tzinfo=timezone.utc))])
def test_invalid_processing_calendar(kwargs):
    with pytest.raises(ValueError):
        ProcessingCalendar(**kwargs)
