"""Optional DuckDB research-layer checks; no private market data required."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("pandas_market_calendars")
import pandas as pd

from Test_Strategies.open_long import ASSETS, load_sessions


def test_paired_entry_times_share_quality_filter_and_dst_safe_clocks(tmp_path):
    zone = ZoneInfo("America/New_York")
    rows = []
    for day in (6, 7, 8):
        date = datetime(2025, 1, day, tzinfo=zone)
        times = [date - timedelta(hours=6), date + timedelta(hours=9, minutes=30),
                 date + timedelta(hours=16, minutes=44)]
        if day == 8:
            times.pop(0)
        for at in times:
            rows.append({"ts_event": at, "instrument_id": 1, "open": 100., "high": 101.,
                         "low": 99., "close": 100., "degraded_day": day == 7})
    frame = pd.DataFrame(rows)
    frame["ts_event"] = pd.to_datetime(frame.ts_event, utc=True)
    path = tmp_path / "continuous/roll_rule=volume/product=MES/data.parquet"
    path.parent.mkdir(parents=True)
    with duckdb.connect() as con:
        con.register("frame", frame)
        con.execute("COPY frame TO ? (FORMAT PARQUET)", [str(path)])
    sessions, quality = load_sessions(tmp_path, "MES")
    assert quality["paired_eligible_sessions"] == 1
    assert quality["excluded_sessions"] == 2
    assert quality["exclusion_reason_counts_overlap"]["degraded_session"] == 1
    assert quality["exclusion_reason_counts_overlap"]["missing_1800"] == 1
    assert sessions["09:30"][0].session == sessions["18:00"][0].session
    assert sessions["09:30"][0].entry_at.astimezone(zone).hour == 9
    assert sessions["18:00"][0].entry_at.astimezone(zone).day == 5
    assert sessions["18:00"][0].close_at.astimezone(zone).hour == 16


def test_commission_equivalent_micro_counts():
    from fractions import Fraction
    counts = {p: int(Fraction(str(v[3])) // Fraction(str(v[4]))) for p, v in ASSETS.items() if v[4]}
    assert counts == {"MES": 3, "MNQ": 3, "M2K": 3, "MCL": 4, "MGC": 2, "SIL": 1}
