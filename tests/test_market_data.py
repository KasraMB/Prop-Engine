from datetime import date, datetime, time, timedelta, timezone
from fractions import Fraction

import numpy as np
import pytest

from propfirm_engine import Instrument, Market, Quote
from propfirm_engine.market_data import MarketTape


AT = datetime(2026, 9, 1, 14, 0, 0, 123456, tzinfo=timezone.utc)
ES = Instrument("ES", 50, .25)
DAYS = (date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3))


def markets():
    return [Market(AT+timedelta(days=i), (Quote("ES", Fraction(-1, 4), 0, 0, 2, None),), seq=7)
            for i in (0, 2)]


def test_round_trip_preserves_negative_ticks_missing_sessions_liquidity_and_microseconds():
    tape = MarketTape(markets(), [ES], sessions=DAYS)
    assert list(tape) == markets()
    assert len(tape) == 2
    assert len(tape.view(1, 2)) == 0
    assert list(tape.view(2, 3)) == markets()[1:]


def test_windows_share_storage_and_cannot_escape_parent_bounds():
    tape = MarketTape(markets(), [ES], sessions=DAYS)
    view = tape.view(1, 3)
    assert view.tape.frames is tape.frames and view.tape.quotes is tape.quotes
    assert view.view(1, 2).sessions == DAYS[2:]
    with pytest.raises(ValueError):
        view.view(-1, 1)


def test_prepared_arrays_cannot_be_made_writeable():
    tape = MarketTape(markets(), [ES], sessions=DAYS)
    for column in (tape.frames, tape.quotes, tape.offsets):
        with pytest.raises(ValueError):
            column.flags.writeable = True
    assert tape.nbytes == tape.frames.nbytes+tape.quotes.nbytes+tape.offsets.nbytes


def test_fingerprint_is_stable_and_sensitive_to_price_and_clock():
    tape = MarketTape(markets(), [ES], sessions=DAYS)
    assert tape.fingerprint == MarketTape(iter(markets()), [ES], sessions=DAYS).fingerprint
    changed = [Market(AT, (Quote("ES", 0, 1, 1),))]
    assert tape.fingerprint != MarketTape(changed, [ES], sessions=DAYS).fingerprint


@pytest.mark.parametrize("stream", [list(reversed(markets())), [markets()[0], markets()[0]],
    [Market(AT, (Quote("X", 1, 1, 1),))], [Market(AT, (Quote("ES", .1, .1, .1),))],
    [Market(AT, (Quote("ES", 1, 1, 1),), seq=2**64)]])
def test_invalid_or_unrepresentable_inputs_are_rejected(stream):
    with pytest.raises(ValueError):
        MarketTape(stream, [ES], sessions=DAYS)
