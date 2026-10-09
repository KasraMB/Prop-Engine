"""Conditional-null mechanics and canonical-engine integration, without market data."""
from datetime import date, datetime, timedelta, timezone
from itertools import product

import numpy as np
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pandas_market_calendars")

from propfirm_engine.market_replay import PriceSession
from Test_Strategies.zero_drift import SCALE, compare, execute, holm, innovations, prepare, random_sessions, reflect


def sample():
    close = datetime(2025, 1, 6, 21, 45, tzinfo=timezone.utc)
    times = np.array([int((close - timedelta(minutes=m)).timestamp() * 1e9) for m in (3, 2, 1)])
    prices = np.array([[100, 103, 99, 102], [103, 105, 101, 104], [102, 104, 100, 101]], dtype=float)
    evening = PriceSession(date(2025, 1, 6), close, times, prices)
    morning = PriceSession(evening.session, close, times[1:], prices[1:])
    return {"18:00": [evening], "09:30": [morning]}


def test_sign_enumeration_zero_expected_innovations_and_valid_bars():
    session = sample()["18:00"][0]
    anchor, changes = innovations(session)
    increments = []
    for signs in product((-1, 1), repeat=3):
        p = reflect(anchor, changes, signs)
        prev = np.r_[p[0, 0], p[:-1, 3]]
        increments.append(p[:, 3] - prev)
        assert np.all(p[:, 1] >= p.max(axis=1))
        assert np.all(p[:, 2] <= p.min(axis=1))
        np.testing.assert_allclose(p[:, 1] - p[:, 2], session.ohlc[:, 1] - session.ohlc[:, 2])
        np.testing.assert_allclose(abs(p[:, 0] - prev), abs(changes[:, 0]) / SCALE)
        np.testing.assert_allclose(abs(p[:, 3] - p[:, 0]), abs(session.ohlc[:, 3] - session.ohlc[:, 0]))
    np.testing.assert_array_equal(np.mean(increments, axis=0), np.zeros(3))
    np.testing.assert_array_equal(reflect(anchor, changes, np.ones(3, dtype=int)), session.ohlc)


def test_seed_determinism_and_shared_entry_tape():
    prepared = prepare(sample())
    first = random_sessions(prepared, 12, "MES", 7)
    second = random_sessions(prepared, 12, "MES", 7)
    np.testing.assert_array_equal(first["18:00"][0].ohlc, second["18:00"][0].ohlc)
    np.testing.assert_array_equal(first["09:30"][0].ohlc, first["18:00"][0].ohlc[1:])
    np.testing.assert_array_equal(first["18:00"][0].timestamps, sample()["18:00"][0].timestamps)


def test_all_positive_signs_reproduce_canonical_execution():
    sessions = sample()
    s, offset, anchor, changes = prepare(sessions)[0]
    identity = PriceSession(s.session, s.close_at, s.timestamps, reflect(anchor, changes, np.ones(3, dtype=int)))
    original = execute("MES", "full_size", sessions["18:00"], 0)
    control = execute("MES", "full_size", [identity], 0)
    assert original == control


def test_tail_ties_and_holm():
    result = compare(2., [1., 2., 3.])
    assert result["upper_tail_exceedances"] == 2
    assert result["upper_tail_rank"] == .75
    assert result["control"]["mean"] == 2.
    assert holm([.04, .01, .03]) == pytest.approx([.06, .03, .06])


def test_reject_invalid_signs_and_unrepresentable_prices():
    session = sample()["18:00"][0]
    anchor, changes = innovations(session)
    with pytest.raises(ValueError):
        reflect(anchor, changes, [0, 1, 1])
    prices = session.ohlc + .0000001
    with pytest.raises(ValueError, match="six decimal"):
        innovations(PriceSession(session.session, session.close_at, session.timestamps, prices))


def test_mismatched_entry_histories_rejected():
    sessions = sample()
    sessions["09:30"] = []
    with pytest.raises(ValueError, match="equal lengths"):
        prepare(sessions)


def test_skipping_another_session_does_not_change_random_tape():
    first = sample()
    following = {}
    for entry, history in first.items():
        s = history[0]
        following[entry] = [PriceSession(s.session + timedelta(days=1), s.close_at + timedelta(days=1),
                                        s.timestamps + 86_400_000_000_000, s.ohlc)]
    combined = {entry: first[entry] + following[entry] for entry in first}
    together = random_sessions(prepare(combined), 12, "MES", 7)
    alone = random_sessions(prepare(following), 12, "MES", 7)
    for entry in combined:
        np.testing.assert_array_equal(together[entry][1].ohlc, alone[entry][0].ohlc)


def test_integer_reconstruction_does_not_perturb_fractional_ticks():
    s = sample()["18:00"][0]
    prices = s.ohlc * .00005 + 1.12345
    precise = PriceSession(s.session, s.close_at, s.timestamps, np.round(prices, 5))
    anchor, changes = innovations(precise)
    restored = reflect(anchor, changes, [1, 1, 1])
    np.testing.assert_array_equal(restored, precise.ohlc)
    reflected = reflect(anchor, changes, [-1, 1, -1])
    np.testing.assert_allclose(reflected / .00005, np.round(reflected / .00005), atol=1e-9, rtol=0)
