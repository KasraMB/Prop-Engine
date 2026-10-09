from dataclasses import replace
from datetime import date, datetime, time, timedelta
from fractions import Fraction
from zoneinfo import ZoneInfo

import pytest

from propfirm_engine import (
    BacktestConfig, BracketHistory, BracketTrade, DollarPolicy, Engine, Fill,
    Instrument, Marks,
)
from propfirm_engine.firms.lucidflex import replay_50k


NY = ZoneInfo("America/New_York")
FIRST = date(2026, 9, 1)
X = Instrument("X", 1, 1)
Y = Instrument("Y", 1, 1)


def at(day=FIRST, hour=10, minute=0):
    return datetime.combine(day, time(hour, minute), NY)


def dates(n):
    out, day = [], FIRST
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return tuple(out)


def spec():
    return replay_50k(eval_fee=105.2, reset_fee=105, contract_type="mini", elapsed_inactivity=True)


def config(**kwargs):
    return BacktestConfig(**(dict(cost_per_contract=0, approval_delay=timedelta(0),
                                 receipt_delay=timedelta(0), activation_delay=timedelta(0)) | kwargs))


def run(events, **kwargs):
    settings = dict(sessions=(FIRST,), fidelity="observed_marks", mark_fills=True,
                    max_mark_age=timedelta(minutes=10), liquidation_fee=2, trace=True)
    settings.update(kwargs)
    profile = settings.pop("spec", spec())
    cfg = settings.pop("config", config())
    instruments = settings.pop("instruments", (X,))
    return Engine().replay_events(profile, iter(events), instruments, cfg, **settings)


def round_trip(day, profit, *, fee=0):
    return [Fill(at(day), "X", 1, 10_000, fee),
            Fill(at(day, minute=5), "X", -1, 10_000 + profit, fee)]


def winning_path(n=7):
    return [e for i, day in enumerate(dates(n))
            for e in round_trip(day, 1500 if i < 2 else 200)]


def test_arbitrary_partial_exits_and_fees():
    events = [Fill(at(), "X", 2, 100, 2), Fill(at(minute=1), "X", -1, 123, 1),
              Fill(at(minute=2), "X", -1, 91, 1)]
    result = run(events)
    assert result.book.balance == 50_010
    assert result.book.realized == 14 and result.book.fees == 4
    assert result.fills == 3 and result.skipped_fills == 0
    assert result.replay.fees == 105.2


def test_unrealized_breach_cannot_recover_on_recorded_exit():
    result = run([Fill(at(), "X", 1, 10_000),
                  Marks(at(minute=1), (("X", 8000),)),
                  Fill(at(minute=2), "X", -1, 11_000)])
    assert result.replay.failed_attempts == 1
    failure, = [e for e in result.replay.events if e.kind == "failure"]
    assert failure.at == at(minute=1) and failure.balance == 47_998
    assert failure.code == "FAIL_TRAILING_DD"
    assert result.fills == 1 and result.skipped_fills == 1
    assert result.book is None


def test_atomic_multi_asset_marks_do_not_create_false_breach():
    result = run([Fill(at(), "X", 1, 10_000), Fill(at(), "Y", 1, 10_000, seq=1),
                  Marks(at(minute=1), (("X", 7000), ("Y", 13_000))),
                  Fill(at(minute=2), "X", -1, 7000),
                  Fill(at(minute=2), "Y", -1, 13_000, seq=1)], instruments=(X, Y))
    assert result.replay.failed_attempts == 0
    assert result.book.balance == 50_000
    assert next(t for t in result.trace if t.kind == "mark").equity == 50_000


def test_full_lucid_path_matches_existing_bracket_lifecycle():
    session_ids = dates(7)
    bracket = BracketHistory(tuple(BracketTrade(at(day), at(day, minute=5), day,
                                                100, 1500 if i < 2 else 200, True)
                                   for i, day in enumerate(session_ids)))
    baseline = Engine().backtest(spec(), bracket, DollarPolicy.constant(100), config())
    result = run(winning_path(), sessions=session_ids)
    kinds = {"evaluation_pass", "request", "approval", "receipt", "failure", "fee"}
    assert [e for e in result.replay.events if e.kind in kinds] == [
        e for e in baseline.events if e.kind in kinds]
    assert result.replay.net_cash == baseline.net_cash == 344.8
    assert result.book.balance == baseline.final_balance == 50_500


def test_open_profit_cannot_pass_evaluation():
    day2 = dates(2)[1]
    events = round_trip(FIRST, 1500) + [Fill(at(day2), "X", 1, 10_000),
              Marks(at(day2, minute=1), (("X", 11_500),)),
              Fill(at(day2, minute=2), "X", -1, 10_000)]
    result = run(events, sessions=dates(2))
    assert not any(e.kind == "evaluation_pass" for e in result.replay.events)
    mark, = [t for t in result.trace if t.kind == "mark"]
    assert mark.equity == 53_000 and mark.balance == 51_500


def test_forced_session_close_and_source_quarantine():
    day2 = dates(2)[1]
    events = [Fill(at(hour=16, minute=44), "X", 1, 10_000),
              Marks(at(hour=16, minute=45), (("X", 10_123),)),
              Fill(at(day2), "X", -1, 9000)] + round_trip(day2, 10)
    events[-2] = replace(events[-2], seq=1)
    result = run(events, sessions=dates(2))
    assert result.book.balance == 50_131
    assert result.skipped_fills == 1 and result.fills == 3
    assert [e.kind for e in result.replay.events].count("liquidation") == 1


def test_retry_does_not_turn_old_exit_into_a_short():
    day2, day3 = dates(3)[1:]
    events = [Fill(at(), "X", 1, 10_000), Marks(at(minute=1), (("X", 8000),)),
              Fill(at(day2), "X", -1, 12_000)] + round_trip(day3, 100)
    result = run(events, sessions=dates(3))
    assert result.replay.attempts == 2 and result.replay.failed_attempts == 1
    assert result.book.balance == 50_100 and result.replay.fees == 210.2
    assert result.skipped_fills == 1


def test_pending_entry_and_its_exit_are_both_skipped():
    day8, day9 = dates(9)[7:]
    events = winning_path() + [Fill(at(day8), "X", 1, 10_000),
                              Fill(at(day9), "X", -1, 12_000)]
    result = run(events, sessions=dates(9), config=config(approval_delay=timedelta(days=1)))
    assert result.skipped_fills == 2 and result.fills == 14
    assert result.book.balance == 50_500 and result.replay.receipts == 450


def test_inactivity_exact_tie_qualifying_close_wins():
    end = FIRST + timedelta(days=30)
    events = [Fill(at(), "X", 1, 10_000), Fill(at(minute=1), "X", -1, 10_001),
              Fill(at(end), "X", 1, 10_000), Fill(at(end, minute=1), "X", -1, 9999)]
    result = run(events, sessions=(FIRST, end))
    assert result.replay.failed_attempts == 0


def test_entry_fee_is_not_qualifying_activity():
    end = FIRST + timedelta(days=30)
    events = [Fill(at(), "X", 1, 10_000), Fill(at(minute=1), "X", -1, 10_001),
              Fill(at(end), "X", 1, 10_000, 5),
              Marks(at(end, minute=1), (("X", 10_000),)),
              Fill(at(end, minute=2), "X", -1, 10_000)]
    result = run(events, sessions=(FIRST, end))
    failure, = [e for e in result.replay.events if e.kind == "failure"]
    assert failure.code == "FAIL_INACTIVITY" and failure.at == at(end, minute=1)


def test_stale_marks_cannot_be_used_for_forced_close():
    with pytest.raises(ValueError, match="stale"):
        run([Fill(at(), "X", 1, 10_000)])


def test_contract_limits_are_account_wide_and_weighted():
    events = [Fill(at(), "X", 3, 10_000), Fill(at(minute=1), "Y", 2, 10_000)]
    with pytest.raises(ValueError, match="contract limit"):
        run(events, instruments=(X, Y))
    events += [Fill(at(minute=2), "X", -3, 10_000), Fill(at(minute=3), "Y", -2, 10_000)]
    result = run(events, instruments=(X, Y), units={"X": 1, "Y": Fraction(1, 10)})
    assert result.fills == 4 and result.replay.failed_attempts == 0


@pytest.mark.parametrize("kwargs,match", [
    ({"fidelity": "strict"}, "observed_marks"),
    ({"max_mark_age": timedelta(seconds=-1)}, "max_mark_age"),
    ({"liquidation_fee": -1}, "liquidation_fee"),
    ({"trace": 1}, "trace"),
    ({"sessions": ()}, "sessions"),
    ({"sessions": (FIRST, FIRST)}, "sessions"),
    ({"sessions": (date(2026, 9, 5),)}, "sessions"),
    ({"config": config(cost_per_contract=1)}, "fees from fills"),
    ({"units": {"Y": 1}}, "every instrument"),
    ({"units": {"X": 0}}, "positive"),
    ({"session_closes": {FIRST: at(hour=17)}}, "firm cutoff"),
    ({"session_closes": {dates(2)[1]: at()}}, "declared sessions"),
])
def test_invalid_configuration_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        run([], **kwargs)


@pytest.mark.parametrize("when", [at(hour=17), at(dates(2)[1]), at(FIRST - timedelta(days=1), hour=17)])
def test_events_outside_declared_sessions_rejected(when):
    with pytest.raises(ValueError, match="outside|horizon"):
        run([Fill(when, "X", 1, 10_000)])


def test_trace_does_not_change_economics():
    traced = run(winning_path(), sessions=dates(7))
    summary = run(winning_path(), sessions=dates(7), trace=False)
    assert traced.replay == summary.replay and traced.book == summary.book
    assert traced.trace and summary.trace == ()


def test_future_events_do_not_change_prior_results():
    short = run(winning_path(), sessions=dates(7))
    long = run(winning_path(8), sessions=dates(8))
    assert tuple(e for e in long.replay.events if e.at <= short.replay.end) == short.replay.events
    assert tuple(e for e in long.trace if e.at <= short.replay.end) == short.trace


def test_live_handoff_restarts_using_the_shared_fee_and_wallet_logic():
    result = run(winning_path(30), sessions=dates(30))
    assert any(e.kind == "live_handoff" for e in result.replay.events)
    assert result.replay.attempts == 2 and result.replay.failed_attempts == 0
    assert result.replay.fees == 210.4
    assert result.book.balance == 50_600


def test_unfunded_wallet_does_not_execute_recorded_positions():
    result = run(round_trip(FIRST, 1500), config=config(initial_wallet=100))
    assert result.replay.status == "INSUFFICIENT_WALLET"
    assert result.replay.attempts == 0 and result.replay.fees == 0
    assert result.skipped_fills == 2 and result.book is None


def test_explicit_marks_allow_fill_prices_to_differ():
    events = [Marks(at(), (("X", 10_000),)),
              Fill(at(), "X", 1, 10_010, 1, seq=1),
              Marks(at(minute=1), (("X", 10_020),)),
              Fill(at(minute=1), "X", -1, 10_019, 1, seq=1)]
    result = run(events, mark_fills=False)
    assert result.trace[0].equity == 49_989
    assert result.book.balance == 50_007


def test_entry_fee_can_breach_and_liquidation_is_charged_once():
    result = run([Fill(at(), "X", 1, 10_000, 2000)])
    failure, = [e for e in result.replay.events if e.kind == "failure"]
    assert failure.balance == 47_998 and result.replay.failed_attempts == 1


def test_funded_mark_breach_preserves_approved_receipt():
    day8 = dates(8)[7]
    events = winning_path() + [Fill(at(day8), "X", 1, 10_000),
                              Marks(at(day8, minute=1), (("X", 9600),)),
                              Fill(at(day8, minute=2), "X", -1, 11_000)]
    result = run(events, sessions=dates(9), config=config(receipt_delay=timedelta(days=1)))
    assert result.replay.failed_attempts == 1 and result.replay.receipts == 450
    failure, = [e for e in result.replay.events if e.kind == "failure"]
    assert failure.balance == 50_098
    assert result.skipped_fills == 1


def test_randomized_equivalent_brackets_preserve_lifecycle_cash():
    from random import Random
    rng = Random(21)
    sessions = dates(100)
    profits = [1500, 1500] + [rng.choice((-100, 200, 500, 1000)) for _ in sessions[2:]]
    events = [e for day, pnl in zip(sessions, profits) for e in round_trip(day, pnl)]
    brackets = BracketHistory(tuple(
        BracketTrade(at(day), at(day, minute=5), day, 100, max(100, pnl), pnl > 0)
        for day, pnl in zip(sessions, profits)))
    baseline = Engine().backtest(spec(), brackets, DollarPolicy.constant(100), config())
    result = run(events, sessions=sessions)
    kinds = {"evaluation_pass", "request", "approval", "receipt", "fee", "live_handoff"}
    assert [e for e in result.replay.events if e.kind in kinds] == [
        e for e in baseline.events if e.kind in kinds]
    assert result.replay.net_cash == baseline.net_cash
    assert result.replay.final_balance == baseline.final_balance
    assert result.replay.failed_attempts == baseline.failed_attempts


@pytest.mark.parametrize("first,last,expiry", [
    (date(2026, 9, 1), date(2026, 10, 1), date(2026, 10, 1)),
    (date(2026, 10, 9), date(2026, 11, 9), date(2026, 11, 8)),
])
def test_calendar_inactivity_uses_local_cutoff_including_weekends_and_dst(first, last, expiry):
    profile = replay_50k(eval_fee=105.2, reset_fee=105, contract_type="mini")
    result = run(round_trip(first, 1), sessions=(first, last), spec=profile)
    failure, = [e for e in result.replay.events if e.kind == "failure"]
    assert failure.at == at(expiry, 16, 15)
    assert failure.code == "FAIL_INACTIVITY"


def test_calendar_inactivity_exact_cutoff_qualifying_close_wins():
    last = date(2026, 10, 1)
    profile = replay_50k(eval_fee=105.2, reset_fee=105, contract_type="mini")
    events = round_trip(FIRST, 1) + [Fill(at(last, 16, 14), "X", 1, 10_000),
                                   Fill(at(last, 16, 15), "X", -1, 10_001)]
    result = run(events, sessions=(FIRST, last), spec=profile)
    assert result.replay.failed_attempts == 0
