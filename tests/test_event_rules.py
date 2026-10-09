from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from propfirm_engine import (
    Account, BacktestConfig, ConsistencyGateRule, DailyLossRule, Engine, Fill,
    Instrument, LifecycleSpec, Marks, MinimumWinningDaysRule, Phase, ProfitTargetRule,
    Severity, StateField, StaticDrawdownRule, Timing, TrailingDrawdownRule,
)
from propfirm_engine.firms.lucidflex import replay_50k


AT = datetime(2026, 9, 1, 14, tzinfo=timezone.utc)
DAY = date(2026, 9, 1)
X, Y = Instrument("X", 1, 1), Instrument("Y", 1, 1)


def run(events, rules, *, sessions=(DAY,), **kwargs):
    spec = LifecycleSpec(Account("test", 50_000, (Phase("eval", "eval", tuple(rules)),)),
                         4, ((float("-inf"), 4),), 0)
    return Engine().replay_events(spec, events, [X, Y],
        BacktestConfig(0, timedelta(0), timedelta(0), timedelta(0)),
        sessions=sessions, fidelity="observed_marks", mark_fills=True,
        max_mark_age=timedelta(minutes=10), liquidation_fee=0, trace=True, **kwargs)


def test_intraday_trailing_ratchets_on_unrealized_peak_before_later_loss():
    events = [Fill(AT, "X", 1, 10_000), Marks(AT+timedelta(minutes=1), (("X", 11_000),)),
              Marks(AT+timedelta(minutes=2), (("X", 9000),)),
              Fill(AT+timedelta(minutes=3), "X", -1, 12_000)]
    result = run(events, [TrailingDrawdownRule(2000), ProfitTargetRule(10_000)])
    failure, = [e for e in result.replay.events if e.kind == "failure"]
    assert failure.floor == 49_000 and failure.balance == 49_000
    assert result.replay.failed_attempts == 1 and result.skipped_fills == 1


def test_eod_trailing_does_not_ratchet_on_an_intraday_peak():
    events = [Fill(AT, "X", 1, 10_000), Marks(AT+timedelta(minutes=1), (("X", 11_000),)),
              Fill(AT+timedelta(minutes=2), "X", -1, 9000)]
    result = run(events, [TrailingDrawdownRule(2000, update_timing=Timing.EOD), ProfitTargetRule(10_000)])
    assert result.replay.failed_attempts == 0 and result.book.balance == 49_000


@pytest.mark.parametrize("timing,failed", [(Timing.CONTINUOUS, 1), (Timing.EOD, 0)])
def test_static_check_timing_controls_whether_recovery_is_possible(timing, failed):
    events = [Fill(AT, "X", 1, 10_000), Marks(AT+timedelta(minutes=1), (("X", 8000),)),
              Fill(AT+timedelta(minutes=2), "X", -1, 10_000)]
    result = run(events, [StaticDrawdownRule(2000, check_timing=timing), ProfitTargetRule(3000)])
    assert result.replay.failed_attempts == failed
    if failed:
        assert next(e for e in result.replay.events if e.kind == "failure").code == "FAIL_STATIC_DD"


def test_daily_loss_suspends_until_next_session_without_counting_a_failure():
    tomorrow = DAY + timedelta(days=1)
    events = [Fill(AT, "X", 1, 10_000), Marks(AT+timedelta(minutes=1), (("X", 9500),)),
              Fill(AT+timedelta(minutes=2), "X", -1, 11_000),
              Fill(AT+timedelta(days=1), "X", 1, 10_000),
              Fill(AT+timedelta(days=1, minutes=1), "X", -1, 10_100)]
    result = run(events, [StaticDrawdownRule(2000), DailyLossRule(500), ProfitTargetRule(3000)],
                 sessions=(DAY, tomorrow))
    assert result.replay.attempts == 1 and result.replay.failed_attempts == 0
    assert result.book.balance == 49_600 and result.skipped_fills == 1
    assert len([e for e in result.replay.events if e.kind == "daily_suspend"]) == 1


def test_hard_daily_loss_is_a_failure_not_a_suspension():
    result = run([Fill(AT, "X", 1, 10_000), Marks(AT+timedelta(minutes=1), (("X", 9500),))],
                 [StaticDrawdownRule(2000), DailyLossRule(500, severity=Severity.HARD), ProfitTargetRule(3000)])
    assert result.replay.failed_attempts == 1
    assert next(e for e in result.replay.events if e.kind == "failure").code == "FAIL_DAILY_LOSS"


def test_hard_drawdown_precedes_soft_daily_loss_at_same_observation():
    result = run([Fill(AT, "X", 1, 10_000), Marks(AT+timedelta(minutes=1), (("X", 8000),))],
                 [DailyLossRule(500), StaticDrawdownRule(2000), ProfitTargetRule(3000)])
    assert next(e for e in result.replay.events if e.kind == "failure").code == "FAIL_STATIC_DD"
    assert not any(e.kind == "daily_suspend" for e in result.replay.events)


def test_funded_closed_loss_with_offsetting_open_profit_does_not_breach_ledger():
    base = replay_50k(eval_fee=0, reset_fee=0, contract_type="mini")
    funded = base.account.phases[1]
    spec = replace(base, account=replace(base.account, phases=(funded,)))
    events = [Fill(AT, "X", 1, 10_000), Fill(AT, "Y", 1, 10_000, seq=1),
              Marks(AT+timedelta(minutes=1), (("X", 7000), ("Y", 13_000))),
              Fill(AT+timedelta(minutes=2), "X", -1, 7000),
              Fill(AT+timedelta(minutes=2), "Y", -1, 13_000, seq=1)]
    result = Engine().replay_events(spec, events, [X, Y],
        BacktestConfig(0, timedelta(0), timedelta(0), timedelta(0)), sessions=(DAY,),
        fidelity="observed_marks", mark_fills=True, max_mark_age=timedelta(minutes=10), liquidation_fee=0)
    assert result.book.balance == 50_000 and result.replay.failed_attempts == 0


def test_funded_consistency_blocks_payout_and_resets_with_the_cycle():
    base = replay_50k(eval_fee=0, reset_fee=0, contract_type="mini")
    funded = base.account.phases[1]
    schema = replace(funded.payout_schema, reset_fields=(StateField.N_QUALIFYING_DAYS, StateField.MAX_DAY_PNL))
    rules = tuple(r for r in funded.rules if not isinstance(r, MinimumWinningDaysRule))
    funded = replace(funded, rules=rules + (MinimumWinningDaysRule(2, 150), ConsistencyGateRule(.5)),
                     payout_schema=schema)
    spec = replace(base, account=replace(base.account, phases=(funded,)))
    days = (DAY, DAY+timedelta(days=1), DAY+timedelta(days=2), DAY+timedelta(days=3), DAY+timedelta(days=6))
    events = []
    for day, profit in zip(days, (1500, 150, 1350, 200, 200)):
        at = datetime.combine(day, AT.timetz())
        events.extend((Fill(at, "X", 1, 10_000), Fill(at+timedelta(minutes=1), "X", -1, 10_000+profit)))
    result = Engine().replay_events(spec, events, [X],
        BacktestConfig(0, timedelta(0), timedelta(0), timedelta(0)), sessions=days,
        fidelity="observed_marks", mark_fills=True, max_mark_age=timedelta(minutes=10), liquidation_fee=0)
    requests = [e for e in result.replay.events if e.kind == "request"]
    assert [e.gross_payout for e in requests] == [1500, 950]
    assert requests[0].at.date() == days[2]
    assert result.replay.receipts == 2205 and result.book.balance == 50_950


def test_static_profile_honors_explicit_request_time_floor_lock():
    base = replay_50k(eval_fee=0, reset_fee=0, contract_type="mini")
    funded = base.account.phases[1]
    funded = replace(funded, rules=(StaticDrawdownRule(2000), MinimumWinningDaysRule(1, 150)))
    profile = replace(base, account=replace(base.account, phases=(funded,)))
    events = [Fill(AT, "X", 1, 10_000), Fill(AT+timedelta(minutes=1), "X", -1, 11_000),
              Fill(AT+timedelta(days=1), "X", 1, 10_000),
              Marks(AT+timedelta(days=1, minutes=1), (("X", 9600),))]
    result = Engine().replay_events(profile, events, [X],
        BacktestConfig(0, timedelta(0), timedelta(0), timedelta(0)), sessions=(DAY, DAY+timedelta(days=1)),
        fidelity="observed_marks", mark_fills=True, max_mark_age=timedelta(minutes=10), liquidation_fee=0)
    assert result.replay.receipts == 450 and result.replay.failed_attempts == 1
    failure, = [e for e in result.replay.events if e.kind == "failure"]
    assert failure.floor == 50_100 and failure.code == "FAIL_STATIC_DD"
