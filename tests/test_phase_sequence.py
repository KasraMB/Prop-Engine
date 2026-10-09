from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from propfirm_engine import (
    Account, BacktestConfig, Engine, Fill, Instrument, InvalidAccountError,
    LifecycleSpec, MinimumWinningDaysRule, Phase, ProfitTargetRule, StaticDrawdownRule,
    validate,
)
from propfirm_engine.firms.lucidflex import replay_50k


AT = datetime(2026, 9, 1, 14, tzinfo=timezone.utc)
DAY = AT.date()


def spec():
    base = replay_50k(eval_fee=100, reset_fee=80, contract_type="mini")
    first = Phase("first", "eval", (StaticDrawdownRule(2000), ProfitTargetRule(100)))
    second = Phase("second", "eval", (StaticDrawdownRule(1000), ProfitTargetRule(200)))
    funded = replace(base.account.phases[1], name="funded")
    return replace(base, account=replace(base.account, phases=(first, second, funded), activation_fee=20))


def events(profits):
    return [event for i, pnl in enumerate(profits) for event in (
        Fill(AT+timedelta(minutes=2*i), "X", 1, 10_000),
        Fill(AT+timedelta(minutes=2*i+1), "X", -1, 10_000+pnl))]


def run(stream, **kwargs):
    profile = kwargs.pop("profile", spec())
    sessions = kwargs.pop("sessions", (DAY,))
    return Engine().replay_events(profile, stream, [Instrument("X", 1, 1)],
        BacktestConfig(0, timedelta(0), timedelta(0), timedelta(0)), sessions=sessions,
        fidelity="observed_marks", mark_fills=True, max_mark_age=timedelta(minutes=10),
        liquidation_fee=0, **kwargs)


def test_multiple_evaluations_reset_balances_and_charge_activation_only_at_funding():
    result = run(events([100, 200, 50]))
    starts = [e for e in result.replay.events if e.kind == "phase_start"]
    assert [e.phase_name for e in starts] == ["first", "second", "funded"]
    assert [e.balance for e in starts] == [50_000]*3
    assert result.replay.attempts == 1 and result.replay.fees == 120
    assert result.book.balance == 50_050


def test_failure_in_second_stage_restarts_from_first_on_next_session():
    stream = events([100, -1000])
    stream += [replace(e, at=e.at+timedelta(days=1)) for e in events([10])]
    result = run(stream, sessions=(DAY, DAY+timedelta(days=1)))
    starts = [e.phase_name for e in result.replay.events if e.kind == "phase_start"]
    assert starts == ["first", "second", "first"]
    assert result.replay.attempts == 2 and result.replay.fees == 180
    assert result.book.balance == 50_010


def test_phase_delay_skips_complete_portfolio_without_creating_a_phantom_exit():
    result = run(events([100, 200, 200, 50]), transition_delays={"second": timedelta(minutes=3)})
    assert result.skipped_fills == 2 and result.book.balance == 50_050
    assert result.replay.fees == 120


def test_eval_only_sequence_stops_after_its_last_gate():
    base = spec()
    result = run(events([100, 200, 999]), profile=replace(base, account=replace(base.account, phases=base.account.phases[:2])))
    assert result.replay.status == "EVALUATION_PASSED" and result.skipped_fills == 2
    assert [e.phase_name for e in result.replay.events if e.kind == "evaluation_pass"] == ["first", "second"]


def test_summary_validation_still_rejects_multiple_evaluation_phases():
    with pytest.raises(InvalidAccountError, match="multiple eval"):
        validate(spec().account)
    validate(spec().account, sequence=True)


def test_optional_retry_policy_can_stop_after_failure():
    stream = events([-2000]) + [replace(e, at=e.at+timedelta(days=1)) for e in events([100])]
    result = run(stream, sessions=(DAY, DAY+timedelta(days=1)), retry_on_failure=False)
    assert result.replay.attempts == 1 and result.replay.status == "ACCOUNT_FAILED"
    assert result.skipped_fills == 2


def test_phase_limit_is_enforced_after_transition():
    stream = events([100]) + [Fill(AT+timedelta(minutes=2), "X", 2, 10_000)]
    with pytest.raises(ValueError, match="contract limit"):
        run(stream, phase_limits={"second": 1})
