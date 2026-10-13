from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from statistics import mean, stdev
from zoneinfo import ZoneInfo

import pytest

from propfirm_engine import Engine, TradeHistory, DollarPolicy, Window
from propfirm_engine import windows as api
from test_trade_log import record, profile, policy
from test_chronological_backtest import config, session_day


NY = ZoneInfo("America/New_York")


def clock(day, hour=16, minute=45):
    return datetime.combine(day, datetime.min.time(), NY)+timedelta(hours=hour, minutes=minute)


def prepared(rows, *, days=None, start=None, end=None, definitions=None, spec=None):
    days = tuple(days or sorted({t.session for t in rows}))
    return Engine().prepare_windows(spec or profile(), TradeHistory(tuple(rows)) if rows else None,
        sessions=days, closes={d: clock(d) for d in days},
        windows=definitions or (Window(start or clock(days[0], 9, 30), end or clock(days[-1])),))


def test_public_calendar_sizes_and_pays_without_research_wrapper():
    rows = [record(0, 375), record(1, 375)] + [record(i, 100) for i in range(2, 7)]
    p = prepared(rows)
    batch = Engine().replay_windows(p, policy(), config(), record_events=True)
    w, = batch.windows
    assert w.receipts == 450 and w.net_cash == 344.8
    assert [e.quantity for e in w.events if e.kind == "trade"] == [4, 4, 2, 2, 2, 2, 2]
    assert batch.visited_regimes == ("evaluation", "funded")
    assert batch.summary["distributions"]["net_cash_per_day"]["mean"] == w.net_cash_per_day
    assert batch.uncertainty["sample_kind"] == "historical_windows"


def test_empty_window_purchases_at_start_and_processes_inactivity():
    days = tuple(session_day(i) for i in range(30))
    p = prepared([], days=days)
    w, = Engine().replay_windows(p, policy(), config(), record_events=True).windows
    assert w.fees == 105.2 and w.attempts == 1
    assert w.executed_trades == 0 and w.failed_attempts == 1
    assert next(e for e in w.events if e.kind == "fee").at == p.windows[0].window.start


def test_pending_cash_arrives_on_no_trade_days_across_dst():
    days = (date(2021, 3, 4), date(2021, 3, 5), date(2021, 3, 8), date(2021, 3, 9),
            date(2021, 3, 10), date(2021, 3, 11), date(2021, 3, 12))
    rows = []
    for i, day in enumerate(days):
        t = record(i, 375 if i < 2 else 100)
        shift = clock(day, 10, 0).astimezone(timezone.utc)-t.entry_at
        rows.append(replace(t, session=day, entry_at=t.entry_at+shift, exit_at=t.exit_at+shift,
                            marks=tuple(replace(m, at=m.at+shift) for m in t.marks)))
    receipt = clock(days[-1]).astimezone(timezone.utc)+timedelta(days=3)
    before = prepared(rows, days=days, end=receipt-timedelta(seconds=1))
    after = prepared(rows, days=days, end=receipt)
    cfg = config(approval_delay=timedelta(days=1), receipt_delay=timedelta(days=2))
    a = Engine().replay_windows(before, policy(), cfg).windows[0]
    b = Engine().replay_windows(after, policy(), cfg).windows[0]
    assert a.receipts == 0 and a.outstanding_payouts == 450
    assert b.receipts == 450 and b.outstanding_payouts == 0


def test_exclusions_remain_visible_and_do_not_become_zero_cash():
    days = (session_day(0), session_day(1))
    definitions = tuple(Window(clock(d, 9), clock(d), "unresolved" if i else None)
                        for i, d in enumerate(days))
    batch = Engine().replay_windows(prepared([], days=days, definitions=definitions), policy(), config())
    assert len(batch.windows) == len(batch.excluded) == 1
    assert batch.summary["distributions"]["net_cash"]["mean"] == -105.2


@pytest.mark.parametrize("which", ["start", "end"])
def test_open_trade_at_window_boundary_is_rejected(which):
    t = record(0)
    with pytest.raises(ValueError, match="open trade"):
        prepared([t], **{which: t.entry_at+timedelta(minutes=2)})


def test_split_must_separate_complete_horizons_not_just_trade_dates():
    train = prepared([record(0)], end=clock(session_day(4)))
    test = prepared([record(3)])
    with pytest.raises(ValueError, match="before holdout"):
        Engine().fit_windows(train, test, config(), policy=policy(), risk_bounds={})


def test_only_scored_windows_determine_visited_regimes(monkeypatch):
    rows = [record(0, 375), record(1, 375), record(2, 10)]
    train, test = prepared(rows), prepared([record(4)])
    class Search:
        def __init__(self, *args, **kwargs): pass
        def optimize(self, objective): objective([.5, 1.])
    monkeypatch.setattr(api, "CMAES", Search)
    def forbidden(*args, **kwargs):
        raise AssertionError("unrelated chronological replay")
    monkeypatch.setattr("propfirm_engine.fitting.backtest", forbidden)
    initial = DollarPolicy((policy().regimes[0], replace(policy().regimes[1], risk_dollars=50)))
    fit = Engine().fit_windows(train, test, config(), policy=initial,
        risk_bounds={"evaluation": (100, 300), "funded": (50, 300)},
        objective=lambda b: sum(w.final_balance or 0 for w in b.windows))
    assert fit.policy.regimes[1].risk_dollars == 300
    assert "funded" in fit.in_sample.visited_regimes


@pytest.mark.parametrize("direction, expected", [("maximize", 300), ("minimize", 100)])
def test_direction_and_custom_policy_objective_survive_restoration(monkeypatch, direction, expected):
    class Search:
        def __init__(self, *args, **kwargs): pass
        def optimize(self, objective): objective([.5, 1.])
    monkeypatch.setattr(api, "CMAES", Search)
    fit = Engine().fit_windows(prepared([record(0)]), prepared([record(3)]), config(), policy=policy(),
        risk_bounds={"evaluation": (100, 300), "funded": (50, 300)}, direction=direction,
        objective=lambda b: b.policy.regimes[1].risk_dollars)
    assert fit.policy.regimes[1].risk_dollars == expected


def test_holdout_outcomes_cannot_change_search_or_policy():
    train = prepared([record(0, 375), record(1, 375), record(2, 100)])
    kwargs = dict(policy=policy(), risk_bounds={"evaluation": (100, 300), "funded": (50, 300)},
                  generations=2, population=4, seed=5)
    a = Engine().fit_windows(train, prepared([record(4, 100)]), config(), **kwargs)
    b = Engine().fit_windows(train, prepared([record(4, -1000)]), config(), **kwargs)
    assert a.policy == b.policy and a.in_sample == b.in_sample
    assert a.evaluations == b.evaluations


def test_prepared_replay_is_repeatable_and_only_retains_events_on_request():
    p = prepared([record(0), record(1)])
    a = Engine().replay_windows(p, policy(), config())
    b = Engine().replay_windows(p, policy(), config(), record_events=True)
    assert not a.windows[0].events and b.windows[0].events
    assert a == Engine().replay_windows(p, policy(), config())
    assert a.windows[0].history_fingerprint == b.windows[0].history_fingerprint


def test_legacy_custom_objective_cannot_be_worsened_by_restoration():
    rows = TradeHistory(tuple(record(i) for i in range(10)))
    seen = []
    def score(result):
        value = result.policy.regimes[1].risk_dollars
        seen.append(value)
        return value
    fit = Engine().fit(profile(), rows, config(), policy=policy(),
        risk_bounds={"evaluation": (100, 300), "funded": (50, 300)},
        objective=score, generations=2, population=4, seed=7)
    assert fit.score == max(seen)


def test_bankroll_report_uses_unrestricted_counterpart():
    from propfirm_engine import RiskConfig
    p = prepared([record(0)])
    batch = Engine().replay_windows(p, policy(), config(initial_wallet=100),
                                   risk=RiskConfig(bankroll=100))
    w, = batch.windows
    assert w.attempts == 0 and w.net_cash == 0
    assert w.required_bankroll == 105.2
    assert batch.risk is not None


def test_no_trade_calendar_day_can_deliver_receipt_before_next_signal():
    rows = [record(0, 375), record(1, 375)] + [record(i, 100) for i in range(2, 7)]
    p = prepared(rows, days=tuple(session_day(i) for i in range(10)))
    batch = Engine().replay_windows(p, policy(), config(receipt_delay=timedelta(days=1)), record_events=True)
    receipt = next(e for e in batch.windows[0].events if e.kind == "receipt")
    assert receipt.at == clock(session_day(6)).astimezone(timezone.utc)+timedelta(days=1)
    assert batch.windows[0].receipts == 450


def test_invalid_clocks_and_unresolved_partitions_fail_explicitly():
    t = record(0)
    with pytest.raises(ValueError, match="cutoff"):
        Engine().prepare_windows(profile(), TradeHistory((t,)), sessions=(t.session,),
            closes={t.session: clock(t.session, 23)}, windows=(Window(t.entry_at, clock(t.session, 23)),))
    with pytest.raises(ValueError, match="exactly"):
        Engine().prepare_windows(profile(), TradeHistory((t,)), sessions=(t.session,), closes={},
                                 windows=(Window(t.entry_at, t.exit_at),))
    excluded = prepared([t], definitions=(Window(t.entry_at, t.exit_at, "missing"),))
    with pytest.raises(ValueError, match="observed window"):
        Engine().fit_windows(excluded, prepared([record(4)]), config(), policy=policy(), risk_bounds={})


def test_early_close_is_used_and_later_trade_is_rejected():
    t = record(0)
    args = dict(sessions=(t.session,), closes={t.session: clock(t.session, 13, 0)},
                windows=(Window(clock(t.session, 9, 30), clock(t.session)),))
    p = Engine().prepare_windows(profile(), TradeHistory((t,)), **args)
    w, = Engine().replay_windows(p, policy(), config(), record_events=True).windows
    assert next(e.at for e in w.events if e.kind == "session_close") == clock(t.session, 13, 0)
    shift = timedelta(hours=4)
    late = replace(t, entry_at=t.entry_at+shift, exit_at=t.exit_at+shift,
                   marks=tuple(replace(m, at=m.at+shift) for m in t.marks))
    with pytest.raises(ValueError, match="calendar"):
        Engine().prepare_windows(profile(), TradeHistory((late,)), **args)


def test_progress_reports_scored_batches_only_and_holdout_is_last():
    events = []
    Engine().fit_windows(prepared([record(0)]), prepared([record(4)]), config(), policy=policy(),
        risk_bounds={"evaluation": (100, 300), "funded": (50, 300)}, generations=0, progress=events.append)
    assert events[-1]["stage"] == "holdout"
    for event in events[:-1]:
        assert event["result"].windows[0].end == clock(session_day(0))
        assert event["result"].policy == event["policy"]


def test_window_guide_runs():
    from pathlib import Path
    import re
    text = (Path(__file__).parents[1] / "docs/WINDOWS.md").read_text(encoding="utf-8")
    for code in re.findall(r"```python\n(.*?)\n```", text, re.DOTALL):
        exec(compile(code, "WINDOWS.md", "exec"), {})
