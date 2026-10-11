from dataclasses import replace
from datetime import timedelta

import pytest

from propfirm_engine import (
    DollarPolicy, Engine, PnlMark, Trade, TradeHistory, PhaseSearch, RollingConfig,
    Fill, Instrument, Marks,
)
from propfirm_engine.firms.lucidflex import replay_50k
from propfirm_engine.ruin import _resampled_history
from test_chronological_backtest import config, session_day, trade


def record(i, pnl=30, risk=50, middle=()):
    t = trade(session_day(i))
    return Trade(t.entry_at, t.exit_at, t.session, risk, pnl,
                 (PnlMark(t.entry_at, 0),) + tuple(
                     PnlMark(t.entry_at+timedelta(minutes=j+1), value) for j, value in enumerate(middle)
                 ) + (PnlMark(t.exit_at, pnl),))


def profile():
    return replay_50k(eval_fee=105.2, reset_fee=105, contract_type="mini", elapsed_inactivity=True)


def policy():
    base = DollarPolicy.constant(200)
    return DollarPolicy((base.regimes[0], replace(base.regimes[1], risk_dollars=100)))


def replay(rows, **kwargs):
    return Engine().backtest(profile(), TradeHistory(tuple(rows)), kwargs.pop("policy", policy()), config(**kwargs))


def test_timed_exit_sizes_from_actual_phase_before_entry():
    result = replay([record(0, 375), record(1, 375), record(2, 30)])
    trades = [e for e in result.events if e.kind == "trade"]
    assert [e.quantity for e in trades] == [4, 4, 2]
    assert [e.phase for e in trades] == ["eval", "eval", "funded"]
    assert [e.balance for e in trades] == [51500, 53000, 50060]
    assert result.net_cash == -105.2


@pytest.mark.parametrize("pnl", [30, 0, -20, -80])
def test_arbitrary_profit_breakeven_partial_loss_and_stop_gap(pnl):
    result = replay([record(0, pnl)])
    assert result.final_balance == 50000+4*pnl


def test_fees_are_in_sizing_and_deducted_exactly_once():
    result = replay([record(0)], cost_per_contract=2, cost_per_trade=1)
    fill = next(e for e in result.events if e.kind == "trade")
    assert fill.quantity == 3  # floor((200-1)/(50+2))
    assert fill.balance == 50000+90-7


def test_observed_breach_cannot_recover_on_later_profit():
    result = replay([record(0, 30, middle=(-500, 100))])
    breach = next(e for e in result.events if e.kind == "failure")
    liquidation = next(e for e in result.events if e.kind == "liquidation")
    assert breach.at == record(0).entry_at+timedelta(minutes=1)
    assert breach.balance == liquidation.balance == 48000
    assert result.failed_attempts == 1
    assert not any(e.kind == "trade" for e in result.events)


def test_fees_participate_in_observed_breach():
    result = replay([record(0, 30, middle=(-666,))], cost_per_contract=2)
    breach = next(e for e in result.events if e.kind == "failure")
    assert breach.balance == 50000-3*666-6


def test_unrealized_target_touch_cannot_pass_or_ratchet_eod_floor():
    result = replay([record(0, 30, middle=(1000,))])
    assert not any(e.kind == "evaluation_pass" for e in result.events)
    assert result.final_balance == 50120
    assert result.events[-1].floor == 48120


def test_future_risk_and_path_cannot_change_entry_size():
    a = replay([record(0, 30, middle=(-20,))])
    b = replay([record(0, 100, middle=(500,))])
    assert next(e.quantity for e in a.events if e.kind == "trade") == 4
    assert next(e.quantity for e in b.events if e.kind == "trade") == 4


@pytest.mark.parametrize("direction", [1, -1])
def test_signed_pnl_matches_long_and_short_recorded_fill_accounting(direction):
    t = record(0, 30, middle=(-20,))
    result = replay([t], policy=DollarPolicy.constant(100), cost_per_contract=2)
    events = [Fill(t.entry_at, "X", direction, 1000, 2),
              Marks(t.marks[1].at, (("X", 1000+direction*(-20)),)),
              Fill(t.exit_at, "X", -direction, 1000+direction*30)]
    fixed = Engine().replay_events(profile(), events, [Instrument("X", 1, 1)], config(),
        sessions=(t.session,), fidelity="observed_marks", mark_fills=True,
        max_mark_age=timedelta(minutes=10), liquidation_fee=0)
    assert result.final_balance == float(fixed.book.balance) == 50028


def test_record_import_and_split_preserve_the_trade_contract():
    rows = [record(i) for i in range(10)]
    dictionaries = [dict(entry_at=t.entry_at.isoformat(), exit_at=t.exit_at.isoformat(),
        session=str(t.session), risk="50", pnl="30",
        marks=[dict(at=m.at.isoformat(), pnl=str(m.pnl)) for m in t.marks]) for t in rows]
    history = TradeHistory.from_records(dictionaries)
    assert history == TradeHistory(tuple(rows))
    a, b = history.split()
    assert type(a) is type(b) is TradeHistory
    assert len(a.sessions) == 7 and len(b.sessions) == 3


def test_rolling_joint_separate_and_comparison_use_same_log_contract():
    history = TradeHistory(tuple(record(i, 375 if i % 3 else -20) for i in range(30)))
    engine, base = Engine(), policy()
    bounds = {r.name: (50, 300) for r in base.regimes}
    normal = engine.fit(profile(), history, config(), policy=base, risk_bounds=bounds, generations=1, population=4)
    assert normal.out_of_sample == engine.backtest(profile(), history.split()[1], normal.policy, config())
    rolling = engine.rolling_backtest(profile(), history, base, config(), rolling=RollingConfig(3))
    assert len(rolling.windows) == 28
    compared = engine.compare_searches(profile(), history, config(), policy=base, risk_bounds=bounds,
        search=PhaseSearch(trade_budget=1000, horizon_sessions=3, stride_sessions=3, archive_size=4, population=4))
    for result in compared.values():
        assert result.out_of_sample == engine.backtest(profile(), history.split()[1], result.policy, config())


def test_resampling_moves_the_entire_observation_path():
    history = TradeHistory(tuple(record(i, middle=(-10,)) for i in range(3)))
    resampled = _resampled_history(profile(), history, [2, 0, 1])
    assert type(resampled) is TradeHistory
    for old, new in zip((history.trades[2], history.trades[0], history.trades[1]), resampled.trades):
        assert [m.at-new.entry_at for m in new.marks] == [m.at-old.entry_at for m in old.marks]
        assert [m.pnl for m in new.marks] == [m.pnl for m in old.marks]


@pytest.mark.parametrize("changes", [dict(risk=0), dict(risk=float("nan")), dict(pnl=float("inf")),
    dict(marks=()), dict(marks=(PnlMark(record(0).entry_at, 0),)),
    dict(marks=tuple(reversed(record(0).marks))), dict(pnl=31)])
def test_bad_or_missing_observations_fail_explicitly(changes):
    with pytest.raises(ValueError):
        replace(record(0), **changes)


def test_capability_check_exposes_sized_arbitrary_exits():
    support = Engine().check_replay(profile(), adapter="trades", features=("sizing", "arbitrary_exits"))
    assert support.fidelity == "observed_marks"


def test_winning_days_use_actual_net_closes_not_open_profit_or_risk():
    result = replay([record(0, 375), record(1, 375)] +
                    [record(i, 30, middle=(500,)) for i in range(2, 7)])
    assert result.final_balance == 50300
    assert result.events[-1].qualifying_days == 0
    assert result.receipts == 0


def test_arbitrary_exits_reach_payout_through_the_shared_ledger():
    result = replay([record(0, 375), record(1, 375)] +
                    [record(i, 100, middle=(-10,)) for i in range(2, 7)])
    request = next(e for e in result.events if e.kind == "request")
    assert request.balance == 51000
    assert request.gross_payout == 500
    assert result.receipts == 450
    assert result.net_cash == 344.8


def test_dense_marks_do_not_expand_the_retained_event_log():
    t = record(0)
    marks = tuple(PnlMark(t.entry_at + timedelta(milliseconds=i), -10)
                  for i in range(1, 1001))
    dense = replace(t, marks=(t.marks[0],) + marks + (t.marks[-1],))
    a, b = replay([t]), replay([dense])
    assert a.events == b.events
    assert a.history_fingerprint != b.history_fingerprint


def test_ruin_runs_sized_observed_logs():
    from propfirm_engine import RuinConfig
    history = TradeHistory(tuple(record(i, middle=(-10,)) for i in range(3)))
    result = Engine().ruin(profile(), history, policy(), config(),
        simulation=RuinConfig(paths=2, sessions=3, cycle_paths=2, cycle_steps=2))
    assert result["source_fingerprint"] == history.fingerprint
    assert result["source_trades"] == 3


@pytest.mark.parametrize("method", ["fit", "fit_phases"])
def test_oos_outcomes_and_marks_cannot_change_selected_policy(method):
    rows = tuple(record(i, 375 if i % 3 else -20) for i in range(20))
    history = TradeHistory(rows)
    changed = TradeHistory(rows[:14] + tuple(record(i, -900, middle=(-1000,)) for i in range(14, 20)))
    kwargs = dict(policy=policy(), risk_bounds={r.name: (50, 300) for r in policy().regimes})
    if method == "fit":
        kwargs.update(generations=1, population=4)
    else:
        kwargs["search"] = PhaseSearch(trade_budget=500, horizon_sessions=3,
                                      stride_sessions=3, archive_size=4, population=4)
    fit = getattr(Engine(), method)
    a = fit(profile(), history, config(), **kwargs)
    b = fit(profile(), changed, config(), **kwargs)
    assert a.policy == b.policy


def test_trade_guide_runs():
    from pathlib import Path
    import re
    text = (Path(__file__).parents[1] / "docs/TRADE_LOG.md").read_text(encoding="utf-8")
    for code in re.findall(r"```python\n(.*?)\n```", text, re.DOTALL):
        exec(compile(code, "TRADE_LOG.md", "exec"), {})
