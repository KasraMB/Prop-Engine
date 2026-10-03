"""Historical-start windows: independent cash examples, canonical parity and leakage."""
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from propfirm_engine import BacktestConfig, BracketHistory, BracketTrade, DollarPolicy, Engine, RollingConfig
from propfirm_engine.firms.lucidflex import replay_50k
from propfirm_engine.rolling import window_slices


def inputs(days=12, trades_per_day=1):
    day, trades = date(2026, 9, 1), []
    for i in range(days):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        for j in range(trades_per_day):
            at = datetime.combine(day, time(10 + j), ZoneInfo("America/New_York"))
            trades.append(BracketTrade(at, at + timedelta(minutes=5), day, 100,
                                       1500 if i < 2 else 200, True))
        day += timedelta(days=1)
    return (replay_50k(eval_fee=105.2, reset_fee=105, contract_type="micro"),
            BracketHistory(tuple(trades)), DollarPolicy.constant(100),
            BacktestConfig(cost_per_contract=0, approval_delay=timedelta(0),
                           receipt_delay=timedelta(0), activation_delay=timedelta(minutes=30)))


def test_whole_session_windows_preserve_exact_trades_and_match_direct_engine():
    spec, h, policy, config = inputs(6, 2)
    rolling = RollingConfig(3, 2)
    assert window_slices(h, rolling) == ((0, 6), (4, 10))
    result = Engine().rolling_backtest(spec, h, policy, config, rolling=rolling)
    assert result.excluded_incomplete_starts == 1
    for w, (start, end) in zip(result.windows, window_slices(h, rolling)):
        part = BracketHistory(h.trades[start:end])
        direct = Engine().backtest(spec, part, policy, config)
        assert w.history_fingerprint == direct.history_fingerprint
        assert w.input_trades == 6
        assert w.first_session == part.sessions[0] and w.last_session == part.sessions[-1]
        assert w.net_cash == direct.net_cash and w.fees == direct.fees
        assert w.net_cash_per_day == direct.net_cash_per_day
        assert w.start == direct.start and w.end == direct.end
    assert result.score == pytest.approx(np.mean([w.net_cash_per_day for w in result.windows]))
    assert result.summary["distributions"]["net_cash"]["median"] == np.median([w.net_cash for w in result.windows])


def test_one_window_independent_payout_fees_and_pass_time():
    args = inputs(7)
    result = Engine().rolling_backtest(*args, rolling=RollingConfig(7))
    w = result.windows[0]
    assert w.first_evaluation == "passed"
    assert w.days_to_first_pass == pytest.approx(1 + 5 / (24 * 60))
    assert w.receipts == 450 and w.fees == 105.2 and w.net_cash == pytest.approx(344.8)
    assert w.max_external_cash_drawdown == 105.2
    assert result.summary["first_evaluation_pass_rate"] == 1
    assert result.summary["payout_probability"] == 1
    assert result.excluded_incomplete_starts == 6


@pytest.mark.parametrize("value", [0, -1, True, 1.5, float("nan"), "2"])
@pytest.mark.parametrize("field", ["window_sessions", "stride_sessions"])
def test_window_configuration_rejects_invalid_counts(field, value):
    with pytest.raises(ValueError, match="positive integer"):
        RollingConfig(**{field: value})


def test_incomplete_only_history_rejected_not_counted_as_failure():
    with pytest.raises(ValueError, match="no complete"):
        Engine().rolling_backtest(*inputs(3), rolling=RollingConfig(4))


def test_unresolved_and_unstarted_evaluations_are_separate_from_failures():
    spec, h, policy, config = inputs(3)
    result = Engine().rolling_backtest(spec, h, policy, config, rolling=RollingConfig(1))
    counts = result.summary["first_evaluation_counts"]
    assert counts["unresolved"] == 3 and counts["failed"] == 0
    assert result.summary["first_evaluation_pass_rate"] == 0
    assert result.summary["days_to_first_pass_among_passes"] is None
    unfunded = Engine().rolling_backtest(spec, h, policy, replace(config, initial_wallet=0),
                                         rolling=RollingConfig(1))
    assert unfunded.summary["first_evaluation_counts"]["not_started"] == 3
    assert unfunded.summary["first_evaluation_pass_rate"] is None


def test_first_eval_failure_not_overwritten_by_successful_retry():
    spec, h, _, config = inputs(3)
    h = BracketHistory((replace(h.trades[0], won=False), h.trades[1], replace(h.trades[2], take_profit=1500)))
    result = Engine().rolling_backtest(spec, h, DollarPolicy.constant(2000), config, rolling=RollingConfig(3))
    w = result.windows[0]
    assert w.first_evaluation == "failed" and w.days_to_first_pass is None
    assert w.any_mll_breach and w.attempts == 2 and w.failed_attempts == 1
    assert result.summary["any_mll_breach_probability"] == 1


def test_direct_funded_pass_rate_is_not_applicable():
    spec, h, policy, config = inputs(3)
    spec = replace(spec, account=replace(spec.account, phases=spec.account.phases[1:]))
    result = Engine().rolling_backtest(spec, h, policy, config, rolling=RollingConfig(2))
    assert result.summary["first_evaluation_pass_rate"] is None
    assert result.summary["first_evaluation_counts"]["not_applicable"] == 2


def test_late_receipt_is_outstanding_not_a_received_payout():
    spec, h, policy, config = inputs(7)
    result = Engine().rolling_backtest(spec, h, policy, replace(config, receipt_delay=timedelta(days=2)),
                                       rolling=RollingConfig(7))
    assert result.windows[0].outstanding_payouts == 450
    assert result.summary["payout_probability"] == 0


@pytest.mark.parametrize("bad", [lambda r: float("nan"), lambda r: True, lambda r: "1", 3])
def test_invalid_objectives_fail_closed(bad):
    with pytest.raises(ValueError, match="objective"):
        Engine().rolling_backtest(*inputs(3), rolling=RollingConfig(2), objective=bad)


def test_custom_objective_averages_windows_not_overlapping_profit_totals():
    result = Engine().rolling_backtest(*inputs(8), rolling=RollingConfig(7), objective=lambda r: r.net_cash)
    assert result.score == sum(w.net_cash for w in result.windows) / 2
    assert result.score != sum(w.net_cash for w in result.windows)


def test_finite_large_objectives_do_not_overflow_when_averaged():
    result = Engine().rolling_backtest(*inputs(3), rolling=RollingConfig(2), objective=lambda r: 1e308)
    assert result.score == 1e308


def test_rolling_fit_never_uses_oos_for_selection_and_retains_single_replay():
    spec, h, policy, config = inputs(30)
    h = BracketHistory(tuple(replace(t, take_profit=1500) for t in h.trades))
    kwargs = dict(policy=policy, risk_bounds={"evaluation": (50, 200), "funded": (50, 200)},
                  rolling=RollingConfig(7, 2), generations=1, population=4, seed=9)
    engine = Engine()
    a = engine.fit(spec, h, config, **kwargs)
    changed = BracketHistory(h.trades[:21] + tuple(replace(t, won=False) for t in h.trades[21:]))
    b = engine.fit(spec, changed, config, **kwargs)
    assert a.policy == b.policy and a.in_sample_score == b.in_sample_score
    assert a.in_sample_rolling == b.in_sample_rolling and a.evaluations == b.evaluations
    assert a.score == a.out_of_sample_rolling.score
    assert a.score != b.score
    train, test = h.split()
    assert all(w.last_session < test.sessions[0] for w in a.in_sample_rolling.windows)
    assert all(w.first_session >= test.sessions[0] for w in a.out_of_sample_rolling.windows)
    assert a.out_of_sample.events == engine.backtest(spec, test, a.policy, config).events
    assert a.out_of_sample_rolling == engine.rolling_backtest(spec, test, a.policy, config, rolling=kwargs["rolling"])


def test_minimize_custom_rolling_objective_and_partition_too_short():
    spec, h, policy, config = inputs(20)
    kwargs = dict(policy=policy, risk_bounds={"evaluation": (50, 200), "funded": (50, 200)}, generations=0)
    result = Engine().fit(spec, h, config, rolling=RollingConfig(4), direction="minimize",
                          objective=lambda r: -r.net_cash, **kwargs)
    assert result.score == pytest.approx(-np.mean([w.net_cash for w in result.out_of_sample_rolling.windows]))
    with pytest.raises(ValueError, match="no complete"):
        Engine().fit(spec, h, config, rolling=RollingConfig(7), **kwargs)
