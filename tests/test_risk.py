"""Independent cash arithmetic, tail weights, capital ranks and lifecycle parity."""
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
from math import comb
from types import SimpleNamespace

import numpy as np
import pytest

from propfirm_engine import BacktestConfig, BracketHistory, BracketTrade, DollarPolicy, Engine, RollingConfig
from propfirm_engine.firms.lucidflex import replay_50k
from propfirm_engine.risk import (
    CashRiskPath, RiskConfig, _confidence_capital, cash_risk_path, distribution, risk_report,
)
from propfirm_engine.target_research import BracketModel, evaluate_targets, lucidflex_example


SPEC = replay_50k(eval_fee=105.20, reset_fee=105, contract_type="micro")
CONFIG = BacktestConfig(0, timedelta(0), timedelta(0), timedelta(0))


def cash_result(flows, *, wallet=None, wait=False):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events = [SimpleNamespace(at=start+timedelta(days=day), cash=amount, kind=kind)
              for day, amount, kind in flows]
    if wait:
        events.append(SimpleNamespace(at=start+timedelta(days=8), cash=0, kind="wallet_wait"))
    return SimpleNamespace(events=events, start=start, end=start+timedelta(days=10),
        net_cash=sum(Fraction(str(e.cash)) for e in events),
        net_cash_per_day=sum(e.cash for e in events)/10,
        calendar_days=10, receipts=sum(e.cash for e in events if e.kind == "receipt"),
        fees=-sum(e.cash for e in events if e.kind == "fee"), outstanding_payouts=0,
        attempts=2, failed_attempts=1, config=replace(CONFIG, initial_wallet=wallet),
        spec=SPEC, policy=DollarPolicy.constant(2000))


def records(cash, required):
    return [CashRiskPath(value, value/10, 10, max(0,value), max(0,-value), 0, 1, 0, int(value>0),
                        max(0,-value), 0, 1 if value>0 else None, capital, False, None, None)
            for value, capital in zip(cash, required)]


def test_cash_deficit_is_not_final_loss_or_peak_to_trough_drawdown():
    result = cash_result([(0,-100,"fee"), (1,-100,"fee"), (2,500,"receipt"), (4,-400,"fee")])
    path = cash_risk_path(result)
    assert path.net_cash == -100
    assert path.required_bankroll == 200
    assert path.max_cash_drawdown == 400
    assert path.longest_underwater_days == 6
    assert path.days_to_first_receipt == 2
    assert path.return_on_initial_wallet is None
    assert cash_risk_path(cash_result([(0,-.1,"fee"), (1,-.2,"fee")])).required_bankroll == .3
    assert cash_risk_path(cash_result([(0,-.301,"fee")])).required_bankroll == .31


def test_wallet_truncation_requires_matching_unrestricted_path():
    stopped = cash_result([(0,-100,"fee")], wallet=100, wait=True)
    full = cash_result([(0,-100,"fee"), (2,-100,"fee"), (3,300,"receipt")])
    with pytest.raises(ValueError, match="unrestricted"):
        cash_risk_path(stopped)
    path = cash_risk_path(stopped, unrestricted=full)
    assert path.required_bankroll == 200
    assert path.net_cash == -100
    assert path.observed_funding_shortfall is True
    full.config = replace(CONFIG, receipt_delay=timedelta(days=2))
    with pytest.raises(ValueError, match="match"):
        cash_risk_path(stopped, unrestricted=full)


def test_variance_percentiles_tail_loss_and_probability_reconcile():
    report = risk_report(records([-100, 0, 100, 300], [100,200,300,400]),
        options=RiskConfig(bankroll=200, target_ruin_probability=.25, tail_probability=.375))
    d = report["distributions"]["net_cash"]
    assert d["variance"] == pytest.approx(np.var([-100,0,100,300]))
    assert d["sample_variance"] == pytest.approx(np.var([-100,0,100,300], ddof=1))
    assert d["percentiles"]["0.5"] == 50
    assert d["percentiles"]["0.99"] == pytest.approx(294)
    assert sum(report[k] for k in ("probability_loss", "probability_profitable", "probability_break_even")) == 1
    assert report["loss_expected_shortfall"] == pytest.approx(100/1.5)
    assert report["worst_tail_mean_net_cash"] == pytest.approx(-100/1.5)
    assert report["ruin_probability"] == .5
    assert report["required_bankroll"] == 300
    assert report["achieved_empirical_ruin_probability"] == .25
    assert report["confidence_supported_bankroll"] is None
    assert report["distributions"]["days_to_first_receipt"]["count"] == 2


def test_empirical_capital_uses_order_statistic_not_interpolated_percentile():
    paths = records([0]*4, [100, 200, 300, 400])
    report = risk_report(paths, options=RiskConfig(bankroll=300, target_ruin_probability=.24))
    assert report["required_bankroll"] == 400
    assert report["ruin_probability"] == .25
    assert risk_report(paths, options=RiskConfig(target_ruin_probability=1))["required_bankroll"] == 0
    assert risk_report(paths, options=RiskConfig(target_ruin_probability=0))["required_bankroll"] == 400


def test_confidence_capital_matches_independent_binomial_calculation():
    n, alpha, confidence = 100, .05, .95
    allowed = [k for k in range(n) if sum(comb(n,j)*alpha**j*(1-alpha)**(n-j) for j in range(k+1)) <= 1-confidence]
    assert max(allowed) == 1
    assert _confidence_capital(list(range(1,101)), alpha, confidence) == 99
    report = risk_report(records([0]*100, list(range(1,101))),
        options=RiskConfig(target_ruin_probability=alpha), sample_kind="independent_model")
    assert report["required_bankroll"] == 95
    assert report["confidence_supported_bankroll"] == 99
    assert report["best_case_zero_failure_upper_bound"] == pytest.approx(1-.05**.01)
    assert report["probability_intervals"]["bounds"]["profitable"][1] > 0
    assert _confidence_capital(list(range(1,31)), .01, .95) is None
    assert _confidence_capital([1,2], 0, .95) is None
    assert _confidence_capital([1,2], 1, .95) == 0
    assert _confidence_capital([100]*298, .01, .95) is None
    assert _confidence_capital([100]*299, .01, .95) == 100


@pytest.mark.parametrize("kind", ["historical_windows", "single_history", "training_model"])
def test_no_confidence_claim_for_dependent_or_selected_training_results(kind):
    report = risk_report(records([0]*100, [100]*100), sample_kind=kind)
    assert report["confidence_supported_bankroll"] is None
    assert report["best_case_zero_failure_upper_bound"] is None
    assert report["probability_intervals"] is None
    assert report["confidence_status"] == "no_independent_sample"


def test_curve_is_monotone_and_equality_with_required_cash_survives():
    report = risk_report(records([0]*4, [100,100,200,300]), options=RiskConfig(bankroll=100))
    assert report["ruin_probability"] == .5
    curve = report["bankroll_curve"]
    assert [p["ruin_probability"] for p in curve] == [1,.5,.25,0]
    assert [p["bankroll"] for p in curve] == [0,100,200,300]


def test_unrestricted_deficit_predicts_actual_wallet_wait_with_delayed_receipts():
    spec = replace(SPEC, account=replace(SPEC.account, activation_fee=150))
    trades, day = [], date(2026,1,5)
    for i in range(20):
        while day.weekday() > 4:
            day += timedelta(days=1)
        start = datetime.fromisoformat(str(day)+"T10:00:00-05:00")
        trades.append(BracketTrade(start, start+timedelta(minutes=5), day, 2000,
                                   1500 if i < 2 else 520, i < 7))
        day += timedelta(days=1)
    history, policy = BracketHistory(tuple(trades)), DollarPolicy.constant(2000)
    config = replace(CONFIG, receipt_delay=timedelta(days=12))
    engine = Engine()
    full = engine.backtest(spec, history, policy, config)
    required = cash_risk_path(full).required_bankroll
    assert required > SPEC.account.eval_fee and full.receipts > 0
    for capital in (0, 105.2, required-.01, required, required+100):
        result = engine.backtest(spec, history, policy, replace(config, initial_wallet=capital))
        assert any(e.kind == "wallet_wait" for e in result.events) == (capital < required)
    rolling = engine.rolling_backtest(spec, history, policy, replace(config, initial_wallet=0),
        rolling=RollingConfig(window_sessions=10, stride_sessions=5), risk=RiskConfig(bankroll=0))
    assert rolling.risk["ruin_probability"] == 1
    assert rolling.risk["required_bankroll"] >= 105.2
    assert rolling.risk["distributions"]["net_cash"]["mean"] == 0


def test_model_report_uses_unrestricted_same_uniform_tape():
    report = evaluate_targets(SPEC, BracketModel(sessions=7), lucidflex_example(),
        replace(CONFIG, initial_wallet=0), [[0]*7, [.999]*7], risk=RiskConfig(bankroll=105.2)).risk
    assert report["sample_kind"] == "independent_model"
    assert report["ruin_probability"] == .5
    assert report["distributions"]["net_cash"]["mean"] == 0
    assert report["required_bankroll"] == pytest.approx(105.2+6*105)


@pytest.mark.parametrize("kwargs", [{"bankroll":-1}, {"bankroll":True}, {"bankroll":.001},
    {"target_ruin_probability":1.01}, {"confidence":1}, {"tail_probability":0},
    {"percentiles":()}, {"percentiles":(.5,.5)}, {"percentiles":(float("nan"),)}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        RiskConfig(**kwargs)


def test_empty_single_and_all_profitable_samples():
    assert distribution([]) is None
    assert distribution([100])["sample_variance"] is None
    report = risk_report(records([100,200],[0,0]), options=RiskConfig(bankroll=0))
    assert report["ruin_probability"] == 0
    assert report["loss_var"] == report["loss_expected_shortfall"] == 0
    assert len(distribution([0,100], [.1000000001, .1000000002])["percentiles"]) == 2
    with pytest.raises(ValueError, match="numeric reporting range"):
        distribution([-1e300, 1e300])
