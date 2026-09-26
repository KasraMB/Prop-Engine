"""Hand-computed receipt/fee timing; no firm-specific rules or invented calendars."""
from datetime import timedelta

import numpy as np
import pytest

from propfirm_engine.cashflows import AttemptCashflows, Cashflow, simulate_cashflow_sequences
from propfirm_engine.renewal import finite_horizon_cashflow


def flow(days, amount, kind):
    return Cashflow(timedelta(days=days), amount, kind)


def test_future_payout_does_not_erase_fee_paid_before_horizon():
    schedule = AttemptCashflows(timedelta(days=14), (flow(0, -100, "fee"), flow(14, 1000, "receipt")))
    result = simulate_cashflow_sequences([schedule], timedelta(days=7), 3)
    assert result.net_cashflow.tolist() == [-100.0] * 3
    assert result.receipts.tolist() == [0.0] * 3
    assert result.fees_paid.tolist() == [100.0] * 3


def test_partial_attempt_keeps_early_receipt_but_not_later_one():
    schedule = AttemptCashflows(timedelta(days=14), (
        flow(0, -100, "fee"), flow(3, 200, "receipt"), flow(10, 1000, "receipt")))
    assert finite_horizon_cashflow([schedule], 1, n_sequences=2).tolist() == [100.0, 100.0]


def test_exact_horizon_receipt_counts_without_starting_another_attempt():
    schedule = AttemptCashflows(timedelta(days=7), (flow(0, -100, "fee"), flow(7, 200, "receipt")))
    result = simulate_cashflow_sequences([schedule], timedelta(days=7), 1)
    assert result.net_cashflow.tolist() == [100.0]
    assert result.attempts_started.tolist() == [1]


def test_pending_receipt_can_settle_after_next_attempt_started():
    schedule = AttemptCashflows(timedelta(days=7), (flow(0, -100, "fee"), flow(14, 300, "receipt")))
    result = simulate_cashflow_sequences([schedule], timedelta(days=14), 1)
    assert result.net_cashflow.tolist() == [100.0]  # two fees; only first receipt due
    assert result.attempts_started.tolist() == [2]


def test_real_elapsed_days_include_weekends_and_no_trade_gaps():
    schedule = AttemptCashflows(timedelta(days=10), (flow(0, -100, "fee"), flow(10, 200, "receipt")))
    assert finite_horizon_cashflow([schedule], 1, n_sequences=1).tolist() == [-100.0]


def test_aggregate_outcomes_cannot_supply_finite_horizon_cash_timing():
    from types import SimpleNamespace
    aggregated = SimpleNamespace(net_payout=np.array([1000.0]), total_trading_days=np.array([14]),
                                  eval_fee=100.0, activation_fee=0.0, reached_funded=np.array([True]),
                                  trading_days_per_week=7.0)
    with pytest.raises(ValueError, match="explicit.*schedules"):
        finite_horizon_cashflow(aggregated, 1, n_sequences=1)


@pytest.mark.parametrize("amount", [None, float("nan"), float("inf"), True])
def test_unknown_or_invalid_cash_amount_rejected(amount):
    with pytest.raises(ValueError, match="amount"):
        flow(0, amount, "fee")


@pytest.mark.parametrize("events", [(flow(0, 0, "fee"),), ()])
def test_zero_fee_is_distinct_from_unknown_fee(events):
    schedule = AttemptCashflows(timedelta(days=1), events)
    result = simulate_cashflow_sequences([schedule], timedelta(days=1), 1)
    assert result.net_cashflow.tolist() == [0.0]


def test_unordered_events_rejected():
    with pytest.raises(ValueError, match="chronological"):
        AttemptCashflows(timedelta(days=10), (flow(5, 20, "receipt"), flow(0, -10, "fee")))


def test_mixed_currency_rejected():
    with pytest.raises(ValueError, match="currencies"):
        simulate_cashflow_sequences([AttemptCashflows(timedelta(days=1), (), "USD"),
                                     AttemptCashflows(timedelta(days=1), (), "CAD")], timedelta(days=1))


def test_deterministic_and_results_read_only():
    schedules = [AttemptCashflows(timedelta(days=2), (flow(0, -10, "fee"),)),
                 AttemptCashflows(timedelta(days=3), (flow(2, 20, "receipt"),))]
    a = simulate_cashflow_sequences(schedules, timedelta(days=10), 50, 22)
    b = simulate_cashflow_sequences(schedules, timedelta(days=10), 50, 22)
    np.testing.assert_array_equal(a.net_cashflow, b.net_cashflow)
    assert not a.net_cashflow.flags.writeable
    assert any("affordability is not modeled" in assumption for assumption in a.assumptions)


@pytest.mark.parametrize("weeks", [float("nan"), float("inf"), 0.0, -1.0, True])
def test_invalid_horizon_is_rejected(weeks):
    with pytest.raises(ValueError, match="horizon"):
        finite_horizon_cashflow([AttemptCashflows(timedelta(days=1), ())], weeks)


@pytest.mark.parametrize("duration", [timedelta(0), timedelta(days=-1), 1.0])
def test_invalid_cycle_duration_is_rejected(duration):
    with pytest.raises(ValueError, match="duration"):
        AttemptCashflows(duration, ())


def test_request_or_approval_is_not_a_cash_receipt():
    with pytest.raises(ValueError, match="not cash"):
        Cashflow(timedelta(0), 100.0, "approval")
