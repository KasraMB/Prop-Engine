"""Independent finite-wallet examples; these are scenarios, not firm evidence."""
from datetime import timedelta

import numpy as np
import pytest

from propfirm_engine.cashflows import (
    AttemptCashflows, Cashflow, simulate_cashflow_sequences, simulate_wallet_sequences,
)


def schedule(*, duration=7, fee=100, receipt=300, receipt_day=14):
    return AttemptCashflows(timedelta(days=duration), (
        Cashflow(timedelta(0), -fee, "fee"),
        Cashflow(timedelta(days=receipt_day), receipt, "receipt"),
    ))


def run(schedules, balance, horizon=21, policy="wait_for_receipts", n=1, seed=0):
    return simulate_wallet_sequences(schedules, timedelta(days=horizon),
                                     initial_balance=balance, retry_policy=policy,
                                     n_sequences=n, seed=seed)


def test_future_receipt_cannot_finance_a_retry():
    result = run([schedule()], 100)
    # Start day 0, wait at day 7, receive 300/start next on day 14.
    assert result.attempts_started.tolist() == [2]
    assert result.fees_paid.tolist() == [200]
    assert result.receipts.tolist() == [300]
    assert result.ending_balance.tolist() == [200]
    assert result.minimum_balance.tolist() == [0]


def test_stop_policy_collects_pending_receipts_without_restarting():
    result = run([schedule()], 100, policy="stop")
    assert result.attempts_started.tolist() == [1]
    assert result.ending_balance.tolist() == [300]
    assert result.funding_shortfall.tolist() == [True]


def test_cannot_pay_first_fee_with_same_instant_receipt():
    result = run([schedule(receipt_day=0)], 0)
    assert result.attempts_started.tolist() == [0]
    assert result.receipts.tolist() == [0]
    assert result.ending_balance.tolist() == [0]
    assert result.funding_shortfall.tolist() == [True]


def test_exact_boundary_receipt_counts_but_no_new_attempt_starts():
    result = run([schedule()], 100, horizon=14)
    assert result.attempts_started.tolist() == [1]
    assert result.ending_balance.tolist() == [300]


def test_wait_collects_several_small_receipts_before_affording_one_fee():
    s = AttemptCashflows(timedelta(days=1), (
        Cashflow(timedelta(0), -100, "fee"),
        Cashflow(timedelta(days=2), 40, "receipt"),
        Cashflow(timedelta(days=3), 60, "receipt"),
    ))
    result = run([s], 100, horizon=4)
    assert result.attempts_started.tolist() == [2]  # days 0 and 3
    assert result.ending_balance.tolist() == [0]


def test_zero_fee_and_cooldown_do_not_create_infinite_retries():
    result = run([schedule(duration=10, fee=0, receipt_day=1, receipt=50)],
                 0, horizon=21)
    assert result.attempts_started.tolist() == [3]
    assert result.receipts.tolist() == [150]  # days 1, 11, 21


def test_large_wallet_matches_unconstrained_cashflow_sampler():
    schedules = [schedule(duration=2, receipt_day=4, receipt=0),
                 schedule(duration=3, receipt_day=5, receipt=250)]
    wallet = run(schedules, 10000, horizon=20, n=40, seed=19)
    unlimited = simulate_cashflow_sequences(schedules, timedelta(days=20), 40, 19)
    for name in ("net_cashflow", "fees_paid", "receipts", "attempts_started"):
        np.testing.assert_array_equal(getattr(wallet, name), getattr(unlimited, name))
    np.testing.assert_allclose(wallet.ending_balance, 10000 + wallet.net_cashflow)
    assert not wallet.funding_shortfall.any()


def test_starts_require_cash_without_overlapping_attempts():
    result = run([schedule(duration=7, receipt_day=1, receipt=1000)], 100, horizon=15)
    assert result.attempts_started.tolist() == [3]  # 0, 7, 14, never on early receipts
    assert result.ending_balance.tolist() == [2800]


@pytest.mark.parametrize("policy", ["stop", "wait_for_receipts"])
def test_results_deterministic_and_read_only(policy):
    schedules = [schedule(receipt=0), schedule(receipt=300)]
    a = run(schedules, 200, n=40, seed=5, policy=policy)
    b = run(schedules, 200, n=40, seed=5, policy=policy)
    for name in ("net_cashflow", "ending_balance", "minimum_balance", "funding_shortfall"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        assert not getattr(a, name).flags.writeable
    assert np.all(a.minimum_balance >= 0)
    assert all("unconstrained" not in text for text in a.assumptions)


def test_post_start_fees_require_a_stateful_billing_model():
    s = AttemptCashflows(timedelta(days=7), (Cashflow(timedelta(days=1), -100, "fee"),))
    with pytest.raises(ValueError, match="upfront"):
        run([s], 100)


def test_future_outcome_cannot_choose_the_known_start_price():
    with pytest.raises(ValueError, match="same.*upfront"):
        run([schedule(fee=100), schedule(fee=200)], 100)


@pytest.mark.parametrize("balance", [None, True, -1, float("nan"), float("inf")])
def test_invalid_wallet(balance):
    with pytest.raises(ValueError, match="initial_balance"):
        run([schedule()], balance)


@pytest.mark.parametrize("policy", [None, "borrow", "wait", True])
def test_unsupported_retry_policy(policy):
    with pytest.raises(ValueError, match="retry_policy"):
        run([schedule()], 100, policy=policy)


def test_decimal_fee_does_not_lose_an_affordable_attempt_to_float_rounding():
    result = run([schedule(duration=1, fee=0.1, receipt=0)], 0.3, horizon=3)
    assert result.attempts_started.tolist() == [3]
    assert result.minimum_balance.tolist() == [0.0]
    assert result.ending_balance.tolist() == [0.0]
