"""Independent dated-payout accounting cases; not intraday execution certification."""
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction

import pytest

from propfirm_engine import PayoutSchema, StateField
from propfirm_engine.payouts import PayoutLedger

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def ledger(**changes):
    sc = PayoutSchema((2000.0,), 0.9, 5, cap_fraction=0.5, min_request=500,
                      fraction_basis="retained_profit", profit_reference_balance=50_000,
                      min_cycle_profit=1, reset_fields=(StateField.N_QUALIFYING_DAYS,))
    return PayoutLedger(replace(sc, **changes), opening_balance=50_000,
                        qualifying_days=5, winning_day_profit=150,
                        initial_floor=48_000, lock_floor_on_request=50_100)


def earn(obj, amounts, start=0):
    for i, amount in enumerate(amounts, start):
        at = T0 + timedelta(days=i)
        obj.record_trade(at, amount)
        obj.close_session(at, at.date())


def ready():
    obj = ledger()
    earn(obj, [200] * 5)
    return obj


def test_request_locks_floor_but_does_not_deduct_or_create_cash():
    obj = ready()
    rid = obj.request(T0 + timedelta(days=4), 500, flat=True)
    assert rid == 1 and obj.floor == 50_100 and obj.balance == 51_000
    assert obj.cycle_profit == 1000 and obj.qualifying_days == 5
    assert obj.receipt_cashflows(T0) == ()
    assert obj.events[-1].kind == "request"


def test_approval_deducts_gross_and_resets_but_is_not_spendable_receipt():
    obj = ready()
    rid = obj.request(T0 + timedelta(days=4), 500, flat=True)
    obj.approve(T0 + timedelta(days=6))
    assert obj.balance == 50_500 and obj.floor == 50_100
    assert obj.qualifying_days == 0 and obj.cycle_profit == 0
    assert obj.receipt_cashflows(T0) == ()
    obj.receive(T0 + timedelta(days=8), rid)
    flow, = obj.receipt_cashflows(T0)
    assert flow.offset == timedelta(days=8) and flow.amount == 450
    assert obj.balance == 50_500  # cash receipt is not another account deduction


def test_no_trading_or_second_request_while_pending():
    obj = ready()
    obj.request(T0 + timedelta(days=4), 500, flat=True)
    before = obj.events
    with pytest.raises(ValueError, match="blocked"):
        obj.record_trade(T0 + timedelta(days=5), 100)
    with pytest.raises(ValueError):
        obj.request(T0 + timedelta(days=5), 500, flat=True)
    assert obj.events == before and obj.balance == 51_000


def test_trading_resumes_after_approval_before_receipt():
    obj = ready()
    rid = obj.request(T0 + timedelta(days=4), 500, flat=True)
    obj.approve(T0 + timedelta(days=5))
    earn(obj, [150] * 5, start=6)
    assert obj.maximum_request() == 625  # retained 500 + fresh 750
    assert obj.cycle_profit == 750
    assert obj.receipt_cashflows(T0) == ()
    obj.receive(T0 + timedelta(days=11), rid, payment_fee=10)
    assert obj.receipt_cashflows(T0)[0].amount == 440


def test_termination_preserves_approved_receipts_and_blocks_recovery():
    obj = ready()
    at = T0 + timedelta(days=4)
    rid = obj.request(at, 500, flat=True)
    obj.approve(at)
    obj.terminate(at, 10)
    assert obj.balance == 50_510 and obj.breached
    assert obj.maximum_request() == 0
    with pytest.raises(ValueError, match="blocked"):
        obj.record_trade(at, 1000)
    with pytest.raises(ValueError, match="terminated"):
        obj.terminate(at)
    obj.receive(at + timedelta(days=1), rid)
    assert obj.receipt_cashflows(T0)[0].amount == 450


def test_termination_can_book_liquidation_after_balance_breach():
    obj = ledger()
    obj.record_trade(T0, -2000)
    obj.terminate(T0, -5)
    assert obj.balance == 47_995 and obj.breached


def test_cycle_losses_do_not_disappear_behind_retained_profit():
    obj = ledger()
    earn(obj, [1000] * 5)
    obj.request(T0 + timedelta(days=4), 2000, flat=True)
    obj.approve(T0 + timedelta(days=5))
    earn(obj, [150] * 5 + [-750], start=6)
    assert obj.balance == 53_000 and obj.cycle_profit == 0
    assert obj.qualifying_days == 5 and obj.maximum_request() == 0
    earn(obj, [1], start=12)
    assert obj.maximum_request() == Fraction(3001, 2)


def test_rejection_preserves_cycle_and_request_time_lock():
    obj = ready()
    obj.request(T0 + timedelta(days=4), 500, flat=True)
    obj.reject(T0 + timedelta(days=5))
    assert obj.balance == 51_000 and obj.floor == 50_100
    assert obj.cycle_profit == 1000 and obj.qualifying_days == 5
    assert obj.maximum_request() == 500
    assert obj.requests[0].status == "rejected"


@pytest.mark.parametrize("gross", [0, -1, 499.99, 500.01, float("nan"), True])
def test_bad_requests_are_atomic(gross):
    obj = ready()
    before = obj.events
    with pytest.raises(ValueError):
        obj.request(T0 + timedelta(days=5), gross, flat=True)
    assert obj.events == before and obj.pending is None and obj.floor == 48_000


def test_open_position_or_unfinalized_session_blocks_request():
    obj = ready()
    with pytest.raises(ValueError, match="open position"):
        obj.request(T0 + timedelta(days=4), 500, flat=False)
    obj.record_trade(T0 + timedelta(days=5), 1)
    with pytest.raises(ValueError):
        obj.request(T0 + timedelta(days=5), 500, flat=True)


def test_receipt_requires_approval_and_cannot_be_duplicated():
    obj = ready()
    rid = obj.request(T0 + timedelta(days=4), 500, flat=True)
    with pytest.raises(ValueError):
        obj.receive(T0 + timedelta(days=5), rid)
    obj.approve(T0 + timedelta(days=5))
    with pytest.raises(ValueError):
        obj.receive(T0 + timedelta(days=6), rid, payment_fee=451)
    obj.receive(T0 + timedelta(days=6), rid)
    with pytest.raises(ValueError):
        obj.receive(T0 + timedelta(days=7), rid)
    assert len(obj.receipt_cashflows(T0)) == 1


def test_final_payout_is_censoring_not_breach_and_still_can_settle():
    obj = ledger(max_payouts=1)
    earn(obj, [200] * 5)
    rid = obj.request(T0 + timedelta(days=4), 500, flat=True)
    obj.approve(T0 + timedelta(days=5))
    assert obj.censored and not obj.breached
    with pytest.raises(ValueError, match="censoring"):
        obj.record_trade(T0 + timedelta(days=6), 1)
    obj.receive(T0 + timedelta(days=6), rid)
    assert obj.receipt_cashflows(T0)[0].amount == 450


def test_realized_floor_breach_cannot_be_recovered():
    obj = ledger()
    obj.record_trade(T0, -2000)
    assert obj.breached
    with pytest.raises(ValueError):
        obj.record_trade(T0, 3000)


def test_clocks_and_sessions_are_explicit_and_monotonic():
    obj = ready()
    with pytest.raises(ValueError, match="chronological"):
        obj.record_trade(T0, 10)
    with pytest.raises(ValueError, match="timezone"):
        obj.record_trade(datetime(2026, 1, 6), 10)
    with pytest.raises(ValueError, match="strictly increase"):
        obj.close_session(T0 + timedelta(days=5), date(2026, 1, 5))
    with pytest.raises(ValueError, match="date identifier"):
        obj.close_session(T0 + timedelta(days=5), T0)


def test_no_trade_session_does_not_become_a_winning_day():
    obj = ledger()
    obj.close_session(T0, T0.date())
    assert obj.qualifying_days == 0


def test_exact_cents_are_retained_without_epsilon_accounting():
    obj = ledger()
    obj.record_trade(T0, 0.1)
    obj.record_trade(T0, 0.2)
    assert obj.cycle_profit == Fraction(3, 10)
