"""Independent hand-calculated lifecycle and boundary examples."""
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from propfirm_engine import (
    BacktestConfig, BracketHistory, BracketTrade, DollarPolicy, Engine, RiskRegime,
)
from propfirm_engine.firms.lucidflex import replay_50k

NY = ZoneInfo("America/New_York")


def session_day(index):
    day = date(2026, 9, 1)
    for _ in range(index):
        day += timedelta(days=1)
        while day.weekday() >= 5:
            day += timedelta(days=1)
    return day


def trade(day, *, won=True, stop=100, target=200, hour=10):
    return BracketTrade(datetime.combine(day, time(hour), NY),
                        datetime.combine(day, time(hour, 5), NY),
                        day, stop, target, won)


def history(*trades):
    return BracketHistory(trades)


def config(**kwargs):
    values = dict(cost_per_contract=0, approval_delay=timedelta(0),
                  receipt_delay=timedelta(0), activation_delay=timedelta(minutes=30))
    return BacktestConfig(**(values | kwargs))


def spec():
    return replay_50k(eval_fee=105.20, reset_fee=105, contract_type="micro", elapsed_inactivity=True)


def funded_history(extra=0):
    first = date(2026, 9, 1)
    return history(*(trade(session_day(i), target=1500 if i < 2 else 200)
                     for i in range(7 + extra)))


def test_full_lucid_path_consistency_payout_floor_and_cash():
    result = Engine().backtest(spec(), funded_history(), DollarPolicy.constant(100), config())
    passed = [e for e in result.events if e.kind == "evaluation_pass"]
    assert len(passed) == 1 and passed[0].at.date() == date(2026, 9, 2)
    request = next(e for e in result.events if e.kind == "request")
    assert request.balance == 51_000 and request.floor == 50_100
    approval = next(e for e in result.events if e.kind == "approval")
    assert approval.balance == 50_500
    assert result.receipts == 450
    assert result.fees == 105.20
    assert result.net_cash == pytest.approx(344.80)
    assert result.attempts == 1 and result.failed_attempts == 0


def test_cached_cash_is_exact_replace_safe_and_export_is_detached():
    from dataclasses import asdict
    from fractions import Fraction
    from dashboard.replay import _jsonable
    result = Engine().backtest(spec(), funded_history(), DollarPolicy.constant(100), config())
    assert result.net_cash == float(sum((Fraction(str(e.cash)) for e in result.events), Fraction(0)))
    assert result.__dict__["net_cash"] == result.net_cash
    changed = replace(result, events=())
    assert changed.net_cash == 0 and result.net_cash == 344.8
    exported = _jsonable(result)
    assert exported == _jsonable(asdict(result))
    assert "net_cash" not in exported  # cache internals are not schema fields
    exported["events"][0]["cash"] = 999
    assert result.events[0].cash != 999


def test_one_giant_day_never_bypasses_consistency():
    day = date(2026, 9, 1)
    result = Engine().backtest(spec(), history(trade(day, target=3000)),
                              DollarPolicy.constant(100), config())
    assert not any(e.kind == "evaluation_pass" for e in result.events)
    assert result.final_balance == 53_000


def test_fee_inclusive_cap_and_touch_is_breach():
    day = date(2026, 9, 1)
    result = Engine().backtest(spec(), history(trade(day, won=False)),
                              DollarPolicy.constant(3000), config(cost_per_contract=25))
    fill = next(e for e in result.events if e.kind == "trade")
    assert fill.quantity == 16
    assert fill.balance == 48_000
    assert result.failed_attempts == 1
    assert next(e for e in result.events if e.kind == "failure").code == "FAIL_TRAILING_DD"


def test_insufficient_buffer_caps_out_then_retries():
    day = date(2026, 9, 1)
    h = history(*(trade(day + timedelta(days=i), won=False) for i in range(3)))
    result = Engine().backtest(spec(), h, DollarPolicy.constant(3000),
                              config(cost_per_trade=1))
    failures = [e for e in result.events if e.kind == "failure"]
    assert failures[0].code == "CAPPED_OUT"
    assert result.attempts == 2
    assert result.fees == pytest.approx(210.20)


def test_policy_skip_does_not_terminate_or_round_up():
    for risk in (0, 99):
        result = Engine().backtest(spec(), funded_history(), DollarPolicy.constant(risk), config())
        assert result.failed_attempts == 0
        assert not any(e.kind == "trade" for e in result.events)


def test_pending_blocks_trading_receipt_is_not_approval():
    result = Engine().backtest(
        spec(), funded_history(extra=2), DollarPolicy.constant(100),
        config(approval_delay=timedelta(days=1), receipt_delay=timedelta(days=2)),
    )
    assert any(e.kind == "pending_skip" for e in result.events)
    assert any(e.kind == "approval" for e in result.events)
    assert result.receipts == 0 and result.outstanding_payouts == 450
    assert all(e.at <= result.end for e in result.events)


def test_named_regime_sees_finalized_days_not_current_trade():
    p = DollarPolicy((
        RiskRegime("evaluation", "eval", 100),
        RiskRegime("last_winning_day", "funded", 200, days_to_payout=1),
        RiskRegime("funded", "funded", 100),
    ))
    result = Engine().backtest(spec(), funded_history(), p, config())
    fills = [e for e in result.events if e.kind == "trade"]
    assert fills[-1].regime == "last_winning_day" and fills[-1].quantity == 2
    assert fills[-2].regime == "funded"


def test_funded_scaling_updates_at_close_not_after_each_trade():
    day = date(2026, 9, 1)
    h = history(trade(day, target=1500), trade(day + timedelta(days=1), target=1500),
                trade(day + timedelta(days=2), target=60, stop=10, hour=10),
                trade(day + timedelta(days=2), target=60, stop=10, hour=11),
                trade(day + timedelta(days=3), target=60, stop=10))
    p = DollarPolicy((RiskRegime("eval", "eval", 100),
                      RiskRegime("funded", "funded", 10_000)))
    result = Engine().backtest(spec(), h, p, config())
    sizes = [e.quantity for e in result.events if e.kind == "trade"]
    assert sizes == [1, 1, 20, 20, 40]


def test_wallet_cannot_spend_future_payouts():
    result = Engine().backtest(spec(), funded_history(), DollarPolicy.constant(100),
                              config(initial_wallet=100))
    assert result.attempts == 0 and result.fees == 0
    assert result.status == "INSUFFICIENT_WALLET"


def test_history_rejects_overlap_and_partial_trade_order():
    day = date(2026, 9, 1)
    t = trade(day)
    with pytest.raises(ValueError, match="non-overlapping"):
        history(t, t)
    with pytest.raises(ValueError, match="outside"):
        Engine().backtest(spec(), history(replace(t, session=day + timedelta(days=1))),
                          DollarPolicy.constant(100), config())


def test_varying_trade_reward_ratios_are_preserved():
    day = date(2026, 9, 1)
    result = Engine().backtest(spec(), history(trade(day, target=200),
                               trade(day, hour=11, target=300)),
                              DollarPolicy.constant(200), config())
    assert result.final_balance == 51_000


def test_holdout_split_is_by_complete_sessions_and_score_is_oos():
    day = date(2026, 9, 1)
    h = history(*(trade(session_day(i), won=i < 7, target=1500)
                  for i in range(10)))
    result = Engine().fit(
        spec(), h, config(), policy=DollarPolicy.constant(100),
        risk_bounds={"evaluation": (50, 200), "funded": (50, 200)},
        generations=1, population=4, seed=5,
    )
    assert len(result.train_sessions) == 7 and len(result.test_sessions) == 3
    assert result.train_sessions[-1] < result.test_sessions[0]
    assert result.score == result.out_of_sample.net_cash_per_day
    assert result.out_of_sample.receipts == 0


def test_custom_loss_is_minimized_and_invalid_objective_rejected():
    kwargs = dict(policy=DollarPolicy.constant(100),
                  risk_bounds={"evaluation": (50, 200), "funded": (50, 200)},
                  generations=0)
    h = BracketHistory(tuple(trade(session_day(i)) for i in range(10)))
    fitted = Engine().fit(spec(), h, config(), objective=lambda r: -r.net_cash_per_day,
                          direction="minimize", **kwargs)
    assert fitted.score == -fitted.out_of_sample.net_cash_per_day
    with pytest.raises(ValueError, match="finite"):
        Engine().fit(spec(), h, config(), objective=lambda r: float("nan"), **kwargs)


def test_decimal_costs_do_not_create_a_false_qualifying_day_shortfall():
    # 150.10 gross less .10 commission is exactly the $150 day threshold.
    day = date(2026, 9, 1)
    h = history(*(trade(session_day(i), target=1500.10 if i < 2 else 150.10)
                  for i in range(7)))
    result = Engine().backtest(spec(), h, DollarPolicy.constant(100.10),
                              config(cost_per_contract=0.10))
    # Five qualifying days but only $750 retained profit: minimum $500 still blocks.
    assert not any(e.kind == "request" for e in result.events)
    assert result.final_balance == 50_750
    assert [e for e in result.events if e.kind == "session_close"][-1].qualifying_days == 5


def test_changing_only_oos_cannot_change_selected_policy():
    day = date(2026, 9, 1)
    base = tuple(trade(session_day(i), target=1500) for i in range(10))
    changed = base[:7] + tuple(replace(t, won=False) for t in base[7:])
    kwargs = dict(policy=DollarPolicy.constant(100),
                  risk_bounds={"evaluation": (50, 500), "funded": (50, 500)},
                  generations=2, population=4, seed=42)
    a = Engine().fit(spec(), history(*base), config(), **kwargs)
    b = Engine().fit(spec(), history(*changed), config(), **kwargs)
    assert a.policy == b.policy
    assert a.in_sample_score == b.in_sample_score
    assert a.evaluations == b.evaluations


def test_exhaustive_eval_paths_against_independent_full_consistency_calculation():
    from itertools import product

    s = spec()
    s = replace(s, account=replace(s.account, phases=(s.account.phases[0],)))
    day = date(2026, 9, 1)
    for wins in product((False, True), repeat=6):
        balance, floor, peak, biggest = 50_000, 48_000, 50_000, 0
        expected = "HORIZON"
        for won in wins:
            if balance - floor < 1000:
                expected = "CAPPED_OUT"
                break
            pnl = 1000 if won else -1000
            balance += pnl
            if balance <= floor:
                expected = "FAIL_TRAILING_DD"
                break
            biggest = max(biggest, pnl)
            profit = balance - 50_000
            if profit >= 3000 and 2 * biggest <= profit:
                expected = "PASSED"
                break
            peak = max(peak, balance)
            floor = min(peak - 2000, 50_100)
        h = history(*(trade(session_day(i), won=won, stop=1000, target=1000)
                      for i, won in enumerate(wins)))
        result = Engine().backtest(s, h, DollarPolicy.constant(1000), config())
        terminal = next((e.code for e in result.events
                         if e.kind in ("failure", "evaluation_pass")), "HORIZON")
        assert terminal == expected, wins


def test_live_handoff_purchases_a_fresh_evaluation_without_counting_a_failure():
    result = Engine().backtest(spec(), funded_history(extra=30),
                              DollarPolicy.constant(100), config())
    assert result.status == "HORIZON"
    assert result.attempts == 2 and result.failed_attempts == 0
    assert len([e for e in result.events if e.kind == "receipt"]) == 5
    handoff = next(e for e in result.events if e.kind == "live_handoff")
    renewed = [e for e in result.events if e.kind == "trade" and e.at > handoff.at]
    assert renewed and renewed[0].phase == "eval" and renewed[0].attempt == 2
    assert renewed[0].balance == 50_200 and renewed[0].floor == 48_000
    assert renewed[0].qualifying_days == 0 and renewed[0].cycle_profit == 0
    assert result.fees == pytest.approx(2 * 105.20)  # purchase, not the $105 reset
    assert result.outstanding_payouts == 0


def test_two_full_handoff_cycles_match_independent_cash_calculation():
    # Each account: two evaluation days, then five five-day payout cycles.
    h = history(*(trade(session_day(i), target=1500 if i % 27 < 2 else 200)
                  for i in range(54)))
    result = Engine().backtest(spec(), h, DollarPolicy.constant(100), config())
    assert result.attempts == 2 and result.failed_attempts == 0
    assert result.status == "RESTART_PENDING" and result.final_balance is None
    handoffs = [e for e in result.events if e.kind == "live_handoff"]
    assert [e.attempt for e in handoffs] == [1, 2]
    # Each cycle retains half of prior retained profit plus five $200 wins.
    gross = [500, 750, 875, 937.5, 968.75]
    receipts = [e for e in result.events if e.kind == "receipt"]
    assert [e.cash for e in receipts] == pytest.approx([g * 0.9 for g in gross] * 2)
    assert result.net_cash == pytest.approx(2 * (sum(gross) * 0.9 - 105.20))
    assert len([e for e in result.events if e.kind == "trade"]) == 54
    assert not any(e.kind == "failure" for e in result.events)


def test_handoff_at_horizon_does_not_charge_an_unstarted_attempt():
    result = Engine().backtest(spec(), funded_history(extra=20),
                              DollarPolicy.constant(100), config(receipt_delay=timedelta(days=3)))
    assert result.status == "RESTART_PENDING"
    assert result.attempts == 1 and result.fees == 105.20
    assert result.failed_attempts == 0 and result.final_balance is None
    assert result.outstanding_payouts == 871.875
    assert len([e for e in result.events if e.kind == "receipt"]) == 4
    assert all(e.at <= result.end for e in result.events)


def test_handoff_retry_delay_and_no_reentry_in_the_handoff_session():
    h = funded_history(extra=35)
    cfg = config(approval_delay=timedelta(hours=12), retry_delay=timedelta(days=3))
    result = Engine().backtest(spec(), h, DollarPolicy.constant(100), cfg)
    handoff = next(e for e in result.events if e.kind == "live_handoff")
    # Approval happens the morning after the qualifying close: exclude that session too.
    expected = next(t for t in h.trades if t.session > handoff.at.astimezone(NY).date()
                    and t.entry_at >= handoff.at + cfg.retry_delay)
    renewal = next(e for e in result.events if e.kind == "phase_start" and e.attempt == 2)
    assert renewal.at == expected.entry_at and renewal.phase == "eval"
    assert result.failed_attempts == 0


def test_handoff_with_zero_delay_still_skips_approval_session():
    h = funded_history(extra=35)
    result = Engine().backtest(spec(), h, DollarPolicy.constant(100),
                              config(approval_delay=timedelta(hours=12)))
    handoff = next(e for e in result.events if e.kind == "live_handoff")
    same_day = [t for t in h.trades if t.session == handoff.at.astimezone(NY).date()]
    assert same_day and same_day[0].entry_at > handoff.at
    expected = next(t for t in h.trades if t.session > handoff.at.astimezone(NY).date())
    renewal = next(e for e in result.events if e.kind == "phase_start" and e.attempt == 2)
    assert renewal.at == expected.entry_at


def test_old_handoff_receipt_survives_new_attempt_without_changing_its_balance():
    result = Engine().backtest(spec(), funded_history(extra=30), DollarPolicy.constant(100),
                              config(receipt_delay=timedelta(days=3)))
    renewal = next(e for e in result.events if e.kind == "phase_start" and e.attempt == 2)
    old_receipts = [e for e in result.events if e.kind == "receipt" and e.at > renewal.at]
    assert len(old_receipts) == 1 and old_receipts[0].attempt == 1
    fills = [e for e in result.events if e.kind == "trade" and e.attempt == 2]
    assert [e.balance for e in fills] == [50_000 + 200 * i for i in range(1, len(fills) + 1)]
    assert result.receipts == 3628.125 and result.outstanding_payouts == 0
    assert result.failed_attempts == 0  # stale inactivity events cannot breach the new account


def test_handoff_wallet_waits_for_real_receipts_before_buying_again():
    h = funded_history(extra=50)
    result = Engine().backtest(spec(), h, DollarPolicy.constant(100),
                              config(initial_wallet=105.20, receipt_delay=timedelta(days=45)))
    handoff = next(e for e in result.events if e.kind == "live_handoff")
    receipt = next(e for e in result.events if e.kind == "receipt")
    renewal = next(e for e in result.events if e.kind == "phase_start" and e.attempt == 2)
    assert handoff.at < receipt.at <= renewal.at
    assert any(e.kind == "wallet_wait" and handoff.at < e.at < receipt.at for e in result.events)
    assert renewal.at == next(t.entry_at for t in h.trades if t.entry_at >= receipt.at)
    wallet = 105.20
    for e in result.events:
        wallet += e.cash
        assert wallet >= -1e-9
    assert result.failed_attempts == 0


def test_funded_only_profile_renews_its_starting_phase_after_handoff():
    s = spec()
    s = replace(s, account=replace(s.account, phases=(s.account.phases[1],)))
    h = history(*(trade(session_day(i)) for i in range(26)))
    result = Engine().backtest(s, h, DollarPolicy.constant(100), config())
    assert result.attempts == 2 and result.failed_attempts == 0
    starts = [e for e in result.events if e.kind == "phase_start"]
    assert [e.phase for e in starts] == ["funded", "funded"]
    assert starts[-1].balance == 50_000 and starts[-1].qualifying_days == 0


def test_optimizer_uses_renewing_lifecycle_in_both_is_and_oos():
    h = history(*(trade(session_day(i), target=1500) for i in range(100)))
    engine, scenario, cfg = Engine(), spec(), config()
    fitted = engine.fit(scenario, h, cfg, policy=DollarPolicy.constant(100),
                        risk_bounds={"evaluation": (50, 200), "funded": (50, 200)},
                        generations=0)
    train, test = h.split(0.7)
    for reported, partition, expected_attempts in (
        (fitted.in_sample, train, 3), (fitted.out_of_sample, test, 2),
    ):
        direct = engine.backtest(scenario, partition, fitted.policy, cfg)
        assert reported.events == direct.events
        assert reported.attempts == expected_attempts and reported.failed_attempts == 0
        assert any(e.kind == "live_handoff" for e in reported.events)
    assert fitted.score == fitted.out_of_sample.net_cash_per_day


def test_csv_record_contract_and_weekend_rejection():
    row = dict(entry_at="2026-09-01T10:00:00-04:00",
               exit_at="2026-09-01T10:05:00-04:00", session="2026-09-01",
               stop_loss="100", take_profit="200", won="true")
    h = BracketHistory.from_records([row])
    assert h.trades[0].won is True and h.trades[0].stop_loss == 100
    with pytest.raises(ValueError, match="won"):
        BracketHistory.from_records([row | {"won": "yes"}])
    with pytest.raises(ValueError, match="weekday"):
        Engine().backtest(spec(), history(trade(date(2026, 9, 5))),
                          DollarPolicy.constant(100), config())


def test_unvisited_funded_regimes_keep_declared_baseline():
    h = BracketHistory(tuple(trade(session_day(i), won=False) for i in range(10)))
    p = DollarPolicy.constant(100)
    result = Engine().fit(
        spec(), h, config(), policy=p,
        risk_bounds={"evaluation": (50, 500), "funded": (50, 500)},
        generations=2, population=4, seed=10,
    )
    assert result.policy.regimes[1].risk_dollars == 100


def test_inactivity_deletes_instead_of_indefinite_zero_risk_survival():
    h = history(trade(date(2026, 9, 1)), trade(date(2026, 10, 5)))
    result = Engine().backtest(spec(), h, DollarPolicy.constant(0), config())
    failures = [e for e in result.events if e.kind == "failure"]
    assert failures[0].code == "FAIL_INACTIVITY"
    assert failures[0].at.date() == date(2026, 10, 1)
    assert result.attempts == 2
    assert result.fees == pytest.approx(210.40)  # new purchase, not reset


def test_expired_reset_uses_new_purchase_fee():
    h = history(trade(date(2026, 9, 1), won=False),
                trade(date(2026, 10, 5)))
    result = Engine().backtest(spec(), h, DollarPolicy.constant(2000), config())
    assert result.attempts == 2 and result.fees == pytest.approx(210.40)


@pytest.mark.parametrize("won", [True, False])
def test_qualifying_close_renews_activity_at_exact_deadline(won):
    h = history(trade(date(2026, 9, 1)), trade(date(2026, 10, 1), won=won),
                trade(date(2026, 10, 2)))
    result = Engine().backtest(spec(), h, DollarPolicy.constant(100), config())
    assert result.attempts == 1 and result.failed_attempts == 0
    assert len([e for e in result.events if e.kind == "trade"]) == 3
    assert result.fees == pytest.approx(105.20)


def test_inactivity_tie_qualification_uses_net_pnl_after_costs():
    first = trade(date(2026, 9, 1), stop=1, target=2)
    last = trade(date(2026, 10, 1), stop=1, target=1)
    with pytest.raises(ValueError, match="inactivity expired during an open trade"):
        Engine().backtest(spec(), history(first, last), DollarPolicy.constant(1.5),
                          config(cost_per_contract=.5))


@pytest.mark.parametrize("won", [True, False])
@pytest.mark.parametrize("cost_key", ["cost_per_contract", "cost_per_trade"])
def test_exact_one_dollar_net_close_qualifies_at_inactivity_tie(won, cost_key):
    first = trade(date(2026, 9, 1), stop=.5, target=2)
    last = trade(date(2026, 10, 1), stop=.5, target=1.5, won=won)
    result = Engine().backtest(spec(), history(first, last), DollarPolicy.constant(1),
                              config(**{cost_key:.5}))
    assert result.attempts == 1 and result.failed_attempts == 0
    fills = [e for e in result.events if e.kind == "trade"]
    assert fills[-1].balance-fills[0].balance == (1 if won else -1)


def test_inactivity_strictly_before_close_is_still_unsupported():
    last = trade(date(2026, 10, 1))
    last = replace(last, exit_at=last.exit_at+timedelta(microseconds=1))
    with pytest.raises(ValueError, match="inactivity expired during an open trade"):
        Engine().backtest(spec(), history(trade(date(2026, 9, 1)), last),
                          DollarPolicy.constant(100), config())


def test_inactivity_at_entry_still_expires_before_new_trade():
    last = trade(date(2026, 10, 1))
    last = replace(last, entry_at=last.entry_at+timedelta(minutes=5),
                   exit_at=last.exit_at+timedelta(minutes=5))
    result = Engine().backtest(spec(), history(trade(date(2026, 9, 1)), last),
                              DollarPolicy.constant(100), config())
    assert any(e.code == "FAIL_INACTIVITY" for e in result.events)
    assert len([e for e in result.events if e.kind == "trade"]) == 1


def test_payment_fee_reduces_receipt_not_account_withdrawal():
    result = Engine().backtest(spec(), funded_history(), DollarPolicy.constant(100),
                              config(payment_fee=10))
    assert result.receipts == 440 and result.final_balance == 50_500


def test_exact_decimal_floor_touch():
    from propfirm_engine import Account, Phase, ProfitTargetRule, TrailingDrawdownRule, Timing

    account = Account("decimal-test", 1, (Phase("eval", "eval", (
        ProfitTargetRule(10),
        TrailingDrawdownRule(0.3, update_timing=Timing.EOD),
    )),))
    scenario = replace(spec(), account=account, request_lock_floor=None)
    h = history(*(trade(session_day(i), won=False, stop=0.1, target=0.1) for i in range(3)))
    result = Engine().backtest(scenario, h, DollarPolicy.constant(0.1), config())
    failure = next(e for e in result.events if e.kind == "failure")
    assert failure.code == "FAIL_TRAILING_DD" and failure.balance == failure.floor == 0.7


def test_existing_generator_runs_through_the_full_holdout_api():
    from propfirm_engine import IIDGenerator

    stream = IIDGenerator(.55, 1.5, trades_per_day=8).generate(80, seed=260926)
    h = history(*(BracketTrade(
        at.replace(tzinfo=NY) - timedelta(minutes=1), at.replace(tzinfo=NY),
        at.date(), 25, 37.5, bool(ret > 0),
    ) for at, ret in zip(stream.rows["timestamp"], stream.rows["return"])))
    result = Engine().fit(
        spec(), h, config(cost_per_contract=1), policy=DollarPolicy.constant(100),
        risk_bounds={"evaluation": (50, 300), "funded": (50, 300)},
        generations=2, population=4, seed=17,
    )
    assert len(result.train_sessions) == 56 and len(result.test_sessions) == 24
    assert result.out_of_sample.history_fingerprint != result.in_sample.history_fingerprint
    assert result.score == result.out_of_sample.net_cash_per_day
