"""Historical execution edge cases and canonical account integration."""
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction as F
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from propfirm_engine import BacktestConfig, DollarPolicy, Engine, Instrument, PriceSession, RiskRegime
from propfirm_engine.firms.lucidflex import replay_50k
from propfirm_engine.market_replay import _resolve, replay_prices

NY = ZoneInfo("America/New_York")
SPEC = replay_50k(eval_fee=105.2, reset_fee=105, contract_type="mini")
CONFIG = BacktestConfig(0, timedelta(0), timedelta(0), timedelta(0))
POLICY = DollarPolicy.constant(2000)
TARGETS = {r.name: 1500 for r in POLICY.regimes}


def session(rows, day=date(2025, 1, 6), hour=9, minute=30):
    start = datetime.combine(day, datetime.min.time(), NY).replace(hour=hour, minute=minute)
    timestamps = np.array([round((start + timedelta(minutes=i)).timestamp() * 1e9) for i in range(len(rows))])
    return PriceSession(day, start + timedelta(minutes=len(rows)), timestamps, np.array(rows))


def resolve(rows, risk=1000, target=1000, costs=0, quantity=1, instrument=None):
    return _resolve(session(rows), instrument or Instrument("X", 100, .01), quantity, F(str(risk)), F(str(target)), F(str(costs)))


def test_ambiguous_bar_is_stop_first():
    fill, *values = resolve([[100, 111, 89, 105]])
    assert fill.pnl == -1000
    assert values[-2:] == ["stop", True]


def test_user_stop_first_convention_applies_even_to_target_gap_collision():
    result = resolve([[100, 101, 99, 100], [112, 113, 80, 90]])
    assert result[0].pnl == -1000
    assert result[-2:] == ("stop", True)


def test_target_gap_without_collision_receives_only_limit_price():
    result = resolve([[100, 101, 99, 100], [112, 113, 111, 112]])
    assert result[0].pnl == 1000
    assert result[-2:] == ("target", False)


def test_stop_gap_keeps_loss_beyond_planned_risk():
    result = resolve([[100, 101, 99, 100], [85, 110, 84, 100]])
    assert result[0].pnl == -1500
    assert result[-2:] == ("stop_gap", True)


def test_session_exit_preserves_partial_profit_and_loss():
    assert resolve([[100, 103, 98, 102]])[0].pnl == 200
    assert resolve([[100, 103, 98, 99]])[0].pnl == -100
    assert resolve([[100, 103, 98, 99]])[-2] == "session_close"


def test_dollar_budget_includes_all_contract_costs_and_tick_rounding():
    result = resolve([[100, 101, 99, 100]], risk=1000, target=500, costs=3,
                     quantity=3, instrument=Instrument("MES", 5, .25))
    fill = result[0]
    assert fill.quantity == 3
    assert result[5] <= 1000
    assert result[6] >= 500
    assert F(str(fill.stop_loss)) * 3 + 3 == F(str(result[5]))
    assert resolve([[100, 101, 99, 100]], risk=3.01, costs=3, quantity=3,
                   instrument=Instrument("MES", 5, .25)) is None


def test_never_use_prices_after_exit():
    result = resolve([[100, 111, 99, 110], [110, 112, 0, 1]])
    assert result[0].pnl == 1000
    assert result[0].low == 0
    assert result[0].exit_at == session([[100, 111, 99, 110]]).close_at


def test_canonical_engine_preserves_timed_exit_pnl_and_actual_quantity():
    prices = [session([[100, 103, 99, 102]])]
    config = replace(CONFIG, cost_per_contract=3.5)
    path = Engine().backtest_prices(SPEC, prices, POLICY, TARGETS, Instrument("X", 100, .01),
                                    config, collision_policy="stop_first")
    assert path.replay.final_balance == 50196.5
    assert path.decisions[0].net_pnl == 196.5
    assert path.decisions[0].quantity == 1
    assert path.replay.net_cash == -105.2
    assert "ideal stop/target fills" not in " ".join(path.replay.assumptions)


def test_exact_two_eval_wins_pass_using_existing_rules():
    prices = [session([[100, 116, 99, 115]], day=date(2025, 1, d)) for d in (6, 7)]
    path = replay_prices(SPEC, prices, POLICY, TARGETS, Instrument("X", 100, .01), CONFIG,
                         collision_policy="stop_first")
    assert sum(e.kind == "evaluation_pass" for e in path.replay.events) == 1
    assert path.replay.failed_attempts == 0


def test_bar_end_qualifying_close_wins_exact_inactivity_tie():
    prices = [session([[100, 100.5, 98.5, 99]], day=d)
              for d in (date(2022, 4, 20), date(2022, 5, 20))]
    policy = DollarPolicy.constant(100)
    path = replay_prices(SPEC, prices, policy, {r.name:1500 for r in policy.regimes},
                         Instrument("X", 100, .01), CONFIG, collision_policy="stop_first")
    assert len(path.decisions) == 2
    assert path.replay.attempts == 1 and path.replay.failed_attempts == 0
    assert path.replay.final_balance == 49800


def test_tiny_remaining_buffer_becomes_capped_out_then_next_session_restarts():
    prices = [session([[100, 101, 40, 80]], day=date(2025, 1, d)) for d in (6, 7, 8)]
    path = replay_prices(SPEC, prices, POLICY, TARGETS, Instrument("ES", 50, .25),
                         replace(CONFIG, cost_per_contract=3.5), collision_policy="stop_first")
    failures = [e for e in path.replay.events if e.kind == "failure"]
    assert failures[0].code == "CAPPED_OUT"
    assert path.replay.attempts == 2
    assert path.replay.fees == 210.2
    assert len(path.decisions) == 2


def test_gap_breach_does_not_recover_with_later_prices():
    prices = [session([[100, 101, 99, 100], [70, 140, 60, 130]])]
    path = replay_prices(SPEC, prices, POLICY, TARGETS, Instrument("X", 100, .01), CONFIG,
                         collision_policy="stop_first")
    assert path.replay.failed_attempts == 1
    assert path.decisions[0].net_pnl == -3000


def test_fixed_micro_count_not_resized_to_budget():
    micro_spec = replay_50k(eval_fee=105.2, reset_fee=105, contract_type="micro")
    prices = [session([[100, 101, 99, 100]])]
    path = replay_prices(micro_spec, prices, POLICY, TARGETS, Instrument("MES", 5, .25),
                         replace(CONFIG, cost_per_contract=1), quantity=3, collision_policy="stop_first")
    assert path.decisions[0].quantity == 3
    assert [e.quantity for e in path.replay.events if e.kind == "trade"] == [3]
    assert path.decisions[0].net_pnl == -3


def test_fingerprint_includes_prices_targets_and_instrument():
    prices = [session([[100, 101, 99, 100]])]
    def fingerprint(instrument, targets=TARGETS):
        return replay_prices(SPEC, prices, POLICY, targets, instrument, CONFIG,
                             collision_policy="stop_first").replay.history_fingerprint
    assert fingerprint(Instrument("X", 100, .01)) != fingerprint(Instrument("X", 50, .01))
    assert fingerprint(Instrument("X", 100, .01)) != fingerprint(Instrument("X", 100, .01), {k: 1600 for k in TARGETS})


def test_approximation_requires_explicit_opt_in():
    with pytest.raises(TypeError):
        replay_prices(SPEC, [session([[100, 101, 99, 100]])], POLICY, TARGETS, Instrument("X", 100, .01), CONFIG)


def test_prices_are_copied_readonly_and_malformed_bars_rejected():
    s = session([[100, 101, 99, 100]])
    with pytest.raises(ValueError): s.ohlc[0, 0] = 99
    with pytest.raises(ValueError): session([[100, 99, 98, 100]])
    with pytest.raises(ValueError): replace(s, close_at=s.close_at + timedelta(minutes=1))
    with pytest.raises(ValueError): replace(s, timestamps=s.timestamps + 1)


def test_declared_globex_session_accepts_previous_evening_and_dst():
    start = datetime(2025, 3, 9, 18, tzinfo=NY)
    s = PriceSession(date(2025, 3, 10), start + timedelta(minutes=1),
                     np.array([round(start.timestamp() * 1e9)]), np.array([[100, 101, 99, 100]]))
    path = replay_prices(SPEC, [s], POLICY, TARGETS, Instrument("X", 100, .01), CONFIG,
                         collision_policy="stop_first")
    assert path.decisions[0].entry_at == datetime(2025, 3, 9, 22, tzinfo=timezone.utc)
    assert path.decisions[0].session == date(2025, 3, 10)


def test_fixed_position_above_firm_limit_rejected():
    with pytest.raises(ValueError, match="contract limit"):
        replay_prices(SPEC, [session([[100, 101, 99, 100]])], POLICY, TARGETS,
                      Instrument("X", 100, .01), CONFIG, quantity=5, collision_policy="stop_first")


def test_insufficient_policy_risk_is_not_an_account_breach():
    path = replay_prices(SPEC, [session([[100, 101, 99, 100]])], DollarPolicy.constant(1),
                         TARGETS, Instrument("ES", 50, .25), CONFIG, collision_policy="stop_first")
    assert path.replay.failed_attempts == 0
    assert not path.decisions
    assert any(e.kind == "execution_skip" for e in path.replay.events)


def test_complete_eval_and_first_payout_follow_same_canonical_ledger():
    policy = DollarPolicy((RiskRegime("eval", "eval", 2000),
                           RiskRegime("build", "funded", 1000, days_to_payout=5, after_payout=False),
                           RiskRegime("fallback", "funded", 1500)))
    prices = [session([[100, 150, 100, 120]], day=date(2025, 1, d)) for d in (6, 7, 8, 9, 10, 13, 14)]
    path = replay_prices(SPEC, prices, policy, {"eval": 1500, "build": 3000, "fallback": 500},
                         Instrument("X", 100, .01), CONFIG, collision_policy="stop_first")
    assert [d.net_pnl for d in path.decisions] == [1500, 1500, 3000, 500, 500, 500, 500]
    assert path.replay.receipts == 1800
    assert path.replay.net_cash == 1694.8
    assert path.replay.final_balance == 53000
    assert sum(e.kind == "approval" for e in path.replay.events) == 1
