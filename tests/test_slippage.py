"""Execution scenarios are causal and budgets are anticipated, not guaranteed."""
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction as F

import numpy as np
import pytest

from propfirm_engine import BacktestConfig, DollarPolicy, Instrument, PriceSession, SlippageModel, TickDistribution
from propfirm_engine.firms.lucidflex import replay_50k
from propfirm_engine.market_replay import _resolve, replay_prices
from propfirm_engine.slippage import prepare_execution


def certain(ticks):
    return TickDistribution((0.,)*ticks+(1.,))


def model(market=0, stop=0, **kwargs):
    return SlippageModel(certain(market), certain(stop), certain(market), certain(stop), 100., **kwargs)


def session(rows, day=6):
    start = datetime(2025, 1, day, 15, tzinfo=timezone.utc)
    t = np.array([int((start+timedelta(minutes=i)).timestamp()*1e9) for i in range(len(rows))])
    return PriceSession(date(2025, 1, day), start+timedelta(minutes=len(rows)), t, np.asarray(rows, dtype=float))


I = Instrument("X", 10, 1)
SPEC = replay_50k(eval_fee=105.2, reset_fee=105, contract_type="mini")


def fill(rows, scenario, risk=100, target=50, costs=0, compensate=True):
    s = session(rows)
    tape = prepare_execution([s], I, scenario)[0]
    details = {}
    resolved = _resolve(s, I, 1, F(risk), F(target), F(costs), slippage=scenario,
                        tape=tape, compensate_slippage=compensate, diagnostics=details)
    return resolved, details


def test_distribution_and_quantile_validation():
    with pytest.raises(ValueError): TickDistribution((.2, .3))
    with pytest.raises(ValueError): TickDistribution((float("nan"),))
    d = TickDistribution((.8, .15, .05))
    assert d.draw(0) == 0
    assert d.draw(.8) == 1
    assert d.quantile(.95) == 1
    assert d.quantile(1) == 2
    with pytest.raises(ValueError): d.draw(1)
    with pytest.raises(ValueError): model(volatility_window=0)
    with pytest.raises(ValueError): model(hour_multipliers=(1.,))


def test_stop_allowance_moves_trigger_but_preserves_budget():
    compensated, details = fill([[100, 101, 80, 95]], model(stop=2), costs=4)
    plain, _ = fill([[100, 101, 80, 95]], model(stop=2), costs=4, compensate=False)
    assert compensated[2] == 93  # 7 ticks + 2 anticipated execution ticks + $4
    assert compensated[4] == 91
    assert compensated[5] == 94
    assert compensated[7] == -94
    assert plain[7] == -114
    assert details["stop_allowance_ticks"] == 2


def test_entry_cost_is_in_fill_not_charged_twice_and_limit_has_no_slippage():
    result, details = fill([[100, 120, 100, 110]], model(market=1), risk=100, target=50, costs=4)
    assert result[1] == 101
    assert result[3] == 107
    assert result[4] == 107
    assert result[7] == 56
    assert details["entry_slippage_ticks"] == 1
    assert details["exit_slippage_ticks"] == 0
    assert result[0].low == -10  # mark-to-market cost at entry remains visible


def test_observed_gap_is_not_charged_twice():
    result, _ = fill([[100, 101, 99, 100], [85, 90, 80, 86]], model(stop=2))
    assert result[-2] == "stop_gap"
    assert result[4] == 83
    assert result[7] == -170


def test_forced_exit_uses_market_distribution():
    result, details = fill([[100, 103, 99, 102]], model(market=1), target=500)
    assert result[-2] == "session_close"
    assert result[1] == result[4] == 101
    assert details["exit_slippage_ticks"] == 1


def test_limit_trade_through_changes_fill_eligibility_not_limit_price():
    rows = [[100, 105, 99, 102]]
    touch, _ = fill(rows, model())
    through, _ = fill(rows, model(target_trade_through_ticks=1))
    assert touch[-2] == "target"
    assert through[-2] == "session_close"
    assert through[7] == 20


def test_current_and_future_bars_cannot_affect_entry_allowance():
    s = session([[100, 102, 99, 101], [101, 103, 100, 102], [102, 103, 101, 102]])
    altered = replace(s, ohlc=np.array([[100, 120, 80, 101], [101, 130, 70, 102], [102, 103, 101, 102]]))
    scenario = replace(model(stop=1), reference_range_ticks=1)
    a = prepare_execution([s], I, scenario)[0]
    b = prepare_execution([altered], I, scenario)[0]
    assert a.multipliers[0] == b.multipliers[0] == 1
    assert a.multipliers[1] == 3
    assert b.multipliers[1] == 8  # capped, using completed first bar only
    np.testing.assert_array_equal(a.uniforms, b.uniforms)


def test_warmup_is_causal_and_readonly():
    s = session([[100, 101, 99, 100]])
    wt = s.timestamps - 60_000_000_000
    warm = replace(s, warmup_timestamps=wt, warmup_ohlc=[[100, 105, 95, 100]])
    scenario = replace(model(), reference_range_ticks=2)
    assert prepare_execution([warm], I, scenario)[0].multipliers[0] == 5
    assert not warm.warmup_ohlc.flags.writeable
    with pytest.raises(ValueError, match="before entry"):
        replace(warm, warmup_timestamps=s.timestamps)


def test_common_draws_do_not_depend_on_entry_time_or_skipped_sessions():
    full = session([[100, 101, 99, 100]]*4)
    suffix = replace(full, timestamps=full.timestamps[2:], ohlc=full.ohlc[2:],
                     warmup_timestamps=full.timestamps[:2], warmup_ohlc=full.ohlc[:2])
    previous = session([[100, 101, 99, 100]], day=3)
    scenario = model()
    a = prepare_execution([full], I, scenario, seed=12, path=3)[0]
    b = prepare_execution([previous, suffix], I, scenario, seed=12, path=3)[1]
    np.testing.assert_array_equal(a.uniforms[2:], b.uniforms)
    assert a.stressed == b.stressed
    assert not a.uniforms.flags.writeable


def test_allowance_does_not_peek_at_realized_stress_and_tail_can_breach():
    scenario = replace(model(), stressed_stop=certain(20), stress_probability=.01)
    s = session([[100, 101, 80, 95]])
    tape = replace(prepare_execution([s], I, scenario)[0], stressed=True)
    details = {}
    out = _resolve(s, I, 1, F(100), F(50), F(0), slippage=scenario, tape=tape, diagnostics=details)
    assert details["stop_allowance_ticks"] == 0
    assert out[7] == -300
    # Canonical engine settles the actual overshoot; it is not clipped to risk.
    config = BacktestConfig(0, timedelta(0), timedelta(0), timedelta(0))
    extreme = replace(scenario, stressed_stop=certain(250), stress_probability=1, allowance_quantile=0)
    narrow = DollarPolicy.constant(100)
    path = replay_prices(SPEC, [s], narrow, {r.name:50 for r in narrow.regimes},
                         I, config, slippage=extreme, compensate_slippage=False, collision_policy="stop_first")
    assert path.replay.failed_attempts == 1
    assert path.decisions[0].net_pnl == -2600


def test_no_slippage_scenario_matches_existing_fills_and_changes_fingerprint():
    s = session([[100, 106, 99, 105]])
    config = BacktestConfig(0, timedelta(0), timedelta(0), timedelta(0))
    policy = DollarPolicy.constant(100)
    kwargs = dict(collision_policy="stop_first")
    args = (SPEC, [s], policy, {r.name:50 for r in policy.regimes}, I, config)
    old = replay_prices(*args, **kwargs)
    new = replay_prices(*args, slippage=model(), **kwargs)
    assert old.replay.events == new.replay.events
    assert old.decisions == new.decisions
    assert old.replay.history_fingerprint != new.replay.history_fingerprint


def test_fractional_timezone_offsets_do_not_share_wrong_hour_bucket():
    s = session([[100,101,99,100]]*60)
    hours = tuple(2. if h == 21 else 1. for h in range(24))
    scenario = model(timezone="Asia/Kolkata", hour_multipliers=hours)
    tape = prepare_execution([s], I, scenario)[0]
    assert tape.multipliers[0] == 1  # 15:00 UTC -> 20:30 local
    assert tape.multipliers[30] == 2  # 15:30 UTC -> 21:00 local


def test_fixed_multiple_contracts_scale_costs_and_allowance_once():
    scenario = model(stop=2)
    s = session([[100,101,80,95]])
    tape = prepare_execution([s], I, scenario)[0]
    result = _resolve(s, I, 3, F(300), F(150), F(3), slippage=scenario, tape=tape)
    assert result[2] == 93
    assert result[5] == 273
    assert result[7] == -273
