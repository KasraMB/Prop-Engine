"""Step 6 — single-path kernel + reference oracle (BUILD_SPEC Step 6, §12, §G6).

Every test runs BOTH the fast kernel and the pure-Python reference through
``_both``, which asserts they agree bit-for-bit (the Level-1 gate), and then
asserts the outcome against a hand-computed expectation (so a shared logic bug is
caught too, not just a transcription slip).

Convention: ``start_equity=0`` and unit size, so equity == cumulative P&L and the
drawdown floor sits at ``-amount`` — hand arithmetic stays trivial.
"""

from __future__ import annotations

import numpy as np
import pytest

from propfirm_engine.compiler import compile_phase
from propfirm_engine.enums import Action, ExitCode, Severity, StateField, Timing
from propfirm_engine.kernels import simulate_one_phase
from propfirm_engine.model import Phase
from propfirm_engine.reference import simulate_reference
from propfirm_engine.rules import (
    ConsistencyGateRule,
    ConsistencyRaisesTargetRule,
    DailyLossRule,
    MinimumTradingDaysRule,
    MinimumWinningDaysRule,
    ProfitTargetRule,
    StaticDrawdownRule,
    TrailingDrawdownRule,
)
from propfirm_engine.schema import PayoutSchema


def _phase(role, rules, schema=None):
    return compile_phase(Phase(role, role, tuple(rules), payout_schema=schema))


def _both(cp, ret, day, trade_low, *, start_equity=0.0, size_base=1.0, policy=(1.0,)):
    ret = np.asarray(ret, np.float64)
    day = np.asarray(day, np.int32)
    trade_low = np.asarray(trade_low, np.float64)
    pol = np.asarray(policy, np.float64)
    kc, ka, kd, ktd = simulate_one_phase(cp, ret, day, trade_low, size_base, pol, start_equity)
    r = simulate_reference(cp, ret, day, trade_low, size_base, pol, start_equity)
    # bitwise parity — exact float compare on payouts, exact int on code/days
    assert kc == r.code, f"code: kernel {kc} vs ref {r.code}"
    assert ka == r.payout_amounts, f"amounts: {ka} vs {r.payout_amounts}"
    assert kd == r.payout_days, f"days: {kd} vs {r.payout_days}"
    assert ktd == r.total_trading_days, f"total_days: {ktd} vs {r.total_trading_days}"
    return kc, ka, kd


# --- profit target / min days (PASS conjunction) ---------------------------- #


def test_profit_target_with_min_days_passes_only_when_both_hold():
    cp = _phase("eval", (ProfitTargetRule(3000.0), MinimumTradingDaysRule(3),
                         TrailingDrawdownRule(10_000.0)))
    # reach the target on day 0 and hold it; three distinct days elapse
    ret = [3000.0, 0.0, 0.0]
    day = [0, 1, 2]
    low = [0.0, 0.0, 0.0]
    code, _, _ = _both(cp, ret, day, low)
    assert code == int(ExitCode.PASSED)


def test_profit_target_hit_but_min_days_unmet_keeps_running():
    cp = _phase("eval", (ProfitTargetRule(3000.0), MinimumTradingDaysRule(5),
                         TrailingDrawdownRule(10_000.0)))
    ret = [3000.0, 0.0, 0.0]  # only 3 days, min is 5
    day = [0, 1, 2]
    low = [0.0, 0.0, 0.0]
    code, _, _ = _both(cp, ret, day, low)
    assert code == int(ExitCode.TIMED_OUT)  # never cleared, ran out of path


def test_profit_target_reached_exactly_passes():
    cp = _phase("eval", (ProfitTargetRule(3000.0), TrailingDrawdownRule(10_000.0)))
    code, _, _ = _both(cp, [3000.0], [0], [0.0])
    assert code == int(ExitCode.PASSED)  # equity - start == target exactly


# --- trailing drawdown breach boundaries ------------------------------------ #


def test_trailing_dd_breach_exactly_on_floor_and_one_tick_either_side():
    cp = _phase("eval", (ProfitTargetRule(10_000.0), TrailingDrawdownRule(1000.0)))
    # floor starts at -1000 (start 0). A trade taking the intraday low to exactly
    # -1000 breaches (<=); to -999.99 does not.
    on = _both(cp, [-1000.0], [0], [0.0])
    assert on[0] == int(ExitCode.FAIL_TRAILING_DD)
    below = _both(cp, [-1000.01], [0], [0.0])
    assert below[0] == int(ExitCode.FAIL_TRAILING_DD)
    above = _both(cp, [-999.99], [0], [0.0])
    assert above[0] == int(ExitCode.TIMED_OUT)


def test_trailing_dd_uses_intraday_low_for_continuous_check():
    # A winning trade that closes at +50 but floated to -1000 intraday breaches a
    # continuous-check floor; the close alone would miss it.
    cp = _phase("eval", (ProfitTargetRule(10_000.0), TrailingDrawdownRule(1000.0)))
    code, _, _ = _both(cp, [50.0], [0], [-1050.0])  # floating low = 50 + (-1050) = -1000
    assert code == int(ExitCode.FAIL_TRAILING_DD)


def test_breach_and_target_same_trade_fail_wins():
    # One trade both hits the target and breaches the DD (via intraday low): the
    # fail precedence (fail before pass) makes it a failure.
    cp = _phase("eval", (ProfitTargetRule(3000.0), TrailingDrawdownRule(1000.0)))
    # closes at +3000 (target) but floated to -1000 (breach)
    code, _, _ = _both(cp, [3000.0], [0], [-4000.0])
    assert code == int(ExitCode.FAIL_TRAILING_DD)


def test_breach_on_the_first_trade():
    cp = _phase("eval", (ProfitTargetRule(10_000.0), TrailingDrawdownRule(500.0)))
    code, _, _ = _both(cp, [-600.0, 100.0], [0, 0], [0.0, 0.0])
    assert code == int(ExitCode.FAIL_TRAILING_DD)


# --- soft breach (daily loss) ----------------------------------------------- #


def test_soft_breach_truncates_day_and_resumes_next():
    # Daily loss soft: after the breach on day 0, day 0's remaining trades are
    # skipped; the account survives into day 1.
    cp = _phase("eval", (ProfitTargetRule(3000.0), DailyLossRule(500.0),
                         TrailingDrawdownRule(10_000.0)))
    ret = [-600.0, -10_000.0, 3000.0]  # trade 1 breaches daily-loss; trade 2 (same day) skipped
    day = [0, 0, 1]
    low = [0.0, 0.0, 0.0]
    code, _, _ = _both(cp, ret, day, low)
    # day 0 truncated after -600 (the -10000 trade is skipped); day 1 +3000 -> equity
    # = -600 + 3000 = 2400, below target 3000 -> times out (survived the soft breach)
    assert code == int(ExitCode.TIMED_OUT)


def test_soft_breached_day_is_not_a_winning_day():
    # A soft-breached day must not count toward the qualifying-day counter even if
    # some earlier same-day profit existed.
    schema = PayoutSchema(dollar_cap=(2000.0,), split=1.0, max_payouts=1,
                          reset_fields=(StateField.N_QUALIFYING_DAYS,))
    cp = _phase("funded", (DailyLossRule(50.0), MinimumWinningDaysRule(1, 100.0),
                           TrailingDrawdownRule(10_000.0)), schema)
    # day 0: +150 then -60 (breaches daily loss 50) -> day truncated, day_pnl=90,
    # not >= 100 anyway, and flagged not-winning; no payout ever fires.
    ret = [150.0, -60.0]
    day = [0, 0]
    low = [0.0, 0.0]
    code, amounts, _ = _both(cp, ret, day, low)
    assert amounts == []  # no qualifying day -> no payout


# --- static drawdown -------------------------------------------------------- #


def test_static_drawdown_uses_fixed_floor():
    cp = _phase("eval", (ProfitTargetRule(10_000.0),
                         StaticDrawdownRule(1000.0, severity=Severity.HARD)))
    # static floor is start-1000 = -1000 and never trails
    code, _, _ = _both(cp, [500.0, -1600.0], [0, 0], [0.0, 0.0])  # equity 500 then -1100
    assert code == int(ExitCode.FAIL_STATIC_DD)


# --- payouts ---------------------------------------------------------------- #


def _funded_1day_payout(max_payouts=2, min_request=0.0, buffer_floor=0.0, cap=2000.0,
                        cap_fraction=1.0, split=0.9):
    schema = PayoutSchema(dollar_cap=(cap,), split=split, max_payouts=max_payouts,
                          min_request=min_request, buffer_floor=buffer_floor,
                          cap_fraction=cap_fraction,
                          reset_fields=(StateField.N_QUALIFYING_DAYS,))
    return _phase("funded", (TrailingDrawdownRule(100_000.0),
                             MinimumWinningDaysRule(1, 100.0)), schema)


def test_payout_fires_on_a_qualifying_day_and_reaches_max():
    cp = _funded_1day_payout(max_payouts=2)
    ret = [150.0, 150.0]
    day = [0, 1]
    low = [0.0, 0.0]
    code, amounts, days = _both(cp, ret, day, low, start_equity=100_000.0)
    assert code == int(ExitCode.MAXED_OUT)
    assert amounts == [135.0, 135.0]  # 0.9 * 150
    assert days == [0, 1]


def test_payout_below_min_request_does_not_fire():
    # Qualifies on winning-days but cycle profit (150) is below min_request (500):
    # no payout, no slot consumed, counter not reset -> keeps accumulating.
    cp = _funded_1day_payout(max_payouts=2, min_request=500.0)
    code, amounts, _ = _both(cp, [150.0], [0], [0.0], start_equity=100_000.0)
    assert amounts == []
    assert code == int(ExitCode.TIMED_OUT)


def test_payout_blocked_by_buffer_floor_does_not_fire():
    # Releasing would drop balance below the non-withdrawable buffer -> withheld.
    cp = _funded_1day_payout(max_payouts=2, buffer_floor=100_100.0)
    # start 100_000, +150 -> 100_150; releasing 150 -> 100_000 < buffer 100_100 -> blocked
    code, amounts, _ = _both(cp, [150.0], [0], [0.0], start_equity=100_000.0)
    assert amounts == []


def test_payout_at_exactly_min_request_fires():
    cp = _funded_1day_payout(max_payouts=1, min_request=150.0)
    code, amounts, _ = _both(cp, [150.0], [0], [0.0], start_equity=100_000.0)
    assert amounts == [135.0]
    assert code == int(ExitCode.MAXED_OUT)


def test_cap_fraction_limits_the_release():
    # cap_fraction 0.5: a 300 cycle releases 150 gross (not the 2000 dollar cap).
    cp = _funded_1day_payout(max_payouts=1, cap_fraction=0.5, split=1.0)
    code, amounts, _ = _both(cp, [300.0], [0], [0.0], start_equity=100_000.0)
    assert amounts == [150.0]  # 0.5 * 300


# --- MAXED_OUT vs PASSED ---------------------------------------------------- #


def test_reaching_max_payouts_returns_maxed_out_not_passed():
    cp = _funded_1day_payout(max_payouts=1)
    code, amounts, _ = _both(cp, [150.0], [0], [0.0], start_equity=100_000.0)
    assert code == int(ExitCode.MAXED_OUT)
    assert code != int(ExitCode.PASSED)


# --- sizing hook ------------------------------------------------------------ #


def test_length_one_policy_reproduces_constant_size():
    cp = _phase("eval", (ProfitTargetRule(3000.0), TrailingDrawdownRule(10_000.0)))
    # size_base 2 with policy [1.0] -> every trade scaled by 2
    ret = [1500.0]  # * size 2 = 3000 -> hits target
    code, _, _ = _both(cp, ret, [0], [0.0], size_base=2.0, policy=(1.0,))
    assert code == int(ExitCode.PASSED)


def test_eval_is_a_single_sizing_regime():
    # The policy conditions on phase: EVAL is one regime (index 0), so the four
    # funded multipliers (indices 1..4) never affect an eval attempt.
    cp = _phase("eval", (ProfitTargetRule(5000.0), TrailingDrawdownRule(100_000.0)))
    ret, day, low = [1000.0, 400.0], [0, 0], [0.0, 0.0]
    # only index 0 is used in eval: changing the funded slots leaves the outcome fixed
    a, _, _ = _both(cp, ret, day, low, policy=[2.0, 99.0, 99.0, 99.0, 99.0])
    b, _, _ = _both(cp, ret, day, low, policy=[2.0, 1.0, 1.0, 1.0, 1.0])
    assert a == b == int(ExitCode.TIMED_OUT)   # size x2 -> 2000+800=2800 < 5000
    # and index 0 genuinely sizes eval: x5 -> +5000 on the first trade -> PASS
    c, _, _ = _both(cp, ret, day, low, policy=[5.0, 1.0, 1.0, 1.0, 1.0])
    assert c == int(ExitCode.PASSED)


def test_funded_regimes_size_the_funded_path():
    # In the FUNDED phase the four regimes (1..4) drive sizing: shrinking them
    # changes the payouts vs full size, so the funded multipliers are live.
    schema = PayoutSchema(dollar_cap=(1000.0,), split=1.0, max_payouts=3,
                          cap_fraction=1.0, min_request=1.0)
    cp = _phase("funded", (TrailingDrawdownRule(100_000.0),
                           MinimumWinningDaysRule(1, 50.0)), schema=schema)
    ret, day, low = [100.0] * 6, [0, 1, 2, 3, 4, 5], [0.0] * 6
    big, big_amts, _ = _both(cp, ret, day, low, policy=[1.0, 1.0, 1.0, 1.0, 1.0])
    small, small_amts, _ = _both(cp, ret, day, low,
                                 policy=[1.0, 0.001, 0.001, 0.001, 0.001])
    assert big_amts != small_amts  # funded regimes size the funded path (parity holds)


# --- timing axes (§6a) ------------------------------------------------------ #


def test_eod_x_eod_breach_terminates_on_an_intermediate_day_not_only_the_last():
    # Pure end-of-day drawdown: floor and breach both at close. A breach at the
    # close of day 3 of a 5-day path must end the attempt on day 3, honored at the
    # intermediate rollover (§B1 at every close).
    r = TrailingDrawdownRule(1000.0, update_timing=Timing.EOD, check_timing=Timing.EOD)
    cp = _phase("eval", (ProfitTargetRule(100_000.0), r))
    # day0 close +500 (floor ratchets to -500); days1-2 flat; day3 closes at -600
    ret = [500.0, 0.0, 0.0, -1100.0, 999.0]
    day = [0, 1, 2, 3, 4]
    low = [0.0, 0.0, 0.0, 0.0, 0.0]
    code, _, _ = _both(cp, ret, day, low)
    assert code == int(ExitCode.FAIL_TRAILING_DD)


def test_eod_update_continuous_check_detects_intraday_breach_against_a_floor_that_trails_at_close():
    # The floor advances only at day close (EOD update), but a breach is detected
    # intraday against the day's low-water mark (continuous check).
    r = TrailingDrawdownRule(1000.0, update_timing=Timing.EOD, check_timing=Timing.CONTINUOUS)
    cp = _phase("eval", (ProfitTargetRule(100_000.0), r))
    # day0 closes +500 -> floor ratchets to -500 at close. day1 floats to -600
    # intraday (below the -500 floor) though it would close at +50.
    ret = [500.0, 50.0]
    day = [0, 1]
    low = [0.0, -1150.0]  # day1 floating low = 550 + (-1150) = -600
    code, _, _ = _both(cp, ret, day, low)
    assert code == int(ExitCode.FAIL_TRAILING_DD)


# --- consistency gate withholds a payout (§C8) ------------------------------ #


def test_consistency_gate_withholds_payout_until_the_ratio_clears():
    schema = PayoutSchema(dollar_cap=(2000.0,), split=1.0, max_payouts=1,
                          reset_fields=(StateField.N_QUALIFYING_DAYS,))
    cp = _phase("funded", (TrailingDrawdownRule(1_000_000.0),
                           MinimumWinningDaysRule(1, 100.0),
                           ConsistencyGateRule(0.5, gate=Action.PAYOUT)), schema)
    # day0 alone: max_day 1000 > 0.5*1000 -> withheld at day0 close.
    # day1: cycle profit 2000, max_day 1000 <= 0.5*2000 -> gate clears, payout fires.
    ret = [1000.0, 1000.0]
    day = [0, 1]
    low = [0.0, 0.0]
    code, amounts, days = _both(cp, ret, day, low)
    assert code == int(ExitCode.MAXED_OUT)
    assert amounts == [2000.0]
    assert days == [1]  # NOT day 0 — the single big day was withheld


# --- EOD consistency-raises-target adjust (§12) ----------------------------- #


def test_eod_adjust_raises_the_target_at_day_close_changing_the_pass():
    adjust = ConsistencyRaisesTargetRule(0.5, raise_to=6000.0, check_timing=Timing.EOD)
    with_adjust = _phase("eval", (ProfitTargetRule(3000.0), adjust,
                                  TrailingDrawdownRule(100_000.0)))
    no_adjust = _phase("eval", (ProfitTargetRule(3000.0), TrailingDrawdownRule(100_000.0)))
    ret = [2000.0, 2000.0]  # reaches 4000 total
    day = [0, 1]
    low = [0.0, 0.0]
    # day0 (+2000) violates consistency at close -> target raised to 6000; 4000 < 6000
    with_code, _, _ = _both(with_adjust, ret, day, low)
    assert with_code == int(ExitCode.TIMED_OUT)
    # without the adjust, 4000 >= the original 3000 target -> passes
    no_code, _, _ = _both(no_adjust, ret, day, low)
    assert no_code == int(ExitCode.PASSED)


# --- post-payout transition: withdrawal + floor recompute ------------------- #


def _funded_recompute(recompute):
    schema = PayoutSchema(dollar_cap=(500.0,), split=1.0, max_payouts=5,
                          reset_fields=(StateField.N_QUALIFYING_DAYS,),
                          withdraw_reduces_equity=True,
                          recompute_floor_on_payout=recompute)
    return _phase("funded", (TrailingDrawdownRule(1000.0),
                             MinimumWinningDaysRule(1, 100.0)), schema)


def test_floor_recompute_on_payout_lowers_the_floor_and_survives():
    cp = _funded_recompute(recompute=True)
    # day0 +600 -> payout 500 at close; recompute drops the floor to (equity-1000).
    # day1 -600 lands at 99_500, below the OLD floor (99_600) but above the new one.
    ret = [600.0, -600.0]
    day = [0, 1]
    low = [0.0, 0.0]
    code, amounts, _ = _both(cp, ret, day, low, start_equity=100_000.0)
    assert amounts == [500.0]
    assert code == int(ExitCode.TIMED_OUT)  # survived on the recomputed floor


def test_payout_immediately_followed_by_a_breach_without_recompute():
    cp = _funded_recompute(recompute=False)
    # same path, but the floor is NOT recomputed, so 99_500 breaches the old floor.
    ret = [600.0, -600.0]
    day = [0, 1]
    low = [0.0, 0.0]
    code, amounts, _ = _both(cp, ret, day, low, start_equity=100_000.0)
    assert amounts == [500.0]  # the payout still fired first
    assert code == int(ExitCode.FAIL_TRAILING_DD)  # then the very next trade breached


# --- trailing-floor lock transition (§6a) ----------------------------------- #


def test_trailing_floor_locks_and_stops_trailing():
    locked = _phase("eval", (ProfitTargetRule(1_000_000.0),
                             TrailingDrawdownRule(1000.0, lock_at=-100.0)))
    unlocked = _phase("eval", (ProfitTargetRule(1_000_000.0),
                               TrailingDrawdownRule(1000.0)))  # never locks
    # rise to 900 (floor reaches -100 and locks), rise to 2000, drop to -50.
    ret = [900.0, 1100.0, -2050.0]
    day = [0, 0, 0]
    low = [0.0, 0.0, 0.0]
    # locked: floor stays -100, so -50 does not breach
    lcode, _, _ = _both(locked, ret, day, low)
    assert lcode == int(ExitCode.TIMED_OUT)
    # unlocked: floor trails to 1000, so -50 breaches
    ucode, _, _ = _both(unlocked, ret, day, low)
    assert ucode == int(ExitCode.FAIL_TRAILING_DD)


# --- breach on the final trade of a day ------------------------------------- #


def test_breach_on_the_final_trade_of_a_day():
    cp = _phase("eval", (ProfitTargetRule(100_000.0), TrailingDrawdownRule(1000.0)))
    ret = [100.0, -1200.0]  # second (final) trade of day 0 breaches
    day = [0, 0]
    low = [0.0, 0.0]
    code, _, _ = _both(cp, ret, day, low)
    assert code == int(ExitCode.FAIL_TRAILING_DD)


# --- trade_low clipping tie-in (§D1) ---------------------------------------- #


def test_clipped_trade_low_does_not_cause_a_spurious_breach():
    cp = _phase("eval", (ProfitTargetRule(100_000.0), TrailingDrawdownRule(1000.0)))
    # a winning trade with a small (properly clipped) floating low does not breach;
    # the same trade with an unclipped, deeper low would.
    ok, _, _ = _both(cp, [50.0], [0], [-50.0])  # floating low 0 -> safe
    assert ok == int(ExitCode.TIMED_OUT)
    breach, _, _ = _both(cp, [50.0], [0], [-1100.0])  # floating low -1050 -> breach
    assert breach == int(ExitCode.FAIL_TRAILING_DD)


# --- more payout boundaries (§6b) ------------------------------------------- #


def test_zero_cycle_profit_does_not_fire_a_dollar_zero_payout():
    # The qualifying conjunction can be met (2 winning days) while cycle profit is
    # exactly 0 — a $0 payout must NOT fire, must not burn a max_payouts slot, and
    # must not reset the counter (§6b). Reachable with discrete P&L.
    schema = PayoutSchema(dollar_cap=(2000.0,), split=0.9, max_payouts=2,
                          reset_fields=(StateField.N_QUALIFYING_DAYS,))
    cp = _phase("funded", (TrailingDrawdownRule(1_000_000.0),
                           MinimumWinningDaysRule(2, 100.0)), schema)
    ret = [100.0, -200.0, 100.0]  # cycle profit nets to exactly 0 by day 2 close
    day = [0, 1, 2]
    low = [0.0, 0.0, 0.0]
    code, amounts, _ = _both(cp, ret, day, low, start_equity=100_000.0)
    assert amounts == []  # no $0 payout recorded
    assert code == int(ExitCode.TIMED_OUT)


def test_payout_gross_binds_exactly_on_the_dollar_cap():
    # cap_fraction*cycle_profit == the dollar cap exactly: gross lands on the cap.
    cp = _funded_1day_payout(max_payouts=1, cap=500.0, cap_fraction=1.0, split=1.0)
    code, amounts, _ = _both(cp, [500.0], [0], [0.0], start_equity=100_000.0)
    assert amounts == [500.0]  # min(500 cap, 1.0*500) == 500
    assert code == int(ExitCode.MAXED_OUT)


def test_tiered_split_pays_the_first_tier_below_the_cumulative_threshold():
    schema = PayoutSchema(dollar_cap=(500.0,), split=0.9, max_payouts=1,
                          split_first_tier=1.0, split_tier_cap=1000.0,
                          reset_fields=(StateField.N_QUALIFYING_DAYS,))
    cp = _phase("funded", (TrailingDrawdownRule(1_000_000.0),
                           MinimumWinningDaysRule(1, 100.0)), schema)
    code, amounts, _ = _both(cp, [500.0], [0], [0.0], start_equity=100_000.0)
    # cumulative_paid (0) < tier_cap (1000) -> 100% first tier, not the 0.9 split
    assert amounts == [500.0]


def test_soft_breach_day_close_is_evaluated_by_an_eod_rule():
    # A day that soft-breaches (daily loss) AND carries an EOD trailing rule: the
    # EOD rule evaluates against the TRUE closing equity (equity after the last
    # executed trade, = the breach trade), not the intraday low (§C5).
    cp = _phase("eval", (ProfitTargetRule(1_000_000.0),
                         DailyLossRule(500.0, severity=Severity.SOFT),
                         TrailingDrawdownRule(500.0, update_timing=Timing.EOD,
                                              check_timing=Timing.EOD)))
    ret = [200.0, -800.0]  # day 0: +200 then -800 -> day_pnl -600 breaches daily loss
    day = [0, 0]
    low = [0.0, 0.0]
    # soft breach truncates at closing equity -600; EOD floor is -500 -> -600 <= -500
    # -> the EOD trailing rule breaches at that truncated close.
    code, _, _ = _both(cp, ret, day, low)
    assert code == int(ExitCode.FAIL_TRAILING_DD)


def test_consistency_gate_on_a_pass_counts_the_in_progress_day():
    # An eval whose PASS = profit target AND a consistency gate. Hitting the target
    # on a dominant day must NOT pass intraday using a stale max_day_pnl: the
    # in-progress day's running P&L is a candidate for the biggest day (§C8).
    cp = _phase("eval", (ProfitTargetRule(3000.0),
                         ConsistencyGateRule(0.5, gate=Action.PASS),
                         TrailingDrawdownRule(100_000.0)))
    # day0 +500, day1 +2500 -> target hit on day1, but day1 is 2500/3000 = 83% > 50%
    dominant, _, _ = _both(cp, [500.0, 2500.0], [0, 1], [0.0, 0.0])
    assert dominant == int(ExitCode.TIMED_OUT)  # consistency blocks the pass
    # two balanced days (1500 each = 50% exactly) DO clear it
    balanced, _, _ = _both(cp, [1500.0, 1500.0], [0, 1], [0.0, 0.0])
    assert balanced == int(ExitCode.PASSED)


def test_consistency_gate_blocks_the_payout_on_the_day_its_own_pnl_tips_the_ratio():
    # A single huge day: at its close the fold makes max_day_pnl == cycle_profit,
    # so the ratio is violated and the payout is withheld; the account continues.
    schema = PayoutSchema(dollar_cap=(2000.0,), split=1.0, max_payouts=1,
                          reset_fields=(StateField.N_QUALIFYING_DAYS,))
    cp = _phase("funded", (TrailingDrawdownRule(1_000_000.0),
                           MinimumWinningDaysRule(1, 100.0),
                           ConsistencyGateRule(0.5, gate=Action.PAYOUT)), schema)
    code, amounts, _ = _both(cp, [1000.0], [0], [0.0], start_equity=100_000.0)
    assert amounts == []  # 1000 > 0.5*1000 -> withheld
    assert code == int(ExitCode.TIMED_OUT)


# --- randomized fuzz parity (the strongest Level-1 evidence, §G6) ----------- #


def _fuzz_configs():
    """A spread of compiled phases exercising every rule kind and timing combo."""
    schema = PayoutSchema(dollar_cap=(2000.0, 2500.0), split=0.9, max_payouts=3,
                          min_request=200.0, cap_fraction=0.5,
                          reset_fields=(StateField.N_QUALIFYING_DAYS,),
                          recompute_floor_on_payout=True)
    return [
        _phase("eval", (ProfitTargetRule(2000.0), MinimumTradingDaysRule(2),
                        TrailingDrawdownRule(1500.0), DailyLossRule(400.0))),
        _phase("eval", (ProfitTargetRule(3000.0),
                        StaticDrawdownRule(1200.0),
                        ConsistencyRaisesTargetRule(0.5, raise_to=5000.0,
                                                    check_timing=Timing.EOD))),
        _phase("eval", (ProfitTargetRule(2500.0),
                        TrailingDrawdownRule(1000.0, update_timing=Timing.EOD,
                                             check_timing=Timing.CONTINUOUS))),
        _phase("funded", (TrailingDrawdownRule(2000.0, lock_at=-500.0),
                          MinimumWinningDaysRule(2, 150.0),
                          ConsistencyGateRule(0.6, gate=Action.PAYOUT),
                          DailyLossRule(600.0)), schema),
    ]


@pytest.mark.parametrize("seed", range(60))
def test_kernel_matches_reference_on_random_paths(seed):
    rng = np.random.default_rng(seed)
    n_days = int(rng.integers(1, 9))
    ret, day, low = [], [], []
    for d in range(n_days):
        for _ in range(int(rng.integers(1, 6))):
            r = float(rng.normal(0.0, 500.0))
            ret.append(r)
            day.append(d)
            low.append(-abs(float(rng.normal(0.0, 300.0))))  # a downward floating low
    cfgs = _fuzz_configs()
    cp = cfgs[seed % len(cfgs)]
    # vary the sizing policy on some seeds so stage-varying sizing is exercised
    # alongside payouts/consistency/timing (not just against trivial rule sets).
    if seed % 3 == 0:
        policy = (1.0,)
    else:
        policy = (0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5)  # IN_PROFIT bit -> size up
    # _both asserts kernel and reference agree bit-for-bit on code + payouts.
    _both(cp, ret, day, low, start_equity=100_000.0, policy=policy)
