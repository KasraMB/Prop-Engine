"""Step 5 — the compiler (BUILD_SPEC Step 5, ARCHITECTURE §6b, §8)."""

from __future__ import annotations

import numpy as np

from propfirm_engine.compiler import (
    NONE_CODE,
    compile_account,
    compile_phase,
    resolve_requirements,
)
from propfirm_engine.enums import Action, ExitCode, Severity, StateField, Timing
from propfirm_engine.model import Account, Phase
from propfirm_engine.rules import (
    NEVER_LOCK,
    ConsistencyGateRule,
    ConsistencyRaisesTargetRule,
    DailyLossRule,
    MinimumTradingDaysRule,
    MinimumWinningDaysRule,
    ProfitTargetRule,
    RuleKind,
    StaticDrawdownRule,
    TrailingDrawdownRule,
)
from propfirm_engine.schema import PayoutSchema


# --- requirements resolver (§8) --------------------------------------------- #


def test_live_state_is_the_union_plus_driving_fields():
    ph = Phase(
        "eval",
        "eval",
        (ProfitTargetRule(3000.0), TrailingDrawdownRule(2500.0), MinimumTradingDaysRule(3)),
    )
    live = resolve_requirements(ph)
    # driving fields always present
    assert StateField.EQUITY in live
    assert StateField.DAY_INDEX in live
    # union of the rules' own requirements
    assert StateField.PEAK_EQUITY in live  # trailing DD
    assert StateField.DD_FLOOR in live
    assert StateField.N_TRADING_DAYS in live  # min-days


def test_only_computes_what_is_required():
    # No consistency rule -> no consistency-only state; no winning-days rule -> no
    # qualifying-day counter.
    ph = Phase("eval", "eval", (ProfitTargetRule(3000.0), TrailingDrawdownRule(2500.0)))
    live = resolve_requirements(ph)
    assert StateField.MAX_DAY_PNL not in live
    assert StateField.N_QUALIFYING_DAYS not in live


def test_shared_state_field_is_not_double_counted_and_union_is_order_independent():
    # Two rules that both read EQUITY produce ONE entry; permuting rule order does
    # not change the resolved set.
    a = Phase("eval", "eval", (ProfitTargetRule(3000.0), TrailingDrawdownRule(2500.0)))
    b = Phase("eval", "eval", (TrailingDrawdownRule(2500.0), ProfitTargetRule(3000.0)))
    assert resolve_requirements(a) == resolve_requirements(b)

    # Non-tautological de-dup check: EQUITY is required by BOTH rules individually,
    # yet the resolved union is strictly smaller than the naive concatenation of
    # every rule's requirements plus the two driving fields — proving overlap was
    # collapsed, not just that a set holds no duplicates.
    live = resolve_requirements(a)
    occurrences = sum(
        1 for r in a.rules for f in r.requirements() if f == StateField.EQUITY
    )
    assert occurrences >= 2  # EQUITY genuinely required more than once
    naive = sum(len(r.requirements()) for r in a.rules) + 2  # + driving fields
    assert len(live) < naive


# --- struct-of-arrays: one aligned entry per rule, order preserved ---------- #


def test_emitted_arrays_have_one_entry_per_rule_in_order():
    rules = (
        ProfitTargetRule(3000.0),  # PASS
        TrailingDrawdownRule(2500.0),  # FAIL/HARD
        DailyLossRule(1000.0),  # FAIL/SOFT
        MinimumWinningDaysRule(5, 150.0),  # PAYOUT
    )
    cp = compile_phase(Phase("funded", "funded", rules,
                             payout_schema=PayoutSchema(dollar_cap=(2000.0,), split=0.9, max_payouts=5)))
    assert cp.n_rules == 4
    # kind array preserves rule order exactly (precedence is order-defined)
    np.testing.assert_array_equal(
        cp.kind,
        [
            int(RuleKind.PROFIT_TARGET),
            int(RuleKind.TRAILING_DD),
            int(RuleKind.DAILY_LOSS),
            int(RuleKind.MIN_WINNING_DAYS),
        ],
    )
    np.testing.assert_array_equal(
        cp.action,
        [int(Action.PASS), int(Action.FAIL), int(Action.FAIL), int(Action.PAYOUT)],
    )


def test_compiled_parameters_and_codes_match_the_rules():
    rules = (
        ProfitTargetRule(3000.0),
        TrailingDrawdownRule(2500.0, lock_at=1000.0, severity=Severity.HARD),
        DailyLossRule(750.0, severity=Severity.SOFT),
        ConsistencyRaisesTargetRule(0.4, raise_to=6000.0),
    )
    cp = compile_phase(Phase("eval", "eval", rules))
    # p0/p1 carried
    np.testing.assert_allclose(cp.p0, [3000.0, 2500.0, 750.0, 0.4])
    np.testing.assert_allclose(cp.p1, [0.0, 1000.0, 0.0, 6000.0])
    # severities
    assert cp.severity[1] == int(Severity.HARD)
    assert cp.severity[2] == int(Severity.SOFT)
    # fail codes present for FAIL rules, sentinel elsewhere
    assert cp.fail_code[1] == int(ExitCode.FAIL_TRAILING_DD)
    assert cp.fail_code[2] == int(ExitCode.FAIL_DAILY_LOSS)
    assert cp.fail_code[0] == NONE_CODE  # profit target is a PASS
    # adjust field named only for the ADJUST rule
    assert cp.adjust_field[3] == int(StateField.PROFIT_TARGET)
    assert cp.adjust_field[0] == NONE_CODE


def test_per_action_index_groupings_are_correct_and_order_preserving():
    rules = (
        DailyLossRule(1000.0),  # 0 FAIL
        ProfitTargetRule(3000.0),  # 1 PASS
        TrailingDrawdownRule(2500.0),  # 2 FAIL
        MinimumWinningDaysRule(5, 150.0),  # 3 PAYOUT
        MinimumTradingDaysRule(3),  # 4 PASS
    )
    cp = compile_phase(Phase("funded", "funded", rules,
                             payout_schema=PayoutSchema(dollar_cap=(2000.0,), split=0.9, max_payouts=5)))
    np.testing.assert_array_equal(cp.fail_idx, [0, 2])
    np.testing.assert_array_equal(cp.pass_idx, [1, 4])
    np.testing.assert_array_equal(cp.payout_idx, [3])
    assert cp.adjust_idx.size == 0
    # timing defaults are carried (continuous where irrelevant)
    assert cp.check_timing[0] == int(Timing.CONTINUOUS)


def test_timing_axes_round_trip():
    r = TrailingDrawdownRule(2500.0, update_timing=Timing.EOD, check_timing=Timing.CONTINUOUS)
    cp = compile_phase(Phase("eval", "eval", (ProfitTargetRule(3000.0), r)))
    assert cp.update_timing[1] == int(Timing.EOD)
    assert cp.check_timing[1] == int(Timing.CONTINUOUS)


def test_never_lock_sentinel_survives_as_infinity_in_p1():
    # A trailing rule that never locks carries p1 = +inf (float64); a regression
    # that int-cast or truncated it would break the "stop trailing" branch.
    cp = compile_phase(Phase("eval", "eval", (ProfitTargetRule(3000.0), TrailingDrawdownRule(2500.0))))
    assert np.isinf(cp.p1[1])
    assert cp.lock_at == NEVER_LOCK


def test_consistency_gate_cushion_rides_in_p1_and_action_selects_grouping():
    rules = (
        TrailingDrawdownRule(2500.0),
        ConsistencyGateRule(0.4, gate=Action.PAYOUT, cushion=0.05),
    )
    cp = compile_phase(
        Phase("funded", "funded", rules,
              payout_schema=PayoutSchema(dollar_cap=(2000.0,), split=0.9, max_payouts=5))
    )
    assert cp.p0[1] == 0.4  # threshold
    assert cp.p1[1] == 0.05  # cushion
    # gate=PAYOUT -> the rule lands in payout_idx, not pass_idx
    np.testing.assert_array_equal(cp.payout_idx, [1])
    assert cp.pass_idx.size == 0


def test_static_drawdown_compiles_with_its_fail_code():
    cp = compile_phase(
        Phase("eval", "eval", (ProfitTargetRule(3000.0), StaticDrawdownRule(2000.0)))
    )
    assert cp.kind[1] == int(RuleKind.STATIC_DD)
    assert cp.fail_code[1] == int(ExitCode.FAIL_STATIC_DD)
    # static DD stores no trailing reference, so it does not set dd_amount
    assert np.isinf(cp.dd_amount)


def test_adjust_idx_is_populated_for_an_adjust_rule():
    rules = (ProfitTargetRule(3000.0), ConsistencyRaisesTargetRule(0.4, raise_to=6000.0))
    cp = compile_phase(Phase("eval", "eval", rules))
    np.testing.assert_array_equal(cp.adjust_idx, [1])


# --- pre-extracted scalar seeds (§12 kernel inputs) ------------------------- #


def test_scalar_seeds_are_extracted_from_the_rules():
    rules = (ProfitTargetRule(3000.0), TrailingDrawdownRule(2500.0, lock_at=1000.0))
    cp = compile_phase(Phase("eval", "eval", rules))
    assert cp.profit_target0 == 3000.0
    assert cp.dd_amount == 2500.0
    assert cp.lock_at == 1000.0


def test_absent_rules_leave_inert_scalar_seeds():
    # A funded phase with only a static DD + payout: no profit target, no trailing
    # DD -> the seeds are the inert +inf sentinels (checks never fire spuriously).
    cp = compile_phase(
        Phase("funded", "funded", (StaticDrawdownRule(2000.0), MinimumWinningDaysRule(5, 150.0)),
              payout_schema=PayoutSchema(dollar_cap=(2000.0,), split=0.9, max_payouts=5))
    )
    assert np.isinf(cp.profit_target0)
    assert np.isinf(cp.dd_amount)


# --- payout schema round-trip (§6b) ----------------------------------------- #


def test_payout_schema_round_trips_all_fields():
    schema = PayoutSchema(
        dollar_cap=(2000.0, 2500.0),
        split=0.9,
        max_payouts=5,
        cap_fraction=0.5,
        min_request=500.0,
        buffer_floor=52_100.0,
        reset_fields=(StateField.N_QUALIFYING_DAYS,),
        withdraw_reduces_equity=True,
        recompute_floor_on_payout=True,
    )
    ph = Phase("funded", "funded", (MinimumWinningDaysRule(5, 150.0),), payout_schema=schema)
    cp = compile_phase(ph)
    assert cp.payout is not None
    np.testing.assert_allclose(cp.payout.dollar_cap, [2000.0, 2500.0])
    assert cp.payout.cap_fraction == 0.5
    assert cp.payout.min_request == 500.0
    assert cp.payout.buffer_floor == 52_100.0
    assert cp.payout.split == 0.9
    assert cp.payout.max_payouts == 5
    assert cp.payout.recompute_floor_on_payout is True
    assert cp.payout.withdraw_reduces_equity is True
    np.testing.assert_array_equal(cp.payout.reset_fields, [int(StateField.N_QUALIFYING_DAYS)])


def test_stepping_dollar_cap_last_element_repeats():
    schema = PayoutSchema(dollar_cap=(2000.0, 2500.0), split=0.9, max_payouts=5)
    cp = compile_phase(Phase("funded", "funded", (MinimumWinningDaysRule(5, 150.0),),
                             payout_schema=schema))
    assert cp.payout.dollar_cap_at(0) == 2000.0
    assert cp.payout.dollar_cap_at(1) == 2500.0
    assert cp.payout.dollar_cap_at(9) == 2500.0  # last repeats


def test_tiered_split_compiles_to_a_uniform_two_tier_form():
    tiered = PayoutSchema(
        dollar_cap=(2000.0,), split=0.9, max_payouts=5,
        split_first_tier=1.0, split_tier_cap=10_000.0,
    )
    cp = compile_phase(Phase("funded", "funded", (MinimumWinningDaysRule(5, 150.0),),
                             payout_schema=tiered))
    assert cp.payout.first_tier_split == 1.0
    assert cp.payout.tier_cap == 10_000.0


def test_flat_split_compiles_to_an_inert_tier():
    flat = PayoutSchema(dollar_cap=(2000.0,), split=0.9, max_payouts=5)
    cp = compile_phase(Phase("funded", "funded", (MinimumWinningDaysRule(5, 150.0),),
                             payout_schema=flat))
    # no legacy tier -> tier is inert: first_tier_split == split, tier_cap == 0
    assert cp.payout.first_tier_split == 0.9
    assert cp.payout.tier_cap == 0.0


def test_eval_phase_has_no_compiled_payout_schema():
    cp = compile_phase(Phase("eval", "eval", (ProfitTargetRule(3000.0),)))
    assert cp.payout is None


# --- whole-account compilation ---------------------------------------------- #


def test_compile_account_carries_metadata_and_phases():
    schema = PayoutSchema(dollar_cap=(2000.0,), split=0.9, max_payouts=5)
    acct = Account(
        "50K",
        50_000,
        phases=(
            Phase("eval", "eval", (ProfitTargetRule(3000.0), TrailingDrawdownRule(2500.0))),
            Phase("funded", "funded", (TrailingDrawdownRule(2500.0), MinimumWinningDaysRule(5, 150.0)),
                  payout_schema=schema),
        ),
        eval_fee=150.0,
        activation_fee=25.0,
    )
    ca = compile_account(acct)
    assert ca.name == "50K"
    assert ca.size == 50_000
    assert ca.eval_fee == 150.0
    assert ca.activation_fee == 25.0
    assert tuple(p.role for p in ca.phases) == ("eval", "funded")
    assert ca.phases[0].payout is None
    assert ca.phases[1].payout is not None
