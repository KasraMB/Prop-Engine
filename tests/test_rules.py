"""Rule base (Step 1) and the concrete rules + registry (Step 2).

The abstract base ships in Step 1, so its two documented guarantees are tested
here: (1) a half-finished rule fails at *instantiation* (the ``@abstractmethod``
guard), and (2) a concrete frozen rule is immutable and hashable by value — the
precondition the fingerprint (§10) relies on. The remainder of the file covers
the Step 2 contract: concrete rule types, their compiled records, and the
``RULE_REGISTRY`` hard-fail (BUILD_SPEC Step 2).
"""

import dataclasses
import math

import pytest

from propfirm_engine.enums import Action, ExitCode, Severity, StateField, Timing
from propfirm_engine.rules import (
    NEVER_LOCK,
    RULE_REGISTRY,
    CompiledRule,
    ConsistencyRaisesTargetRule,
    ConsistencyGateRule,
    DailyLossRule,
    MinimumTradingDaysRule,
    MinimumWinningDaysRule,
    ProfitTargetRule,
    Rule,
    RuleKind,
    StaticDrawdownRule,
    TrailingDrawdownRule,
    UnknownRuleError,
    assert_kernel_supports,
)


def test_abstract_rule_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Rule()  # type: ignore[abstract]


def test_rule_missing_a_required_method_cannot_be_instantiated():
    @dataclasses.dataclass(frozen=True)
    class HalfFinished(Rule):
        # implements requirements() but NOT compile() -> still abstract
        def requirements(self):
            return (StateField.EQUITY,)

    with pytest.raises(TypeError):
        HalfFinished()  # type: ignore[abstract]


def _make_concrete():
    @dataclasses.dataclass(frozen=True)
    class DummyRule(Rule):
        amount: float

        def requirements(self):
            return (StateField.EQUITY,)

        def compile(self):
            return ("dummy", self.amount)

    return DummyRule


def test_concrete_rule_is_frozen_and_hashable_by_value():
    DummyRule = _make_concrete()
    a = DummyRule(2500.0)
    b = DummyRule(2500.0)
    # frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.amount = 1.0  # type: ignore[misc]
    # hashable by value: equal instances are interchangeable in sets/dicts
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
    # a differing parameter is a different value (same type)
    c = DummyRule(1000.0)
    assert a != c
    assert type(a) is type(c)


def test_concrete_rule_reports_its_requirements():
    DummyRule = _make_concrete()
    assert DummyRule(2500.0).requirements() == (StateField.EQUITY,)


# --------------------------------------------------------------------------- #
# Step 2 — concrete rules and the registry (BUILD_SPEC Step 2).                #
# --------------------------------------------------------------------------- #

# One representative instance of every registered kind, for parametrized sweeps.
_ALL_RULES: tuple[Rule, ...] = (
    ProfitTargetRule(3000.0),
    TrailingDrawdownRule(2500.0),
    StaticDrawdownRule(2000.0),
    DailyLossRule(1000.0),
    MinimumTradingDaysRule(3),
    MinimumWinningDaysRule(count=5, threshold=150.0),
    ConsistencyRaisesTargetRule(0.5, raise_to=6000.0),
    ConsistencyGateRule(0.5, gate=Action.PASS),
)


# --- "a new number is a new instance, not a new type" ----------------------- #


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (ProfitTargetRule(3000.0), ProfitTargetRule(3500.0)),
        (TrailingDrawdownRule(2500.0), TrailingDrawdownRule(2600.0)),
        (DailyLossRule(1000.0), DailyLossRule(1200.0)),
        (ConsistencyGateRule(0.4), ConsistencyGateRule(0.5)),
        (
            MinimumWinningDaysRule(5, 150.0),
            MinimumWinningDaysRule(5, 200.0),
        ),
    ],
)
def test_numeric_difference_is_same_type_unequal_value(a, b):
    assert type(a) is type(b)
    assert a != b
    # the differing number must also reach the compiled record (not just the DSL
    # object): two rules unequal by value compile to unequal records.
    assert a.compile() != b.compile()


def test_non_numeric_fields_participate_in_equality():
    # Step 2 tests otherwise only vary a number; severity/timing/gate must also be
    # part of identity, or Step 7's fingerprint (which hashes them) is unsound.
    assert DailyLossRule(1000.0, severity=Severity.SOFT) != DailyLossRule(
        1000.0, severity=Severity.HARD
    )
    assert TrailingDrawdownRule(2500.0, check_timing=Timing.EOD) != TrailingDrawdownRule(
        2500.0, check_timing=Timing.CONTINUOUS
    )
    assert TrailingDrawdownRule(2500.0, update_timing=Timing.EOD) != TrailingDrawdownRule(
        2500.0, update_timing=Timing.CONTINUOUS
    )
    assert TrailingDrawdownRule(2500.0, lock_at=None) != TrailingDrawdownRule(
        2500.0, lock_at=0.0
    )
    assert ConsistencyGateRule(0.5, gate=Action.PASS) != ConsistencyGateRule(
        0.5, gate=Action.PAYOUT
    )


def test_identical_rules_are_equal_and_hash_equal():
    assert ConsistencyGateRule(0.5) == ConsistencyGateRule(0.5)
    assert hash(ConsistencyGateRule(0.5)) == hash(ConsistencyGateRule(0.5))
    assert len({TrailingDrawdownRule(2500.0), TrailingDrawdownRule(2500.0)}) == 1


# --- requirements are consistent with behaviour ----------------------------- #


def test_trailing_dd_requires_equity_and_floor_state():
    reqs = TrailingDrawdownRule(2500.0).requirements()
    assert StateField.EQUITY in reqs
    assert StateField.PEAK_EQUITY in reqs
    assert StateField.DD_FLOOR in reqs


def test_trailing_dd_needs_intraday_low_only_when_checked_continuously():
    intraday = TrailingDrawdownRule(2500.0, check_timing=Timing.CONTINUOUS)
    eod = TrailingDrawdownRule(2500.0, check_timing=Timing.EOD)
    assert StateField.DAY_LOW in intraday.requirements()
    assert StateField.DAY_LOW not in eod.requirements()


def test_trailing_dd_needs_lock_state_only_when_it_can_lock():
    assert StateField.DD_LOCKED not in TrailingDrawdownRule(2500.0).requirements()
    assert StateField.DD_LOCKED in TrailingDrawdownRule(2500.0, lock_at=0.0).requirements()


def test_static_dd_needs_intraday_low_only_when_checked_continuously():
    intraday = StaticDrawdownRule(2000.0, check_timing=Timing.CONTINUOUS)
    eod = StaticDrawdownRule(2000.0, check_timing=Timing.EOD)
    assert StateField.DAY_LOW in intraday.requirements()
    assert StateField.DAY_LOW not in eod.requirements()


def test_exact_requirements_for_the_simple_rules():
    # These three carry a single required field; pin the exact tuple so a swap to
    # the wrong StateField (which the isinstance sweep would miss) is caught.
    assert ProfitTargetRule(3000.0).requirements() == (StateField.EQUITY,)
    assert MinimumTradingDaysRule(3).requirements() == (StateField.N_TRADING_DAYS,)
    assert DailyLossRule(1000.0).requirements() == (StateField.DAY_PNL,)


def test_static_dd_stores_no_trailing_reference():
    # Static DD recomputes start - amount inline (§8): it must NOT pull in the
    # trailing-reference state, or it would falsely widen the live-state union.
    reqs = StaticDrawdownRule(2000.0).requirements()
    assert StateField.EQUITY in reqs
    assert StateField.PEAK_EQUITY not in reqs
    assert StateField.DD_FLOOR not in reqs
    assert StateField.DD_LOCKED not in reqs


def test_min_winning_days_requires_the_qualifying_counter():
    assert MinimumWinningDaysRule(5, 150.0).requirements() == (
        StateField.N_QUALIFYING_DAYS,
    )


def test_consistency_adjust_reads_the_full_consistency_state_plus_target():
    # It must read PROFIT_TARGET (it mutates it) on top of the *whole* consistency
    # state — pin the exact set, not just membership of two fields.
    reqs = ConsistencyRaisesTargetRule(0.5, raise_to=6000.0).requirements()
    assert set(reqs) == {
        StateField.DAY_PNL,
        StateField.TOTAL_PNL,
        StateField.MAX_DAY_PNL,
        StateField.PROFIT_TARGET,
    }


# --- consistency gate (the mechanic our firms actually use) ---------------- #


def test_consistency_gate_compiles_to_the_chosen_success_action():
    # The ONLY thing that varies between eval and funded use is which success it
    # gates: PASS (eval) vs PAYOUT (funded).
    assert ConsistencyGateRule(0.5, gate=Action.PASS).compile().action == Action.PASS
    assert ConsistencyGateRule(0.4, gate=Action.PAYOUT).compile().action == Action.PAYOUT


def test_consistency_gate_defaults_to_payout():
    # Funded payout-gating is the common case, so PAYOUT is the default (ARCH §5).
    assert ConsistencyGateRule(0.5).compile().action == Action.PAYOUT


def test_consistency_gate_carries_the_flex_cushion_in_p1():
    # The Flex allowance (ARCHITECTURE §5) rides in the compiled record's p1;
    # default 0.0 means no cushion.
    assert ConsistencyGateRule(0.5).compile().p1 == 0.0
    assert ConsistencyGateRule(0.5, cushion=60.0).compile().p1 == 60.0
    # cushion is part of identity (feeds the fingerprint, §10)
    assert ConsistencyGateRule(0.5, cushion=0.0) != ConsistencyGateRule(0.5, cushion=60.0)


@pytest.mark.parametrize("bad", [Action.FAIL, Action.ADJUST])
def test_consistency_gate_rejects_fail_and_adjust(bad):
    # A gate only ever withholds a success; it never fails or adjusts. Constructing
    # one with a non-gate action must fail loudly at instantiation.
    with pytest.raises(ValueError):
        ConsistencyGateRule(0.5, gate=bad)


def test_consistency_gate_carries_no_severity_or_fail_code():
    # It never fails, so it carries no failure semantics (unlike a FAIL rule).
    for gate in (Action.PASS, Action.PAYOUT):
        c = ConsistencyGateRule(0.5, gate=gate).compile()
        assert c.fail_code is None
        assert c.adjust_field is None


def test_consistency_gate_denominator_is_cycle_scoped():
    # Cycle profit = EQUITY - CYCLE_START_EQUITY generalizes eval (no reset) and
    # funded (resets per payout). Pin the exact read-set that encodes that.
    reqs = ConsistencyGateRule(0.5).requirements()
    assert set(reqs) == {
        StateField.MAX_DAY_PNL,
        StateField.EQUITY,
        StateField.CYCLE_START_EQUITY,
    }
    # It must NOT read the un-resettable TOTAL_PNL, or the funded per-cycle reset
    # would be silently defeated.
    assert StateField.TOTAL_PNL not in reqs


def test_consistency_gate_timing_is_inert():
    # The gate is only ever read inside the payout/pass conjunction (MODEL_RISKS
    # §C8), so its check_timing carries no semantic choice — it exposes no timing
    # knob and compiles to the default CONTINUOUS. Behavior must not depend on it.
    assert ConsistencyGateRule(0.5).compile().check_timing == Timing.CONTINUOUS
    assert not hasattr(ConsistencyGateRule(0.5), "check_timing")


def test_consistency_gate_threshold_difference_is_same_type_unequal():
    a = ConsistencyGateRule(0.4, gate=Action.PAYOUT)
    b = ConsistencyGateRule(0.5, gate=Action.PAYOUT)
    assert type(a) is type(b)
    assert a != b
    assert a.compile() != b.compile()
    # gate is part of identity too
    assert ConsistencyGateRule(0.5, gate=Action.PASS) != ConsistencyGateRule(
        0.5, gate=Action.PAYOUT
    )


def test_every_requirement_is_a_state_field():
    for rule in _ALL_RULES:
        for sf in rule.requirements():
            assert isinstance(sf, StateField)


# --- compiled action / severity / fail-code -------------------------------- #


def test_compiled_numeric_parameters_match_the_rule_definition():
    # The core half of the Step 2 contract (BUILD_SPEC): the compiled record's
    # parameters must match the definition. Pin p0/p1 for EVERY rule so a
    # compile() that drops or zeroes a threshold/amount/count is caught.
    cases = [
        (ProfitTargetRule(3000.0), {"p0": 3000.0}),
        (TrailingDrawdownRule(2500.0, lock_at=1000.0), {"p0": 2500.0, "p1": 1000.0}),
        (StaticDrawdownRule(2000.0), {"p0": 2000.0}),
        (DailyLossRule(1000.0), {"p0": 1000.0}),
        (MinimumTradingDaysRule(3), {"p0": 3}),
        (MinimumWinningDaysRule(count=5, threshold=150.0), {"p0": 5, "p1": 150.0}),
        (
            ConsistencyRaisesTargetRule(0.4, raise_to=6000.0),
            {"p0": 0.4, "p1": 6000.0},
        ),
        (ConsistencyGateRule(0.5, gate=Action.PASS, cushion=25.0), {"p0": 0.5, "p1": 25.0}),
        (ConsistencyGateRule(0.4, gate=Action.PAYOUT), {"p0": 0.4, "p1": 0.0}),
    ]
    for rule, expected in cases:
        c = rule.compile()
        for field_name, val in expected.items():
            assert getattr(c, field_name) == val, (
                f"{type(rule).__name__}.compile().{field_name}"
            )


def test_fail_rule_severity_defaults():
    # Contractual defaults for every FAIL rule — a flipped default (e.g. a
    # consistency breach becoming SOFT) is a real behaviour change the kernel obeys.
    assert DailyLossRule(1000.0).compile().severity == Severity.SOFT
    assert TrailingDrawdownRule(2500.0).compile().severity == Severity.HARD
    assert StaticDrawdownRule(2000.0).compile().severity == Severity.HARD


def test_fail_rules_carry_the_right_action_and_fail_code():
    cases = {
        TrailingDrawdownRule(2500.0): ExitCode.FAIL_TRAILING_DD,
        StaticDrawdownRule(2000.0): ExitCode.FAIL_STATIC_DD,
        DailyLossRule(1000.0): ExitCode.FAIL_DAILY_LOSS,
    }
    for rule, code in cases.items():
        c = rule.compile()
        assert c.action == Action.FAIL
        assert c.fail_code == code


def test_daily_loss_severity_is_overridable():
    hard = DailyLossRule(1000.0, severity=Severity.HARD).compile()
    assert hard.severity == Severity.HARD
    assert hard.action == Action.FAIL


def test_pass_rules_yield_pass_action_and_no_fail_code():
    for rule in (ProfitTargetRule(3000.0), MinimumTradingDaysRule(3)):
        c = rule.compile()
        assert c.action == Action.PASS
        assert c.fail_code is None


def test_payout_rule_yields_payout_action_with_both_parameters():
    c = MinimumWinningDaysRule(count=5, threshold=150.0).compile()
    assert c.action == Action.PAYOUT
    assert c.p0 == 5
    assert c.p1 == 150.0


def test_consistency_adjust_yields_adjust_action_naming_the_target():
    c = ConsistencyRaisesTargetRule(0.5, raise_to=6000.0).compile()
    assert c.action == Action.ADJUST
    assert c.adjust_field == StateField.PROFIT_TARGET
    assert c.p1 == 6000.0  # the raise-to value rides in p1
    assert c.fail_code is None  # adjust neither fails nor passes


# --- timing fields --------------------------------------------------------- #


def test_every_compiled_rule_carries_two_timing_fields():
    for rule in _ALL_RULES:
        c = rule.compile()
        assert isinstance(c.update_timing, Timing)
        assert isinstance(c.check_timing, Timing)


def test_timing_irrelevant_rules_default_to_continuous():
    # Profit target / min-days / daily-loss have no timing meaning -> both CONTINUOUS.
    for rule in (ProfitTargetRule(3000.0), MinimumTradingDaysRule(3), DailyLossRule(1.0)):
        c = rule.compile()
        assert c.update_timing == Timing.CONTINUOUS
        assert c.check_timing == Timing.CONTINUOUS


def test_consistency_adjust_defaults_to_eod_timing():
    # The adjust variant is a whole-day property, evaluated at day close.
    assert ConsistencyRaisesTargetRule(0.5, 6000.0).compile().check_timing == Timing.EOD


def test_trailing_dd_preserves_both_timing_axes_independently():
    c = TrailingDrawdownRule(
        2500.0, update_timing=Timing.EOD, check_timing=Timing.CONTINUOUS
    ).compile()
    assert c.update_timing == Timing.EOD
    assert c.check_timing == Timing.CONTINUOUS


def test_trailing_dd_lock_at_absent_compiles_to_never_lock_sentinel():
    assert TrailingDrawdownRule(2500.0).compile().p1 == NEVER_LOCK
    assert math.isinf(TrailingDrawdownRule(2500.0).compile().p1)
    assert TrailingDrawdownRule(2500.0, lock_at=0.0).compile().p1 == 0.0


# --- the compiled record itself -------------------------------------------- #


def test_compiled_rule_is_frozen_and_hashable_by_value():
    a = ProfitTargetRule(3000.0).compile()
    b = ProfitTargetRule(3000.0).compile()
    assert a == b
    assert hash(a) == hash(b)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.p0 = 1.0  # type: ignore[misc]


def test_compiled_kind_matches_the_registry_entry():
    # Each rule compiles to the kind under which its class is registered.
    for rule in _ALL_RULES:
        kind = rule.compile().kind
        assert isinstance(kind, RuleKind)
        assert RULE_REGISTRY[kind] is type(rule)


# --- the registry hard-fail ------------------------------------------------ #


def test_registry_covers_every_rule_kind_enum_member():
    # A RuleKind with no registry entry is a rule the kernel would silently fail
    # to run — the gap assert_kernel_supports exists to close.
    assert set(RULE_REGISTRY) == set(RuleKind)


def test_assert_kernel_supports_passes_for_every_registered_kind():
    for kind in RULE_REGISTRY:
        assert_kernel_supports(kind)  # must not raise


def test_assert_kernel_supports_raises_on_unregistered_kind():
    unregistered = max(int(k) for k in RuleKind) + 1
    assert unregistered not in RULE_REGISTRY
    with pytest.raises(UnknownRuleError):
        assert_kernel_supports(unregistered)


def test_guard_keys_on_the_registry_not_on_rulekind_membership(monkeypatch):
    # The safety net exists for exactly one scenario (§5): a RuleKind member added
    # WITHOUT a registry entry — a kernel branch that doesn't exist yet. A guard
    # that merely checked `kind in RuleKind` would pass such a rule silently.
    # Simulate the missing entry and require the guard to still fail loudly.
    patched = dict(RULE_REGISTRY)
    del patched[RuleKind.CONSISTENCY_GATE]
    monkeypatch.setattr("propfirm_engine.rules.RULE_REGISTRY", patched)
    assert RuleKind.CONSISTENCY_GATE in RuleKind  # still a valid enum member...
    with pytest.raises(UnknownRuleError):  # ...but no kernel implementation
        assert_kernel_supports(RuleKind.CONSISTENCY_GATE)


def test_every_registered_rule_compiles_to_a_compiled_rule():
    # _ALL_RULES holds one instance per registered kind (asserted equal to the
    # registry key set above), so this exercises the whole registry.
    assert {r.compile().kind for r in _ALL_RULES} == set(RULE_REGISTRY)
    for rule in _ALL_RULES:
        assert isinstance(rule.compile(), CompiledRule)
