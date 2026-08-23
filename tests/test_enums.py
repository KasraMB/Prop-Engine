"""Step 1 — exit-code and enum semantics (BUILD_SPEC Step 1)."""

from propfirm_engine.enums import (
    FAILURE_THRESHOLD,
    Action,
    ExitCode,
    Severity,
    Stage,
    StateField,
    Timing,
)


def _failure_codes() -> list[ExitCode]:
    return [c for c in ExitCode if c.name.startswith("FAIL_")]


def test_alive_is_zero():
    assert ExitCode.ALIVE == 0


def test_passed_is_a_distinct_success_code():
    # PASSED is neither ALIVE nor any failure code.
    assert ExitCode.PASSED != ExitCode.ALIVE
    assert ExitCode.PASSED.value < FAILURE_THRESHOLD
    assert not ExitCode.PASSED.is_failure


def test_every_failure_code_is_at_least_ten():
    codes = _failure_codes()
    assert codes, "expected at least one FAIL_* code"
    for c in codes:
        assert c.value >= FAILURE_THRESHOLD
        assert c.is_failure


def _has_no_alias_collision(enum_cls) -> bool:
    """A duplicate integer value in an IntEnum does not raise — it silently becomes
    an *alias* that is absent from normal iteration, so ``len(list(cls))`` cannot
    detect it. ``__members__`` DOES include aliases, so comparing the member-name
    count to the distinct-value count is what actually catches a collision."""
    names = list(enum_cls.__members__)  # includes aliases
    values = [enum_cls.__members__[n].value for n in names]
    return len(set(values)) == len(names)


def test_all_failure_codes_are_mutually_distinct():
    # Guard against a real regression: e.g. FAIL_STATIC_DD = 11 (colliding with
    # FAIL_TRAILING_DD) would become a hidden alias and pass a naive set() check.
    fail_names = [n for n in ExitCode.__members__ if n.startswith("FAIL_")]
    fail_values = [ExitCode.__members__[n].value for n in fail_names]
    assert len(set(fail_values)) == len(fail_names), (
        "a FAIL_* code collided with another and became a silent alias"
    )


def test_no_exitcode_value_collisions_at_all():
    assert _has_no_alias_collision(ExitCode)


def test_non_failure_terminals_are_not_flagged_as_failures():
    for c in (ExitCode.TIMED_OUT, ExitCode.CAPPED_OUT, ExitCode.MAXED_OUT):
        assert not c.is_failure


def test_state_field_indices_are_distinct():
    # Two StateFields sharing an index would alias to the same state-vector slot —
    # a serious bug. Detect it via __members__ (aliases included), not iteration.
    assert _has_no_alias_collision(StateField)


def test_action_severity_timing_members_present():
    assert {a.name for a in Action} == {"FAIL", "PASS", "PAYOUT", "ADJUST"}
    assert {s.name for s in Severity} == {"HARD", "SOFT"}
    assert {t.name for t in Timing} == {"CONTINUOUS", "EOD"}


def test_stage_values_are_bit_positions_not_masks():
    # The kernel builds the mask as (1 << Stage.X); that is only correct if the
    # values are POSITIONS (0,1,2,3), not already-shifted masks (1,2,4,8). Pin it.
    assert sorted(s.value for s in Stage) == list(range(len(Stage)))


def test_stage_bits_compose_independently_in_a_mask():
    mask = (1 << Stage.IN_PROFIT) | (1 << Stage.PAYOUT_ELIGIBLE)
    assert mask & (1 << Stage.IN_PROFIT)
    assert mask & (1 << Stage.PAYOUT_ELIGIBLE)
    assert not mask & (1 << Stage.CONSISTENCY_SATISFIED)
    # OR-ing all stages sets exactly len(Stage) distinct bits (no overlap).
    full = 0
    for s in Stage:
        full |= 1 << s
    assert bin(full).count("1") == len(Stage)
