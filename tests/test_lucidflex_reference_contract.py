"""Reference-account fixes do not certify the remaining legacy payout model."""
import numpy as np
import pytest

from propfirm_engine import (
    Engine, ExitCode, RunConfig, Timing, UnsupportedInputCapabilityError,
    compile_phase, preprocess, simulate_reference,
)
from propfirm_engine.firms import lucidflex
from propfirm_engine.rules import RuleKind


@pytest.mark.parametrize("size", list(lucidflex.SPECS))
def test_lucidflex_trails_eod_but_checks_loss_continuously(size):
    for phase in lucidflex.build_account(size).phases:
        cp = compile_phase(phase)
        mll = np.flatnonzero(cp.kind == RuleKind.TRAILING_DD)
        assert len(mll) == 1
        assert cp.update_timing[mll[0]] == Timing.EOD
        assert cp.check_timing[mll[0]] == Timing.CONTINUOUS
        assert not np.any(cp.kind == RuleKind.DAILY_LOSS)
        if cp.payout is not None:
            assert cp.payout.min_request == 500.0


def test_selected_reference_account_cannot_silently_run_in_strict_mode():
    data = preprocess([{"timestamp": "2026-01-05T10:00:00", "return": 100.0}])
    with pytest.raises(UnsupportedInputCapabilityError):
        Engine().run(lucidflex.build_account(50_000), data, RunConfig(n_paths=2))


def run_summary(role, returns, lows):
    phase = next(p for p in lucidflex.build_account(50_000).phases if p.role == role)
    return simulate_reference(
        compile_phase(phase), np.array(returns, dtype=float),
        np.arange(len(returns), dtype=np.int32), np.array(lows, dtype=float),
        1.0, np.array([1.0]), 50_000.0,
    )


@pytest.mark.parametrize("role", ["eval", "funded"])
def test_intratrade_floor_touch_cannot_be_erased_by_recovery(role):
    result = run_summary(role, [100.0], [-2000.0])
    assert result.code == ExitCode.FAIL_TRAILING_DD


def test_five_winning_days_do_not_allow_a_subminimum_request():
    # $750 profit, legacy 50% cycle basis => $375 gross: below $500.
    result = run_summary("funded", [150.0] * 5, [0.0] * 5)
    assert result.payouts_taken == 0


def test_minimum_request_is_gross_before_split():
    result = run_summary("funded", [200.0] * 5, [0.0] * 5)
    assert result.payouts_taken == 1
    assert result.payout_amounts == [450.0]
