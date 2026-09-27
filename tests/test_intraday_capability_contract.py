"""Closed-trade evidence cannot silently authorize intraday-rule execution."""
from dataclasses import replace

import pytest

from propfirm_engine import (
    Account, DailyLossRule, Engine, Phase, ProfitTargetRule, RunConfig,
    StaticDrawdownRule, Timing, TrailingDrawdownRule, preprocess,
)


def dataset():
    return preprocess([dict(timestamp="2026-01-05T10:00:00", pnl=10, size=1, mae=5)])


def account(rule, role="eval"):
    gates = (ProfitTargetRule(100),) if role == "eval" else ()
    return Account("synthetic", 1000, phases=(Phase(role, role, (rule,) + gates),))


@pytest.mark.parametrize("rule", [
    StaticDrawdownRule(100), DailyLossRule(100), TrailingDrawdownRule(100),
    TrailingDrawdownRule(100, update_timing=Timing.EOD),
    TrailingDrawdownRule(100, check_timing=Timing.EOD),
])
@pytest.mark.parametrize("role", ["eval", "funded"])
def test_default_rejects_intraday_requirements(rule, role):
    from propfirm_engine import UnsupportedInputCapabilityError
    with pytest.raises(UnsupportedInputCapabilityError, match="ordered.*mark-to-market"):
        Engine().run(account(rule, role), dataset(), RunConfig(n_paths=2))


@pytest.mark.parametrize("rule", [
    StaticDrawdownRule(100, check_timing=Timing.EOD),
    TrailingDrawdownRule(100, update_timing=Timing.EOD, check_timing=Timing.EOD),
])
def test_eod_only_rules_and_closed_profit_gate_do_not_need_opt_in(rule):
    out = Engine().run(account(rule), dataset(), RunConfig(n_paths=2))
    assert out.intraday_mode == "strict"
    assert out.execution_model == "closed_trade_summary"
    assert out.approximation_reasons == ()


def test_explicit_opt_in_is_recorded_and_prepared_runs_are_guarded():
    from propfirm_engine import UnsupportedInputCapabilityError
    eng = Engine()
    cfg = RunConfig(n_paths=2, intraday_mode="summary_approximation")
    prepared = eng.prepare(account(TrailingDrawdownRule(100)), cfg)
    out = eng.run_prepared(prepared, dataset(), cfg)
    assert out.intraday_mode == "summary_approximation"
    assert out.execution_model == "closed_trade_summary"
    assert len(out.approximation_reasons) == 2
    with pytest.raises(UnsupportedInputCapabilityError):
        eng.run_prepared(prepared, dataset(), replace(cfg, intraday_mode="strict"))


@pytest.mark.parametrize("mode", [None, True, "", "approximate", "STRICT", 1, []])
def test_invalid_mode_rejected_at_construction_and_use(mode):
    with pytest.raises(ValueError, match="intraday_mode"):
        RunConfig(intraday_mode=mode)
    cfg = RunConfig(n_paths=2)
    prepared = Engine().prepare(account(StaticDrawdownRule(100)), cfg)
    cfg.intraday_mode = mode
    with pytest.raises(ValueError, match="intraday_mode"):
        Engine().run_prepared(prepared, dataset(), cfg)


def test_guard_precedes_resampling_and_cannot_be_bypassed_by_dataset_label():
    from propfirm_engine import UnsupportedInputCapabilityError

    class NeverSample:
        def generate(self, *args):
            pytest.fail("unsupported run must not resample")

    ds = dataset()
    object.__setattr__(ds, "execution_model", "ordered_mtm")  # a label cannot create evidence
    with pytest.raises(UnsupportedInputCapabilityError):
        Engine().run(account(DailyLossRule(100)), ds, RunConfig(resampler=NeverSample()))


@pytest.mark.parametrize("extra", [{}, {"mae": 5}, {
    "entry_time": "2026-01-05T09:00:00",
    "mae_bars": [("2026-01-05T09:30:00", 5)],
}])
def test_optional_excursions_do_not_supply_an_ordered_path(extra):
    from propfirm_engine import UnsupportedInputCapabilityError
    ds = preprocess([dict(timestamp="2026-01-05T10:00:00", pnl=10, size=1, **extra)])
    with pytest.raises(UnsupportedInputCapabilityError):
        Engine().run(account(StaticDrawdownRule(100)), ds, RunConfig(n_paths=2))


def test_optimizer_evaluation_retains_explicit_mode_and_reasons():
    from propfirm_engine.optimizer import PolicySpace, evaluate_policy
    cfg = RunConfig(n_paths=2, intraday_mode="summary_approximation")
    result = evaluate_policy(Engine(), account(DailyLossRule(100)), dataset(), cfg,
                             [1], PolicySpace(), lambda o: 0.0, return_details=True)
    assert dict(result.run_parameters)["intraday_mode"] == cfg.intraday_mode
    summary = dict(result.data_summary)
    assert summary["intraday_mode"] == cfg.intraday_mode
    assert summary["execution_model"] == result.outcomes.execution_model
    assert summary["approximation_reasons"] == result.outcomes.approximation_reasons
    assert summary["approximation_reasons"]
