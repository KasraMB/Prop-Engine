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


def load_consumer(path, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    full = Path(__file__).resolve().parents[1] / path
    monkeypatch.syspath_prepend(str(full.parent))
    monkeypatch.delitem(sys.modules, "accounts", raising=False)
    spec = importlib.util.spec_from_file_location("capability_consumer", full)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", ["dashboard/montecarlo.py", "docs/py/mc_engine.py"])
def test_monte_carlo_consumers_default_to_rejection_and_record_opt_in(path, monkeypatch):
    from propfirm_engine import UnsupportedInputCapabilityError
    module = load_consumer(path, monkeypatch)
    # Test the rule capability, not the truth of a legacy firm preset's timing.
    monkeypatch.setattr(module, "_account", lambda *args: account(StaticDrawdownRule(100)))
    params = dict(size="50K", n_paths=4, n_days=5, L_eval=2, L_funded=2,
                  win_rate=0.5, rr=1.5)
    with pytest.raises(UnsupportedInputCapabilityError):
        module.run(params)
    result = module.run(dict(params, intraday_mode="summary_approximation"))
    assert result["model_scope"]["intraday_mode"] == "summary_approximation"
    assert result["model_scope"]["approximation_reasons"]


@pytest.mark.parametrize("path", ["dashboard/bridge.py", "docs/py/bridge.py"])
def test_manual_consumers_default_to_rejection_and_record_opt_in(path, monkeypatch):
    from propfirm_engine import UnsupportedInputCapabilityError
    module = load_consumer(path, monkeypatch)
    args = ("Lucid", "LucidFlex", "50K", "eval", [[dict(pnl=10)]])
    with pytest.raises(UnsupportedInputCapabilityError):
        module.evaluate(*args)
    result = module.evaluate(*args, intraday_mode="summary_approximation")
    assert result["model_scope"]["intraday_mode"] == "summary_approximation"
    assert result["model_scope"]["execution_model"] == "manual_closed_trade_summary"
    state = module._lucid_payout_state(
        [[dict(pnl=3000)], [dict(type="payout", amount=1500)], [dict(pnl=100)], []],
        module.payout_params("50K"))
    assert state["max_amount"] == 800.0  # retained 1600, not lifetime profit 3100


@pytest.mark.parametrize("path", [
    "dashboard/index.html", "docs/interactive.html",
    "dashboard/montecarlo.html", "docs/montecarlo.html",
])
def test_ui_opt_in_is_unchecked_and_transmits_explicit_mode(path):
    import re
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1] / path).read_text(encoding="utf-8")
    checkbox = re.search(r'<input[^>]*id="summary_approximation"[^>]*>', html).group()
    assert "checked" not in checkbox
    assert '.checked?"summary_approximation":"strict"' in html
    assert "model_scope" in html
