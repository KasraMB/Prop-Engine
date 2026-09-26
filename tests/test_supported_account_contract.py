"""Known-answer rejection tests, independent of simulator/reference agreement.

These are execution-capability limits, not claims about any firm's rulebook.
Unsupported structures must fail before compilation or simulation starts.
"""

from dataclasses import dataclass

import pytest

from propfirm_engine.engine import Engine, RunConfig
from propfirm_engine.enums import Action
from propfirm_engine.model import Account, Phase
from propfirm_engine.rules import (
    CompiledRule,
    ConsistencyGateRule,
    DailyLossRule,
    MinimumTradingDaysRule,
    MinimumWinningDaysRule,
    ProfitTargetRule,
    Rule,
    RuleKind,
    StaticDrawdownRule,
    TrailingDrawdownRule,
)
from propfirm_engine.schema import PayoutSchema
from propfirm_engine.validate import InvalidAccountError, validate


def eval_phase(name="evaluation"):
    return Phase(name, "eval", (ProfitTargetRule(100.0),))


def funded_phase(name="funded"):
    return Phase(name, "funded", (StaticDrawdownRule(100.0),))


@pytest.fixture(params=["validate", "prepare"])
def check_account(request):
    if request.param == "validate":
        return validate
    return lambda account: Engine().prepare(account, RunConfig())


@pytest.mark.parametrize(
    "phases, message",
    [
        ((eval_phase(), eval_phase("second evaluation")), "multiple eval phases"),
        ((funded_phase(), funded_phase("second funded")), "multiple funded phases"),
        ((funded_phase(), eval_phase()), "phase order"),
        ((Phase("live", "live", (StaticDrawdownRule(100.0),)),), "unsupported phase role"),
    ],
)
def test_rejects_unexecutable_lifecycle(check_account, phases, message):
    with pytest.raises(InvalidAccountError, match=message):
        check_account(Account("test", 1000, phases))


@pytest.mark.parametrize(
    "phases",
    [(eval_phase(),), (funded_phase(),), (eval_phase(), funded_phase())],
)
def test_preserves_supported_lifecycles(check_account, phases):
    check_account(Account("test", 1000, phases))


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "kind, pair",
    [
        ("PROFIT_TARGET", (ProfitTargetRule(100.0), ProfitTargetRule(900.0))),
        ("TRAILING_DD", (TrailingDrawdownRule(100.0), TrailingDrawdownRule(200.0))),
        ("MIN_WINNING_DAYS", (MinimumWinningDaysRule(2, 10.0), MinimumWinningDaysRule(3, 20.0))),
    ],
)
def test_rejects_colliding_scalar_rules_in_either_order(check_account, reverse, kind, pair):
    rules = pair[::-1] if reverse else pair
    if kind == "MIN_WINNING_DAYS":
        phase = Phase("funded", "funded", rules, PayoutSchema((100.0,), 1.0, 2))
    else:
        if kind != "PROFIT_TARGET":
            rules = (ProfitTargetRule(100.0), *rules)
        phase = Phase("evaluation", "eval", rules)
    with pytest.raises(InvalidAccountError, match=kind):
        check_account(Account("test", 1000, (phase,)))


@dataclass(frozen=True)
class AlternateTarget(Rule):
    """A different Python class still collides with the same compiled scalar."""

    def requirements(self):
        return ()

    def compile(self):
        return CompiledRule(RuleKind.PROFIT_TARGET, Action.PASS, p0=500.0)


def test_collision_check_uses_compiled_kind_not_python_class(check_account):
    phase = Phase("evaluation", "eval", (ProfitTargetRule(100.0), AlternateTarget()))
    with pytest.raises(InvalidAccountError, match="PROFIT_TARGET"):
        check_account(Account("test", 1000, (phase,)))


@pytest.mark.parametrize(
    "extra",
    [
        (DailyLossRule(50.0), DailyLossRule(80.0)),
        (StaticDrawdownRule(100.0), StaticDrawdownRule(200.0)),
        (MinimumTradingDaysRule(2), MinimumTradingDaysRule(3)),
        (ConsistencyGateRule(0.4, gate=Action.PASS), ConsistencyGateRule(0.5, gate=Action.PASS)),
    ],
)
def test_does_not_blanket_ban_repeated_rule_kinds(check_account, extra):
    phase = Phase("evaluation", "eval", (ProfitTargetRule(100.0), *extra))
    check_account(Account("test", 1000, (phase,)))


@pytest.mark.parametrize(
    "rule", [ProfitTargetRule(100.0), ConsistencyGateRule(0.5, gate=Action.PASS)]
)
def test_rejects_pass_action_in_funded_phase(check_account, rule):
    phase = Phase("funded", "funded", (StaticDrawdownRule(100.0), rule))
    with pytest.raises(InvalidAccountError, match="funded phase.*PASS"):
        check_account(Account("test", 1000, (phase,)))


@pytest.mark.parametrize(
    "rule", [MinimumWinningDaysRule(2, 10.0), ConsistencyGateRule(0.5, gate=Action.PAYOUT)]
)
def test_rejects_payout_action_in_eval_even_when_pass_is_present(check_account, rule):
    phase = Phase("evaluation", "eval", (ProfitTargetRule(100.0), rule))
    with pytest.raises(InvalidAccountError, match="eval phase.*PAYOUT"):
        check_account(Account("test", 1000, (phase,)))


def test_rejection_precedes_compilation(monkeypatch):
    engine = Engine()

    def unexpected_compilation(*args, **kwargs):
        pytest.fail("unsupported account reached the compiler/cache")

    monkeypatch.setattr(engine.caches.accounts, "get", unexpected_compilation)
    account = Account("test", 1000, (eval_phase(), eval_phase("second")))
    with pytest.raises(InvalidAccountError, match="multiple eval phases"):
        engine.prepare(account, RunConfig())
