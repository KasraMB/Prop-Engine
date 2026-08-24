"""propfirm_engine — valuing prop-firm accounts as convex structured products."""

from .enums import (
    FAILURE_THRESHOLD,
    Action,
    ExitCode,
    Severity,
    Stage,
    StateField,
    Timing,
)
from .model import Account, Firm, Phase, Program, Variant
from .rules import (
    RULE_REGISTRY,
    CompiledRule,
    ConsistencyGateRule,
    ConsistencyRaisesTargetRule,
    DailyLossRule,
    MinimumTradingDaysRule,
    MinimumWinningDaysRule,
    NEVER_LOCK,
    ProfitTargetRule,
    Rule,
    RuleKind,
    StaticDrawdownRule,
    TrailingDrawdownRule,
    UnknownRuleError,
    assert_kernel_supports,
)

__all__ = [
    # enums
    "FAILURE_THRESHOLD",
    "ExitCode",
    "StateField",
    "Action",
    "Severity",
    "Timing",
    "Stage",
    # model
    "Firm",
    "Program",
    "Variant",
    "Account",
    "Phase",
    # rules
    "Rule",
    "RuleKind",
    "CompiledRule",
    "NEVER_LOCK",
    "TrailingDrawdownRule",
    "StaticDrawdownRule",
    "DailyLossRule",
    "ProfitTargetRule",
    "MinimumTradingDaysRule",
    "MinimumWinningDaysRule",
    "ConsistencyRaisesTargetRule",
    "ConsistencyGateRule",
    "RULE_REGISTRY",
    "UnknownRuleError",
    "assert_kernel_supports",
]
