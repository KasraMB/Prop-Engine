"""propfirm_engine — valuing prop-firm accounts as convex structured products."""

from . import firms, statistics

from .enums import (
    FAILURE_THRESHOLD,
    Action,
    ExitCode,
    Severity,
    Stage,
    StateField,
    Timing,
)
from .compiler import (
    CompiledAccount,
    CompiledPayoutSchema,
    CompiledPhase,
    compile_account,
    compile_phase,
    resolve_requirements,
)
from .cache import Caches, CompiledAccountCache, CompiledRuleCache, TradeCache
from .config import build_accounts, scaled
from .fingerprint import fingerprint
from .kernels import simulate_one_phase
from .reference import SimResult, simulate_reference
from .data import (
    InvalidTradeDataError,
    TradeDataset,
    clip_mae_to_holding_interval,
    preprocess,
)
from .engine import Engine, Outcomes, RunConfig
from .model import Account, Firm, Phase, Program, Variant
from .objectives import (
    annualized_return_on_fee,
    expected_net_payoff,
    expected_payout_st_profitable,
)
from .renewal import (
    fee_bankroll_efficiency,
    finite_horizon_cashflow,
    prob_profitable_sequence,
    r_path,
    r_renewal,
)
from .results import Results
from .resampling import (
    DayResampler,
    IIDDayBootstrap,
    StationaryDayBootstrap,
    gather_days,
)
from .feasibility import (
    FeasibilityAgg,
    FeasibilityDiag,
    FeasibilitySpec,
    project_position,
)
from .ladder import Band, LadderResult, Rung, default_ladder, run_ladder
from .optimizer import (
    CMAES,
    OptConfig,
    PolicySpace,
    RenewalObjective,
    WalkForwardResult,
    optimize,
    walk_forward,
)
from .schema import PayoutSchema
from .validate import InvalidAccountError, validate
from .synthetic import (
    IIDGenerator,
    Provenance,
    RegimeSwitchingGenerator,
    StochasticVolGenerator,
    SyntheticStream,
    TradeStreamGenerator,
)
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
    # data
    "TradeDataset",
    "InvalidTradeDataError",
    "preprocess",
    "clip_mae_to_holding_interval",
    # synthetic
    "TradeStreamGenerator",
    "IIDGenerator",
    "RegimeSwitchingGenerator",
    "StochasticVolGenerator",
    "SyntheticStream",
    "Provenance",
    # schema / config / validate
    "PayoutSchema",
    "scaled",
    "build_accounts",
    "validate",
    "InvalidAccountError",
    # compiler
    "resolve_requirements",
    "compile_phase",
    "compile_account",
    "CompiledPhase",
    "CompiledAccount",
    "CompiledPayoutSchema",
    # kernel + reference oracle
    "simulate_one_phase",
    "simulate_reference",
    "SimResult",
    # fingerprint + caches
    "fingerprint",
    "Caches",
    "TradeCache",
    "CompiledAccountCache",
    "CompiledRuleCache",
    # resampling
    "DayResampler",
    "IIDDayBootstrap",
    "StationaryDayBootstrap",
    "gather_days",
    # engine
    "Engine",
    "RunConfig",
    "Outcomes",
    # statistics / objectives / results
    "statistics",
    "Results",
    "expected_net_payoff",
    "expected_payout_st_profitable",
    "annualized_return_on_fee",
    # renewal
    "r_renewal",
    "r_path",
    "finite_horizon_cashflow",
    "prob_profitable_sequence",
    "fee_bankroll_efficiency",
    # feasibility (§16.4b)
    "FeasibilitySpec",
    "FeasibilityDiag",
    "FeasibilityAgg",
    "project_position",
    # generator ladder (§G1 / Step 13)
    "Band",
    "Rung",
    "LadderResult",
    "default_ladder",
    "run_ladder",
    # optimizer (§16 / Step 14)
    "PolicySpace",
    "RenewalObjective",
    "CMAES",
    "OptConfig",
    "optimize",
    "walk_forward",
    "WalkForwardResult",
    # firms
    "firms",
]
