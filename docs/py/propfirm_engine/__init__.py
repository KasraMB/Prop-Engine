"""Research simulator for futures prop-firm strategies and attempt economics."""

from . import firms, statistics
from .execution import (
    BracketTrade, BracketHistory, RiskRegime, DollarPolicy, LifecycleSpec, BacktestConfig,
)
from .backtest import BacktestEvent, BacktestResult
from .fitting import HoldoutFit
from .rolling import RollingConfig, RollingWindow, RollingResult
from .risk import RiskConfig, CashRiskPath, cash_risk_path, risk_report
from .payouts import PayoutEvent, PayoutRequest, PayoutLedger
from .analytical import (
    BarrierEstimate, SessionMoments, barrier_pass_probability,
    estimate_session_moments, expected_funding_cost,
)
from .cashflows import (
    Cashflow, AttemptCashflows, CashflowSequences, WalletSequences,
    simulate_cashflow_sequences, simulate_wallet_sequences,
)

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
    slice_days,
)
from .engine import Engine, Outcomes, RunConfig, UnsupportedInputCapabilityError
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
    PolicyEvaluation,
    RenewalObjective,
    RollingResult,
    WalkForwardResult,
    factorized_optimize,
    optimize,
    rolling_walk_forward,
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
    "RiskConfig", "CashRiskPath", "cash_risk_path", "risk_report",
    "BracketTrade", "BracketHistory", "RiskRegime", "DollarPolicy",
    "LifecycleSpec", "BacktestConfig", "BacktestEvent", "BacktestResult", "HoldoutFit",
    "RollingConfig", "RollingWindow", "RollingResult",
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
    "PayoutEvent",
    "PayoutRequest",
    "PayoutLedger",
    "BarrierEstimate",
    "SessionMoments",
    "barrier_pass_probability",
    "estimate_session_moments",
    "expected_funding_cost",
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
    "UnsupportedInputCapabilityError",
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
    "Cashflow",
    "AttemptCashflows",
    "CashflowSequences",
    "simulate_cashflow_sequences",
    "WalletSequences",
    "simulate_wallet_sequences",
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
    "PolicyEvaluation",
    "RenewalObjective",
    "CMAES",
    "OptConfig",
    "optimize",
    "walk_forward",
    "WalkForwardResult",
    "rolling_walk_forward",
    "RollingResult",
    "factorized_optimize",
    "slice_days",
    # firms
    "firms",
]
