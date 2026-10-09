"""Research simulator for futures prop-firm strategies and attempt economics."""

__version__ = "0.2.0"

from . import firms, statistics
from .execution import (
    BracketTrade, BracketHistory, RiskRegime, DollarPolicy, LifecycleSpec, BacktestConfig,
)
from .backtest import BacktestEvent, BacktestResult
from .instruments import Instrument
from .market_replay import PriceSession, PriceDecision, PriceReplay
from .events import Fill, Marks, merge_events
from .portfolio import Book, BookState, Position
from .event_replay import EventReplay, EventState, PayoutContext
from .calendars import ProcessingCalendar
from .orders import Quote, Market, MarketSource, Order, Basket, Cancel, Amend, Abandon, OrderState, OrderEvent, QuoteModel
from .strategy import Context, StrategyReplay, ReplayCancelled
from .market_data import MarketTape, MarketView
from .feeds import Bar, TradeTick, MarketFeed, bar_quotes, trade_quotes, merge_markets
from .opportunities import Opportunity, OpportunityStrategy, OpportunityReplay
from .randomness import RandomStream
from .strategy_fitting import (
    Parameter, InfeasiblePolicy, StrategyPath, StrategyEvaluation, StrategyFit,
    StrategyTrial, StrategyCheckpoint, SearchCancelled, evaluate_strategy,
    StrategyWalkForward, evaluate_scenarios, SearchRun, estimate_strategy_work,
)
from .slippage import TickDistribution, SlippageModel
from .price_fitting import PriceEvaluation, PriceFit, evaluate_prices
from .fitting import HoldoutFit
from .rolling import RollingConfig, RollingWindow, RollingResult as RollingReplayResult
from .risk import RiskConfig, CashRiskPath, cash_risk_path, risk_report
from .ruin import RuinConfig, CashCycle, bootstrap_ruin, ultimate_cycle_ruin
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
from .capabilities import ReplaySupport, check_replay
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
    "SearchRun", "estimate_strategy_work",
    "ReplaySupport", "check_replay",
    "Basket",
    "Opportunity", "OpportunityStrategy", "OpportunityReplay",
    "MarketSource",
    "Bar", "TradeTick", "MarketFeed", "bar_quotes", "trade_quotes", "merge_markets",
    "RandomStream",
    "StrategyWalkForward", "evaluate_scenarios",
    "Parameter", "InfeasiblePolicy", "StrategyPath", "StrategyEvaluation", "StrategyFit",
    "StrategyTrial", "StrategyCheckpoint", "SearchCancelled", "ReplayCancelled", "evaluate_strategy",
    "MarketTape", "MarketView",
    "PayoutContext", "ProcessingCalendar",
    "Quote", "Market", "Order", "Cancel", "Amend", "Abandon", "OrderState", "OrderEvent",
    "QuoteModel", "Context", "StrategyReplay",
    "EventReplay", "EventState",
    "Fill", "Marks", "merge_events", "Book", "BookState", "Position",
    "Instrument", "PriceSession", "PriceDecision", "PriceReplay",
    "TickDistribution", "SlippageModel", "PriceEvaluation", "PriceFit", "evaluate_prices",
    "RiskConfig", "CashRiskPath", "cash_risk_path", "risk_report",
    "BracketTrade", "BracketHistory", "RiskRegime", "DollarPolicy",
    "LifecycleSpec", "BacktestConfig", "BacktestEvent", "BacktestResult", "HoldoutFit",
    "RollingConfig", "RollingWindow", "RollingReplayResult",
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
    "RuinConfig", "CashCycle", "bootstrap_ruin", "ultimate_cycle_ruin",
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
    # feasibility
    "FeasibilitySpec",
    "FeasibilityDiag",
    "FeasibilityAgg",
    "project_position",
    # generators
    "Band",
    "Rung",
    "LadderResult",
    "default_ladder",
    "run_ladder",
    # optimizer
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
