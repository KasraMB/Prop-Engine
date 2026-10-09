"""Preflight the declared account and execution contract without running a path."""
from dataclasses import dataclass

from .backtest import _check_support
from .engine import UnsupportedInputCapabilityError
from .execution import LifecycleSpec


@dataclass(frozen=True)
class ReplaySupport:
    adapter: str
    fidelity: str
    features: tuple[str, ...]
    limitations: tuple[str, ...]


def check_replay(spec, *, adapter="strategy", fidelity="observed_marks", features=()):
    if not isinstance(spec, LifecycleSpec):
        raise TypeError("preflight requires LifecycleSpec")
    if adapter not in ("recorded", "strategy", "opportunities"):
        raise UnsupportedInputCapabilityError("select recorded, strategy or opportunities replay")
    if fidelity not in ("observed_marks", "last_trade", "ohlc_path", "mixed_scenario"):
        raise UnsupportedInputCapabilityError("unknown input fidelity")
    if adapter == "recorded" and fidelity != "observed_marks":
        raise UnsupportedInputCapabilityError("recorded fills require supplied marks, not generated quotes")
    _check_support(spec, observations=True)
    supported = {"partial_exits", "reversals", "multi_asset", "observed_equity", "multiple_phases",
                 "static_drawdown", "eod_drawdown", "intraday_drawdown", "daily_suspension",
                 "withdrawal_policy", "processing_calendar"}
    if not spec.flatten_at_close:
        supported.add("overnight")
    if adapter != "recorded":
        supported |= {"orders", "trailing_exits", "oco", "atomic_baskets", "causal_callbacks", "sizing"}
    if adapter == "opportunities":
        supported.add("external_signals")
    missing = set(features)-supported
    if missing:
        raise UnsupportedInputCapabilityError(f"unsupported by this profile/input contract: {', '.join(sorted(missing))}")
    return ReplaySupport(adapter, fidelity, tuple(sorted(supported)), (
        "rules observe supplied marks; unobserved intrabar paths are not inferred",
        "custom fees, liquidity, slippage and liquidation assumptions must be supplied",
        "manual firm reviews and unknown agreements are not certified or inferred",
        "pending exposure is reserved conservatively; atomic baskets require explicit all-or-none fills",
    ))
