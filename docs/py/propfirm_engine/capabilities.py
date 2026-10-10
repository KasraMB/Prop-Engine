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


def check_replay(spec, *, adapter="recorded", fidelity=None, features=()):
    if not isinstance(spec, LifecycleSpec):
        raise TypeError("preflight requires LifecycleSpec")
    if adapter not in ("recorded", "brackets"):
        raise UnsupportedInputCapabilityError("select recorded fills or closed brackets replay")
    if fidelity is None:
        fidelity = "closed_brackets" if adapter == "brackets" else "observed_marks"
    if adapter == "brackets":
        expected = "closed_brackets"
        if fidelity != expected:
            raise UnsupportedInputCapabilityError(f"{adapter} replay requires {expected}")
        _check_support(spec, observations=False)
        supported = {"long", "short", "mixed_directions", "eod_drawdown", "sizing"}
        missing = set(features)-supported
        if missing:
            raise UnsupportedInputCapabilityError(f"unsupported by {adapter} replay: {', '.join(sorted(missing))}")
        return ReplaySupport(adapter, fidelity, tuple(sorted(supported)), (
            "closed brackets hold recorded outcomes and reward-to-risk ratios fixed",
            "partial exits and concurrent positions require recorded fills",
        ))
    if fidelity != "observed_marks":
        raise UnsupportedInputCapabilityError("recorded fills require supplied marks, not generated quotes")
    _check_support(spec, observations=True)
    supported = {"long", "short", "mixed_directions", "partial_exits", "reversals", "multi_asset", "observed_equity", "multiple_phases",
                 "static_drawdown", "eod_drawdown", "intraday_drawdown", "daily_suspension",
                 "withdrawal_policy", "processing_calendar"}
    if not spec.flatten_at_close:
        supported.add("overnight")
    missing = set(features)-supported
    if missing:
        raise UnsupportedInputCapabilityError(f"unsupported by this profile/input contract: {', '.join(sorted(missing))}")
    return ReplaySupport(adapter, fidelity, tuple(sorted(supported)), (
        "rules observe supplied marks; unobserved intrabar paths are not inferred",
        "custom fees, liquidity, slippage and liquidation assumptions must be supplied",
        "manual firm reviews and unknown agreements are not certified or inferred",
    ))
