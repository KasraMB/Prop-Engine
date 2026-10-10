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


def check_replay(spec, *, adapter="strategy", fidelity=None, features=()):
    if not isinstance(spec, LifecycleSpec):
        raise TypeError("preflight requires LifecycleSpec")
    if adapter not in ("recorded", "strategy", "opportunities", "prices", "brackets"):
        raise UnsupportedInputCapabilityError("select recorded, strategy, opportunities, prices or brackets replay")
    if fidelity is None:
        fidelity = {"prices": "ohlc_stop_first", "brackets": "closed_brackets"}.get(adapter, "observed_marks")
    if adapter in ("prices", "brackets"):
        expected = "ohlc_stop_first" if adapter == "prices" else "closed_brackets"
        if fidelity != expected:
            raise UnsupportedInputCapabilityError(f"{adapter} replay requires {expected}")
        _check_support(spec, observations=False)
        supported = {"long", "short", "mixed_directions", "eod_drawdown", "sizing"}
        if adapter == "prices":
            supported = (supported-{"sizing"}) | {"fixed_quantity", "retargeting", "stop_gaps", "session_close_exits"}
        missing = set(features)-supported
        if missing:
            raise UnsupportedInputCapabilityError(f"unsupported by {adapter} replay: {', '.join(sorted(missing))}")
        return ReplaySupport(adapter, fidelity, tuple(sorted(supported)), (
            "closed brackets hold realized outcomes fixed; price replay resolves supplied minute bars",
            "partial exits, concurrent positions and causal entry callbacks use strategy replay",
        ))
    if fidelity not in ("observed_marks", "last_trade", "ohlc_path", "mixed_scenario"):
        raise UnsupportedInputCapabilityError("unknown input fidelity")
    if adapter == "recorded" and fidelity != "observed_marks":
        raise UnsupportedInputCapabilityError("recorded fills require supplied marks, not generated quotes")
    _check_support(spec, observations=True)
    supported = {"long", "short", "mixed_directions", "partial_exits", "reversals", "multi_asset", "observed_equity", "multiple_phases",
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
