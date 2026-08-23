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
from .rules import Rule

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
]
