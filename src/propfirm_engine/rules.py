"""Rule base class (ARCHITECTURE §5).

Step 1 defines only the abstract interface. The concrete rule types (profit
target, trailing drawdown, daily loss, consistency, ...) and the
``RULE_REGISTRY`` that guarantees the kernel can execute them are Step 2.

A rule is a *predicate plus an action*. The base declares two things and no
shared behaviour: what per-simulation state the rule needs (``requirements``),
and how it lowers to a numeric record the kernel consumes (``compile``). All
actual rule behaviour lives in the kernel, keyed by rule kind — never in the
class hierarchy — so every concrete rule is exactly one level deep.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .enums import StateField


@dataclass(frozen=True)
class Rule(ABC):
    """Abstract base for every rule. Frozen and hashable by value, so any DSL
    object containing rules is itself hashable (required for the fingerprint,
    §10). ``@abstractmethod`` makes a half-finished rule fail at instantiation."""

    @abstractmethod
    def requirements(self) -> tuple[StateField, ...]:
        """The ``StateField``s this rule's predicate reads. The compiler unions
        these across a phase to decide which state is live (§8)."""
        raise NotImplementedError

    @abstractmethod
    def compile(self) -> "CompiledRule":  # noqa: F821 — defined in Step 2
        """Lower this rule to its numeric record (kind/params/action/severity/
        timing/fail-code). Implemented by concrete rules in Step 2."""
        raise NotImplementedError


__all__ = ["Rule"]
