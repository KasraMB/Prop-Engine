"""The prop-firm DSL tree (ARCHITECTURE §4).

``Firm → Program → Variant → Account → Phase`` — all frozen dataclasses,
hashable by value. No dicts or lists appear anywhere reachable from an
``Account``; tuples are used throughout so the structural fingerprint (§10) is
well-defined and two objects built from identical values are equal *and*
hash-equal.

Design invariants this module enforces (§ "Design invariants"):

- **The variant level is always present.** A program with a single account type
  carries one implicit ``default`` variant (built via
  :meth:`Program.with_default_variant`), so every consumer traverses
  ``firm.program(x).variant(y).account(z)`` with one shape and no branching.
- **Phases are independent gates.** An account holds ``(eval, funded)`` or just
  ``(funded,)``; no special-case code distinguishes them — ``Phase.role`` does.
- **The fee is the entire downside (§0).** ``eval_fee``/``activation_fee`` are
  part of account identity and the fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing rule/schema machinery at model-definition time
    from .rules import Rule
    from .schema import PayoutSchema


@dataclass(frozen=True)
class Phase:
    """One gate of an account. An eval phase is cleared by a ``PASS`` predicate;
    a funded phase may issue repeatable ``PAYOUT``s and need not be terminable."""

    name: str
    role: str  # "eval" | "funded" — drives role-aware validation & payouts
    rules: tuple[Rule, ...]
    payout_schema: PayoutSchema | None = None  # funded phases carry one (§6b)


@dataclass(frozen=True)
class Account:
    """A single account size. ``phases`` is ``(eval, funded)`` or ``(funded,)``."""

    name: str
    size: int
    phases: tuple[Phase, ...]
    eval_fee: float = 0.0  # cost to start an attempt — the entire downside (§0)
    activation_fee: float = 0.0  # additional cost on reaching funded (some types)
    currency: str = "USD"


@dataclass(frozen=True)
class Variant:
    """A named group of account sizes. The ``default`` variant is the common case
    (a program with a single account type); firms with genuinely distinct
    account families use additional named variants."""

    name: str
    accounts: tuple[Account, ...]

    def account(self, name: str) -> Account:
        for a in self.accounts:
            if a.name == name:
                return a
        raise KeyError(name)


@dataclass(frozen=True)
class Program:
    """A firm product line. ``version`` is an effective-date/version string that is
    part of the structural hash (§10), so a rules change is a new fingerprint."""

    name: str
    variants: tuple[Variant, ...]
    version: str = "v1"

    def variant(self, name: str = "default") -> Variant:
        for v in self.variants:
            if v.name == name:
                return v
        raise KeyError(name)

    @classmethod
    def with_default_variant(
        cls, name: str, accounts: tuple[Account, ...], version: str = "v1"
    ) -> "Program":
        """Build a program whose single account family lives under the implicit
        ``default`` variant, so ``program.variant().account(z)`` works whether or
        not variants were authored by hand (§4)."""
        return cls(name=name, variants=(Variant("default", accounts),), version=version)


@dataclass(frozen=True)
class Firm:
    """The top of the tree: a firm and its programs."""

    name: str
    programs: tuple[Program, ...]

    def program(self, name: str) -> Program:
        for p in self.programs:
            if p.name == name:
                return p
        raise KeyError(name)


__all__ = ["Phase", "Account", "Variant", "Program", "Firm"]
