"""The payout schema — the product's upside leg (ARCHITECTURE §6b).

A ``PAYOUT`` predicate decides *whether* a payout can be requested; the
:class:`PayoutSchema` decides *how much* is released and *what it does to account
state*. Because the entire economic value of a funded account is the sum of its
released payouts, this is not an add-on — it is the product definition, and it is
per-account-type config (§6b). Every field varies independently across firms, so
each is explicit.

Field **defaults are firm-neutral**: either an identity value meaning "no such
constraint" (``cap_fraction=1.0``, ``min_request=0.0``, ``buffer_floor=0.0``) or
*required* (no default) where there is no neutral norm — ``dollar_cap``,
``split``, ``max_payouts``. Real firm values live in ``firms/`` and are verified
per firm, never hardcoded here as engine norms (the §5 intrinsic-vs-firm-varying
discipline). This class is pure data; the fire-gate arithmetic and the
post-payout transition live in the compiler/kernel (§6b, §12).
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import StateField


@dataclass(frozen=True)
class PayoutSchema:
    """How a funded account's payouts are sized and what a taken payout does (§6b).

    Frozen and hashable by value, so it is part of the account fingerprint (§10):
    two funded accounts differing only in ``split``/``dollar_cap``/``cap_fraction``
    or a post-payout flag must not collide to one cache key.
    """

    # --- REQUIRED (no firm-neutral norm — every firm must state these) ------- #
    dollar_cap: tuple[float, ...]  # per-payout-index ceiling; last value repeats
    split: float  # trader share of gross (e.g. 0.90)
    max_payouts: int  # count after which the account reaches its terminal state

    # --- optional; neutral defaults = "no such constraint" ------------------ #
    cap_fraction: float = 1.0  # 1.0 = no fraction limit; e.g. 0.5 = "50% of profit up to cap"
    min_request: float = 0.0  # 0.0 = no minimum cycle profit to request
    buffer_floor: float = 0.0  # 0.0 = no buffer; else an ABSOLUTE non-withdrawable balance

    # --- trader's share, optionally tiered (grandfathered "100% first $X") -- #
    split_first_tier: float | None = None  # split up to a cumulative-paid threshold
    split_tier_cap: float = 0.0  # the cumulative-paid threshold where the tier changes

    # --- the post-payout transition (§6b.1) --------------------------------- #
    reset_fields: tuple[StateField, ...] = ()  # counters zeroed each payout
    withdraw_reduces_equity: bool = True  # does the paid amount leave the balance?
    recompute_floor_on_payout: bool = False  # does DD_FLOOR re-derive post-withdrawal?

    def dollar_cap_at(self, payout_index: int) -> float:
        """The per-request dollar ceiling for the ``payout_index``-th payout (§6b).

        The cap tuple steps with the payout number and its **last element repeats**
        for every later payout — a flat cap ``(2000,)`` applies 2000 to all, while
        ``(2000, 2500)`` applies 2000 to the first and 2500 thereafter.
        """
        if payout_index < 0:
            raise IndexError("payout_index must be >= 0")
        i = min(payout_index, len(self.dollar_cap) - 1)
        return self.dollar_cap[i]


__all__ = ["PayoutSchema"]
