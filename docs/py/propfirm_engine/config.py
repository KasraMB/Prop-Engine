"""Firm config — the three-layer format (ARCHITECTURE §7; BUILD_SPEC Step 4).

The taxonomy is ``Firm → Program → (default) Variant → Account(size) → Phase →
Rule``. The format is built for *arbitrary irregularity first*, so regular
accounts are the cheap special case:

* **Layer 1 — unconditional floor: tables of real rule objects.** A "cell" is a
  mapping ``{role: (rules...)}`` (e.g. ``{"eval": (...), "funded": (...)}``); a
  table is ``{size_name: cell}``. Anything expressible as rules — per-account
  severity, which counters a payout resets, wholly different structure per size —
  is expressible here, and this floor never narrows. Hand-written cells are the
  primitive; the helpers below only *produce* Layer-1 data.
* **Layer 2 — opt-in sugar: :func:`scaled` + :func:`build_accounts`.** For
  account types whose sizes share structure and only scale in value. They never
  restrict what a table may contain; a type can scale its regular sizes and
  hand-write a quirky one in the same table.
* **Layer 3 — safety net: ``validate`` + ``RULE_REGISTRY``** (in
  :mod:`propfirm_engine.validate` and :mod:`propfirm_engine.rules`). Because
  Layer 1 accepts anything, these catch what a permissive format would pass
  silently: the registry rejects *unimplemented* rules, the validator rejects
  *broken* accounts — without rejecting merely *irregular* ones.
"""

from __future__ import annotations

from collections.abc import Mapping

from .model import Account, Phase
from .schema import PayoutSchema


def scaled(rule_cls, per_size: Mapping[str, float], **fixed) -> dict:
    """``{size: rule_cls(value, **fixed)}`` — Layer-2 sugar for the regular case.

    Builds one rule instance per size from a ``{size: value}`` mapping, threading
    the same fixed keyword arguments into each (e.g. a shared ``severity`` or a
    winning-day ``threshold``). Purely a convenience that *produces* Layer-1 rule
    objects; it constrains nothing.
    """
    return {size: rule_cls(value, **fixed) for size, value in per_size.items()}


def build_accounts(
    name_prefix: str,
    sizes: Mapping[str, int],
    cells: Mapping[str, Mapping],
    *,
    payouts: PayoutSchema | Mapping[str, PayoutSchema] | None = None,
    fees: tuple[float, float] | Mapping[str, tuple[float, float]] | None = None,
) -> tuple[Account, ...]:
    """Turn a Layer-1 table into :class:`Account` objects (§7).

    ``cells`` is exactly a Layer-1 table — ``{size_name: {role: (rules...)}}`` —
    however it was produced (by hand or from :func:`scaled`). Each size becomes an
    account whose phases mirror the cell's roles in insertion order.

    ``payouts`` optionally attaches a :class:`PayoutSchema` to each account's
    **funded** phase — one schema shared across sizes, or a ``{size: schema}``
    mapping. ``fees`` optionally sets ``(eval_fee, activation_fee)`` per account —
    one pair shared, or a ``{size: (eval, activation)}`` mapping; the fee is the
    entire downside (§0) and part of account identity.

    ``name_prefix`` is the caller's product label (the program name); accounts are
    named by their size key, matching the architecture sketch.
    """
    accounts: list[Account] = []
    for size_name, size_val in sizes.items():
        cell = cells[size_name]
        phases = tuple(
            Phase(
                name=role,
                role=role,
                rules=tuple(rules),
                payout_schema=(
                    _select(payouts, size_name) if role == "funded" else None
                ),
            )
            for role, rules in cell.items()
        )
        eval_fee, activation_fee = _select_fees(fees, size_name)
        accounts.append(
            Account(
                name=size_name,
                size=size_val,
                phases=phases,
                eval_fee=eval_fee,
                activation_fee=activation_fee,
            )
        )
    return tuple(accounts)


def _select(per_size, size_name):
    """Return a shared value, a per-size value, or ``None``. A per-size *mapping*
    that omits a size is an authoring error (a size silently losing its schema),
    so raise rather than degrade to ``None``."""
    if per_size is None:
        return None
    if isinstance(per_size, Mapping):
        if size_name not in per_size:
            raise KeyError(
                f"per-size payouts mapping is missing size {size_name!r}"
            )
        return per_size[size_name]
    return per_size


def _select_fees(fees, size_name) -> tuple[float, float]:
    if fees is None:
        return (0.0, 0.0)
    if isinstance(fees, Mapping):
        if size_name not in fees:
            raise KeyError(f"per-size fees mapping is missing size {size_name!r}")
        pair = fees[size_name]
    else:
        pair = fees
    eval_fee, activation_fee = pair
    return (float(eval_fee), float(activation_fee))


__all__ = ["scaled", "build_accounts"]
