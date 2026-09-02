"""The structural fingerprint — one authoritative identity for an account
(ARCHITECTURE §10; BUILD_SPEC Step 7).

Because every DSL object is a frozen tuple of primitives, a structural hash is
well-defined and stable. The fingerprint is both the **compiled-account cache
key** and the **reproducibility record**: a result is tied to config ``a3f9…``
(e.g. LucidFlexDLL 50K v2026_08). The program **version string lives inside the
hashed content**, so it cannot disagree with the hash, and per-account resolved
values are hashed, so a size-specific quirk (a differing ``min_days`` on one
size, a rule's ``severity``) gets its own fingerprint.

**The payout schema is part of the identity** (§10, MODEL_RISKS §A1): two funded
accounts differing only in ``split``/``dollar_cap``/``cap_fraction`` or a
post-payout flag must not collide to one cache key. So are the fees — the fee is
the entire downside (§0), so two accounts differing only in fee are different
products.

**Deviation from the §10 sketch — rule ORDER is preserved, not sorted.** The
sketch sorts each phase's rule list; but rule order defines fail-precedence
(§6/§C4), so two accounts with the same rules in a different order compile to
different arrays and *behave differently*. Sorting would collide them to one
cache key and hand back the wrong compiled artifact. The fingerprint therefore
keeps rules in author order.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass


def _rule_terms(rule) -> list[str]:
    """A stable, value-derived description of one rule: its type name followed by
    its ``field=value`` pairs in field order (dataclass field order is stable)."""
    terms = [type(rule).__name__]
    for f in fields(rule):
        terms.append(f"{f.name}={getattr(rule, f.name)!r}")
    return terms


def _schema_terms(schema) -> list[str]:
    return [f"{f.name}={getattr(schema, f.name)!r}" for f in fields(schema)]


def fingerprint(account, program_version: str = "v1") -> str:
    """A 16-hex-char structural hash of ``account`` under ``program_version`` (§10).

    Identical configs hash identically; any value change — a rule parameter, a
    severity, a timing field, the version string, a fee, or any ``PayoutSchema``
    field — changes the hash. Rule order within a phase is significant.
    """
    payload = {
        "version": program_version,
        "name": account.name,
        "size": account.size,
        "currency": account.currency,
        "eval_fee": account.eval_fee,
        "activation_fee": account.activation_fee,
        "phases": [
            {
                "name": ph.name,
                "role": ph.role,
                # author order preserved (precedence is order-defined, §6/§C4)
                "rules": [_rule_terms(r) for r in ph.rules],
                "payout": (
                    _schema_terms(ph.payout_schema)
                    if ph.payout_schema is not None
                    else None
                ),
            }
            for ph in account.phases
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _assert_hashable_tree(account) -> None:
    """Best-effort guard that the account is the all-frozen tree the fingerprint
    assumes (used only in tests / debugging)."""
    assert is_dataclass(account)


__all__ = ["fingerprint"]
