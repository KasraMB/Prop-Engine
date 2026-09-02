"""Firm configs (ARCHITECTURE §7, §12). Each firm assembles engine rules into a
``Firm → Program → Variant → Account`` tree; adding one is config, not kernel
changes (a genuinely new mechanic would trip ``RULE_REGISTRY`` first).

``FIRMS`` is the registry keyed by firm name, so tools (the dashboard, sweeps)
can enumerate every implemented account.
"""

from __future__ import annotations

from . import lucidflex

FIRMS = {
    "Lucid": lucidflex.firm(),  # firm "Lucid", account type (program) "LucidFlex"
}


def all_accounts():
    """Yield ``(firm_name, program_name, account)`` for every implemented account."""
    for firm_name, firm in FIRMS.items():
        for program in firm.programs:
            for variant in program.variants:
                for account in variant.accounts:
                    yield firm_name, program.name, account


__all__ = ["FIRMS", "lucidflex", "all_accounts"]
