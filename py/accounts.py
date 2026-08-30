"""Test-account registry for the validation dashboard.

Standalone — imports the engine's DSL only; changes nothing in the pipeline.

The **Test Firm** account (per the current spec):

* **Eval:** a **\\$3,000 profit target**, a **\\$2,000 end-of-day Max Loss Limit**
  that trails the EOD balance but **stops trailing once the balance reaches
  \\$52,100** (the floor locks at \\$50,100, i.e. breakeven + \\$100), and a **50%
  consistency rule** (no single day may exceed 50% of the profit, enforced as a
  pass gate). Nothing else.
* **Funded:** the same trailing/locking MLL, and **manual payouts** whose rules
  (choose the amount and timing; \\$500–\\$2,000; at most 50% of total profit; cycle
  profit ≥ \\$1) are modeled by the dashboard bridge, not the engine's auto-firing
  ``PayoutSchema``. So the funded phase here is just the MLL survival rule; the
  bridge injects a chosen payout as a withdrawal so the real engine's MLL sees the
  reduced balance.

Sizes scale: the 100K doubles the target/MLL and locks at balance = size + \\$2,100.
"""

from __future__ import annotations



from propfirm_engine.enums import Action, Timing  # noqa: E402
from propfirm_engine.model import Account, Phase  # noqa: E402
from propfirm_engine.rules import (  # noqa: E402
    ConsistencyGateRule,
    ProfitTargetRule,
    TrailingDrawdownRule,
)

# Dashboard-modeled payout rules (funded stage) — NOT the engine's PayoutSchema.
PAYOUT_MIN = 500.0
PAYOUT_MAX = 2000.0
PAYOUT_PCT_OF_TOTAL = 0.5  # a payout may be at most 50% of total profit
MIN_CYCLE_PROFIT = 1.0  # cycle profit must be at least this to request


def build_test_account(size: int) -> Account:
    mll = round(size * 0.04)  # 50K -> 2,000
    target = round(size * 0.06)  # 50K -> 3,000
    # the floor locks at breakeven + $100 (reached when balance = size + mll + 100,
    # i.e. $52,100 for the 50K), and never trails higher after that.
    lock_at = float(size + 100)

    mll_rule = TrailingDrawdownRule(
        float(mll), update_timing=Timing.EOD, check_timing=Timing.EOD, lock_at=lock_at
    )

    eval_phase = Phase(
        "eval", "eval",
        (
            ProfitTargetRule(float(target)),
            mll_rule,
            ConsistencyGateRule(0.5, gate=Action.PASS),  # 50% consistency to pass
        ),
    )
    # funded: just the MLL; payouts are dashboard-modeled (manual) — no schema.
    funded_phase = Phase("funded", "funded", (mll_rule,))

    return Account(
        name=f"{size // 1000}K",
        size=size,
        phases=(eval_phase, funded_phase),
        eval_fee=150.0,
        activation_fee=0.0,
    )


REGISTRY: dict[str, dict[str, dict[str, Account]]] = {
    "Test Firm": {
        "Standard": {
            "50K": build_test_account(50_000),
            "100K": build_test_account(100_000),
        },
    },
}


def list_registry() -> dict:
    out: dict = {}
    for firm, types in REGISTRY.items():
        out[firm] = {}
        for atype, sizes in types.items():
            out[firm][atype] = {}
            for size_name, acct in sizes.items():
                out[firm][atype][size_name] = {
                    "phases": [p.role for p in acct.phases],
                    "eval_fee": acct.eval_fee,
                    "size": acct.size,
                }
    return out


def get_account(firm: str, atype: str, size: str) -> Account:
    return REGISTRY[firm][atype][size]
