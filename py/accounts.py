"""LucidFlex accounts for the validation dashboard.

Imports the engine DSL only. The **eval** phase is fully engine-driven (profit
target + 50% consistency gate + trailing/locking MLL). The **funded** phase used
by the dashboard is just the MLL (survival) — the LucidFlex payout is a *manual*
request whose exact rules (5 qualifying days, day-after timing, $500–min(cap, 50%
of total profit), 90/10 split, 5 max) are enforced in :mod:`bridge`, injected
into the real simulator as withdrawals so the MLL sees the reduced balance.

Per-size specification (from the firm's published rules):

    size     MLL     lock floor    target    min daily    payout cap
    25K    $1,000    $25,100       $1,250      $100         $1,000
    50K    $2,000    $50,100       $3,000      $150         $2,000
    100K   $3,000   $100,100       $6,000      $200         $2,500
    150K   $4,500   $150,100       $9,000      $250         $3,000
"""

from __future__ import annotations



from propfirm_engine.enums import Action, Timing  # noqa: E402
from propfirm_engine.model import Account, Phase  # noqa: E402
from propfirm_engine.rules import (  # noqa: E402
    ConsistencyGateRule,
    ProfitTargetRule,
    TrailingDrawdownRule,
)

# size -> spec
SPECS = {
    "25K": dict(size=25_000, mll=1_000, target=1_250, min_daily=100, cap=1_000),
    "50K": dict(size=50_000, mll=2_000, target=3_000, min_daily=150, cap=2_000),
    "100K": dict(size=100_000, mll=3_000, target=6_000, min_daily=200, cap=2_500),
    "150K": dict(size=150_000, mll=4_500, target=9_000, min_daily=250, cap=3_000),
}

# LucidFlex funded payout rules (enforced in bridge.py)
CONSISTENCY = 0.5
SPLIT = 0.9  # trader receives 90% of the requested amount
MIN_PAYOUT = 500.0
PCT_OF_TOTAL = 0.5  # max request = 50% of total profit, up to the per-size cap
QUALIFYING_DAYS = 5  # closed days with day P&L >= min-daily, reset per payout
MIN_CYCLE_PROFIT = 1.0
MAX_PAYOUTS = 5


def build_account(spec) -> Account:
    size = spec["size"]
    mll = TrailingDrawdownRule(
        float(spec["mll"]), update_timing=Timing.EOD, check_timing=Timing.EOD,
        lock_at=float(size + 100),
    )
    eval_phase = Phase(
        "eval", "eval",
        (ProfitTargetRule(float(spec["target"])), mll,
         ConsistencyGateRule(CONSISTENCY, gate=Action.PASS)),
    )
    funded_phase = Phase("funded", "funded", (mll,))  # payouts are dashboard-modeled
    return Account(f"{size // 1000}K", size, phases=(eval_phase, funded_phase),
                   eval_fee=0.0, activation_fee=0.0)


REGISTRY = {
    "Lucid": {
        "LucidFlex": {name: build_account(spec) for name, spec in SPECS.items()},
    },
}


def payout_params(size_name: str) -> dict:
    s = SPECS[size_name]
    return dict(cap=float(s["cap"]), min_daily=float(s["min_daily"]),
                min_payout=MIN_PAYOUT, split=SPLIT, pct=PCT_OF_TOTAL,
                qualifying_days=QUALIFYING_DAYS, min_cycle=MIN_CYCLE_PROFIT,
                max_payouts=MAX_PAYOUTS)


def list_registry() -> dict:
    out: dict = {}
    for firm, types in REGISTRY.items():
        out[firm] = {}
        for atype, sizes in types.items():
            out[firm][atype] = {}
            for size_name, acct in sizes.items():
                out[firm][atype][size_name] = {
                    "phases": [p.role for p in acct.phases],
                    "eval_fee": acct.eval_fee, "size": acct.size,
                }
    return out


def get_account(firm: str, atype: str, size: str) -> Account:
    return REGISTRY[firm][atype][size]
