"""LucidFlex — a concrete firm config (ARCHITECTURE §7, §12 "additional firms").

Assembled from the engine's existing rules, so nothing new is compiled: an eval
phase (profit target + 50% consistency + trailing MLL) and a funded phase (the
same MLL + a payout schema). Four account sizes.

Per-size specification (from the firm's published rules):

    size     MLL     lock floor    profit target   min daily (funded)   payout cap
    25K    $1,000    $25,100          $1,250            $100               $1,000
    50K    $2,000    $50,100          $3,000            $150               $2,000
    100K   $3,000   $100,100          $6,000            $200               $2,500
    150K   $4,500   $150,100          $9,000            $250               $3,000

**MLL** trails the end-of-day balance and locks at ``size + 100`` once the balance
reaches ``size + amount + 100`` (the trail), after which it stops rising. Modeled
as a ``TrailingDrawdownRule`` with ``update=EOD`` and ``lock_at = size + 100``.
The breach check is **EOD** here (matching the validated test account); real-time
breach is a one-line switch to ``check_timing=CONTINUOUS``.

**Payout model — batch approximation.** The real LucidFlex payout is a *manual*
request (the trader chooses the amount, $500–min(50% of total profit, cap)),
allowed only the *day after* five qualifying days (day P&L ≥ min-daily, reset per
payout) and a positive cycle. The auto-firing ``PayoutSchema`` cannot express the
manual amount, the "50% of *total* profit" cap, or the one-day delay, so for
Monte-Carlo valuation this config fires at the qualifying close for
``min(cap, 50% of cycle profit)``, 90/10 split, up to 5 payouts, no buffer. The
dashboard enforces the exact rules interactively; extend the engine's payout
amount (total-profit basis + day-after gate) for exact batch valuation.
"""

from __future__ import annotations

from ..enums import Action, StateField, Timing
from ..model import Account, Firm, Phase, Program
from ..rules import (
    ConsistencyGateRule,
    MinimumWinningDaysRule,
    ProfitTargetRule,
    TrailingDrawdownRule,
)
from ..schema import PayoutSchema

#: size -> spec. ``eval_fee``/``activation_fee`` are left 0.0 (the firm's prices
#: were not provided); set them before trusting any fee-denominated statistic.
SPECS: dict[int, dict] = {
    25_000: dict(mll=1_000, target=1_250, min_daily=100, cap=1_000),
    50_000: dict(mll=2_000, target=3_000, min_daily=150, cap=2_000),
    100_000: dict(mll=3_000, target=6_000, min_daily=200, cap=2_500),
    150_000: dict(mll=4_500, target=9_000, min_daily=250, cap=3_000),
}

CONSISTENCY = 0.5  # eval: no single day may exceed 50% of the profit
SPLIT = 0.9  # 90% to the trader
MAX_PAYOUTS = 5
QUALIFYING_DAYS = 5  # trading days with the minimum daily profit


def _mll_rule(size: int, amount: int) -> TrailingDrawdownRule:
    return TrailingDrawdownRule(
        float(amount),
        update_timing=Timing.EOD,
        check_timing=Timing.EOD,
        lock_at=float(size + 100),
    )


def build_account(size: int) -> Account:
    """The eval + funded LucidFlex account for ``size``."""
    spec = SPECS[size]
    mll = _mll_rule(size, spec["mll"])

    eval_phase = Phase(
        "eval",
        "eval",
        (
            ProfitTargetRule(float(spec["target"])),
            mll,
            ConsistencyGateRule(CONSISTENCY, gate=Action.PASS),
        ),
    )

    schema = PayoutSchema(
        dollar_cap=(float(spec["cap"]),),  # flat per-size cap (does not scale with count)
        split=SPLIT,
        max_payouts=MAX_PAYOUTS,
        min_request=1.0,  # positive cycle profit (>= $1) to be eligible
        cap_fraction=CONSISTENCY,  # "50% of profit" (cycle-profit basis in batch — see docstring)
        buffer_floor=0.0,  # no buffer
        reset_fields=(StateField.N_QUALIFYING_DAYS,),  # qualifying days reset per payout
        withdraw_reduces_equity=True,
    )
    funded_phase = Phase(
        "funded",
        "funded",
        (mll, MinimumWinningDaysRule(QUALIFYING_DAYS, float(spec["min_daily"]))),
        payout_schema=schema,
    )

    return Account(
        name=f"{size // 1000}K",
        size=size,
        phases=(eval_phase, funded_phase),
        eval_fee=0.0,  # firm prices not provided — set before fee-based valuation
        activation_fee=0.0,
    )


def firm() -> Firm:
    """The Lucid firm with its LucidFlex program (account type): four sizes under
    the default variant."""
    accounts = tuple(build_account(size) for size in SPECS)
    program = Program.with_default_variant("LucidFlex", accounts, version="2026_lucidflex")
    return Firm("Lucid", (program,))


__all__ = ["SPECS", "build_account", "firm"]
