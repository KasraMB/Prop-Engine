"""LucidFlex DLL-off research presets; not verified account replicas.

2026-09-23: first end-to-end reference account selected by the user is 50K,
DLL off. Shared numeric corrections also apply to the other published sizes.
See docs/BRACKET_BACKTEST.md for official sources and modeling assumptions.

The floor trails closing balance at EOD and locks at nominal size + 100.
Continuous breach checking follows the user's explicit real-time-equity
confirmation; the public drawdown page does not explicitly define open equity.
The Engine therefore rejects these closed-summary presets in strict mode.

The legacy payout approximation auto-requests at the qualifying close, uses
50% of retained profit above nominal balance, requires $1 fresh cycle profit,
enforces a $500 gross minimum and applies the 90% split. It does NOT implement
separate request/approval/payment events, request-time floor locking or live handoff. The public
wording does not establish a mandatory extra trading day after qualification.

Funded contract scaling remains missing from the legacy batch path. The
chronological replay_50k profile supplies scaling and dated payout integration;
see docs/BRACKET_BACKTEST.md for its assumptions and capability boundary.
The consistency cushion is deliberately
excluded by user decision: the evaluation uses a conservative strict 50% threshold.
Neither this preset nor the manual dashboard is certified against firm rules.
Zero evaluation fees below are unspecified placeholders, not official prices."""

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

RULE_VERSION = "lucidflex-50k-2026-10-08"
SOURCES = (
    ("evaluation", "https://support.lucidtrading.com/en/articles/12945790-lucidflex-evaluation-account"),
    ("drawdown", "https://support.lucidtrading.com/en/articles/12945815-lucidflex-drawdown"),
    ("payouts", "https://support.lucidtrading.com/en/articles/12945796-lucidflex-payouts"),
    ("scaling", "https://support.lucidtrading.com/en/articles/12945808-lucidflex-scaling-plan"),
    ("hours", "https://support.lucidtrading.com/en/articles/11404729-allowed-trading-times"),
    ("inactivity", "https://support.lucidtrading.com/en/articles/11404632-inactivity-policy"),
    ("commissions", "https://support.lucidtrading.com/en/articles/11508978-approved-products-and-commissions"),
    ("review", "https://support.lucidtrading.com/en/articles/11404742-prohibited-microscalping"),
)


def _mll_rule(size: int, amount: int) -> TrailingDrawdownRule:
    return TrailingDrawdownRule(
        float(amount),
        update_timing=Timing.EOD,
        check_timing=Timing.CONTINUOUS,
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
        min_request=500.0,  # minimum gross request, distinct from positive cycle eligibility
        cap_fraction=0.5,
        fraction_basis="retained_profit",  # user-confirmed, distinct from cycle eligibility
        profit_reference_balance=float(size),
        min_cycle_profit=1.0,
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


def replay_50k(*, eval_fee, reset_fee, contract_type, elapsed_inactivity=False):
    """50K DLL-off lifecycle scenario. Fees and mini/micro choice are explicit.

    Public rules checked 2026-10-08. See the execution guide for evidence limits.
    Cost per contract belongs to BacktestConfig, not this firm preset.
    """
    from dataclasses import replace
    from datetime import time
    from ..execution import LifecycleSpec

    if type(elapsed_inactivity) is not bool:
        raise ValueError("elapsed_inactivity must be bool")
    if contract_type not in ("mini", "micro"):
        raise ValueError("contract_type must be mini or micro")
    multiplier = 10 if contract_type == "micro" else 1
    return LifecycleSpec(
        account=replace(build_account(50_000), eval_fee=eval_fee),
        rule_version=RULE_VERSION,
        eval_contract_limit=4 * multiplier,
        funded_tiers=((float("-inf"), 2 * multiplier),
                      (1000.0, 3 * multiplier), (2000.0, 4 * multiplier)),
        reset_fee=reset_fee,
        request_lock_floor=50_100.0,
        inactivity_days=30,
        inactivity_close=None if elapsed_inactivity else time(16, 15),
        reset_valid_days=30,
        assumptions=(
            "LucidFlex 50K DLL off; public numeric rules checked 2026-10-08",
            "strict 50% evaluation consistency; undocumented cushion excluded by user",
            "MLL uses open equity: user-confirmed interpretation",
            "retained-profit payout basis and $1 fresh cycle profit: user-confirmed",
            "negative funded profit retains the starting scaling tier (scenario interpretation)",
            "scaling bands use exact lower thresholds, not rounded displayed dollars",
            ("inactivity clock restarts on a trade with absolute net P&L >= $1; exact elapsed 30-day boundary"
             if elapsed_inactivity else
             "inactivity expires at 16:15 New York on the qualifying close's local date plus 30 calendar days, including weekends; user-selected scenario"),
            "five approved payouts end each simulated-funded account; earlier live transfer unmodeled",
            "fresh evaluation after live handoff: user-selected research lifecycle, not a firm entitlement",
            "session schedule excludes holidays/early closes unless input is filtered by its producer",
        ),
    )


def info(spec=None, config=None, *, print_output=True):
    """Describe the 50K DLL-off reference, or an explicitly supplied configuration.

    No-argument preview uses mini units and leaves checkout/reset fees unspecified.
    Pass the actual replay spec and config to include your fees and lifecycle choices.
    """
    from ..info import _emit, _report

    preview = spec is None
    if preview:
        spec = replay_50k(eval_fee=0, reset_fee=0, contract_type="mini")
    notes = [
        "Reference values are versioned configuration, not current checkout prices or live-verified terms.",
        "Unknown agreements, discretionary reviews, microscalping review and earlier live transfer are not simulated.",
    ]
    if preview:
        micro = replay_50k(eval_fee=0, reset_fee=0, contract_type="micro")
        notes.append(f"Preview limits use mini units. Micro reference: eval {micro.eval_contract_limit}; funded tiers {micro.funded_tiers}.")
    else:
        notes.append("Supplied configuration may override the LucidFlex reference; limits use its configured contract units.")
    return _emit(_report(spec, config,
        title="LucidFlex 50K DLL-off reference (mini units)" if preview else "LucidFlex - supplied configuration",
        sources=SOURCES, unknown_fees=("eval_fee", "reset_fee") if preview else (), notes=notes), print_output)


__all__ = ["SPECS", "RULE_VERSION", "SOURCES", "build_account", "firm", "replay_50k", "info"]
