"""Step 4 — firm config, the three-layer format (BUILD_SPEC Step 4, ARCHITECTURE §7)."""

from __future__ import annotations

from propfirm_engine.config import build_accounts, scaled
from propfirm_engine.model import Account, Phase
from propfirm_engine.rules import (
    DailyLossRule,
    MinimumTradingDaysRule,
    MinimumWinningDaysRule,
    ProfitTargetRule,
    TrailingDrawdownRule,
)
from propfirm_engine.schema import PayoutSchema

SIZES = {"50K": 50_000, "100K": 100_000}


def _eval(target, dd, min_days):
    return (
        ProfitTargetRule(target),
        TrailingDrawdownRule(dd),
        MinimumTradingDaysRule(min_days),
    )


def _funded(dd):
    return (TrailingDrawdownRule(dd), MinimumWinningDaysRule(5, 150.0))


# --- Layer 1: hand-written cells -------------------------------------------- #


def test_hand_written_cell_builds_the_intended_account():
    cells = {"50K": {"eval": _eval(3000.0, 2500.0, 3), "funded": _funded(2500.0)}}
    (acct,) = build_accounts("Lucid", {"50K": 50_000}, cells)
    assert acct.name == "50K"
    assert acct.size == 50_000
    assert tuple(p.role for p in acct.phases) == ("eval", "funded")
    assert acct.phases[0].rules == _eval(3000.0, 2500.0, 3)
    assert acct.phases[1].rules == _funded(2500.0)


def test_two_sizes_with_different_structure_coexist():
    # 50K has a daily-loss rule in eval; 100K does not — genuinely different
    # structure in one table (BUILD_SPEC Step 4).
    cells = {
        "50K": {
            "eval": (*_eval(3000.0, 2500.0, 3), DailyLossRule(1000.0)),
            "funded": _funded(2500.0),
        },
        "100K": {"eval": _eval(6000.0, 3000.0, 3), "funded": _funded(3000.0)},
    }
    accts = build_accounts("Lucid", SIZES, cells)
    a50, a100 = accts
    assert any(isinstance(r, DailyLossRule) for r in a50.phases[0].rules)
    assert not any(isinstance(r, DailyLossRule) for r in a100.phases[0].rules)


# --- Layer 2: sugar produces value-identical Layer-1 accounts --------------- #


def test_scaled_produces_per_size_rule_instances():
    got = scaled(ProfitTargetRule, {"50K": 3000.0, "100K": 6000.0})
    assert got == {"50K": ProfitTargetRule(3000.0), "100K": ProfitTargetRule(6000.0)}


def test_scaled_threads_fixed_keyword_arguments():
    got = scaled(MinimumWinningDaysRule, {"50K": 5, "100K": 5}, threshold=150.0)
    assert got["50K"] == MinimumWinningDaysRule(5, threshold=150.0)


def test_sugar_accounts_are_value_identical_to_hand_written():
    # Build the same accounts two ways; frozen dataclasses compare by value, so
    # identity means the sugar constrained nothing.
    cells = {
        "50K": {"eval": _eval(3000.0, 2500.0, 3), "funded": _funded(2500.0)},
        "100K": {"eval": _eval(6000.0, 3000.0, 3), "funded": _funded(3000.0)},
    }
    built = build_accounts("Lucid", SIZES, cells)

    hand = (
        Account(
            "50K",
            50_000,
            phases=(
                Phase("eval", "eval", _eval(3000.0, 2500.0, 3)),
                Phase("funded", "funded", _funded(2500.0)),
            ),
        ),
        Account(
            "100K",
            100_000,
            phases=(
                Phase("eval", "eval", _eval(6000.0, 3000.0, 3)),
                Phase("funded", "funded", _funded(3000.0)),
            ),
        ),
    )
    assert built == hand


# --- payout schema + fees attach correctly ---------------------------------- #


def test_payout_schema_attaches_only_to_the_funded_phase():
    schema = PayoutSchema(dollar_cap=(2000.0,), split=0.9, max_payouts=5)
    cells = {"50K": {"eval": _eval(3000.0, 2500.0, 3), "funded": _funded(2500.0)}}
    (acct,) = build_accounts("Lucid", {"50K": 50_000}, cells, payouts=schema)
    assert acct.phases[0].payout_schema is None  # eval
    assert acct.phases[1].payout_schema is schema  # funded


def test_per_size_payouts_and_fees_are_selected():
    schemas = {
        "50K": PayoutSchema(dollar_cap=(2000.0,), split=0.9, max_payouts=5),
        "100K": PayoutSchema(dollar_cap=(2500.0,), split=0.9, max_payouts=5),
    }
    fees = {"50K": (150.0, 0.0), "100K": (300.0, 50.0)}
    cells = {
        "50K": {"eval": _eval(3000.0, 2500.0, 3), "funded": _funded(2500.0)},
        "100K": {"eval": _eval(6000.0, 3000.0, 3), "funded": _funded(3000.0)},
    }
    a50, a100 = build_accounts("Lucid", SIZES, cells, payouts=schemas, fees=fees)
    assert a50.phases[1].payout_schema.dollar_cap == (2000.0,)
    assert a100.phases[1].payout_schema.dollar_cap == (2500.0,)
    assert (a50.eval_fee, a50.activation_fee) == (150.0, 0.0)
    assert (a100.eval_fee, a100.activation_fee) == (300.0, 50.0)


def test_shared_fees_pair_applies_to_every_size():
    cells = {
        "50K": {"eval": _eval(3000.0, 2500.0, 3), "funded": _funded(2500.0)},
        "100K": {"eval": _eval(6000.0, 3000.0, 3), "funded": _funded(3000.0)},
    }
    a50, a100 = build_accounts("Lucid", SIZES, cells, fees=(100.0, 25.0))
    assert (a50.eval_fee, a50.activation_fee) == (100.0, 25.0)
    assert (a100.eval_fee, a100.activation_fee) == (100.0, 25.0)


def test_direct_funded_account_has_a_single_phase():
    cells = {"25K": {"funded": _funded(1500.0)}}
    (acct,) = build_accounts("Direct", {"25K": 25_000}, cells)
    assert tuple(p.role for p in acct.phases) == ("funded",)


def test_per_size_fees_mapping_missing_a_size_raises():
    # A per-size mapping that omits a size would silently zero its fees — an
    # authoring footgun on large tables — so build_accounts raises instead.
    import pytest

    cells = {
        "50K": {"eval": _eval(3000.0, 2500.0, 3), "funded": _funded(2500.0)},
        "100K": {"eval": _eval(6000.0, 3000.0, 3), "funded": _funded(3000.0)},
    }
    with pytest.raises(KeyError):
        build_accounts("Lucid", SIZES, cells, fees={"50K": (150.0, 0.0)})
