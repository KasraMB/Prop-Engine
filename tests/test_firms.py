"""LucidFlex firm config (ARCHITECTURE §7, §12)."""

from __future__ import annotations

import numpy as np

from propfirm_engine.compiler import compile_phase
from propfirm_engine.enums import ExitCode
from propfirm_engine.fingerprint import fingerprint
from propfirm_engine.firms import FIRMS, all_accounts, lucidflex
from propfirm_engine.reference import simulate_reference
from propfirm_engine.validate import validate


def _phase(acct, role):
    return next(p for p in acct.phases if p.role == role)


def test_every_lucidflex_account_validates():
    for _firm, _prog, acct in all_accounts():
        validate(acct)  # raises on any broken account


def test_registry_exposes_lucidflex_with_four_sizes():
    assert "Lucid" in FIRMS  # firm "Lucid"
    assert FIRMS["Lucid"].programs[0].name == "LucidFlex"  # account type "LucidFlex"
    sizes = [a.name for _f, _p, a in all_accounts()]
    assert sizes == ["25K", "50K", "100K", "150K"]


def test_mll_amount_and_lock_match_the_spec():
    # lock floor = size + 100; amount per the published table.
    expected = {25_000: (1_000, 25_100), 50_000: (2_000, 50_100),
                100_000: (3_000, 100_100), 150_000: (4_500, 150_100)}
    for size, (amt, lock) in expected.items():
        cp = compile_phase(_phase(lucidflex.build_account(size), "eval"))
        assert cp.dd_amount == float(amt)
        assert cp.lock_at == float(lock)


def test_profit_targets_match_the_spec():
    targets = {25_000: 1_250, 50_000: 3_000, 100_000: 6_000, 150_000: 9_000}
    for size, target in targets.items():
        cp = compile_phase(_phase(lucidflex.build_account(size), "eval"))
        assert cp.profit_target0 == float(target)


def test_funded_payout_schema_matches_the_spec():
    caps = {25_000: 1_000, 50_000: 2_000, 100_000: 2_500, 150_000: 3_000}
    for size, cap in caps.items():
        cp = compile_phase(_phase(lucidflex.build_account(size), "funded"))
        assert cp.payout.dollar_cap[0] == float(cap)
        assert cp.payout.split == 0.9
        assert cp.payout.max_payouts == 5
        assert cp.winning_day_threshold == float(lucidflex.SPECS[size]["min_daily"])


def test_all_accounts_have_distinct_fingerprints():
    fps = {fingerprint(a, "2026_lucidflex") for _f, _p, a in all_accounts()}
    assert len(fps) == 4


# --- behavioral smoke: the eval MLL + consistency + target, via the real oracle - #


def _run(cp, ret, day, low, start):
    ret = np.asarray(ret, np.float64); day = np.asarray(day, np.int32)
    low = np.asarray(low, np.float64)
    return simulate_reference(cp, ret, day, low, 1.0, np.array([1.0]), float(start))


def test_50k_eval_passes_on_target_with_balanced_consistency():
    cp = compile_phase(_phase(lucidflex.build_account(50_000), "eval"))
    # 50K target 3000; two $1500 days = 50% consistency exactly -> passes
    r = _run(cp, [1500.0, 1500.0], [0, 1], [0.0, 0.0], 50_000)
    assert r.code == int(ExitCode.PASSED)


def test_25k_eval_breaches_when_balance_hits_the_mll():
    cp = compile_phase(_phase(lucidflex.build_account(25_000), "eval"))
    # 25K MLL 1000: day0 +500 -> EOD 25500, floor trails to 24500; day1 -1100 ->
    # 24400 <= 24500 -> breach
    r = _run(cp, [500.0, -1100.0], [0, 1], [0.0, 0.0], 25_000)
    assert r.code == int(ExitCode.FAIL_TRAILING_DD)


def test_100k_funded_fires_a_payout_after_five_qualifying_days():
    cp = compile_phase(_phase(lucidflex.build_account(100_000), "funded"))
    # 100K: 5 days each >= $200 (min daily). cycle profit 5*250=1250; payout =
    # min(cap 2500, 0.5*1250=625) = 625, net 0.9*625 = 562.5 at the 5th close.
    ret = [250.0] * 5
    day = [0, 1, 2, 3, 4]
    low = [0.0] * 5
    r = _run(cp, ret, day, low, 100_000)
    assert r.payouts_taken == 1
    assert r.payout_amounts[0] == 0.9 * 625.0
