"""Step 9 — batch Monte Carlo + engine (BUILD_SPEC Step 9, ARCHITECTURE §17)."""

from __future__ import annotations

import numpy as np

from propfirm_engine.cache import Caches
from propfirm_engine.compiler import compile_phase
from propfirm_engine.data import preprocess
from propfirm_engine.engine import Engine, Outcomes, RunConfig
from propfirm_engine.enums import ExitCode, StateField
from propfirm_engine.kernels import simulate_one_phase
from propfirm_engine.model import Account, Phase
from propfirm_engine.resampling import IIDDayBootstrap, gather_days
from propfirm_engine.rules import (
    MinimumTradingDaysRule,
    MinimumWinningDaysRule,
    ProfitTargetRule,
    TrailingDrawdownRule,
)
from propfirm_engine.schema import PayoutSchema
from propfirm_engine.synthetic import IIDGenerator

SIZE = 50_000
SIZE_BASE = 100.0  # each R (unit return) = $100
_SCHEMA = PayoutSchema(dollar_cap=(2000.0,), split=0.9, max_payouts=3,
                       reset_fields=(StateField.N_QUALIFYING_DAYS,))


def _dataset(seed=1, n_days=40):
    stream = IIDGenerator(win_rate=0.4, rr=2.0, trades_per_day=3).generate(n_days, seed)
    return preprocess(stream.rows)


def _account(target=1000.0, dd=1200.0, min_days=1):
    return Account(
        "50K",
        SIZE,
        phases=(
            Phase("eval", "eval", (ProfitTargetRule(target), TrailingDrawdownRule(dd),
                                   MinimumTradingDaysRule(min_days))),
            Phase("funded", "funded", (TrailingDrawdownRule(dd),
                                       MinimumWinningDaysRule(1, 1.5)), payout_schema=_SCHEMA),
        ),
        eval_fee=150.0,
        activation_fee=50.0,
    )


def _cfg(**kw):
    base = dict(n_paths=200, L_eval=20, L_funded=30, seed=3,
                resampler=IIDDayBootstrap(), size_base=SIZE_BASE)
    base["intraday_mode"] = "summary_approximation"  # legacy summary-model scenarios
    base.update(kw)
    return RunConfig(**base)


# --- batch <-> single-path oracle agreement (the core Step 9 contract) ------ #


def test_batch_matches_the_single_path_kernel_on_each_path():
    from propfirm_engine.simulate import simulate_phase_batch

    ds = _dataset()
    cp = compile_phase(Phase("eval", "eval",
                             (ProfitTargetRule(1000.0), TrailingDrawdownRule(1200.0))))
    paths = IIDDayBootstrap().generate(ds.n_days, 25, 40, seed=5)
    policy = np.array([1.0])
    br = simulate_phase_batch(cp, ds, paths, SIZE_BASE, policy, float(SIZE))
    for i in range(paths.shape[0]):
        ret, day, low = gather_days(ds, paths[i])
        c, amts, days, nd = simulate_one_phase(cp, ret, day, low, SIZE_BASE, policy, float(SIZE))
        assert br.code[i] == c
        assert br.payouts_taken[i] == len(amts)
        assert br.total_trading_days[i] == nd
        assert br.net_payout[i] == (sum(amts) if amts else 0.0)


def test_non_trivial_policy_threads_unchanged_to_the_kernel():
    from propfirm_engine.simulate import simulate_phase_batch

    ds = _dataset()
    cp = compile_phase(Phase("eval", "eval",
                             (ProfitTargetRule(1000.0), TrailingDrawdownRule(5000.0))))
    paths = IIDDayBootstrap().generate(ds.n_days, 20, 30, seed=6)
    policy = np.array([0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5])  # stage-varying
    br = simulate_phase_batch(cp, ds, paths, SIZE_BASE, policy, float(SIZE))
    for i in range(paths.shape[0]):
        ret, day, low = gather_days(ds, paths[i])
        c, amts, days, nd = simulate_one_phase(cp, ret, day, low, SIZE_BASE, policy, float(SIZE))
        assert br.code[i] == c  # the hook is a pure passthrough


# --- survivors-only multi-phase (§17) --------------------------------------- #


def test_funded_runs_only_for_eval_survivors():
    out = Engine().run(_account(), _dataset(), _cfg())
    # every attempt that reached funded must have passed eval; the rest must have a
    # non-PASSED eval terminal and no funded economics.
    assert out.reached_funded.sum() > 0  # some pass
    assert (~out.reached_funded).sum() > 0  # some fail
    failed = ~out.reached_funded
    assert np.all(out.net_payout[failed] == 0.0)
    assert np.all(out.payouts_taken[failed] == 0)
    assert np.all(out.first_payout_day[failed] == -1)
    assert np.all(out.code[failed] != int(ExitCode.PASSED))


def test_total_trading_days_actually_sums_eval_and_funded(monkeypatch):
    # Verify the §H4 sum directly: intercept the per-phase batch results and confirm
    # reached-funded attempts' total == eval_days + funded_days (not eval-only,
    # funded-only, or an overwrite).
    import propfirm_engine.engine as eng_mod

    captured = []
    orig = eng_mod.simulate_phase_batch

    def spy(*a, **k):
        r = orig(*a, **k)
        captured.append(r.total_trading_days.copy())
        return r

    monkeypatch.setattr(eng_mod, "simulate_phase_batch", spy)
    out = Engine().run(_account(), _dataset(), _cfg())
    eval_days, funded_days = captured  # two phases, in order
    reached = out.reached_funded
    survivors = np.where(reached)[0]
    expected = eval_days.copy()
    expected[survivors] += funded_days  # funded_days is in survivor order
    np.testing.assert_array_equal(out.total_trading_days, expected)


def test_time_to_first_payout_is_measured_from_attempt_start_not_funded_start():
    # first_payout_day must include the eval duration (§H4): for a reached-funded
    # attempt with a payout, it should be >= that attempt's eval day-count, never a
    # funded-relative index that ignores the eval phase.
    import propfirm_engine.engine as eng_mod

    captured = []
    orig = eng_mod.simulate_phase_batch

    def spy(*a, **k):
        r = orig(*a, **k)
        captured.append(r)
        return r

    import pytest
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(eng_mod, "simulate_phase_batch", spy)
    out = Engine().run(_account(), _dataset(), _cfg())
    monkeypatch.undo()
    eval_r, funded_r = captured
    survivors = np.where(out.reached_funded)[0]
    eval_days = eval_r.total_trading_days
    for j, i in enumerate(survivors):
        if funded_r.first_payout_day[j] >= 0:
            assert out.first_payout_day[i] == eval_days[i] + funded_r.first_payout_day[j]
            assert out.first_payout_day[i] >= eval_days[i]


def test_direct_funded_account_runs_funded_for_all_attempts():
    acct = Account("25K", 25_000, phases=(
        Phase("funded", "funded", (TrailingDrawdownRule(2000.0),
                                   MinimumWinningDaysRule(1, 1.5)), payout_schema=_SCHEMA),),
        eval_fee=100.0)
    out = Engine().run(acct, _dataset(), _cfg(n_paths=50))
    assert np.all(out.reached_funded)  # trivially true from t=0 (§14.1)


# --- determinism / batch-size invariance ------------------------------------ #


def test_same_seed_and_config_are_deterministic():
    a = Engine().run(_account(), _dataset(), _cfg())
    b = Engine().run(_account(), _dataset(), _cfg())
    np.testing.assert_array_equal(a.code, b.code)
    np.testing.assert_array_equal(a.net_payout, b.net_payout)
    np.testing.assert_array_equal(a.total_trading_days, b.total_trading_days)


def test_batch_size_is_a_memory_knob_not_a_result_changer():
    small = Engine().run(_account(), _dataset(), _cfg(batch_size=13))
    large = Engine().run(_account(), _dataset(), _cfg(batch_size=5000))
    np.testing.assert_array_equal(small.code, large.code)
    np.testing.assert_array_equal(small.net_payout, large.net_payout)


# --- caches (§10) ----------------------------------------------------------- #


def test_running_the_same_account_twice_hits_the_caches():
    caches = Caches()
    eng = Engine(caches)
    ds = _dataset()
    first = eng.run(_account(), ds, _cfg())
    second = eng.run(_account(), ds, _cfg())
    # the compiled-account cache served the second run without recompiling
    assert caches.accounts.hits == 1
    assert caches.accounts.misses == 1
    # identical results
    np.testing.assert_array_equal(first.code, second.code)


def test_trade_cache_is_used_when_raw_rows_are_passed():
    caches = Caches()
    eng = Engine(caches)
    rows = IIDGenerator(win_rate=0.4, rr=2.0, trades_per_day=3).generate(40, 1).rows
    eng.run(_account(), rows, _cfg())
    eng.run(_account(), rows, _cfg())
    assert caches.trades.hits == 1  # second run reused the preprocessed dataset


# --- raw outcomes retain the fields the stats layer needs ------------------- #


def test_outcomes_carry_the_metadata_the_statistics_layer_needs():
    out = Engine().run(_account(), _dataset(), _cfg())
    assert isinstance(out, Outcomes)
    assert out.size == SIZE
    assert out.max_payouts == 3  # from the funded schema (§H2)
    assert out.eval_fee == 150.0
    assert out.activation_fee == 50.0
    assert out.trading_days_per_week > 0
    assert len(out.fingerprint) == 16
    assert out.n_attempts == 200
    assert out.size_base == SIZE_BASE  # the P&L-driving scalar, distinct from `size`


def test_provenance_is_carried_through_to_the_outcomes():
    out = Engine().run(_account(), _dataset(), _cfg(provenance="held-out 2023 OOS"))
    assert out.provenance == "held-out 2023 OOS"


def test_engine_threads_a_non_trivial_policy_end_to_end():
    # An IN_PROFIT-scaling policy through the full engine must match a re-run of the
    # same account under that policy (deterministic passthrough, no engine behavior).
    policy = [0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0]
    a = Engine().run(_account(), _dataset(), _cfg(), policy_params=policy)
    b = Engine().run(_account(), _dataset(), _cfg(), policy_params=policy)
    np.testing.assert_array_equal(a.code, b.code)
    np.testing.assert_array_equal(a.net_payout, b.net_payout)
    # and it differs from the constant-size run (the policy actually did something)
    const = Engine().run(_account(), _dataset(), _cfg())
    assert not np.array_equal(a.total_trading_days, const.total_trading_days)
