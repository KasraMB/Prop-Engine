"""Step 8 — resampling (BUILD_SPEC Step 8, ARCHITECTURE §11.4, MODEL_RISKS §G1)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from propfirm_engine.data import preprocess
from propfirm_engine.resampling import (
    IIDDayBootstrap,
    StationaryDayBootstrap,
    gather_days,
)


def _dataset(n_days=20, trades_per_day=3, multi_asset=False):
    rows = []
    for d in range(n_days):
        for k in range(trades_per_day):
            ts = datetime(2024, 1, 1 + d, 10 + k)
            row = {"timestamp": ts, "return": float(d * 10 + k)}
            if multi_asset:
                row["symbol"] = "ES" if k % 2 == 0 else "NQ"
            rows.append(row)
    return preprocess(rows, session_reset="17:00")


def _consecutive_fraction(path, n_days):
    """Fraction of steps where day[i] == (day[i-1]+1) mod n_days (block continuation)."""
    nxt = (path[:, :-1] + 1) % n_days
    return float(np.mean(path[:, 1:] == nxt))


# --- shape / validity contract ---------------------------------------------- #


@pytest.mark.parametrize("gen", [IIDDayBootstrap(), StationaryDayBootstrap(3.0)])
def test_path_is_exactly_L_days_and_references_only_real_days(gen):
    n_days, L, n_paths = 20, 15, 50
    paths = gen.generate(n_days, L, n_paths, seed=1)
    assert paths.shape == (n_paths, L)  # exactly L, an explicit input
    assert paths.min() >= 0 and paths.max() < n_days  # only real source days


def test_L_is_explicit_not_inferred_from_source_length():
    gen = IIDDayBootstrap()
    short = gen.generate(20, 5, 10, seed=1)
    long = gen.generate(20, 200, 10, seed=1)  # L can exceed the 20 source days
    assert short.shape[1] == 5
    assert long.shape[1] == 200


def test_eval_and_funded_may_use_different_L():
    gen = StationaryDayBootstrap(4.0)
    eval_paths = gen.generate(20, 10, 5, seed=7)
    funded_paths = gen.generate(20, 40, 5, seed=8)
    assert eval_paths.shape == (5, 10)
    assert funded_paths.shape == (5, 40)


# --- determinism ------------------------------------------------------------ #


@pytest.mark.parametrize("gen", [IIDDayBootstrap(), StationaryDayBootstrap(3.0)])
def test_fixed_seed_reproduces_identical_paths(gen):
    a = gen.generate(20, 30, 20, seed=42)
    b = gen.generate(20, 30, 20, seed=42)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("gen", [IIDDayBootstrap(), StationaryDayBootstrap(3.0)])
def test_different_seed_differs(gen):
    a = gen.generate(20, 30, 20, seed=1)
    b = gen.generate(20, 30, 20, seed=2)
    assert not np.array_equal(a, b)


# --- block structure (§G1) -------------------------------------------------- #


def test_iid_has_essentially_no_consecutive_runs():
    paths = IIDDayBootstrap().generate(50, 200, 200, seed=3)
    # chance level of a consecutive step is 1/n_days = 0.02
    assert _consecutive_fraction(paths, 50) < 0.06


def test_stationary_block_structure_scales_with_mean_block():
    n_days = 50
    f2 = _consecutive_fraction(StationaryDayBootstrap(2.0).generate(n_days, 300, 200, 5), n_days)
    f10 = _consecutive_fraction(StationaryDayBootstrap(10.0).generate(n_days, 300, 200, 5), n_days)
    # P(continue) = 1 - 1/mean_block: ~0.5 for mean 2, ~0.9 for mean 10
    assert 0.4 < f2 < 0.6
    assert 0.85 < f10 < 0.95
    assert f10 > f2  # longer blocks -> more consecutive-day runs


def test_iid_is_the_short_block_limit_of_the_stationary_bootstrap():
    n_days = 50
    stat1 = StationaryDayBootstrap(1.0).generate(n_days, 300, 200, 9)
    iid = IIDDayBootstrap().generate(n_days, 300, 200, 9)
    # mean_block=1 -> p=1 -> every day is a fresh draw, so the stationary path is
    # not merely IID-like but BIT-IDENTICAL to the IID bootstrap for the same seed
    # (both consume the same first rng.integers draw). Strongest possible form.
    np.testing.assert_array_equal(stat1, iid)


def test_block_starts_cover_all_source_days_roughly_uniformly():
    # Guard against a generator that draws starts from a restricted sub-range: over
    # a large sample every source day must appear, ~uniformly.
    n_days = 30
    for gen in (IIDDayBootstrap(), StationaryDayBootstrap(3.0)):
        paths = gen.generate(n_days, 100, 300, seed=13)
        counts = np.bincount(paths.ravel(), minlength=n_days)
        assert np.all(counts > 0)  # every source day is reachable
        # no day is wildly over/under-represented (loose uniformity band)
        expected = paths.size / n_days
        assert counts.max() < 2.0 * expected
        assert counts.min() > 0.4 * expected


# --- gather_days: day integrity, trade order, joint assets ------------------- #


def test_gather_preserves_each_days_trades_in_order():
    ds = _dataset(n_days=10, trades_per_day=3)
    day_path = np.array([2, 5, 2])  # a day may repeat
    ret, day, low = gather_days(ds, day_path)
    # exactly the three days' trades, relabelled 0,1,2
    assert ret.shape[0] == 9  # 3 days * 3 trades
    np.testing.assert_array_equal(day, [0, 0, 0, 1, 1, 1, 2, 2, 2])
    # the trades of source day 2 (returns 20,21,22) appear intact and in order
    np.testing.assert_allclose(ret[day == 0], ds.ret[ds.day_slice(2)])
    np.testing.assert_allclose(ret[day == 2], ds.ret[ds.day_slice(2)])
    np.testing.assert_allclose(ret[day == 1], ds.ret[ds.day_slice(5)])


def test_gather_matches_the_source_day_slice_exactly():
    # Day identity used for resampling IS the canonical session day from Step 3.
    ds = _dataset(n_days=8, trades_per_day=4)
    ret, day, low = gather_days(ds, np.array([3]))
    np.testing.assert_allclose(ret, ds.ret[ds.day_slice(3)])
    np.testing.assert_allclose(low, ds.trade_low[ds.day_slice(3)])


def test_gather_keeps_all_assets_of_a_day_together():
    ds = _dataset(n_days=6, trades_per_day=4, multi_asset=True)
    # each source day has both ES and NQ trades; a gathered day must carry both
    ret, day, low = gather_days(ds, np.array([0, 1]))
    for d in (0, 1):
        src = ds.symbol[ds.day_slice(d)]
        assert len(set(src.tolist())) == 2  # both symbols present in the source day
    # and the gathered day 0 has exactly the source day 0 trade count (all assets)
    assert np.sum(day == 0) == int(ds.day_count[0])


def test_no_day_is_split_across_the_boundary():
    ds = _dataset(n_days=10, trades_per_day=3)
    paths = StationaryDayBootstrap(3.0).generate(ds.n_days, 20, 1, seed=4)
    ret, day, low = gather_days(ds, paths[0])
    # every relabelled day holds a whole source day's worth of trades (3 here)
    _, counts = np.unique(day, return_counts=True)
    assert np.all(counts == 3)


# --- single source day edge case -------------------------------------------- #


def test_single_source_day_yields_all_zero_paths():
    for gen in (IIDDayBootstrap(), StationaryDayBootstrap(5.0)):
        paths = gen.generate(1, 10, 4, seed=1)
        assert np.all(paths == 0)


def test_generators_reject_bad_parameters():
    with pytest.raises(ValueError):
        IIDDayBootstrap().generate(0, 5, 1, seed=1)  # no source days
    with pytest.raises(ValueError):
        IIDDayBootstrap().generate(10, 0, 1, seed=1)  # L must be >= 1
    with pytest.raises(ValueError):
        StationaryDayBootstrap(0.5)  # mean_block must be >= 1
