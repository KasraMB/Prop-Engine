"""Cache random indices, not mutable trade values or per-attempt Python work."""
from dataclasses import replace

import numpy as np
import pytest

from propfirm_engine.data import preprocess
from propfirm_engine.engine import Engine, RunConfig
from propfirm_engine.feasibility import FeasibilitySpec
from propfirm_engine.model import Account, Phase
from propfirm_engine.resampling import IIDDayBootstrap, StationaryDayBootstrap
from propfirm_engine.rules import MinimumWinningDaysRule, TrailingDrawdownRule
from propfirm_engine.schema import PayoutSchema
from propfirm_engine.synthetic import IIDGenerator


def fixture():
    ds = preprocess(IIDGenerator(win_rate=0.45, rr=1.5, trades_per_day=3).generate(40, 3).rows)
    account = Account("fixture", 1000, (Phase("funded", "funded", (
        MinimumWinningDaysRule(1, 0.1),), PayoutSchema((10000.0,), 1.0, 1)),))
    config = RunConfig(intraday_mode="summary_approximation", n_paths=50, L_funded=10, size_base=10.0)
    engine = Engine()
    return engine, account, engine.prepare(account, config), ds, config


def test_cache_does_not_freeze_trade_returns():
    engine, _, prep, ds, cfg = fixture()
    cache = {}
    engine.run_prepared(prep, ds, cfg, path_cache=cache)
    ds.ret[:] = 10.0  # finite mutation is legal; next run must see it
    expected = engine.run_prepared(prep, ds, cfg)
    cached = engine.run_prepared(prep, ds, cfg, path_cache=cache)
    np.testing.assert_array_equal(cached.net_payout, expected.net_payout)


def test_mutating_stationary_parameter_changes_cache_key():
    engine, _, _, ds, _ = fixture()
    sampler, cache = StationaryDayBootstrap(2.0), {}
    engine._resample(cache, ds, sampler, ds.n_days, 20, 30, 4)
    sampler.mean_block = 20.0
    paths, _ = engine._resample(cache, ds, sampler, ds.n_days, 20, 30, 4)
    np.testing.assert_array_equal(paths, sampler.generate(ds.n_days, 20, 30, 4))


def test_cached_paths_use_fused_execution_and_are_read_only():
    engine, _, _, ds, _ = fixture()
    cache = {}
    first, material = engine._resample(cache, ds, IIDDayBootstrap(), ds.n_days, 5, 10, 3)
    second, again = engine._resample(cache, ds, IIDDayBootstrap(), ds.n_days, 5, 10, 3)
    assert material is None and again is None
    assert first is second
    assert not first.flags.writeable


def test_unknown_mutable_resampler_not_cached_by_identity():
    class Custom(IIDDayBootstrap):
        calls = 0

        def generate(self, *args):
            self.calls += 1
            return super().generate(*args)

    engine, _, _, ds, _ = fixture()
    sampler, cache = Custom(), {}
    for _ in range(2):
        engine._resample(cache, ds, sampler, ds.n_days, 5, 10, 3)
    assert sampler.calls == 2
    assert cache == {}


@pytest.mark.parametrize("feasibility", [None, FeasibilitySpec(1.0, 1.0)])
def test_cached_fused_results_and_diagnostics_equal_uncached(feasibility):
    engine, account, _, ds, cfg = fixture()
    ph = account.phases[0]
    account = replace(account, phases=(replace(ph, rules=ph.rules + (TrailingDrawdownRule(100.0),)),))
    prep = engine.prepare(account, cfg)
    direct = engine.run_prepared(prep, ds, cfg, feasibility=feasibility)
    cached = engine.run_prepared(prep, ds, cfg, feasibility=feasibility, path_cache={})
    for field in ("code", "reached_funded", "net_payout", "payouts_taken", "first_payout_day",
                  "total_trading_days", "eval_trading_days"):
        np.testing.assert_array_equal(getattr(direct, field), getattr(cached, field))
    assert direct.feas_agg == cached.feas_agg
