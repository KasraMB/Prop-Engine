"""Reproducible local microbenchmark; timings are workload-specific, not CI gates.

Run from the repo: python benchmarks/path_execution.py
Compares warmed fused gathering vs pre-materialized Python dispatch, excluding
materialization cost (deliberately favorable to the old cache). Checks exact
outcomes and feasibility diagnostics before timing. No downloads or file writes.
"""
from pathlib import Path
import statistics
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from propfirm_engine.compiler import compile_phase
from propfirm_engine.data import preprocess
from propfirm_engine.enums import StateField
from propfirm_engine.feasibility import FeasibilitySpec
from propfirm_engine.model import Phase
from propfirm_engine.resampling import StationaryDayBootstrap, materialize
from propfirm_engine.rules import MinimumWinningDaysRule, TrailingDrawdownRule
from propfirm_engine.schema import PayoutSchema
from propfirm_engine.simulate import simulate_phase_batch
from propfirm_engine.synthetic import IIDGenerator


def main():
    dataset = preprocess(IIDGenerator(win_rate=0.45, rr=1.5, trades_per_day=6).generate(250, 2026).rows)
    phase = compile_phase(Phase("synthetic", "funded", (
        TrailingDrawdownRule(2000.0), MinimumWinningDaysRule(3, 100.0)),
        PayoutSchema((1000.0,), 0.9, 5, cap_fraction=0.5,
                     reset_fields=(StateField.N_QUALIFYING_DAYS,))))
    paths = StationaryDayBootstrap(5.0).generate(dataset.n_days, 120, 2000, 22)
    materials = materialize(dataset, paths)
    args = (phase, dataset, paths, 100.0, np.ones(5), 50000.0)
    for feasibility in (None, FeasibilitySpec(1.0, 1.0)):
        calls = {
            "fused": lambda: simulate_phase_batch(*args, feasibility=feasibility),
            "materialized": lambda: simulate_phase_batch(*args, feasibility=feasibility, materials=materials),
        }
        results = [call() for call in calls.values()]  # warm JIT before measurements
        for name in ("code", "payouts_taken", "net_payout", "first_payout_day", "total_trading_days"):
            np.testing.assert_array_equal(getattr(results[0], name), getattr(results[1], name))
        assert results[0].feas_agg == results[1].feas_agg
        timings = {}
        for name, call in calls.items():
            elapsed = []
            for _ in range(5):
                start = perf_counter()
                call()
                elapsed.append(perf_counter() - start)
            timings[name] = statistics.median(elapsed)
        print(f"feasibility={feasibility is not None}: {timings}; "
              f"materialized/fused={timings['materialized']/timings['fused']:.2f}x")


if __name__ == "__main__":
    main()
