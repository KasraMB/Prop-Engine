"""Batch Monte Carlo driver (ARCHITECTURE §12, §17; BUILD_SPEC Step 9).

Runs many resampled attempts of one compiled phase through the single-path
kernel and collects rich per-attempt raw outcomes. Batching is purely a
memory-throughput device (§13): the resampled day-paths are generated once
(deterministic under the seed), so results do not depend on the batch size —
that is only how many attempts are materialized and run at a time.

For Step 9 the per-attempt work is a Python loop over
:func:`propfirm_engine.kernels.simulate_one_phase` (correctness first; the doc's
``prange``/``@njit`` batch is a deferred throughput optimization, §18/§19 — the
loop already agrees with the single-path oracle by construction). The stationary
resampler's inner loop and the per-path :func:`gather_days` are the throughput
hot spots to vectorize later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kernels import simulate_one_phase
from .resampling import gather_days


@dataclass
class PhaseBatchResult:
    """Per-attempt raw outcomes for one phase over a batch of resampled paths.

    Arrays are length ``B`` (the number of attempts). The time axis (payout day
    indices via ``first_payout_day``, and ``total_trading_days``) is retained, not
    pre-aggregated away (§13).
    """

    code: np.ndarray  # int32[B] — terminal ExitCode
    payouts_taken: np.ndarray  # int32[B]
    net_payout: np.ndarray  # float64[B] — sum of the attempt's net payouts
    first_payout_day: np.ndarray  # int32[B] — day-index of the first payout, -1 if none
    total_trading_days: np.ndarray  # int32[B] — trading days the attempt ran
    feas_agg: object = None  # a FeasibilityAgg (§16.9) when feasibility was active


def simulate_phase_batch(
    cp_phase,
    dataset,
    day_paths: np.ndarray,
    size_base: float,
    policy_params: np.ndarray,
    start_equity: float,
    batch_size: int | None = None,
    feasibility=None,
    materials=None,
    trade_cost: float = 0.0,
) -> PhaseBatchResult:
    """Run ``day_paths.shape[0]`` attempts of ``cp_phase`` over the resampled paths.

    ``day_paths`` is ``int[B, L]`` source-day indices (from a resampler). Each row
    is materialized with :func:`gather_days` and run through the kernel; the
    outcome equals the single-path kernel on that path by construction (batch ⇄
    oracle agreement). ``batch_size`` bounds how many attempts are materialized at
    a time — a pure memory knob: attempts are independent, so the result is
    identical for any chunking (§13).
    """
    day_paths = np.asarray(day_paths)
    b = day_paths.shape[0]

    # Fast path: when materialization is NOT precomputed, run the whole batch inside
    # the fused compiled kernel (no per-attempt Python dispatch or allocation, §13).
    # The materials path (optimizer's cross-candidate cache, §18) keeps the Python
    # loop, since a jitted core can't take a Python list of variable-length arrays.
    if materials is None:
        from .kernels import simulate_batch

        oc, optk, onet, ofpd, odays, agg_i, agg_f = simulate_batch(
            cp_phase, dataset, day_paths, size_base, policy_params, start_equity,
            feasibility=feasibility, want_diag=feasibility is not None,
            trade_cost=trade_cost,
        )
        feas_agg = None
        if feasibility is not None:
            from .feasibility import FeasibilityAgg

            feas_agg = FeasibilityAgg(
                attempts=int(agg_i[0]), nontradable_failures=int(agg_i[1]),
                actual_breach_failures=int(agg_i[2]),
                _sum_frac_reduced=float(agg_f[0]), _sum_frac_constrained=float(agg_f[1]),
                _sum_avg_requested=float(agg_f[2]), _sum_avg_executed=float(agg_f[3]),
            )
        return PhaseBatchResult(
            code=oc.astype(np.int32),
            payouts_taken=optk.astype(np.int32),
            net_payout=onet,
            first_payout_day=ofpd.astype(np.int32),
            total_trading_days=odays.astype(np.int32),
            feas_agg=feas_agg,
        )

    code = np.empty(b, dtype=np.int32)
    payouts_taken = np.zeros(b, dtype=np.int32)
    net_payout = np.zeros(b, dtype=np.float64)
    first_payout_day = np.full(b, -1, dtype=np.int32)
    total_days = np.zeros(b, dtype=np.int32)

    policy = np.asarray(policy_params, dtype=np.float64)
    # Feasibility diagnostics are aggregated on the fly only when the projection is
    # active (§16.9); inactive -> no diag objects allocated, zero overhead.
    agg = None
    if feasibility is not None:
        from .feasibility import FeasibilityAgg, FeasibilityDiag

        agg = FeasibilityAgg()
    step = b if not batch_size or batch_size < 1 else int(batch_size)
    for chunk_start in range(0, b, step):
        for i in range(chunk_start, min(chunk_start + step, b)):
            # `materials` (when supplied) is the policy-INDEPENDENT (ret, day,
            # trade_low) materialization of each path, precomputed once and reused
            # across sizing-policy candidates (§18); identical to gathering here.
            if materials is not None:
                ret, day, trade_low = materials[i]
            else:
                ret, day, trade_low = gather_days(dataset, day_paths[i])
            diag = FeasibilityDiag() if feasibility is not None else None
            c, amounts, days, ndays = simulate_one_phase(
                cp_phase, ret, day, trade_low, size_base, policy, start_equity,
                feasibility=feasibility, diag_out=diag, trade_cost=trade_cost,
            )
            code[i] = c
            total_days[i] = ndays
            if amounts:
                payouts_taken[i] = len(amounts)
                net_payout[i] = float(sum(amounts))
                first_payout_day[i] = days[0]
            if agg is not None:
                agg.add(diag)

    return PhaseBatchResult(
        code=code,
        payouts_taken=payouts_taken,
        net_payout=net_payout,
        first_payout_day=first_payout_day,
        total_trading_days=total_days,
        feas_agg=agg,
    )


__all__ = ["PhaseBatchResult", "simulate_phase_batch"]
