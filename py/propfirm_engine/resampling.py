"""Resampling — whole-day bootstrap path generators (ARCHITECTURE §11.4;
BUILD_SPEC Step 8; MODEL_RISKS §G1, §C7).

An attempt is simulated over a *resampled path*: a sequence of **whole trading
days** drawn from the source dataset, each day carried intact with its trades in
their original intra-day order. Because a "day" in :class:`TradeDataset` is
already every asset's trades on one canonical session day (§11.4, fixed in Step
3), resampling *day indices* keeps all assets of a day together for free — the
resamplers here never see assets, so they cannot manufacture cross-asset
alignment (§G1).

Two generators, both operating at day granularity:

* :class:`IIDDayBootstrap` — each day of the path is an independent uniform draw.
  The optimistic baseline: it destroys inter-day dependence (§B1).
* :class:`StationaryDayBootstrap` — the Politis–Romano stationary bootstrap:
  geometric-length blocks of *consecutive* source days (circular), with mean
  block length a first-class modeling parameter (§G1). ``mean_block = 1`` recovers
  the i.i.d. case exactly (every day starts a new block), so IID is the
  short-block limit.

**Path length ``L`` is an explicit input, never inferred** (§C7): it is a
load-bearing modeling parameter, and eval and funded phases may use different
``L`` (they draw independent paths). A generated path is exactly ``L`` days long
and references only real source days; :func:`gather_days` materializes a path
into the ``(ret, day, trade_low)`` arrays the kernel consumes, relabelling days
``0..L-1`` in path order while preserving each day's trades.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numba import njit


class DayResampler(ABC):
    """Common interface: produce ``int32[n_paths, L]`` of source-day indices."""

    @abstractmethod
    def generate(self, n_days: int, path_length: int, n_paths: int, seed: int) -> np.ndarray:
        """Return an ``int32[n_paths, path_length]`` array of source-day indices in
        ``[0, n_days)``. Deterministic under ``seed``."""
        raise NotImplementedError

    @staticmethod
    def _check(n_days, path_length, n_paths):
        if n_days < 1:
            raise ValueError(f"n_days must be >= 1, got {n_days}")
        if path_length < 1:
            raise ValueError(f"path_length (L) must be >= 1, got {path_length}")
        if n_paths < 1:
            raise ValueError(f"n_paths must be >= 1, got {n_paths}")


class IIDDayBootstrap(DayResampler):
    """Each day of each path is an independent uniform draw from the source days."""

    def generate(self, n_days, path_length, n_paths, seed):
        self._check(n_days, path_length, n_paths)
        rng = np.random.default_rng(seed)
        return rng.integers(0, n_days, size=(n_paths, path_length), dtype=np.int64).astype(
            np.int32
        )


class StationaryDayBootstrap(DayResampler):
    """Stationary (geometric-block) bootstrap over consecutive source days (§G1).

    With ``p = 1 / mean_block``: each step, with probability ``p`` a new block
    begins at a fresh uniform day, otherwise the block continues to the next
    source day (circularly, ``(d + 1) mod n_days``). Block lengths are geometric
    with mean ``mean_block``. ``mean_block = 1`` gives ``p = 1`` — every day starts
    a new block — i.e. exactly the i.i.d. bootstrap.
    """

    def __init__(self, mean_block: float):
        if mean_block < 1.0:
            raise ValueError(f"mean_block must be >= 1, got {mean_block}")
        self.mean_block = float(mean_block)

    def generate(self, n_days, path_length, n_paths, seed):
        self._check(n_days, path_length, n_paths)
        rng = np.random.default_rng(seed)
        p = 1.0 / self.mean_block
        # Pre-draw the randomness vectorized, then thread the sequential dependence
        # in a compiled loop (identical output to the former Python double loop, so
        # the seed→stream mapping is byte-for-byte unchanged; the RNG draws stay in
        # numpy so the stream itself does not move).
        starts = rng.integers(0, n_days, size=(n_paths, path_length), dtype=np.int64)
        new_block = rng.random(size=(n_paths, path_length)) < p
        return _thread_stationary(starts, new_block, int(n_days))


@njit(cache=True)
def _thread_stationary(starts, new_block, n_days):  # pragma: no cover - jitted
    """Thread geometric-block dependence: each step continues the block (``d+1`` mod
    ``n_days``) unless ``new_block`` opens a fresh block at ``starts``. Compiled, but
    numerically identical to the reference Python loop."""
    n_paths, path_length = starts.shape
    out = np.empty((n_paths, path_length), dtype=np.int32)
    for b in range(n_paths):
        d = starts[b, 0]  # the first day always starts a block
        out[b, 0] = np.int32(d)
        for i in range(1, path_length):
            if new_block[b, i]:
                d = starts[b, i]
            else:
                d = (d + 1) % n_days
            out[b, i] = np.int32(d)
    return out


@njit(cache=True)
def _gather_core(day_path, ds_ret, ds_trade_low, day_first, day_count):  # pragma: no cover - jitted
    total = 0
    for k in range(day_path.shape[0]):
        total += day_count[day_path[k]]
    ret = np.empty(total, dtype=np.float64)
    trade_low = np.empty(total, dtype=np.float64)
    day = np.empty(total, dtype=np.int32)
    pos = 0
    for new_d in range(day_path.shape[0]):
        src_d = day_path[new_d]
        start = day_first[src_d]
        cnt = day_count[src_d]
        for j in range(cnt):
            ret[pos] = ds_ret[start + j]
            trade_low[pos] = ds_trade_low[start + j]
            day[pos] = new_d
            pos += 1
    return ret, day, trade_low


def gather_days(dataset, day_path: np.ndarray):
    """Materialize one resampled path into ``(ret, day, trade_low)`` (§11.4).

    ``day_path`` is a 1-D array of source-day indices. Each day's trades are
    concatenated in path order, in their original intra-day order (no day split,
    no trade orphaned), with the ``day`` array relabelled ``0..L-1`` so the kernel
    sees a fresh day sequence. All of a source day's trades — every asset — come
    together, since a day index gathers that day's whole slice. The copy runs in a
    compiled loop (identical output to the former slice-copy Python loop).
    """
    day_path = np.ascontiguousarray(day_path, dtype=np.int64)
    return _gather_core(day_path, dataset.ret, dataset.trade_low,
                        dataset.day_first, dataset.day_count)


def materialize(dataset, day_paths: np.ndarray) -> list:
    """Materialize every path row of ``day_paths`` into its ``(ret, day, trade_low)``
    arrays (a list, one entry per attempt). This is **policy-independent** — a path's
    materialization is identical regardless of the sizing policy — so an optimizer
    sweeping policies over fixed resampled paths computes it once and reuses it
    (§18), instead of re-gathering per candidate."""
    day_paths = np.asarray(day_paths)
    return [gather_days(dataset, day_paths[i]) for i in range(day_paths.shape[0])]


__all__ = [
    "DayResampler",
    "IIDDayBootstrap",
    "StationaryDayBootstrap",
    "gather_days",
    "materialize",
]
