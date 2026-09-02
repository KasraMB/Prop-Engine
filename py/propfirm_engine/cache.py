"""The three caches — reuse across the test matrix (ARCHITECTURE §10, §18;
BUILD_SPEC Step 7).

Testing many firms re-touches the same artifacts constantly, so three caches
turn repeated work into lookups:

* :class:`CompiledAccountCache` — keyed on the structural :func:`fingerprint`
  (version + fees + payout schema inside the hash, §10), so identical configs
  compile once and a size-specific quirk gets its own entry.
* :class:`TradeCache` — keyed on a content hash of the raw input **plus the
  session-boundary parameter** (§11.5), so the same trades preprocess once.
* :class:`CompiledRuleCache` — keyed on the (frozen, value-hashable) rule object,
  so a rule shared across sizes compiles once.

Each cache exposes ``hits``/``misses`` so a caller (and the Step 9 "no recompute"
test) can confirm a second lookup did not recompute. Backends are plain dicts;
the contract is stability + correct hit/miss behavior, not a particular store.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from .compiler import compile_account
from .data import preprocess
from .fingerprint import fingerprint


@dataclass
class _Stats:
    hits: int = 0
    misses: int = 0


def _dataset_key(rows, session_reset) -> str:
    """A stable content hash of the raw trade input plus the session parameter.

    Handles both input shapes (a mapping of columns, or a sequence of row
    mappings) by canonicalizing to sorted columns of ``repr``-ed cells. O(N) in
    the input — computed once per distinct dataset, which is the point.
    """
    if isinstance(rows, Mapping):
        cols = {str(k): list(v) for k, v in rows.items()}
    else:
        row_list = list(rows)
        keys: list[str] = []
        for r in row_list:
            for k in r:
                if k not in keys:
                    keys.append(k)
        cols = {k: [r.get(k) for r in row_list] for k in keys}
    h = hashlib.sha256()
    h.update(repr(session_reset).encode())
    for name in sorted(cols):
        h.update(b"\x00")
        h.update(name.encode())
        _hash_column(h, cols[name])
    return h.hexdigest()


def _hash_column(h, values) -> None:
    """Hash one column's content. Fast path: coerce to a homogeneous numpy array
    and hash its raw bytes (+ dtype/shape) — far cheaper than ``repr``-ing every
    cell. Falls back to per-cell ``repr`` only for genuinely heterogeneous/object
    columns. Deterministic and content-distinguishing either way (the properties
    the cache correctness rests on): the same values hash the same, different
    values hash differently, and the two input shapes canonicalize to the same
    per-column list so they agree."""
    arr = values if isinstance(values, np.ndarray) else None
    if arr is None:
        try:
            cand = np.asarray(values)
            if cand.dtype != object:
                arr = cand
        except Exception:  # pragma: no cover - exotic inputs fall back below
            arr = None
    if arr is not None:
        h.update(b"A")
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    else:
        for cell in values:
            h.update(b"\x01")
            h.update(repr(cell).encode())


class TradeCache:
    """Preprocessed-trades cache keyed on raw input + session boundary (§11.5)."""

    def __init__(self) -> None:
        self._store: dict[str, object] = {}
        self.stats = _Stats()

    def get(self, rows, *, session_reset="17:00", trading_days_per_week=None):
        key = _dataset_key(rows, (session_reset, trading_days_per_week))
        cached = self._store.get(key)
        if cached is not None:
            self.stats.hits += 1
            return cached
        self.stats.misses += 1
        ds = preprocess(rows, session_reset=session_reset,
                        trading_days_per_week=trading_days_per_week)
        self._store[key] = ds
        return ds

    @property
    def hits(self) -> int:
        return self.stats.hits

    @property
    def misses(self) -> int:
        return self.stats.misses


class CompiledAccountCache:
    """Compiled-account cache keyed on the structural fingerprint (§10)."""

    def __init__(self) -> None:
        self._store: dict[str, object] = {}
        self.stats = _Stats()

    def get(self, account, program_version: str = "v1", *, key: str | None = None):
        key = key if key is not None else fingerprint(account, program_version)
        cached = self._store.get(key)
        if cached is not None:
            self.stats.hits += 1
            return cached
        self.stats.misses += 1
        compiled = compile_account(account)
        self._store[key] = compiled
        return compiled

    def fingerprint_of(self, account, program_version: str = "v1") -> str:
        return fingerprint(account, program_version)

    @property
    def hits(self) -> int:
        return self.stats.hits

    @property
    def misses(self) -> int:
        return self.stats.misses


class CompiledRuleCache:
    """Compiled-rule cache keyed on the frozen rule object itself (§10)."""

    def __init__(self) -> None:
        self._store: dict[object, object] = {}
        self.stats = _Stats()

    def get(self, rule):
        cached = self._store.get(rule)
        if cached is not None:
            self.stats.hits += 1
            return cached
        self.stats.misses += 1
        compiled = rule.compile()
        self._store[rule] = compiled
        return compiled

    @property
    def hits(self) -> int:
        return self.stats.hits

    @property
    def misses(self) -> int:
        return self.stats.misses


@dataclass
class Caches:
    """The three caches bundled, so an engine threads one object through a run."""

    trades: TradeCache = field(default_factory=TradeCache)
    accounts: CompiledAccountCache = field(default_factory=CompiledAccountCache)
    rules: CompiledRuleCache = field(default_factory=CompiledRuleCache)


__all__ = ["TradeCache", "CompiledAccountCache", "CompiledRuleCache", "Caches"]
