"""Trade dataset and preprocessing (ARCHITECTURE §11; BUILD_SPEC Step 3).

Two layers, exactly as §11 frames them:

* **The raw input contract (§11.1)** — one row per *closed* trade, supplied by
  the caller in any tabular form. :func:`preprocess` validates it and derives ...
* **The derived representation (:class:`TradeDataset`, §11.2)** — contiguous
  NumPy arrays the kernel reads: per-unit ``ret``, the session-``day`` index, the
  per-trade floating ``trade_low``, and a small per-day side table
  (``day_first``/``day_count``) plus the calendar cadence
  ``trading_days_per_week`` that bridges simulated trading-days to wall-clock
  time (§11.5, §14).

Design decisions worth stating loudly (each is a place a wrong guess would hide):

* **``mae`` arrives already holding-interval-clipped.** §11.1 specifies the
  ``mae`` column as "computed from bar excursions clipped to the position's
  ``[entry_time, exit_time]``" — i.e. the *producer* does the clip, and that
  producer is explicitly a separate, out-of-scope layer (MODEL_RISKS §D4).
  So :func:`preprocess` *trusts* ``mae`` and sets ``trade_low = -mae``; it does
  **not** re-derive it from bars (it has no bars). When ``mae`` is absent,
  ``trade_low`` falls back to the realized down-move ``min(ret, 0)`` — the
  documented lower-fidelity path (§11.2), never zero and never an error.
* BUILD_SPEC Step 3 nonetheless contracts a "*excursion outside the holding
  interval must not lower ``trade_low``*" test on the *derived* field. To honor
  that end-to-end (not just via a standalone helper), :func:`preprocess` also
  accepts an **optional in-pipeline clip**: a row may carry ``entry_time`` plus
  ``mae_bars`` (per-bar adverse-excursion samples), and preprocessing clips them
  to ``[entry_time, timestamp]`` via :func:`clip_mae_to_holding_interval` to
  derive that row's ``mae``. This keeps the clip a genuinely *tested* part of the
  pipeline while the producer path (a pre-clipped scalar ``mae``) still works —
  both satisfy §11.1, and neither pulls raw 1-minute bars into engine scope.
* **The session boundary is a parameter, never a hardcoded midnight** (§11.3):
  a trade at/after the reset time belongs to the *next* trading day even on the
  same calendar date. ``reset`` defaults to 17:00 (the common futures reset) but
  is caller-set; ``reset="00:00"`` recovers a midnight split as one option.

Naive timestamps retain the caller-normalized local-time contract. Aware inputs
require an explicit IANA session_timezone: sorting uses absolute UTC instants,
session grouping uses local wall time, and ambiguous localization is not guessed.
This module keeps to NumPy + stdlib (no pandas).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np


# --------------------------------------------------------------------------- #
# Derived representation (§11.2)                                                #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TradeDataset:
    """Simulation-ready arrays derived once from raw rows and reused everywhere.

    Per-trade arrays are length ``N`` in trade (timestamp) order; the per-day
    side table is length ``n_days`` and is indexed by the values in ``day``.
    ``symbol`` is retained (as small integer codes) for multi-asset inspection
    and for resampling's joint-day integrity (§11.4); the kernel itself does not
    need it, since ``day`` already groups every asset's trades of one session
    day together.
    """

    # --- per-trade arrays, length N (trade order) ---
    ret: np.ndarray  # float64[N] — per-unit return, after fees
    day: np.ndarray  # int32[N]   — 0-based trading-day index, monotonic non-decreasing
    trade_low: np.ndarray  # float64[N] — per-unit floating low of THIS trade (≤ 0)
    symbol: np.ndarray  # int32[N]   — instrument code (0 for a single-asset account)
    # --- per-day side table, length n_days; index by TradeDataset.day ---
    day_first: np.ndarray  # int32[n_days] — index of first trade of each day
    day_count: np.ndarray  # int32[n_days] — number of trades in each day
    n_days: int
    # --- calendar cadence (§11.5): the only bridge to wall-clock time ---
    trading_days_per_week: float
    # --- provenance: instrument names in code order; parameters used ---
    symbol_names: tuple[str, ...]
    session_reset: str
    session_timezone: str | None = None  # None: caller-normalized naive local times
    # Observed input clocks, NOT a simulated cashflow calendar. Aware inputs use
    # UTC; naive inputs retain the declared caller-local clock. Missing entry
    # observations are NaT. None permits legacy manually constructed datasets.
    exit_timestamps: np.ndarray | None = None  # datetime64[ns], aligned with trades
    entry_timestamps: np.ndarray | None = None  # datetime64[ns], aligned with trades
    cadence_source: str = "supplied"  # "derived" from observed span, or an explicit caller assumption

    @property
    def n_trades(self) -> int:
        return int(self.ret.shape[0])

    def day_slice(self, d: int) -> slice:
        """Half-open trade-index slice for trading day ``d`` (§11.2 O(1) lookup)."""
        first = int(self.day_first[d])
        return slice(first, first + int(self.day_count[d]))


class InvalidTradeDataError(ValueError):
    """Raised for invalid raw trade data or simulation-ready numeric inputs."""


def _finite_number(value, name: str) -> float:
    """Convert one numeric value, preserving a useful field/row error context."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidTradeDataError(f"{name} must be a finite number") from exc
    if not np.isfinite(number):
        raise InvalidTradeDataError(f"{name} must be a finite number")
    return number


def _assert_finite(values: np.ndarray, name: str) -> None:
    bad = np.flatnonzero(~np.isfinite(values))
    if bad.size:
        raise InvalidTradeDataError(f"{name} must be finite (row {int(bad[0]) + 1})")


def validate_trade_dataset_numeric(dataset: TradeDataset) -> None:
    """Check numeric arrays at use, including manually built or mutated datasets.

    Checks numeric and day/symbol layout safety, not calendars or path
    sufficiency. Frozen dataclasses still contain mutable arrays, so the engine
    calls this before each run rather than trusting preprocessing/cache history.
    """
    ret, low = dataset.ret, dataset.trade_low
    for name, values in (("ret", ret), ("trade_low", low)):
        if not isinstance(values, np.ndarray) or values.ndim != 1:
            raise InvalidTradeDataError(f"{name} must be a one-dimensional NumPy array")
        if values.dtype.kind not in "fiu":
            raise InvalidTradeDataError(f"{name} must contain real numeric values")
        _assert_finite(values, name)
    if ret.size == 0 or ret.shape != low.shape:
        raise InvalidTradeDataError("ret and trade_low must have equal, nonzero lengths")
    if np.any(low > 0.0):
        raise InvalidTradeDataError("trade_low must be non-positive")
    cadence = dataset.trading_days_per_week
    if isinstance(cadence, (bool, np.bool_)) or not isinstance(cadence, (int, float, np.integer, np.floating)):
        raise InvalidTradeDataError("trading_days_per_week must be a finite positive number")
    if _finite_number(cadence, "trading_days_per_week") <= 0.0:
        raise InvalidTradeDataError("trading_days_per_week must be positive")
    n_days = dataset.n_days
    if isinstance(n_days, (bool, np.bool_)) or not isinstance(n_days, (int, np.integer)) or n_days < 1:
        raise InvalidTradeDataError("n_days must be a positive integer")
    for name, length in (("day", ret.size), ("symbol", ret.size),
                         ("day_first", n_days), ("day_count", n_days)):
        values = getattr(dataset, name)
        if (not isinstance(values, np.ndarray) or values.ndim != 1
                or values.size != length or values.dtype.kind not in "iu"):
            raise InvalidTradeDataError(f"{name} must be a one-dimensional integer array of length {length}")
    counts = dataset.day_count
    # Python integer summation cannot wrap on malicious uint64 counts.
    if np.any(counts < 1) or sum(int(v) for v in counts) != ret.size:
        raise InvalidTradeDataError("day_count must partition every trade into nonempty days")
    expected_first = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))
    if not np.array_equal(dataset.day_first, expected_first):
        raise InvalidTradeDataError("day_first must describe contiguous, non-overlapping day slices")
    expected_day = np.repeat(np.arange(n_days), counts.astype(np.int64))
    if not np.array_equal(dataset.day, expected_day):
        raise InvalidTradeDataError("day must match the chronological day slices")
    if not dataset.symbol_names or np.any(dataset.symbol < 0) or np.any(dataset.symbol >= len(dataset.symbol_names)):
        raise InvalidTradeDataError("symbol codes must index nonempty symbol_names")
    _validate_trade_clocks(dataset)
    if dataset.cadence_source not in ("supplied", "derived"):
        raise InvalidTradeDataError("cadence_source must be supplied or derived")
    if dataset.cadence_source == "derived" and dataset.exit_timestamps is None:
        raise InvalidTradeDataError("derived cadence requires observed exit_timestamps")


def _validate_trade_clocks(dataset):
    """Reject observed overlap; missing holding intervals remain unknown."""
    exits, entries = dataset.exit_timestamps, dataset.entry_timestamps
    for name, values in (("exit_timestamps", exits), ("entry_timestamps", entries)):
        if values is not None and (not isinstance(values, np.ndarray)
                or values.shape != dataset.ret.shape or values.dtype != np.dtype("datetime64[ns]")):
            raise InvalidTradeDataError(f"{name} must be an aligned datetime64[ns] array")
    if exits is None:
        if entries is not None:
            raise InvalidTradeDataError("entry_timestamps require exit_timestamps")
        return
    if np.any(np.isnat(exits)) or np.any(exits[1:] < exits[:-1]):
        raise InvalidTradeDataError("exit_timestamps must be present and chronologically ordered")
    if entries is None:
        return
    known = ~np.isnat(entries)
    if np.any(entries[known] > exits[known]):
        raise InvalidTradeDataError("entry_timestamps cannot follow exit_timestamps")
    known_entries, known_exits = entries[known], exits[known]
    if np.any(known_entries[1:] < known_exits[:-1]):
        raise InvalidTradeDataError(
            "known holding intervals overlap; the closed-trade summary engine "
            "requires sequential positions, not concurrent-position MTM")


# --------------------------------------------------------------------------- #
# Producer-boundary helper (MODEL_RISKS §D1/§D4)                                #
# --------------------------------------------------------------------------- #


def clip_mae_to_holding_interval(
    entry_time: datetime,
    exit_time: datetime,
    bar_excursions: Sequence[tuple[datetime, float]],
) -> float:
    """Clip a trade's adverse excursion to its ``[entry_time, exit_time]`` window.

    **This is producer-side logic (MODEL_RISKS §D4), exposed here only so
    BUILD_SPEC Step 3's clipping contract is testable at the engine boundary.**
    Given the per-bar adverse-excursion magnitudes a trade spanned, return the
    worst (largest) magnitude that occurred *while the position was actually
    open* — a bar whose timestamp is before ``entry_time`` or after ``exit_time``
    is not exposure and must not count (§D1). Magnitudes are per-unit and
    non-negative (a loss depth); the result is the ``mae`` to feed
    :func:`preprocess`. Returns ``0.0`` if no bar falls inside the interval.
    """
    if exit_time < entry_time:
        raise ValueError("exit_time precedes entry_time")
    worst = 0.0
    for index, (bar_time, magnitude) in enumerate(bar_excursions):
        magnitude = _finite_number(magnitude, f"bar excursion {index + 1}")
        if magnitude < 0:
            raise InvalidTradeDataError("bar excursion magnitude must be non-negative")
        if entry_time <= bar_time <= exit_time:  # inside the holding interval
            if magnitude > worst:
                worst = magnitude
    return worst


# --------------------------------------------------------------------------- #
# Input normalization                                                           #
# --------------------------------------------------------------------------- #


def _as_columns(rows) -> dict[str, list]:
    """Accept either a sequence of row-mappings or a mapping of columns; return
    a column-oriented dict. Keeps the public API forgiving without pandas."""
    if isinstance(rows, Mapping):
        cols = {k: list(v) for k, v in rows.items()}
        lengths = {len(v) for v in cols.values()}
        if len(lengths) > 1:
            raise InvalidTradeDataError(
                f"column-oriented input has ragged lengths: "
                f"{ {k: len(v) for k, v in cols.items()} }"
            )
        return cols
    # sequence of mappings
    row_list = list(rows)
    if not row_list:
        raise InvalidTradeDataError("no trades supplied")
    keys: list[str] = []
    for r in row_list:
        if not isinstance(r, Mapping):
            raise InvalidTradeDataError("row-oriented input rows must be mappings")
        for k in r:
            if k not in keys:
                keys.append(k)
    return {k: [r.get(k) for r in row_list] for k in keys}


def _to_datetime64(values) -> np.ndarray:
    """Coerce a column of datetimes / ISO strings / datetime64 to datetime64[ns]."""
    out = np.empty(len(values), dtype="datetime64[ns]")
    for i, v in enumerate(values):
        if v is None:
            raise InvalidTradeDataError("timestamp is missing on at least one row")
        try:
            if isinstance(v, (np.datetime64, datetime, date, str)):
                raw = np.datetime64(v)
                out[i] = raw
                if not np.isnat(raw) and out[i].astype("datetime64[us]") != raw.astype("datetime64[us]"):
                    raise InvalidTradeDataError(f"timestamp outside supported datetime64[ns] range on row {i + 1}")
            else:
                raise InvalidTradeDataError(f"unparseable timestamp on row {i + 1}")
        except InvalidTradeDataError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidTradeDataError(f"unparseable timestamp on row {i + 1}") from exc
        if np.isnat(out[i]):
            raise InvalidTradeDataError(f"timestamp is NaT on row {i + 1}")
    return out


def _normalize_times(values, session_timezone):
    """Return absolute ordering timestamps and local session-wall timestamps.

    A timezone must be explicitly supplied for aware inputs. Naive inputs keep
    the legacy caller-normalized contract; we never guess their timezone/fold.
    """
    parsed = []
    for value in values:
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                pass  # the canonical parser below supplies row-specific errors
        parsed.append(value)
    aware = [isinstance(value, datetime) and value.utcoffset() is not None for value in parsed]
    if not any(aware):
        if session_timezone is not None:
            raise InvalidTradeDataError("session_timezone requires timezone-aware timestamps; do not silently localize naive times")
        times = _to_datetime64(values)
        return times, times
    if not all(aware):
        raise InvalidTradeDataError("cannot mix timezone-aware and naive timestamps")
    if session_timezone is None:
        raise InvalidTradeDataError("timezone-aware input requires an explicit session_timezone")
    try:
        zone = ZoneInfo(session_timezone)
    except (ZoneInfoNotFoundError, ValueError, TypeError) as exc:
        raise InvalidTradeDataError("invalid/unavailable session_timezone; supply an IANA timezone with timezone data installed") from exc
    utc = []
    local = []
    for raw, value in zip(values, parsed):
        round_trip = value.astimezone(timezone.utc).astimezone(value.tzinfo)
        if round_trip.replace(tzinfo=None) != value.replace(tzinfo=None):
            raise InvalidTradeDataError("nonexistent local timestamp; provide a valid timezone-aware instant")
        # Preserve nanosecond ISO fractions while removing only the parsed offset.
        naive = (re.sub(r"(?:Z|[+-]\d{2}(?::?\d{2})?(?::?\d{2}(?:[.,]\d+)?)?)$", "", raw)
                 if isinstance(raw, str) else value.replace(tzinfo=None))
        base = _to_datetime64([naive])[0]
        offset = np.timedelta64(int(value.utcoffset().total_seconds() * 1_000_000), "us")
        absolute = base - offset
        expected = _to_datetime64([value.astimezone(timezone.utc).replace(tzinfo=None)])[0]
        if absolute.astype("datetime64[us]") != expected.astype("datetime64[us]"):
            raise InvalidTradeDataError("timestamp outside supported datetime64[ns] range after timezone conversion")
        utc.append(absolute)
        local.append(value.astimezone(zone).replace(tzinfo=None))
    return np.asarray(utc, dtype="datetime64[ns]"), _to_datetime64(local)


def _parse_reset(reset: str | time) -> time:
    if isinstance(reset, time):
        return reset
    hh, _, mm = reset.partition(":")
    return time(int(hh), int(mm) if mm else 0)


def _floats(values, name: str) -> np.ndarray:
    """Coerce a required numeric column, rejecting missing/non-finite cells."""
    out = np.empty(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        out[i] = _finite_number(v, f"'{name}' on row {i + 1}")
    return out


def _derive_trade_low(n: int, ret: np.ndarray, ts_raw: list, cols: dict) -> np.ndarray:
    """Per-row floating low (§11.2). See :func:`preprocess` for the priority order.

    ``ret`` here is in the original (pre-sort) row order, matching ``ts_raw`` and
    the ``cols`` columns; :func:`preprocess` reorders the result alongside the
    other per-trade arrays.
    """
    mae_col = cols.get("mae")
    entry_col = cols.get("entry_time")
    bars_col = cols.get("mae_bars")
    tl = np.empty(n, dtype=np.float64)
    for i in range(n):
        bars_i = bars_col[i] if bars_col is not None else None
        if bars_i:  # non-empty per-row [(time, magnitude), ...]: clip in-pipeline
            entry_i = entry_col[i] if entry_col is not None else None
            if entry_i is None:
                raise InvalidTradeDataError(
                    "a row provides 'mae_bars' but no 'entry_time' to clip against "
                    "(the holding interval is [entry_time, timestamp], §D1)"
                )
            mae_i = clip_mae_to_holding_interval(entry_i, ts_raw[i], bars_i)
            tl[i] = -mae_i
            continue
        mv = mae_col[i] if mae_col is not None else None
        if mv is not None:  # scalar mae, already holding-interval-clipped (§11.1/§D4)
            m = _finite_number(mv, f"'mae' on row {i + 1}")
            if m < 0:
                raise InvalidTradeDataError(
                    "'mae' is a loss magnitude and must be non-negative "
                    "(the per-unit worst adverse excursion, §11.1)"
                )
            tl[i] = -m
        else:  # documented fallback: realized down-move (§11.2), never zero-by-default
            tl[i] = min(float(ret[i]), 0.0)
    return tl


# --------------------------------------------------------------------------- #
# The pipeline (§11.5)                                                          #
# --------------------------------------------------------------------------- #


def preprocess(
    rows,
    *,
    session_reset: str | time = "17:00",
    trading_days_per_week: float | None = None,
    session_timezone: str | None = None,
) -> TradeDataset:
    """Convert raw closed-trade rows into a :class:`TradeDataset` (§11.5).

    ``rows`` is either a sequence of per-trade mappings or a mapping of
    column-name → sequence. Required: ``timestamp`` and exactly one of ``return``
    or (``pnl`` and ``size``). Optional: ``mae`` (already holding-interval-clipped,
    §11.1), ``symbol`` (required only for multi-asset accounts).

    Numeric inputs must be finite. ``size`` is a positive normalization
    magnitude, not signed direction; fractional scales remain allowed here.
    An absent/None MAE uses the documented fallback; explicit NaN/inf is invalid.
    Per-bar magnitudes are checked even outside the holding interval. The
    existing priority remains bars, then scalar MAE, then realized fallback.

    Aware timestamps require ``session_timezone`` (IANA name). UTC determines
    chronological ordering, local time determines session membership. Naive times
    must already be caller-normalized and cannot be combined with that parameter.
    This is session assignment, not a holiday calendar or elapsed-time ledger.

    ``session_reset`` is the daily session boundary (§11.3); a trade at/after it
    rolls into the next trading day. ``trading_days_per_week`` may be supplied to
    override the value derived from the calendar span (useful for tiny/synthetic
    inputs); otherwise it is ``distinct trading days ÷ calendar-week span`` (§11.5).
    """
    cols = _as_columns(rows)
    if "timestamp" not in cols:
        raise InvalidTradeDataError("required column 'timestamp' is missing")

    n = len(cols["timestamp"])
    if n == 0:
        raise InvalidTradeDataError("no trades supplied")

    # --- per-unit return: EXACTLY one of `return` or (`pnl` + `size`) (§11.1) ---
    has_return = "return" in cols and any(v is not None for v in cols["return"])
    has_pnl = "pnl" in cols and any(v is not None for v in cols["pnl"])
    has_size = "size" in cols and any(v is not None for v in cols["size"])
    if not has_return and not (has_pnl and has_size):
        raise InvalidTradeDataError(
            "input must provide either a 'return' column or both 'pnl' and 'size' "
            "(to normalize dollar P&L to per-unit return, §11.1)"
        )
    if has_return and (has_pnl or has_size):
        raise InvalidTradeDataError(
            "input provides BOTH 'return' and 'pnl'/'size' — exactly one must be "
            "present (§11.1) so the per-unit return has a single unambiguous source"
        )
    if has_return:
        ret = _floats(cols["return"], "return")
    else:
        pnl = _floats(cols["pnl"], "pnl")
        size = _floats(cols.get("size", [None] * n), "size")
        if np.any(size <= 0.0):
            raise InvalidTradeDataError("'size' must be a positive normalization magnitude")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            ret = pnl / size
        _assert_finite(ret, "normalized return (pnl / size)")

    # --- timestamps: keep the raw objects (for the holding-interval clip) and a
    #     parallel datetime64 view (for ordering + session-day assignment) ---
    ts_raw = list(cols["timestamp"])
    ts, local_ts = _normalize_times(ts_raw, session_timezone)
    # Normalize all clipping operands to the same absolute clock. This also
    # supports ISO-string entry/bar times and avoids ambiguous DST wall ordering.
    cols = dict(cols)
    if "entry_time" in cols:
        cols["entry_time"] = [None if v is None else _normalize_times([v], session_timezone)[0][0]
                              for v in cols["entry_time"]]
        for i, entry in enumerate(cols["entry_time"]):
            if entry is not None and entry > ts[i]:
                raise InvalidTradeDataError("entry_time follows exit timestamp")
    if "mae_bars" in cols:
        cols["mae_bars"] = [None if not bars else [
            (_normalize_times([when], session_timezone)[0][0], magnitude) for when, magnitude in bars]
            for bars in cols["mae_bars"]]

    # --- trade_low, decided PER ROW (§11.2, §D1) with priority:
    #       1. bar excursions clipped to [entry_time, timestamp]  — in-pipeline clip
    #          (only when a row carries 'mae_bars'; keeps the clip a *tested* part of
    #          the derived field rather than trusting an off-pipeline value);
    #       2. a scalar 'mae', already holding-interval-clipped upstream (§11.1/§D4);
    #       3. the realized down-move min(ret,0) — the documented fallback (§11.2).
    #     Choosing per row means a partly-populated 'mae' column does NOT zero the
    #     un-annotated rows; each such row independently takes the fallback. ---
    trade_low = _derive_trade_low(n, ret, ts, cols)

    # --- symbol codes (single synthetic instrument when absent) ---
    if "symbol" in cols and any(v is not None for v in cols["symbol"]):
        raw_syms = [str(s) if s is not None else "" for s in cols["symbol"]]
    else:
        raw_syms = ["_"] * n
    symbol_names = tuple(sorted(set(raw_syms)))
    sym_code = {name: i for i, name in enumerate(symbol_names)}
    symbol = np.array([sym_code[s] for s in raw_syms], dtype=np.int32)

    # --- stable timestamp order (multi-asset ties broken by symbol code) ---
    order = np.lexsort((symbol, ts))
    ts = ts[order]
    local_ts = local_ts[order]
    ret = ret[order]
    symbol = symbol[order]
    trade_low = trade_low[order]

    # --- session-day index (§11.3): reset-shifted calendar date, ranked 0-based ---
    reset = _parse_reset(session_reset)
    day = _assign_session_days(local_ts, reset)
    if np.any(np.diff(day) < 0):
        raise InvalidTradeDataError("session_reset crosses an ambiguous DST boundary; choose an unambiguous session boundary")

    # --- per-day side table (§11.2) ---
    n_days = int(day[-1]) + 1
    # day is monotonic non-decreasing in [0, n_days); counts are a bincount and the
    # first-index offsets are the exclusive prefix sum of those counts (identical to
    # the former two Python passes, vectorized).
    day_count = np.bincount(day, minlength=n_days).astype(np.int32)
    day_first = np.zeros(n_days, dtype=np.int32)
    if n_days > 1:
        day_first[1:] = np.cumsum(day_count)[:-1].astype(np.int32)

    # --- calendar cadence (§11.5) ---
    cadence_source = "derived" if trading_days_per_week is None else "supplied"
    if trading_days_per_week is None:
        trading_days_per_week = _derive_cadence(ts, n_days)
    trading_days_per_week = _finite_number(trading_days_per_week, "trading_days_per_week")

    dataset = TradeDataset(
        ret=ret,
        day=day,
        trade_low=trade_low,
        symbol=symbol,
        day_first=day_first,
        day_count=day_count,
        n_days=n_days,
        trading_days_per_week=float(trading_days_per_week),
        symbol_names=symbol_names,
        session_reset=reset.strftime("%H:%M"),
        session_timezone=session_timezone,
        exit_timestamps=ts.copy(),
        entry_timestamps=np.array(cols.get("entry_time", [None] * n),
                                  dtype="datetime64[ns]")[order],
        cadence_source=cadence_source,
    )
    validate_trade_dataset_numeric(dataset)
    return dataset


def slice_days(ds: TradeDataset, d0: int, d1: int) -> TradeDataset:
    """A contiguous session-day range ``[d0, d1)`` of ``ds`` as its own dataset, with
    day indices re-based to 0 (§11.2). Days are chronological, so this is a *time* slice
    — the primitive behind rolling time-separated train/test folds. Every day index in
    the source carries ≥1 trade, so the derived side table stays dense.
    """
    if not (0 <= d0 < d1 <= ds.n_days):
        raise ValueError(f"slice_days: bad range [{d0}, {d1}) for n_days={ds.n_days}")
    validate_trade_dataset_numeric(ds)
    mask = (ds.day >= d0) & (ds.day < d1)
    day = (ds.day[mask] - d0).astype(np.int32)
    n_days = int(d1 - d0)
    day_count = np.bincount(day, minlength=n_days).astype(np.int32)
    day_first = np.zeros(n_days, dtype=np.int32)
    day_first[1:] = np.cumsum(day_count)[:-1]
    # Only caller-supplied cadence is shared. A derived fold estimate must not
    # use future held-out timestamps from the parent dataset.
    cadence = (_derive_cadence(ds.exit_timestamps[mask], n_days)
               if ds.cadence_source == "derived" else ds.trading_days_per_week)
    return replace(ds, ret=ds.ret[mask], day=day, trade_low=ds.trade_low[mask],
                   symbol=ds.symbol[mask], day_first=day_first, day_count=day_count,
                   n_days=n_days,
                   trading_days_per_week=cadence,
                   exit_timestamps=None if ds.exit_timestamps is None else ds.exit_timestamps[mask],
                   entry_timestamps=None if ds.entry_timestamps is None else ds.entry_timestamps[mask])


def _assign_session_days(ts: np.ndarray, reset: time) -> np.ndarray:
    """0-based session-day index for each (already sorted) timestamp (§11.3).

    A trade whose time-of-day is at/after ``reset`` belongs to the *next*
    calendar session; the resulting session dates are then ranked to a dense
    0-based index that is monotonic non-decreasing in timestamp order.
    """
    reset_delta = np.timedelta64(
        reset.hour * 3600 + reset.minute * 60 + reset.second, "s"
    )
    one_day = np.timedelta64(1, "D")
    # shift so the reset boundary lands on midnight, then floor to the day:
    #   time >= reset  ->  (ts - reset) is on/after midnight of the SAME date,
    #                       and adding one day pushes it to the next session date.
    # Equivalent, branch-free: session_date = floor_day(ts - reset) + 1 day.
    shifted = (ts - reset_delta).astype("datetime64[D]") + one_day
    # rank distinct session dates to a dense 0-based index (monotonic since ts sorted)
    _, inverse = np.unique(shifted, return_inverse=True)
    return inverse.astype(np.int32).reshape(-1)


def _derive_cadence(ts: np.ndarray, n_days: int) -> float:
    """``distinct trading days ÷ calendar-week span`` (§11.5).

    The span is measured between the first and last trade's calendar dates. A
    span shorter than a week is floored to one week so a single-week (or single-
    day) input yields a finite, sensible cadence rather than dividing by zero.
    """
    first = ts[0].astype("datetime64[D]")
    last = ts[-1].astype("datetime64[D]")
    span_days = int((last - first) / np.timedelta64(1, "D"))
    weeks = max(span_days, 7) / 7.0
    return n_days / weeks


__all__ = [
    "TradeDataset",
    "InvalidTradeDataError",
    "preprocess",
    "clip_mae_to_holding_interval",
    "validate_trade_dataset_numeric",
]
