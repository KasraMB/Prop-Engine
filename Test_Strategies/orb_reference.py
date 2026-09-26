"""Standalone research strategy, outside the engine's public API.

Causal opening-range breakout signals, not an execution or profit backtest.

Default research baseline: first 15 minutes after 09:30 America/New_York,
strict close breakout, next contiguous bar open, one entry per local session,
stop at the opposite opening-range boundary. No OHLC intrabar order is inferred.
"""
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from numbers import Integral, Real
from zoneinfo import ZoneInfo
import math


@dataclass(frozen=True)
class PriceBar:
    start: datetime
    end: datetime
    open: float
    high: float
    low: float
    close: float

    def __post_init__(self):
        for at in (self.start, self.end):
            if not isinstance(at, datetime) or at.tzinfo is None or at.utcoffset() is None:
                raise ValueError("bars require timezone-aware start and end")
        if self.end.timestamp() <= self.start.timestamp():
            raise ValueError("bar end must follow its start")
        for value in (self.open, self.high, self.low, self.close):
            try:
                finite = isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("OHLC must be finite real numbers")
        if not self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high:
            raise ValueError("OHLC extrema do not enclose open and close")


@dataclass(frozen=True)
class ORBConfig:
    range_minutes: int = 15
    open_time: time = time(9, 30)
    entry_cutoff: time = time(16, 0)
    session_timezone: str = "America/New_York"

    def __post_init__(self):
        if (isinstance(self.range_minutes, bool) or not isinstance(self.range_minutes, Integral)
                or self.range_minutes < 1):
            raise ValueError("range_minutes must be a positive integer")
        if any(not isinstance(t, time) or t.tzinfo is not None
               for t in (self.open_time, self.entry_cutoff)):
            raise ValueError("session boundaries must be timezone-free local times")
        start = datetime.combine(datetime(2026, 1, 1).date(), self.open_time)
        end = datetime.combine(start.date(), self.entry_cutoff)
        if start + timedelta(minutes=int(self.range_minutes)) >= end:
            raise ValueError("opening range must finish before same-day entry cutoff")
        ZoneInfo(self.session_timezone)


@dataclass(frozen=True)
class ORBEntry:
    session: str
    side: int
    signal_at: datetime
    entry_at: datetime
    reference_entry_price: float
    stop_price: float
    range_high: float
    range_low: float

    @property
    def risk_points(self):
        return self.side * (self.reference_entry_price - self.stop_price)


@dataclass(frozen=True)
class ORBResult:
    entries: tuple[ORBEntry, ...]
    skipped: tuple[tuple[str, str], ...]
    execution_model: str = "signals_only_next_bar_open_reference"
    assumptions: tuple[str, ...] = (
        "OHLC supplies completed-bar signals, not ordered MTM or executable fills",
        "one entry per local day; strict close breakout; next contiguous bar open",
        "stop is the opposite opening-range boundary, not a constant range-width distance",
        "no inferred holiday calendar, slippage, quantities, exits or prop-account performance",
    )


def opening_range_breakout(bars, config=None):
    """Build signals causally; incomplete opening ranges and entry gaps are reported.

    Input must be a single instrument's chronological, non-overlapping bars.
    Higher-frequency ordered price evidence is still required by strict intraday
    account execution. This function cannot upgrade OHLC to that capability.
    """
    config = ORBConfig() if config is None else config
    if not isinstance(config, ORBConfig):
        raise ValueError("config must be ORBConfig")
    zone = ZoneInfo(config.session_timezone)
    entries, skipped = [], []
    previous_end = None
    day = None
    pending = None
    for bar in bars:
        if not isinstance(bar, PriceBar):
            raise ValueError("bars must be PriceBar values")
        if previous_end is not None and bar.start.timestamp() < previous_end:
            raise ValueError("bars must be chronological and non-overlapping")
        previous_end = bar.end.timestamp()
        local = bar.start.astimezone(zone)
        if local.date() != day:
            if pending is not None:
                skipped.append((str(day), "missing_entry_bar"))
            if day is not None and valid and (covered != range_end.timestamp() or high <= low):
                skipped.append((str(day), "incomplete_or_zero_width_opening_range"))
            day = local.date()
            opening = datetime.combine(day, config.open_time, zone)
            range_end = opening + timedelta(minutes=int(config.range_minutes))
            cutoff = datetime.combine(day, config.entry_cutoff, zone)
            covered = opening.timestamp()
            signal_coverage = range_end.timestamp()
            high, low = -math.inf, math.inf
            valid = True
            attempted = False
            pending = None
        start, end = bar.start.timestamp(), bar.end.timestamp()
        if pending is not None:
            side, signal_at = pending
            pending = None
            if start != signal_at.timestamp() or start >= cutoff.timestamp():
                skipped.append((str(day), "no_contiguous_entry_before_cutoff"))
            else:
                stop = low if side == 1 else high
                if side * (bar.open - stop) <= 0:
                    skipped.append((str(day), "entry_gapped_beyond_stop"))
                else:
                    entries.append(ORBEntry(str(day), side, signal_at, bar.start,
                                            float(bar.open), float(stop), high, low))
        if end <= opening.timestamp() or start >= cutoff.timestamp():
            continue
        if start < range_end.timestamp():
            if start != covered or start < opening.timestamp() or end > range_end.timestamp():
                if valid:
                    skipped.append((str(day), "incomplete_or_misaligned_opening_range"))
                valid = False
            high, low = max(high, float(bar.high)), min(low, float(bar.low))
            covered = end
            continue
        if not valid or attempted:
            continue
        if covered != range_end.timestamp() or high <= low:
            skipped.append((str(day), "incomplete_or_zero_width_opening_range"))
            valid = False
            continue
        if end > cutoff.timestamp():
            continue
        if start != signal_coverage:
            skipped.append((str(day), "missing_signal_bar"))
            valid = False
            continue
        signal_coverage = end
        side = 1 if bar.close > high else -1 if bar.close < low else 0
        if side:
            attempted = True
            pending = (side, bar.end)
    if pending is not None:
        skipped.append((str(day), "missing_entry_bar"))
    if day is not None and valid and (covered != range_end.timestamp() or high <= low):
        skipped.append((str(day), "incomplete_or_zero_width_opening_range"))
    return ORBResult(tuple(entries), tuple(skipped))
