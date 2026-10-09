"""Explicit scenario distributions for simulated execution, not venue calibration.

All quantities are adverse ticks beyond the raw-price execution reference.
The model includes spread/latency uncertainty in that increment; do not add a
second spread charge. Observed price gaps remain the price adapter's concern.
"""
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from hashlib import sha256
from math import ceil, isfinite
from zoneinfo import ZoneInfo

import numpy as np


@dataclass(frozen=True)
class TickDistribution:
    """Probability at each integer tick from zero through len(probabilities)-1.

    A finite categorical tail is deliberately inspectable. Its upper endpoint
    is a scenario assumption, not an assertion that real slippage is bounded.
    """
    probabilities: tuple[float, ...]

    def __post_init__(self):
        p = tuple(float(v) for v in self.probabilities)
        if not p or any(not isfinite(v) or v < 0 for v in p) or abs(sum(p) - 1) > 1e-10:
            raise ValueError("tick probabilities must be nonnegative, finite and sum to one")
        object.__setattr__(self, "probabilities", p)

    def quantile(self, probability):
        if not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("quantile probability must be in [0,1]")
        cumulative = 0.
        last = 0
        for ticks, weight in enumerate(self.probabilities):
            if weight:
                last = ticks
                cumulative += weight
                if probability <= cumulative + 1e-14:
                    return ticks
        return last

    def draw(self, uniform):
        if not 0 <= uniform < 1:
            raise ValueError("uniform must be in [0,1)")
        cumulative = 0.
        for ticks, weight in enumerate(self.probabilities):
            cumulative += weight
            if uniform < cumulative:
                return ticks
        return max(i for i, p in enumerate(self.probabilities) if p)


@dataclass(frozen=True)
class SlippageModel:
    """Immutable assumptions for one instrument/platform scenario.

    Callers supply separate models per instrument. Fit reference_range_ticks
    on training data only; no inference of a platform's fills is performed.
    The latent stress state is shared by fills in a trading session. Bracket
    allowances use its mixture distribution, never the latent realized state.
    """
    market: TickDistribution
    stop: TickDistribution
    stressed_market: TickDistribution
    stressed_stop: TickDistribution
    reference_range_ticks: float
    stress_probability: float = .05
    allowance_quantile: float = .95
    volatility_window: int = 20
    max_multiplier: float = 8.
    hour_multipliers: tuple[float, ...] = (1.,) * 24
    timezone: str = "America/New_York"
    target_trade_through_ticks: int = 0
    label: str = "uncalibrated execution scenario"

    def __post_init__(self):
        if any(not isinstance(getattr(self, k), TickDistribution)
               for k in ("market", "stop", "stressed_market", "stressed_stop")):
            raise TypeError("execution distributions must be TickDistribution")
        if not isfinite(self.reference_range_ticks) or self.reference_range_ticks <= 0:
            raise ValueError("reference_range_ticks must be positive and finite")
        for k in ("stress_probability", "allowance_quantile"):
            if not isfinite(getattr(self, k)) or not 0 <= getattr(self, k) <= 1:
                raise ValueError(f"{k} must be in [0,1]")
        if type(self.volatility_window) is not int or self.volatility_window < 1:
            raise ValueError("volatility_window must be a positive integer")
        if not isfinite(self.max_multiplier) or self.max_multiplier < 1:
            raise ValueError("max_multiplier must be finite and >= 1")
        hours = tuple(self.hour_multipliers)
        if len(hours) != 24 or any(not isfinite(v) or v <= 0 for v in hours):
            raise ValueError("hour_multipliers requires 24 positive finite values")
        object.__setattr__(self, "hour_multipliers", hours)
        if type(self.target_trade_through_ticks) is not int or self.target_trade_through_ticks < 0:
            raise ValueError("target_trade_through_ticks must be a nonnegative integer")
        ZoneInfo(self.timezone)
        if not self.label:
            raise ValueError("scenario label is required")

    def allowance(self, multiplier):
        size = max(len(self.stop.probabilities), len(self.stressed_stop.probabilities))
        weights = np.zeros(size)
        for weight, dist in ((1-self.stress_probability, self.stop),
                             (self.stress_probability, self.stressed_stop)):
            weights[:len(dist.probabilities)] += weight * np.asarray(dist.probabilities)
        return ceil(TickDistribution(tuple(weights)).quantile(self.allowance_quantile) * multiplier)


@dataclass(frozen=True)
class ExecutionTape:
    """Read-only causal multipliers and exogenous uniforms for one session."""
    multipliers: np.ndarray
    uniforms: np.ndarray
    stressed: bool

    def ticks(self, model, index, channel):
        name = "stop" if channel == 1 else "market"
        distribution = getattr(model, ("stressed_" if self.stressed else "") + name)
        return ceil(distribution.draw(float(self.uniforms[index, channel])) * self.multipliers[index])


def prepare_execution(sessions, instrument, model, *, seed=0, path=0):
    """Precompute policy-independent execution inputs once, reusable in search.

    Each draw is keyed by instrument, trading date, absolute minute and order
    channel, independent of skipped trades, bracket choices and entry timing.
    Available prior bars seed volatility; otherwise multiplier starts at one.
    No full-sample volatility normalizer or current-bar range enters a decision.
    """
    if not isinstance(model, SlippageModel):
        raise TypeError("model must be SlippageModel")
    for value in (seed, path):
        if type(value) is not int or value < 0:
            raise ValueError("execution seed/path must be nonnegative integers")
    key = int.from_bytes(sha256(instrument.symbol.encode()).digest()[:4], "little")
    zone = ZoneInfo(model.timezone)
    carry = np.array([], dtype=float)
    tapes = []
    for session in sessions:
        prior = carry
        if session.warmup_ohlc is not None:
            prior = (session.warmup_ohlc[:, 1] - session.warmup_ohlc[:, 2]) / instrument.tick_size
        prior = prior[-model.volatility_window:]
        ranges = (session.ohlc[:, 1] - session.ohlc[:, 2]) / instrument.tick_size
        combined = np.r_[prior, ranges]
        cumulative = np.r_[0., np.cumsum(combined)]
        end = np.arange(len(ranges)) + len(prior)
        start = np.maximum(0, end - model.volatility_window)
        count = end - start
        recent = np.divide(cumulative[end] - cumulative[start], count,
                           out=np.full(len(ranges), model.reference_range_ticks, dtype=float), where=count > 0)
        if model.timezone in ("UTC", "America/New_York", "America/Chicago"):
            utc_hours, inverse = np.unique(session.timestamps // 3_600_000_000_000, return_inverse=True)
            local_hours = np.fromiter((datetime.fromtimestamp(int(h)*3600, timezone.utc).astimezone(zone).hour
                                       for h in utc_hours), dtype=int, count=len(utc_hours))
            hours = local_hours[inverse]
        else:
            # Fractional offsets and non-hour DST changes can cross a local-hour
            # boundary within one UTC hour. Do not reuse an incorrect hour bucket.
            hours = np.fromiter((datetime.fromtimestamp(int(t)/1e9, timezone.utc).astimezone(zone).hour
                                 for t in session.timestamps), dtype=int, count=len(ranges))
        multipliers = np.minimum(model.max_multiplier,
                                np.maximum(1., recent / model.reference_range_ticks) * np.asarray(model.hour_multipliers)[hours])
        rng = np.random.default_rng(np.random.SeedSequence([seed, path, key, session.session.toordinal()]))
        stressed = bool(rng.random() < model.stress_probability)
        # Full two-UTC-day tape gives identical draws to morning/evening suffixes.
        start_at = datetime.combine(session.session - timedelta(days=1), time(), timezone.utc)
        offsets = (session.timestamps // 60_000_000_000 - int(start_at.timestamp() // 60)).astype(int)
        if np.any(offsets < 0) or np.any(offsets >= 2880):
            raise ValueError("price session lies outside its two-day execution tape")
        uniforms = rng.random((2880, 3))[offsets]
        multipliers.flags.writeable = uniforms.flags.writeable = False
        tapes.append(ExecutionTape(multipliers, uniforms, stressed))
        carry = combined[-model.volatility_window:]
    return tuple(tapes)
