"""Drift-aware diffusion analytics, separate from account-rule simulation.

Fernandez (2026), supplied paper equations (3) and (5). Equation (3) is an
infinite-horizon constant-coefficient Brownian barrier model. Equation (5) is
the paper's unverified two-clock approximation, not an exact discrete solver.
No consistency, qualifying days, sizing changes, payouts or calendar is inferred.
"""
from dataclasses import dataclass
from numbers import Real
import math

import numpy as np

from .data import TradeDataset, validate_trade_dataset_numeric

BARRIER_CORRECTION = 0.5826


def _number(value, name, *, positive=False, nonnegative=False, allow_inf=False):
    try:
        valid = (not isinstance(value, (bool, np.bool_)) and isinstance(value, Real)
                 and (math.isfinite(value) or (allow_inf and value == math.inf)))
        x = float(value) if valid else math.nan
    except (OverflowError, ValueError):
        valid, x = False, math.nan
    if not valid or (positive and x <= 0) or (nonnegative and x < 0):
        raise ValueError(f"{name} has an invalid numeric value")
    return x


@dataclass(frozen=True)
class SessionMoments:
    n_days: int
    n_trades: int
    mean_trades_per_day: float
    mu: float
    sigma: float
    trade_sigma: float
    size_base: float
    trade_cost: float
    variance_basis: str = "population variance of empirical session totals (ddof=0)"
    units: str = "mu: USD/session; sigma: USD/sqrt(session)"


def estimate_session_moments(dataset: TradeDataset, *, size_base, trade_cost=0.0):
    """Fit moments of the SAME net-dollar day distribution an IID day bootstrap uses.

    Every supplied session participates, before account-dependent stopping. Cost
    is subtracted once per trade, as in RunConfig. No independent-trade variance
    assumption: sigma is measured from actual session totals. This is not a
    long-run variance estimator for correlated days or an ordered-MTM inference.
    """
    validate_trade_dataset_numeric(dataset)
    scale = _number(size_base, "size_base", positive=True)
    cost = _number(trade_cost, "trade_cost", nonnegative=True)
    if dataset.n_days < 2:
        raise ValueError("at least two sessions are required")
    with np.errstate(over="ignore", invalid="ignore"):
        net = dataset.ret.astype(np.float64) * scale - cost
        daily = np.add.reduceat(net, dataset.day_first)
        mu, sigma, trade_sigma = float(daily.mean()), float(daily.std()), float(net.std())
    if not all(math.isfinite(x) for x in (mu, sigma, trade_sigma)):
        raise ValueError("dollar moments exceed the finite numeric range")
    return SessionMoments(dataset.n_days, dataset.n_trades,
                          dataset.n_trades / dataset.n_days, mu, sigma, trade_sigma, scale, cost)


@dataclass(frozen=True)
class BarrierEstimate:
    probability: float
    nu: float
    trailing_distance: float
    frozen_distance: float
    effective_trailing_buffer: float
    effective_frozen_buffer: float
    monitoring_ratio: float
    method: str
    warnings: tuple[str, ...]
    scope: str = "infinite-horizon barrier-only constant-drift diffusion"


def _z_over_expm1(z):
    if abs(z) < 1e-5:
        return 1 - z / 2 + z * z / 12 - z ** 4 / 720
    if z > 50:
        return z * math.exp(-z) / (-math.expm1(-z))
    return z / math.expm1(z)


def _log_frozen_probability(nu, down, remaining):
    total = down + remaining
    if nu == 0:
        return math.log(down / total)
    # Small-drift expansion of the general drifted formula, not a zero-drift assumption.
    if abs(nu) * total < 1e-5:
        return (math.log(down / total) + nu * remaining / 2
                + (nu * down) ** 2 / 24 - (nu * total) ** 2 / 24)
    a = abs(nu)
    log_ratio = math.log(-math.expm1(-a * down)) - math.log(-math.expm1(-a * total))
    return log_ratio if nu > 0 else log_ratio - a * remaining


def barrier_pass_probability(*, target, drawdown, freeze_level, mu, sigma,
                             floor_updates=math.inf, breach_checks=math.inf):
    """Equation (3), optionally with equation (5)'s two monitoring clocks.

    Dollar target/drawdown/freeze are relative to opening balance. mu is net
    dollars/session and sigma dollars/sqrt(session), ALREADY at the chosen size.
    Clock frequencies are observations/session; infinity means continuous.
    freeze_level=-drawdown is static; >= target-drawdown never freezes before pass.
    Zero sigma is rejected: deterministic paths need a different treatment.
    """
    target = _number(target, "target", positive=True)
    drawdown = _number(drawdown, "drawdown", positive=True)
    freeze = _number(freeze_level, "freeze_level", allow_inf=True)
    mu = _number(mu, "mu")
    sigma = _number(sigma, "sigma", positive=True)
    fm = _number(floor_updates, "floor_updates", positive=True, allow_inf=True)
    fb = _number(breach_checks, "breach_checks", positive=True, allow_inf=True)
    if freeze < -drawdown:
        raise ValueError("freeze_level must be at least -drawdown")
    trailing = min(drawdown + freeze, target)
    remaining = max(0.0, target - drawdown - freeze)
    sm, sb = sigma / math.sqrt(fm), sigma / math.sqrt(fb)
    d1 = drawdown + BARRIER_CORRECTION * max(sm, sb)
    d2 = drawdown + BARRIER_CORRECTION * sb
    nu = 2 * (mu / sigma) / sigma
    if not all(math.isfinite(v) for v in (nu, d1, d2, nu * d1, nu * (d2 + remaining))):
        raise ValueError("scaled diffusion parameters exceed the finite numeric range")
    log_probability = -(trailing / d1) * _z_over_expm1(nu * d1)
    if remaining > 0:
        log_probability += _log_frozen_probability(nu, d2, remaining)
    probability = min(1.0, max(0.0, math.exp(min(0.0, log_probability))))
    ratio = max(sm if trailing > 0 else 0.0, sb) / drawdown
    corrected = (sm > 0 and trailing > 0) or sb > 0
    warnings = []
    if corrected:
        warnings.append("Two-clock monitoring correction is approximate; accuracy under these inputs is unverified.")
    if ratio > 0.5:
        warnings.append("Monitoring step/buffer ratio exceeds the paper's stated 0.5 range.")
    return BarrierEstimate(probability, nu, trailing, remaining, d1, d2, ratio,
                           "paper_eq3_eq5" if corrected else "paper_eq3_continuous",
                           tuple(warnings))


def expected_funding_cost(pass_probability, *, entry_fee, reset_fee, activation_fee=0.0):
    """Equation (9), per-attempt fees: IID unlimited affordable retries until pass.

    Not a finite-wallet forecast, subscription model or complete-account value.
    """
    p = _number(pass_probability, "pass_probability", nonnegative=True)
    if p > 1:
        raise ValueError("pass_probability must not exceed one")
    entry = _number(entry_fee, "entry_fee", nonnegative=True)
    reset = _number(reset_fee, "reset_fee", nonnegative=True)
    activation = _number(activation_fee, "activation_fee", nonnegative=True)
    return math.inf if p == 0 else entry + (1 / p - 1) * reset + activation


__all__ = ["SessionMoments", "BarrierEstimate", "estimate_session_moments",
           "barrier_pass_probability", "expected_funding_cost"]
