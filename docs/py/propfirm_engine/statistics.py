"""Decision statistics over closed-trade attempt outcomes.

Fees here include only configured evaluation and conditional activation charges.
Payouts are model-recorded withdrawals, not independently confirmed cash receipts.
Calendar-named compatibility helpers use a trading cadence estimate: no settlement,
holiday, billing or retry-delay calendar is represented.

Distribution summaries and user-selected objectives answer different questions;
neither convexity nor a universal optimal decision metric is assumed. Pass rate
counts PASSED terminal codes and is useful only for evaluation-phase outcomes.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

from .enums import ExitCode

_PASSED = int(ExitCode.PASSED)
def _positive_ratio(numerator, denominator) -> np.ndarray:
    """Undefined nonpositive denominators yield NaN, never a finite proxy."""
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
    )
    return np.divide(numerator, denominator,
                     out=np.full(numerator.shape, np.nan),
                     where=denominator > 0)


# --------------------------------------------------------------------------- #
# Path-dependent fees (§H1)                                                     #
# --------------------------------------------------------------------------- #


def attributable_fee(o) -> np.ndarray:
    """Per-attempt attributable fee: ``eval_fee`` always, ``activation_fee`` only
    for attempts that reached the funded phase (§H1). Returns a ``float64[B]``."""
    return o.eval_fee + o.activation_fee * o.reached_funded.astype(np.float64)


# --------------------------------------------------------------------------- #
# Distribution axis (§14.1) — because convexity lives in the shape             #
# --------------------------------------------------------------------------- #


def prob_profitable(o) -> float:
    """P(modeled net payouts exceed this attempt's configured fees)."""
    if o.net_payout.size == 0:
        return float("nan")
    return float(np.mean(o.net_payout > attributable_fee(o)))


def payout_count_dist(o) -> np.ndarray:
    """``[P(0), P(1), …, P(max_payouts)]`` — the product's natural payoff profile.
    Sums to 1. ``max_payouts`` is schema config (§H2), taken from the outcomes.
    Any (should-not-happen) count above ``max_payouts`` is folded into the last
    bin so the distribution always sums to 1, never silently dropping mass."""
    n = len(o.payouts_taken)
    if n == 0:
        return np.full(o.max_payouts + 1, np.nan)
    clipped = np.minimum(o.payouts_taken, o.max_payouts)
    counts = np.bincount(clipped, minlength=o.max_payouts + 1)
    return counts[: o.max_payouts + 1] / n


def return_on_fee(o) -> np.ndarray:
    """Payout/fee multiple per attempt (not net ROI); zero fees yield NaN.

    No epsilon substitutes for a real fee. Use a dollar-based objective when
    fees are zero; averages deliberately do not discard undefined paths.
    """
    return _positive_ratio(o.net_payout, attributable_fee(o))


def payoff_quantiles(o, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> np.ndarray:
    """Quantiles of net payoff (``net_payout − attributable_fee``) — the full
    profile, including the low quantile an optimizer may target."""
    net = o.net_payout - attributable_fee(o)
    return np.quantile(net, qs)


def mean_payout(o) -> float:
    """``E[net_payout]`` — available, but one number among the above, not the summary."""
    if o.net_payout.size == 0:
        return float("nan")
    return float(np.mean(o.net_payout))


# --------------------------------------------------------------------------- #
# Time axis (§14.2) — because the product is a rate, not a lump sum            #
# --------------------------------------------------------------------------- #


def calendar_weeks(trading_days, trading_days_per_week) -> np.ndarray:
    """Cadence-estimated weeks; legacy name, NOT actual elapsed calendar time."""
    return np.asarray(trading_days, dtype=np.float64) / trading_days_per_week


def payout_velocity(o, weeks_per_month: float = 4.345) -> float:
    """Mean per-attempt payout/month using cadence-estimated duration.

    Not a repeated-attempt long-run rate. Zero duration makes the mean undefined.
    """
    if o.net_payout.size == 0:
        return float("nan")
    months = calendar_weeks(o.total_trading_days, o.trading_days_per_week) / weeks_per_month
    return float(np.mean(_positive_ratio(o.net_payout, months)))


def time_to_first_payout(o) -> np.ndarray:
    """Cadence-estimated weeks to model payout, conditional on taking one.

    first_payout_day is a zero-based attempt-relative index; an EOD payout
    on index zero consumes one trading day. This is not cash settlement time.
    """
    reached = o.first_payout_day[o.payouts_taken > 0]
    if np.any(reached < 0):
        raise ValueError("paid attempts require nonnegative first_payout_day")
    return calendar_weeks(reached.astype(np.float64) + 1.0, o.trading_days_per_week)


def return_on_fee_per_year(o) -> float:
    """Mean annualized payout/fee multiple using cadence-estimated duration.

    Not net ROI, an IRR, or a repeated-attempt rate. Undefined paths propagate.
    """
    if o.net_payout.size == 0:
        return float("nan")
    years = calendar_weeks(o.total_trading_days, o.trading_days_per_week) / 52.0
    return float(np.mean(_positive_ratio(return_on_fee(o), years)))


# --------------------------------------------------------------------------- #
# Pass rate + confidence intervals (§13) — eval-phase only for pass_rate       #
# --------------------------------------------------------------------------- #


def pass_rate(codes) -> float:
    """Fraction of attempts with ``code == PASSED``. An **eval-phase** metric only
    (§H3) — never a funded success measure."""
    return float(np.mean(np.asarray(codes) == _PASSED))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval for IID Bernoulli trials under a fixed model/policy.

    No trials means undefined, not certainty at zero. This is not an interval
    for history uncertainty, model error, or a policy selected on these trials.
    """
    if (any(isinstance(v, bool) or not isinstance(v, Integral) for v in (k, n))
            or n < 0 or not 0 <= k <= n):
        raise ValueError("counts must be integers satisfying 0 <= k <= n")
    if isinstance(z, bool) or not isinstance(z, Real) or not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    if n == 0:
        return (float("nan"), float("nan"))
    try:
        scale = float(z) / math.sqrt(float(n))
    except OverflowError as exc:
        raise ValueError("counts exceed the numerical reporting range") from exc
    p = int(k) / int(n)
    # q=n/(n+z^2), w=z^2/(n+z^2), avoiding overflow in z^2.
    if scale <= 1:
        squared = scale * scale
        q, w = 1 / (1 + squared), squared / (1 + squared)
    else:
        inverse = (1 / scale) ** 2
        q, w = inverse / (1 + inverse), 1 / (1 + inverse)
    center = q * p + w / 2
    half = math.sqrt(q * w * p * (1 - p) + w * w / 4)
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_ci(values, stat=np.mean, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """IID percentile-bootstrap interval conditional on the supplied sample.

    Requires exchangeable scalar observations and a fixed scalar statistic.
    It does not refit a sizing policy, resample original trading history, or
    incorporate model uncertainty. Do not pool correlated folds/generator
    scenarios as independent observations. Empty samples return undefined bounds.
    """
    for name, value, minimum in (("n_boot", n_boot, 1), ("seed", seed, 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if (isinstance(alpha, bool) or not isinstance(alpha, Real)
            or not math.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError("alpha must be finite and in (0, 1)")
    if not callable(stat):
        raise ValueError("stat must be callable")
    try:
        values = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("values must be a finite one-dimensional numeric array") from exc
    if values.ndim != 1 or values.dtype.kind not in "fiu" or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite one-dimensional numeric array")
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        estimate = np.asarray(stat(values[rng.integers(0, n, n)]))
        if estimate.ndim != 0 or estimate.dtype.kind not in "fiu" or not np.isfinite(estimate):
            raise ValueError("stat must return one finite scalar")
        boot[i] = float(estimate)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return tuple(np.percentile(boot, [lo, hi]))


__all__ = [
    "attributable_fee",
    "prob_profitable",
    "payout_count_dist",
    "return_on_fee",
    "payoff_quantiles",
    "mean_payout",
    "calendar_weeks",
    "payout_velocity",
    "time_to_first_payout",
    "return_on_fee_per_year",
    "pass_rate",
    "wilson_ci",
    "bootstrap_ci",
]
