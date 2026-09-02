"""Decision statistics over raw outcomes (ARCHITECTURE §13, §14; BUILD_SPEC Step 10).

The account is a convex structured product (§0): bounded downside (the fee),
capped path-dependent upside (the payouts). For a convex payoff **the shape of
the outcome distribution carries the value, not its mean** — two accounts with
identical ``E[payout]`` can be entirely different products. So this layer reports
the payoff distribution and decision statistics along **two axes the mean
discards**: the *distribution* of outcomes, and their *normalization by time*.
Everything here is pure post-processing over the rich raw :class:`Outcomes`
(§13) — no kernel or data-model re-entry.

Two definitions are fixed (BUILD_SPEC Step 10):

* **Profitable attempt** = cumulative net payouts − the fees attributable to *this*
  attempt (evaluation + activation) > 0. Fees are **path-dependent** (§H1): an
  attempt that fails the eval never pays the activation fee, so the attributable
  fee is ``eval_fee + activation_fee·reached_funded`` per attempt, never a scalar.
* **Calendar time of an attempt** = its trading-day count ÷ ``trading_days_per_week``
  (§11.5). Duration is measured from cadence, **never as a fraction of the source
  dataset** (a resampled path can be longer or shorter than the history).

``pass_rate`` is an **eval-phase** metric only (§H3): a funded attempt that
succeeds returns ``TIMED_OUT``/``MAXED_OUT``, never ``PASSED``, so funded economic
performance is read from ``net_payout``/``payouts_taken``/the payout-count
distribution — never from ``pass_rate``.
"""

from __future__ import annotations

import numpy as np

from .enums import ExitCode

_PASSED = int(ExitCode.PASSED)
_EPS = 1e-9


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
    """P(this attempt's net payouts exceed the fees attributable to it). The single
    most decision-relevant scalar, and provably not recoverable from ``E[payout]``."""
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
    """Per-attempt ``net_payout / attributable_fee`` — the instrument's yield,
    comparable across sizes and prices. Summarize with mean AND quantiles. The
    denominator is floored at ``_EPS`` so a zero-fee (free) account yields a large
    finite proxy rather than ``nan``/``inf`` (a fee-less account has no well-defined
    return-on-fee; §0's downside is the fee)."""
    return o.net_payout / np.maximum(attributable_fee(o), _EPS)


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
    """Convert trading days to calendar weeks via cadence (§11.5), never a dataset
    fraction."""
    return np.asarray(trading_days, dtype=np.float64) / trading_days_per_week


def payout_velocity(o, weeks_per_month: float = 4.345) -> float:
    """Expected net payout per calendar month — makes a fast Pro account and a slow
    Flex account comparable. Time comes from cadence (§11.5)."""
    if o.net_payout.size == 0:
        return float("nan")
    months = calendar_weeks(o.total_trading_days, o.trading_days_per_week) / weeks_per_month
    return float(np.mean(o.net_payout / np.maximum(months, _EPS)))


def time_to_first_payout(o) -> np.ndarray:
    """Distribution (report median + tail) of calendar weeks to the first payout,
    over attempts that took at least one. ``first_payout_day`` is attempt-relative
    (eval + funded, §H4), so this is the true capital-at-risk duration."""
    reached = o.first_payout_day[o.payouts_taken > 0]
    return calendar_weeks(reached, o.trading_days_per_week)


def return_on_fee_per_year(o) -> float:
    """Annualized yield on the attributable fee — the closest thing to the
    instrument's true rate of return (a likely optimizer objective, §15)."""
    if o.net_payout.size == 0:
        return float("nan")
    years = calendar_weeks(o.total_trading_days, o.trading_days_per_week) / 52.0
    fee = np.maximum(attributable_fee(o), _EPS)  # floor: zero-fee has no defined yield
    return float(np.mean((o.net_payout / fee) / np.maximum(years, _EPS)))


# --------------------------------------------------------------------------- #
# Pass rate + confidence intervals (§13) — eval-phase only for pass_rate       #
# --------------------------------------------------------------------------- #


def pass_rate(codes) -> float:
    """Fraction of attempts with ``code == PASSED``. An **eval-phase** metric only
    (§H3) — never a funded success measure."""
    return float(np.mean(np.asarray(codes) == _PASSED))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (e.g. a pass rate)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def bootstrap_ci(values, stat=np.mean, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for ``stat`` over ``values``."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = stat(values[rng.integers(0, n, n)])
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
