"""Renewal economics — analysis layer above the engine (ARCHITECTURE §15;
BUILD_SPEC Step 11).

A single account is one **renewal cycle**, not the terminal unit of analysis. The
real process is sequential attempts: pay a fee → run an attempt → it terminates →
pay another fee → retry. The economically meaningful quantity is the **long-run
cashflow rate** of this renewal-reward process, which makes otherwise-incomparable
accounts rankable (a 20%-chance-of-\\$5k-in-5-days attempt vs a 60%-chance-of-\\$1.5k-
in-30-days attempt are only comparable as repeated capital-recycling machines).

The long-run rate consumes aggregate attempt outcomes. Finite-horizon cashflow
instead requires explicit elapsed-time fee/receipt schedules; totals alone do
not identify cash timing. Neither API implements stateful billing or affordability.

Definitions fixed (BUILD_SPEC Step 11):

* **Per-attempt reward** ``R_i`` = net payouts − fees attributable to the attempt
  (§H1); **per-attempt time** ``T_i`` = the **whole attempt's** calendar duration,
  eval + funded (§H4), from ``total_trading_days / trading_days_per_week``.
* **Profitable renewal sequence** = cumulative net payouts − cumulative fees across
  *all* attempts in the sequence > 0 (distinct from the single-attempt definition).

Two reward rates are reported **separately** (§15.2):

* ``r_renewal = E[R]/E[T]`` — the closed form (ratio of means), valid under i.i.d.
  light-tailed cycles.
* ``r_path`` — the realized finite-horizon rate from i.i.d. attempt sequences.

Their divergence measures **ratio-estimator / finite-horizon (Jensen) bias** and
*only* that: because ``r_path`` draws attempts **i.i.d.**, it removes cross-cycle
correlation by construction and therefore **cannot** diagnose it (§H5). Diagnosing
correlation needs order-preserving sequence simulation, deferred with the
generator ladder. So this layer reports both rates plus a *named, uncaptured*
correlation gap — never "the two agree, therefore cycles are independent."
"""

from __future__ import annotations

import numpy as np
from datetime import timedelta
from numbers import Integral, Real

from .cashflows import simulate_cashflow_sequences

from .statistics import attributable_fee


def per_attempt_reward(o) -> np.ndarray:
    """``R_i = net_payout − attributable_fee`` (§15). Net profit of one attempt."""
    return o.net_payout - attributable_fee(o)


def per_attempt_time(o) -> np.ndarray:
    """``T_i`` = the whole attempt's calendar duration in weeks (eval + funded, §H4),
    from ``total_trading_days / trading_days_per_week`` — never funded-only."""
    return o.total_trading_days.astype(np.float64) / o.trading_days_per_week


def r_renewal(o) -> float:
    """Closed-form reward rate ``E[R]/E[T]`` (reward per calendar week), valid under
    i.i.d. light-tailed cycles."""
    t_mean = float(np.mean(per_attempt_time(o)))
    if t_mean <= 0.0:
        return float("nan")
    return float(np.mean(per_attempt_reward(o)) / t_mean)


def _simulate_sequences(reward, time, horizon_weeks, n_sequences, seed):
    """Draw i.i.d. attempts until accumulated time reaches ``horizon_weeks``;
    return per-sequence (total_reward, total_time) for :func:`r_path`.
    This stopping-cycle diagnostic includes an overshooting final cycle; it is
    not finite-horizon cashflow and is never used by that API.

    Guards the degenerate cases that would otherwise never terminate: an empty
    batch, or one whose attempts all have zero calendar time (the accumulated time
    could never reach the horizon). These return ``nan`` — matching
    :func:`r_renewal`'s convention — rather than hanging."""
    if isinstance(horizon_weeks, bool) or not isinstance(horizon_weeks, Real) or not np.isfinite(horizon_weeks) or horizon_weeks <= 0:
        raise ValueError(f"horizon_weeks must be > 0, got {horizon_weeks}")
    if isinstance(n_sequences, bool) or not isinstance(n_sequences, Integral) or n_sequences < 1:
        raise ValueError("n_sequences must be a positive integer")
    if reward.shape != time.shape or reward.ndim != 1 or not np.all(np.isfinite(reward)) or not np.all(np.isfinite(time)) or np.any(time < 0):
        raise ValueError("reward/time must be aligned finite arrays with nonnegative durations")
    n = reward.shape[0]
    if n == 0 or not np.any(time > 0):
        return np.full(n_sequences, np.nan), np.full(n_sequences, np.nan)
    rng = np.random.default_rng(seed)
    tot_r = np.empty(n_sequences)
    tot_t = np.empty(n_sequences)
    for s in range(n_sequences):
        r = t = 0.0
        while t < horizon_weeks:
            i = rng.integers(0, n)
            r += reward[i]
            t += time[i]
        tot_r[s] = r
        tot_t[s] = t
    return tot_r, tot_t


def r_path(o, horizon_weeks: float, n_sequences: int = 2000, seed: int = 0) -> np.ndarray:
    """Empirical finite-horizon rate under **i.i.d.** attempt draws (§15.2).

    Returns the distribution of ``ΣR / ΣT`` over ``n_sequences`` sequences, each run
    until its accumulated calendar time reaches ``horizon_weeks``. Its departure
    from :func:`r_renewal` isolates ratio-estimator / finite-horizon (Jensen) bias;
    it **cannot** reveal cross-cycle correlation (i.i.d. draws remove it, §H5).
    """
    reward = per_attempt_reward(o)
    time = per_attempt_time(o)
    tot_r, tot_t = _simulate_sequences(reward, time, horizon_weeks, n_sequences, seed)
    return tot_r / tot_t


def finite_horizon_cashflow(schedules, horizon_weeks: float, n_sequences: int = 2000,
                            seed: int = 0) -> np.ndarray:
    """Actual horizon filtering over explicit elapsed-time cashflow schedules.

    Aggregate Outcomes are deliberately unsupported: payout totals and trading
    cadence cannot locate fee payments or receipts relative to a horizon.
    Supply AttemptCashflows scenarios. IID retries assume unconstrained capital;
    this is not a wallet-affordability or stateful firm-billing model.
    """
    if isinstance(horizon_weeks, bool) or not isinstance(horizon_weeks, Real) or not np.isfinite(horizon_weeks) or horizon_weeks <= 0:
        raise ValueError("horizon_weeks must be a positive finite number")
    try:
        horizon = timedelta(weeks=float(horizon_weeks))
    except OverflowError as exc:
        raise ValueError("horizon_weeks exceeds timedelta range") from exc
    return simulate_cashflow_sequences(schedules, horizon, n_sequences, seed).net_cashflow


def prob_profitable_sequence(schedules, horizon_weeks: float, n_sequences: int = 2000,
                             seed: int = 0) -> float:
    """P(a renewal sequence's cumulative reward > 0) — the sequence-level analogue
    of the single-attempt ``prob_profitable`` (one funded run can pay for many
    failed evals, so this differs from the attempt-level number)."""
    cash = finite_horizon_cashflow(schedules, horizon_weeks, n_sequences, seed)
    return float(np.mean(cash > 0.0))


def fee_bankroll_efficiency(o, fee: float | None = None, bankroll: float = 1000.0,
                            weeks_per_month: float = 4.345) -> float:
    """Expected income per calendar month per ``bankroll`` of fees — the headline
    renewal number (§15.2). ``fee`` is the **deterministic per-cycle entry cost**
    (the eval fee), *not* the path-dependent attributable fee: at the renewal level
    every cycle pays the eval fee to start, so a scalar is correct here. Doubling
    ``fee`` (all else equal) halves income-per-bankroll — fewer accounts fit a fixed
    fee bankroll.
    """
    fee = float(o.eval_fee if fee is None else fee)
    if fee <= 0.0:
        return float("nan")
    rate_per_week = r_renewal(o)  # net cashflow per calendar week
    return rate_per_week * weeks_per_month * (bankroll / fee)


def renewal_report(o, horizon_weeks: float, n_sequences: int = 2000, seed: int = 0) -> dict:
    """Report **both** reward rates and their divergence as a diagnostic (§15.2).

    The layer must not silently report only the closed form: this returns
    ``r_renewal`` (closed), the mean of the i.i.d. ``r_path`` distribution
    (empirical), and their gap — which measures ratio-estimator / finite-horizon
    (Jensen) bias *only*. It also names the **uncaptured** cross-cycle correlation
    gap (§H5): because ``r_path`` draws attempts i.i.d., neither rate can diagnose
    correlation; that awaits order-preserving sequence simulation. Reporting the
    named gap is the honest position — never "the two agree, therefore independent".
    """
    closed = r_renewal(o)
    empirical = r_path(o, horizon_weeks, n_sequences, seed)
    empirical_mean = float(np.nanmean(empirical))
    return {
        "r_renewal": closed,
        "r_path_mean": empirical_mean,
        "jensen_divergence": abs(closed - empirical_mean),
        "correlation_gap": "uncaptured — r_path is i.i.d.; needs order-preserving "
        "sequence simulation (§H5)",
    }


__all__ = [
    "per_attempt_reward",
    "per_attempt_time",
    "r_renewal",
    "r_path",
    "finite_horizon_cashflow",
    "prob_profitable_sequence",
    "fee_bankroll_efficiency",
    "renewal_report",
]
