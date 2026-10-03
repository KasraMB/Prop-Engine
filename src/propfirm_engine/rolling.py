"""Whole-session historical starts, without resampling or account-state carryover."""
from dataclasses import dataclass, replace
from datetime import date, datetime
from hashlib import sha256
from itertools import groupby
from math import fsum, isfinite
from numbers import Integral, Real

import numpy as np

from .backtest import backtest
from .execution import BracketHistory
from .risk import cash_risk_path, risk_report


@dataclass(frozen=True)
class RollingConfig:
    window_sessions: int = 20
    stride_sessions: int = 1

    def __post_init__(self):
        for name in ("window_sessions", "stride_sessions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def window_slices(history, rolling):
    """Return trade-index slices; each covers complete declared observed sessions."""
    if not isinstance(history, BracketHistory) or not isinstance(rolling, RollingConfig):
        raise TypeError("rolling evaluation requires BracketHistory and RollingConfig")
    offsets = [0]
    for _, trades in groupby(history.trades, lambda t: t.session):
        offsets.append(offsets[-1] + sum(1 for _ in trades))
    count = len(offsets) - 1
    if rolling.window_sessions > count:
        raise ValueError("window_sessions exceeds available sessions; no complete rolling windows")
    return tuple((offsets[i], offsets[i + rolling.window_sessions])
                 for i in range(0, count - rolling.window_sessions + 1, rolling.stride_sessions))


@dataclass(frozen=True)
class RollingWindow:
    first_session: date
    last_session: date
    start: datetime
    end: datetime
    history_fingerprint: str
    input_trades: int
    first_evaluation: str
    days_to_first_pass: float | None
    any_mll_breach: bool
    received_payout: bool
    receipts: float
    fees: float
    net_cash: float
    net_cash_per_day: float
    calendar_days: float
    outstanding_payouts: float
    max_external_cash_drawdown: float
    attempts: int
    failed_attempts: int
    executed_trades: int
    handoffs: int
    status: str
    objective_score: float


def _spread(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                minimum=float(values.min()), maximum=float(values.max()),
                p05=float(np.quantile(values, .05)), p95=float(np.quantile(values, .95)))


@dataclass(frozen=True)
class RollingResult:
    windows: tuple[RollingWindow, ...]
    rolling: RollingConfig
    excluded_incomplete_starts: int
    history_fingerprint: str
    visited_regimes: tuple[str, ...]
    risk: dict | None = None

    @property
    def score(self):
        """Equal-weight mean of the supplied per-window objective, not summed profit."""
        return fsum(w.objective_score / len(self.windows) for w in self.windows)

    @property
    def summary(self):
        counts = {name: sum(w.first_evaluation == name for w in self.windows)
                  for name in ("passed", "failed", "unresolved", "not_started", "not_applicable")}
        started = counts["passed"] + counts["failed"] + counts["unresolved"]
        return {
            "windows": len(self.windows), "first_evaluation_counts": counts,
            "first_evaluation_started": started,
            "first_evaluation_pass_rate": counts["passed"] / started if started else None,
            "payout_probability": sum(w.received_payout for w in self.windows) / len(self.windows),
            "any_mll_breach_probability": sum(w.any_mll_breach for w in self.windows) / len(self.windows),
            "days_to_first_pass_among_passes": _spread([w.days_to_first_pass for w in self.windows
                                                        if w.days_to_first_pass is not None]),
            "distributions": {field: _spread([getattr(w, field) for w in self.windows])
                              for field in ("receipts", "fees", "net_cash", "net_cash_per_day",
                                            "calendar_days", "executed_trades", "failed_attempts",
                                            "max_external_cash_drawdown")},
            "score": self.score,
            "interpretation": "Overlapping historical starts are dependent; quantiles describe window outcomes, not confidence intervals. Each window starts a fresh account and wallet.",
        }


def rolling_backtest(spec, history, policy, config, *, rolling, objective=None, risk=None):
    """Evaluate each complete window through the canonical repeated-attempt engine.

    The objective receives one BacktestResult; its equally weighted mean is score.
    First-evaluation pass rates include unresolved started evaluations in the
    denominator, but exclude accounts that could not start and direct-funded cases.
    Window ledgers are discarded after summary; replay a window to inspect its events.
    """
    slices = window_slices(history, rolling)
    objective = objective if objective is not None else lambda r: r.net_cash_per_day
    if not callable(objective):
        raise ValueError("objective must be callable")
    windows, visited, risk_paths = [], set(), []
    has_eval = any(p.role == "eval" for p in spec.account.phases)
    for begin, end in slices:
        part = BracketHistory(history.trades[begin:end])
        result = backtest(spec, part, policy, config)
        if risk is not None:
            unlimited = (backtest(spec, part, policy, replace(config, initial_wallet=None))
                         if any(e.kind == "wallet_wait" for e in result.events) else None)
            risk_paths.append(cash_risk_path(result, unrestricted=unlimited))
        score = objective(result)
        if isinstance(score, bool) or not isinstance(score, Real) or not isfinite(score):
            raise ValueError("objective must return a finite real scalar")
        visited.update(e.regime for e in result.events if e.regime is not None)
        first = [e for e in result.events if e.attempt == 1 and e.phase == "eval"]
        start = next((e for e in first if e.kind == "phase_start"), None)
        passed = next((e for e in first if e.kind == "evaluation_pass"), None)
        failed = next((e for e in first if e.kind == "failure"), None)
        outcome = ("not_applicable" if not has_eval else "not_started" if start is None
                   else "passed" if passed else "failed" if failed else "unresolved")
        cash = peak = drawdown = 0.0
        for e in result.events:
            cash += e.cash
            peak = max(peak, cash)
            drawdown = max(drawdown, peak - cash)
        windows.append(RollingWindow(
            part.sessions[0], part.sessions[-1], result.start, result.end,
            result.history_fingerprint, len(part.trades), outcome,
            (passed.at - start.at).total_seconds() / 86400 if passed else None,
            any(e.kind == "failure" and e.code == "FAIL_TRAILING_DD" for e in result.events),
            any(e.kind == "receipt" for e in result.events), result.receipts, result.fees,
            result.net_cash, result.net_cash_per_day, result.calendar_days,
            result.outstanding_payouts, drawdown, result.attempts, result.failed_attempts,
            sum(e.kind == "trade" for e in result.events),
            sum(e.kind == "live_handoff" for e in result.events), result.status, float(score),
        ))
    candidate_starts = len(range(0, len(history.sessions), rolling.stride_sessions))
    return RollingResult(tuple(windows), rolling, candidate_starts - len(slices),
                         sha256(repr(history.trades).encode("utf-8")).hexdigest(), tuple(sorted(visited)),
                         risk_report(risk_paths, options=risk) if risk is not None else None)
