"""Prepared finite-calendar experiments using the canonical account replay."""
from bisect import bisect_left
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from math import isfinite
from numbers import Integral, Real
from statistics import mean, stdev, variance
from zoneinfo import ZoneInfo

import numpy as np

from .backtest import _Replay, _check_support
from .execution import BacktestConfig, BracketHistory, DollarPolicy, LifecycleSpec, _aware
from .optimizer import CMAES
from .risk import cash_risk_path, risk_report
from .rolling import _spread


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime
    exclusion: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "start", _aware(self.start, "window start"))
        object.__setattr__(self, "end", _aware(self.end, "window end"))
        if self.end <= self.start:
            raise ValueError("window end must follow start")
        if self.exclusion is not None and (not isinstance(self.exclusion, str) or not self.exclusion.strip()):
            raise ValueError("exclusion must be a nonempty reason")


@dataclass(frozen=True)
class _PreparedWindow:
    window: Window
    history: BracketHistory | None
    sessions: tuple
    closes: tuple
    grouped: tuple
    fingerprint: str


@dataclass(frozen=True)
class PreparedWindows:
    spec: LifecycleSpec
    windows: tuple[_PreparedWindow, ...]


@dataclass(frozen=True)
class WindowOutcome:
    start: datetime
    end: datetime
    history_fingerprint: str
    net_cash: float
    net_cash_per_day: float
    calendar_days: float
    receipts: float
    fees: float
    outstanding_payouts: float
    attempts: int
    failed_attempts: int
    evaluation_passes: int
    payouts: int
    executed_trades: int
    required_bankroll: float
    max_cash_drawdown: float
    longest_underwater_days: float
    days_to_first_receipt: float | None
    observed_funding_shortfall: bool
    return_on_initial_wallet: float | None
    any_mll_breach: bool
    handoffs: int
    final_balance: float | None
    status: str
    events: tuple = ()


@dataclass(frozen=True)
class WindowBatch:
    windows: tuple[WindowOutcome, ...]
    excluded: tuple[Window, ...]
    visited_regimes: tuple[str, ...]
    policy: DollarPolicy
    risk: dict | None = None

    @property
    def summary(self):
        fields = ("net_cash", "net_cash_per_day", "calendar_days", "receipts", "fees",
                  "outstanding_payouts", "attempts", "failed_attempts", "evaluation_passes",
                  "payouts", "executed_trades", "required_bankroll", "max_cash_drawdown",
                  "longest_underwater_days", "days_to_first_receipt", "return_on_initial_wallet", "handoffs")
        distributions = {}
        for field in fields:
            values = [getattr(w, field) for w in self.windows if getattr(w, field) is not None]
            distributions[field] = (_spread(values) | {
                "count": len(values),
                "sd": stdev(values) if len(values) > 1 else None,
                "variance": variance(values) if len(values) > 1 else None,
            }) if values else None
        return {"observed_windows": len(self.windows), "excluded_windows": len(self.excluded),
                "distributions": distributions,
                "funding_shortfall_fraction": mean(w.observed_funding_shortfall for w in self.windows) if self.windows else None,
                "mll_breach_fraction": mean(w.any_mll_breach for w in self.windows) if self.windows else None,
                "receipt_fraction": mean(w.receipts > 0 for w in self.windows) if self.windows else None,
                "interpretation": "Historical window outcomes, not independent future trials; excluded windows are not zero returns."}

    @property
    def uncertainty(self):
        from .uncertainty import uncertainty_report
        return uncertainty_report({k: [getattr(w, k) for w in self.windows]
                                   for k in ("net_cash", "net_cash_per_day")},
                                  sample_kind="historical_windows")


@dataclass(frozen=True)
class WindowFit:
    policy: DollarPolicy
    in_sample: WindowBatch
    out_of_sample: WindowBatch
    in_sample_score: float
    out_of_sample_score: float
    evaluations: int
    seed: int
    direction: str

    @property
    def score(self):
        return self.out_of_sample_score

    @property
    def uncertainty(self):
        return self.out_of_sample.uncertainty


def prepare_windows(spec, history, *, sessions, closes, windows):
    """Freeze explicit windows and clocks once; never infer missing sessions."""
    if not isinstance(spec, LifecycleSpec) or (history is not None and not isinstance(history, BracketHistory)):
        raise TypeError("prepare_windows requires LifecycleSpec and a trade history or None")
    _check_support(spec)
    days, definitions = tuple(sessions), tuple(windows)
    if not days or any(type(d) is not date for d in days) or tuple(sorted(set(days))) != days:
        raise ValueError("sessions must be nonempty, unique, increasing dates")
    if not definitions or any(not isinstance(w, Window) for w in definitions):
        raise ValueError("windows must contain Window objects")
    if any(b.start <= a.start for a, b in zip(definitions, definitions[1:])):
        raise ValueError("window starts must be unique and increasing")
    zone = ZoneInfo(spec.session_timezone)
    clocks = dict(closes)
    if set(clocks) != set(days):
        raise ValueError("closes must specify exactly the declared sessions")
    openings = {}
    for day in days:
        if day.weekday() not in spec.session_weekdays:
            raise ValueError("session lies on a closed weekday")
        opening = datetime.combine(day-timedelta(days=1), spec.session_open, zone).astimezone(timezone.utc)
        normal = datetime.combine(day, spec.session_close, zone).astimezone(timezone.utc)
        close = _aware(clocks[day], "session close")
        if not opening < close <= normal:
            raise ValueError("session close must follow opening and not exceed firm cutoff")
        clocks[day], openings[day] = close, opening
    trades = history.trades if history is not None else ()
    for trade in trades:
        if trade.session not in clocks or not openings[trade.session] <= trade.entry_at < trade.exit_at <= clocks[trade.session]:
            raise ValueError("trade lies outside the declared calendar")
    entries = tuple(t.entry_at for t in trades)
    prepared = []
    for window in definitions:
        selected_days = tuple(d for d in days if openings[d] < window.end and clocks[d] >= window.start)
        if not selected_days:
            raise ValueError("window has no declared sessions")
        a, b = bisect_left(entries, window.start), bisect_left(entries, window.end)
        if not window.exclusion and ((a and trades[a-1].exit_at > window.start)
                                     or (b > a and trades[b-1].exit_at > window.end)):
            raise ValueError("window boundary cuts an open trade; supply a terminal execution or change the window")
        selected = trades[a:b] if not window.exclusion else ()
        part = type(history)(selected) if selected else None
        grouped = {d: [] for d in selected_days}
        for trade in selected:
            grouped[trade.session].append(trade)
        clock = tuple((d, clocks[d]) for d in selected_days)
        digest = sha256(repr((window, clock)).encode())
        if part is not None:
            digest.update((part.fingerprint if hasattr(part, "fingerprint")
                           else sha256(repr(part.trades).encode()).hexdigest()).encode())
        prepared.append(_PreparedWindow(window, part, selected_days, clock,
                         tuple(tuple(grouped[d]) for d in selected_days), digest.hexdigest()))
    return PreparedWindows(spec, tuple(prepared))


def _replay(prepared, part, policy, config):
    run = _Replay(prepared.spec, part.history, policy, config, sessions=part.sessions,
                  session_closes=dict(part.closes))
    run.start, run.end, run.available_at = part.window.start, part.window.end, part.window.start
    run.current_session = part.sessions[0]
    run.start_phase(run.start)
    for i, ((day, close), trades) in enumerate(zip(part.closes, part.grouped)):
        run.current_session = day
        for trade in trades:
            run.trade(trade, i)
        if run.start <= close <= run.end:
            run.close(day)
    run.advance(run.end)
    result = run.result(fingerprint=part.fingerprint)
    return replace(result, assumptions=result.assumptions + (
        "explicit calendar and finite horizon; no-trade session closes included",
        "first account purchased at window start; retries wait for the next eligible trade",
        "close clocks normalized to UTC; processing delays are elapsed durations",
    ))


def replay_windows(prepared, policy, config, *, record_events=False, risk=None):
    if not isinstance(prepared, PreparedWindows) or not isinstance(policy, DollarPolicy) or not isinstance(config, BacktestConfig):
        raise TypeError("replay_windows requires PreparedWindows, DollarPolicy and BacktestConfig")
    if type(record_events) is not bool:
        raise ValueError("record_events must be boolean")
    outcomes, excluded, visited, paths = [], [], set(), []
    for part in prepared.windows:
        if part.window.exclusion:
            excluded.append(part.window)
            continue
        result = _replay(prepared, part, policy, config)
        unrestricted = (_replay(prepared, part, policy, replace(config, initial_wallet=None))
                        if any(e.kind == "wallet_wait" for e in result.events) else None)
        capital = cash_risk_path(result, unrestricted=unrestricted)
        if risk is not None:
            paths.append(capital)
        visited.update(e.regime for e in result.events if e.regime is not None)
        outcomes.append(WindowOutcome(result.start, result.end, result.history_fingerprint,
            result.net_cash, result.net_cash_per_day, result.calendar_days, result.receipts,
            result.fees, result.outstanding_payouts, result.attempts, result.failed_attempts,
            sum(e.kind == "evaluation_pass" for e in result.events),
            sum(e.kind == "receipt" for e in result.events),
            sum(e.kind in ("trade", "liquidation") for e in result.events),
            capital.required_bankroll, capital.max_cash_drawdown, capital.longest_underwater_days,
            capital.days_to_first_receipt, capital.observed_funding_shortfall,
            capital.return_on_initial_wallet,
            any(e.kind == "failure" and e.code == "FAIL_TRAILING_DD" for e in result.events),
            sum(e.kind == "live_handoff" for e in result.events), result.final_balance, result.status,
            result.events if record_events else ()))
    return WindowBatch(tuple(outcomes), tuple(excluded), tuple(sorted(visited)), policy,
                       risk_report(paths, options=risk, sample_kind="historical_windows") if risk is not None else None)


def fit_windows(training, holdout, config, *, policy, risk_bounds, objective=None,
                direction="maximize", seed=0, generations=20, population=12,
                progress=None, record_events=False, risk=None):
    """Search only IS windows; freeze before replaying chronological holdout windows."""
    if not isinstance(training, PreparedWindows) or not isinstance(holdout, PreparedWindows):
        raise TypeError("fit_windows requires prepared training and holdout windows")
    if not isinstance(policy, DollarPolicy) or not isinstance(config, BacktestConfig):
        raise TypeError("fit_windows requires DollarPolicy and BacktestConfig")
    if type(record_events) is not bool:
        raise ValueError("record_events must be boolean")
    if training.spec != holdout.spec:
        raise ValueError("training and holdout must use the same account specification")
    if max(p.window.end for p in training.windows) > min(p.window.start for p in holdout.windows):
        raise ValueError("training windows must end before holdout starts")
    if any(not any(not p.window.exclusion for p in batch.windows) for batch in (training, holdout)):
        raise ValueError("each partition needs an observed window")
    if direction not in ("maximize", "minimize"):
        raise ValueError("direction must be maximize or minimize")
    for name, value, minimum in (("seed", seed, 0), ("generations", generations, 0), ("population", population, 2)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    names = tuple(r.name for r in policy.regimes)
    if set(risk_bounds) != set(names):
        raise ValueError("risk_bounds must specify exactly the policy's regime names")
    bounds = np.asarray([risk_bounds[n] for n in names], dtype=float)
    if bounds.shape != (len(names), 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("each risk bound must satisfy 0 <= minimum < maximum")
    initial = np.asarray([r.risk_dollars for r in policy.regimes], dtype=float)
    if np.any(initial < bounds[:, 0]) or np.any(initial > bounds[:, 1]):
        raise ValueError("initial policy must lie within risk_bounds")
    objective = objective if objective is not None else lambda batch: mean(w.net_cash_per_day for w in batch.windows)
    if not callable(objective):
        raise ValueError("objective must be callable")
    def score(batch):
        value = objective(batch)
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError("objective must return a finite real scalar")
        return float(value)
    lo, width = bounds[:, 0], bounds[:, 1]-bounds[:, 0]
    sign, cache = (1 if direction == "maximize" else -1), {}
    def evaluate(x):
        key = tuple(float(v) for v in x)
        if key not in cache:
            candidate = DollarPolicy(tuple(replace(r, risk_dollars=float(v))
                                    for r, v in zip(policy.regimes, lo+np.asarray(x)*width)))
            batch = replay_windows(training, candidate, config)
            cache[key] = (score(batch), candidate)
            if progress:
                progress({"stage": "search", "evaluations": len(cache),
                          "maximum_evaluations": generations*population+2,
                          "policy": candidate, "score": cache[key][0], "result": batch})
        return sign*cache[key][0]
    x0 = (initial-lo)/width
    evaluate(x0)
    CMAES(x0, 0.25, popsize=population, bounds=(np.zeros(len(lo)), np.ones(len(lo))),
          seed=seed, max_gen=generations).optimize(evaluate)
    best_score, best = max(cache.values(), key=lambda item: sign*item[0])
    observed = replay_windows(training, best, config, record_events=record_events, risk=risk)
    restored = DollarPolicy(tuple(r if r.name in observed.visited_regimes else original
                                 for r, original in zip(best.regimes, policy.regimes)))
    if restored != best:
        updated = replay_windows(training, restored, config, record_events=record_events, risk=risk)
        # A policy-dependent custom objective need not be invariant to restoration.
        if sign*score(updated) >= sign*best_score:
            best, observed = restored, updated
    if progress:
        progress({"stage": "holdout", "evaluations": len(cache)})
    held_out = replay_windows(holdout, best, config, record_events=record_events, risk=risk)
    return WindowFit(best, observed, held_out, score(observed), score(held_out), len(cache), seed, direction)
