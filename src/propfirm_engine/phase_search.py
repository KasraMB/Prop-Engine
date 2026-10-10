"""Budgeted single-process phase search with chronological lifecycle selection."""
from dataclasses import dataclass, replace
from collections import OrderedDict
from functools import lru_cache
from math import fsum, isfinite
from numbers import Integral, Real
from time import perf_counter

import numpy as np

from .backtest import _Replay, _check_support, backtest
from .execution import BacktestConfig, BracketHistory, DollarPolicy, LifecycleSpec
from .fitting import HoldoutFit
from .optimizer import CMAES
from .risk import RiskConfig, cash_risk_path, risk_report
from .rolling import RollingConfig, window_slices


@dataclass(frozen=True)
class PhaseSearch:
    trade_budget: int = 200_000
    horizon_sessions: int = 20
    stride_sessions: int = 20
    archive_size: int = 12
    population: int = 8

    def __post_init__(self):
        for name, minimum in (("trade_budget", 1), ("horizon_sessions", 1),
                              ("stride_sessions", 1), ("archive_size", 4), ("population", 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class PhaseSample:
    outcome: str
    duration_days: float
    receipts: float
    fees: float
    outstanding: float


@dataclass(frozen=True)
class PhaseCandidate:
    policy: DollarPolicy
    metrics: dict
    samples: tuple[PhaseSample, ...]


@dataclass(frozen=True)
class SearchWork:
    budget: int
    search_trade_visits: int
    report_trade_visits: int
    phase_evaluations: int
    lifecycle_evaluations: int
    search_seconds: float
    report_seconds: float

    @property
    def unused_budget(self):
        return self.budget - self.search_trade_visits


@dataclass(frozen=True)
class PhaseFit:
    fit: HoldoutFit
    architecture: str
    settings: PhaseSearch
    evaluation_candidates: tuple[PhaseCandidate, ...]
    funded_candidates: tuple[PhaseCandidate, ...]
    work: SearchWork
    in_sample_risk: dict
    out_of_sample_risk: dict

    @property
    def policy(self):
        return self.fit.policy

    @property
    def score(self):
        return self.fit.score

    @property
    def in_sample_score(self):
        return self.fit.in_sample_score

    @property
    def out_of_sample_score(self):
        return self.fit.out_of_sample_score

    @property
    def train_sessions(self):
        return self.fit.train_sessions

    @property
    def test_sessions(self):
        return self.fit.test_sessions

    @property
    def in_sample(self):
        return self.fit.in_sample

    @property
    def out_of_sample(self):
        return self.fit.out_of_sample

    @property
    def uncertainty(self):
        return self.fit.uncertainty


class _Attempt(_Replay):
    def result(self):
        return super().result(fingerprint="phase-screen")

    def restart(self, at, fee):
        super().restart(at, fee)
        self.handoff = True
        self.status = "PHASE_ENDED"


def _phase_sample(spec, history, policy, config):
    result = _Attempt(spec, history, policy, config).run()
    terminal = next((e for e in result.events if e.kind in
                     ("evaluation_pass", "failure", "live_handoff")), None)
    outcome = ({"evaluation_pass": "passed", "failure": "failed", "live_handoff": "handoff"}
               [terminal.kind] if terminal else "unresolved")
    duration = ((terminal.at if terminal else result.end) - result.start).total_seconds() / 86400
    return PhaseSample(outcome, duration, result.receipts, result.fees, result.outstanding_payouts)


def _metrics(samples):
    n = len(samples)
    passed = sum(s.outcome == "passed" for s in samples)
    days = fsum(s.duration_days for s in samples)
    fees = fsum(s.fees for s in samples)
    cash = np.asarray([s.receipts - s.fees for s in samples])
    return dict(pass_probability=passed/n, passes_per_day=passed/days,
                cost_per_observed_pass=fees/passed if passed else None,
                mean_duration_days=days/n, mean_net_payout=float(cash.mean()),
                payout_variance=float(cash.var()),
                failure_probability=sum(s.outcome == "failed" for s in samples)/n,
                unresolved_fraction=sum(s.outcome == "unresolved" for s in samples)/n,
                mean_outstanding=fsum(s.outstanding for s in samples)/n)


def _axes(candidate, phase):
    m = candidate.metrics
    if phase == "eval":
        cost = m["cost_per_observed_pass"]
        return (m["passes_per_day"], m["pass_probability"],
                -cost if cost is not None else -float("inf"), -m["mean_duration_days"])
    return (m["mean_net_payout"], -m["mean_duration_days"],
            -m["payout_variance"], -m["failure_probability"])


def _retain(archive, candidate, phase, limit):
    values = _axes(candidate, phase)
    for old in archive:
        axes = _axes(old, phase)
        if axes == values or all(a >= b for a, b in zip(axes, values)):
            return archive
    archive = [old for old in archive if not all(a >= b for a, b in zip(values, _axes(old, phase)))]
    archive.append(candidate)
    if len(archive) <= limit:
        return archive
    # Normalized crowding preserves boundary trade-offs without an unbounded archive.
    distance = np.zeros(len(archive))
    for axis in range(4):
        order = sorted(range(len(archive)), key=lambda i: _axes(archive[i], phase)[axis])
        low, high = (_axes(archive[order[i]], phase)[axis] for i in (0, -1))
        if high == low:
            continue
        distance[order[0]] = distance[order[-1]] = float("inf")
        for j in range(1, len(order)-1):
            before, after = (_axes(archive[order[k]], phase)[axis] for k in (j-1, j+1))
            distance[order[j]] += (after-before)/(high-low) if isfinite(high-low) else 0
    remove = min(range(len(archive)), key=lambda i: (distance[i], -i))
    return [c for i, c in enumerate(archive) if i != remove]


class _Limit(Exception):
    pass


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError("objective must return a finite real scalar")
    return float(value)


def fit_phases(spec, history, config, *, policy, risk_bounds, search=PhaseSearch(),
               architecture="separate", objective=None, direction="maximize",
               train_fraction=.7, seed=0, constraint=None, risk=RiskConfig()):
    """Search only IS; score the frozen complete policy on chronological OOS.

    Phase screens use unlimited wallets and fresh accounts on common IS windows.
    Final pair scores use the original configuration, dates and lifecycle.
    constraint(result) is an IS-only boolean eligibility predicate.
    The trade budget includes phase screens and pair selection, not final reports.
    """
    if not isinstance(search, PhaseSearch):
        raise TypeError("search must be PhaseSearch")
    if not isinstance(spec, LifecycleSpec) or not isinstance(history, BracketHistory):
        raise TypeError("fit_phases requires LifecycleSpec and BracketHistory")
    if not isinstance(policy, DollarPolicy) or not isinstance(config, BacktestConfig):
        raise TypeError("fit_phases requires DollarPolicy and BacktestConfig")
    if architecture not in ("separate", "joint") or direction not in ("maximize", "minimize"):
        raise ValueError("select separate/joint architecture and maximize/minimize direction")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    objective = objective if objective is not None else lambda r: r.net_cash_per_day
    if not callable(objective) or constraint is not None and not callable(constraint):
        raise TypeError("objective and constraint must be callable")
    if not isinstance(risk, RiskConfig):
        raise TypeError("risk must be RiskConfig")
    _check_support(spec)
    if tuple(p.role for p in spec.account.phases) != ("eval", "funded"):
        raise ValueError("phase search requires one evaluation followed by one funded phase")
    names = tuple(r.name for r in policy.regimes)
    if set(risk_bounds) != set(names):
        raise ValueError("risk_bounds must specify exactly the policy's regime names")
    bounds = np.asarray([risk_bounds[n] for n in names], dtype=float)
    if (bounds.shape != (len(names), 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] <= bounds[:, 0])):
        raise ValueError("each risk bound must satisfy 0 <= minimum < maximum")
    lo, width = bounds[:, 0], bounds[:, 1]-bounds[:, 0]
    initial = np.asarray([r.risk_dollars for r in policy.regimes])
    if np.any(initial < lo) or np.any(initial > bounds[:, 1]):
        raise ValueError("initial policy must lie within risk_bounds")
    train, test = history.split(train_fraction)
    full_cost = len(train.trades)
    if search.trade_budget < full_cost:
        raise ValueError("trade_budget cannot cover the baseline lifecycle replay")
    x0, sign = (initial-lo)/width, 1 if direction == "maximize" else -1
    visits = phase_calls = lifecycle_calls = 0
    best = None
    best_score = -float("inf")
    archives = {"eval": [], "funded": []}
    started = perf_counter()

    def decode(x):
        return DollarPolicy(tuple(replace(r, risk_dollars=float(v))
                                  for r, v in zip(policy.regimes, lo+np.asarray(x)*width)))

    def lifecycle(candidate):
        nonlocal visits, lifecycle_calls, best, best_score
        if visits + full_cost > search.trade_budget:
            raise _Limit
        visits += full_cost
        lifecycle_calls += 1
        result = backtest(spec, train, candidate, config)
        score = _number(objective(result))
        allowed = True if constraint is None else constraint(result)
        if type(allowed) is not bool:
            raise ValueError("constraint must return bool")
        if allowed and sign*score > best_score:
            visited = {e.regime for e in result.events if e.regime is not None}
            best = DollarPolicy(tuple(r if r.name in visited else old
                                     for r, old in zip(candidate.regimes, policy.regimes)))
            best_score = sign*score
        return sign*score if allowed else -float("inf")

    lifecycle(policy)
    if architecture == "joint":
        cache = OrderedDict({tuple(x0): best_score})

        def evaluate(x):
            key = tuple(x)
            if key not in cache:
                cache[key] = lifecycle(decode(x))
                if len(cache) > 256:
                    cache.popitem(last=False)
            return cache[key]

        try:
            CMAES(x0, .25, popsize=search.population, bounds=(np.zeros(len(x0)), np.ones(len(x0))),
                  seed=seed, max_gen=search.trade_budget//full_cost).optimize(evaluate)
        except _Limit:
            pass
    else:
        windows = window_slices(train, RollingConfig(search.horizon_sessions, search.stride_sessions))
        phase_cost = sum(b-a for a, b in windows)

        @lru_cache(maxsize=64)
        def window(a, b):
            return BracketHistory(train.trades[a:b])
        quota = (search.trade_budget-full_cost)//4
        if quota < 4*phase_cost or search.trade_budget-full_cost-2*quota < full_cost:
            raise ValueError("trade_budget must cover four candidate sweeps per phase and a lifecycle pair")
        screening = replace(config, initial_wallet=None)
        for index, phase in enumerate(("eval", "funded")):
            phase_spec = replace(spec, account=replace(spec.account,
                phases=(spec.account.phases[index],),
                eval_fee=spec.account.eval_fee if phase == "eval" else 0,
                activation_fee=spec.account.activation_fee if phase == "funded" else 0))
            mask = np.asarray([i for i, r in enumerate(policy.regimes) if r.phase == phase])
            objectives = (0, 1, 2) if phase == "eval" else (0, 1, 2, 3)
            slots = quota//phase_cost
            for run, axis in enumerate(objectives):
                allowance = slots//len(objectives) + (run < slots % len(objectives))
                cache = OrderedDict()
                calls = 0

                def evaluate(x):
                    nonlocal visits, phase_calls, calls
                    key = tuple(x)
                    if key not in cache:
                        if calls >= allowance:
                            raise _Limit
                        whole = x0.copy()
                        whole[mask] = x
                        candidate = decode(whole)
                        visits += phase_cost
                        phase_calls += 1
                        calls += 1
                        samples = tuple(_phase_sample(phase_spec, window(a, b), candidate, screening)
                                        for a, b in windows)
                        record = PhaseCandidate(candidate, _metrics(samples), samples)
                        archives[phase] = _retain(archives[phase], record, phase, search.archive_size)
                        value = _axes(record, phase)[axis]
                        cache[key] = value if isfinite(value) else -1e300
                        if len(cache) > 256:
                            cache.popitem(last=False)
                    return cache[key]

                evaluate(x0[mask])
                try:
                    CMAES(x0[mask], .25, popsize=search.population,
                          bounds=(np.zeros(len(mask)), np.ones(len(mask))),
                          seed=seed+1+index*4+run, max_gen=allowance).optimize(evaluate)
                except _Limit:
                    pass
        choices = [[policy] + [c.policy for c in archives[p]] for p in ("eval", "funded")]
        # Shuffle pair order with an IS-only seed, not a preference for one phase objective.
        pairs = [(a, b) for a in choices[0] for b in choices[1]]
        rng = np.random.default_rng(seed)
        rng.shuffle(pairs)
        seen = {policy}
        for a, b in pairs:
            candidate = DollarPolicy(tuple(x if x.phase == "eval" else y
                                          for x, y in zip(a.regimes, b.regimes)))
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                lifecycle(candidate)
            except _Limit:
                break
    if best is None:
        raise ValueError("no searched policy satisfies the IS constraint")
    search_seconds = perf_counter()-started
    reporting = perf_counter()
    report_visits = 0

    def report(part):
        nonlocal report_visits
        result = backtest(spec, part, best, config)
        report_visits += len(part.trades)
        unlimited = None
        if any(e.kind == "wallet_wait" for e in result.events):
            unlimited = backtest(spec, part, best, replace(config, initial_wallet=None))
            report_visits += len(part.trades)
        stats = risk_report([cash_risk_path(result, unrestricted=unlimited)],
                            options=risk, sample_kind="single_history")
        return result, stats

    training, train_risk = report(train)
    held, held_risk = report(test)
    fit = HoldoutFit(best, training, held, _number(objective(training)), _number(objective(held)),
                     train.sessions, test.sessions, lifecycle_calls, seed, direction)
    return PhaseFit(fit, architecture, search, tuple(archives["eval"]), tuple(archives["funded"]),
                    SearchWork(search.trade_budget, visits, report_visits, phase_calls, lifecycle_calls,
                               search_seconds, perf_counter()-reporting), train_risk, held_risk)


def compare_searches(spec, history, config, **kwargs):
    """Paired search-budget comparison; never selects an architecture using OOS."""
    if "architecture" in kwargs:
        raise ValueError("compare_searches evaluates both architectures")
    return {name: fit_phases(spec, history, config, architecture=name, **kwargs)
            for name in ("joint", "separate")}
