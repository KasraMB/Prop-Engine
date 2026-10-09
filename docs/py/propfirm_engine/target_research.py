"""Conditional risk/target research using the canonical account lifecycle.

This is an ideal one-bracket-per-session model, not historical retargeting or
a simulation of Brownian passage times. Every target changes its hit probability.
"""
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from math import exp, fsum, isfinite, sqrt
from numbers import Integral, Real
from zoneinfo import ZoneInfo

import numpy as np

from .analytical import _log_frozen_probability, _number
from .backtest import BacktestResult, _Replay
from .execution import BacktestConfig, BracketHistory, BracketTrade, DollarPolicy, RiskRegime
from .optimizer import CMAES
from .risk import RiskConfig, cash_risk_path, distribution, risk_report
from .uncertainty import _from_distributions, paired_uncertainty


def bracket_probability(stop, target, *, mu=0.0, sigma=1000.0):
    """Probability of hitting +target before -stop for dX=mu*dt+sigma*dW.

    Gross dollar barriers and gross drift, before round-trip execution costs.
    At zero drift this is stop/(stop+target). This is an eventual hit probability,
    not a probability of completing the bracket before a session deadline.
    """
    stop = _number(stop, "stop", positive=True)
    target = _number(target, "target", positive=True)
    mu = _number(mu, "mu")
    sigma = _number(sigma, "sigma", positive=True)
    nu = 2 * (mu / sigma) / sigma
    if not isfinite(nu) or not isfinite(stop + target):
        raise ValueError("barrier parameters exceed the supported numeric range")
    return min(1.0, exp(_log_frozen_probability(nu, stop, target)))


@dataclass(frozen=True)
class TargetPolicy:
    """Net dollar loss budgets and net profit targets, aligned by regime name.

    Kept distinct from DollarPolicy so historical replay cannot silently ignore
    the target part. The theoretical model uses one contract and adjusts brackets.
    """
    sizing: DollarPolicy
    targets: tuple[float, ...]

    def __post_init__(self):
        if not isinstance(self.sizing, DollarPolicy):
            raise TypeError("sizing must be a DollarPolicy")
        object.__setattr__(self, "targets", tuple(self.targets))
        if len(self.targets) != len(self.sizing.regimes):
            raise ValueError("one target is required for each sizing regime")
        for value in self.targets:
            _number(value, "target", positive=True)


@dataclass(frozen=True)
class BracketModel:
    sessions: int = 60
    mu: float = 0.0
    sigma: float = 1000.0
    start_date: date = date(2026, 1, 5)

    def __post_init__(self):
        if isinstance(self.sessions, bool) or not isinstance(self.sessions, Integral) or self.sessions < 1:
            raise ValueError("sessions must be a positive integer")
        _number(self.mu, "mu")
        _number(self.sigma, "sigma", positive=True)
        if not isinstance(self.start_date, date) or isinstance(self.start_date, datetime):
            raise ValueError("start_date must be a date")


@dataclass(frozen=True)
class TargetDecision:
    session_index: int
    regime: str
    buffer: float
    net_risk: float
    net_target: float
    probability: float
    won: bool


@dataclass(frozen=True)
class ResearchPath:
    replay: BacktestResult
    decisions: tuple[TargetDecision, ...]


def _calendar(spec, model):
    return _calendar_template(spec.session_timezone, tuple(spec.session_weekdays), model.start_date, model.sessions)


@lru_cache(maxsize=4)
def _calendar_template(timezone, weekdays, start_date, sessions):
    """Bounded cache of immutable templates only; never account state or outcomes."""
    day, trades = start_date, []
    zone = ZoneInfo(timezone)
    while len(trades) < sessions:
        if day.weekday() in weekdays:
            start = datetime.combine(day, time(10), zone)
            trades.append(BracketTrade(start, start + timedelta(minutes=5), day, 1, 1, True))
        day += timedelta(days=1)
    return BracketHistory(tuple(trades))


def research_path(spec, model, policy, config, uniforms):
    """Run one supplied [0,1) tape; all firm mechanics remain in _Replay.

    Each calendar session owns a draw even if the account skips that session.
    Reusing tapes across policies is variance reduction, not a price-path replay.
    Costs are translated into gross barriers once, then charged by the engine.
    """
    if not isinstance(model, BracketModel) or not isinstance(policy, TargetPolicy):
        raise TypeError("research_path requires BracketModel and TargetPolicy")
    if not isinstance(config, BacktestConfig):
        raise TypeError("config must be BacktestConfig")
    tape = tuple(uniforms)
    if len(tape) != model.sessions or any(
        isinstance(u, bool) or not isinstance(u, Real) or not isfinite(u) or not 0 <= u < 1
        for u in tape
    ):
        raise ValueError("uniforms needs one finite value in [0,1) per session")
    targets = {r.name: Fraction(str(t)) for r, t in zip(policy.sizing.regimes, policy.targets)}
    decisions = []
    costs = Fraction(str(config.cost_per_contract)) + Fraction(str(config.cost_per_trade))

    def factory(template, regime, runner, index):
        buffer = runner.sim.equity - runner.sim.dd_floor
        risk = min(runner.risk_budgets[regime.name], buffer)
        # A minimum one-cent gross bracket allows canonical projection to decide
        # CAPPED_OUT versus a voluntary sub-contract policy skip.
        if risk < costs + Fraction(1, 100):
            return replace(template, stop_loss=0.01, take_profit=0.01, won=False)
        target = targets[regime.name]
        stop, take = risk - costs, target + costs
        probability = bracket_probability(float(stop), float(take), mu=model.mu, sigma=model.sigma)
        won = bool(tape[index] < probability)
        decisions.append(TargetDecision(index, regime.name, float(buffer), float(risk),
                                        float(target), probability, won))
        return replace(template, stop_loss=stop, take_profit=take, won=won)

    replay = _Replay(spec, _calendar(spec, model), policy.sizing, config, bracket_factory=factory).run()
    assumptions = tuple(a for a in replay.assumptions if
                        a != "gross per-contract historical stop/target outcomes are held fixed") + (
        "theoretical independent barrier outcomes; not historical strategy performance",
        "one contract, freely adjustable dollar barriers; no instrument tick grid",
        "one eventual bracket outcome assigned to each available session; passage times are not modeled",
        "gross drift and volatility determine hit probabilities, not calendar duration",
        "common uniform tapes couple policies statistically, not through shared market paths",
    )
    fingerprint = sha256(repr((model, policy, tape)).encode()).hexdigest()
    return ResearchPath(replace(replay, assumptions=assumptions, history_fingerprint=fingerprint), tuple(decisions))


@dataclass(frozen=True)
class ResearchSummary:
    paths: int
    score: float
    score_standard_error: float
    mean_net_cash: float
    mean_receipts: float
    mean_fees: float
    payout_probability: float
    net_cash_p05: float
    net_cash_median: float
    net_cash_p95: float
    objective_values: tuple[float, ...]
    visited_regimes: tuple[str, ...]
    scope: str = "independent model paths; uncertainty is sampling error only, not model risk"
    distributions: dict | None = None
    risk: dict | None = None
    selected_on_sample: bool = False

    @property
    def uncertainty(self):
        return _from_distributions(self.distributions or {"objective": distribution(self.objective_values)},
            sample_kind="independent_model", count=self.paths, selected_on_sample=self.selected_on_sample)


def evaluate_targets(spec, model, policy, config, tapes, *, objective=None, risk=RiskConfig()):
    """Equal-weight path objective; default received external net cash/calendar day."""
    objective = objective if objective is not None else lambda r: r.net_cash_per_day
    if not callable(objective):
        raise ValueError("objective must be callable")
    scores, cash, cash_per_day, receipts, fees, paid, visited = [], [], [], [], [], [], set()
    risk_paths = []
    for tape in tapes:
        tape = tuple(tape)
        path = research_path(spec, model, policy, config, tape)
        result = path.replay
        value = objective(result)
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError("objective must return a finite real scalar")
        scores.append(float(value))
        cash.append(result.net_cash)
        cash_per_day.append(result.net_cash_per_day)
        receipts.append(result.receipts)
        fees.append(result.fees)
        paid.append(result.receipts > 0)
        visited.update(d.regime for d in path.decisions)
        if risk is not None:
            unlimited = (research_path(spec, model, policy, replace(config, initial_wallet=None), tape).replay
                         if any(e.kind == "wallet_wait" for e in result.events) else None)
            risk_paths.append(cash_risk_path(result, unrestricted=unlimited))
    n = len(scores)
    if n < 2:
        raise ValueError("at least two independent tapes are required for a summary")
    return ResearchSummary(n, fsum(scores) / n, float(np.std(scores, ddof=1) / sqrt(n)),
                           fsum(cash) / n, fsum(receipts) / n, fsum(fees) / n, sum(paid) / n,
                           *map(float, np.quantile(cash, [0.05, 0.5, 0.95])),
                           tuple(scores), tuple(sorted(visited)),
                           distributions={"objective": distribution(scores), "net_cash": distribution(cash),
                                          "net_cash_per_day": distribution(cash_per_day)},
                           risk=risk_report(risk_paths, options=risk, sample_kind="independent_model")
                           if risk is not None else None)


@dataclass(frozen=True)
class TargetFit:
    policy: TargetPolicy
    training: ResearchSummary
    holdout: ResearchSummary
    baseline_holdout: ResearchSummary
    evaluations: int
    seed: int
    holdout_seed: int
    candidate_seeds: int
    direction: str
    paired_holdout_gain: float
    paired_gain_standard_error: float
    scope: str = "70/30 split of independent model paths; NOT historical IS/OOS validation"

    @property
    def uncertainty(self):
        sign = 1 if self.direction == "maximize" else -1
        paired = paired_uncertainty(
            {"objective_gain": [sign*v for v in self.holdout.objective_values]},
            {"objective_gain": [sign*v for v in self.baseline_holdout.objective_values]},
            sample_kind="independent_model")
        return {"holdout": self.holdout.uncertainty, "paired_gain": paired,
                "selection": "frozen IS-selected policy; no correction for repeated holdout reuse"}


def fit_targets(spec, model, config, *, policy, risk_bounds, target_bounds,
                paths=40, seed=0, holdout_seed=1, generations=10, population=8,
                objective=None, direction="maximize", candidates=(), progress=None,
                risk_choices=None, target_choices=None, risk=RiskConfig()):
    """Joint bounded CMA-ES search. Holdout tapes are evaluated after selection.

    Bounds map regime names to dollar intervals. Candidate seeds are explicit;
    including a known policy tests it but must not be described as discovery.
    Search actions round to cents. Unvisited regimes retain baseline parameters.
    Optional per-regime choices snap search proposals to declared dollar levels;
    useful for exact contractual thresholds that a continuous search rarely hits.
    """
    for name, value, minimum in (("paths", paths, 7), ("seed", seed, 0),
                                  ("holdout_seed", holdout_seed, 0),
                                  ("generations", generations, 0), ("population", population, 2)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if direction not in ("maximize", "minimize"):
        raise ValueError("direction must be maximize or minimize")
    if not isinstance(policy, TargetPolicy):
        raise TypeError("policy must be TargetPolicy")
    names = tuple(r.name for r in policy.sizing.regimes)
    if set(risk_bounds) != set(names) or set(target_bounds) != set(names):
        raise ValueError("both bounds must specify exactly the policy regime names")
    bounds = np.asarray([risk_bounds[n] for n in names] + [target_bounds[n] for n in names], dtype=float)
    if (bounds.shape != (2 * len(names), 2) or not np.all(np.isfinite(bounds))
            or np.any(bounds[:, 0] < 0.01) or np.any(bounds[:, 1] <= bounds[:, 0])
            or not np.allclose(bounds * 100, np.round(bounds * 100), rtol=0, atol=1e-7)):
        raise ValueError("bounds must be cent-valued and satisfy 0.01 <= minimum < maximum")
    candidates = tuple(candidates)
    structure = lambda p: tuple(replace(r, risk_dollars=0) for r in p.sizing.regimes)
    values = lambda p: np.array([r.risk_dollars for r in p.sizing.regimes] + list(p.targets))
    lo, width = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    choices = []
    for offset, mapping in ((0, risk_choices), (len(names), target_choices)):
        if mapping is not None and set(mapping) != set(names):
            raise ValueError("choices must specify exactly the policy regime names")
        for i, name in enumerate(names):
            levels = None if mapping is None else np.asarray(mapping[name], dtype=float)
            if levels is not None and (
                levels.ndim != 1 or not len(levels) or not np.all(np.isfinite(levels))
                or np.any(levels < bounds[offset+i, 0]) or np.any(levels > bounds[offset+i, 1])
                or not np.allclose(levels * 100, np.round(levels * 100), rtol=0, atol=1e-7)
            ):
                raise ValueError("choices must be nonempty cent-valued levels within bounds")
            choices.append(None if levels is None else np.unique(levels))
    for candidate in (policy,) + candidates:
        if not isinstance(candidate, TargetPolicy) or structure(candidate) != structure(policy):
            raise ValueError("candidate policies must have identical regime conditions")
        if np.any(values(candidate) < lo) or np.any(values(candidate) > bounds[:, 1]):
            raise ValueError("initial and candidate policies must lie within bounds")
    train_count = paths * 7 // 10
    # Separate streams even if the caller uses identical integer seeds.
    train = np.random.default_rng(np.random.SeedSequence([seed, 0])).random((train_count, model.sessions))
    sign = 1 if direction == "maximize" else -1
    cache = {}

    def score(candidate):
        if candidate not in cache:
            summary = evaluate_targets(spec, model, candidate, config, train, objective=objective, risk=None)
            cache[candidate] = summary.score, summary.visited_regimes
            if progress:
                progress({"stage": "search", "evaluations": len(cache)})
        return sign * cache[candidate][0]

    def decode(x):
        v = np.round(lo + np.asarray(x) * width, 2)
        for i, levels in enumerate(choices):
            if levels is not None:
                v[i] = levels[np.argmin(abs(levels - v[i]))]
        return TargetPolicy(DollarPolicy(tuple(replace(r, risk_dollars=float(risk)) for r, risk
                            in zip(policy.sizing.regimes, v[:len(names)]))), tuple(v[len(names):]))

    for candidate in (policy,) + candidates:
        score(candidate)
    search = CMAES((values(policy) - lo) / width, 0.25, popsize=population,
                   bounds=(np.zeros(len(lo)), np.ones(len(lo))), seed=seed, max_gen=generations)
    search.optimize(lambda x: score(decode(x)))
    selected = max(cache, key=lambda p: sign * cache[p][0])
    visited = cache[selected][1]
    selected = TargetPolicy(DollarPolicy(tuple(fitted if fitted.name in visited else original
        for fitted, original in zip(selected.sizing.regimes, policy.sizing.regimes))),
        tuple(t if n in visited else old for n, t, old in zip(names, selected.targets, policy.targets)))
    training = replace(evaluate_targets(spec, model, selected, config, train, objective=objective, risk=None),
                       selected_on_sample=True)
    test = np.random.default_rng(np.random.SeedSequence([holdout_seed, 1])).random((paths - train_count, model.sessions))
    held = evaluate_targets(spec, model, selected, config, test, objective=objective, risk=risk)
    baseline = evaluate_targets(spec, model, policy, config, test, objective=objective, risk=risk)
    gains = sign * (np.asarray(held.objective_values) - baseline.objective_values)
    return TargetFit(selected, training, held, baseline, len(cache), seed, holdout_seed,
                     len(candidates), direction, float(gains.mean()), float(gains.std(ddof=1) / sqrt(len(gains))))


def lucidflex_example():
    """The declared 2000/1500 -> 2000/2000 -> 2000/150 research benchmark.

    A named example, not firm-rule configuration or an optimizer-discovered policy.
    Surviving losses and later payout cycles use the same fallback rule.
    """
    return TargetPolicy(DollarPolicy((
        RiskRegime("evaluation", "eval", 2000),
        RiskRegime("funded_build", "funded", 2000, days_to_payout=5, after_payout=False),
        RiskRegime("funded_days", "funded", 2000),
    )), (1500, 2000, 150))
