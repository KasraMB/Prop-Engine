"""Chronological search over external causal strategies and shared market tapes."""
from dataclasses import dataclass, replace
from functools import cached_property
from fractions import Fraction
from hashlib import sha256
from itertools import product
from math import floor, fsum, isfinite
from numbers import Real
from types import MappingProxyType

import numpy as np

from .market_data import MarketTape, MarketView
from .optimizer import CMAES
from .risk import RiskConfig, cash_risk_path, distribution, risk_report
from .rolling import RollingConfig
from .strategy import ReplayCancelled, replay_strategy


def _scalar(value):
    return (type(value) in (str, bool, int) or value is None
            or type(value) is float and isfinite(value))


@dataclass(frozen=True)
class Parameter:
    kind: str = "continuous"
    low: float = 0
    high: float = 1
    choices: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "choices", tuple(self.choices))
        if self.kind == "categorical":
            if (len(self.choices) < 2 or any(not _scalar(v) for v in self.choices)
                    or len(set(self.choices)) != len(self.choices)):
                raise ValueError("categorical choices must be distinct finite scalar values")
        elif self.kind in ("continuous", "integer"):
            if (any(type(v) not in (float, int) or not isfinite(v) for v in (self.low, self.high))
                    or self.low >= self.high or self.choices):
                raise ValueError("numeric bounds must be finite and increasing")
            if self.kind == "integer" and any(type(v) is not int for v in (self.low, self.high)):
                raise ValueError("integer bounds must be integers")
        else:
            raise ValueError("parameter kind must be continuous, integer or categorical")

    def decode(self, value):
        value = min(1.0, max(0.0, float(value)))
        if self.kind == "categorical":
            return self.choices[min(len(self.choices)-1, floor(value*len(self.choices)))]
        if self.kind == "integer":
            return min(self.high, self.low+floor(value*(self.high-self.low+1)))
        return self.low + value*(self.high-self.low)

    def encode(self, value):
        if self.kind == "categorical":
            if value not in self.choices:
                raise ValueError("baseline is outside categorical choices")
            return (self.choices.index(value)+.5)/len(self.choices)
        if (type(value) not in (float, int) or not isfinite(value) or not self.low <= value <= self.high
                or self.kind == "integer" and type(value) is not int):
            raise ValueError("baseline is outside its parameter domain")
        if self.kind == "integer":
            return (value-self.low+.5)/(self.high-self.low+1)
        return (value-self.low)/(self.high-self.low)


class InfeasiblePolicy(ValueError):
    """An explicitly declared policy constraint, not a simulator failure."""


@dataclass(frozen=True)
class StrategyPath:
    first_session: object
    last_session: object
    seed: int
    cash: object
    balance: float | None
    equity: float | None
    fills: int
    rejected_orders: int
    passed_stages: int


@dataclass(frozen=True)
class StrategyEvaluation:
    parameters: tuple
    paths: tuple[StrategyPath, ...]
    risk: RiskConfig
    fingerprint: str
    score: float | None = None
    sample_kind: str | None = None

    @property
    def params(self):
        return dict(self.parameters)

    @property
    def mean_net_cash(self):
        return fsum(p.cash.net_cash for p in self.paths)/len(self.paths)

    @property
    def mean_cash_per_day(self):
        return fsum(p.cash.net_cash_per_day for p in self.paths)/len(self.paths)

    @cached_property
    def metrics(self):
        report = risk_report((p.cash for p in self.paths), options=self.risk,
            sample_kind=self.sample_kind or ("single_history" if len(self.paths) == 1 else "historical_windows"))
        for name in ("balance", "equity", "fills", "rejected_orders", "passed_stages"):
            report["distributions"][name] = distribution(
                (getattr(p, name) for p in self.paths if getattr(p, name) is not None), self.risk.percentiles)
        report["uncertainty"] = {
            "historical": "chronological windows; overlapping windows are dependent",
            "execution": "conditional on setup(seed); repeated seeds are not independent market histories",
            "market": ("conditional on the declared independent scenario generator"
                       if self.sample_kind == "independent_model" else "not inferred from one historical tape"),
            "horizon": "fresh account and wallet per window; warmup is observation-only",
            "open_positions": "marked at the horizon, not credited as external cash",
            "ultimate_ruin": "not inferred; use the separately labelled complete-cycle approximation",
        }
        return report


def _windows(view, rolling):
    if rolling is None:
        return (view,)
    if not isinstance(rolling, RollingConfig):
        raise TypeError("rolling must be RollingConfig")
    count = len(view.sessions)
    if rolling.window_sessions > count:
        raise ValueError("rolling window exceeds the available partition")
    return tuple(view.view(i, i+rolling.window_sessions)
                 for i in range(0, count-rolling.window_sessions+1, rolling.stride_sessions))


def _view(tape):
    if isinstance(tape, MarketTape):
        return tape.view()
    if not isinstance(tape, MarketView):
        raise TypeError("strategy fitting requires a prepared MarketTape or MarketView")
    return tape


def _params(params):
    params = dict(params)
    if any(not isinstance(k, str) or not k or not _scalar(v) for k, v in params.items()):
        raise ValueError("parameters must map nonempty names to finite scalar values")
    return tuple(sorted(params.items()))


def _seeds(seeds):
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    return seeds


def evaluate_strategy(spec, tape, config, factory, *, params, setup, seeds=(0,), rolling=None,
                      warmup_sessions=0, risk=RiskConfig(), objective=None, cancel=None):
    """Evaluate frozen parameters. setup(seed) returns fresh replay models and options."""
    view, parameters, seeds = _view(tape), _params(params), _seeds(seeds)
    if not callable(factory) or not callable(setup):
        raise TypeError("factory and setup must be callables")
    if type(warmup_sessions) is not int or warmup_sessions < 0:
        raise ValueError("warmup_sessions must be a nonnegative integer")
    if not isinstance(risk, RiskConfig):
        raise TypeError("risk must be RiskConfig")
    paths = []
    digest = sha256(repr((view.fingerprint, parameters, seeds, warmup_sessions, rolling)).encode())
    for window in _windows(view, rolling):
        warmup = () if not warmup_sessions or window.first == 0 else window.tape.view(
            max(0, window.first-warmup_sessions), window.first)
        for seed in seeds:
            def run(run_config):
                if cancel is not None and cancel():
                    raise ReplayCancelled("strategy evaluation cancelled")
                options = dict(setup(seed))
                if "trace" not in options:
                    options.setdefault("recording", "search")
                if {"sessions", "warmup", "cancel"} & options.keys():
                    raise ValueError("setup cannot override sessions, warmup or cancellation")
                strategy = factory(MappingProxyType(dict(parameters)), seed)
                return replay_strategy(spec, window, window.instruments, run_config, strategy,
                                       sessions=window.sessions, warmup=warmup, cancel=cancel, **options)

            result = run(config)
            replay = result.result.replay
            unrestricted = None
            if any(e.kind == "wallet_wait" for e in replay.events):
                unrestricted = run(replace(config, initial_wallet=None)).result.replay
            cash = cash_risk_path(replay, unrestricted=unrestricted)
            book = result.result.book
            paths.append(StrategyPath(window.sessions[0], window.sessions[-1], seed, cash,
                float(book.balance) if book else None, float(book.equity) if book else None,
                result.result.fills, dict(result.order_counts).get("rejected", 0),
                dict(result.result.event_counts).get("evaluation_pass", 0)))
            digest.update(replay.history_fingerprint.encode())
    evaluation = StrategyEvaluation(parameters, tuple(paths), risk, digest.hexdigest())
    score = evaluation.mean_cash_per_day if objective is None else objective(evaluation)
    if isinstance(score, bool) or not np.isscalar(score) or not isfinite(float(score)):
        raise ValueError("objective must return a finite scalar")
    return replace(evaluation, score=float(score))


@dataclass(frozen=True)
class StrategyTrial:
    partition: str
    parameters: tuple
    score: float | None
    feasible: bool
    reason: str = ""


@dataclass(frozen=True)
class StrategyCheckpoint:
    fingerprint: str
    trials: tuple[StrategyTrial, ...]


class SearchCancelled(ReplayCancelled):
    def __init__(self, checkpoint):
        super().__init__("strategy search cancelled; completed trials can be resumed")
        self.checkpoint = checkpoint


@dataclass(frozen=True)
class StrategyFit:
    selected: StrategyEvaluation
    baseline: StrategyEvaluation
    training: StrategyEvaluation
    validation: StrategyEvaluation | None
    trials: tuple[StrategyTrial, ...]
    checkpoint: StrategyCheckpoint
    split_session: object
    work: dict

    @property
    def params(self):
        return self.selected.params

    @property
    def score(self):
        return self.selected.score

    @property
    def metrics(self):
        return self.selected.metrics

    @property
    def stability(self):
        feasible = [t for t in self.trials if t.feasible]
        return {"trials": len(self.trials), "infeasible": len(self.trials)-len(feasible),
                "selected_parameters": self.params,
                "training_score": self.training.score,
                "validation_score": None if self.validation is None else self.validation.score,
                "oos_score": self.score,
                "baseline_oos_score": self.baseline.score,
                "note": "diagnostics only; OOS must not select settings or stopping"}


def fit_strategy(spec, tape, config, factory, *, baseline, space, setup, train_fraction=.7,
                 validation_fraction=0, finalists=3, generations=20, population=8, seed=0,
                 seeds=(0,), rolling=None, warmup_sessions=0, risk=RiskConfig(), objective=None,
                 direction="maximize", constraint=None, study_id=None, resume=None,
                 checkpoint=None, progress=None, cancel=None):
    """Search IS only; report selected and baseline performance on untouched OOS."""
    view, base, seeds = _view(tape), _params(baseline), _seeds(seeds)
    space = dict(sorted(dict(space).items()))
    if not space or any(k not in dict(base) or not isinstance(v, Parameter) for k, v in space.items()):
        raise ValueError("space must declare Parameter domains for baseline names")
    for name, value, low in (("generations", generations, 0), ("population", population, 4),
                             ("finalists", finalists, 1), ("seed", seed, 0)):
        if type(value) is not int or value < low:
            raise ValueError(f"{name} must be an integer >= {low}")
    if (isinstance(train_fraction, bool) or not isinstance(train_fraction, Real) or not 0 < train_fraction < 1
            or isinstance(validation_fraction, bool) or not isinstance(validation_fraction, Real)
            or not 0 <= validation_fraction < 1):
        raise ValueError("train_fraction must be in (0,1); validation_fraction in [0,1)")
    if direction not in ("maximize", "minimize"):
        raise ValueError("direction must be maximize or minimize")
    if (resume is not None or checkpoint is not None) and (not isinstance(study_id, str) or not study_id):
        raise ValueError("checkpoint/resume requires a study_id identifying code and execution settings")
    split = floor(len(view.sessions)*Fraction(str(train_fraction)))
    if not 0 < split < len(view.sessions):
        raise ValueError("split must leave complete sessions in IS and OOS")
    training_end = floor(split*(1-Fraction(str(validation_fraction))))
    if not 0 < training_end <= split or validation_fraction and training_end == split:
        raise ValueError("inner validation must leave complete training and validation sessions")
    training, oos = view.view(0, training_end), view.view(split)
    validation = view.view(training_end, split) if training_end < split else None
    partitions = {"training": training, "oos": oos}
    if validation is not None:
        partitions["validation"] = validation
    for part in partitions.values():
        _windows(part, rolling)
    x0 = [domain.encode(dict(base)[name]) for name, domain in space.items()]
    key = sha256(repr(("strategy-fit-v1", view.fingerprint, spec, config, base, tuple(space.items()),
        train_fraction, validation_fraction, finalists, generations, population, seed, seeds,
        rolling, warmup_sessions, risk, direction, study_id)).encode()).hexdigest()
    trials, cache = [], {}
    if resume is not None:
        if not isinstance(resume, StrategyCheckpoint) or resume.fingerprint != key:
            raise ValueError("checkpoint does not match this study, inputs or settings")
        for trial in resume.trials:
            if (not isinstance(trial, StrategyTrial) or trial.partition not in ("training", "validation")
                    or trial.feasible and (trial.score is None or not isfinite(trial.score))):
                raise ValueError("invalid checkpoint trial")
            trial_key = trial.partition, trial.parameters
            if trial_key in cache:
                raise ValueError("duplicate checkpoint trial")
            cache[trial_key] = trial
            trials.append(trial)

    def state():
        return StrategyCheckpoint(key, tuple(trials))

    def check_cancel():
        if cancel is not None and cancel():
            raise SearchCancelled(state())

    def evaluate(parameters, part):
        check_cancel()
        if constraint is not None and not constraint(MappingProxyType(dict(parameters))):
            raise InfeasiblePolicy("declared parameter constraint")
        return evaluate_strategy(spec, partitions[part], config, factory, params=dict(parameters),
            setup=setup, seeds=seeds, rolling=rolling, warmup_sessions=warmup_sessions,
            risk=risk, objective=objective, cancel=cancel)

    sign = 1 if direction == "maximize" else -1

    def trial(parameters, part="training"):
        check_cancel()
        trial_key = part, parameters
        if trial_key not in cache:
            try:
                result = evaluate(parameters, part)
                item = StrategyTrial(part, parameters, result.score, True)
            except InfeasiblePolicy as exc:
                item = StrategyTrial(part, parameters, None, False, str(exc))
            except ReplayCancelled:
                raise SearchCancelled(state()) from None
            cache[trial_key] = item
            trials.append(item)
            if checkpoint is not None:
                checkpoint(state())
            if progress is not None:
                progress(item)
        item = cache[trial_key]
        return sign*item.score if item.feasible else -np.inf

    def decode(values):
        params = dict(base)
        params.update((name, domain.decode(value)) for (name, domain), value in zip(space.items(), values))
        return _params(params)

    if not isfinite(trial(base)):
        raise ValueError("baseline must be feasible")
    grid, count = [], 1
    for domain in space.values():
        if domain.kind == "continuous":
            count = float("inf")
            break
        values = domain.choices if domain.kind == "categorical" else range(domain.low, domain.high+1)
        grid.append(values)
        count *= len(values)
    if generations and count <= 2+generations*population:
        for values in product(*grid):
            trial(_params(dict(base) | dict(zip(space, values))))
    else:
        CMAES(x0, .3, popsize=population, bounds=(0, 1), seed=seed, max_gen=generations).optimize(
            lambda values: trial(decode(values)))
    ranked = sorted((t for t in trials if t.partition == "training" and t.feasible),
                    key=lambda t: -sign*t.score)
    if validation is None:
        chosen = ranked[0].parameters
    else:
        candidates = [base] + [t.parameters for t in ranked[:finalists] if t.parameters != base]
        chosen = max(candidates, key=lambda p: trial(p, "validation"))
    try:
        trained = evaluate(chosen, "training")
        validated = evaluate(chosen, "validation") if validation is not None else None
        selected = evaluate(chosen, "oos")
        original = selected if chosen == base else evaluate(base, "oos")
    except ReplayCancelled:
        raise SearchCancelled(state()) from None
    training_events = sum(len(w) for w in _windows(training, rolling))*len(seeds)
    work = {"max_search_candidates": 2+generations*population,
            "training_observations_per_candidate": training_events,
            "search_observation_upper_bound": (2+generations*population)*training_events,
            "tape_bytes": view.tape.nbytes,
            "retention": "shared input arrays; compact trial scores and final path summaries",
            "caveat": "excludes validation, reporting, callbacks and unrestricted-wallet reruns"}
    return StrategyFit(selected, original, trained, validated, tuple(trials), state(),
                       oos.sessions[0], work)


@dataclass(frozen=True)
class StrategyWalkForward:
    folds: tuple[StrategyFit, ...]

    @cached_property
    def metrics(self):
        paths = tuple(path for fold in self.folds for path in fold.selected.paths)
        return StrategyEvaluation((), paths, self.folds[0].selected.risk, "walk-forward").metrics


def walk_strategy(spec, tape, config, factory, *, train_sessions, test_sessions,
                  step_sessions=None, **kwargs):
    """Refit each fold on its past only. Accounts and wallets start fresh each fold."""
    view = _view(tape)
    step_sessions = test_sessions if step_sessions is None else step_sessions
    if any(type(v) is not int or v < 1 for v in (train_sessions, test_sessions, step_sessions)):
        raise ValueError("walk-forward session counts must be positive integers")
    if {"train_fraction", "resume", "checkpoint"} & kwargs.keys():
        raise ValueError("walk-forward owns fold splits; use fit_strategy for checkpointed individual folds")
    size = train_sessions+test_sessions
    if size > len(view.sessions):
        raise ValueError("not enough sessions for a complete walk-forward fold")
    folds = tuple(fit_strategy(spec, view.view(i, i+size), config, factory,
                               train_fraction=Fraction(train_sessions, size), **kwargs)
                  for i in range(0, len(view.sessions)-size+1, step_sessions))
    return StrategyWalkForward(folds)


def evaluate_scenarios(spec, tapes, config, factory, *, params, setup, sample_kind, **kwargs):
    """Frozen-policy evaluation of ordered full-market scenarios, consumed one at a time."""
    if sample_kind not in ("historical_windows", "independent_model"):
        raise ValueError("declare historical_windows or independent_model for scenario uncertainty")
    if "objective" in kwargs or "rolling" in kwargs or "seeds" in kwargs:
        raise ValueError("scenario evaluation uses one seed and complete path per supplied tape")
    paths, digest = [], sha256()
    risk = kwargs.get("risk", RiskConfig())
    for index, tape in enumerate(tapes):
        result = evaluate_strategy(spec, tape, config, factory, params=params, setup=setup,
                                   seeds=(index,), **kwargs)
        paths.extend(result.paths)
        digest.update(result.fingerprint.encode())
    if not paths:
        raise ValueError("at least one complete market scenario is required")
    evaluation = StrategyEvaluation(_params(params), tuple(paths), risk, digest.hexdigest(), sample_kind=sample_kind)
    return replace(evaluation, score=evaluation.mean_cash_per_day)
