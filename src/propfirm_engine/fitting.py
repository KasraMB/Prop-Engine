"""Chronological 70/30 holdout fitting using the existing CMA-ES optimizer."""
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Integral, Real

import numpy as np

from .backtest import BacktestResult, backtest
from .execution import BracketHistory, DollarPolicy
from .optimizer import CMAES
from .rolling import RollingConfig, RollingResult, rolling_backtest, window_slices


@dataclass(frozen=True)
class HoldoutFit:
    policy: DollarPolicy
    in_sample: BacktestResult
    out_of_sample: BacktestResult
    in_sample_score: float
    out_of_sample_score: float
    train_sessions: tuple
    test_sessions: tuple
    evaluations: int
    seed: int
    direction: str
    boundary_policy: str = "fresh account and wallet at the OOS boundary"
    in_sample_rolling: RollingResult | None = None
    out_of_sample_rolling: RollingResult | None = None

    @property
    def score(self):
        """The headline score is ALWAYS the untouched OOS score."""
        return self.out_of_sample_score


def fit_holdout(spec, history: BracketHistory, config, *, policy: DollarPolicy,
                risk_bounds, objective=None, direction="maximize",
                train_fraction=0.7, seed=0, generations=20, population=12,
                rolling: RollingConfig | None = None, progress=None):
    """Select dollar sizing on IS only, then replay the frozen policy once on OOS.

    risk_bounds maps every regime name to a (minimum, maximum) dollar budget.
    objective accepts BacktestResult. Default: observed net cash/calendar day.
    With rolling supplied, optimize the equal-weight mean window objective on
    IS and report the same statistic on OOS. Full chronological replays remain
    available alongside rolling summaries. Windows never cross the split.
    This empirical criterion is not claimed to be an unbiased expected value.
    """
    if direction not in ("maximize", "minimize"):
        raise ValueError("direction must be maximize or minimize")
    for name, value, minimum in (("seed", seed, 0), ("generations", generations, 0),
                                  ("population", population, 2)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    names = tuple(r.name for r in policy.regimes)
    if set(risk_bounds) != set(names):
        raise ValueError("risk_bounds must specify exactly the policy's regime names")
    bounds = np.asarray([risk_bounds[n] for n in names], dtype=float)
    if (bounds.shape != (len(names), 2) or not np.all(np.isfinite(bounds))
            or np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] <= bounds[:, 0])):
        raise ValueError("each risk bound must satisfy 0 <= minimum < maximum")
    train, test = history.split(train_fraction)
    if rolling is not None:
        window_slices(train, rolling)
        window_slices(test, rolling)
    objective = objective if objective is not None else lambda result: result.net_cash_per_day
    if not callable(objective):
        raise ValueError("objective must be callable")
    sign = 1 if direction == "maximize" else -1
    lo, width = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    initial = np.asarray([r.risk_dollars for r in policy.regimes], dtype=float)
    if np.any(initial < lo) or np.any(initial > bounds[:, 1]):
        raise ValueError("initial policy must lie within risk_bounds")
    cache = {}

    def decode(x):
        values = lo + np.asarray(x) * width
        return DollarPolicy(tuple(replace(r, risk_dollars=float(v))
                                  for r, v in zip(policy.regimes, values)))

    def score(result):
        value = objective(result)
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError("objective must return a finite real scalar")
        return float(value)

    def evaluate(x):
        key = tuple(float(v) for v in x)
        if key not in cache:
            candidate = decode(x)
            result = (rolling_backtest(spec, train, candidate, config, rolling=rolling, objective=objective)
                      if rolling is not None else backtest(spec, train, candidate, config))
            visited = (frozenset(result.visited_regimes) if rolling is not None else
                       frozenset(e.regime for e in result.events if e.regime is not None))
            # Do not retain a full event ledger for every search candidate.
            cache[key] = (result.score if rolling is not None else score(result), candidate, visited)
            if progress:
                progress({"stage": "search", "evaluations": len(cache),
                          "maximum_evaluations": generations * population + 2})
        return sign * cache[key][0]

    x0 = (initial - lo) / width
    evaluate(x0)  # the supplied baseline participates in IS selection
    search = CMAES(x0, 0.25, popsize=population, bounds=(np.zeros(len(lo)), np.ones(len(lo))),
                   seed=seed, max_gen=generations)
    search.optimize(evaluate)
    best_score, best_policy, visited = max(cache.values(), key=lambda item: sign * item[0])
    # No data supports tuning an IS-unvisited regime: retain its declared baseline.
    best_policy = DollarPolicy(tuple(
        fitted if fitted.name in visited else original
        for fitted, original in zip(best_policy.regimes, policy.regimes)
    ))
    if progress:
        progress({"stage": "holdout", "evaluations": len(cache)})
    training = backtest(spec, train, best_policy, config)
    if rolling is None:
        best_score = score(training)
    # No OOS result can affect candidate selection, regime design or search stopping.
    held_out = backtest(spec, test, best_policy, config)
    train_rolling = test_rolling = None
    if rolling is not None:
        train_rolling = rolling_backtest(spec, train, best_policy, config, rolling=rolling, objective=objective)
        test_rolling = rolling_backtest(spec, test, best_policy, config, rolling=rolling, objective=objective)
        best_score = train_rolling.score
    return HoldoutFit(best_policy, training, held_out, best_score,
                      test_rolling.score if test_rolling is not None else score(held_out),
                      train.sessions, test.sessions, len(cache), seed, direction,
                      in_sample_rolling=train_rolling, out_of_sample_rolling=test_rolling)
