"""IS-only dollar risk/target search on prices using the existing CMA-ES engine.

Execution scenarios are fixed inputs, never search dimensions. Monte Carlo
here repeats execution uncertainty on one historical tape, not market futures.
"""
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from math import isfinite
from numbers import Real

import numpy as np

from .execution import DollarPolicy
from .market_replay import replay_prices
from .optimizer import CMAES
from .risk import cash_risk_path, distribution
from .slippage import prepare_execution
from .target_research import TargetPolicy


@dataclass(frozen=True)
class PriceEvaluation:
    paths: tuple
    score: float
    distributions: dict
    objective_values: tuple[float, ...]
    scope: str = "execution uncertainty conditional on one historical price tape; not independent market histories"


@dataclass(frozen=True)
class PriceFit:
    policy: TargetPolicy
    in_sample: PriceEvaluation
    out_of_sample: PriceEvaluation
    baseline_out_of_sample: PriceEvaluation
    evaluations: int
    train_sessions: tuple
    test_sessions: tuple
    direction: str
    boundary_policy: str = "fresh account and wallet at the chronological OOS boundary"

    @property
    def score(self):
        return self.out_of_sample.score


def _objective(objective, result):
    value = (objective(result) if objective is not None else result.net_cash_per_day)
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError("objective must return a finite real scalar")
    return float(value)


def evaluate_prices(spec, sessions, policy, instrument, config, *, slippage=None,
                    quantity=1, paths=1, execution_seed=0, path_offset=0,
                    compensate_slippage=True, collision_policy, objective=None):
    """All performance distributions are returned irrespective of the objective."""
    if not isinstance(policy, TargetPolicy):
        raise TypeError("policy must be TargetPolicy")
    if type(paths) is not int or paths < 1 or type(path_offset) is not int or path_offset < 0:
        raise ValueError("paths must be positive and path_offset nonnegative integers")
    sessions = tuple(sessions)
    targets = dict(zip((r.name for r in policy.sizing.regimes), policy.targets))
    results = tuple(replay_prices(spec, sessions, policy.sizing, targets, instrument, config,
                                 quantity=quantity, slippage=slippage, execution_seed=execution_seed,
                                 execution_path=path_offset+i, compensate_slippage=compensate_slippage,
                                 collision_policy=collision_policy) for i in range(paths if slippage else 1))
    scores = tuple(_objective(objective, r.replay) for r in results)
    metrics = []
    for i, path in enumerate(results):
        unrestricted = None
        if any(e.kind == "wallet_wait" for e in path.replay.events):
            unrestricted = replay_prices(spec, sessions, policy.sizing, targets, instrument,
                replace(config, initial_wallet=None), quantity=quantity, slippage=slippage,
                execution_seed=execution_seed, execution_path=path_offset+i,
                compensate_slippage=compensate_slippage, collision_policy=collision_policy).replay
        row = asdict(cash_risk_path(path.replay, unrestricted=unrestricted))
        events, decisions = path.replay.events, path.decisions
        row.update(trades=len(decisions), evaluation_passes=sum(e.kind == "evaluation_pass" for e in events),
                   collisions=sum(d.collision for d in decisions),
                   slippage_ticks=sum(d.entry_slippage_ticks+d.exit_slippage_ticks for d in decisions),
                   allowance_exceedances=sum(d.net_pnl < -d.planned_net_risk-1e-8 for d in decisions),
                   session_close_exits=sum(d.reason == "session_close" for d in decisions))
        metrics.append(row)
    distributions = {}
    for key in metrics[0]:
        values = [row[key] for row in metrics]
        present = [v for v in values if v is not None]
        if present and all(isinstance(v, Real) for v in present):
            distributions[key] = {"missing": len(values)-len(present), "distribution": distribution(present)}
        elif not present:
            distributions[key] = {"missing": len(values), "distribution": None}
    return PriceEvaluation(results, float(np.mean(scores)), distributions, scores)


def fit_prices(spec, sessions, instrument, config, *, policy, risk_bounds, target_bounds,
               slippage=None, quantity=1, train_fraction=.7, paths=3, execution_seed=0,
               seed=0, generations=10, population=8, direction="maximize", objective=None,
               collision_policy, compensate_slippage=True, progress=None):
    """Optimize both dollar budgets and targets on IS; holdout never ranks policies.

    Reuses TargetPolicy and CMAES; every candidate runs the canonical account
    lifecycle through price replay. Caches only scores/visited regimes, not
    candidate ledgers. Causal volatility and common random tapes are prepared
    once for IS. OOS uses distinct execution draws and fresh account state.
    """
    if not isinstance(policy, TargetPolicy):
        raise TypeError("policy must be TargetPolicy")
    if collision_policy != "stop_first":
        raise ValueError("OHLC fitting requires explicit collision_policy='stop_first'")
    if direction not in ("maximize", "minimize"):
        raise ValueError("direction must be maximize or minimize")
    for name, value, minimum in (("paths", paths, 1), ("seed", seed, 0),
                                  ("execution_seed", execution_seed, 0),
                                  ("generations", generations, 0), ("population", population, 2)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isfinite(train_fraction) or not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie between zero and one")
    sessions = tuple(sessions)
    if any(b.session <= a.session for a, b in zip(sessions, sessions[1:])):
        raise ValueError("sessions must be unique and chronological")
    split = int(len(sessions)*Fraction(str(train_fraction)))
    if not 0 < split < len(sessions):
        raise ValueError("split requires nonempty training and holdout histories")
    train, test = sessions[:split], sessions[split:]
    names = tuple(r.name for r in policy.sizing.regimes)
    if set(risk_bounds) != set(names) or set(target_bounds) != set(names):
        raise ValueError("bounds must cover exactly the policy regimes")
    bounds = np.asarray([risk_bounds[n] for n in names]+[target_bounds[n] for n in names], dtype=float)
    if (bounds.shape != (2*len(names), 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] < .01) or np.any(bounds[:, 1] <= bounds[:, 0])
            or not np.allclose(bounds*100, np.round(bounds*100), rtol=0, atol=1e-7)):
        raise ValueError("bounds must be cent-valued and satisfy .01 <= lower < upper")
    initial = np.array([r.risk_dollars for r in policy.sizing.regimes]+list(policy.targets))
    lo, width = bounds[:, 0], bounds[:, 1]-bounds[:, 0]
    if np.any(initial < lo) or np.any(initial > bounds[:, 1]):
        raise ValueError("initial policy must lie within bounds")
    count = paths if slippage else 1
    tapes = tuple(prepare_execution(train, instrument, slippage, seed=execution_seed, path=i)
                  if slippage else None for i in range(count))
    cache = {}
    sign = 1 if direction == "maximize" else -1

    def decode(x):
        v = np.round(lo+np.clip(x, 0, 1)*width, 2)
        return TargetPolicy(DollarPolicy(tuple(replace(r, risk_dollars=float(value))
                            for r, value in zip(policy.sizing.regimes, v[:len(names)]))), tuple(v[len(names):]))

    def score(candidate):
        if candidate not in cache:
            values, visited = [], set()
            targets = dict(zip(names, candidate.targets))
            for i, tape in enumerate(tapes):
                path = replay_prices(spec, train, candidate.sizing, targets, instrument, config,
                                     slippage=slippage, execution_seed=execution_seed, execution_path=i,
                                     _execution_tapes=tape, quantity=quantity, collision_policy=collision_policy,
                                     compensate_slippage=compensate_slippage)
                values.append(_objective(objective, path.replay))
                visited.update(d.regime for d in path.decisions)
            cache[candidate] = float(np.mean(values)), visited
            if progress:
                progress({"stage": "search", "evaluations": len(cache)})
        return sign*cache[candidate][0]

    score(policy)
    optimizer = CMAES((initial-lo)/width, .25, popsize=population,
                      bounds=(np.zeros(len(initial)), np.ones(len(initial))), seed=seed, max_gen=generations)
    optimizer.optimize(lambda x: score(decode(x)))
    selected = max(cache, key=lambda p: sign*cache[p][0])
    visited = cache[selected][1]
    selected = TargetPolicy(DollarPolicy(tuple(new if old.name in visited else old
        for old, new in zip(policy.sizing.regimes, selected.sizing.regimes))),
        tuple(new if name in visited else old for name, old, new in zip(names, policy.targets, selected.targets)))
    common = dict(slippage=slippage, quantity=quantity, paths=paths, execution_seed=execution_seed,
                  compensate_slippage=compensate_slippage, collision_policy=collision_policy, objective=objective)
    training = evaluate_prices(spec, train, selected, instrument, config, **common)
    # Distinct stream identifiers, even when IS and OOS share the supplied seed.
    held = evaluate_prices(spec, test, selected, instrument, config, path_offset=paths, **common)
    baseline = evaluate_prices(spec, test, policy, instrument, config, path_offset=paths, **common)
    return PriceFit(selected, training, held, baseline, len(cache), tuple(s.session for s in train),
                    tuple(s.session for s in test), direction)
