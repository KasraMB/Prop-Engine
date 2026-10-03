"""Reproducible model-based risk/target search; no historical validation.

Run: python benchmarks/target_policy.py --paths 100 --generations 20
The named example is measured AFTER selection, never seeded into the search.
"""
import argparse
from dataclasses import asdict, replace
from datetime import timedelta
import json
from math import isfinite
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from propfirm_engine import BacktestConfig, DollarPolicy, RiskConfig
from propfirm_engine.firms.lucidflex import replay_50k
from propfirm_engine.target_research import (
    BracketModel, TargetPolicy, evaluate_targets, fit_targets,
    lucidflex_example, research_path,
)


def run_experiment(*, paths=100, sessions=30, generations=20, seed=42, mu=0.0, sigma=1000.0,
                   bankroll=2000.0, ruin_target=.01, confidence=.95):
    spec = replay_50k(eval_fee=105.20, reset_fee=105.00, contract_type="micro")
    config = BacktestConfig(0, timedelta(0), timedelta(0), timedelta(0))
    model = BracketModel(sessions=sessions, mu=mu, sigma=sigma)
    risk = RiskConfig(bankroll=bankroll, target_ruin_probability=ruin_target, confidence=confidence)
    example = lucidflex_example()
    baseline = TargetPolicy(DollarPolicy(tuple(replace(r, risk_dollars=500)
                            for r in example.sizing.regimes)), (500, 500, 500))
    names = [r.name for r in baseline.sizing.regimes]
    risk_choices = {n: (500, 1000, 1500, 2000) for n in names}
    target_choices = {n: (150, 500, 1000, 1500, 2000, 3000) for n in names}
    fitted = fit_targets(spec, model, config, policy=baseline,
        risk_bounds={n: (500, 2000) for n in names}, target_bounds={n: (150, 3000) for n in names},
        risk_choices=risk_choices, target_choices=target_choices,
        paths=paths, seed=seed, holdout_seed=seed+1, generations=generations, population=8, risk=risk)
    test = np.random.default_rng(np.random.SeedSequence([seed+1, 1])).random((paths-int(paths*.7), sessions))
    example_holdout = evaluate_targets(spec, model, example, config, test, risk=risk)
    trace = research_path(spec, model, fitted.policy, config, test[0])
    return {"model": asdict(model), "spec": asdict(spec), "config": asdict(config),
            "baseline_policy": asdict(baseline), "risk_choices": risk_choices,
            "target_choices": target_choices, "generations": generations, "population": 8,
            "fit": asdict(fitted), "named_example_holdout": asdict(example_holdout),
            "example_used_for_selection": False, "representative_holdout_path": asdict(trace),
            "zero_drift_zero_cost_seven_session_reference": {
                "first_attempt_pass_probability": (4/7)**2,
                "first_payout_by_seventh_session_probability": (4/7)**2 * .5 * (40/43)**4,
                "first_receipt_on_fastest_path": 1170,
                "scope": "closed-form check of fastest path, not a full lifecycle EV"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=100)
    parser.add_argument("--sessions", type=int, default=30)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mu", type=float, default=0)
    parser.add_argument("--sigma", type=float, default=1000)
    parser.add_argument("--bankroll", type=float, default=2000,
                        help="Ruin-analysis bankroll; performance still uses an unlimited wallet")
    parser.add_argument("--ruin-target", type=float, default=.01, help="Target fraction, e.g. .01 for 1%%")
    parser.add_argument("--confidence", type=float, default=.95)
    args = parser.parse_args()
    result = run_experiment(**vars(args))
    print(json.dumps(exportable(result), allow_nan=False, indent=2))


def exportable(value):
    """Keep unbounded profile thresholds explicit without emitting invalid JSON."""
    if isinstance(value, dict):
        return {key: exportable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [exportable(item) for item in value]
    if isinstance(value, timedelta):
        return {"seconds": value.total_seconds()}
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float) and not isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else None
    return value


if __name__ == "__main__":
    main()
