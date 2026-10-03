"""Joint target-search dashboard adapter; no manual history or seeded solution."""
from dataclasses import replace
from datetime import date, timedelta

import numpy as np

from propfirm_engine import BacktestConfig, DollarPolicy, RiskConfig, RiskRegime
from propfirm_engine.firms.lucidflex import replay_50k
from propfirm_engine.ruin import _settled_cycles, ultimate_cycle_ruin
from propfirm_engine.target_research import (
    BracketModel, TargetPolicy, evaluate_targets, fit_targets, lucidflex_example, research_path,
)


def run_research(request, progress=None):
    # Shared JSON/scenario validation; import after the replay dispatcher is loaded.
    if __package__:
        from .replay import _finite, _jsonable, OBJECTIVES
    else:
        from replay import _finite, _jsonable, OBJECTIVES
    if request.get("profile") != "lucidflex_50k_dll_off":
        raise ValueError("Only LucidFlex 50K DLL-off is exposed in target research")
    spec = replay_50k(**request["account"])
    values = dict(request["config"])
    for name in ("approval_delay", "receipt_delay", "activation_delay", "retry_delay"):
        values[name] = timedelta(hours=_finite(values.pop(name+"_hours"), name))
    config = BacktestConfig(**values)
    risk = RiskConfig(**{"bankroll": config.initial_wallet, **request.get("risk", {})})
    settings = dict(request["model"])
    settings["start_date"] = date.fromisoformat(settings["start_date"])
    model = BracketModel(**settings)
    search = dict(request["search"])
    if set(search) != {"paths", "generations", "population", "seed", "holdout_seed"}:
        raise ValueError("Supply only the declared search controls; candidate seeds are not accepted")
    for name, low, high in (("paths",7,2000), ("generations",0,100), ("population",2,32),
                            ("seed",0,2**32-1), ("holdout_seed",0,2**32-1)):
        _finite(search[name], name, low, high, integer=True)
    if model.sessions > 10000:
        raise ValueError("Dashboard research supports up to 10,000 sessions per path")
    initial = request["initial"]
    for key in ("risk", "target"):
        _finite(initial[key], key, .01)
    regimes = [RiskRegime("evaluation", "eval", initial["risk"]),
               RiskRegime("funded_build", "funded", initial["risk"], days_to_payout=5, after_payout=False)]
    if request.get("regime_set", "compact") == "detailed":
        regimes[1] = replace(regimes[1], in_profit=False)
        regimes += [RiskRegime(f"funded_days_left_{n}", "funded", initial["risk"], days_to_payout=n)
                    for n in (0,1,2,3,4)]
        regimes += [RiskRegime("funded_after_payout", "funded", initial["risk"], after_payout=True),
                    RiskRegime("funded_in_profit", "funded", initial["risk"], in_profit=True)]
    elif request.get("regime_set", "compact") != "compact":
        raise ValueError("regime_set must be compact or detailed")
    regimes.append(RiskRegime("funded_fallback", "funded", initial["risk"]))
    policy = TargetPolicy(DollarPolicy(tuple(regimes)), (initial["target"],)*len(regimes))
    names = [r.name for r in regimes]
    objective = request.get("objective", "net_cash_per_day")
    if objective not in OBJECTIVES:
        raise ValueError("Unsupported objective")
    choices = request.get("choices")
    bound = request["bounds"]
    maximum = search["generations"]*search["population"]+2

    def report(update):
        if progress:
            progress({**update, "maximum_evaluations": maximum})

    fitted = fit_targets(spec, model, config, policy=policy,
        risk_bounds={n: bound["risk"] for n in names}, target_bounds={n: bound["target"] for n in names},
        risk_choices=None if choices is None else {n: choices["risk"] for n in names},
        target_choices=None if choices is None else {n: choices["target"] for n in names},
        objective=OBJECTIVES[objective], risk=risk, progress=report, **search)
    report({"stage": "holdout"})
    tapes = np.random.default_rng(np.random.SeedSequence([search["holdout_seed"],1])).random(
        (search["paths"]-int(.7*search["paths"]), model.sessions))
    # Only after selection: never pass the named example as a search candidate.
    reference = evaluate_targets(spec, model, lucidflex_example(), config, tapes,
                                 objective=OBJECTIVES[objective], risk=None)
    trace = research_path(spec, model, fitted.policy, config, tapes[0])
    cycles, excluded = [], 0
    for i, tape in enumerate(tapes):
        full = research_path(spec, model, fitted.policy, replace(config, initial_wallet=None), tape)
        complete, missing = _settled_cycles(full.replay)
        cycles.extend(complete)
        excluded += missing
        report({"stage": "ruin", "completed_paths": i+1, "paths": len(tapes)})
    ultimate = (ultimate_cycle_ruin(cycles, bankroll=risk.bankroll,
                target=risk.target_ruin_probability, confidence=risk.confidence,
                seed=search["holdout_seed"]) if cycles and risk.bankroll is not None else None)
    return _jsonable({"schema_version":1, "mode":"model_search", "request":request,
        "model":model, "spec":spec, "config":config, "objective":objective, "fit":fitted,
        "initial_policy":policy, "reference_policy":lucidflex_example(), "reference_holdout":reference,
        "reference_used_for_selection":False, "risk":fitted.holdout.risk,
        "cycle_approximation":ultimate, "cycle_records":cycles,
        "excluded_unsettled_or_open_accounts":excluded, "representative_path":trace,
        "representative_path_selection":"First holdout tape, not best path",
        "assumptions":trace.replay.assumptions,
        "scope":"70/30 independent model-path split; not historical strategy IS/OOS"})
