"""Staged execution sensitivity and IS-only policy refitting on local prices.

Run: python -m Test_Strategies.execution_study --assets MES
Profiles are explicit uncalibrated assumptions, NOT estimates of Lucid fills.
"""
from argparse import ArgumentParser
from dataclasses import asdict, replace
from datetime import timedelta
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from propfirm_engine import Engine, Instrument, SlippageModel, TickDistribution
from propfirm_engine.price_fitting import evaluate_prices
from propfirm_engine.target_research import TargetPolicy
from propfirm_engine.execution import BacktestConfig
from propfirm_engine.firms.lucidflex import replay_50k
from propfirm_engine.risk import cash_risk_path
from Test_Strategies.open_long import ASSETS, POLICY, TARGETS, SOURCES, load_sessions, write_records


def scenarios(reference_range_ticks):
    """Illustrative inputs only; none of these probabilities come from fill logs."""
    hours = tuple(1.25 if h == 9 else 1.5 if h == 18 else 1. for h in range(24))
    central = SlippageModel(
        TickDistribution((.65, .30, .05)), TickDistribution((.25, .50, .20, .05)),
        TickDistribution((0, 0, .15, .25, .25, .20, .10, .05)),
        TickDistribution((0, 0, 0, .10, .15, .20, .20, .15, .10, .10)),
        reference_range_ticks, hour_multipliers=hours, label="moderate illustrative scenario; uncalibrated")
    return {
        "low": replace(central, market=TickDistribution((.9, .1)), stop=TickDistribution((.6, .35, .05)),
                       stress_probability=.01, hour_multipliers=(1.,)*24, label="low illustrative scenario; uncalibrated"),
        "moderate": central,
        "stressed": replace(central, stress_probability=.15, max_multiplier=12.,
                            hour_multipliers=tuple(h*1.5 for h in hours), target_trade_through_ticks=1,
                            label="stressed illustrative scenario with stricter limit fills; uncalibrated"),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Data"))
    parser.add_argument("--output", type=Path, default=Path("results/execution_study"))
    parser.add_argument("--assets", nargs="+", choices=sorted(ASSETS), default=["MES"])
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--paths", type=int, default=3, help="common execution draws per IS candidate")
    parser.add_argument("--evaluation-paths", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--max-sessions", type=int, help="last N eligible sessions; a bounded integration run, not the full study")
    args = parser.parse_args()
    if args.evaluation_paths < 1 or args.paths < 1 or args.seed < 0 or args.generations < 0 or args.population < 2:
        parser.error("positive path counts, nonnegative seed/generations, population >=2 required")
    if args.max_sessions is not None and args.max_sessions < 10:
        parser.error("max-sessions must be at least 10")
    args.output.mkdir(parents=True, exist_ok=True)
    initial = TargetPolicy(POLICY, tuple(TARGETS[r.name] for r in POLICY.regimes))
    records, models, fits, quality = [], {}, {}, {}
    for product in dict.fromkeys(args.assets):
        histories, quality[product] = load_sessions(args.data, product, execution_context=True)
        if args.max_sessions:
            histories = {k:v[-args.max_sessions:] for k,v in histories.items()}
        full, pv, tick, commission, *_ = ASSETS[product]
        instrument = Instrument(full, pv, tick)
        config = BacktestConfig(commission, timedelta(0), timedelta(0), timedelta(0))
        spec = replay_50k(eval_fee=105.2, reset_fee=105, contract_type="mini")
        split = int(len(histories["18:00"])*.7)
        # Use IS full-session ranges for both clocks; never normalize from OOS.
        ranges = np.concatenate([(s.ohlc[:,1]-s.ohlc[:,2])/tick for s in histories["18:00"][:split]])
        reference = max(1., float(np.median(ranges)))
        profiles = scenarios(reference)
        models[product] = {name:asdict(m) for name,m in profiles.items()}
        for entry, sessions in histories.items():
            print(f"{product} {entry}: fitting {split} IS / {len(sessions)-split} OOS sessions", flush=True)
            fitted = Engine().fit_prices(
                spec, sessions, instrument, config, policy=initial,
                risk_bounds={r.name:(100, 2000) for r in POLICY.regimes},
                target_bounds={r.name:(100, 4000) for r in POLICY.regimes},
                slippage=profiles["moderate"], paths=args.paths, seed=args.seed, execution_seed=args.seed,
                generations=args.generations, population=args.population, collision_policy="stop_first",
                progress=lambda p: print(f"  evaluated {p['evaluations']} IS policies", flush=True))
            fits[f"{product}_{entry}"] = {"selected_policy": asdict(fitted.policy), "evaluations": fitted.evaluations,
                                         "IS_score": fitted.in_sample.score, "train_sessions": len(fitted.train_sessions),
                                         "test_sessions": len(fitted.test_sessions)}
            stages = [("commission_only", initial, None, False),
                      ("uncompensated_slippage", initial, profiles["moderate"], False),
                      ("aware_brackets", initial, profiles["moderate"], True),
                      ("IS_refitted", fitted.policy, profiles["moderate"], True),
                      ("refitted_low", fitted.policy, profiles["low"], True),
                      ("refitted_stressed", fitted.policy, profiles["stressed"], True)]
            for name, policy, scenario, compensate in stages:
                evaluated = evaluate_prices(spec, sessions[split:], policy, instrument, config, slippage=scenario,
                                            paths=args.evaluation_paths, execution_seed=args.seed,
                                            path_offset=args.paths, compensate_slippage=compensate,
                                            collision_policy="stop_first")
                records.append({"asset": product, "execution_symbol": full, "entry_time": entry, "stage": name,
                                "sessions": len(sessions)-split, "first_session": str(sessions[split].session),
                                "last_session": str(sessions[-1].session), "score": evaluated.score,
                                "scope": evaluated.scope, "distributions": evaluated.distributions})
                prefix = f"{product}_{entry.replace(':','')}_{name}"
                path_rows = []
                for i, path in enumerate(evaluated.paths):
                    path_rows.append({"execution_path": args.paths+i, **asdict(cash_risk_path(path.replay))})
                write_records(args.output / f"{prefix}_paths.csv", path_rows)
                # One inspectable execution ledger per stage, not every search candidate.
                write_records(args.output / f"{prefix}_example_fills.csv", [asdict(d) for d in evaluated.paths[0].decisions])
                print(f"  {name}: OOS mean cash/day={evaluated.score:.2f}", flush=True)
        (args.output / f"{product}.json").write_text(json.dumps([r for r in records if r['asset']==product], indent=2, allow_nan=False)+"\n", encoding="utf-8")
    report = {"settings": {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              "profiles": models, "fits": fits, "data_quality": quality, "sources": SOURCES,
              "initial_policy": asdict(initial), "results": records,
              "implementation_sha256": {str(p):sha256(p.read_bytes()).hexdigest() for p in
                  [Path(__file__), Path(__file__).with_name("open_long.py"),
                   *sorted(Path("src/propfirm_engine").glob("*.py"))]},
              "caveats": [
                  "Scenario tick probabilities, stress rates and hourly multipliers are illustrative, not calibrated Lucid/platform fills.",
                  "Reference volatility is fitted on IS only. One fixed-size mini-equivalent position, same commissions and account lifecycle as open_long.",
                  "Every stage replays brackets and account mechanics. No spread charge is added on top of the modeled adverse tick increment.",
                  "70/30 chronological split; policy selected on IS only. This previously inspected historical OOS is research validation, not newly untouched evidence.",
                  "Repeated paths vary execution on a fixed price history; their spread is not market uncertainty or lifetime ruin risk.",
                  "Current-bar/future ranges never inform entry allowances. Warmup uses available completed bars, up to 60 supplied by the loader; missing warmup uses available prior replay bars or the reference range.",
                  "Stop allowance is an entry-time quantile, not a guaranteed loss cap. Unexpected fills can breach MLL; losses are not clipped.",
                  "Stressed scenario combines worse market/stop execution with one-tick trade-through limits; its change is not attributable solely to slippage.",
                  "Approval, receipt and activation are immediate scenarios; all payouts assumed approved. Existing data selection and micro-to-mini price-proxy limitations remain.",
                  "Stage comparisons are conditional, order-dependent changes, not an additive causal attribution. Stress comparisons hold the learned policy fixed.",
              ]}
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    summary = [{k:v for k,v in r.items() if k not in ("distributions", "scope")} for r in records]
    write_records(args.output / "summary.csv", summary)
    print(f"Results: {args.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
