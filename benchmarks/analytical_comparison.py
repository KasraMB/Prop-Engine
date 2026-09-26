"""Reproducible synthetic calibration -> paper analytics -> existing Engine.

Run from repository root:
python benchmarks/analytical_comparison.py --output results.json --trades-dir local_trades
No real market data, estimated edge, optimized policy or firm certification.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from propfirm_engine import Engine, ExitCode, IIDGenerator, RunConfig, preprocess
from propfirm_engine.analytical import (
    barrier_pass_probability, estimate_session_moments, expected_funding_cost,
)
from propfirm_engine.firms.lucidflex import build_account
from propfirm_engine.rules import ConsistencyGateRule
from propfirm_engine.statistics import wilson_ci

RULE_INPUTS = dict(target=3000.0, drawdown=2000.0, freeze_level=100.0)
FEE_SCENARIO = dict(entry_fee=105.20, reset_fee=105.0, activation_fee=0.0)
SOURCES = {
    "paper": "7260819.pdf, Fernandez (2026), equations 3, 5, 9",
    "evaluation": "https://support.lucidtrading.com/en/articles/12945790-lucidflex-evaluation-account",
    "drawdown": "https://support.lucidtrading.com/en/articles/12945815-lucidflex-drawdown",
    "consistency": "https://support.lucidtrading.com/en/articles/12945805-lucidflex-consistency-percentage",
    "dll": "https://support.lucidtrading.com/en/articles/16226050-lucidflex-customization",
}


def outcome_summary(outcome):
    n = outcome.n_attempts
    passed = int(np.count_nonzero(outcome.code == ExitCode.PASSED))
    censored = int(np.count_nonzero(outcome.code == ExitCode.TIMED_OUT))
    breached = int(np.count_nonzero((outcome.code >= 10) & (outcome.code < 20)))
    if passed + censored + breached != n:
        raise RuntimeError("unexpected evaluation terminal state")
    p = passed / n
    return {
        "n_paths": n, "passed": passed, "breached": breached, "censored": censored,
        "pass_probability_by_horizon": p,
        "conditional_mc_wilson95": list(wilson_ci(passed, n)),
        "eventual_probability_identification_bounds": [p, (passed + censored) / n],
        "mean_observed_trading_sessions": float(outcome.total_trading_days.mean()),
        "iid_retry_cost_scenario": (expected_funding_cost(p, **FEE_SCENARIO) if p > 0 and not censored else None),
        "execution_model": outcome.execution_model, "intraday_mode": outcome.intraday_mode,
        "approximation_reasons": list(outcome.approximation_reasons),
    }


def evaluate_case(dataset, *, risk, cost, n_paths, horizon, seed):
    moments = estimate_session_moments(dataset, size_base=risk, trade_cost=cost)
    inputs = dict(**RULE_INPUTS, mu=moments.mu, sigma=moments.sigma)
    continuous = barrier_pass_probability(**inputs)
    eod = barrier_pass_probability(**inputs, floor_updates=1)
    # A sensitivity scenario, NOT an assertion that the firm's clock equals trades/day.
    trade_clock = barrier_pass_probability(**inputs, floor_updates=1,
                                            breach_checks=moments.mean_trades_per_day)
    preset = build_account(50_000)
    full = replace(preset, phases=(preset.phases[0],), eval_fee=FEE_SCENARIO["entry_fee"])
    phase = full.phases[0]
    barrier_only = replace(full, name="50K barrier-only diagnostic",
                           phases=(replace(phase, rules=tuple(
                               r for r in phase.rules if not isinstance(r, ConsistencyGateRule))),))
    config = RunConfig(n_paths=n_paths, L_eval=horizon, seed=seed,
                       size_base=risk, trade_cost=cost, intraday_mode="summary_approximation")
    engine = Engine()
    # Same seed/day paths and source data; no policy is fitted or selected.
    base = engine.run(barrier_only, dataset, config, policy_params=[1.0])
    with_consistency = engine.run(full, dataset, config, policy_params=[1.0])
    b, f = outcome_summary(base), outcome_summary(with_consistency)
    delta = ((with_consistency.code == ExitCode.PASSED).astype(float)
             - (base.code == ExitCode.PASSED).astype(float))
    paired_se = float(delta.std(ddof=1) / math.sqrt(n_paths)) if n_paths > 1 else None
    return {
        "moments": asdict(moments),
        "paper_continuous": asdict(continuous), "paper_eod_continuous_breach": asdict(eod),
        "paper_trade_clock_sensitivity": asdict(trade_clock),
        "engine_barrier_only": b, "engine_with_consistency": f,
        "engine_minus_paper_percentage_points": 100 * (b["pass_probability_by_horizon"] - eod.probability),
        "paired_consistency_effect_percentage_points": 100 * float(delta.mean()),
        "paired_consistency_effect_mc_se_percentage_points": None if paired_se is None else 100 * paired_se,
        "paper_iid_retry_cost_scenario": expected_funding_cost(eod.probability, **FEE_SCENARIO),
        "account_fingerprints": [base.fingerprint, with_consistency.fingerprint],
    }


def run_comparison(*, n_days=5000, n_paths=20000, horizon=600, trades_dir=None):
    if n_days < 2 or n_paths < 2 or horizon < 1:
        raise ValueError("need >=2 calibration days/paths and positive horizon")
    cases = []
    for j, win_rate in enumerate((.48, .50, .52)):
        seed = 260926 + j
        generator = IIDGenerator(win_rate, 1.0, trades_per_day=20, intraday_excursion=0.0)
        stream = generator.generate(n_days, seed)
        dataset = preprocess(stream.rows)
        payload = {
            "return_r": dataset.ret, "trade_low_r": dataset.trade_low, "day": dataset.day,
            "exit_timestamp_ns": dataset.exit_timestamps.astype("int64"),
        }
        digest = hashlib.sha256()
        for key, values in payload.items():
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(values).tobytes())
        source = {
            "generator": asdict(stream.provenance), "n_trades": dataset.n_trades,
            "n_days": dataset.n_days, "canonical_arrays_sha256": digest.hexdigest(),
            "first_rows": [
                {"timestamp": str(stream.rows["timestamp"][i]), "return_r": float(dataset.ret[i]),
                 "mae_r": float(-dataset.trade_low[i])} for i in range(min(5, dataset.n_trades))
            ],
        }
        if trades_dir is not None:
            trades_dir = Path(trades_dir)
            trades_dir.mkdir(parents=True, exist_ok=True)
            path = trades_dir / f"iid-{j}-seed-{seed}.npz"
            np.savez_compressed(path, **payload)
            source["trade_archive"] = str(path)
        for risk in (100.0, 200.0):
            result = evaluate_case(dataset, risk=risk, cost=1.0, n_paths=n_paths,
                                   horizon=horizon, seed=742019)
            cases.append(dict(name=f"p={win_rate:.2f}; risk={risk:.0f}; win_MAE=0", source=source, **result))
        if win_rate == .50:
            # Identical closed returns, different winner MAE: moment matching cannot identify it.
            adverse_stream = replace(generator, intraday_excursion=.5).generate(n_days, seed)
            adverse = preprocess(adverse_stream.rows)
            assert np.array_equal(adverse.ret, dataset.ret)
            result = evaluate_case(adverse, risk=100.0, cost=1.0, n_paths=n_paths,
                                   horizon=horizon, seed=742019)
            cases.append(dict(name="p=0.50; risk=100; win_MAE=0.5",
                              source={"same_closed_returns_as": source["canonical_arrays_sha256"],
                                      "generator": asdict(adverse_stream.provenance)},
                              **result))
    return {
        "experiment": "2026-09-26 synthetic LucidFlex 50K DLL-off evaluation comparison",
        "python": platform.python_version(), "numpy": np.__version__,
        "n_days": n_days, "n_paths": n_paths, "horizon_sessions": horizon, "engine_seed": 742019,
        "contract_inputs": RULE_INPUTS, "fee_scenario": FEE_SCENARIO, "sources": SOURCES,
        "limitations": [
            "Synthetic IID +/-1R trades, not historical strategy performance or fitted real edge.",
            "USD1 per-trade cost is a declared generic scenario, not verified instrument commission.",
            "Fee amounts are the user's dated screenshot scenario, not refreshed checkout prices.",
            "Moments use all generated net-dollar session totals; no account survivor selection.",
            "Paper models eventual barrier-only success; engine reports success within the stated horizon.",
            "Engine explicitly uses closed-trade summary approximation, not ordered-MTM certification.",
            "Continuous breach is user-confirmed; official drawdown page does not specify unrealized equity.",
            "Trade-clock correction is sensitivity only, not a firm-rule interpretation.",
            "Wilson intervals measure conditional Monte Carlo error, not uncertainty in moments or model.",
            "MAE and discrete trade overshoots are not identified by mu and sigma.",
            "Consistency excluded from paper/barrier comparator, added separately using the chosen 50% rule.",
            "No funded payouts, live value, calendar billing, optimization or finite-wallet profitability is claimed.",
            "IID retry cost is a derived fee scenario, not an independent engine cashflow validation.",
        ],
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-days", type=int, default=5000)
    parser.add_argument("--n-paths", type=int, default=20000)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trades-dir", type=Path)
    args = parser.parse_args()
    report = run_comparison(n_days=args.n_days, n_paths=args.n_paths,
                            horizon=args.horizon, trades_dir=args.trades_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for case in report["cases"]:
        m, b, f = case["moments"], case["engine_barrier_only"], case["engine_with_consistency"]
        print(f"{case['name']}: mu={m['mu']:.3f} sigma={m['sigma']:.3f}; "
              f"paper={case['paper_eod_continuous_breach']['probability']:.5f}; "
              f"engine={b['pass_probability_by_horizon']:.5f}; "
              f"consistency={f['pass_probability_by_horizon']:.5f}; "
              f"censored={b['censored']}/{f['censored']}", flush=True)


if __name__ == "__main__":
    main()
