"""Serial dashboard-workload benchmark; no downloads, workers or file writes.

Run: python benchmarks/chronological.py
Reports median of three runs, then peak traced Python allocations in a separate
warmed run. Result hashes include full ledgers, metrics, policies and provenance.
Timings are workload-specific observations, not CI performance thresholds.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dashboard.replay import generate, run


def scenarios():
    account = dict(eval_fee=105.2, reset_fee=105, contract_type="micro")
    config = dict(cost_per_contract=0, cost_per_trade=0, payment_fee=0,
                  initial_wallet=2000, approval_delay_hours=0, receipt_delay_hours=0,
                  activation_delay_hours=.5, retry_delay_hours=0)
    history = generate(dict(generator="iid", win_rate=.55, rr=1.5, stop_loss=100,
        trades_per_day=4, sessions=60, seed=7, start_date="2026-01-05"))["csv"]
    normal = dict(profile="lucidflex_50k_dll_off", mode="fit", csv=history,
        account=account, config=config, objective="net_cash_per_day",
        regimes=[dict(name="evaluation", phase="eval", risk_dollars=100),
                 dict(name="funded", phase="funded", risk_dollars=100)],
        risk_bounds={name: [50,2000] for name in ("evaluation", "funded")},
        search=dict(generations=3, population=6, seed=42))
    rolling = dict(normal, rolling=dict(window_sessions=10, stride_sessions=3))
    target = dict(profile="lucidflex_50k_dll_off", mode="model_search",
        account=account, config=dict(config, activation_delay_hours=0),
        objective="net_cash_per_day",
        model=dict(mu=0, sigma=1000, sessions=20, start_date="2026-01-05"),
        search=dict(paths=20, generations=3, population=6, seed=42, holdout_seed=43),
        initial=dict(risk=500, target=500), regime_set="compact",
        bounds=dict(risk=[500,2000], target=[150,3000]),
        choices=dict(risk=[500,1000,1500,2000], target=[150,500,1000,1500,2000,3000]),
        risk=dict(target_ruin_probability=.01))
    return {"normal": normal, "rolling": rolling, "target": target}


def main():
    for name, request in scenarios().items():
        original = deepcopy(request)
        timings, hashes = [], []
        for _ in range(3):
            start = perf_counter()
            result = run(request)
            timings.append(perf_counter() - start)
            hashes.append(sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest())
        assert len(set(hashes)) == 1 and request == original
        tracemalloc.start()
        run(request)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        print(json.dumps(dict(name=name, seconds=median(timings), peak_bytes=peak,
                              result_sha256=hashes[0])), flush=True)


if __name__ == "__main__":
    main()
