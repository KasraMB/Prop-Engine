"""Synthetic release workloads with preparation, compilation and retention costs."""
import argparse
import ctypes
from datetime import date, datetime, timedelta, timezone
import gc
import json
import os
from pathlib import Path
import platform
from statistics import median
import sys
import tempfile
from time import perf_counter
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def peak_rss():
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]

        info = Counters()
        info.cb = ctypes.sizeof(info)
        kernel, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(info), info.cb):
            raise OSError("cannot read process memory counters")
        return info.peak
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value*1024)


def hardware():
    import numpy as np
    return dict(system=platform.system(), machine=platform.machine(),
                cpu=os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
                python=".".join(platform.python_version_tuple()[:2]), numpy=np.__version__)


def measure(call, repeat):
    start = perf_counter()
    call()
    first = perf_counter()-start
    times = []
    for _ in range(repeat):
        start = perf_counter()
        call()
        times.append(perf_counter()-start)
    gc.collect()
    tracemalloc.start()
    call()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return dict(first_seconds=first, warm_median_seconds=median(times), peak_python_bytes=peak,
                process_peak_rss_bytes=peak_rss())


def workloads(scale):
    import numpy as np
    from propfirm_engine import (
        BacktestConfig, Bar, Basket, Engine, Instrument, Market, MarketTape, Order, Parameter,
        Quote, QuoteModel, RunConfig, IIDGenerator, bar_quotes, evaluate_scenarios,
    )
    from propfirm_engine.firms.lucidflex import replay_50k
    from propfirm_engine.data import preprocess

    engine = Engine()
    spec = replay_50k(eval_fee=105.2, reset_fee=105, contract_type="mini")
    config = BacktestConfig(0, timedelta(0), timedelta(0), timedelta(0))
    start = datetime(2026, 9, 1, 14, tzinfo=timezone.utc)
    offsets = [i for i in range(50) if (start+timedelta(days=i)).weekday() < 5][:20]
    sessions = tuple(date(2026, 9, 1)+timedelta(days=i) for i in offsets)
    instruments = tuple(Instrument(s, 1, 1) for s in ("X", "Y", "Z"))

    def prepare(count=1, bars=False, seed=0):
        rng = np.random.default_rng(seed)
        if bars:
            stream = bar_quotes((Bar(start+timedelta(days=d, minutes=i),
                start+timedelta(days=d, minutes=i, seconds=59), "X", 10000, 10020, 9990, 10010, seq=i)
                for d in offsets for i in range(12*scale)), instruments[0], path="olhc",
                half_spread_ticks=1, liquidity=None)
        else:
            stream = (Market(start+timedelta(days=d, seconds=i), tuple(
                Quote(x.symbol, p, p, p) for x in instruments[:count]), seq=i)
                for d in offsets for i in range(48*scale)
                for p in (10000+(i % 12)*int(rng.choice((-1, 1))),))
        return MarketTape(stream, instruments[:count], sessions=sessions)

    class Policy:
        def __init__(self, params, seed):
            self.quantity = max(1, round(params.get("quantity", 1)))

        def on_market(self, context, market):
            if context.warmup or not context.available:
                return
            held = context.book.positions if context.book else ()
            orders = []
            if market.seq % 12 == 0 and not held:
                count = max(0, context.contract_limit//self.quantity)
                orders = [Order(f"{market.at}:{q.symbol}:in", q.symbol, self.quantity)
                          for q in market.quotes[:count]]
            elif market.seq % 12 == 9 and held:
                orders = [Order(f"{market.at}:{p.symbol}:out", p.symbol, -p.quantity, reduce_only=True)
                          for p in held]
            return [Basket(str(market.at), tuple(orders))] if len(orders) > 1 else orders

    def setup_for(data, recording="search"):
        return dict(models={i.symbol: QuoteModel(fee=.1) for i in data.instruments},
            fidelity=data.fidelity, liquidation_fee=.1, max_mark_age=timedelta(hours=8), recording=recording)

    tracemalloc.start()
    start_prep = perf_counter()
    quotes, multi, bars = prepare(), prepare(3), prepare(bars=True)
    dataset = preprocess(IIDGenerator(win_rate=.45, rr=1.5, trades_per_day=6).generate(150, 7).rows)
    preparation = dict(seconds=perf_counter()-start_prep, peak_python_bytes=tracemalloc.get_traced_memory()[1],
                       tape_bytes=sum(t.nbytes for t in (quotes, multi, bars)))
    tracemalloc.stop()

    def replay(data, mode="search"):
        result = engine.replay_strategy(spec, data, data.instruments, config, Policy({}, 0),
            sessions=data.sessions, **setup_for(data, mode))
        assert result.result.fills > 0
        return result

    compact, research = replay(quotes), replay(quotes, "research")
    assert compact.result.book == research.result.book
    assert compact.result.replay.net_cash == research.result.replay.net_cash
    assert compact.order_counts == research.order_counts
    mc = RunConfig(n_paths=1000*scale, L_eval=20, L_funded=40, seed=7,
                   size_base=100, intraday_mode="summary_approximation")
    calls = {
        "quote_search": lambda: replay(quotes),
        "quote_research": lambda: replay(quotes, "research"),
        "quote_trace": lambda: replay(quotes, "trace"),
        "concurrent_assets": lambda: replay(multi),
        "bar_path": lambda: replay(bars),
        "strategy_fit": lambda: engine.fit_strategy(spec, quotes.view(0, 10), config, Policy,
            baseline={"quantity": 1}, space={"quantity": Parameter("continuous", 1, 3)},
            setup=lambda seed: setup_for(quotes), generations=2, population=4, seed=7),
        "full_scenarios": lambda: evaluate_scenarios(spec, (prepare(seed=s) for s in range(3)), config,
            Policy, params={"quantity": 1}, setup=lambda seed: setup_for(quotes), sample_kind="independent_model"),
        "summary_mc": lambda: engine.run(spec.account, dataset, mc),
    }
    return calls, preparation


def check_budget(report, budget):
    if report["hardware"] != budget["hardware"] or report["scale"] != budget["scale"]:
        raise ValueError("performance budget hardware or workload scale does not match")
    for name, limits in budget["limits"].items():
        for metric, maximum in limits.items():
            actual = report["results"][name][metric]
            if actual > maximum:
                raise ValueError(f"performance budget exceeded: {name}.{metric} = {actual} > {maximum}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--budget", type=Path)
    args = parser.parse_args()
    if args.repeat < 1 or args.scale < 1:
        parser.error("repeat and scale must be positive")
    if args.cold:
        directory = ROOT / "results"
        directory.mkdir(exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="jit-benchmark-", dir=directory)
    calls, preparation = workloads(args.scale)
    results = {}
    for name, call in calls.items():
        results[name] = measure(call, args.repeat)
        print(f"completed {name}", file=sys.stderr, flush=True)
    report = dict(hardware=hardware(), scale=args.scale, repeat=args.repeat, preparation=preparation,
        fresh_jit_cache=args.cold, results=results,
        notes=["process RSS peaks are cumulative, not per-case deltas", "Python allocation peaks exclude untracked native allocations",
               "summary_mc uses the explicit legacy summary approximation; other cases use the general replay core",
               "full_scenarios includes preparation; other replays share prepared inputs", "single process, no multicore search"])
    print(json.dumps(report, indent=2))
    if args.budget:
        check_budget(report, json.loads(args.budget.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
