"""Representative serial rolling search. Optional real Chromium/Pyodide timing."""
import argparse
import csv
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
import json
from pathlib import Path
import sys
from threading import Thread
from time import perf_counter
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dashboard.replay import generate, run
from datetime import timedelta
from propfirm_engine import BacktestConfig, BracketHistory, DollarPolicy, RiskRegime, RollingConfig
from propfirm_engine.firms.lucidflex import replay_50k
from propfirm_engine.backtest import net_cash
from propfirm_engine.rolling import _prepare_rolling, _score_cash, rolling_backtest


def request(sessions=1000, generations=8, population=8):
    regimes = [dict(name="evaluation", phase="eval", risk_dollars=200),
        dict(name="payout_days_complete", phase="funded", risk_dollars=150, days_to_payout=0),
        dict(name="one_day_to_payout", phase="funded", risk_dollars=150, days_to_payout=1),
        dict(name="two_days_to_payout", phase="funded", risk_dollars=150, days_to_payout=2),
        dict(name="after_first_payout", phase="funded", risk_dollars=150, after_payout=True),
        dict(name="funded_in_profit", phase="funded", risk_dollars=200, in_profit=True),
        dict(name="funded_fallback", phase="funded", risk_dollars=150)]
    return dict(profile="lucidflex_50k_dll_off", mode="fit", objective="net_cash",
        csv=generate(dict(generator="iid", win_rate=.55, rr=1.5, stop_loss=100,
            trades_per_day=1, sessions=sessions, seed=7, start_date="2020-01-06"))["csv"],
        account=dict(eval_fee=105.2, reset_fee=105, contract_type="micro"),
        config=dict(cost_per_contract=1, cost_per_trade=0, payment_fee=0, initial_wallet=2000,
            approval_delay_hours=24, receipt_delay_hours=48, activation_delay_hours=.5, retry_delay_hours=0),
        regimes=regimes, risk_bounds={r["name"]: [50, 2000] for r in regimes},
        search=dict(generations=generations, population=population, seed=42),
        rolling=dict(window_sessions=40, stride_sessions=1),
        risk=dict(target_ruin_probability=.01, tail_probability=.05))


def brief(result, seconds, runtime):
    return dict(runtime=runtime, seconds=seconds, evaluations=result["evaluations"],
                score=result["score"], training_score=result["training_score"], policy=result["policy"],
                estimated_trade_visits=result["estimated_trade_visits"])


def browser_run(payload):
    from playwright.sync_api import sync_playwright
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(ROOT / "docs")))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            page.wait_for_function("ready || !document.getElementById('error').hidden", timeout=180000)
            if not page.locator("#error").is_hidden():
                raise RuntimeError(page.locator("#error").inner_text())
            data = page.evaluate("""async payload => {
                const started = performance.now();
                const result = await rpc('run', payload);
                return {result, seconds: (performance.now()-started)/1000};
            }""", payload)
            browser.close()
            return brief(data["result"], data["seconds"], "chromium_pyodide")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=1000)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--browser", action="store_true")
    args = parser.parse_args()
    payload = request(args.sessions, args.generations, args.population)
    if args.browser:
        print(json.dumps(browser_run(payload)), flush=True)
        return
    history = BracketHistory.from_records(csv.DictReader(StringIO(payload["csv"])))
    train, _ = history.split()
    rolling = RollingConfig(**payload["rolling"])
    spec = replay_50k(**payload["account"])
    config = BacktestConfig(1, timedelta(hours=24), timedelta(hours=48),
                            timedelta(minutes=30), initial_wallet=2000)
    policy = DollarPolicy(tuple(RiskRegime(**r) for r in payload["regimes"]))
    start = perf_counter()
    prepared = _prepare_rolling(train, rolling)
    preparation = perf_counter()-start
    start = perf_counter()
    compact = _score_cash(spec, prepared, policy, config, net_cash)
    compact_time = perf_counter()-start
    start = perf_counter()
    full = rolling_backtest(spec, train, policy, config, rolling=rolling, objective=net_cash)
    full_time = perf_counter()-start
    assert compact.score == full.score and compact.visited_regimes == full.visited_regimes
    tracemalloc.start()
    _score_cash(spec, prepared, policy, config, net_cash)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    print(json.dumps(dict(runtime="native", windows=len(prepared.parts), preparation_seconds=preparation,
        compact_seconds=compact_time, detailed_seconds=full_time, compact_peak_python_bytes=peak,
        score=compact.score, detailed_parity=True)), flush=True)
    if args.fit:
        start = perf_counter()
        result = run(payload)
        print(json.dumps(brief(result, perf_counter()-start, "native_fit")), flush=True)


if __name__ == "__main__":
    main()
