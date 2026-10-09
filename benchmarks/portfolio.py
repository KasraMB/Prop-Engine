"""Synthetic portfolio throughput and retained-memory scaling."""

import argparse
import json
import platform
from statistics import median
from datetime import date, datetime, timedelta, timezone
from time import perf_counter
import tracemalloc

from propfirm_engine.events import Fill
from propfirm_engine.instruments import Instrument
from propfirm_engine.portfolio import Book


def run(count, trace=False):
    if trace:
        tracemalloc.start()
    book = Book([Instrument("ES", 50, .25)])
    at = datetime(2026, 1, 5, tzinfo=timezone.utc)
    start = perf_counter()
    for seq in range(count):
        book.apply(Fill(at, "ES", 1 if seq % 2 == 0 else -1, 100, seq=seq))
    seconds = perf_counter() - start
    peak = tracemalloc.get_traced_memory()[1] if trace else None
    if trace:
        tracemalloc.stop()
    assert book.equity == 0 and not book.snapshot().positions
    return {"events": count, "seconds": seconds, "events_per_second": count / seconds,
            "peak_python_bytes": peak}


def replay_run(count, trace=False, record=False, strategy=False):
    from propfirm_engine import BacktestConfig, Engine, Marks, Market, Order, Quote, QuoteModel
    from propfirm_engine.firms.lucidflex import replay_50k
    spec = replay_50k(eval_fee=105.2, reset_fee=105, contract_type="mini")
    config = BacktestConfig(0, timedelta(0), timedelta(0), timedelta(0))
    at = datetime(2026, 1, 5, 15, tzinfo=timezone.utc)

    def events():
        yield Fill(at, "ES", 1, 100)
        for seq in range(1, count - 1):
            yield Marks(at, (("ES", 100),), seq=seq)
        yield Fill(at, "ES", -1, 100, seq=count - 1)

    class Hold:
        def on_market(self, context, market):
            if market.seq == 0:
                return [Order("entry", "ES", 1)]
            if market.seq == count - 2:
                return [Order("exit", "ES", -1, reduce_only=True)]

    if trace:
        tracemalloc.start()
    start = perf_counter()
    options = dict(sessions=(date(2026, 1, 5),), fidelity="observed_marks",
                   max_mark_age=timedelta(0), liquidation_fee=0, trace=record)
    if strategy:
        markets = (Market(at, (Quote("ES", 100, 100, 100),), seq=i) for i in range(count))
        result = Engine().replay_strategy(spec, markets, [Instrument("ES", 50, .25)], config,
                                          Hold(), models={"ES": QuoteModel()}, **options).result
    else:
        result = Engine().replay_events(spec, events(), [Instrument("ES", 50, .25)], config,
                                        mark_fills=True, **options)
    seconds = perf_counter() - start
    peak = tracemalloc.get_traced_memory()[1] if trace else None
    if trace:
        tracemalloc.stop()
    assert result.book.equity == 50_000 and result.replay.failed_attempts == 0
    return {"events": count, "seconds": seconds, "events_per_second": count / seconds,
            "peak_python_bytes": peak}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--mode", choices=("book", "replay", "strategy"), default="book")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    if args.events < 2 or args.events % 2:
        parser.error("events must be an even integer >= 2")
    if args.repeat < 1:
        parser.error("repeat must be positive")
    if args.record and args.mode == "book":
        parser.error("--record requires --mode replay or strategy")
    execute = run if args.mode == "book" else lambda n, trace=False: replay_run(
        n, trace, args.record, strategy=args.mode == "strategy")
    execute(100)
    rows = []
    for count in (args.events, args.events * 10):
        seconds = [execute(count)["seconds"] for _ in range(args.repeat)]
        rows.append({"events": count, "median_seconds": median(seconds),
                     "events_per_second": count / median(seconds),
                     "peak_python_bytes": execute(count, True)["peak_python_bytes"]})
    print(json.dumps({"python": platform.python_version(), "platform": platform.platform(),
                      "mode": args.mode, "record": args.record,
                      "includes_event_creation": True, "results": rows}, indent=2))
