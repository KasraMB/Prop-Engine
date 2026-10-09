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


def replay_run(count, trace=False, record=False, strategy=False, prepared=False, recording=None,
               churn=False):
    from propfirm_engine import BacktestConfig, Engine, Marks, Market, MarketTape, Order, Quote, QuoteModel
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
            if churn and market.seq < count-1:
                return [Order(str(market.seq), "ES", 1 if market.seq % 2 == 0 else -1)]
            if market.seq == 0:
                return [Order("entry", "ES", 1)]
            if market.seq == count - 2:
                return [Order("exit", "ES", -1, reduce_only=True)]

    instruments = [Instrument("ES", 50, .25)]
    data = None
    prep_start = perf_counter()
    if prepared:
        data = MarketTape((Market(at, (Quote("ES", 100, 100, 100),), seq=i) for i in range(count)),
                          instruments, sessions=(date(2026, 1, 5),))
    prep_seconds = perf_counter()-prep_start
    if trace:
        tracemalloc.start()
    start = perf_counter()
    options = dict(sessions=(date(2026, 1, 5),), fidelity="observed_marks",
                   max_mark_age=timedelta(days=1), liquidation_fee=0, trace=record, recording=recording)
    if strategy:
        markets = data if data is not None else (Market(at, (Quote("ES", 100, 100, 100),), seq=i) for i in range(count))
        result = Engine().replay_strategy(spec, markets, instruments, config,
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
            "peak_python_bytes": peak, "preparation_seconds": prep_seconds,
            "tape_bytes": data.nbytes if data is not None else 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--mode", choices=("book", "replay", "strategy", "prepared", "orders"), default="book")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--recording", choices=("search", "research", "trace"))
    args = parser.parse_args()
    if args.events < 2 or args.events % 2:
        parser.error("events must be an even integer >= 2")
    if args.repeat < 1:
        parser.error("repeat must be positive")
    if args.record and args.mode == "book":
        parser.error("--record requires --mode replay or strategy")
    execute = run if args.mode == "book" else lambda n, trace=False: replay_run(
        n, trace, args.record, strategy=args.mode in ("strategy", "prepared", "orders"),
        prepared=args.mode in ("prepared", "orders"), recording=args.recording, churn=args.mode == "orders")
    execute(100)
    rows = []
    for count in (args.events, args.events * 10):
        runs = [execute(count) for _ in range(args.repeat)]
        seconds = [row["seconds"] for row in runs]
        rows.append({"events": count, "median_seconds": median(seconds),
                     "events_per_second": count / median(seconds),
                     "peak_python_bytes": execute(count, True)["peak_python_bytes"],
                     "preparation_seconds": median(row.get("preparation_seconds", 0) for row in runs),
                     "tape_bytes": runs[0].get("tape_bytes", 0)})
    print(json.dumps({"python": platform.python_version(), "platform": platform.platform(),
                      "mode": args.mode, "record": args.record, "recording": args.recording,
                      "includes_event_creation": args.mode not in ("prepared", "orders"), "results": rows}, indent=2))
