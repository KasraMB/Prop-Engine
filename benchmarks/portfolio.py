"""Synthetic portfolio throughput and retained-memory scaling."""

import argparse
import json
import platform
from statistics import median
from datetime import datetime, timezone
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.events < 2 or args.events % 2:
        parser.error("events must be an even integer >= 2")
    if args.repeat < 1:
        parser.error("repeat must be positive")
    run(100)
    rows = []
    for count in (args.events, args.events * 10):
        seconds = [run(count)["seconds"] for _ in range(args.repeat)]
        rows.append({"events": count, "median_seconds": median(seconds),
                     "events_per_second": count / median(seconds),
                     "peak_python_bytes": run(count, True)["peak_python_bytes"]})
    print(json.dumps({"python": platform.python_version(), "platform": platform.platform(),
                      "includes_event_creation": True, "results": rows}, indent=2))
