# Performance

The general strategy engine is single-process Python with exact price/accounting
arithmetic. Immutable columnar inputs share storage across folds and candidates.
Search drops detailed histories after callbacks while retaining economic events
and compact order identities. It has the same economics as research/trace modes,
not constant memory. Trial caches retain scores, not candidate replay ledgers.

The legacy summary Monte Carlo path uses fused Numba kernels and reusable batch
arrays. Its narrower summary assumptions do not make it an exact substitute for
general quote replay. Arbitrary Python strategies are not silently compiled or
restricted to the legacy model. Profiling justified cached tick fractions,
flat-position fast paths, bounded compilation caches and direct payout lookup;
it did not justify replacing the general coordinator with a second simulator.

## Reproduce

```sh
python benchmarks/research.py --cold --repeat 3
python benchmarks/portfolio.py --mode orders --events 200 --recording search
```

`--cold` selects a fresh ignored JIT cache. The matrix records preparation, first
call, warm median, separate peak traced Python allocation and cumulative process
peak RSS. RSS is not a per-case delta; Python tracing excludes untracked native
allocations. The complete-scenario case includes tape preparation; other replay
cases share prepared inputs. All inputs are synthetic and public.

## Measured baseline

2026-10-09, Windows AMD64, Intel Family 6 Model 158 Stepping 9, Python 3.12.7,
NumPy 2.2.6, scale 1, median of three warm runs. Twenty sessions of 48 quote
observations; concurrent replay uses three instruments. Search uses ten sessions;
the scenario case evaluates three full paths. Summary Monte Carlo uses 1,000
paths with 20 evaluation and 40 funded days.

| Workload | Warm seconds | Peak Python bytes |
| --- | ---: | ---: |
| Quote search | 0.1345 | 193,020 |
| Quote research | 0.1308 | 314,184 |
| Quote trace | 0.1357 | 569,376 |
| Concurrent assets | 0.2314 | 323,098 |
| Explicit bar path | 0.1336 | 184,531 |
| Complete strategy fit | 0.3027 | 169,091 |
| Full-path scenarios | 0.5954 | 647,116 |
| Summary Monte Carlo | 0.0157 | 622,540 |

Preparation took 1.017 seconds under allocation tracing, with 1,216,296 peak
Python bytes and 303,864 shared tape bytes. Fresh-cache summary compilation plus
first execution took 19.366 seconds; its cumulative process RSS peak reached
256,569,344 bytes. General replay cases reached 74,723,328 bytes before that.
Search's demonstrated benefit in this matrix is memory, not faster execution.
These are small reference workloads, not production latency guarantees.

## Regression gates

```sh
python benchmarks/research.py --repeat 3 --budget benchmarks/budgets/windows-py312.json
```

The budget rejects mismatched hardware, Python/NumPy versions or workload scale.
It passed a separate local run after calibration. Record a fresh baseline before
adding budgets for different machines; do not relax limits merely to pass a change.
CI records the matrix and gates search/research allocation and runtime ratios,
since shared runner hardware varies. Reference/accelerated, prepared/streamed,
recording-mode and deterministic-resume parity tests remain correctness gates.

`estimate_strategy_work` bounds input observation visits and compact trial counts
before a search. It includes warmup, validation and wallet counterparts. It cannot
bound arbitrary user callback allocations, order counts or wall time. Use the
recording sink for incremental external logs; cancellation discards partial
performance, and checkpoints preserve completed candidate scores only.
