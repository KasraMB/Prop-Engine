# Performance

The dated trade-log engine is single-process Python with exact monetary
accounting. The separate summary simulator uses compiled Numba kernels.
They have different input contracts; speed is not a reason to substitute
closed summaries for missing intratrade evidence.

## Reproduce

```sh
python benchmarks/chronological.py
python benchmarks/portfolio.py --mode replay --events 1000 --recording search
python benchmarks/portfolio.py --mode book --events 10000
```

The chronological benchmark runs fixed-seed normal, rolling and synthetic
risk/target workloads three times, checks identical complete-output hashes and
unchanged requests, and reports median wall time. Peak traced Python allocation
is measured separately to avoid including tracing overhead in timing.

It also compares joint and separate-phase searches with identical trade-visit
caps, reporting actual visits, unused budget, IS/OOS scores and repeated-output
hashes. Timing is excluded from deterministic hashes. Neither architecture is
assumed to win; this public fixture measures wiring and cost, not general efficacy.

The portfolio benchmark measures recorded fills and valuation marks, including
event construction. It compares N and 10N events. Search recording avoids
retaining per-mark snapshots; accounting results and economic ledger events
remain available. Trace mode deliberately retains more detail.

CI retains benchmark artifacts and checks that the 10N streaming-mark workload
uses less than three times the peak Python allocation of N. This is a regression
gate for that workload, not a constant-memory claim for arbitrary histories:
account transitions, payouts and requested trace records can grow with input.

## Existing implementation choices

- Fixed costs and sizing budgets are converted to exact rational values once.
- Immutable result cash totals are cached; account state is never shared.
- Portfolio accounting drops closed lots and uses cached instrument tick values.
- Compiled account and model-calendar caches are bounded.
- Summary kernels reuse batch arrays.
- Bootstrap work is batched to bound temporary index storage.
- Objective selection does not suppress performance metrics.

Wall time depends on hardware, runtime and workload. Python allocation tracing
does not include all native/process/browser memory. The browser runs without
Numba, so Python timing is not a browser-performance guarantee.
