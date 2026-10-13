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

## Rolling dashboard search

The dashboard's cash and cash/day objectives use a compact scoring path through
the same account state machine. It retains cash events and visited regimes, not
full trade/session ledgers or per-window distribution reports for every candidate.
Selected-policy IS/OOS and baseline reports still use detailed replay. Custom
objectives receive full results; they are not silently routed through this path.
Python callers can use the exported `net_cash` / `net_cash_per_day` objectives;
the default `Engine.fit` cash/day objective is also recognized.

Whole-dollar cash uses Python integers, not rounded floats. Nonintegral amounts
remain exact fractions. The decimal conversion cache is bounded. Contract sizing
uses exact floor division for the existing integer-contract constraint, and tests
compare it with the general feasibility projector. Desired sizes are shared only
within one candidate; account buffer and scaling caps are always evaluated live.

Reproduce a workload with seven regimes, 40-session windows, spacing one, 70/30
holdout and eight generations of eight candidates:

```powershell
python benchmarks/rolling.py --sessions 1000 --fit
python benchmarks/rolling.py --sessions 1000 --browser
```

The browser command requires Playwright and Chromium and runs the real published
Pyodide worker. It excludes runtime download/boot from search timing. These are
synthetic inputs, not the user's MNQ CSV. Native tracing measures additional Python
allocations after input/window preparation, not process or browser memory.

An observed 1,000-session fixture has 661 training windows and 26,440 trade visits
per candidate. Before this change, the measured detailed candidate took 6.60 s;
compact prepared scoring measured 1.33 s, with identical score/visited regimes.
Additional traced Python allocation fell from about 830 KB for detailed rolling
output to 53 KB for compact scoring. These are workload-specific observations,
not universal speed or constant-memory guarantees. The complete search remains
millions of visits and can still take minutes. Window spacing is never changed
automatically to obtain a speedup.

The complete eight-by-eight fixture was also run: 66 candidates and approximately
1.79 million estimated trade visits, taking 173 s in desktop Python and 277 s in
Chromium/Pyodide on the test machine. Other validation processes were active, so
these timings are indicative rather than isolated hardware comparisons. Both
returned IS score 2739.5773354765506 and OOS score 3655.537619731801. Selected
continuous risk values differed between numerical runtimes; identical seeds do
not guarantee cross-platform CMA-ES trajectories or identical parameter vectors.
Compact versus full-path policy equality is tested within the same runtime.

## Existing implementation choices

The calendar-window benchmark reuses prepared sessions, trade groups and hashes
across serial policy replays. Preparation shares immutable observations. Candidate
search caches retain scores and policies, not event ledgers. Final traces are
opt-in. Bankroll metrics require an additional unlimited-wallet replay only when
the specified wallet constrained an outcome; this extra work is intentional.

The chronological benchmark also reports sized trade logs with 1,000 and 10,000
observations. Input construction is excluded from this replay-allocation measure.
Marks are immutable slotted objects, replay retains lifecycle events rather than
each valuation, and each history's fingerprint is streamed and cached. Work still
grows linearly with marks processed; phase-search trade-visit budgets are not
equal wall-time budgets when trades contain different numbers of observations.

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
