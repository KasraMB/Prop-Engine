# Validation dashboard (testing only)

A small **standalone** tool to drive the engine and check its behavior. Two pages:

- **Interactive** (`/`) — hand-enter trades day by day and watch one account's rule
  decisions, computed by the real `propfirm_engine.reference` oracle.
- **Monte Carlo** (`/montecarlo`) — pick trade-stream **generator** parameters
  (win-rate, RR, generator type + its dependence params) and an account, run the
  *full batch engine* (generator → `preprocess` → `Engine.run` over thousands of
  resampled attempts → statistics/renewal), and see the results as **headline
  numbers** plus a set of **charts**: outcome breakdown, payout-count distribution,
  net-payoff & return-on-fee histograms, attempt-duration & time-to-first-payout
  histograms, a reward-vs-time scatter, sample equity curves, and an optional
  win-rate × RR **surface**. Uses the real firm accounts (with payout schemas) and
  user-set fees, so the fee/renewal metrics are well posed. Optionally enable the
  feasibility projection to see attempts wither (`CAPPED_OUT`). An **Optimize
  sizing** toggle runs the Tier-1 CMA-ES `walk_forward` (fit on a TRAIN stream,
  reported on an independent HELD-OUT stream — nested OOS) and charts the baseline
  vs the fitted policy side by side, with the fitted per-stage risk multipliers and
  the OOS improvement in a dedicated optimizer panel.

It is deliberately *not* part of the pipeline — it only **imports** the engine.
Every number shown is the engine's own.

## Run it

```bash
python dashboard/server.py
# then open http://localhost:8000  (Interactive)
#           http://localhost:8000/montecarlo  (Monte Carlo explorer)
```

No third-party dependencies — standard library only, plus the engine in `src/`
(the scripts add `src/` to the path themselves). Set `PORT=1234` to change port.

Prefer the terminal? `python dashboard/selfcheck.py` prints a handful of
hand-checkable eval/funded scenarios and the engine's verdict on each.

## What you can do

1. **Browse** the implemented firms → account types → sizes (top-left). Only the
   built-in *Test Firm* demo accounts exist today; real firm configs will appear
   here automatically once they're added under `propfirm_engine/firms/`.
2. Pick a **stage** (Eval or Funded) and click **Start / Reset account**.
3. **Add a trade** by entering its P&L (and optionally its intraday floating low).
   Watch the balance, day P&L, the **Max Loss Limit** floor, and the distance to
   the profit target update.
4. **Next day →** closes the current day and starts a new one — this is when
   end-of-day rules fire (the MLL trails your EOD balance, winning days are
   counted, consistency is checked, payouts are released).
5. The status badge tells you **PASSED / FAILED / in-progress** (eval) or shows
   **payouts** and **COMPLETE** (funded).

## The test account

As requested: a **50K** account whose eval is exactly a **\$2,000 end-of-day Max
Loss Limit** (a drawdown that trails your EOD balance by \$2k and is checked at
day close) and a **\$3,000 profit target**, and nothing else. The **100K** size
scales both. The **funded** stage of the same account adds winning-day,
consistency, and payout mechanics so those can be exercised too:

- **3 qualifying days** each ≥ \$150,
- **cycle profit ≥ \$500** to request,
- **consistency ≤ 40%** (no single day may exceed 40% of the cycle's profit),
- payout = 90% of the released amount (cap \$2,000), up to **5 payouts**.

Payouts **fire automatically at a day's close** once every condition holds — that
is exactly what the engine's `PayoutSchema` models — so you validate the payout
logic by engineering days that should (or shouldn't) trigger one and watching the
balance drop and the qualifying-day counter reset.

## How it stays faithful

Each action re-sends the full day/trade list; the server builds the path arrays,
runs `_ReferenceSim` over them, and reads its internal state (equity, the trailing
floor, `max_day_pnl`, qualifying days, cycle profit, payouts, the terminal exit
code). Because the last (in-progress) day is closed by the engine's end-of-path
logic, the panel shows its **projected end-of-day** state — add a recovery trade
and it updates, which is exactly how an EOD limit behaves. Nothing here
re-implements a rule; the engine makes every decision.

## GitHub Pages (browser build)

`docs/` is the static Pages site — both pages run the **real engine in the browser**
via Pyodide (pure Python, no server). `index.html` is the interactive validator
(reference simulator); `montecarlo.html` is the Monte Carlo explorer (full batch
engine + optimizer). The `@njit` kernels run as plain Python through a tiny `numba`
shim — bit-identical to the compiled path (the Level-1 parity gate and the golden
hash both hold under the shim).

Rebuild the Monte Carlo Python bundle after engine changes with:

```
python dashboard/build_pages.py
```

It copies the exact `propfirm_engine` modules the pipeline imports into `docs/py/`,
writes the `numba` shim and `mc_engine.py`, and regenerates `py/manifest_mc.json`
(the file list the page fetches). Keep `ENGINE_MODULES` in that script in sync if
`montecarlo.py`'s imports change.
