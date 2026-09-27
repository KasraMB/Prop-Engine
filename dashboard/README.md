# Strategy replay dashboard

Updated 2026-09-26. The homepage calls `Engine.backtest` or `Engine.fit` through
`replay.py`; it contains no separate accounting or optimization implementation.

## Run locally

```sh
python -m pip install -e ".[dev]"
python dashboard/server.py
```

Open `http://localhost:8000`. Set `PORT` to change the loopback port.
The local endpoint is `POST /api/replay`; it accepts the same JSON scenario used
by the browser worker. It is a local research server, not a public API service.

1. Generate a synthetic history, enter manual trades, or upload a six-column CSV:
   `entry_at,exit_at,session,stop_loss,take_profit,won`.
2. Set the contract type, dated account fees, trading costs, wallet and processing delays.
3. Inputs follow the [ideal-fill input contract](../docs/BRACKET_BACKTEST.md);
   there is no acknowledgment checkbox.
4. Configure ordered regimes and search bounds before inspecting OOS results.
5. Optimize on the first 70% of whole sessions; report the final 30%. Both selected
   and initial policies start fresh accounts/wallets on the same OOS period.
6. Export the full JSON before changing inputs; edits invalidate the displayed result.

The fixed-policy alternative replays the entire history, clearly labeled **not OOS**.
The cumulative cash chart shows the selected policy over IS and OOS together, with
a vertical dashed line at OOS start. Cashflows are concatenated for display only:
OOS still starts with a fresh account and wallet. The gray initial-policy OOS curve
uses the same IS cash offset for comparison. Headline metrics remain OOS-only.
Headline economics are received payouts minus account fees, per elapsed calendar day
or in total. Trading balance, outstanding payouts and unvalued live handoff are distinct.
After live handoff, the replay starts another paid evaluation under the existing
retry delay and wallet constraints, without counting a failure. Approved receipts
from the old attempt still arrive on schedule. This is a research lifecycle choice;
the transferred live account is not traded or valued.
The ledger display is capped at 500 events; JSON retains all events and the full model
configuration. The original CSV is not included; preserve it alongside its export hash.
Generated histories can be downloaded as CSV; their generator parameters and realized
statistics are included in result provenance.

## Synthetic inputs and account trace

The synthetic input controls win probability, gross reward/risk ratio, base stop
per contract, trades per session, session count, start date and seed. Choose IID,
regime-switching (spread and persistence), or stochastic volatility (persistence
and volatility noise). These call the existing Python trade generators. Volatility
scales stop and target together, preserving the selected ratio. Realized sample
statistics can differ from generator parameters. Sessions skip weekends, not exchange
holidays. Changing parameters invalidates the prior history and result.

Open `/trace.html` for the same account, cost and policy controls with fixed-policy
execution. Manual wins/losses specify gross stop/target amounts per contract,
an Eastern entry time and duration. The manual editor supports weekday daytime
trades from 09:30 through 16:45; CSV inputs can cover other valid account hours.
Each edit replays the supplied history through `Engine.backtest`, including session
finalization. Use the event selector to inspect the state before or after each trade,
session close, evaluation pass or payout event. The chart shows balance and loss
floor for the selected attempt and phase. There is no separate manual rule engine.
The trace can also consume synthetic histories and uploaded CSVs.

Local helper endpoints `POST /api/generate` and `POST /api/manual` construct validated
histories; `POST /api/replay` is the shared engine adapter.

Dashboard limits: 5 MB, 20,000 trades, 32 regimes, 100 generations, 32 candidates per
generation and two million candidate-trade evaluations. Use the Python API for larger
jobs and custom callable objectives. Arbitrary Python text is never executed by the UI.

## Pages and reproducible builds

```sh
python dashboard/build_pages.py
```

The build normalizes line endings, mirrors every canonical package module, copies
the shared adapter and replay assets, and writes content hashes in
`docs/py/manifest_replay.json`. The worker refuses a mixed or incomplete bundle.
The homepage runs Python in a worker with Pyodide 0.26.4, its NumPy package and
tzdata 2025.2. Trade contents stay in the browser; dependencies come from external
CDNs/PyPI. No backend or credential is needed. Cancel terminates the worker and
reloads the runtime. The local server mode does not expose cancellation.

`Tests` checks Python 3.11–3.13, generated-file synchronization and real Chromium
integration. After those checks pass on `main`, `Pages` publishes the `docs/`
artifact using GitHub Actions. Configure the repository's Pages source as
**GitHub Actions**, not the older `gh-pages` branch. Deployment identifies the
tested source commit in `version.json`. Failed tests never trigger publication.

## Tests

```sh
python -m pytest -q
python -m pip install playwright
python -m playwright install chromium
# Set RUN_BROWSER_TESTS=1 in your shell, then:
python -m pytest tests/test_dashboard_browser.py -q
```

Browser tests start temporary loopback servers, generate/upload histories,
fit/report/export and enter manual payout scenarios. They compare real browser
results and trace events against canonical Python, check validation recovery,
worker cancellation, bundle integrity and narrow-screen overflow.
They require network access for runtime dependencies.
Without the environment opt-in they skip. `BROWSER_SCREENSHOT_DIR` optionally
collects screenshots; `BROWSER_BASE_URL` tests an already deployed static site.

## Research scope

As of 2026-09-26, the old manual and Monte Carlo dashboards and their independent
adapters are removed. The resampled `Engine.run` Python API remains available.
Removing the UI acknowledgment does not change execution assumptions or establish
real-market fidelity. Fees, holidays, discretionary enforcement and live value
still require the treatment described in the execution contract.
