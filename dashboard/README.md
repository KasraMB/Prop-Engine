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

1. Upload a six-column bracket CSV: `entry_at,exit_at,session,stop_loss,take_profit,won`.
2. Set the contract type, dated account fees, trading costs, wallet and processing delays.
3. Confirm the [ideal-fill input contract](../docs/BRACKET_BACKTEST.md).
4. Configure ordered regimes and search bounds before inspecting OOS results.
5. Optimize on the first 70% of whole sessions; report the final 30%. Both selected
   and initial policies start fresh accounts/wallets on the same OOS period.
6. Export the full JSON before changing inputs; edits invalidate the displayed result.

The fixed-policy alternative replays the entire history, clearly labeled **not OOS**.
Headline economics are received payouts minus account fees, per elapsed calendar day
or in total. Trading balance, outstanding payouts and unvalued live handoff are distinct.
The ledger display is capped at 500 events; JSON retains all events and the full model
configuration. The original CSV is not included; preserve it alongside its export hash.

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

Browser tests start temporary loopback servers, upload a CSV, fit/report/export,
compare real browser results against canonical Python, check validation recovery
and narrow-screen overflow. They require network access for runtime dependencies.
Without the environment opt-in they skip. `BROWSER_SCREENSHOT_DIR` optionally
collects screenshots; `BROWSER_BASE_URL` tests an already deployed static site.

## Legacy research interfaces

- `/interactive.html`: manual trace explorer with explicit summary-approximation opt-in.
- `/montecarlo.html`: resampled Monte Carlo research with its own legacy sizing fitter.

Neither is the chronological replay workflow. Their approximation warnings remain
active. Fees, holidays, discretionary enforcement, live value and execution realism
are not made verified simply by displaying a result. Read the execution contract.
