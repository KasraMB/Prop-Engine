# Propfirm Engine

A Python engine for replaying trading strategies through futures prop-firm
accounts and optimizing account-state-dependent position sizing.

**One account at a time, repeated attempts, explicit cashflows, untouched
out-of-sample reporting.** LucidFlex 50K DLL-off is the first reference profile;
firm rules and lifecycle settings remain separate from execution and optimization.

## Status

General order and portfolio execution is being developed under the
[research engine plan](docs/RESEARCH_ENGINE_PLAN.md). Recorded fills and ordered
portfolio marks now support arbitrary exits, partials and concurrent instruments
through `Engine.replay_events`. See the [event replay API](docs/EVENT_REPLAY.md)
for its explicit observation and liquidation assumptions. Causal strategy
callbacks and quote-based orders are available through `Engine.replay_strategy`;
see the [strategy API](docs/STRATEGY_REPLAY.md). `Engine.fit_strategy` adds shared
market tapes, declared parameter search, chronological OOS, rolling windows and
walk-forward refits. See [strategy fitting](docs/STRATEGY_FITTING.md). Additional
input adapters include external opportunities, explicit bar/trade scenarios and
atomic or legged multi-instrument orders. See [market inputs](docs/MARKET_INPUTS.md).
The remaining release gates are tracked in the roadmap.

The bracket API implements the agreed **sequential stop-or-target model**.
It reuses the existing rule interpreter, feasibility projection, payout ledger,
and CMA-ES optimizer. It does not reconstruct market paths from closed trades.

This is **not a claim of complete contractual or live-execution fidelity**.
The consistency cushion is intentionally excluded; fills and processing delays
are explicit scenarios. Live-account valuation, discretionary reviews, holidays
and platform-specific agreements need additional treatment.
Read the [execution contract](docs/BRACKET_BACKTEST.md) before relying on results.

An explicit minute-OHLC approximation also supports state-dependent dollar
brackets at fixed contract count, gap fills and timed session exits. See the
[historical price replay guide](docs/PRICE_REPLAY.md) for the API and the
DuckDB-backed 09:30 versus 18:00 long-only experiment.

Price replay also supports explicit state-dependent slippage scenarios,
slippage-aware brackets, and joint dollar risk/target fitting through
`Engine.fit_prices`. See [execution scenarios and fitting](docs/PRICE_REPLAY.md#execution-scenarios-and-price-policy-fitting).

## Install

Requires Python 3.11+.

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Research state-dependent risk and targets

Historical replay keeps recorded brackets fixed. For **model-based** joint
risk/target search, use the separate `propfirm_engine.target_research` API:

```python
from datetime import timedelta
from propfirm_engine import BacktestConfig, RiskConfig
from propfirm_engine.firms.lucidflex import replay_50k
from propfirm_engine.target_research import BracketModel, fit_targets, lucidflex_example

spec = replay_50k(eval_fee=105.20, reset_fee=105, contract_type="micro")
config = BacktestConfig(0, timedelta(0), timedelta(0), timedelta(0))
policy = lucidflex_example()  # explicit baseline, not a discovered result
names = [r.name for r in policy.sizing.regimes]
fit = fit_targets(
    spec, BracketModel(sessions=60, mu=0, sigma=1000), config,
    policy=policy,
    risk_bounds={n: (50, 2000) for n in names},
    target_bounds={n: (150, 3000) for n in names},
    paths=100, seed=42, holdout_seed=43,
    risk=RiskConfig(bankroll=2000, target_ruin_probability=0.01),
    objective=lambda result: result.net_cash_per_day,
)
print(fit.policy, fit.holdout.score, fit.holdout.score_standard_error)
print(fit.holdout.risk["required_bankroll"])
print(fit.holdout.risk["confidence_status"])
```

Each changed bracket gets a drift-aware hit probability; the existing account
engine handles consistency, winning days, drawdown, payouts, fees and restarts.
The assumed clock is one completed bracket per available session, **not simulated
market passage time**. The 70/30 split here is independent model paths, not
historical IS/OOS. Read [assumptions and verification](docs/ANALYTICAL_MODEL.md#joint-risk-and-target-research)
before interpreting the cash/day result. For a separate historical minute-bar
test with explicit execution assumptions, see [price replay](docs/PRICE_REPLAY.md).

To search from a flat $500/$500 policy without supplying the example as a seed:

```sh
python benchmarks/target_policy.py --paths 100 --sessions 30 --generations 20
```

## Backtest a trade history

Each row describes one completed trade, including its original stop and target
in **gross dollars per contract**. Fixed and varying reward-to-risk ratios are
both supported. Fees are supplied separately and must not already be deducted.

```csv
entry_at,exit_at,session,stop_loss,take_profit,won
2026-09-01T10:00:00-04:00,2026-09-01T10:05:00-04:00,2026-09-01,100,200,true
2026-09-02T10:00:00-04:00,2026-09-02T10:07:00-04:00,2026-09-02,100,300,false
```

Use your complete history, not these two illustrative rows, for fitting:

```python
import csv
from datetime import timedelta

from propfirm_engine import (
    BacktestConfig, BracketHistory, DollarPolicy, Engine, RiskRegime,
)
from propfirm_engine.firms.lucidflex import replay_50k

with open("trades.csv", newline="", encoding="utf-8") as stream:
    history = BracketHistory.from_records(csv.DictReader(stream))

# Dated fee scenario, NOT a current checkout quote.
spec = replay_50k(eval_fee=105.20, reset_fee=105.00, contract_type="micro")

# Illustrative costs and elapsed-time delays; replace with your actual inputs.
config = BacktestConfig(
    cost_per_contract=1.00,          # round trip, for each executed contract
    approval_delay=timedelta(days=1),
    receipt_delay=timedelta(days=2),
    activation_delay=timedelta(minutes=30),
    initial_wallet=2_000,
)

# First matching regime wins; place specific regimes before phase fallbacks.
policy = DollarPolicy((
    RiskRegime("evaluation", "eval", 200),
    RiskRegime("one_day_to_payout", "funded", 150, days_to_payout=1),
    RiskRegime("funded_in_profit", "funded", 200, in_profit=True),
    RiskRegime("funded", "funded", 100),
))

engine = Engine()
replay = engine.backtest(spec, history, policy, config)
print(replay.net_cash, replay.net_cash_per_day, replay.status)
```

Risk budgets include modeled trading costs. Positions round down to whole
contracts and obey the active drawdown buffer and scaling limit. If the buffer
cannot support one contract, the attempt ends as `CAPPED_OUT`. A deliberate
zero-risk or sub-contract policy budget is a skip, not an account breach.

The reward for each trade follows its historical ratio and executed quantity.
A $200 risk budget does not guarantee exactly $200 executed risk after rounding.
This API does **not** independently change historical take-profit targets.

## Optimize on 70%; report on 30%

```python
fit = engine.fit(
    spec, history, config,
    policy=policy,
    risk_bounds={regime.name: (50, 500) for regime in policy.regimes},
    train_fraction=0.70,
    generations=20,
    population=12,
    seed=42,
)

print(fit.policy)                          # frozen, selected using IS only
print(fit.score)                           # OOS score, never training score
print(fit.out_of_sample.net_cash_per_day)  # observed OOS cash / elapsed days
```

The split uses complete chronological sessions. OOS starts with a fresh account
and wallet; it is never used to choose candidates. Regimes not visited during IS
retain their declared baseline risk. Configure regime definitions and search
bounds **before** inspecting OOS results.

The default objective maximizes net external cash per elapsed calendar day:
received payouts minus account/reset/activation fees, divided by the observation
period. It includes waiting time and weekends. This is an empirical performance
measurement, **not an unbiased estimate of population EV** or a valuation of
unreceived payouts/live accounts.

Live handoff ends that simulated-funded account, then the replay buys a fresh
evaluation using the existing retry delay and wallet checks. Handoffs do not count
as failures; old approved payouts still arrive on schedule. Continuing this way
is a research lifecycle assumption, not a valuation or simulation of the live account.

Custom objectives receive the complete replay result:

```python
fit = engine.fit(
    spec, history, config,
    policy=policy,
    risk_bounds={regime.name: (50, 500) for regime in policy.regimes},
    objective=lambda result: -result.net_cash_per_day,
    direction="minimize",
    seed=42,
)
```

## Rolling historical starts

Keep the market sequence intact and reset the account and wallet at each selected
session start. Each window uses the same repeated-attempt engine, including retries
after breaches and live handoffs. No trades are shuffled or dates rebased.

```python
from propfirm_engine import RollingConfig, RiskConfig

# Five sessions fits both partitions of the small example above.
rolling = RollingConfig(window_sessions=5, stride_sessions=1)
starts = engine.rolling_backtest(spec, history, policy, config, rolling=rolling)
print(starts.summary)  # per-window distributions, first-evaluation outcomes and rates

fit = engine.fit(
    spec, history, config,
    policy=policy,
    risk_bounds={r.name: (50, 500) for r in policy.regimes},
    rolling=rolling, seed=42,
    risk=RiskConfig(bankroll=2000, target_ruin_probability=0.01),
)
print(fit.score)  # mean OOS-window net cash / calendar day
print(fit.out_of_sample_rolling.summary)
print(fit.out_of_sample_rolling.risk)
```

Fitting splits sessions 70/30 **before** building windows. Each partition must
contain at least one complete window. The objective is the equal-weight mean of
per-window scores; a custom callable still receives a single `BacktestResult`.
The full chronological IS/OOS replays remain in `fit.in_sample` / `fit.out_of_sample`.
Incomplete tail starts are counted and excluded. Unresolved evaluations within
complete windows are reported separately, not treated as failures.

Overlapping starts are dependent historical scenarios, not independent future
simulations. Their cashflows must not be summed as portfolio profit; reported
percentiles describe window outcomes, not confidence intervals. See the
[rolling evaluation contract](docs/BRACKET_BACKTEST.md#rolling-historical-starts).

## Cash risk and bankroll

The dashboard's **Cash risk & bankroll** panel reports variance, standard
deviation, P1/P5/P10/P25/P50/P75/P90/P95/P99, loss frequency, loss VaR and expected
shortfall, cash drawdowns and underwater duration, payout timing, and required
starting capital. Set the ruin target and worst-tail percentage in Run & evaluate;
the initial wallet supplies the bankroll for the ruin-frequency calculation.
Use rolling starts for distributions; a single replay shows observed path facts
only. **Print results** creates a print-friendly view; the JSON export retains
all risk inputs, definitions, curve points and per-path records.

Finite-horizon funding failure means being unable to fund the next required evaluation, reset or activation
within the chosen horizon, not merely breaching a prop account. Capital needs
come from each path's maximum cash deficit with wallet constraints removed, so
stopping early cannot make an underfunded policy appear cheap to finance.

Historical-window frequencies are descriptive, not independent future estimates.
Independent model holdouts additionally expose probability intervals and a
confidence-supported bankroll, or an explicit insufficient-sample result.
Risk reporting never changes candidate selection. See the
[risk contract](docs/BRACKET_BACKTEST.md#cash-risk-and-bankroll-reporting) for
formulas, sample requirements, finite-horizon limits and the separate research API.

### Multi-path and ultimate ruin

In **Multi-path and ultimate ruin analysis**, enable the full-engine bootstrap,
choose the path count, sessions per path, mean block length and seed. All payouts
remain in the bankroll. After fitting, the frozen policy is simulated using only
the OOS source sessions. Whole-session blocks retain trade order and stop/target
outcomes; account state and delayed receipts persist across attempts within each path.

```python
from propfirm_engine import RuinConfig

_, holdout = history.split(0.70)
ruin = engine.ruin(
    spec, holdout, fit.policy, config,
    simulation=RuinConfig(paths=20, sessions=60, mean_block=5, seed=1729),
    risk=RiskConfig(bankroll=2000, target_ruin_probability=0.01),
)
print(ruin["risk"]["ruin_probability"])  # finite-horizon bootstrap frequency
print(ruin["cycle_approximation"])       # ultimate, conditional on a separate model
```

The **ultimate** calculation is a separately labelled approximation: it resamples
complete, settled account cash cycles independently, with full payout retention.
It reports a Monte Carlo range, a confidence range including unresolved paths,
and a **sufficient** bankroll bound for the requested ultimate risk—not a claimed
minimum. Positive-drift simulations stop only with a bounded remaining tail;
reaching the computational limit does not make survivors safe.

This cycle model does **not** preserve receipt overlap between accounts, dependence
between successive fees or market cycles. Open/unsettled accounts are excluded
and counted, which can bias the sampled cycle law. Its confidence bounds cover
simulation error, not these model errors. The full engine's ultimate probability
remains unidentified by finite simulations. See the
[ruin model contract](docs/BRACKET_BACKTEST.md#multi-path-and-ultimate-ruin).

## API map

| Task | Entry point |
| --- | --- |
| Import sequential stop/target records | `BracketHistory.from_records(...)` |
| Replay dated account attempts | `Engine.backtest(...)` |
| Evaluate ordered historical starting windows | `Engine.rolling_backtest(...)` |
| Cash distributions, ruin and bankroll | `RiskConfig`, `cash_risk_path(...)`, `risk_report(...)` |
| Dated lifecycle Monte Carlo and ultimate cycle approximation | `Engine.ruin(...)`, `RuinConfig`, `ultimate_cycle_ruin(...)` |
| Fit dollar regimes and evaluate held-out history | `Engine.fit(...)` |
| Resampled Monte Carlo research on trade summaries | `Engine.run(...)` |
| Drift-aware barrier approximations | [Analytical API](docs/ANALYTICAL_MODEL.md) |
| Explicit payout event accounting | [Payout lifecycle](docs/PAYOUT_LIFECYCLE.md) |

The legacy Monte Carlo path remains a separate research mode.
Its strict capability guard is unchanged: arbitrary closed summaries cannot
certify intraday compliance. Choosing bracket replay explicitly asserts the
ideal sequential stop/target contract; it does not relabel arbitrary MAE data.

## Dashboard

Open the [live dashboard](https://kasramb.github.io/Prop-Engine/) or run locally:

```sh
python dashboard/server.py
# Open http://localhost:8000
```

To discover **risk and profit targets together**, open
[Target search](https://kasramb.github.io/Prop-Engine/research.html) and click
**Discover policy**. No trade entry or CSV is required. Set drift, volatility,
path count, horizon and account costs; the optimizer searches independent risk
and target amounts for evaluation, funded build and remaining funded trades.
An expanded regime set distinguishes payout proximity, profit and payout history.
The default starts flat at $500/$500, not from the named example. Dollar grids
include exact eligibility thresholds; continuous-dollar search is also available.

Results show the discovered policy, independent 30% model holdout performance,
cash percentiles, variance, finite-horizon funding risk and a separately labelled
ultimate-cycle approximation. Selecting an objective changes policy selection,
not report coverage: total cash and cash/day statistics and both selected risk
and target amounts remain available. Cash/day means, uncertainty and percentiles
also remain in the baseline/reference comparison and JSON export.
The named example is compared **after selection**,
never supplied as a candidate. Search can miss it or underperform it; no global
optimum is claimed. These are bracket-model experiments, **not historical strategy
validation**. Each available session resolves one eventual bracket; market holding
time and price-level tick constraints are not simulated. Defaults use zero costs
and delays as an explicit idealized scenario. Export JSON to retain all inputs.

Upload the bracket CSV described above or generate a synthetic history with
configurable win rate, reward/risk ratio, stop size, sessions, trades per session,
seed and return model. Set fees/costs/delays and ordered dollar-risk regimes.
The default workflow
fits on 70% of complete sessions and shows only the final 30% as headline OOS
performance, with the initial policy evaluated on the same holdout. A fixed-policy
full-history replay is available separately and is explicitly not labeled OOS.
Download the full JSON for scenario inputs, policy, split, fingerprints and ledgers.

The hosted replay runs in a browser worker; trade contents are not uploaded.
Local mode sends them to your loopback Python server. Runtime dependencies load
from external CDNs on Pages. The synthetic example is a UI demonstration, not
market data or evidence of strategy profitability. The account trace uses the same
engine and interface: enter manual wins/losses or replay a CSV, then step through
trades, session closes and payout events. The Monte Carlo dashboard has been removed;
the resampled Python research API remains available. No acknowledgment checkbox is
required. See [dashboard setup and deployment](dashboard/README.md).

## Repository guide

- `src/propfirm_engine/`: canonical package, generic rules, execution and optimization.
- `tests/`: boundary, independent hand-calculation, leakage and regression tests.
- `docs/`: maintained API contracts, model limitations and browser distribution.
- `Test_Strategies/`: standalone input producers; strategies are not engine API features.
- `benchmarks/`: reproducible performance/model-comparison programs, not scratch scripts.
- `dashboard/`: chronological replay, holdout fitting and account trace with a shared adapter.

See [contributing](CONTRIBUTING.md), the [design overview](docs/DESIGN.md) and
the [execution contract](docs/BRACKET_BACKTEST.md) for supported behavior,
validation requirements and remaining limitations.

Firm-rule evidence must come from the firm's own website/help center or
published attachments. Official facts, user decisions and assumptions stay distinct.
