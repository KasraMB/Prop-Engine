# Propfirm Engine

A Python engine for replaying trading strategies through futures prop-firm
accounts and optimizing account-state-dependent position sizing.

**One account at a time, repeated attempts, explicit cashflows, untouched
out-of-sample reporting.** LucidFlex 50K DLL-off is the first reference profile;
firm rules and lifecycle settings remain separate from execution and optimization.

## Status

The chronological API implements the agreed **sequential stop-or-target model**.
It reuses the existing rule interpreter, feasibility projection, payout ledger,
and CMA-ES optimizer. It does not reconstruct market paths from closed trades.

This is **not a claim of complete contractual or live-execution fidelity**.
The consistency cushion is intentionally excluded; fills and processing delays
are explicit scenarios. Live-account valuation, discretionary reviews, holidays
and platform-specific agreements need additional treatment.
Read the [execution contract](docs/BRACKET_BACKTEST.md) before relying on results.

## Install

Requires Python 3.11+.

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
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

## API map

| Task | Entry point |
| --- | --- |
| Import sequential stop/target records | `BracketHistory.from_records(...)` |
| Replay dated account attempts | `Engine.backtest(...)` |
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
