# Historical price replay

`Engine.backtest_prices` evaluates state-dependent dollar brackets against real
minute OHLC bars, while reusing the chronological account engine and payout
ledger. The entry strategy and data queries stay outside the engine API.

This is an **explicit OHLC execution approximation**, not tick-level execution
or a guarantee that a live account would reproduce the cashflows.

## Run the long-only experiment

```sh
python -m pip install -e ".[research]"
python -m Test_Strategies.open_long --data Data --output results/open_long
```

The runner uses DuckDB with one query thread, reading
`Data/continuous/roll_rule=volume/product=*/data.parquet`. Source data is never
modified. Use `--assets MES MNQ` for a subset and `--approval-hours 48` for an
elapsed-calendar-hour processing sensitivity, not an official business-day SLA.

It compares independent 09:30 and 18:00 New York entries, each long once per
session. The evening entry belongs to the following calendar trading date.
It uses the same eligible dates for both timings within an asset. The fixed
policy is:

| Regime | Net dollar loss budget | Net dollar target |
|---|---:|---:|
| Evaluation | 2,000 | 1,500 |
| Funded, before first payout and five qualifying days left | 1,000 | 3,000 |
| Other funded states | 1,500 | 500 |

The main experiment uses one full-size equivalent: MES/ES, MNQ/NQ, M2K/RTY,
MCL/CL, MGC/GC, SIL/SI, and 6E/6E. Micro source prices are only a proxy for
full-size execution; different liquidity, spreads and prints are not modeled.
For commodities, "full size" means CL, GC and SI, not QM or other reduced-size
contracts. Dollar-per-point and tick-size inputs are explicit in the runner.

### Commissions and micro sensitivity

Lucid's [official commission table](https://support.lucidtrading.com/en/articles/11508978-approved-products-and-commissions),
checked 2026-10-03, lists per-side rates. The runner doubles them:

| Full size | Round trip | Micro | Round trip | Commission-equivalent quantity |
|---|---:|---|---:|---:|
| ES | 3.50 | MES | 1.00 | 3 |
| NQ | 3.50 | MNQ | 1.00 | 3 |
| RTY | 3.50 | M2K | 1.00 | 3 |
| CL | 4.00 | MCL | 1.00 | 4 |
| GC | 4.60 | MGC | 1.60 | 2 |
| SI | 4.60 | SIL | 3.20 | 1 |
| 6E | 4.80 | Not configured | - | - |

A sensitivity is triggered if at least five first-touch bars, representing at
least 1% of executed IS trades, touch both barriers in either entry-time run.
Both micro timings are then reported separately. Contract count is
`floor(full_round_trip / micro_round_trip)`; dollar targets and risk budgets
stay unchanged. OOS results never select the variant. An OOS-only collision
increase adds an explicitly labelled post-hoc micro sensitivity, not permission
to relabel that result as untouched OOS. Original full-size OOS stays unchanged.

### Data and execution conventions

- Raw, not back-adjusted, held-contract prices; prior-session-volume rolls.
- DuckDB joins bars to exchange-calendar windows, respecting DST. Vendor trade
  dates are not used to merge separate holiday calendar days into one trade.
- One contract must cover the whole session. Known degraded sessions, missing
  exact 18:00/09:30 entry bars, and missing final bars are excluded from both
  entry-time experiments. Exclusion reasons overlap and are reported per asset.
  This complete-case filter uses session availability and can introduce
  selection bias; early sparse micro gold/silver are particularly affected.
- Missing intermediate minutes are not forward-filled: the source documents
  them as minutes without reported trades. Unknown feed errors remain a risk.
- Exit by 16:45 New York, or one minute before an earlier exchange close,
  following Lucid's [trading-hours rule](https://support.lucidtrading.com/en/articles/11404729-allowed-trading-times).
  The calendar package is an implementation dependency, not Lucid's official
  holiday notice. Historical calendar inaccuracies remain possible.
- Exactly the configured quantity. SL distance rounds inward to whole ticks
  after commissions; TP rounds outward. Remaining buffer caps the risk. If
  fixed quantity plus one stop tick no longer fits, the account is capped out.
  A small policy budget alone instead skips; it does not breach a viable account.
- If both levels are spanned by the first exit bar, the stop fills first,
  following the selected convention even if its opening suggests target-first.
  Stop gaps still receive the worse opening fill. This is an adverse bar
  convention, not a proof of a global lower bound on lifecycle cash.
- Stops gapping through their level fill at the observed opening price rounded
  down to ticks. Targets receive their limit price, not favorable gap improvement.
  Buy proxy entries round up. No additional spread or slippage is assumed.
- Intrabar exits are timestamped at minute-end. Later bars cannot affect fills.
  Forced closes retain actual realized P&L; they are not relabeled as binary bets.
- Current rules, fees and commissions apply retrospectively. Evaluation costs
  105.20, evaluation reset 105, and no activation charge. Default approvals,
  receipts and activation are immediate scenario assumptions. All eligible
  requests are assumed approved. No trading while a request is pending.
- Five approved payouts restart a new paid evaluation, as requested for this
  research; this is not a contractual entitlement to bypass live onboarding.

## Outputs and interpretation

`results/open_long/` contains `REPORT.md`, `results.json`, `summary.csv`, and
trade/event CSVs for every asset, timing, variant and partition. These generated
files and market data are ignored by version control.

`full` is an uninterrupted chronological lifecycle. `IS` is the first 70% of
eligible sessions, and `OOS` the last 30%, each starting with a fresh account.
The policy is fixed, not fitted on this historical IS. Splits differ by asset
because source coverage differs; OOS is not a universal common date window.

All variants report cash/calendar-day, receipts, fees, payouts, evaluation
passes, failures, live handoffs, cash drawdown, per-trade P&L distributions,
holding-time distributions, exit types, collisions, and annual external cash.
Evaluation pass rate uses resolved evaluations; unfinished ones are not failures.
Trade P&L percentiles are not percentiles of future lifecycle profits.

External funding is unrestricted for performance measurement. The reported
required bankroll is the maximum deficit of **that observed path**, not a 5%
ultimate-ruin capital recommendation. These runs do not estimate future ruin
probabilities or establish that positive cash/day is a statistically reliable
market drift effect. Resampling price sessions would need to rerun the account
and adaptive brackets, not shuffle the resulting fixed-policy trade CSVs.

## Engine API

```python
from propfirm_engine import Engine, Instrument, PriceSession

# Each PriceSession has a trading date, aware close timestamp,
# UTC bar-open timestamps as int64 nanoseconds, and an N-by-4 OHLC array.
# The first bar is the entry; the last bar ends at the supplied session close.
result = Engine().backtest_prices(
    spec, sessions, policy,
    targets={"evaluation": 1500, "funded_build": 3000, "funded_fallback": 500},
    instrument=Instrument("ES", point_value=50, tick_size=0.25),
    config=config,  # cost_per_contract is ROUND TRIP for each actual contract
    quantity=1,
    collision_policy="stop_first",  # required explicit approximation opt-in
)
print(result.replay.net_cash, result.replay.net_cash_per_day)
print(result.decisions[0])
```

Targets must cover the sizing policy's exact regime names. The supplied firm's
rules must be supported by chronological replay (including EOD trailing-floor
updates and continuous breach checks). The fixed quantity must fit its current
contract limit. Sessions and input arrays are validated and copied read-only.
Historical fingerprints include price data, targets, instrument and quantity.

## Matched zero-drift comparison

The research companion runs the same fixed policy on sign-randomized minute
prices rather than comparing historical fills with ideal unlimited-duration
binary bets:

```powershell
$env:PYTHONPATH = 'src;.'
python -m Test_Strategies.zero_drift --paths 200 --seed 20261003
```

For each minute, express OHLC relative to the preceding close (the first minute
uses the session opening). Multiply the complete innovation by an independent
fair sign, exchange high/low when reflecting, and reconstruct cumulative prices.
Integer millionths of a price unit prevent floating accumulation from changing
tick rounding. Each session re-anchors at its observed open; there are no
overnight positions. The 09:30 tape is a suffix of the randomized 18:00 tape.

This preserves absolute minute movements, bar ranges, gap magnitudes, volatility
clustering, timestamps and missing-minute patterns. Expected close-to-close
changes are zero. It removes directional dependence along with average drift;
it does **not** establish a continuous intrabar martingale. All execution and
account assumptions above still apply, including stops on ambiguous bars.

The last 30% of eligible sessions is replayed from fresh accounts without policy
optimization. Both entry times and the exploratory gold/silver micro variants
share each asset's random tape. Outputs in `results/zero_drift/` include:

- Per-path metrics in CSV, including cash/day, cash, payouts, fees, attempts,
  cash drawdown and finite-horizon funding deficit.
- Observed historical metrics and control distributions in `results.json`.
- A compact comparison in `REPORT.md` and `summary.csv`.

P5–P95 describes the outcome distribution, not uncertainty in its mean. The
reported mean Monte Carlo standard error measures simulation noise only. The
upper-tail rank is `(1 + controls at least as high as observed) / (paths + 1)`;
Holm adjustment covers the full-size cash/day comparisons, not every reported
metric. With 200 paths and 14 primary comparisons, resolution alone prevents a
5% Holm rejection: use substantially more paths for precise extreme-tail tests.

This is a retrospective benchmark conditional on observed OOS magnitudes and
data eligibility, not an independently forecast price distribution. An unusual
historical result could reflect directional dependence, data selection or
execution assumptions, not necessarily exploitable drift. Micro alternatives
remain exploratory. Funding deficits are not lifetime gambler's-ruin estimates.

## Execution scenarios and price-policy fitting

`slippage=None` preserves the existing execution model. Opting into a
`SlippageModel` adds adverse ticks beyond the raw-price fill reference. The
increment represents combined execution friction; do not charge spread again.
It is not a reconstruction of a platform's order book or simulator.

`TickDistribution((.8, .15, .05))`, for example, assigns probabilities to zero,
one and two adverse ticks. Supply separate ordinary/stressed distributions for
market and stop orders. The finite tail, stress probability, hourly multipliers
and volatility cap are explicit assumptions, not physical bounds or estimated
Lucid parameters. Use separate profiles per asset/platform when evidence supports
them. The research example's low/moderate/stressed profiles are uncalibrated.

At each minute the execution multiplier uses the mean range of the last
`volatility_window` completed available bars, divided by an IS-only reference
range, floored at one, multiplied by the local-hour factor and capped at
`max_multiplier`. It never reads the current bar's eventual range. Optional
`PriceSession.warmup_timestamps` and `warmup_ohlc` must precede entry and seed the
calculation. Otherwise it uses earlier supplied sessions, or the declared
reference for a cold start. Gaps in observations are not forward-filled.

Each trading session has a shared latent stress state. Draws are reproducible by
seed, path, instrument, session, absolute minute and order channel. A skipped
trade does not consume or shift later draws. The brackets do not see the latent
stress state or future execution draws.

With `compensate_slippage=True` (default when using a model):

- Brackets are anchored to the actual market entry fill, including entry ticks.
- Stop distance is reduced by the entry-time stop-mixture quantile allowance;
  trigger loss plus allowance plus commissions fits the net policy budget and
  remaining account buffer. `planned_net_risk` includes this allowance.
- A target limit is placed to cover the requested net profit and commissions.
  No adverse fill-price slippage is imposed on a limit. The optional
  `target_trade_through_ticks` requires price to pass the limit before filling at
  the limit price; it is a conservative fill convention, not queue simulation.
- Actual stop execution uses conditions at the triggering bar's opening, based
  on prior completed bars. Observed gaps are applied once, followed by the extra
  execution ticks. Forced closes use the market distribution. Surprise fills
  can exceed the allowance and breach MLL; realized P&L is never clipped.
- Stop-first still resolves ambiguous OHLC bars. Subminute ordering, latency and
  MLL touch times remain unobserved. A minute-end timestamp is not a measured
  execution timestamp.

Set `compensate_slippage=False` to measure execution effects without the stop
allowance. Both modes still anchor brackets to the actual entry fill. Too little
policy budget causes a skip; insufficient account buffer for the selected
scenario's minimum executable bracket causes `CAPPED_OUT`.

```python
from propfirm_engine import Engine, SlippageModel, TickDistribution, evaluate_prices
from propfirm_engine.target_research import TargetPolicy

# Illustrative probabilities only; reference_range_ticks must be chosen using IS.
execution = SlippageModel(
    market=TickDistribution((.8, .15, .05)),
    stop=TickDistribution((.4, .4, .2)),
    stressed_market=TickDistribution((0, .2, .3, .3, .2)),
    stressed_stop=TickDistribution((0, 0, .2, .3, .3, .2)),
    reference_range_ticks=10,
    label="illustrative sensitivity, not calibrated fills",
)
initial = TargetPolicy(policy, tuple(targets[r.name] for r in policy.regimes))
fit = Engine().fit_prices(
    spec, sessions, instrument, config,
    policy=initial,
    risk_bounds={r.name: (100, 2000) for r in policy.regimes},
    target_bounds={r.name: (100, 4000) for r in policy.regimes},
    slippage=execution, paths=3, execution_seed=12,
    generations=5, population=8, seed=42,
    objective=lambda result: result.net_cash_per_day,
    direction="maximize", collision_policy="stop_first",
)
print(fit.policy, fit.score)  # score is OOS, never IS
print(fit.out_of_sample.distributions["net_cash_per_day"])
```

The search reuses the existing CMA-ES optimizer and canonical account lifecycle.
It searches risk/target dollars only, not favorable execution parameters. The
chronological split is 70/30 by default, with fresh accounts/wallets at OOS.
Training candidates share precomputed execution tapes; final OOS draws use
separate path identifiers. Unvisited IS regimes retain their initial settings.
Search caches contain scores and visited regimes, not every candidate ledger.
The objective may be any finite scalar function of `BacktestResult`; distributions
of cash/day, cash, fees, payouts, drawdowns, funding requirements and execution
diagnostics remain available regardless of the selected objective.

Repeated execution paths on one price history quantify execution uncertainty,
not independent future market histories. The historical OOS already inspected in
this project is not fresh confirmation of subsequent research decisions. Reserve
new data for independent validation. A larger search or more execution paths does
not repair that limitation.

The staged runner is `python -m Test_Strategies.execution_study --assets MES`.
Its JSON records every scenario, selected policy, split, source, code hash and
metric distribution; CSVs contain per-path cash metrics and one sample fill
ledger per stage. The final stress stages keep the learned policy fixed.
