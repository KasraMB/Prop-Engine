# Strategy fitting

`Engine.fit_strategy` optimizes only declared parameters of an external causal
strategy. Every candidate and baseline uses `Engine.replay_strategy` and the
same account coordinator. Quantity, dollar risk, targets, named regimes and
exit logic can be separate parameters; omitted parameters stay fixed. The
strategy decides how those parameters translate into executable orders.

Prepare the market stream once. `MarketTape` stores immutable integer-tick
columns and UTC microsecond clocks. Session views share those columns. Quotes
must include the selected mark basis; they do not imply an unseen intrabar path.

## Example

This small synthetic example demonstrates the API, not a profitable strategy.
The factory creates a fresh strategy for each window and seed. `setup(seed)`
must likewise create fresh execution models and scenario callbacks. Do not share
mutable strategy or execution state across candidates.

```python
from datetime import date, datetime, timedelta, timezone
from propfirm_engine import (
    BacktestConfig, Engine, Instrument, Market, MarketTape, Order, Parameter,
    Quote, QuoteModel, RiskConfig,
)
from propfirm_engine.firms.lucidflex import replay_50k

instrument = Instrument("X", point_value=1, tick_size=1)
start = datetime(2026, 9, 1, 14, tzinfo=timezone.utc)
offsets = [i for i in range(20) if (start+timedelta(days=i)).weekday() < 5][:10]
markets = [
    Market(start+timedelta(days=day, minutes=minute), (Quote("X", price, price, price),))
    for day in offsets
    for minute, price in enumerate((10000, 10000, 10010))
]
tape = MarketTape(markets, [instrument], sessions=[
    date(2026, 9, 1)+timedelta(days=day) for day in offsets
])

class Daily:
    def __init__(self, params, seed):
        self.quantity = params["quantity"]

    def on_market(self, context, market):
        if context.warmup or not context.available:
            return
        if market.at.minute == 0:
            return [Order(str(market.at), "X", self.quantity)]
        if market.at.minute == 1:
            return [Order(str(market.at), "X", -self.quantity, reduce_only=True)]

def setup(seed):
    return dict(models={"X": QuoteModel(fee=1)}, fidelity="observed_marks",
                max_mark_age=timedelta(hours=8), liquidation_fee=1)

spec = replay_50k(eval_fee=105.20, reset_fee=105, contract_type="mini")
config = BacktestConfig(0, timedelta(0), timedelta(0), timedelta(0))
fit = Engine().fit_strategy(
    spec, tape, config, Daily,
    baseline={"quantity": 1}, space={"quantity": Parameter("integer", 1, 3)},
    setup=setup, generations=2, population=4,
    risk=RiskConfig(bankroll=1000, target_ruin_probability=.05),
    objective=lambda result: result.mean_net_cash,
)
print(fit.params, fit.score)
print(fit.metrics["distributions"]["net_cash_per_day"])
print(fit.metrics["required_bankroll"])
```

`Parameter()` is continuous on [0,1]. Numeric bounds are explicit; categorical
parameters use `Parameter("categorical", choices=("signal", "time"))`. Small
fully discrete spaces are enumerated if they fit the budget. Other spaces use
the existing CMA-ES search with decoded-parameter memoization. A categorical
coordinate has no inherent economic ordering and search is not a global proof.

`constraint(params)` can reject an explicitly infeasible portfolio/policy.
Strategies can raise `InfeasiblePolicy` for the same purpose. Simulator,
configuration and non-finite objective errors propagate instead of becoming
poor candidate scores. The baseline must be feasible.

## Validation and reporting

- The default split is 70% IS and 30% OOS, by complete declared sessions.
- `validation_fraction=.2` reserves the final 20% of IS for selecting among
  `finalists` training candidates and the baseline. OOS never selects them.
- `rolling=RollingConfig(window_sessions=20, stride_sessions=5)` starts a fresh
  account, wallet and strategy in each complete window of each partition.
- `seeds=(1,2,3)` repeats the same declared execution/strategy seeds for every
  candidate. The factory and setup must actually use those seeds where random.
- `warmup_sessions=10` supplies preceding observations, including the IS prefix
  before OOS, with trading disabled. No account or open position crosses folds.
- Permitted horizon positions remain marked. Unrealized balance is not a payout
  or external cash. Each fold honors the profile's session-close rule.
- `direction="minimize"` supports loss functions; the default maximizes mean
  external cash per calendar day. A custom objective receives every compact path
  and the complete distribution report. Changing objectives drops no metrics.

`fit.score`, `fit.selected` and `fit.metrics` are OOS. `fit.baseline` is the same
OOS replay with the initial parameters. `fit.training`, `fit.validation`,
`fit.trials` and `fit.stability` are separate diagnostics. Parameters used only
in unvisited regimes are not identified by the data; a generic external factory
cannot promise automatic identification. Restrict the declared search space.

`Engine.walk_strategy(..., train_sessions=100, test_sessions=30)` refits each
chronological fold. `step_sessions` defaults to the test length. Accounts restart
per fold; cash is not a continuous-wallet backtest across folds. Overlapping
windows and repeated seeds are not independent future observations. Reusing an
inspected OOS period makes it research evidence, not an untouched final holdout.

## Scenarios and ruin

`evaluate_strategy` evaluates frozen parameters without optimizing.
`evaluate_scenarios(..., sample_kind="independent_model")` consumes externally
generated, complete ordered market tapes one at a time. Use `historical_windows`
for historical starts. Cross-asset quotes and signal ordering stay intact; the
engine does not shuffle individual quotes or splice price levels together.
Independence is a declared generator assumption, not inferred from path count.

Reports retain variance, percentiles, drawdown, receipts, costs and finite-horizon
funding risk. Wallet-stopped paths are replayed with an unrestricted wallet using
fresh identical strategy and execution seeds for capital estimation. No finite
historical sample establishes ultimate ruin. The existing complete-cycle
approximation remains separately labelled in the [risk API](BRACKET_BACKTEST.md).

Use `RandomStream(seed, channel)` in stochastic strategies/models for keyed
`uniform(*key)` or `normal(*key)` draws. Keys should identify a market opportunity,
symbol and execution leg, not the count of previously executed trades. Policies
that skip a trade then leave unrelated draws unchanged. This does not repair a
strategy that embeds future information or a mutable unseeded external model.

## Cancellation and resume

`cancel()` is polled between candidates, sessions and every 1,024 observations.
A cancelled replay returns no partial performance. Search raises `SearchCancelled`
with `.checkpoint` containing completed compact trial scores. Pass it as `resume`
to reproduce the deterministic search without rerunning completed candidates.
Candidate evaluation interrupted mid-path is rerun in full.

`checkpoint(state)` can persist completed trials; `progress(trial)` reports them.
Checkpoint use requires a stable `study_id` identifying the strategy, objective,
constraint and execution code/settings. Change it when any opaque callback changes.
Input arrays, profile and declared search settings are checked automatically.
Persist dataclasses with an application-owned safe format; do not load untrusted
pickles. Replaying cached optimizer steps reconstructs search state, so resume
cost grows with completed trial count but does not repeat their simulations.

`fit.work` reports shared tape bytes and a search observation upper bound. It
excludes final reporting, inner validation and unrestricted-wallet counterparts;
it is not a measured runtime or process memory budget. Full order/lifecycle
recording is currently per replay, not retained across all search candidates.
