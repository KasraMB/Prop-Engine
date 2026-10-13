# Calendar-window research

Use this API for experiments such as "fresh account and $2,000 wallet for each
60-calendar-day start". Research generates trades and declares calendars. The API
handles sizing, account mechanics, optimization, exclusions and IS/OOS reports.
No private `_Replay` subclass, nested simulation objective or separate optimizer
is needed. This does not generate trades from market data.

## Runnable example

```python
from datetime import date, datetime, time, timedelta
from statistics import mean, stdev
from zoneinfo import ZoneInfo
from propfirm_engine import Engine, Window, Trade, TradeHistory, PnlMark
from propfirm_engine import DollarPolicy, BacktestConfig
from propfirm_engine.firms.lucidflex import replay_50k

zone = ZoneInfo("America/New_York")
days = tuple(date(2022, 1, 3) + timedelta(days=i) for i in range(11)
             if (date(2022, 1, 3) + timedelta(days=i)).weekday() < 5)
closes = {d: datetime.combine(d, time(16, 45), zone) for d in days}
rows = []
for day in days:
    entry = datetime.combine(day, time(10), zone)
    exit = entry + timedelta(minutes=30)
    rows.append(Trade(entry, exit, day, 50, 30,
                      (PnlMark(entry, 0), PnlMark(exit, 30))))
history = TradeHistory(tuple(rows))
spec = replay_50k(eval_fee=105.20, reset_fee=105, contract_type="micro")
config = BacktestConfig(cost_per_contract=1, approval_delay=timedelta(days=1),
    receipt_delay=timedelta(days=2), activation_delay=timedelta(minutes=30),
    initial_wallet=2000)
engine = Engine()

def window(first, last):
    return Window(datetime.combine(first, time(9, 30), zone), closes[last])

training = engine.prepare_windows(spec, history, sessions=days, closes=closes,
    windows=(window(days[0], days[1]), window(days[2], days[3])))
holdout = engine.prepare_windows(spec, history, sessions=days, closes=closes,
    windows=(window(days[4], days[5]), window(days[6], days[7])))

def income_score(batch):
    cash = [w.net_cash for w in batch.windows]
    return mean(cash) - stdev(cash)

fit = engine.fit_windows(training, holdout, config,
    policy=DollarPolicy.constant(200),
    risk_bounds={"evaluation": (100, 1900), "funded": (100, 1000)},
    objective=income_score, direction="maximize", seed=42,
    generations=1, population=4)
print(fit.policy, fit.score)  # score is the frozen holdout score
print(fit.out_of_sample.summary)  # all metrics, regardless of objective
trace = engine.replay_windows(holdout, fit.policy, config, record_events=True)
assert trace.windows[0].events
```

Fees, fill assumptions and delay values above are examples, not verified execution
costs. Supply your actual scenario. A production input needs sufficiently dense
ordered valuations; the two endpoint marks here only illustrate the contract.

## Calendar and horizons

- `sessions` are unique increasing session dates, including no-trade dates.
- `closes` maps every declared session to an aware close timestamp, including
  known early closes. It must not exceed the profile's cutoff. Supply the firm's
  processing/session clock, not automatically the instrument's RTH closing time.
- `Window(start, end)` uses aware timestamps. Entry inclusion is `[start, end)`;
  scheduled receipts at `end` are included. Local timestamps become UTC internally,
  so configured payout delays are elapsed durations across DST.
- The first evaluation is purchased at `start`, even without a signal. Each
  window starts fresh with the configured wallet. Subsequent purchases use the
  engine's retry and signal rules. Unreceived payouts remain outstanding, not cash.
- A boundary cutting an open trade is rejected. Research must supply an actual
  terminal execution or change the window, not silently discard the position.
- Supply the full known calendar and enough source coverage for each horizon.
  The API cannot distinguish a deliberately empty session from omitted data.
  `history=None` explicitly represents no trades, not missing observations.
- `Window(start, end, exclusion="unresolved data")` retains an excluded window
  in `batch.excluded`. It is not simulated or converted to zero cash. Freeze
  exclusions independently of policy. Results describe the observed subset only.
- Window starts must increase. Within-partition overlap is allowed and creates
  dependent observations. Every training horizon must end before holdout starts.
  No window may straddle that boundary. Partition by complete horizons, not just
  entry dates. Preparing holdout inputs does not execute or score them.

## Objective and results

`objective` receives a compact `WindowBatch`, not an individual trade or replay.
Its `windows` contain observed per-window cash, cash/day, fees, payouts, durations,
failures, passes, executed trades, terminal status and required bankroll. The
default objective is mean net external cash per calendar day across windows.
An aggregate objective can use sample SD or quantiles across windows. Define its
minimum sample size explicitly; sample SD is not defined for one observation.
Keep it deterministic and based on outcome fields, not optional traces/reports.

`fit.in_sample` and `fit.out_of_sample` are the exact window batches for the
selected policy, not unrelated continuous replays. `fit.score` is OOS only.
The summary retains all metric distributions including SD, variance and
percentiles even if the objective selects only one metric. Cash drawdown,
underwater duration, receipt timing, wallet shortfalls and MLL breaches are also
reported. Undefined receipt times/returns are omitted from their distributions;
each distribution includes its observation count. Optional
`risk=RiskConfig(...)` adds the existing cash-risk report to final batches.
Bankroll requirements use an unrestricted counterpart when a finite wallet
truncates a replay. Historical frequencies are not independent-future estimates.

`progress` receives search events containing `evaluations`, `policy`, `score`
and compact `result` for candidate-ledger logging. "Evaluations" counts candidate
scorings, not prop-firm accounts. Serialize dataclasses with `dataclasses.asdict`.
After IS selection a `holdout` event is emitted, then the frozen policy is scored
on holdout. Never use holdout reports to revise this same experiment unnoticed.

Regime detection uses only the scored IS windows. Unvisited regimes may revert
to their initial budgets only if rescoring does not worsen the objective.
The search is serial CMA-ES, with the baseline included; it does not promise a
global optimum. Cached candidate entries contain scores/policies, not full traces.
Prepared windows share immutable trade observations and reuse grouping/hashes.

## Migration from the private income adapter

1. Replace `CalendarEngine` with `Engine.prepare_windows` and `replay_windows`.
2. Pass the existing frozen training/check windows, calendar, exclusions and
   trade logs. Preserve fees, wallet, delays, bounds, seed and search budget.
3. Replace `FiniteObjective`'s replay loop with `income_score(batch)` above and
   call `fit_windows`. Do not also pass `policy_objective` or call `Engine.fit`.
4. Log candidates through `progress`; use `record_events=True` for selected-policy
   ledger audits. Compare common candidate cash and events before accepting migration.
5. Preserve old artifacts. The regime-selection correction can change the final
   policy even when every candidate score matches; this is not necessarily a
   performance-only restart. Label the reused development check as development.

This API currently supports the same sequential sized-trade account rules as
`backtest`. It does not add overlapping positions, market execution, or calendar
window support to `fit_phases`; use `fit_windows` for this experiment.
