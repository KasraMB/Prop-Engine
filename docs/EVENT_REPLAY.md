# Recorded event replay

`Engine.replay_events` accepts a chronological iterable of executed `Fill` and
`Marks` events. It uses the same rules, payout ledger, fees, wallet and retry
coordinator as bracket replay. It does not require stops, targets or binary exits.
Longs, shorts, partial exits, scale-ins, reversals and concurrent instruments use
FIFO accounting. Futures P&L uses each instrument's tick size and point value.

This is recorded execution, not an order simulator. Quantities, prices and fees
remain fixed. It cannot retarget exits or regenerate strategy decisions. The
existing optimization APIs still use their existing input contracts.

## Example

```python
from datetime import date, datetime, timedelta, timezone
from propfirm_engine import BacktestConfig, Engine, Fill, Instrument, Marks
from propfirm_engine.firms.lucidflex import replay_50k

at = datetime(2026, 9, 1, 14, tzinfo=timezone.utc)
events = [
    Fill(at, "ES", 2, 6000, fee=4),
    Marks(at + timedelta(minutes=1), (("ES", 6001),)),
    Fill(at + timedelta(minutes=2), "ES", -1, 6002, fee=2),
    Fill(at + timedelta(minutes=3), "ES", -1, 5999, fee=2),
]
spec = replay_50k(eval_fee=105.20, reset_fee=105, contract_type="mini")
config = BacktestConfig(0, timedelta(0), timedelta(0), timedelta(0))
result = Engine().replay_events(
    spec, events, [Instrument("ES", point_value=50, tick_size=.25)], config,
    sessions=(date(2026, 9, 1),),
    fidelity="observed_marks",
    mark_fills=True,
    max_mark_age=timedelta(minutes=2),
    liquidation_fee=2,
    trace=True,
)
assert result.book.balance == 50_042
assert result.book.realized == 50 and result.book.fees == 8
assert result.replay.net_cash == -105.20
```

Fees in this example are illustrative, not a current platform quote. `Fill.fee`
is the total fee for that fill, not a per-contract or round-trip amount. Both
trading-cost fields in `BacktestConfig` must be zero to prevent double counting.
Evaluation, reset, activation and payment fees retain their existing meanings.

## Observations and clocks

- Explicit `fidelity="observed_marks"` is required. Breach is detected at the
  first supplied observation at or below the floor. No exact intrabar path is
  inferred from sparse marks or closed trades. Input resolution remains your
  responsibility.
- Timestamps must be aware. Events are processed in strictly increasing
  `(at, seq)` order. Resolve timestamp ties yourself; ambiguous ties are rejected.
  `merge_events` lazily combines already ordered feeds.
- A `Marks` event values all its prices atomically. Put simultaneous multi-asset
  observations together to avoid creating an artificial intermediate portfolio.
- `mark_fills=True` also treats execution prices as fresh marks. With `False`,
  an explicit mark must precede each instrument's first fill. Open positions must
  have marks no older than `max_mark_age` at every processed valuation and forced
  close. This is a validation bound, not a generated quote stream.
- `sessions` lists observed trading dates, including no-trade dates in the chosen
  horizon. The normal overnight-open/session-close clock comes from the profile.
  `session_closes={date: aware_datetime}` can supply early closes. Events outside
  these windows are rejected; holidays are not guessed.
- Market events at the cutoff run before session finalization. Lifecycle events
  due at a market timestamp run first, except an eligible closing fill wins an
  exact inactivity tie. Activity uses FIFO realized P&L net of allocated entry
  and exit fees. Entry fees alone do not qualify.
- LucidFlex now defaults to the user-selected 16:15 New York cutoff on the local
  activity date plus 30 calendar days, including weekends. To reproduce old
  elapsed-time research, pass `elapsed_inactivity=True` to `replay_50k`.

## Account behavior

Closed balance feeds profit targets, consistency and winning-day calculations.
Balance plus unrealized portfolio P&L feeds continuous MLL checks. Evaluation
transitions wait until the portfolio is flat. The default session policy forces
flat, including LucidFlex. A profile can explicitly allow overnight holding with
`flatten_at_close=False`; fresh closing marks are still required.

Forced closes use the last fresh marks and the explicit `liquidation_fee` per
contract. These are scenario prices, not guaranteed executable quotes. A hard
breach terminates the account even if liquidation or later observations would
recover. Missing or stale liquidation marks raise an error, never use a future
price. Approved receipts remain payable after account failure.

Limits use total absolute exposure across instruments. Optional `units` must
define a positive contract weight for every instrument in the profile's cap
units. All instruments default to weight 1; no symbol-based inference is made.
For a mini-denominated profile, an example is `{"ES": 1, "MES": 0.1}`.
Recorded exposure above a cap raises an error; the adapter does not silently
resize or invent a rejected-fill outcome. It does not model pending orders.

While a payout is pending, recorded fills are skipped. Retries start no earlier
than the next observed session. If an entry was skipped or an account was
liquidated, its later source exit cannot become a new opposite position. The
adapter waits for the entire recorded source portfolio to become flat before
following another entry. This can skip multiple instruments together. Strategies
whose future signals depend on fills or account outcomes need the future callback
adapter, not this recorded-history assumption.

## Results and performance

`result.replay` is the existing `BacktestResult`: external receipts, fees, net
cash, cash/day, attempts, failures, outstanding payouts and lifecycle events.
Its policy is `None` because recorded quantities are not a sizing policy.
`result.book` is the current portfolio snapshot, or `None` after account failure
or handoff until another phase starts. It includes open equity separately.
`fills`, `skipped_fills` and `marks` count supplied events; generated liquidations
have their own lifecycle events and are not included in the fill count.

`trace=True` additionally retains account balance, equity and floor at each
accepted observation. The default does not retain mark history. Lifecycle and
fill events remain available in either mode. Cash analysis remains compatible:

```python
from propfirm_engine import cash_risk_path, risk_report
report = risk_report([cash_risk_path(result.replay)], sample_kind="single_history")
```

One history does not establish a future ruin probability. Supply independently
justified scenarios or historical windows for distributions and label them
accordingly. General event search, resampling and compiled execution are pending.

Inputs are streamed with one-event lookahead. Marks update cached portfolio
equity; snapshots are not copied per event unless requested. FIFO storage tracks
open lots, not closed trades. Source and account books are separate to preserve
skipped-entry semantics. Existing lifecycle/fill logs still grow with executions.
For a reproducible synthetic mark-heavy benchmark, run:

```sh
python benchmarks/portfolio.py --mode replay --events 2000 --repeat 3
python benchmarks/portfolio.py --mode replay --events 2000 --repeat 3 --record
```

## Rule and payout controls

Ordered observation replay supports one hard static or trailing drawdown rule per
phase. Trailing updates can be continuous (observed open-equity peaks) or EOD
(closed balance). Breach checks follow the rule's continuous/EOD setting. A
continuous peak cannot be applied retroactively to an earlier observation.
Daily loss uses closed session P&L plus current unrealized P&L; hard DLL ends the
account, while soft DLL liquidates and suspends that session without counting an
account failure. A suspended day is not a winning day. Hard breach takes priority
over soft suspension at the same observation.

Funded consistency gates use closed profit and the largest closed session profit.
Declare `StateField.MAX_DAY_PNL` in the payout schema's reset fields when the
largest-day statistic resets after approval. Withdrawals are not trading losses.
The portfolio executor, not the payout ledger's closed balance alone, decides
whether open equity breached a rule.

Optional replay arguments apply to both recorded events and strategy replay:

- `withdrawal(context)` receives immutable `PayoutContext` with balance, floor,
  maximum eligible amount, cycle profit, qualifying days and payout count.
  Return zero to skip or a valid amount up to the maximum. The minimum-request
  rule still applies. This can express retained buffers or partial withdrawal.
- `decision(at, request)` returns `"approve"` or `"deny"`. The default approves.
  Denial leaves cycle profit, qualifying days and the request-time floor lock
  intact; it creates no receipt. A later eligible session can request again.
- `processing=ProcessingCalendar(...)` supplies timezone, weekdays, explicit
  holidays and opening/closing times. First add the elapsed configured delay,
  then roll forward into an open processing interval. Closing time is exclusive.
  This is not a business-duration counter or a built-in holiday service.

Unsupported rule combinations are still rejected. Broader execution adapters
remain on the roadmap. The
legacy bracket API retains its narrower guards rather than treating a closed
summary as an observed intraday path.

An account may declare multiple uniquely named evaluation phases followed by at
most one funded phase. Observation replay executes them in that order, resetting
balances and rule state at each transition. No additional evaluation fee is
charged for promotion; activation is charged when entering funding. Optional
`phase_limits={name: contracts}` and `transition_delays={name: timedelta}` control
per-stage caps and delays before entering each named stage. A funded override
can lower, but cannot exceed, the scaling tier. Failure restarts from stage one.
Legacy summary/bracket validation still rejects multi-stage accounts.

`retry_on_failure=False` stops after failure; `restart_on_handoff=False` stops at
live handoff. Delayed approved receipts remain processed within the horizon.
These choices do not change whether the account failed its firm rules.

`drawdown_basis="rule"` uses open equity for continuous peaks and closed balance
for EOD peaks. Explicit `"balance"` or `"equity"` selects one basis for both.
`daily_loss_basis="balance"` resets daily loss from prior closed balance;
`"equity"` uses prior closing marked equity. Payout deductions adjust the daily
baseline rather than counting as trading loss. Declare these choices for profiles
that allow overnight exposure; they are not inferred from a firm's name.
