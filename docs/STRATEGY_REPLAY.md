# Strategy replay

`Engine.replay_strategy` runs an external strategy against timestamped quotes.
It shares the event portfolio, account rules, payout ledger, wallet and retry
coordinator. Strategies stay outside the package. This is a deterministic quote
execution scenario, not a reconstruction of exchange queue priority.

```python
from datetime import date, datetime, timedelta, timezone
from propfirm_engine import (
    BacktestConfig, Engine, Instrument, Market, Order, Quote, QuoteModel,
)
from propfirm_engine.firms.lucidflex import replay_50k

class Example:
    def __init__(self):
        self.step = 0

    def on_market(self, context, market):
        self.step += 1
        if self.step == 1:
            return [Order("entry", "ES", 1)]
        if self.step == 2:
            return [Order("exit", "ES", -1, reduce_only=True)]

at = datetime(2026, 9, 1, 14, tzinfo=timezone.utc)
markets = [Market(at + timedelta(minutes=i), (Quote("ES", p, p, p),))
           for i, p in enumerate((6000, 6001, 6003))]
result = Engine().replay_strategy(
    replay_50k(eval_fee=105.2, reset_fee=105, contract_type="mini"),
    markets, [Instrument("ES", 50, .25)],
    BacktestConfig(0, timedelta(0), timedelta(0), timedelta(0)), Example(),
    sessions=(date(2026, 9, 1),), fidelity="observed_marks",
    models={"ES": QuoteModel(fee=2)}, max_mark_age=timedelta(minutes=2),
    liquidation_fee=2,
)
assert result.result.book.balance == 50_096
assert result.result.replay.net_cash == -105.2
```

Fees and prices here are illustrative. The entry executes on the second quote
and the exit on the third. No stop or target is needed for either order.

## Clock and callbacks

Each market observation has aware time, an explicit sequence number, and one or
more simultaneous instrument quotes. Each quote supplies bid, ask and valuation
mark independently. All three must be on the instrument's tick grid. Quotes are
atomic portfolio observations; separate observations are explicitly ordered.

At each observation, due lifecycle events run first, then portfolio marks and
breach checks, then working orders, order notifications and `on_market`. A hard
breach cancels orders and liquidates before a hypothetical stop or target fill.
Orders returned by callbacks can execute only on a later observed market event.

`on_market(context, market)` is required. Optional callbacks are
`on_order(context, event)` and `on_session(context, session_date)`. Return an
iterable of `Order`, `Amend` and `Cancel`, or None. Notification-generated actions
are processed once per callback batch, not recursively. Invalid action types
raise errors; account/limit rejection is a reported order event.

Context is immutable: current book snapshot, floor, wallet, phase, next phase,
attempt, contract cap, remaining qualifying days, payout count, availability and
working orders. There is no future-data handle. Keep your own strategy state.
`warmup` can supply earlier market observations with `context.warmup=True`;
orders are disabled and no account is purchased during warm-up. All warm-up
timestamps must precede the account horizon.

## Orders

Use signed integer quantities. Supported kinds are `market`, `limit`, `stop`,
`stop_limit` and `trailing`. Provide only the applicable `limit`, `stop` or
positive `trail` fields. `tif` is `day`, `gtc` or `ioc`. Absolute `expires` is
optional; expiry precedes matching at an identical timestamp.

- Market orders buy at ask and sell at bid, plus the execution model.
- Stops trigger from the supplied mark, not an invented intrabar price. Gaps
  execute at the available quote, not the old stop level.
- Stop-limit orders remain triggered until executable within their limit.
- Trailing orders anchor at the mark known when submitted, then advance only
  on observed favorable marks. Linked trailing exits initialize after entry.
- Amendments specify a replacement order with the same ID. Quantity is the new
  total including fills already received. Amendments cannot change instrument,
  side or parent, or reduce total quantity to/below already-filled quantity.
  Replacing a trigger resets its trigger/anchor state.
- `reduce_only=True` prevents an exit from opening the opposite position.
- `parent="entry_id"` links a reduce-only exit to filled parent quantity.
  An unfilled cancelled parent cancels its children. After a partial parent fill,
  children remain available for the executed exposure.
- Shared `oco` IDs cancel siblings after any fill, including a partial fill.
  Partial exits can therefore leave exposure without the sibling protection;
  use fill feedback to submit the desired remaining protection.
- Signal/time exits are ordinary orders returned by callbacks. Multiple linked
  exits can express multiple targets. Pyramiding and reversals use signed orders.
  Multiple legs execute in declared order, not as an atomic spread fill.

Quote `bid_size` and `ask_size` share finite per-side capacity across orders in
submission order. None explicitly means unlimited scenario liquidity. A partial
market order remains working unless IOC. IOC cancels its remainder after its
first eligible observed quote. OCO orders conservatively reserve entry exposure
independently; there is no speculative reduction for mutual exclusion.

`QuoteModel` supplies per-contract fill fees, a fixed first-fill fee per order,
and adverse tick slippage. Limit fills never exceed the limit after slippage.
Each instrument gets its own model. Custom models implement
`price(order, quote, instrument)` and `cost(quantity, first=bool)`; they may return
None for no executable price. Prices, fees and limits are validated. There is no
implied realism from selecting a fixed slippage setting.

## Account interaction

The first accepted order starts the paid account, even if its limit never fills.
Rejected over-cap orders do not start one. Pending exposure is reserved using
the maximum long/short outcome per instrument, summed in profile cap units.
Use the same `units` mapping as [event replay](EVENT_REPLAY.md).

Orders are rejected while the account is unavailable. Failure, evaluation
transition and session cutoff cancel working orders. The current flat-at-cutoff
profile cancels GTC too: GTC does not override the firm's close requirement.
When `flatten_at_close=False`, open positions and GTC orders can carry into the
next declared session, while DAY orders still expire. LucidFlex remains flat at
cutoff. `Context.phase_name` and `next_phase_name` identify stages in a multi-step
evaluation; `phase` continues to distinguish eval from funded.
Session callbacks cannot reopen the just-closed session. The source-history
quarantine used for recorded fills is unnecessary because strategy decisions
are regenerated from actual account outcomes.

`result.result` is the canonical `EventReplay`; `result.orders` contains order
acceptance, rejection, cancellation and fill events. Cash/risk reporting uses
`result.result.replay`. Tracing is optional and does not change economics.

Return `Abandon(reason="...", retry=True)` to liquidate and abandon an active
account for a research reason. It is recorded separately and does not increment
firm failures. Retry buys a new account no earlier than the next session; False
stops this path. Actual breaches during liquidation remain actual failures.
Returning no order is a voluntary skip, not abandonment or a breach.

The order interface does not yet imply support for every firm profile, atomic
spread fills, bar-path assumptions or general strategy optimization. Those remain
separately tracked acceptance items.
