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
- The profile still uses its documented elapsed-day inactivity scenario. The
  requested 16:15 expiry-calendar convention is not yet implemented.

## Account behavior

Closed balance feeds profit targets, consistency and winning-day calculations.
Balance plus unrealized portfolio P&L feeds continuous MLL checks. The floor
ratchets at EOD for the currently supported profiles. Evaluation transitions
wait until the portfolio is flat. Every session cutoff forces flat.

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

Only the existing chronological profile capabilities are enabled: EOD trailing
floor with continuous hard breach, evaluation profit/consistency/day gates and
the dated funded payout ledger. Unsupported rule combinations are rejected.
Static/intraday trailing drawdown, DLL suspension, general orders, strategy
callbacks and overnight holding remain on the research engine roadmap.
