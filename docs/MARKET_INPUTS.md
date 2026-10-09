# Market inputs and execution scenarios

Call `Engine.check_replay(spec, adapter="strategy", features=("atomic_baskets",))`
to preflight account rules and requested features before preparing data. The
recorded adapter does not silently enable sizing, and overnight holding remains
unavailable when the profile requires flattening. Missing intrabar history is
never reported as exact. Replay still validates actual event data as it is read.

Recorded fills, external opportunities and causal strategies are different input
contracts. They share account rules and cash accounting, not an assumption that
every strategy uses fixed stop/target exits.

- `Engine.replay_events`: import actual signed fills and ordered portfolio marks.
  Arbitrary exits, partials and overlapping positions are supported. Changing
  historical quantity or exits requires another source contract.
- `Engine.replay_opportunities`: release external signals to a policy which
  chooses orders using actual account state. The signal list itself stays fixed.
- `Engine.replay_strategy`: regenerate signals, sizing and position management
  through callbacks. Use this when a different exit changes later signals.

## Input fidelity

Observed quotes use `fidelity="observed_marks"`. Missing bid/ask, intrabar order,
liquidity and timestamps cannot be recovered from a closed trade or an OHLC bar.
The following adapters require an explicit approximation selection:

```python
from datetime import date, datetime, timedelta, timezone
from propfirm_engine import Bar, Instrument, MarketTape, bar_quotes

instrument = Instrument("ES", point_value=50, tick_size=.25)
at = datetime(2026, 9, 1, 14, tzinfo=timezone.utc)
bars = [Bar(at, at+timedelta(minutes=1), "ES", 5000, 5002, 4999, 5001)]
feed = bar_quotes(bars, instrument, path="ohlc", half_spread_ticks=1, liquidity=None)
tape = MarketTape(feed, [instrument], sessions=(date(2026, 9, 1),))
assert tape.fidelity == "ohlc_path"
```

Pass `fidelity="ohlc_path"` when replaying this tape. `ohlc` assigns open, high,
low and close to four synthetic times across the bar; `olhc` reverses the extrema.
This is a scenario, not a reconstruction or a general worst-case fill model.
The strategy sees each synthetic point only when released. Stops can gap and
limits use the selected execution model. OHLC volume is not queue depth.
Adjoining bars need increasing `seq` so a prior close precedes the next open at
the same timestamp. Test both paths where that ambiguity matters; neither is
automatically conservative for every long/short, trailing or multi-leg policy.
The existing fixed-bracket price adapter still has its separate stop-first rule.

`trade_quotes(ticks, instruments, half_spread_ticks=1, liquidity=None)` takes
`TradeTick(at, symbol, price, seq=0)` and requires `fidelity="last_trade"` on replay.
Marks come from prints; bid/ask are the selected symmetric spread scenario.
`liquidity` is the per-side capacity at each generated observation; explicit None
means unlimited scenario liquidity. Printed volume is not used as quote depth.

`MarketSource` provenance travels on individual events and through MarketTape.
Converting a feed to an iterator does not silently turn bars into observed quotes.
Mixed fidelities must be declared, not relabelled. A custom `MarketFeed` can carry
a replacement scenario and assumptions; it must still produce valid Market data.
Execution models implement `price(order, quote, instrument)` and
`cost(quantity, first=...)`; the engine does not certify a custom fill model.

`merge_markets(*feeds, ties="atomic")` combines disjoint instrument quotes at
identical timestamp/sequence keys. `ties="reject"` rejects collisions. Each feed
must already be ordered; merging is incremental, not a full-history sort. Preserve
feed wrappers when merging approximations. Different `seq` values retain explicit
ordering. Matching timestamps do not prove that historical quotes were synchronous.
Merging independently approximated bar extrema is a declared joint scenario, not
observed cross-asset intrabar dependence.

## External opportunities

An `Opportunity` contains an aware release time, symbol, side (+/-1), optional
absolute stop/target prices, tag, optional expiry, and a sequence for timestamp
ties. Those levels are metadata for the policy, not mandatory exits. A policy
implements `on_opportunity(context, opportunity, market)` and returns normal
orders. It may also implement the strategy's market/order/session callbacks.

The first market observation on or after release delivers an unexpired signal.
Orders still execute no earlier than the next market observation. Earlier-than-
horizon opportunities are skipped. Expiry is exclusive: a signal is unavailable
at its expiry instant. Unknown future signals and signals after the last market
observation are not delivered. The result exposes offered, expired and pre-horizon
counts; offered is not a fill count and includes observation-only warmup delivery.

Use `OpportunityStrategy(opportunities, policy)` from a fresh fitting factory to
optimize a resizable opportunity policy through `Engine.fit_strategy`. Keep the
complete external signal series or filter it by window; pre-horizon signals cannot
be replayed as fresh OOS entries. Do not use fixed signals when altered exits would
have changed their generation. That requires a full causal strategy callback.

## Multi-leg execution

Separate `Order` actions are legged execution: liquidity, cost and firm checks can
occur between legs. `Basket("spread", (order_a, order_b))` selects an all-or-none
atomic scenario. Its distinct instruments must all have current quotes, executable
prices and full size in the same market observation. Partial basket fills are not
invented. Fees apply per leg and rules see combined portfolio settlement. Existing
positions are marked before execution, so a real observed pre-fill breach still
terminates the account.

Baskets accept market/limit legs with one shared time-in-force. IOC either fills
every leg or cancels. Cancelling the group or one working leg cancels the group;
amendment requires cancel/replace. Nested parents and OCO links are rejected for
atomic legs. For spread stops or more complex conditions, the strategy can submit
the basket when its causal condition is observed. Pending exposure reservations
remain conservative; atomicity is not a permission to exceed contract limits.

This is an explicit simulation contract, not a claim that an exchange, platform
or prop firm guarantees atomic fills across instruments.
