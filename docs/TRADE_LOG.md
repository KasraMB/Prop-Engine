# Sized trade logs

Use `TradeHistory` when the engine should choose quantity from the account state,
but your strategy has already determined entry, exit and the intervening path.
A timed exit, partial profit, breakeven exit or loss beyond the planned stop is
valid. This is trade-log replay, not market-data backtesting.

```python
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from propfirm_engine import BacktestConfig, DollarPolicy, Engine, PnlMark, Trade, TradeHistory
from propfirm_engine.firms.lucidflex import replay_50k

entry = datetime(2026, 9, 1, 10, tzinfo=ZoneInfo("America/New_York"))
exit = entry + timedelta(minutes=60)
history = TradeHistory((Trade(
    entry_at=entry, exit_at=exit, session=entry.date(),
    risk=50, pnl=30,
    marks=(PnlMark(entry, 0), PnlMark(entry + timedelta(minutes=20), -20),
           PnlMark(exit, 30)),
),))
spec = replay_50k(eval_fee=105.20, reset_fee=105, contract_type="mini",
                  elapsed_inactivity=True)
config = BacktestConfig(cost_per_contract=0, approval_delay=timedelta(0),
                        receipt_delay=timedelta(0), activation_delay=timedelta(minutes=30))
policy = DollarPolicy.constant(200)
engine = Engine()
result = engine.backtest(spec, history, policy, config)
assert next(e.quantity for e in result.events if e.kind == "trade") == 4
assert result.final_balance == 50120
print(result.net_cash, result.net_cash_per_day)
```

The zero cost above is for arithmetic illustration, not a fee recommendation.
Set actual round-trip `cost_per_contract` and any fixed `cost_per_trade`.
Sizing reserves those costs, caps quantity by loss buffer and contract limits,
and applies the selected `DollarPolicy` regime before entry. With a $100 funded
risk budget, the same $50-risk trade uses two contracts before costs.

## Input contract

- `risk`: original planned gross dollar loss for one contract, known at entry.
- `pnl`: actual gross realized dollars for one contract, not account P&L or R.
- `marks`: strictly ordered gross per-contract P&L observations, including entry
  and exit. The final mark must equal `pnl`. Entry P&L may include an immediate
  spread loss. Do not include separately configured fees in these values.
- `session`: the account's trading-session date; timestamps must be timezone aware.

P&L is already signed for the trade direction. Long and short trades use exactly
the same interface. Research supplies strategy exits and valuations; the engine
supplies quantity, account rules, lifecycle transitions, payouts and fees.
`TradeHistory.from_records` accepts dictionaries with these same fields, ISO
timestamps/date strings and nested `marks` dictionaries containing `at` and `pnl`.

Full round-trip costs are deducted at entry as an explicit conservative timing
assumption. The first observed drawdown breach liquidates at that mark, including
gap overshoot, and ignores the remainder of the recorded trade. No additional
liquidation slippage is inferred. Unrealized profit cannot pass evaluation or
count as a winning day. Only supplied marks are checked; sparse endpoints cannot
certify what happened between observations. Supply the full ordered valuation
history available to your research, not just the final return or unordered MAE.

## Optimization and scope

Pass this same history to `Engine.fit`, `Engine.fit_phases`,
`Engine.compare_searches`, `Engine.rolling_backtest` or `Engine.ruin`.
The [sizing example](../README.md#optimize-on-70-report-on-30) and
[phase-search guide](PHASE_SEARCH.md) use the same policies and settings.
Fitting splits complete sessions chronologically, sizes each trade using only
the account state available at entry, and holds recorded exits fixed.
Changing quantity assumes gross per-contract execution is unchanged; it does
not model size-dependent fills or retune targets using future outcomes.

The sized adapter currently requires sequential, non-overlapping, same-session
trades and supported EOD-trailing account profiles. Quantity is constant within
each trade. An inactivity expiry strictly inside an open trade is rejected.
Use `Engine.check_replay(spec, adapter="trades")` for account preflight.
Partial position exits, overlapping positions and other supported account rule
families remain available through fixed-quantity `replay_events`; they are not
silently converted into independent dynamically sized trades.
