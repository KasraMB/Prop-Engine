# Dated payout ledger

This accounting component is integrated by the chronological bracket replay API;
it is not the legacy resampled batch engine's execution model.
It consumes realized net trade P&L, explicit session finalization and externally
supplied request/approval/receipt times. It neither invents delays nor proves
intraday compliance from trade summaries.

The selected request policy requires a flat position and a finalized session.
It permits a request immediately after that close; no extra trading day is
imposed. All trading is blocked until approval or rejection. Approval deducts
gross and resets cycle profit and configured day counters; a later receipt
creates spendable cash. A payment fee is explicit and reduces that receipt only.
Floor locking is an event at request, not at receipt. The ledger tracks realized
balance breaches only. The bracket adapter bounds stop exposure and updates the
ledger floor through `advance_floor`; arbitrary MTM paths still require a different
executor. This class does not independently calculate EOD ratchets or fills.

Payout-count exhaustion is labelled research censoring, not account failure.
The caller must not silently restart a new evaluation at that boundary: live
handoff/continuation still requires its own model. Rejected requests preserve
cycle counters and retain any request-time floor lock.

The following is a synthetic timing scenario, not a promise of Lucid's approval
speed or a current checkout price:

```python
from datetime import datetime, timedelta, timezone
from propfirm_engine import PayoutLedger
from propfirm_engine.firms import lucidflex

schema = lucidflex.build_account(50_000).phases[1].payout_schema
ledger = PayoutLedger(schema, opening_balance=50_000,
                      qualifying_days=5, winning_day_profit=150,
                      initial_floor=48_000, lock_floor_on_request=50_100)
start = datetime(2026, 1, 1, 21, tzinfo=timezone.utc)
for day in range(5):
    at = start + timedelta(days=day)
    ledger.record_trade(at, 200)  # net realized P&L, costs already included
    ledger.close_session(at, at.date())

request = ledger.request(start + timedelta(days=4), 500, flat=True)
assert ledger.balance == 51_000 and ledger.floor == 50_100
assert ledger.receipt_cashflows(start) == ()

ledger.approve(start + timedelta(days=5))
assert ledger.balance == 50_500 and ledger.cycle_profit == 0
assert ledger.qualifying_days == 0
assert ledger.receipt_cashflows(start) == ()  # approval is not cash

ledger.receive(start + timedelta(days=7), request)
receipt, = ledger.receipt_cashflows(start)
assert receipt.offset == timedelta(days=7) and receipt.amount == 450
```

The exported receipts can be combined with separately known fee events for the
cashflow tools. They do not alone establish complete attempt duration, upfront
fees, permission to retry, a live continuation value or a finite-wallet policy.
Money arithmetic retains exact rational decimal input amounts; actual provider
rounding is not guessed. Cashflow exports use the existing finite-float API.
