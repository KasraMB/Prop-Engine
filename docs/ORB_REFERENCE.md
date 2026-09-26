# Opening-range breakout reference signal

Research baseline, 2026-09-23. The user chose a simple ORB for the first
LucidFlex 50K DLL-off test. These strategy settings are **implementation choices**,
not Lucid rules or parameters selected on a final test set.

The strategy lives in `Test_Strategies/orb_reference.py`, outside the installed
engine package and browser bundle. It is a standalone example, not an engine API
feature. The public API is for the engine, optimization and related generic
components; strategies supply inputs through adapters. This separation follows
the user's clarification on 2026-09-23. Run the example from the repository root.

## Exact baseline

- One instrument, chronological, non-overlapping, timezone-aware OHLC bars.
- Opening range: 09:30 inclusive to 09:45 exclusive, America/New_York.
  Contiguous bars must cover the complete interval without straddling its edges.
- From 09:45, first completed close strictly above/below the range triggers a
  long/short. Touching the boundary does not trigger. Missing bars before a
  signal invalidate that day's remaining search; an earlier signal might be lost.
- Entry reference: next contiguous bar's open, strictly before 16:00 local.
  No entry without that bar; no carry to the following date.
- Initial stop reference: the opposite range boundary. Risk distance is the
  actual reference entry minus that boundary, not simply opening-range width.
  If the next open crosses the stop, skip rather than create negative risk.
- At most one entry attempt per local date. No retry after an invalid entry gap.
- No automatic holiday calendar. The module reports incomplete/zero-width ranges
  and missing entry data; it cannot detect days wholly absent from the input.

An observed next-bar open is a **reference price**, not an executable fill
guarantee. These are signals, not trades or account returns. Quantity, commissions,
slippage, stop/target/time exits and forced firm flattening belong to the executor.
The 16:00 entry cutoff is a strategy choice, not the firm's forced-flat deadline.
No take-profit or exit schedule is silently inferred here.

OHLC highs/lows do not specify their order. This module cannot satisfy or bypass
the existing strict intraday account guard. The historical strategy adapter in
Test_Strategies remains a separate legacy experiment, not this implementation.

## Academic context, not replication

Holmberg, Lönnbark and Lundström (2013), *Assessing the profitability of intraday
opening range breakout strategies*, Finance Research Letters 10(1), 27–33,
DOI 10.1016/j.frl.2012.09.001.

The [university author record](https://umu.diva-portal.org/smash/person.jsf?pid=authority-person%3A66742)
provides the bibliography and abstract. That abstract describes a
normal-distribution-based threshold approach tested on crude-oil futures.
It does not establish the precise 15-minute clock-range recipe used here.
The [publisher article](https://www.sciencedirect.com/science/article/pii/S1544612312000438)
returned HTTP 403 during this research; the linked university item also presented
a bot challenge. Full methods were not read. This baseline is therefore **not a
paper replication**, and no published profitability result is transferred to it.
No third-party firm-rule source is used.

## Executable synthetic example

This demonstrates causal signal timing and the stop anchor only.

```python
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from Test_Strategies.orb_reference import PriceBar, opening_range_breakout

start = datetime(2026, 1, 5, 9, 30, tzinfo=ZoneInfo("America/New_York"))
bars = [
    PriceBar(start + timedelta(minutes=i), start + timedelta(minutes=i + 1),
             100, 101, 99, 100)
    for i in range(15)
]
bars.append(PriceBar(start + timedelta(minutes=15), start + timedelta(minutes=16),
                     100, 103, 99, 102))
bars.append(PriceBar(start + timedelta(minutes=16), start + timedelta(minutes=17),
                     102, 104, 100, 103))
result = opening_range_breakout(bars)
entry, = result.entries
assert entry.signal_at == entry.entry_at == start + timedelta(minutes=16)
assert entry.reference_entry_price == 102
assert entry.stop_price == 99
assert entry.risk_points == 3
assert result.execution_model == "signals_only_next_bar_open_reference"
```

Before performance evaluation: join these signals to a separately specified
ordered-price executor, integer-contract sizing/costs, session/account transitions
and the dated payout ledger. Preserve an untouched chronological test partition.
None of those steps is supplied by a successful signal test.
