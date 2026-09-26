# Explicit cashflow scenarios

Updated 2026-09-23. These APIs evaluate caller-supplied scenarios; they do not
certify firm rules or reconstruct receipt dates from trading summaries.

## Choose the economic question

- `simulate_cashflow_sequences`: cash paid/received through an elapsed horizon,
  with unlimited external funding for every retry.
- `simulate_wallet_sequences`: one active attempt at a time, finite initial
  wallet, no borrowing or deposits, and an explicit retry policy.
- `r_renewal` / `r_path`: legacy cadence-based rate diagnostics, not these
  finite-horizon wallet distributions.

The wallet variant currently requires the **same known total upfront fee** for
every possible attempt outcome. It rejects later fees and outcome-dependent
entry prices. Conditional activation, recurring billing, resets/reactivations,
price changes, taxes and bank fees need a stateful lifecycle model; do not hide
them in a zero or move them to entry merely to satisfy this restriction.

## Example: delayed receipts cannot pay today's entry fee

This synthetic example is not a Lucid, Topstep, Alpha or Tradeify preset.

```python
from datetime import timedelta
from propfirm_engine import (
    AttemptCashflows, Cashflow, simulate_wallet_sequences,
)

attempt = AttemptCashflows(
    duration=timedelta(days=7),  # includes any retry cooldown
    events=(
        Cashflow(timedelta(0), -100, "fee"),
        Cashflow(timedelta(days=14), 300, "receipt"),
    ),
)
out = simulate_wallet_sequences(
    [attempt], timedelta(days=21), initial_balance=100,
    retry_policy="wait_for_receipts", n_sequences=1, seed=0,
)
assert out.attempts_started[0] == 2   # days 0 and 14, not day 7
assert out.ending_balance[0] == 200
assert out.net_cashflow[0] == 100    # excludes the initial wallet deposit
```

The account ends after seven elapsed days, but its first cash receipt arrives on
day 14. With no fee money on day 7, the wallet waits. The second receipt falls
after the horizon and is excluded. Receipt events are cash reaching the trader,
not trading profit, eligibility, withdrawal requests or approvals.

## Boundaries and outputs

`retry_policy="stop"` permanently abandons new attempts at the first funding
shortfall; already-earned pending receipts still settle through the horizon.
`"wait_for_receipts"` waits only for pending receipts from started attempts.
Neither policy looks at future sampled outcomes to decide whether entry is
affordable. Every outcome is sampled IID only after the fee can be paid.

Attempts start strictly before the horizon; cash events exactly at it count.
Receipts at an attempt's start cannot finance that same attempt's fee.
Early receipts do not permit overlapping attempts. Duration must include all
caller-assumed nontrading delay before the next attempt can begin.

Outputs retain net cashflow, positive fees paid, receipts, attempts started,
ending and minimum wallet balances, and `funding_shortfall` (whether any entry
decision lacked funds, even if later funded). Arrays are read-only on return.
Assumptions, horizon, currency, initial balance and retry policy accompany them.
Affordability uses exact arithmetic on supplied decimal representations; there
is no assumed currency rounding rule. Reporting arrays use float64.

Use an objective on the desired result distribution explicitly: expected cash,
loss probability, or a defined tail statistic are different decisions. This
layer imposes no survival threshold and is not yet wired to the trading-policy
optimizer. An engine-generated, dated event ledger is still required for that
integration. Unknown cost or settlement assumptions remain unknown.
