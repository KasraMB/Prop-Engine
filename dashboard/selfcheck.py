"""Standalone sanity scenarios for the LucidFlex dashboard bridge.

Run: ``python dashboard/selfcheck.py``. Every number is from the real reference
simulator; the LucidFlex payout rules (5 qualifying days, day-after, caps, 90/10)
are enforced in the bridge.
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bridge import evaluate


def T(pnl, low=None):
    return {"type": "trade", "pnl": pnl, "low": low}


def P(amount):
    return {"type": "payout", "amount": amount}


def show(title, size, role, days):
    s = evaluate("Lucid", "LucidFlex", size, role, days)["snapshot"]
    print(f"\n=== {title} ({size} {role}) ===")
    print(f"  status : {s['status_label']}  [{s['code_name']}]")
    print(f"  balance {s['balance']:.0f}  total P&L {s['total_pnl']:.0f}  MLL floor {s['mll_floor']:.0f}"
          f"{'  LOCKED' if s['mll_locked'] else ''}")
    if s["profit_target_level"]:
        print(f"  target {s['profit_target_level']:.0f}  consistency "
              f"{('%.0f%%' % (100*s['consistency_ratio'])) if s['consistency_ratio'] is not None else 'n/a'}"
              f" (limit {int(100*s['consistency_limit'])}%)")
    if s["payout"]["enabled"]:
        p = s["payout"]
        print(f"  qual days {p['qual_days']}/{p['qual_needed']} (min ${p['min_daily']:.0f}/day)"
              f"  total profit {p['total_profit']:.0f}  cycle {p['cycle_profit']:.0f}")
        print(f"  can request: {p['can_request']}  range ${p['min']:.0f}-${p['max']:.0f}"
              f"  paid {p['payouts_taken']}/{p['max_payouts']}  received(net) {p['total_received']:.0f}"
              + (f"  [{p['reason']}]" if p['reason'] else ""))


print("Driving the REAL reference simulator; LucidFlex payout rules enforced by the bridge.")
print("=" * 66)

# EVAL (50K: target 3000, 50% consistency, 2k MLL locking at 50,100)
show("Two balanced days clear the 3k target", "50K", "eval", [[T(1500)], [T(1500)]])
show("Target hit but one day is 75% of profit -> consistency blocks", "50K", "eval", [[T(750)], [T(2250)]])
show("Climb to 52,100 -> MLL locks at 50,100", "50K", "eval", [[T(2100)], [T(1000)]])

# FUNDED (50K: 5 qual days >= $150, day-after, min $500, max min($2000, 50% total))
show("4 qualifying days (closed) — not yet eligible", "50K", "funded",
     [[T(200)], [T(200)], [T(200)], [T(200)], []])
show("5 qualifying days closed, now on day 6 -> eligible", "50K", "funded",
     [[T(200)], [T(200)], [T(200)], [T(200)], [T(200)], []])
show("5th qualifying day is the CURRENT day -> NOT eligible (day-after)", "50K", "funded",
     [[T(200)], [T(200)], [T(200)], [T(200)], [T(200)]])
show("Eligible, then take a $500 payout (net $450, cycle+quals reset)", "50K", "funded",
     [[T(300)], [T(300)], [T(300)], [T(300)], [T(300)], [P(500)]])

# 25K funded: min daily $100, cap $1,000
show("25K: 5 days >= $100, next day eligible (max = 50% of total, capped $1,000)", "25K", "funded",
     [[T(150)], [T(150)], [T(150)], [T(150)], [T(150)], []])
