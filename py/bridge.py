"""Bridge between the dashboard UI and the REAL engine (LucidFlex).

Every rule decision — the trailing/locking MLL, the eval profit target, and the
50% consistency gate — is computed by the actual
:class:`propfirm_engine.reference._ReferenceSim`. Nothing here re-implements a
rule the engine owns.

What this layer models (LucidFlex's payout mechanic, which the auto-firing engine
schema cannot express):

* **Manual payouts.** The trader chooses the amount; it is injected into the
  engine path as a **withdrawal** so the real MLL sees the reduced balance.
* **Eligibility:** 5 **qualifying days** (a *closed* day whose P&L ≥ the size's
  minimum daily profit, reset after each payout) **and** positive cycle profit —
  and, crucially, a payout may be requested **only the day after** those are met
  (so the qualifying days are counted over *closed* days, never the in-progress one).
* **Amount:** $500 – min(per-size cap, 50% of total profit). **Split 90/10** — the
  requested amount leaves the account; the trader receives 90%. Up to 5 payouts.

"Held" end-of-day state: the displayed MLL floor is computed over the *committed*
days so it moves only on "Next day".
"""

from __future__ import annotations

import os
import sys

import numpy as np


from propfirm_engine.compiler import compile_phase  # noqa: E402
from propfirm_engine.enums import ExitCode  # noqa: E402
from propfirm_engine.reference import _ReferenceSim  # noqa: E402
from propfirm_engine.rules import RuleKind  # noqa: E402

from accounts import get_account, payout_params  # noqa: E402

_CODE_NAMES = {int(c): c.name for c in ExitCode}


def _phase_by_role(account, role):
    for p in account.phases:
        if p.role == role:
            return p
    raise KeyError(role)


def _events_to_arrays(days):
    """Flatten day events into (ret, day, trade_low). A payout event becomes a
    withdrawal (negative return, no floating low). Empty days skipped, re-indexed."""
    ret, day, low = [], [], []
    d = 0
    for events in days:
        if not events:
            continue
        for e in events:
            if e.get("type") == "payout":
                ret.append(-abs(float(e["amount"])))
                day.append(d)
                low.append(0.0)
            else:
                pnl = float(e["pnl"])
                ret.append(pnl)
                day.append(d)
                lo = e.get("low")
                low.append(float(lo) if lo is not None else min(pnl, 0.0))
        d += 1
    return (np.asarray(ret, np.float64), np.asarray(day, np.int32),
            np.asarray(low, np.float64))


def _run(cp, days, start_equity, trace=False):
    ret, day, low = _events_to_arrays(days)
    sim = _ReferenceSim(cp, 1.0, np.array([1.0]), float(start_equity), trace=trace)
    res = sim.run(ret, day, low)
    return sim, res


def _requirements(cp):
    req = {"profit_target": None, "consistency": None, "mll_amount": None, "lock_at": None}
    for i in range(cp.n_rules):
        k = int(cp.kind[i])
        if k == int(RuleKind.PROFIT_TARGET):
            req["profit_target"] = float(cp.p0[i])
        elif k == int(RuleKind.CONSISTENCY_GATE):
            req["consistency"] = float(cp.p0[i])
        elif k == int(RuleKind.TRAILING_DD):
            req["mll_amount"] = float(cp.p0[i])
            req["lock_at"] = float(cp.lock_at)
    return req


def _status(code):
    c = int(code)
    if c in (int(ExitCode.ALIVE), int(ExitCode.TIMED_OUT)):
        return "in_progress", "In progress"
    if c == int(ExitCode.PASSED):
        return "passed", "PASSED ✅ — eval cleared"
    if c == int(ExitCode.MAXED_OUT):
        return "complete", "COMPLETE \U0001F389"
    labels = {int(ExitCode.FAIL_TRAILING_DD): "FAILED ❌ — Max Loss Limit breached",
              int(ExitCode.FAIL_STATIC_DD): "FAILED ❌ — max loss breached",
              int(ExitCode.FAIL_DAILY_LOSS): "FAILED ❌ — daily loss breached"}
    return "failed", labels.get(c, f"FAILED ❌ ({_CODE_NAMES.get(c, c)})")


def _lucid_payout_state(days, params):
    """Walk the event list and compute LucidFlex payout state. Qualifying days and
    the positive-cycle requirement are counted over CLOSED days only (the last
    non-empty day is in progress) — this is what enforces "day after"."""
    total_trading = 0.0
    total_gross = 0.0
    total_net = 0.0
    cycle_all = 0.0  # trading P&L since last payout, incl. the current day
    cycle_closed = 0.0  # ...over closed days only
    qual_closed = 0  # closed qualifying days since last payout
    payouts_taken = 0
    # The in-progress day is the LAST element of `days` (even if empty); every day
    # before it is closed. Qualifying days and the day-after positive-cycle are
    # counted over closed days only, so a payout is requestable only the day after.
    last_idx = len(days) - 1
    for idx, events in enumerate(days):
        is_current = idx == last_idx
        day_trading = 0.0
        for e in events:
            if e.get("type") == "payout":
                amt = abs(float(e["amount"]))
                total_gross += amt
                total_net += amt * params["split"]
                payouts_taken += 1
                cycle_all = 0.0
                cycle_closed = 0.0
                qual_closed = 0  # payout resets the cycle + qualifying days
            else:
                p = float(e["pnl"])
                day_trading += p
                total_trading += p
                cycle_all += p
        if events and not is_current:  # a day counts only once closed
            cycle_closed += day_trading
            if day_trading >= params["min_daily"]:
                qual_closed += 1

    max_amount = min(params["cap"], params["pct"] * total_trading)
    reason = ""
    if payouts_taken >= params["max_payouts"]:
        reason = "reached the 5-payout limit"
    elif qual_closed < params["qualifying_days"]:
        reason = f"{qual_closed}/{int(params['qualifying_days'])} qualifying days (need them on CLOSED days)"
    elif cycle_closed < params["min_cycle"]:
        reason = "cycle profit must be positive (≥ $1) as of a prior day"
    elif max_amount < params["min_payout"]:
        reason = f"50% of total profit (${max_amount:.0f}) is below the ${params['min_payout']:.0f} minimum"
    eligible = reason == ""
    return dict(total_trading=total_trading, total_gross_paid=total_gross,
                total_net_received=total_net, cycle_profit=cycle_all,
                qual_days=qual_closed, qualifying_needed=int(params["qualifying_days"]),
                payouts_taken=payouts_taken, max_payouts=int(params["max_payouts"]),
                max_amount=max_amount, min_amount=params["min_payout"],
                split=params["split"], eligible=eligible, reason=reason)


def evaluate(firm, atype, size, role, days):
    account = get_account(firm, atype, size)
    cp = compile_phase(_phase_by_role(account, role))
    req = _requirements(cp)
    start = float(account.size)

    non_empty = [d for d in days if d]
    committed = days[:-1] if days else []

    full_sim, full_res = _run(cp, days, start, trace=True)
    committed_sim, _ = _run(cp, committed, start)
    status, label = _status(full_res.code)

    params = payout_params(size)
    pstate = _lucid_payout_state(days, params) if role == "funded" else None
    # balance from the event arithmetic (engine equity matches: withdrawals injected)
    total_trading = pstate["total_trading"] if pstate else float(np.sum([
        float(e["pnl"]) for d in days for e in d if e.get("type") != "payout"]))
    total_paid = pstate["total_gross_paid"] if pstate else 0.0
    balance = start + total_trading - total_paid
    cur_day_trading = sum(float(e["pnl"]) for e in (days[-1] if days else [])
                          if e.get("type") != "payout")

    consistency_ratio = None
    if req["consistency"] is not None:
        cyc = full_sim.equity - full_sim.cycle_start_equity
        if full_sim.max_day_pnl > 0 and cyc > 0:
            consistency_ratio = full_sim.max_day_pnl / cyc

    payout = {"enabled": role == "funded"}
    if pstate:
        payout.update({
            "can_request": pstate["eligible"] and status == "in_progress",
            "min": pstate["min_amount"], "max": pstate["max_amount"],
            "reason": pstate["reason"], "total_profit": pstate["total_trading"],
            "cycle_profit": pstate["cycle_profit"], "total_paid": pstate["total_gross_paid"],
            "total_received": pstate["total_net_received"],
            "qual_days": pstate["qual_days"], "qual_needed": pstate["qualifying_needed"],
            "payouts_taken": pstate["payouts_taken"], "max_payouts": pstate["max_payouts"],
            "min_daily": params["min_daily"], "split": pstate["split"],
        })

    target_level = (start + cp.profit_target0) if req["profit_target"] else None
    snap = {
        "status": status, "status_label": label,
        "code_name": _CODE_NAMES.get(int(full_res.code), str(full_res.code)),
        "balance": balance, "total_pnl": total_trading, "day_pnl": cur_day_trading,
        "n_days": len(non_empty),
        "mll_floor": committed_sim.dd_floor,
        "mll_distance": balance - committed_sim.dd_floor,
        "mll_locked": bool(committed_sim.dd_locked),
        "mll_amount": req["mll_amount"], "lock_at": req["lock_at"],
        "profit_target_level": target_level,
        "profit_target_distance": (target_level - balance) if target_level else None,
        "consistency_ratio": consistency_ratio, "consistency_limit": req["consistency"],
        "requirements": req, "payout": payout,
    }
    return {"snapshot": snap, "days_log": _days_log(cp, non_empty, start, params, role),
            "calendar": _calendar(days, params, role),
            "equity_series": _equity_series(full_res, start, target_level, req)}


def _days_log(cp, non_empty, start, params, role):
    log = []
    for k in range(1, len(non_empty) + 1):
        s, r = _run(cp, non_empty[:k], start)
        st, _ = _status(r.code)
        events = non_empty[k - 1]
        trades = sum(1 for e in events if e.get("type") != "payout")
        day_trading = sum(float(e["pnl"]) for e in events if e.get("type") != "payout")
        pays = sum(abs(float(e["amount"])) for e in events if e.get("type") == "payout")
        qualifies = role == "funded" and trades and day_trading >= params["min_daily"]
        log.append({"day": k - 1, "trades": trades, "day_pnl": day_trading,
                    "balance": s.equity, "mll_floor": s.dd_floor,
                    "payout": pays, "qualifies": bool(qualifies),
                    "status": st, "code_name": _CODE_NAMES.get(int(r.code), str(r.code))})
    return log


def _calendar(days, params, role):
    cells = []
    d = 0
    for events in days:
        if not events:
            continue
        trading = sum(float(e["pnl"]) for e in events if e.get("type") != "payout")
        paid = sum(abs(float(e["amount"])) for e in events if e.get("type") == "payout")
        cells.append({"day": d, "week": d // 5, "weekday": d % 5, "pnl": trading,
                      "payout": paid,
                      "qualifies": bool(role == "funded" and trading >= params["min_daily"])})
        d += 1
    return cells


def _equity_series(res, start, target_level, req):
    pts = [{"i": 0, "equity": start, "floor": start - (req["mll_amount"] or 0)}]
    for e in res.trace:
        pts.append({"i": e["t"] + 1, "equity": e["equity"], "floor": e["dd_floor"]})
    return {"points": pts, "target": target_level}
