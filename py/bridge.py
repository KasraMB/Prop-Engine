"""Bridge between the dashboard UI and the REAL engine.

Every rule decision (the trailing/locking MLL, the profit target, the eval
consistency gate, and pass/fail) is computed by the actual
:class:`propfirm_engine.reference._ReferenceSim` — the oracle the fast kernel is
proven against — run over the trades you enter. Nothing here re-implements a rule.

Two things this layer models itself, clearly separated from the engine:

* **Manual payouts (funded).** The engine's ``PayoutSchema`` auto-fires payouts;
  the spec here is a *manual* payout (you choose the amount and when), bounded by
  \\$500–\\$2,000, at most 50% of total profit, requiring cycle profit ≥ \\$1. A
  requested payout is injected into the engine path as a **withdrawal** (a
  negative "trade") so the real MLL correctly sees the reduced balance; the
  amount/timing and the bounds are enforced here.
* **"Held" end-of-day state.** An EOD Max Loss Limit only moves at day close, so
  the displayed floor is computed over the *committed* days (everything before the
  in-progress day) and updates only when you click "Next day" — while the live
  balance and status reflect the in-progress day too.

Day/event model: ``days`` is a list of days; each day is a list of events, either
``{"type":"trade","pnl":..,"low":..?}`` or ``{"type":"payout","amount":..}``.
"""

from __future__ import annotations


import numpy as np


from propfirm_engine.compiler import compile_phase  # noqa: E402
from propfirm_engine.enums import ExitCode  # noqa: E402
from propfirm_engine.reference import _ReferenceSim  # noqa: E402
from propfirm_engine.rules import RuleKind  # noqa: E402

from accounts import (  # noqa: E402
    MIN_CYCLE_PROFIT,
    PAYOUT_MAX,
    PAYOUT_MIN,
    PAYOUT_PCT_OF_TOTAL,
    get_account,
)

_CODE_NAMES = {int(c): c.name for c in ExitCode}


def _phase_by_role(account, role):
    for p in account.phases:
        if p.role == role:
            return p
    raise KeyError(role)


def _events_to_arrays(days):
    """Flatten day events into (ret, day, trade_low). A payout event becomes a
    withdrawal: a negative return with no floating low. Empty days are skipped and
    the rest re-indexed 0..D-1."""
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
    return (
        np.asarray(ret, np.float64),
        np.asarray(day, np.int32),
        np.asarray(low, np.float64),
    )


def _run(cp, days, start_equity, trace=False):
    ret, day, low = _events_to_arrays(days)
    sim = _ReferenceSim(cp, 1.0, np.array([1.0]), float(start_equity), trace=trace)
    res = sim.run(ret, day, low)
    return sim, res


def _requirements(cp):
    req = {"profit_target": None, "consistency": None, "mll_amount": None,
           "lock_at": None}
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
    labels = {
        int(ExitCode.FAIL_TRAILING_DD): "FAILED ❌ — Max Loss Limit breached",
        int(ExitCode.FAIL_STATIC_DD): "FAILED ❌ — max loss breached",
        int(ExitCode.FAIL_DAILY_LOSS): "FAILED ❌ — daily loss breached",
    }
    return "failed", labels.get(c, f"FAILED ❌ ({_CODE_NAMES.get(c, c)})")


def _profit_totals(days):
    """Pure arithmetic over the event list (independent of the engine):
    trading P&L totals, withdrawals, cycle profit (profit since the last payout)."""
    total_trading = 0.0  # sum of all trade P&Ls (gross profit, before withdrawals)
    total_paid = 0.0
    cycle_profit = 0.0  # trading P&L since the last payout
    cur_day_trading = 0.0
    for events in days:
        cur_day_trading = 0.0
        for e in events:
            if e.get("type") == "payout":
                total_paid += abs(float(e["amount"]))
                cycle_profit = 0.0  # a payout resets the cycle
            else:
                pnl = float(e["pnl"])
                total_trading += pnl
                cycle_profit += pnl
                cur_day_trading += pnl
    return {
        "total_trading": total_trading,
        "total_paid": total_paid,
        "cycle_profit": cycle_profit,
        "cur_day_trading": cur_day_trading,
    }


def _payout_bounds(totals):
    """The manual-payout rules: [500, min(2000, 50% of total profit)], requiring
    cycle profit >= $1. Returns (can_request, min, max, reason)."""
    total_profit = totals["total_trading"]
    cap = min(PAYOUT_MAX, PAYOUT_PCT_OF_TOTAL * total_profit)
    if totals["cycle_profit"] < MIN_CYCLE_PROFIT:
        return False, PAYOUT_MIN, cap, "cycle profit must be at least $1"
    if cap < PAYOUT_MIN:
        return False, PAYOUT_MIN, cap, f"need 50% of total profit ≥ ${PAYOUT_MIN:.0f} (have ${cap:.0f})"
    return True, PAYOUT_MIN, cap, ""


def evaluate(firm, atype, size, role, days):
    account = get_account(firm, atype, size)
    cp = compile_phase(_phase_by_role(account, role))
    req = _requirements(cp)
    start = float(account.size)

    non_empty = [d for d in days if d]
    committed = days[:-1] if days else []  # everything before the in-progress day

    full_sim, full_res = _run(cp, days, start, trace=True)
    committed_sim, _ = _run(cp, committed, start)

    totals = _profit_totals(days)
    balance = start + totals["total_trading"] - totals["total_paid"]
    status, label = _status(full_res.code)

    # consistency ratio (display): biggest day vs cycle profit
    consistency_ratio = None
    if req["consistency"] is not None and full_sim.max_day_pnl > 0 and (balance - full_sim.cycle_start_equity) != 0:
        cyc = full_sim.equity - full_sim.cycle_start_equity
        if cyc > 0:
            consistency_ratio = full_sim.max_day_pnl / cyc

    # payout availability (funded only)
    can_pay, pmin, pmax, preason = _payout_bounds(totals)
    payout = {
        "enabled": role == "funded",
        "can_request": role == "funded" and can_pay and status == "in_progress",
        "min": pmin, "max": pmax, "reason": preason,
        "total_profit": totals["total_trading"],
        "cycle_profit": totals["cycle_profit"],
        "total_paid": totals["total_paid"],
    }

    target_level = (start + cp.profit_target0) if req["profit_target"] else None
    snap = {
        "status": status, "status_label": label,
        "code_name": _CODE_NAMES.get(int(full_res.code), str(full_res.code)),
        "balance": balance,
        "total_pnl": totals["total_trading"],  # trading P&L (gross of withdrawals)
        "day_pnl": totals["cur_day_trading"],
        "n_days": len(non_empty),
        # held floor: from committed days only, so it updates on Next Day
        "mll_floor": committed_sim.dd_floor,
        "mll_distance": balance - committed_sim.dd_floor,
        "mll_locked": bool(committed_sim.dd_locked),
        "mll_amount": req["mll_amount"], "lock_at": req["lock_at"],
        "profit_target_level": target_level,
        "profit_target_distance": (target_level - balance) if target_level else None,
        "max_day_pnl": full_sim.max_day_pnl,
        "consistency_ratio": consistency_ratio,
        "consistency_limit": req["consistency"],
        "requirements": req,
        "payout": payout,
    }

    return {
        "snapshot": snap,
        "days_log": _days_log(cp, non_empty, start),
        "calendar": _calendar(days),
        "equity_series": _equity_series(full_res, start, target_level, req),
    }


def _days_log(cp, non_empty, start):
    log = []
    for k in range(1, len(non_empty) + 1):
        s, r = _run(cp, non_empty[:k], start)
        st, _ = _status(r.code)
        events = non_empty[k - 1]
        trades = sum(1 for e in events if e.get("type") != "payout")
        pays = [abs(float(e["amount"])) for e in events if e.get("type") == "payout"]
        day_trading = sum(float(e["pnl"]) for e in events if e.get("type") != "payout")
        log.append({
            "day": k - 1, "trades": trades, "day_pnl": day_trading,
            "balance": s.equity, "mll_floor": s.dd_floor,
            "payout": sum(pays) if pays else 0.0,
            "status": st, "code_name": _CODE_NAMES.get(int(r.code), str(r.code)),
        })
    return log


def _calendar(days):
    """Per-trading-day trading P&L laid out for a Mon-Fri weekly grid."""
    cells = []
    d = 0
    for events in days:
        if not events:
            continue
        trading = sum(float(e["pnl"]) for e in events if e.get("type") != "payout")
        paid = sum(abs(float(e["amount"])) for e in events if e.get("type") == "payout")
        cells.append({"day": d, "week": d // 5, "weekday": d % 5,
                      "pnl": trading, "payout": paid})
        d += 1
    return cells


def _equity_series(res, start, target_level, req):
    """Per-step equity + the (step-function) MLL floor, from the real trace."""
    pts = [{"i": 0, "equity": start, "floor": start - (req["mll_amount"] or 0)}]
    for e in res.trace:
        pts.append({"i": e["t"] + 1, "equity": e["equity"], "floor": e["dd_floor"]})
    return {"points": pts, "target": target_level}
