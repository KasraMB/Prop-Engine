"""Dated payout accounting, integrated by the chronological replay adapter.

This ledger consumes realized net trades and explicit session-close/approval/
receipt events. It does not infer market paths, approval delays or live value.
Request policy: flat, after a finalized session; no trading while pending.
Amounts use exact rational decimal inputs. No payment rounding is invented.
"""
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from fractions import Fraction
from numbers import Integral, Real
import math

from .cashflows import Cashflow
from .enums import StateField
from .model import Account, Phase
from .rules import MinimumWinningDaysRule
from .schema import PayoutSchema
from .validate import validate


def _money(value, name, *, nonnegative=False):
    try:
        valid = (not isinstance(value, bool) and isinstance(value, Real)
                 and math.isfinite(value))
    except OverflowError:
        valid = False
    if not valid or (nonnegative and value < 0):
        raise ValueError(f"{name} must be finite real" + (" and nonnegative" if nonnegative else ""))
    return value if isinstance(value, Fraction) else Fraction(str(value))


def _utc(at):
    if not isinstance(at, datetime) or at.tzinfo is None or at.utcoffset() is None:
        raise ValueError("event time must be a timezone-aware datetime")
    return at.astimezone(timezone.utc)


@dataclass(frozen=True)
class PayoutEvent:
    at: datetime
    kind: str
    request_id: int | None
    amount: Fraction
    balance: Fraction
    floor: Fraction | None


@dataclass(frozen=True)
class PayoutRequest:
    request_id: int
    requested_at: datetime
    gross: Fraction
    net: Fraction
    status: str = "pending"
    approved_at: datetime | None = None
    received_at: datetime | None = None


class PayoutLedger:
    """Accounting component, not a complete prop-firm simulator.

    Caller must provide actual/scenario clocks and session IDs. Request-time
    floor changes are exposed for the execution engine to enforce against MTM.
    This component only detects floor breaches in its observed realized balance.
    Payout-count exhaustion is explicit research censoring, not a failed account.
    """

    def __init__(self, schema: PayoutSchema, *, opening_balance, qualifying_days,
                 winning_day_profit, initial_floor=None, lock_floor_on_request=None):
        validate(Account("payout-ledger", 1, (Phase("funded", "funded",
            (MinimumWinningDaysRule(qualifying_days, winning_day_profit),), schema),)))
        if schema.recompute_floor_on_payout:
            raise ValueError("recompute_floor_on_payout is unsupported by the dated ledger")
        if any(f != StateField.N_QUALIFYING_DAYS for f in schema.reset_fields):
            raise ValueError("dated ledger supports only qualifying-day counter resets")
        self.schema = schema
        self._balance = _money(opening_balance, "opening_balance")
        self._reference = (self._balance if schema.profit_reference_balance is None
                           else _money(schema.profit_reference_balance, "profit_reference_balance"))
        self._floor = None if initial_floor is None else _money(initial_floor, "initial_floor")
        self._lock_floor = (None if lock_floor_on_request is None
                            else _money(lock_floor_on_request, "lock_floor_on_request"))
        if self._floor is not None and self._balance <= self._floor:
            raise ValueError("opening balance must exceed initial floor")
        if self._floor is not None and self._lock_floor is not None and self._lock_floor < self._floor:
            raise ValueError("request lock may not lower the floor")
        self._required_days = int(qualifying_days)
        self._winning_threshold = _money(winning_day_profit, "winning_day_profit")
        self._qualifying = 0
        self._cycle_profit = Fraction(0)
        self._session_profit = Fraction(0)
        self._dirty_session = False
        self._last_session = None
        self._last_at = None
        self._pending = None
        self._requests = {}
        self._events = []
        self._cumulative_gross = Fraction(0)
        self._approved_count = 0

    @property
    def balance(self):
        return self._balance

    @property
    def floor(self):
        return self._floor

    @property
    def cycle_profit(self):
        return self._cycle_profit

    @property
    def qualifying_days(self):
        return self._qualifying

    @property
    def events(self):
        return tuple(self._events)

    @property
    def requests(self):
        return tuple(self._requests.values())

    @property
    def pending(self):
        return self._pending

    @property
    def breached(self):
        return self._floor is not None and self._balance <= self._floor

    @property
    def censored(self):
        return self._approved_count >= self.schema.max_payouts

    def _time(self, at):
        at = _utc(at)
        if self._last_at is not None and at < self._last_at:
            raise ValueError("events must be chronological")
        return at

    def _emit(self, at, kind, amount=Fraction(0), request_id=None):
        self._last_at = at
        self._events.append(PayoutEvent(at, kind, request_id, amount, self._balance, self._floor))

    def record_trade(self, at, net_pnl):
        at = self._time(at)
        pnl = _money(net_pnl, "net_pnl")
        if self._pending is not None or self.breached or self.censored:
            raise ValueError("trading is blocked: pending payout, breach or research censoring")
        self._balance += pnl
        self._cycle_profit += pnl
        self._session_profit += pnl
        self._dirty_session = True
        self._emit(at, "trade", pnl)

    def advance_floor(self, at, floor):
        """Apply an executor's EOD ratchet; a floor must never move downward."""
        at = self._time(at)
        floor = _money(floor, "floor")
        if self._floor is not None and floor < self._floor:
            raise ValueError("drawdown floor cannot decrease")
        self._floor = floor
        self._emit(at, "floor_update")

    def close_session(self, at, session):
        at = self._time(at)
        if not isinstance(session, date) or isinstance(session, datetime):
            raise ValueError("session must be an explicit date identifier")
        if self._last_session is not None and session <= self._last_session:
            raise ValueError("session identifiers must strictly increase")
        if self._dirty_session and self._session_profit >= self._winning_threshold:
            self._qualifying += 1
        self._session_profit = Fraction(0)
        self._dirty_session = False
        self._last_session = session
        self._emit(at, "session_close")

    def maximum_request(self):
        s = self.schema
        if (self._pending is not None or self.breached or self.censored
                or self._dirty_session or self._qualifying < self._required_days
                or self._cycle_profit < _money(s.min_cycle_profit, "min_cycle_profit")):
            return Fraction(0)
        basis = self._balance - self._reference if s.fraction_basis == "retained_profit" else self._cycle_profit
        cap = s.dollar_cap_at(self._approved_count)
        gross = _money(s.cap_fraction, "cap_fraction") * basis
        if math.isfinite(cap):
            gross = min(gross, _money(cap, "dollar_cap"))
        gross = min(gross, self._balance - _money(s.buffer_floor, "buffer_floor"))
        return gross if gross > 0 and gross >= _money(s.min_request, "min_request") else Fraction(0)

    def request(self, at, gross, *, flat):
        at = self._time(at)
        gross = _money(gross, "gross", nonnegative=True)
        if flat is not True:
            raise ValueError("requests require the executor to confirm no open position")
        if gross <= 0 or gross < _money(self.schema.min_request, "min_request") or gross > self.maximum_request():
            raise ValueError("request does not meet eligibility or amount limits")
        s = self.schema
        first = (min(gross, max(Fraction(0), _money(s.split_tier_cap, "split_tier_cap")
                               - self._cumulative_gross)) if s.split_first_tier is not None else Fraction(0))
        net = first * _money(s.split_first_tier, "first split") if first else Fraction(0)
        net += (gross - first) * _money(s.split, "split")
        rid = len(self._requests) + 1
        self._requests[rid] = PayoutRequest(rid, at, gross, net)
        self._pending = rid
        if self._lock_floor is not None:
            self._floor = self._lock_floor if self._floor is None else max(self._floor, self._lock_floor)
        self._emit(at, "request", gross, rid)
        return rid

    def approve(self, at):
        at = self._time(at)
        if self._pending is None:
            raise ValueError("no pending payout")
        rid = self._pending
        req = self._requests[rid]
        if self.schema.withdraw_reduces_equity:
            self._balance -= req.gross
        self._cumulative_gross += req.gross
        self._approved_count += 1
        self._cycle_profit = Fraction(0)
        if any(f == StateField.N_QUALIFYING_DAYS for f in self.schema.reset_fields):
            self._qualifying = 0
        self._requests[rid] = replace(req, status="approved", approved_at=at)
        self._pending = None
        self._emit(at, "approval", -req.gross if self.schema.withdraw_reduces_equity else Fraction(0), rid)

    def reject(self, at):
        at = self._time(at)
        if self._pending is None:
            raise ValueError("no pending payout")
        rid = self._pending
        self._requests[rid] = replace(self._requests[rid], status="rejected")
        self._pending = None
        # Request-time floor lock remains; no withdrawal or cycle reset occurred.
        self._emit(at, "rejection", request_id=rid)

    def receive(self, at, request_id, *, payment_fee=0):
        at = self._time(at)
        fee = _money(payment_fee, "payment_fee", nonnegative=True)
        if isinstance(request_id, bool) or not isinstance(request_id, Integral):
            raise ValueError("request_id must be an integer")
        req = self._requests.get(request_id)
        if req is None or req.status != "approved" or fee > req.net:
            raise ValueError("receipt requires an unpaid approved request and fee <= net")
        self._requests[request_id] = replace(req, status="received", received_at=at)
        self._emit(at, "receipt", req.net - fee, request_id)

    def receipt_cashflows(self, origin):
        """Only observed receipts become spendable cash; approvals are excluded."""
        origin = _utc(origin)
        receipts = tuple(e for e in self._events if e.kind == "receipt")
        if any(e.at < origin for e in receipts):
            raise ValueError("cashflow origin must not follow a receipt")
        return tuple(Cashflow(e.at - origin, float(e.amount), "receipt") for e in receipts)
