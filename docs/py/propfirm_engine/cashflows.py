"""Explicit cash timing for independent repeated-attempt scenarios.

Offsets/durations are elapsed calendar timedeltas, never inferred from trading
cadence. Only external fee payments and net trader receipts belong here: trading
P&L, payout requests and approvals are NOT spendable receipts. These schedules
must be supplied by a caller/event simulator; legacy Outcomes cannot reconstruct
them. The finite-wallet variant supports known upfront fees only; neither
variant certifies a firm's costs or implements stateful billing.
"""
from dataclasses import dataclass
from datetime import timedelta
from fractions import Fraction
import heapq
import math
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class Cashflow:
    offset: timedelta
    amount: float  # signed external cash: fee <= 0, receipt >= 0
    kind: str  # "fee" or "receipt"

    def __post_init__(self):
        if not isinstance(self.offset, timedelta) or self.offset < timedelta(0):
            raise ValueError("cashflow offset must be a nonnegative elapsed timedelta")
        if isinstance(self.amount, bool) or not isinstance(self.amount, Real) or not np.isfinite(self.amount):
            raise ValueError("cashflow amount must be finite and explicitly known")
        if self.kind not in ("fee", "receipt"):
            raise ValueError("cashflow kind must be fee or receipt; requests/approvals are not cash")
        if (self.kind == "fee" and self.amount > 0) or (self.kind == "receipt" and self.amount < 0):
            raise ValueError("fee cashflows must be nonpositive; receipts must be nonnegative")


@dataclass(frozen=True)
class AttemptCashflows:
    duration: timedelta  # elapsed time until the next attempt may start, including cooldown
    events: tuple[Cashflow, ...]
    currency: str = "USD"

    def __post_init__(self):
        if not isinstance(self.duration, timedelta) or self.duration <= timedelta(0):
            raise ValueError("attempt duration must be a positive elapsed timedelta")
        if not isinstance(self.events, tuple) or any(not isinstance(event, Cashflow) for event in self.events):
            raise ValueError("events must be an immutable tuple of Cashflow values")
        if any(a.offset > b.offset for a, b in zip(self.events, self.events[1:])):
            raise ValueError("cashflow events must be in chronological order")
        if not isinstance(self.currency, str) or not self.currency:
            raise ValueError("currency must be explicitly identified")
        # Receipts after duration are legitimate pending settlements. Their
        # timestamps still govern inclusion while subsequent attempts may run.


@dataclass(frozen=True)
class CashflowSequences:
    net_cashflow: np.ndarray
    fees_paid: np.ndarray
    receipts: np.ndarray
    attempts_started: np.ndarray
    horizon: timedelta
    currency: str
    assumptions: tuple[str, ...] = (
        "IID draws from caller-supplied elapsed-time cashflow schedules",
        "one active attempt; duration includes caller-specified retry delay",
        "unconstrained external capital; affordability is not modeled",
        "receipt timing is supplied, not inferred from approval or trading profit",
    )


def _validated_sequence_inputs(schedules, horizon, n_sequences, seed):
    if not isinstance(horizon, timedelta) or horizon <= timedelta(0):
        raise ValueError("horizon must be a positive elapsed timedelta")
    if isinstance(n_sequences, bool) or not isinstance(n_sequences, Integral) or n_sequences < 1:
        raise ValueError("n_sequences must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    try:
        schedules = tuple(schedules)
    except TypeError as exc:
        raise ValueError("finite-horizon cashflow requires explicit AttemptCashflows schedules; aggregate Outcomes lack receipt/fee timing") from exc
    if not schedules or any(not isinstance(schedule, AttemptCashflows) for schedule in schedules):
        raise ValueError("schedules must contain explicit AttemptCashflows, not aggregate Outcomes")
    currencies = {schedule.currency for schedule in schedules}
    if len(currencies) != 1:
        raise ValueError("cannot combine currencies without an explicit conversion model")
    return schedules


def simulate_cashflow_sequences(schedules, horizon, n_sequences=2000, seed=0):
    """Sample sequential attempts, retaining only cash at/before the horizon.

    Attempts start strictly BEFORE the horizon; events exactly AT it count.
    An attempt crossing it contributes only its already-due fees/receipts.
    Pending receipts may settle after the attempt ends. No per-attempt lump-sum
    proxy or proportional accrual is used. The schedules are scenario inputs,
    not a general stateful retry/affordability simulator.
    """
    schedules = _validated_sequence_inputs(schedules, horizon, n_sequences, seed)
    fees = np.zeros(n_sequences)
    receipts = np.zeros(n_sequences)
    starts = np.zeros(n_sequences, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for sequence in range(n_sequences):
        elapsed = timedelta(0)
        while elapsed < horizon:
            schedule = schedules[int(rng.integers(len(schedules)))]
            starts[sequence] += 1
            remaining = horizon - elapsed
            for event in schedule.events:
                if event.offset > remaining:
                    break
                if event.kind == "fee":
                    fees[sequence] -= event.amount
                else:
                    receipts[sequence] += event.amount
            if schedule.duration >= remaining:
                break
            elapsed += schedule.duration
    net = receipts - fees
    for values in (net, fees, receipts, starts):
        values.flags.writeable = False
    return CashflowSequences(net, fees, receipts, starts, horizon, schedules[0].currency)


@dataclass(frozen=True)
class WalletSequences:
    net_cashflow: np.ndarray
    fees_paid: np.ndarray
    receipts: np.ndarray
    attempts_started: np.ndarray
    ending_balance: np.ndarray
    minimum_balance: np.ndarray
    funding_shortfall: np.ndarray  # any retry/start decision lacked the fee, even if later funded
    initial_balance: float
    horizon: timedelta
    currency: str
    retry_policy: str
    assumptions: tuple[str, ...] = (
        "IID outcomes drawn only after the known upfront fee is affordable",
        "one active attempt; duration includes caller-specified retry delay",
        "all fees upfront at the same known total price on every attempt",
        "no borrowing, new deposits, recurring bills or conditional activation fees",
        "only supplied cash receipts fund retries; pending receipts are not cash",
    )


def simulate_wallet_sequences(schedules, horizon, *, initial_balance, retry_policy,
                              n_sequences=2000, seed=0):
    """Finite-wallet retries for explicit upfront-fee scenarios.

    Policy must be 'stop' (abandon retries at first funding shortfall) or
    'wait_for_receipts' (wait for already-earned pending receipts). Receipts from
    previously started attempts are collected through the inclusive horizon even
    after stopping. Attempts start strictly before it, never overlap, and their
    own offset-zero receipts cannot finance their entry fee.

    All schedule fees must occur at offset zero and have the same total across
    outcomes: drawing an outcome must not reveal a future conditional entry price.
    Mid-attempt fees need a stateful billing/activation model and are rejected.
    This scenario layer does not reconstruct cash timing from Engine Outcomes.
    Affordability uses exact rational arithmetic over the decimal strings of
    supplied amounts, with no currency rounding. Report arrays are float64.
    """
    schedules = _validated_sequence_inputs(schedules, horizon, n_sequences, seed)
    if (isinstance(initial_balance, bool) or not isinstance(initial_balance, Real)
            or not math.isfinite(initial_balance) or initial_balance < 0):
        raise ValueError("initial_balance must be a finite nonnegative real amount")
    if retry_policy not in ("stop", "wait_for_receipts"):
        raise ValueError("retry_policy must be 'stop' or 'wait_for_receipts'")
    prices = []
    for schedule in schedules:
        if any(event.kind == "fee" and event.offset != timedelta(0)
               for event in schedule.events):
            raise ValueError("wallet schedules require all fees upfront; later fees need stateful billing")
        price = sum((-Fraction(str(event.amount)) for event in schedule.events
                     if event.kind == "fee"), Fraction(0))
        prices.append(price)
    upfront = prices[0]
    if any(price != upfront for price in prices):
        raise ValueError("all outcomes must have the same known upfront fee")
    receipt_events = [tuple((event.offset, Fraction(str(event.amount)))
                            for event in schedule.events if event.kind == "receipt")
                      for schedule in schedules]

    def reported_cash(value):
        try:
            reported = float(value)
        except OverflowError as exc:
            raise ValueError("wallet cash arithmetic exceeds float64 reporting range") from exc
        if not math.isfinite(reported):
            raise ValueError("wallet cash arithmetic exceeds float64 reporting range")
        return reported

    fees = np.zeros(n_sequences)
    receipts = np.zeros(n_sequences)
    starts = np.zeros(n_sequences, dtype=np.int64)
    ending = np.zeros(n_sequences)
    net = np.zeros(n_sequences)
    minimum = np.full(n_sequences, float(initial_balance))
    shortfall = np.zeros(n_sequences, dtype=bool)
    rng = np.random.default_rng(seed)
    for sequence in range(n_sequences):
        cash = Fraction(str(initial_balance))
        minimum_cash = cash
        received = Fraction(0)
        paid = Fraction(0)
        pending = []
        elapsed = timedelta(0)

        def receive_through(cutoff):
            nonlocal cash, received
            while pending and pending[0][0] <= cutoff:
                _, amount = heapq.heappop(pending)
                cash += amount
                received += amount

        while elapsed < horizon:
            receive_through(elapsed)
            if cash < upfront:
                shortfall[sequence] = True
                if retry_policy == "stop":
                    break
                while cash < upfront and pending:
                    elapsed = pending[0][0]
                    receive_through(elapsed)
                if cash < upfront or elapsed >= horizon:
                    break
            # The affordability decision never looks at an unstarted outcome.
            schedule_index = int(rng.integers(len(schedules)))
            schedule = schedules[schedule_index]
            cash -= upfront
            paid += upfront
            minimum_cash = min(minimum_cash, cash)
            starts[sequence] += 1
            remaining = horizon - elapsed
            for offset, amount in receipt_events[schedule_index]:
                if offset > remaining:
                    break
                heapq.heappush(pending, (elapsed + offset, amount))
            elapsed = horizon if schedule.duration >= remaining else elapsed + schedule.duration

        receive_through(horizon)
        ending[sequence] = reported_cash(cash)
        receipts[sequence] = reported_cash(received)
        fees[sequence] = reported_cash(paid)
        net[sequence] = reported_cash(received - paid)
        minimum[sequence] = reported_cash(minimum_cash)
    for values in (net, fees, receipts, starts, ending, minimum, shortfall):
        values.flags.writeable = False
    return WalletSequences(net, fees, receipts, starts, ending, minimum, shortfall,
                           float(initial_balance), horizon, schedules[0].currency, retry_policy)


__all__ = ["Cashflow", "AttemptCashflows", "CashflowSequences", "simulate_cashflow_sequences",
           "WalletSequences", "simulate_wallet_sequences"]
