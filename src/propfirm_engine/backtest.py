"""Chronological lifecycle adapter over the existing reference rule interpreter.

No resampling, strategy construction or stop/target retuning happens here.
The fast Monte Carlo API remains available separately via Engine.run.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from heapq import heappop, heappush
from itertools import groupby
from fractions import Fraction
from hashlib import sha256
from math import isfinite
from zoneinfo import ZoneInfo

import numpy as np

from .compiler import compile_account
from .enums import ExitCode, Severity, Timing
from .execution import BacktestConfig, BracketHistory, DollarPolicy, LifecycleSpec
from .feasibility import project_position
from .payouts import PayoutLedger
from .reference import _ReferenceSim
from .rules import RuleKind
from .validate import validate


@dataclass(frozen=True)
class BacktestEvent:
    at: datetime
    kind: str
    attempt: int
    phase: str
    balance: float
    floor: float
    quantity: int = 0
    cash: float = 0.0
    regime: str | None = None
    code: str | None = None
    qualifying_days: int = 0
    cycle_profit: float = 0.0
    gross_payout: float = 0.0


@dataclass(frozen=True)
class BacktestResult:
    """Observed cash economics, NOT an unbiased population-EV estimate."""

    events: tuple[BacktestEvent, ...]
    start: datetime
    end: datetime
    attempts: int
    failed_attempts: int
    status: str
    final_balance: float | None
    outstanding_payouts: float
    assumptions: tuple[str, ...]
    spec: LifecycleSpec
    policy: DollarPolicy
    config: BacktestConfig
    history_fingerprint: str

    @property
    def net_cash(self):
        return float(sum((Fraction(str(e.cash)) for e in self.events), Fraction(0)))

    @property
    def receipts(self):
        return sum(e.cash for e in self.events if e.kind == "receipt")

    @property
    def fees(self):
        return -sum(e.cash for e in self.events if e.kind == "fee")

    @property
    def calendar_days(self):
        return (self.end - self.start).total_seconds() / 86400

    @property
    def net_cash_per_day(self):
        return self.net_cash / self.calendar_days


def _check_support(spec):
    validate(spec.account)
    compiled = compile_account(spec.account)
    allowed = {RuleKind.TRAILING_DD, RuleKind.PROFIT_TARGET,
               RuleKind.CONSISTENCY_GATE, RuleKind.MIN_WINNING_DAYS, RuleKind.MIN_DAYS}
    for p in compiled.phases:
        if not isfinite(p.dd_amount) or p.dd_update_timing != Timing.EOD:
            raise ValueError("bracket replay currently requires an EOD trailing floor in each phase")
        for i in range(p.n_rules):
            kind = RuleKind(int(p.kind[i]))
            if kind not in allowed:
                raise ValueError(f"bracket replay does not support {kind.name}")
            if kind == RuleKind.TRAILING_DD and (
                p.severity[i] != Severity.HARD or p.check_timing[i] != Timing.CONTINUOUS
            ):
                raise ValueError("bracket replay requires hard continuous drawdown checks")
            if kind == RuleKind.CONSISTENCY_GATE and p.role != "eval":
                raise ValueError("funded consistency is not supported by the dated payout ledger")
        if p.role == "funded" and p.payout is None:
            raise ValueError("funded replay requires an explicit payout schema")
    return compiled


class _Replay:
    """Coordinates existing rules, feasibility projection and dated payout accounting."""

    def __init__(self, spec, history, policy, config):
        self.spec, self.history, self.policy, self.config = spec, history, policy, config
        compiled = _check_support(spec)
        self.phases = {p.role: p for p in compiled.phases}
        self.source = {p.role: p for p in spec.account.phases}
        self.tz = ZoneInfo(spec.session_timezone)
        self.events, self.queue, self.ledgers = [], [], []
        self.serial = self.attempts = self.failures = 0
        self.sim = self.ledger = None
        self.role = "eval" if "eval" in self.phases else "funded"
        self.next_role = self.role
        self.available_at = history.trades[0].entry_at
        self.wallet = (None if config.initial_wallet is None
                       else Fraction(str(config.initial_wallet)))
        self.status = "HORIZON"
        self.handoff = False
        self.next_fee = spec.account.eval_fee
        self.last_ended_session = None
        self.failed_at = None
        self.activity_epoch = 0
        # The horizon consists of complete declared sessions, not N sessions / 5.
        first, last = history.sessions[0], history.sessions[-1]
        self.start = datetime.combine(first - timedelta(days=1), spec.session_open,
                                      self.tz).astimezone(timezone.utc)
        self.end = self.close_at(last)
        for trade in history.trades:
            if trade.session.weekday() not in spec.session_weekdays:
                raise ValueError("trade lies on a closed weekday")
            opening = datetime.combine(trade.session - timedelta(days=1),
                                       spec.session_open, self.tz).astimezone(timezone.utc)
            if not opening <= trade.entry_at < trade.exit_at <= self.close_at(trade.session):
                raise ValueError("trade lies outside its declared firm's trading session")

    def close_at(self, session):
        return datetime.combine(session, self.spec.session_close, self.tz).astimezone(timezone.utc)

    def emit(self, at, kind, *, quantity=0, cash=0.0, regime=None, code=None,
             gross_payout=0.0):
        self.events.append(BacktestEvent(
            at, kind, self.attempts, self.role,
            float(self.sim.equity) if self.sim else 0.0,
            float(self.sim.dd_floor) if self.sim else 0.0,
            quantity, cash, regime, code,
            self.ledger.qualifying_days if self.ledger else 0,
            float(self.ledger.cycle_profit) if self.ledger else 0.0,
            gross_payout,
        ))
        if self.wallet is not None:
            self.wallet += Fraction(str(cash))

    def charge(self, at, fee):
        fee = Fraction(str(fee))
        if self.wallet is not None and self.wallet < fee:
            self.emit(at, "wallet_wait")
            self.status = "INSUFFICIENT_WALLET"
            return False
        self.emit(at, "fee", cash=-float(fee))
        self.status = "HORIZON"
        return True

    def schedule(self, at, kind, ledger, attempt, request=None):
        self.serial += 1
        heappush(self.queue, (at, self.serial, kind, ledger, attempt, request))

    def advance(self, until):
        while self.queue and self.queue[0][0] <= until:
            at, _, kind, ledger, attempt, request = heappop(self.queue)
            if kind == "inactivity":
                if (self.sim is None or self.handoff or self.next_role is not None
                        or attempt != self.attempts or request != self.activity_epoch):
                    continue
                if self.ledger is not None and self.ledger.pending is not None:
                    raise ValueError("inactivity during a pending payout needs an explicit firm decision")
                self.fail(at, "FAIL_INACTIVITY")
                continue
            if kind == "approval":
                request = ledger.pending
                ledger.approve(at)
                if ledger is self.ledger:
                    self.sim.equity = ledger.balance
                    self.sim.n_qual_days = ledger.qualifying_days
                    self.sim.cycle_start_equity = self.sim.equity
                    self.sim.payouts_taken += 1
                    # The public scaling page does not define intraday withdrawal
                    # timing. Conservatively apply downward payout changes now;
                    # trading-profit increases still wait for session close.
                    self.limit = min(self.limit, self.spec.funded_limit(
                        self.sim.equity - self.sim.start_equity))
                    self.emit(at, "approval", gross_payout=float(ledger.requests[request - 1].gross))
                    if ledger.breached:
                        self.fail(at, "FAIL_TRAILING_DD")
                    elif ledger.censored:
                        self.emit(at, "live_handoff", code="LIVE_HANDOFF")
                        self.status = "RESTART_PENDING"
                        self.failed_at = None
                        self.restart(at, self.spec.account.eval_fee)
                self.schedule(at + self.config.receipt_delay, "receipt", ledger, attempt, request)
            else:
                ledger.receive(at, request, payment_fee=self.config.payment_fee)
                amount = float(ledger.requests[request - 1].net
                               - Fraction(str(self.config.payment_fee)))
                self.events.append(BacktestEvent(
                    at, "receipt", attempt, "funded",
                    float(ledger.balance), float(ledger.floor), cash=amount,
                    qualifying_days=ledger.qualifying_days,
                    cycle_profit=float(ledger.cycle_profit),
                    gross_payout=float(ledger.requests[request - 1].gross),
                ))
                if self.wallet is not None:
                    self.wallet += Fraction(str(amount))

    def start_phase(self, at):
        role = self.next_role
        if self.sim is None:
            self.role = role
            if (self.failed_at is not None and self.spec.reset_valid_days is not None
                    and at >= self.failed_at + timedelta(days=self.spec.reset_valid_days)):
                self.next_fee = self.spec.account.eval_fee
            upfront = Fraction(str(self.next_fee)) + (
                Fraction(str(self.spec.account.activation_fee)) if role == "funded" else 0)
            if self.wallet is not None and self.wallet < upfront:
                self.emit(at, "wallet_wait")
                self.status = "INSUFFICIENT_WALLET"
                return False
            self.attempts += 1
            if not self.charge(at, self.next_fee):
                return False
        if role == "funded" and not self.charge(at, self.spec.account.activation_fee):
            return False
        self.role = role
        cp = self.phases[role]
        opening = cp.start_equity if cp.start_equity is not None else self.spec.account.size
        self.sim = _ReferenceSim(cp, 1.0, [1.0], opening, False,
                                 external_payouts=True, exact_money=True)
        self.ledger = None
        self.limit = self.spec.eval_contract_limit
        if role == "funded":
            p = self.source[role]
            required = [float(cp.p0[i]) for i in cp.payout_idx
                        if int(cp.kind[i]) == int(RuleKind.MIN_WINNING_DAYS)]
            if len(required) != 1:
                raise ValueError("dated replay requires exactly one winning-day rule")
            self.required_days = int(required[0])
            self.ledger = PayoutLedger(
                p.payout_schema, opening_balance=opening,
                qualifying_days=self.required_days, winning_day_profit=cp.winning_day_threshold,
                initial_floor=self.sim.dd_floor, lock_floor_on_request=self.spec.request_lock_floor,
            )
            self.ledgers.append(self.ledger)
            self.limit = self.spec.funded_limit(0)
        self.emit(at, "phase_start")
        self.next_role = None
        self.activity(at)
        return True

    def activity(self, at):
        self.activity_epoch += 1
        if self.spec.inactivity_days is not None:
            self.schedule(at + timedelta(days=self.spec.inactivity_days),
                          "inactivity", self.ledger, self.attempts, self.activity_epoch)

    def fail(self, at, code, regime=None):
        self.emit(at, "failure", code=code, regime=regime)
        self.failures += 1
        self.failed_at = at
        fee = (self.spec.reset_fee if self.role == "eval" and code != "FAIL_INACTIVITY"
               else self.spec.account.eval_fee)
        self.restart(at, fee)

    def restart(self, at, fee):
        """Release the old account, retaining its scheduled receipts and the wallet."""
        local = at.astimezone(self.tz)
        self.last_ended_session = (local.date() + timedelta(days=1)
                                  if local.time() >= self.spec.session_open else local.date())
        self.next_fee = fee
        self.next_role = "eval" if "eval" in self.phases else "funded"
        self.available_at = at + self.config.retry_delay
        self.sim = self.ledger = None

    def trade(self, trade, day_index):
        self.current_session = trade.session
        self.advance(trade.entry_at)
        if self.handoff or trade.session == self.last_ended_session or trade.entry_at < self.available_at:
            return
        if self.next_role is not None and not self.start_phase(trade.entry_at):
            return
        if self.ledger is not None and self.ledger.pending is not None:
            self.emit(trade.entry_at, "pending_skip")
            return
        sim = self.sim
        remaining = (max(0, self.required_days - self.ledger.qualifying_days)
                     if self.ledger else 0)
        regime = self.policy.select(
            phase=self.role, in_profit=sim.equity > sim.start_equity,
            days_to_payout=remaining, after_payout=sim.payouts_taken > 0,
        )
        if regime.risk_dollars == 0:
            self.emit(trade.entry_at, "policy_skip", regime=regime.name)
            return
        fixed = Fraction(str(self.config.cost_per_trade))
        cost = Fraction(str(self.config.cost_per_contract))
        per_unit = Fraction(str(trade.stop_loss)) + cost
        desired = (Fraction(str(regime.risk_dollars)) - fixed) / per_unit
        # Reuse the projection arithmetic with rational dollars on this slow path.
        # Fast Monte Carlo keeps its compiled float64 implementation.
        project = getattr(project_position, "py_func", project_position)
        quantity, capped, _, _ = project(
            max(Fraction(0), desired), sim.equity - sim.dd_floor,
            Fraction(1), per_unit, Fraction(1), fixed_cost=fixed,
        )
        if capped:
            self.fail(trade.entry_at, "CAPPED_OUT", regime=regime.name)
            return
        # A small policy budget is not evidence that the account is untradeable.
        if desired < 1:
            self.emit(trade.entry_at, "policy_skip", regime=regime.name)
            return
        quantity = min(int(quantity), self.limit)
        sim.policy = np.array([float(quantity)])
        sim.trade_cost = fixed + quantity * cost
        pnl_per_unit = trade.take_profit if trade.won else -trade.stop_loss
        self.advance(trade.exit_at)
        if self.sim is not sim:
            raise ValueError("inactivity expired during an open trade; that execution is unsupported")
        before = sim.equity
        result = sim.run(np.array([pnl_per_unit]), np.array([day_index]),
                         np.array([min(0.0, pnl_per_unit)]), finalize=False)
        if self.ledger:
            self.ledger.record_trade(trade.exit_at, sim.equity - before)
        if abs(sim.equity - before) >= Fraction(str(self.spec.activity_threshold)):
            self.activity(trade.exit_at)
        self.emit(trade.exit_at, "trade", quantity=quantity, regime=regime.name)
        if result.code == ExitCode.PASSED:
            self.emit(trade.exit_at, "evaluation_pass", code="PASSED")
            self.next_role = "funded" if "funded" in self.phases else None
            self.available_at = trade.exit_at + self.config.activation_delay
            if self.next_role is None:
                self.handoff = True
                self.status = "EVALUATION_PASSED"
        elif result.code != ExitCode.ALIVE:
            self.fail(trade.exit_at, ExitCode(result.code).name)

    def close(self, session):
        at = self.close_at(session)
        self.advance(at)
        sim = self.sim
        if sim is None or self.handoff or self.next_role is not None:
            return
        if sim.cur_day != -1:
            code = sim._close_day(sim.equity, winning_allowed=True)
            sim.cur_day = -1  # the adapter has consumed this session-close event
            if code not in (ExitCode.ALIVE, ExitCode.PASSED):
                self.fail(at, ExitCode(code).name)
                return
        if self.ledger:
            ledger = self.ledger
            ledger.close_session(at, session)
            ledger.advance_floor(at, sim.dd_floor)
            self.emit(at, "session_close")
            amount = ledger.maximum_request()
            if amount > 0:
                ledger.request(at, amount, flat=True)
                sim.dd_floor = ledger.floor
                if self.spec.request_lock_floor is not None:
                    sim.dd_locked = True
                self.emit(at, "request", gross_payout=float(amount))
                self.schedule(at + self.config.approval_delay, "approval",
                              ledger, self.attempts)
                self.advance(at)
            if self.sim is not None:
                self.limit = self.spec.funded_limit(self.sim.equity - self.sim.start_equity)
        else:
            self.emit(at, "session_close")

    def run(self):
        for index, (session, trades) in enumerate(groupby(self.history.trades, lambda t: t.session)):
            self.current_session = session
            for trade in trades:
                self.trade(trade, index)
            self.close(session)
        self.advance(self.end)
        outstanding = sum(float(r.net) for ledger in self.ledgers for r in ledger.requests
                          if r.status in ("pending", "approved"))
        return BacktestResult(
            tuple(self.events), self.start, self.end, self.attempts, self.failures,
            self.status, float(self.sim.equity) if self.sim else None, outstanding,
            self.spec.assumptions + (
                "sequential ideal stop/target fills; no gaps or slippage",
                "gross per-contract historical stop/target outcomes are held fixed",
                "maximum eligible payout requested at session close; all requests approved on scenario clock",
                "approval deducts gross immediately; no trading while pending",
                "payout-driven scaling decreases apply at approval (conservative scenario)",
                "retries start no earlier than the next observed session",
                "live handoff starts a fresh paid attempt under the same retry delay and wallet constraints; not a failure",
                "live-account value and receipts after the observation horizon are excluded",
            ),
            self.spec, self.policy, self.config,
            sha256(repr(self.history.trades).encode("utf-8")).hexdigest(),
        )


def backtest(spec: LifecycleSpec, history: BracketHistory, policy: DollarPolicy,
             config: BacktestConfig) -> BacktestResult:
    """Replay one chronological history with repeated attempts and dated cashflows."""
    if not isinstance(spec, LifecycleSpec) or not isinstance(history, BracketHistory):
        raise TypeError("backtest requires LifecycleSpec and BracketHistory")
    if not isinstance(policy, DollarPolicy) or not isinstance(config, BacktestConfig):
        raise TypeError("backtest requires DollarPolicy and BacktestConfig")
    return _Replay(spec, history, policy, config).run()
