"""Chronological lifecycle adapter over the existing reference rule interpreter.

No resampling, strategy construction or stop/target retuning happens here.
The fast Monte Carlo API remains available separately via Engine.run.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from heapq import heappop, heappush
from itertools import groupby
from fractions import Fraction
from functools import cached_property, lru_cache
from hashlib import sha256
from math import isfinite
from zoneinfo import ZoneInfo

from .compiler import compile_account
from .cache import freeze_compiled
from .enums import ExitCode, Severity, StateField, Timing
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
    phase_name: str | None = None


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
    policy: DollarPolicy | None
    config: BacktestConfig
    history_fingerprint: str

    @cached_property
    def uncertainty(self):
        from .uncertainty import uncertainty_report
        return uncertainty_report({"net_cash": [self.net_cash],
            "net_cash_per_day": [self.net_cash_per_day]}, sample_kind="single_history")

    @cached_property
    def net_cash(self):
        # The frozen ledger never changes; most rule events carry no cash.
        return float(sum((Fraction(str(e.cash)) for e in self.events if e.cash), Fraction(0)))

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


@lru_cache(maxsize=128)
def _check_support(spec, *, observations=False):
    if not observations and not spec.flatten_at_close:
        raise ValueError("overnight positions require observation replay")
    validate(spec.account, sequence=observations)
    for phase in spec.account.phases:
        if phase.role == "funded":
            if phase.payout_schema is not None and phase.payout_schema.recompute_floor_on_payout:
                raise ValueError("recompute_floor_on_payout is unsupported by the dated ledger")
            if sum(r.compile().kind == RuleKind.MIN_WINNING_DAYS for r in phase.rules) != 1:
                raise ValueError("dated replay requires exactly one winning-day rule")
    compiled = compile_account(spec.account)
    allowed = {RuleKind.TRAILING_DD, RuleKind.PROFIT_TARGET,
               RuleKind.CONSISTENCY_GATE, RuleKind.MIN_WINNING_DAYS, RuleKind.MIN_DAYS}
    if observations:
        allowed |= {RuleKind.STATIC_DD, RuleKind.DAILY_LOSS}
    for p in compiled.phases:
        if not observations and (not isfinite(p.dd_amount) or p.dd_update_timing != Timing.EOD):
            raise ValueError("bracket replay currently requires an EOD trailing floor in each phase")
        if observations and sum(int(k) in (RuleKind.STATIC_DD, RuleKind.TRAILING_DD) for k in p.kind) != 1:
            raise ValueError("observation replay requires one static or trailing drawdown rule per phase")
        for i in range(p.n_rules):
            kind = RuleKind(int(p.kind[i]))
            if kind not in allowed:
                raise ValueError(f"bracket replay does not support {kind.name}")
            if not observations and kind == RuleKind.TRAILING_DD and (
                p.severity[i] != Severity.HARD or p.check_timing[i] != Timing.CONTINUOUS
            ):
                raise ValueError("bracket replay requires hard continuous drawdown checks")
            if observations and kind in (RuleKind.TRAILING_DD, RuleKind.STATIC_DD) and p.severity[i] != Severity.HARD:
                raise ValueError("drawdown must terminate the account; only daily loss supports suspension")
            if not observations and kind == RuleKind.CONSISTENCY_GATE and p.role != "eval":
                raise ValueError("funded consistency is not supported by the dated payout ledger")
        if p.role == "funded" and p.payout is None:
            raise ValueError("funded replay requires an explicit payout schema")
    return freeze_compiled(compiled)


class _Replay:
    """Coordinates existing rules, feasibility projection and dated payout accounting."""

    def __init__(self, spec, history, policy, config, *, bracket_factory=None,
                 session_closes=None, sessions=None):
        self.spec, self.history, self.policy, self.config = spec, history, policy, config
        self.fixed_cost = Fraction(str(config.cost_per_trade))
        self.contract_cost = Fraction(str(config.cost_per_contract))
        self.risk_budgets = ({r.name: Fraction(str(r.risk_dollars)) for r in policy.regimes}
                             if policy is not None else {})
        self.activity_threshold = Fraction(str(spec.activity_threshold))
        # Internal research hook; the public historical API never retargets trades.
        self.bracket_factory = bracket_factory
        self.session_closes = dict(session_closes or {})
        compiled = self.check_support(spec)
        self.phase_list = compiled.phases
        self.named_phases = sum(p.role == "eval" for p in self.phase_list) > 1
        self.phase_index = self.next_index = 0
        self.ledger_names = {}
        self.phases = {p.role: p for p in compiled.phases}
        self.source = {p.role: p for p in spec.account.phases}
        self.tz = ZoneInfo(spec.session_timezone)
        self.events, self.queue, self.ledgers = [], [], []
        self.serial = self.attempts = self.failures = 0
        self.sim = self.ledger = None
        self.role = "eval" if "eval" in self.phases else "funded"
        self.next_role = self.role
        self.wallet = (None if config.initial_wallet is None
                       else Fraction(str(config.initial_wallet)))
        self.status = "HORIZON"
        self.handoff = False
        self.next_fee = spec.account.eval_fee
        self.last_ended_session = None
        self.failed_at = None
        self.activity_epoch = 0
        # The horizon consists of complete declared sessions, not N sessions / 5.
        sessions = history.sessions if history is not None else sessions
        first, last = sessions[0], sessions[-1]
        self.start = datetime.combine(first - timedelta(days=1), spec.session_open,
                                      self.tz).astimezone(timezone.utc)
        self.available_at = history.trades[0].entry_at if history is not None else self.start
        self.end = self.close_at(last)
        for trade in history.trades if history is not None else ():
            if trade.session.weekday() not in spec.session_weekdays:
                raise ValueError("trade lies on a closed weekday")
            opening = datetime.combine(trade.session - timedelta(days=1),
                                       spec.session_open, self.tz).astimezone(timezone.utc)
            if not opening <= trade.entry_at < trade.exit_at <= self.close_at(trade.session):
                raise ValueError("trade lies outside its declared firm's trading session")

    def close_at(self, session):
        normal = datetime.combine(session, self.spec.session_close, self.tz).astimezone(timezone.utc)
        actual = self.session_closes.get(session, normal)
        if actual.tzinfo is None or actual > normal:
            raise ValueError("session close must be aware and no later than the firm cutoff")
        return actual

    def check_support(self, spec):
        return _check_support(spec)

    def winning_allowed(self, session):
        return True

    def needs_day_close(self):
        return self.sim.cur_day != -1

    def closing_equity(self):
        return self.sim.equity

    def payout_allowed(self):
        return True

    def ledger_options(self):
        return {}

    def payout_amount(self, at, maximum):
        return maximum

    def payout_decision(self, at, request):
        return "approve"

    def processing_at(self, at, delay):
        return at + delay

    def phase_limit(self, index):
        return (self.spec.eval_contract_limit if self.phase_list[index].role == "eval"
                else self.spec.funded_limit(0))

    def funded_limit(self, profit):
        return self.spec.funded_limit(profit)

    def transition_delay(self, index):
        return self.config.activation_delay

    def phase_name(self):
        return self.spec.account.phases[self.phase_index].name

    def append_event(self, event):
        self.events.append(event)

    def emit(self, at, kind, *, quantity=0, cash=0.0, regime=None, code=None,
             gross_payout=0.0):
        self.append_event(BacktestEvent(
            at, kind, self.attempts, self.role,
            float(self.sim.equity) if self.sim else 0.0,
            float(self.sim.dd_floor) if self.sim else 0.0,
            quantity, cash, regime, code,
            self.ledger.qualifying_days if self.ledger else 0,
            float(self.ledger.cycle_profit) if self.ledger else 0.0,
            gross_payout,
            self.phase_name() if self.named_phases else None,
        ))
        if self.wallet is not None and cash:
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

    def advance(self, until, *, qualifying_close=False):
        deferred = []
        while self.queue and self.queue[0][0] <= until:
            event = heappop(self.queue)
            at, _, kind, ledger, attempt, request = event
            if kind == "inactivity":
                if (self.sim is None or self.handoff or self.next_role is not None
                        or attempt != self.attempts or request != self.activity_epoch):
                    continue
                if self.ledger is not None and self.ledger.pending is not None:
                    raise ValueError("inactivity during a pending payout needs an explicit firm decision")
                if qualifying_close and at == until:
                    # An eligible close wins an exact timestamp tie, not a
                    # deadline crossed strictly before that close.
                    deferred.append(event)
                    continue
                self.fail(at, "FAIL_INACTIVITY")
                continue
            if kind == "approval":
                request = ledger.pending
                decision = self.payout_decision(at, ledger.get_request(request))
                if decision not in ("approve", "deny"):
                    raise ValueError("payout decision must be approve or deny")
                if decision == "deny":
                    ledger.reject(at)
                    self.emit(at, "rejection", gross_payout=float(ledger.get_request(request).gross))
                    continue
                ledger.approve(at)
                if ledger is self.ledger:
                    self.sim.equity = ledger.balance
                    self.sim.n_qual_days = ledger.qualifying_days
                    self.sim.cycle_start_equity = self.sim.equity
                    self.sim.payouts_taken += 1
                    if StateField.MAX_DAY_PNL in self.source[self.role].payout_schema.reset_fields:
                        self.sim.max_day_pnl = self.sim._cash(0)
                    # The public scaling page does not define intraday withdrawal
                    # timing. Conservatively apply downward payout changes now;
                    # trading-profit increases still wait for session close.
                    self.limit = min(self.limit, self.funded_limit(
                        self.sim.equity - self.sim.start_equity))
                    self.emit(at, "approval", gross_payout=float(ledger.get_request(request).gross))
                    if ledger.breached or (ledger.floor is not None and self.sim.equity <= ledger.floor):
                        self.fail(at, "FAIL_TRAILING_DD")
                    elif ledger.censored:
                        self.emit(at, "live_handoff", code="LIVE_HANDOFF")
                        self.status = "RESTART_PENDING"
                        self.failed_at = None
                        self.restart(at, self.spec.account.eval_fee)
                self.schedule(self.processing_at(at, self.config.receipt_delay), "receipt", ledger, attempt, request)
            else:
                ledger.receive(at, request, payment_fee=self.config.payment_fee)
                amount = float(ledger.get_request(request).net
                               - Fraction(str(self.config.payment_fee)))
                self.append_event(BacktestEvent(
                    at, "receipt", attempt, "funded",
                    float(ledger.balance), float(ledger.floor), cash=amount,
                    qualifying_days=ledger.qualifying_days,
                    cycle_profit=float(ledger.cycle_profit),
                    gross_payout=float(ledger.get_request(request).gross),
                    phase_name=self.ledger_names[ledger] if self.named_phases else None,
                ))
                if self.wallet is not None:
                    self.wallet += Fraction(str(amount))
        for event in deferred:
            heappush(self.queue, event)

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
        self.phase_index = self.next_index
        cp = self.phase_list[self.phase_index]
        opening = cp.start_equity if cp.start_equity is not None else self.spec.account.size
        self.sim = _ReferenceSim(cp, 1.0, [1.0], opening, False,
                                 external_payouts=True, exact_money=True)
        self.ledger = None
        self.limit = self.phase_limit(self.phase_index)
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
                **self.ledger_options(),
            )
            self.ledgers.append(self.ledger)
            self.ledger_names[self.ledger] = self.phase_name()
            self.limit = self.funded_limit(0)
        self.emit(at, "phase_start")
        self.next_role = None
        self.activity(at)
        return True

    def activity(self, at):
        self.activity_epoch += 1
        if self.spec.inactivity_days is not None:
            deadline = at + timedelta(days=self.spec.inactivity_days)
            if self.spec.inactivity_close is not None:
                expiry_date = at.astimezone(self.tz).date() + timedelta(days=self.spec.inactivity_days)
                deadline = datetime.combine(expiry_date, self.spec.inactivity_close,
                                            self.tz).astimezone(timezone.utc)
            self.schedule(deadline,
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
        self.next_index = 0
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
        if self.bracket_factory is not None:
            trade = self.bracket_factory(trade, regime, self, day_index)
        fixed, cost = self.fixed_cost, self.contract_cost
        stop = trade.stop_loss if isinstance(trade.stop_loss, Fraction) else Fraction(str(trade.stop_loss))
        per_unit = stop + cost
        desired = (self.risk_budgets[regime.name] - fixed) / per_unit
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
        pnl_per_unit = trade.take_profit if trade.won else -trade.stop_loss
        self.settle_trade(trade, day_index, regime, quantity, pnl_per_unit,
                          min(0.0, pnl_per_unit), trade.exit_at)

    def settle_trade(self, trade, day_index, regime, quantity, pnl_per_unit, low, exit_at):
        """One settlement path shared by fixed brackets and historical fills."""
        sim = self.sim
        sim.trade_cost = self.fixed_cost + quantity * self.contract_cost
        net = Fraction(str(pnl_per_unit)) * quantity - sim.trade_cost
        self.advance(exit_at, qualifying_close=abs(net) >= self.activity_threshold)
        if self.sim is not sim:
            raise ValueError("inactivity expired during an open trade; that execution is unsupported")
        before = sim.equity
        balance = before + net
        observed = min(before + quantity * Fraction(str(low)), balance)
        code = sim.observe(balance, observed, day_index, traded=True, allow_pass=True)
        if self.ledger:
            self.ledger.record_trade(exit_at, sim.equity - before)
        if abs(sim.equity - before) >= self.activity_threshold:
            self.activity(exit_at)
        self.emit(exit_at, "trade", quantity=quantity, regime=regime.name)
        self.handle_result(exit_at, code, regime.name)
        self.advance(exit_at)

    def handle_result(self, at, code, regime=None):
        if code == ExitCode.PASSED:
            self.emit(at, "evaluation_pass", code="PASSED")
            self.next_index = self.phase_index + 1
            self.next_role = self.phase_list[self.next_index].role if self.next_index < len(self.phase_list) else None
            self.available_at = at + (self.transition_delay(self.next_index) if self.next_role else timedelta(0))
            if self.next_role is None:
                self.handoff = True
                self.status = "EVALUATION_PASSED"
        elif code != ExitCode.ALIVE:
            self.fail(at, ExitCode(code).name, regime=regime)

    def close(self, session):
        at = self.close_at(session)
        self.advance(at)
        sim = self.sim
        if sim is None or self.handoff or self.next_role is not None:
            return
        winning = self.winning_allowed(session)
        if self.needs_day_close():
            code = sim._close_day(sim.equity, winning_allowed=winning, test_equity=self.closing_equity())
            sim.cur_day = -1  # the adapter has consumed this session-close event
            if code not in (ExitCode.ALIVE, ExitCode.PASSED):
                self.fail(at, ExitCode(code).name)
                return
        if self.ledger:
            ledger = self.ledger
            ledger.close_session(at, session, winning_allowed=winning)
            ledger.advance_floor(at, sim.dd_floor)
            self.emit(at, "session_close")
            maximum = ledger.maximum_request() if self.payout_allowed() else 0
            amount = self.payout_amount(at, maximum) if maximum else 0
            if amount > 0:
                ledger.request(at, amount, flat=True)
                sim.dd_floor = ledger.floor
                if self.spec.request_lock_floor is not None:
                    sim.dd_locked = True
                self.emit(at, "request", gross_payout=float(amount))
                self.schedule(self.processing_at(at, self.config.approval_delay), "approval",
                              ledger, self.attempts)
                self.advance(at)
            if self.sim is not None:
                self.limit = self.funded_limit(self.sim.equity - self.sim.start_equity)
        else:
            self.emit(at, "session_close")

    def run(self):
        for index, (session, trades) in enumerate(groupby(self.history.trades, lambda t: t.session)):
            self.current_session = session
            for trade in trades:
                self.trade(trade, index)
            self.close(session)
        self.advance(self.end)
        return self.result()

    def result(self, *, fingerprint=None, assumptions=None):
        outstanding = sum(float(r.net) for ledger in self.ledgers for r in ledger.requests
                          if r.status in ("pending", "approved"))
        return BacktestResult(
            tuple(self.events), self.start, self.end, self.attempts, self.failures,
            self.status, float(self.sim.equity) if self.sim else None, outstanding,
            self.spec.assumptions + (assumptions if assumptions is not None else (
                "qualifying trade close precedes inactivity at the identical timestamp; strictly earlier expiry remains unsupported",
                "sequential ideal stop/target fills; no gaps or slippage",
                "gross per-contract historical stop/target outcomes are held fixed",
                "maximum eligible payout requested at session close; all requests approved on scenario clock",
                "approval deducts gross immediately; no trading while pending",
                "payout-driven scaling decreases apply at approval (conservative scenario)",
                "retries start no earlier than the next observed session",
                "live handoff starts a fresh paid attempt under the same retry delay and wallet constraints; not a failure",
                "live-account value and receipts after the observation horizon are excluded",
            )),
            self.spec, self.policy, self.config,
            fingerprint if fingerprint is not None else sha256(repr(self.history.trades).encode("utf-8")).hexdigest(),
        )


def backtest(spec: LifecycleSpec, history: BracketHistory, policy: DollarPolicy,
             config: BacktestConfig) -> BacktestResult:
    """Replay one chronological history with repeated attempts and dated cashflows."""
    if not isinstance(spec, LifecycleSpec) or not isinstance(history, BracketHistory):
        raise TypeError("backtest requires LifecycleSpec and BracketHistory")
    if not isinstance(policy, DollarPolicy) or not isinstance(config, BacktestConfig):
        raise TypeError("backtest requires DollarPolicy and BacktestConfig")
    return _Replay(spec, history, policy, config).run()
