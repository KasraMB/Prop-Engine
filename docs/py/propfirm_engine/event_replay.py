"""Recorded portfolio executions on the shared account lifecycle."""
from dataclasses import dataclass
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
from hashlib import sha256

from .backtest import BacktestResult, _Replay, _check_support
from .calendars import ProcessingCalendar
from .enums import ExitCode
from .events import Fill, Marks, money, ordered
from .execution import BacktestConfig, LifecycleSpec
from .portfolio import Book, BookState
from .rules import RuleKind


@dataclass(frozen=True, slots=True)
class EventState:
    at: datetime
    kind: str
    attempt: int
    phase: str
    balance: Fraction
    equity: Fraction
    floor: Fraction


@dataclass(frozen=True)
class EventReplay:
    replay: BacktestResult
    book: BookState | None
    trace: tuple[EventState, ...]
    fills: int
    skipped_fills: int
    marks: int
    event_counts: tuple[tuple[str, int], ...] = ()
    recording: str = "research"


@dataclass(frozen=True, slots=True)
class PayoutContext:
    at: datetime
    balance: Fraction
    floor: Fraction
    maximum: Fraction
    cycle_profit: Fraction
    qualifying_days: int
    payouts: int


class _EventReplay(_Replay):
    def __init__(self, spec, events, instruments, config, *, sessions, fidelity,
                 mark_fills, max_mark_age, liquidation_fee, trace=False, units=None,
                 session_closes=None, withdrawal=None, decision=None, processing=None,
                 phase_limits=None, transition_delays=None, retry_on_failure=True,
                 restart_on_handoff=True, drawdown_basis="rule", daily_loss_basis="balance",
                 recording=None, sink=None):
        if not isinstance(spec, LifecycleSpec) or not isinstance(config, BacktestConfig):
            raise TypeError("event replay requires LifecycleSpec and BacktestConfig")
        if fidelity != "observed_marks":
            raise ValueError("explicit fidelity='observed_marks' is required; unseen paths are not inferred")
        if config.cost_per_contract or config.cost_per_trade:
            raise ValueError("event replay takes trading fees from fills, not BacktestConfig")
        if not isinstance(max_mark_age, timedelta) or max_mark_age < timedelta(0):
            raise ValueError("max_mark_age must be a nonnegative timedelta")
        if type(trace) is not bool:
            raise ValueError("trace must be bool")
        recording = ("trace" if trace else "research") if recording is None else recording
        if recording not in ("search", "research", "trace") or trace and recording != "trace":
            raise ValueError("recording must be search, research or trace; trace=True requires trace mode")
        if sink is not None and not callable(sink):
            raise TypeError("sink must be callable")
        self.recording, self.sink = recording, sink
        self.event_counts = Counter()
        if any(callback is not None and not callable(callback) for callback in (withdrawal, decision)):
            raise TypeError("withdrawal and decision must be callbacks or None")
        if processing is not None and not isinstance(processing, ProcessingCalendar):
            raise TypeError("processing must be a ProcessingCalendar or None")
        self.withdrawal, self.decision, self.processing = withdrawal, decision, processing
        if type(retry_on_failure) is not bool or type(restart_on_handoff) is not bool:
            raise ValueError("restart policies must be bool")
        self.retry_on_failure, self.restart_on_handoff = retry_on_failure, restart_on_handoff
        if drawdown_basis not in ("rule", "balance", "equity") or daily_loss_basis not in ("balance", "equity"):
            raise ValueError("invalid drawdown or daily loss basis")
        self.drawdown_basis, self.daily_loss_basis = drawdown_basis, daily_loss_basis
        self.phase_limits, self.transition_delays = dict(phase_limits or {}), dict(transition_delays or {})
        names = {p.name for p in spec.account.phases}
        if (set(self.phase_limits) | set(self.transition_delays)) - names:
            raise ValueError("phase configuration names must belong to the account")
        if any(type(n) is not int or n < 1 for n in self.phase_limits.values()):
            raise ValueError("phase limits must be positive integers")
        if any(not isinstance(d, timedelta) or d < timedelta(0) for d in self.transition_delays.values()):
            raise ValueError("transition delays must be nonnegative timedeltas")
        self.liquidation_fee = money(liquidation_fee)
        if self.liquidation_fee < 0:
            raise ValueError("liquidation_fee must be nonnegative")
        self.sessions = tuple(sessions)
        if (not self.sessions or any(type(s) is not date for s in self.sessions)
                or any(b <= a for a, b in zip(self.sessions, self.sessions[1:]))
                or any(s.weekday() not in spec.session_weekdays for s in self.sessions)):
            raise ValueError("sessions must be increasing open-weekday date identifiers")
        self.instruments = tuple(instruments)
        self.source_book = Book(self.instruments, mark_fills=mark_fills)
        self.units = {i.symbol: Fraction(1) for i in self.instruments}
        if units is not None:
            if set(units) != set(self.units):
                raise ValueError("units must define every instrument")
            self.units = {k: money(v) for k, v in units.items()}
            if any(v <= 0 for v in self.units.values()):
                raise ValueError("contract units must be positive")
        self.mark_fills, self.max_mark_age = mark_fills, max_mark_age
        self.keep_trace, self.trace = recording == "trace", []
        self.book = None
        self.following = False
        self.suspended_session = None
        self.current_day = 0
        self.fill_count = self.skip_count = self.mark_count = 0
        self.stream = iter(ordered(events))
        self.digest = sha256()
        super().__init__(spec, None, None, config, sessions=self.sessions,
                         session_closes=session_closes)
        if set(self.session_closes) - set(self.sessions):
            raise ValueError("session close overrides must belong to the declared sessions")
        for session in self.sessions:
            if self.close_at(session) <= self.open_at(session):
                raise ValueError("session close must follow its opening")
        self.digest.update(repr((self.instruments, self.sessions, self.session_closes,
                                 mark_fills, max_mark_age, self.liquidation_fee,
                                 self.units)).encode())

    def open_at(self, session):
        return datetime.combine(session - timedelta(days=1), self.spec.session_open,
                                self.tz).astimezone(timezone.utc)

    def check_support(self, spec):
        return _check_support(spec, observations=True)

    def ledger_options(self):
        return {"floor_checks": "executor", "record_events": self.recording != "search"}

    def append_event(self, event):
        self.event_counts[event.kind] += 1
        if self.sink is not None:
            self.sink(event)
        if self.recording != "search" or event.cash or event.kind == "wallet_wait":
            super().append_event(event)

    def phase_limit(self, index):
        baseline = super().phase_limit(index)
        limit = self.phase_limits.get(self.spec.account.phases[index].name, baseline)
        return min(limit, baseline) if self.phase_list[index].role == "funded" else limit

    def abandon(self, at, *, retry, reason):
        if self.book is None or self.next_role is not None or (self.ledger and self.ledger.pending is not None):
            raise ValueError("abandonment requires an active account without a pending payout")
        self.book.check_marks(at, self.max_mark_age)
        for fill in tuple(self.book.liquidation(at, self.liquidation_fee)):
            if self.book is None:
                return
            self.settle(fill, self.current_day, kind="abandon_close")
        self.emit(at, "abandonment", code=reason)
        self.book = None
        self.following = False
        self.failed_at = None
        _Replay.restart(self, at, self.spec.account.eval_fee)
        self.handoff = not retry
        self.status = "RESTART_PENDING" if retry else "ABANDONED"

    def funded_limit(self, profit):
        return min(super().funded_limit(profit), self.phase_limits.get(self.phase_name(), float("inf")))

    def transition_delay(self, index):
        return self.transition_delays.get(self.spec.account.phases[index].name, self.config.activation_delay)

    def payout_amount(self, at, maximum):
        if self.withdrawal is None:
            return maximum
        view = PayoutContext(at, self.ledger.balance, self.ledger.floor, maximum,
                             self.ledger.cycle_profit, self.ledger.qualifying_days,
                             self.sim.payouts_taken)
        amount = money(self.withdrawal(view))
        if not 0 <= amount <= maximum:
            raise ValueError("withdrawal must be between zero and the maximum eligible amount")
        return amount

    def payout_decision(self, at, request):
        return "approve" if self.decision is None else self.decision(at, request)

    def processing_at(self, at, delay):
        return at + delay if self.processing is None else self.processing.after(at, delay)

    def winning_allowed(self, session):
        return session != self.suspended_session and self.sim.cur_day != -1

    def needs_day_close(self):
        return super().needs_day_close() or (self.book is not None and not self.book.flat)

    def closing_equity(self):
        return self.book.equity

    def payout_allowed(self):
        return self.book.flat and all(self.sim._consistency_gate_ok(i) for i in self.sim.cp.payout_idx
                   if self.sim.cp.kind[i] == RuleKind.CONSISTENCY_GATE)

    def handle_result(self, at, code, regime=None):
        if code != ExitCode.ALIVE and self.sim is not None and getattr(self.sim, "observation_soft", False):
            if self.suspended_session != self.current_session:
                self.suspend(at)
            return
        super().handle_result(at, code, regime)

    def suspend(self, at):
        self.suspended_session = self.current_session
        self.following = False
        self.book.check_marks(at, self.max_mark_age)
        for fill in self.book.liquidation(at, self.liquidation_fee):
            before = self.book.balance
            self.book.apply(fill)
            if self.ledger is not None:
                self.ledger.record_trade(at, self.book.balance - before)
            code = self.sim.observe(self.book.balance, self.book.equity, self.current_day, traded=True)
            self.emit(at, "liquidation", quantity=fill.quantity)
            if code != ExitCode.ALIVE and not self.sim.observation_soft:
                self.fail(at, ExitCode(code).name)
                return
        self.emit(at, "daily_suspend", code="FAIL_DAILY_LOSS")
        self.record(at, "daily_suspend")

    def record(self, at, kind):
        if (self.keep_trace or self.sink is not None) and self.book is not None:
            event = EventState(at, kind, self.attempts, self.role,
                               self.book.balance, self.book.equity, self.sim.dd_floor)
            if self.sink is not None:
                self.sink(event)
            if self.keep_trace:
                self.trace.append(event)

    def emit(self, at, kind, **kwargs):
        if kind == "approval" and self.book is not None:
            delta = self.sim.equity - self.book.balance
            self.book.adjust(at, delta)
            for name in ("day_base", "close_equity"):
                if hasattr(self.sim, name):
                    setattr(self.sim, name, getattr(self.sim, name) + delta)
            self.record(at, kind)
        super().emit(at, kind, **kwargs)

    def start_phase(self, at):
        if not super().start_phase(at):
            return False
        self.book = Book(self.instruments, balance=self.sim.equity, mark_fills=self.mark_fills)
        self.sim.drawdown_basis = self.drawdown_basis
        self.sim.daily_loss_basis = self.daily_loss_basis
        self.book.copy_marks(self.source_book)
        return True

    def restart(self, at, fee):
        stop = not (self.retry_on_failure if self.failed_at is not None else self.restart_on_handoff)
        self.book = None
        self.following = False
        super().restart(at, fee)
        if stop:
            self.handoff = True
            self.status = "ACCOUNT_FAILED" if self.failed_at is not None else "LIVE_HANDOFF"

    def fail(self, at, code, regime=None):
        if self.book is not None:
            self.book.check_marks(at, self.max_mark_age)
            before = self.book.balance
            for fill in self.book.liquidation(at, self.liquidation_fee):
                self.book.apply(fill)
                self.sim.equity = self.book.balance
                self.emit(at, "liquidation", quantity=fill.quantity)
            delta = self.book.balance - before
            self.sim.equity = self.book.balance
            self.sim.total_pnl += delta
            self.sim.day_pnl += delta
            if self.ledger is not None:
                self.ledger.terminate(at, delta)
            self.record(at, "failure")
        super().fail(at, code, regime)

    def eligible(self, at):
        return (not self.handoff and self.current_session != self.last_ended_session
                and self.current_session != self.suspended_session
                and at >= self.available_at
                and (self.ledger is None or self.ledger.pending is None))

    def settle(self, fill, day, *, kind="fill"):
        before = self.book.balance
        effect = self.book.apply(fill)
        self.book.check_marks(fill.at, self.max_mark_age)
        code = self.sim.observe(self.book.balance, self.book.equity, day,
                                traded=True, allow_pass=self.book.flat)
        if self.ledger is not None:
            self.ledger.record_trade(fill.at, self.book.balance - before)
        if effect.closed and abs(effect.closed_net) >= self.activity_threshold:
            self.activity(fill.at)
        self.emit(fill.at, kind, quantity=fill.quantity)
        self.record(fill.at, kind)
        self.handle_result(fill.at, code)
        if code != ExitCode.ALIVE:
            self.following = False

    def consume(self, event, day):
        self.current_day = day
        self.digest.update(repr(event).encode())
        qualifies = False
        if isinstance(event, Fill) and self.following and self.book is not None:
            effect = self.book.preview(event)
            qualifies = bool(effect.closed and abs(effect.closed_net) >= self.activity_threshold)
        self.advance(event.at, qualifying_close=qualifies)
        if isinstance(event, Marks):
            self.mark_count += 1
            self.source_book.apply(event)
            if self.book is not None and self.following:
                self.book.apply(event)
                self.book.check_marks(event.at, self.max_mark_age)
                code = self.sim.observe(self.book.balance, self.book.equity, day)
                self.record(event.at, "mark")
                self.handle_result(event.at, code)
            return
        if self.source_book.flat and self.eligible(event.at):
            if self.next_role is None or self.start_phase(event.at):
                if not self.following:
                    self.book.copy_marks(self.source_book)
                self.following = True
        if self.following and self.eligible(event.at):
            held = self.book.quantity(event.symbol)
            exposure = sum(abs(self.book.quantity(s)) * u for s, u in self.units.items())
            exposure += (abs(held + event.quantity) - abs(held)) * self.units[event.symbol]
            if exposure > self.limit:
                raise ValueError("recorded fill exceeds the account contract limit; resizing needs a strategy adapter")
            self.source_book.apply(event)
            self.fill_count += 1
            self.settle(event, day)
        else:
            self.source_book.apply(event)
            self.following = False
            self.skip_count += 1
            self.emit(event.at, "fill_skip", quantity=event.quantity)
        self.advance(event.at)

    def finish_session(self, session, day):
        self.current_day = day
        at = self.close_at(session)
        self.advance(at)
        if self.book is not None and not self.book.flat:
            self.book.check_marks(at, self.max_mark_age)
            closing = tuple(self.book.liquidation(at, self.liquidation_fee)) if self.spec.flatten_at_close else ()
            for fill in closing:
                if self.book is None:
                    break
                self.settle(fill, day, kind="liquidation")
            if self.spec.flatten_at_close:
                self.following = False
        super().close(session)
        if self.book is not None:
            self.sim.close_equity = self.book.equity
        self.record(at, "session_close")

    def run(self):
        event = next(self.stream, None)
        for day, session in enumerate(self.sessions):
            self.current_session = session
            opening, closing = self.open_at(session), self.close_at(session)
            while event is not None and event.at <= closing:
                if event.at < opening:
                    raise ValueError("event lies outside a declared trading session")
                self.consume(event, day)
                event = next(self.stream, None)
            self.finish_session(session, day)
        if event is not None:
            raise ValueError("event lies after the declared observation horizon")
        self.advance(self.end)
        return self.finish_result()

    def finish_result(self, assumptions=None):
        result = self.result(fingerprint=self.digest.hexdigest(), assumptions=(assumptions or (
            "recorded quantities and arbitrary exits; no sizing or target optimization",
            "observed marks only; no claim about unobserved intrabar equity",
            f"fill prices update marks: {self.mark_fills}; maximum mark age: {self.max_mark_age}",
            f"forced closes use last fresh marks plus {self.liquidation_fee} fee per contract",
            "a breached account cannot recover on later fills",
            "skipped source portfolios are quarantined until flat; dependent signals are not regenerated",
            "account-wide absolute exposure uses explicit contract units; excess recorded exposure is rejected",
            "qualifying FIFO closes use realized profit net of allocated entry and exit fees",
            "qualifying close precedes inactivity at the identical timestamp",
            "payout requests at session close use the configured withdrawal and decision policies",
            "approval deducts gross; no trading while pending; downward scaling applies at approval",
            "retry and live handoff start fresh paid attempts no earlier than the next observed session",
            "live value and receipts after the horizon are excluded",
        )) + (
            f"flatten at session cutoff: {self.spec.flatten_at_close}; open horizon positions remain marked, not counted as cash",
            f"drawdown peak basis: {self.drawdown_basis}; daily loss reset basis: {self.daily_loss_basis}",
            f"retry after failure: {self.retry_on_failure}; restart after handoff: {self.restart_on_handoff}",
            "maximum eligible withdrawal" if self.withdrawal is None else "custom causal withdrawal callback",
            "all payout requests approved" if self.decision is None else "custom payout approval/denial scenario",
            "elapsed processing delays" if self.processing is None else f"elapsed delays rolled through {self.processing!r}",
        ))
        return EventReplay(result, self.book.snapshot() if self.book is not None else None,
                           tuple(self.trace), self.fill_count, self.skip_count, self.mark_count,
                           tuple(sorted(self.event_counts.items())), self.recording)


def replay_events(spec, events, instruments, config, **kwargs):
    return _EventReplay(spec, events, instruments, config, **kwargs).run()
