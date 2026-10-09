"""Recorded portfolio executions on the shared account lifecycle."""
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
from hashlib import sha256

from .backtest import BacktestResult, _Replay
from .enums import ExitCode
from .events import Fill, Marks, money, ordered
from .execution import BacktestConfig, LifecycleSpec
from .portfolio import Book, BookState


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


class _EventReplay(_Replay):
    def __init__(self, spec, events, instruments, config, *, sessions, fidelity,
                 mark_fills, max_mark_age, liquidation_fee, trace=False, units=None,
                 session_closes=None):
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
        self.keep_trace, self.trace = trace, []
        self.book = None
        self.following = False
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

    def record(self, at, kind):
        if self.keep_trace and self.book is not None:
            self.trace.append(EventState(at, kind, self.attempts, self.role,
                                        self.book.balance, self.book.equity, self.sim.dd_floor))

    def emit(self, at, kind, **kwargs):
        if kind == "approval" and self.book is not None:
            self.book.adjust(at, self.sim.equity - self.book.balance)
            self.record(at, kind)
        super().emit(at, kind, **kwargs)

    def start_phase(self, at):
        if not super().start_phase(at):
            return False
        self.book = Book(self.instruments, balance=self.sim.equity, mark_fills=self.mark_fills)
        self.book.copy_marks(self.source_book)
        return True

    def restart(self, at, fee):
        self.book = None
        self.following = False
        super().restart(at, fee)

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
        at = self.close_at(session)
        self.advance(at)
        if self.book is not None and not self.book.flat:
            self.book.check_marks(at, self.max_mark_age)
            closing = tuple(self.book.liquidation(at, self.liquidation_fee))
            for fill in closing:
                if self.book is None:
                    break
                self.settle(fill, day, kind="liquidation")
            self.following = False
        super().close(session)
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
        result = self.result(fingerprint=self.digest.hexdigest(), assumptions=(
            "recorded quantities and arbitrary exits; no sizing or target optimization",
            "observed marks only; no claim about unobserved intrabar equity",
            f"fill prices update marks: {self.mark_fills}; maximum mark age: {self.max_mark_age}",
            f"forced closes use last fresh marks plus {self.liquidation_fee} fee per contract",
            "session cutoff forces flat; a breached account cannot recover on later fills",
            "skipped source portfolios are quarantined until flat; dependent signals are not regenerated",
            "account-wide absolute exposure uses explicit contract units; excess recorded exposure is rejected",
            "qualifying FIFO closes use realized profit net of allocated entry and exit fees",
            "qualifying close precedes inactivity at the identical timestamp",
            "maximum eligible payout at session close; requests approved on the scenario clock",
            "approval deducts gross; no trading while pending; downward scaling applies at approval",
            "retry and live handoff start fresh paid attempts no earlier than the next observed session",
            "live value and receipts after the horizon are excluded",
        ))
        return EventReplay(result, self.book.snapshot() if self.book is not None else None,
                           tuple(self.trace), self.fill_count, self.skip_count, self.mark_count)


def replay_events(spec, events, instruments, config, **kwargs):
    return _EventReplay(spec, events, instruments, config, **kwargs).run()
