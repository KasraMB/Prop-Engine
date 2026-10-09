"""External strategy callbacks over the shared event and account core."""
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction

from .enums import ExitCode
from .event_replay import EventReplay, _EventReplay
from .events import Fill, Marks, event_key
from .orders import Abandon, Broker, Market, OrderEvent, OrderState
from .portfolio import BookState


@dataclass(frozen=True, slots=True)
class Context:
    at: datetime
    phase: str
    next_phase: str | None
    attempt: int
    book: BookState | None
    floor: Fraction | None
    wallet: Fraction | None
    contract_limit: int
    days_to_payout: int
    payouts: int
    available: bool
    warmup: bool
    orders: tuple[OrderState, ...]
    phase_name: str
    next_phase_name: str | None


@dataclass(frozen=True)
class StrategyReplay:
    result: EventReplay
    orders: tuple[OrderEvent, ...]


class _StrategyReplay(_EventReplay):
    def __init__(self, spec, markets, instruments, config, strategy, *, models,
                 warmup=(), **kwargs):
        if not callable(getattr(strategy, "on_market", None)):
            raise TypeError("strategy must implement on_market(context, market)")
        if "mark_fills" in kwargs:
            raise ValueError("strategy execution uses explicit market marks")
        super().__init__(spec, (), instruments, config, mark_fills=False, **kwargs)
        self.strategy = strategy
        self.markets, self.warmup = iter(markets), iter(warmup)
        self.broker = Broker(self.instruments, models, self.units)
        self.last_market = None
        self.clock = self.start
        self.order_cursor = 0
        self.seq = 0
        self.day = 0

    def quantities(self):
        return {i.symbol: self.book.quantity(i.symbol) if self.book is not None else 0
                for i in self.instruments}

    def cap(self):
        if self.next_role is not None:
            return self.phase_limit(self.next_index)
        return self.limit

    def context(self, *, warmup=False):
        remaining = max(0, self.required_days-self.ledger.qualifying_days) if self.ledger else 0
        return Context(self.clock, self.role, self.next_role, self.attempts,
                       self.book.snapshot() if self.book else None,
                       self.sim.dd_floor if self.sim else None, self.wallet, self.cap(),
                       remaining, self.sim.payouts_taken if self.sim else 0,
                       not warmup and self.eligible(self.clock), warmup,
                       tuple(self.broker.pending.values()), self.phase_name(),
                       self.spec.account.phases[self.next_index].name if self.next_role else None)

    def actions(self, actions, *, enabled=True):
        if actions is None:
            return
        for action in actions:
            if isinstance(action, Abandon):
                if not enabled or not self.eligible(self.clock):
                    raise ValueError("abandonment is disabled outside an active trading session")
                self.broker.cancel_all(self.clock, "abandonment")
                self.abandon(self.clock, retry=action.retry, reason=action.reason)
                continue
            accepted = self.broker.submit(action, self.clock, self.quantities(), self.cap(),
                                           enabled=enabled and self.eligible(self.clock))
            if accepted and self.next_role is not None:
                if not self.start_phase(self.clock):
                    self.broker.cancel_all(self.clock, "wallet")
                else:
                    self.following = True

    def notifications(self, *, enabled=True):
        end = len(self.broker.events)
        callback = getattr(self.strategy, "on_order", None)
        events = self.broker.events[self.order_cursor:end]
        self.order_cursor = end
        if callback is not None:
            for event in events:
                self.actions(callback(self.context(), event), enabled=enabled)

    def restart(self, at, fee):
        self.broker.cancel_all(at, "account_ended")
        super().restart(at, fee)

    def handle_result(self, at, code, regime=None):
        if code == ExitCode.PASSED:
            self.broker.cancel_all(at, "phase_ended")
        super().handle_result(at, code, regime)

    def suspend(self, at):
        self.broker.cancel_all(at, "daily_suspend")
        super().suspend(at)

    def validate_market(self, market):
        if not isinstance(market, Market):
            raise TypeError("strategy replay requires Market observations")
        if self.last_market is not None and event_key(market) <= self.last_market:
            raise ValueError("market keys must strictly increase")
        self.last_market = event_key(market)
        for quote in market.quotes:
            instrument = self.broker.instruments.get(quote.symbol)
            if instrument is None:
                raise ValueError(f"unknown instrument: {quote.symbol}")
            for price in (quote.bid, quote.ask, quote.mark):
                instrument.ticks(price)
        self.digest.update(repr(market).encode())

    def mark(self, market):
        self.current_day = self.day
        self.broker.observe(market)
        self.seq += 1
        event = Marks(market.at, tuple((q.symbol, q.mark) for q in market.quotes), self.seq)
        self.source_book.apply(event)
        self.mark_count += 1
        if self.book is not None and self.next_role is None:
            self.book.apply(event)
            self.book.check_marks(event.at, self.max_mark_age)
            code = self.sim.observe(self.book.balance, self.book.equity, self.day)
            self.record(event.at, "mark")
            self.handle_result(event.at, code)

    def execute(self, order, quantity, price, fee):
        if not self.eligible(self.clock) or self.next_role is not None:
            return None
        holdings = self.quantities()
        holdings[order.symbol] += quantity
        exposure = sum(abs(n) * self.units[s] for s, n in holdings.items())
        if not order.reduce_only and exposure > self.limit:
            self.broker.cancel(order.id, self.clock, "contract_limit")
            return None
        self.seq += 1
        fill = Fill(self.clock, order.symbol, quantity, price, fee, self.seq)
        self.fill_count += 1
        self.settle(fill, self.day)
        return fill

    def finish_session(self, session, day):
        self.clock = self.close_at(session)
        self.broker.cancel_all(self.clock, "session_close", day_only=not self.spec.flatten_at_close)
        super().finish_session(session, day)
        self.notifications(enabled=False)
        callback = getattr(self.strategy, "on_session", None)
        if callback is not None:
            self.actions(callback(self.context(), session), enabled=False)

    def run(self):
        self.current_session = self.sessions[0]
        for market in self.warmup:
            self.validate_market(market)
            if market.at >= self.start:
                raise ValueError("warmup observations must precede the account horizon")
            self.clock = market.at
            self.actions(self.strategy.on_market(self.context(warmup=True), market), enabled=False)
            self.notifications(enabled=False)
        market = self.next_market()
        for self.day, session in enumerate(self.sessions):
            self.current_session = session
            opening, closing = self.open_at(session), self.close_at(session)
            while market is not None and market.at <= closing:
                self.validate_market(market)
                if market.at < opening:
                    raise ValueError("market observation lies outside a declared session")
                self.clock = market.at
                self.advance(market.at)
                self.mark(market)
                if self.book is not None and self.eligible(market.at) and self.next_role is None:
                    self.broker.match(market, self.quantities, self.execute)
                self.notifications()
                self.actions(self.strategy.on_market(self.context(), market))
                market = self.next_market()
            self.finish_session(session, self.day)
        if market is not None:
            raise ValueError("market observation lies after the declared horizon")
        self.advance(self.end)
        result = self.finish_result((
            "causal strategy callbacks; orders become executable on the next observed market event",
            "observed quote execution; no inferred intrabar paths or order book queue priority",
            "atomic quote marks precede order matching and callbacks; hard breach cancels all working orders",
            "quote sizes bound shared per-side liquidity; omitted sizes mean unlimited scenario liquidity",
            "stop triggers use supplied marks; market and stop gaps execute at supplied bid/ask plus the execution model",
            "OCO cancels siblings on any fill; linked reduce-only exits cannot reverse the portfolio",
            "session cutoff follows the profile; DAY orders expire, GTC survives only when holding is allowed",
            f"forced closes use fresh last marks and {self.liquidation_fee} fee per contract",
            "configured withdrawal, approval/denial and processing scenarios; no trading while pending",
            "retry and live handoff use the shared next-session, fee and wallet rules",
        ))
        return StrategyReplay(result, tuple(self.broker.events))

    def next_market(self):
        market = next(self.markets, None)
        if market is not None and not isinstance(market, Market):
            raise TypeError("strategy replay requires Market observations")
        return market


def replay_strategy(spec, markets, instruments, config, strategy, **kwargs):
    return _StrategyReplay(spec, markets, instruments, config, strategy, **kwargs).run()
