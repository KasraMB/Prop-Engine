from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from fractions import Fraction

from .events import Fill, Marks, event_key, money
from .instruments import Instrument


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    quantity: int
    average_price: Fraction
    mark: Fraction
    unrealized: Fraction
    mark_at: datetime


@dataclass(frozen=True, slots=True)
class BookState:
    balance: Fraction
    realized: Fraction
    fees: Fraction
    unrealized: Fraction
    equity: Fraction
    positions: tuple[Position, ...]
    at: datetime | None


@dataclass(frozen=True, slots=True)
class FillEffect:
    realized: Fraction
    closed_net: Fraction
    closed: int


class _Position:
    __slots__ = ("instrument", "tick", "value", "lots", "quantity", "cost", "mark", "mark_at", "unrealized")

    def __init__(self, instrument):
        self.instrument = instrument
        self.tick = instrument.tick
        self.value = instrument.tick_value
        self.lots = deque()
        self.quantity = self.cost = 0
        self.mark = None
        self.mark_at = None
        self.unrealized = Fraction(0)

    def ticks(self, price):
        ticks = price / self.tick
        if ticks.denominator != 1:
            raise ValueError(f"price is off the tick grid for {self.instrument.symbol}")
        return int(ticks)

    def revalue(self):
        self.unrealized = (self.quantity * self.mark - self.cost) * self.value

    def preview(self, quantity, price, fee):
        total, closed, gross, costs = abs(quantity), 0, 0, Fraction(0)
        for held, entry, paid in self.lots:
            if not quantity or (quantity > 0) == (held > 0):
                break
            side = 1 if held > 0 else -1
            n = min(abs(quantity), abs(held))
            closed += n
            gross += n * side * (price - entry)
            costs += paid * n / abs(held)
            quantity += n * side
        return FillEffect(gross * self.value,
                          gross * self.value - costs - fee * closed / total, closed)

    def fill(self, quantity, price, fee):
        total = abs(quantity)
        realized = 0
        entry_cost = Fraction(0)
        closed_count = 0
        while quantity and self.lots and (quantity > 0) != (self.lots[0][0] > 0):
            held, entry, paid = self.lots[0]
            side = 1 if held > 0 else -1
            closed = min(abs(quantity), abs(held))
            allocated = paid * closed / abs(held)
            entry_cost += allocated
            closed_count += closed
            realized += closed * side * (price - entry)
            self.cost -= closed * side * entry
            self.quantity -= closed * side
            quantity += closed * side
            held -= closed * side
            if held:
                self.lots[0] = (held, entry, paid - allocated)
            else:
                self.lots.popleft()
        if quantity:
            paid = fee * abs(quantity) / total
            if self.lots and self.lots[-1][1:] == (price, 0) and paid == 0:
                self.lots[-1] = (self.lots[-1][0] + quantity, price, paid)
            else:
                self.lots.append((quantity, price, paid))
            self.quantity += quantity
            self.cost += quantity * price
        gross = realized * self.value
        return FillEffect(gross, gross - entry_cost - fee * closed_count / total, closed_count)


class Book:
    """FIFO futures accounting, without account rules or execution assumptions."""

    def __init__(self, instruments, *, balance=0, mark_fills=True):
        instruments = tuple(instruments)
        if (not instruments or any(not isinstance(i, Instrument) for i in instruments)
                or len({i.symbol for i in instruments}) != len(instruments)):
            raise ValueError("book needs distinct instruments")
        if type(mark_fills) is not bool:
            raise ValueError("mark_fills must be bool")
        self._positions = {i.symbol: _Position(i) for i in instruments}
        self._key = None
        self._balance = money(balance)
        self._realized = self._fees = self._unrealized = Fraction(0)
        self._mark_fills = mark_fills

    @property
    def mark_fills(self):
        return self._mark_fills

    @property
    def balance(self):
        return self._balance

    @property
    def equity(self):
        return self._balance + self._unrealized

    @property
    def unrealized(self):
        return self._unrealized

    @property
    def fees(self):
        return self._fees

    @property
    def realized(self):
        return self._realized

    @property
    def flat(self):
        return not any(p.quantity for p in self._positions.values())

    def quantity(self, symbol):
        return self._position(symbol).quantity

    def preview(self, fill):
        if not isinstance(fill, Fill):
            raise TypeError("expected Fill")
        p = self._position(fill.symbol)
        return p.preview(fill.quantity, p.ticks(fill.price), fill.fee)

    def copy_marks(self, other):
        if not isinstance(other, Book) or not self.flat or self._positions.keys() != other._positions.keys():
            raise ValueError("mark copying requires a flat book with matching instruments")
        for symbol, p in self._positions.items():
            source = other._positions[symbol]
            if p.instrument != source.instrument:
                raise ValueError("instrument definitions must match")
        for symbol, p in self._positions.items():
            source = other._positions[symbol]
            p.mark, p.mark_at = source.mark, source.mark_at
        if other._key is not None and (self._key is None or other._key > self._key):
            self._key = other._key

    def _check_time(self, at):
        if (not isinstance(at, datetime) or at.tzinfo is None or at.utcoffset() is None
                or (self._key is not None and at < self._key[0])):
            raise ValueError("book time must be aware and chronological")

    def adjust(self, at, amount):
        self._check_time(at)
        amount = money(amount)
        self._balance += amount
        if self._key is None or at > self._key[0]:
            self._key = at, -1

    def check_marks(self, at, max_age):
        self._check_time(at)
        if not isinstance(max_age, timedelta) or max_age < timedelta(0):
            raise ValueError("max_age must be a nonnegative timedelta")
        for symbol, p in self._positions.items():
            if p.quantity and (p.mark_at is None or at - p.mark_at > max_age):
                raise ValueError(f"stale or missing mark for {symbol} at {at.isoformat()}")

    def liquidation(self, at, fee):
        self._check_time(at)
        fee = money(fee)
        if fee < 0:
            raise ValueError("liquidation fee must be nonnegative")
        seq = self._key[1] + 1 if self._key is not None and self._key[0] == at else 0
        for p in self._positions.values():
            if p.quantity:
                yield Fill(at, p.instrument.symbol, -p.quantity, p.mark * p.tick,
                           fee * abs(p.quantity), seq)
                seq += 1

    def _position(self, symbol):
        try:
            return self._positions[symbol]
        except KeyError as exc:
            raise ValueError(f"unknown instrument: {symbol}") from exc

    def apply(self, event):
        if not isinstance(event, (Fill, Marks)):
            raise TypeError("expected Fill or Marks")
        key = event_key(event)
        if self._key is not None and key <= self._key:
            raise ValueError("event keys must strictly increase; order timestamp ties with seq")
        if isinstance(event, Marks):
            updates = [(self._position(symbol), price) for symbol, price in event.prices]
            updates = [(p, p.ticks(price)) for p, price in updates]
            for p, price in updates:
                if p.quantity:
                    self._unrealized -= p.unrealized
                p.mark = price
                p.mark_at = event.at
                if p.quantity:
                    p.revalue()
                    self._unrealized += p.unrealized
        else:
            p = self._position(event.symbol)
            price = p.ticks(event.price)
            if not self.mark_fills and p.mark is None:
                raise ValueError("a price mark is required before filling this instrument")
            effect = p.fill(event.quantity, price, event.fee)
            self._realized += effect.realized
            self._fees += event.fee
            self._balance += effect.realized - event.fee
            self._unrealized -= p.unrealized
            if self.mark_fills:
                p.mark = price
                p.mark_at = event.at
            p.revalue()
            self._unrealized += p.unrealized
        self._key = key
        return effect if isinstance(event, Fill) else None

    def snapshot(self):
        positions = tuple(Position(symbol, p.quantity, p.cost * p.tick / p.quantity,
                                   p.mark * p.tick, p.unrealized, p.mark_at)
                          for symbol, p in self._positions.items() if p.quantity)
        return BookState(self.balance, self.realized, self.fees, self.unrealized,
                         self.equity, positions, self._key[0] if self._key else None)
