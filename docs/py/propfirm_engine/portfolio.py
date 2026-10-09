from collections import deque
from dataclasses import dataclass
from datetime import datetime
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


class _Position:
    __slots__ = ("instrument", "tick", "value", "lots", "quantity", "cost", "mark", "mark_at", "unrealized")

    def __init__(self, instrument):
        self.instrument = instrument
        self.tick = Fraction(str(instrument.tick_size))
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

    def fill(self, quantity, price):
        realized = 0
        while quantity and self.lots and (quantity > 0) != (self.lots[0][0] > 0):
            held, entry = self.lots[0]
            side = 1 if held > 0 else -1
            closed = min(abs(quantity), abs(held))
            realized += closed * side * (price - entry)
            self.cost -= closed * side * entry
            self.quantity -= closed * side
            quantity += closed * side
            held -= closed * side
            if held:
                self.lots[0] = (held, entry)
            else:
                self.lots.popleft()
        if quantity:
            if self.lots and self.lots[-1][1] == price:
                self.lots[-1] = (self.lots[-1][0] + quantity, price)
            else:
                self.lots.append((quantity, price))
            self.quantity += quantity
            self.cost += quantity * price
        return realized * self.value


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
                self._unrealized -= p.unrealized
                p.mark = price
                p.mark_at = event.at
                p.revalue()
                self._unrealized += p.unrealized
        else:
            p = self._position(event.symbol)
            price = p.ticks(event.price)
            if not self.mark_fills and p.mark is None:
                raise ValueError("a price mark is required before filling this instrument")
            realized = p.fill(event.quantity, price)
            self._realized += realized
            self._fees += event.fee
            self._balance += realized - event.fee
            self._unrealized -= p.unrealized
            if self.mark_fills:
                p.mark = price
                p.mark_at = event.at
            p.revalue()
            self._unrealized += p.unrealized
        self._key = key

    def snapshot(self):
        positions = tuple(Position(symbol, p.quantity, p.cost * p.tick / p.quantity,
                                   p.mark * p.tick, p.unrealized, p.mark_at)
                          for symbol, p in self._positions.items() if p.quantity)
        return BookState(self.balance, self.realized, self.fees, self.unrealized,
                         self.equity, positions, self._key[0] if self._key else None)
