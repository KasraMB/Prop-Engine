from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from heapq import merge
from numbers import Integral, Real


def money(value):
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError("price and fee must be finite numbers")
    try:
        return value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError("price and fee must be finite numbers") from exc


def _header(event):
    if (not isinstance(event.at, datetime) or event.at.tzinfo is None
            or event.at.utcoffset() is None):
        raise ValueError("event time must be timezone-aware")
    if isinstance(event.seq, bool) or not isinstance(event.seq, Integral) or event.seq < 0:
        raise ValueError("event sequence must be a nonnegative integer")
    object.__setattr__(event, "at", event.at.astimezone(timezone.utc))
    object.__setattr__(event, "seq", int(event.seq))


def _symbol(symbol):
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("symbol must be a nonempty string")


@dataclass(frozen=True, slots=True)
class Fill:
    at: datetime
    symbol: str
    quantity: int
    price: Fraction
    fee: Fraction = Fraction(0)
    seq: int = 0

    def __post_init__(self):
        _header(self)
        _symbol(self.symbol)
        if (isinstance(self.quantity, bool) or not isinstance(self.quantity, Integral)
                or self.quantity == 0):
            raise ValueError("fill quantity must be a nonzero signed integer")
        object.__setattr__(self, "quantity", int(self.quantity))
        object.__setattr__(self, "price", money(self.price))
        object.__setattr__(self, "fee", money(self.fee))
        if self.fee < 0:
            raise ValueError("fill fee must be nonnegative")


@dataclass(frozen=True, slots=True)
class Marks:
    """Prices observed together; portfolio valuation updates atomically."""

    at: datetime
    prices: tuple[tuple[str, Fraction], ...]
    seq: int = 0

    def __post_init__(self):
        _header(self)
        prices = tuple((symbol, money(price)) for symbol, price in self.prices)
        for symbol, _ in prices:
            _symbol(symbol)
        if not prices or len({symbol for symbol, _ in prices}) != len(prices):
            raise ValueError("marks need distinct instruments")
        object.__setattr__(self, "prices", prices)


def event_key(event):
    return event.at, event.seq


def ordered(events):
    """Validate lazily without sorting or repairing the supplied history."""
    previous = None
    for event in events:
        if not isinstance(event, (Fill, Marks)):
            raise TypeError("expected Fill or Marks")
        key = event_key(event)
        if previous is not None and key <= previous:
            raise ValueError("event keys must strictly increase; order timestamp ties with seq")
        previous = key
        yield event


def merge_events(*feeds):
    return ordered(merge(*(ordered(feed) for feed in feeds), key=event_key))
