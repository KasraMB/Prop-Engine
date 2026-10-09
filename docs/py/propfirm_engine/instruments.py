from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from math import isfinite
from numbers import Real


@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: str
    point_value: float
    tick_size: float

    def __post_init__(self):
        if (not isinstance(self.symbol, str) or not self.symbol
                or any(isinstance(x, bool) or not isinstance(x, (Real, Decimal))
                       or not isfinite(x) or x <= 0
                       for x in (self.point_value, self.tick_size))):
            raise ValueError("instrument requires a symbol and positive point/tick values")

    @property
    def tick_value(self):
        return Fraction(str(self.point_value)) * Fraction(str(self.tick_size))

    def ticks(self, price):
        value = Fraction(str(price)) / Fraction(str(self.tick_size))
        if value.denominator != 1:
            raise ValueError(f"price is off the tick grid for {self.symbol}")
        return int(value)
