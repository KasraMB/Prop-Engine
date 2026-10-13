"""Exact decimal values with an integer representation when no fraction is needed."""
from fractions import Fraction
from functools import lru_cache


@lru_cache(maxsize=4096, typed=True)
def decimal_cash(value):
    if type(value) is int:
        return value
    amount = value if isinstance(value, Fraction) else Fraction(str(value))
    return amount.numerator if amount.denominator == 1 else amount
