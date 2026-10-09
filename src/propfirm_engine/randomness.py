"""Keyed random draws that do not depend on policy execution counts."""
from dataclasses import dataclass
from hashlib import blake2b
from json import dumps
from math import cos, isfinite, log1p, pi, sqrt


def _key(value):
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and isfinite(value):
        return value
    if isinstance(value, (tuple, list)):
        return [_key(v) for v in value]
    raise ValueError("random keys must contain finite scalar values or sequences")


@dataclass(frozen=True)
class RandomStream:
    seed: int
    channel: str = "execution"

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0 or not isinstance(self.channel, str):
            raise ValueError("random stream requires a nonnegative seed and string channel")

    def uniform(self, *key):
        encoded = dumps([self.seed, self.channel, _key(key)], ensure_ascii=True,
                        separators=(",", ":"), allow_nan=False).encode()
        value = int.from_bytes(blake2b(encoded, digest_size=8).digest(), "big")
        return (value >> 11) / 2**53

    def normal(self, *key):
        radius = sqrt(-2*log1p(-self.uniform("radius", key)))
        return radius*cos(2*pi*self.uniform("angle", key))
