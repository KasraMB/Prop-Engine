"""Immutable columnar quotes with zero-copy session windows."""
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone as utc_zone
from fractions import Fraction
from hashlib import sha256
from zoneinfo import ZoneInfo

import numpy as np

from .events import event_key
from .instruments import Instrument
from .orders import Market, Quote


_EPOCH = datetime(1970, 1, 1, tzinfo=utc_zone.utc)
_FRAME = np.dtype([("time", "<i8"), ("seq", "<i8"), ("offset", "<i8"),
                   ("count", "<i4"), ("session", "<i4")])
_QUOTE = np.dtype([("instrument", "<i4"), ("bid", "<i8"), ("ask", "<i8"),
                   ("mark", "<i8"), ("bid_size", "<i8"), ("ask_size", "<i8")])


def _freeze(rows, dtype):
    try:
        array = np.asarray(rows, dtype=dtype)
        return np.frombuffer(array.tobytes(), dtype=dtype)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("prepared market values exceed their integer storage range") from exc


@dataclass(frozen=True, init=False)
class MarketTape:
    instruments: tuple[Instrument, ...]
    sessions: tuple[date, ...]
    timezone: str
    session_open: time
    frames: np.ndarray
    quotes: np.ndarray
    offsets: np.ndarray
    fingerprint: str

    def __init__(self, markets, instruments, *, sessions, timezone="America/New_York", session_open=time(18)):
        instruments, sessions = tuple(instruments), tuple(sessions)
        if (not instruments or any(not isinstance(i, Instrument) for i in instruments)
                or len({i.symbol for i in instruments}) != len(instruments)):
            raise ValueError("prepared quotes require distinct instruments")
        if (not sessions or any(type(s) is not date for s in sessions)
                or any(b <= a for a, b in zip(sessions, sessions[1:]))):
            raise ValueError("sessions must be increasing date identifiers")
        if not isinstance(session_open, time) or session_open.tzinfo is not None:
            raise ValueError("session_open must be a local time")
        zone = ZoneInfo(timezone)
        lookup = {i.symbol: (n, i) for n, i in enumerate(instruments)}
        session_index = {day: n for n, day in enumerate(sessions)}
        frames, quotes = [], []
        counts = [0] * len(sessions)
        previous = None
        for market in markets:
            if not isinstance(market, Market):
                raise TypeError("MarketTape requires Market observations")
            key = event_key(market)
            if previous is not None and key <= previous:
                raise ValueError("market keys must strictly increase")
            previous = key
            local = market.at.astimezone(zone)
            day = local.date() + (timedelta(days=1) if local.time() >= session_open else timedelta(0))
            if day not in session_index:
                raise ValueError("market observation is outside the declared sessions")
            session = session_index[day]
            delta = market.at - _EPOCH
            micros = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
            frames.append((micros, market.seq, len(quotes), len(market.quotes), session))
            counts[session] += 1
            for quote in market.quotes:
                if quote.symbol not in lookup:
                    raise ValueError(f"unknown instrument: {quote.symbol}")
                index, instrument = lookup[quote.symbol]
                quotes.append((index, instrument.ticks(quote.bid), instrument.ticks(quote.ask),
                               instrument.ticks(quote.mark), -1 if quote.bid_size is None else quote.bid_size,
                               -1 if quote.ask_size is None else quote.ask_size))
        frame_data, quote_data = _freeze(frames, _FRAME), _freeze(quotes, _QUOTE)
        offsets = _freeze([0, *np.cumsum(counts).tolist()], np.dtype("<i8"))
        digest = sha256(repr((instruments, sessions, timezone, session_open)).encode())
        digest.update(frame_data.tobytes())
        digest.update(quote_data.tobytes())
        for name, value in dict(instruments=instruments, sessions=sessions, timezone=timezone,
                                session_open=session_open, frames=frame_data, quotes=quote_data,
                                offsets=offsets, fingerprint=digest.hexdigest()).items():
            object.__setattr__(self, name, value)

    @property
    def nbytes(self):
        return self.frames.nbytes + self.quotes.nbytes + self.offsets.nbytes

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        return iter(self.view())

    def view(self, first=0, last=None):
        return MarketView(self, first, len(self.sessions) if last is None else last)

    def market(self, index):
        frame = self.frames[index]
        rows = self.quotes[int(frame["offset"]):int(frame["offset"])+int(frame["count"])]
        quotes = []
        for row in rows:
            instrument = self.instruments[int(row["instrument"])]
            tick = instrument.tick
            quotes.append(Quote(instrument.symbol, int(row["bid"])*tick, int(row["ask"])*tick,
                                int(row["mark"])*tick,
                                None if row["bid_size"] < 0 else int(row["bid_size"]),
                                None if row["ask_size"] < 0 else int(row["ask_size"])))
        return Market(_EPOCH + timedelta(microseconds=int(frame["time"])), tuple(quotes), int(frame["seq"]))


@dataclass(frozen=True)
class MarketView:
    tape: MarketTape
    first: int
    last: int

    def __post_init__(self):
        if (not isinstance(self.tape, MarketTape) or type(self.first) is not int or type(self.last) is not int
                or not 0 <= self.first < self.last <= len(self.tape.sessions)):
            raise ValueError("market window must contain complete declared sessions")

    @property
    def sessions(self):
        return self.tape.sessions[self.first:self.last]

    @property
    def instruments(self):
        return self.tape.instruments

    @property
    def fingerprint(self):
        return sha256(f"{self.tape.fingerprint}:{self.first}:{self.last}".encode()).hexdigest()

    def __len__(self):
        return int(self.tape.offsets[self.last] - self.tape.offsets[self.first])

    def __iter__(self):
        for index in range(int(self.tape.offsets[self.first]), int(self.tape.offsets[self.last])):
            yield self.tape.market(index)

    def view(self, first=0, last=None):
        end = self.last-self.first if last is None else last
        if type(first) is not int or type(end) is not int or not 0 <= first < end <= self.last-self.first:
            raise ValueError("market window is outside this view")
        return MarketView(self.tape, self.first+first, self.first+end)
