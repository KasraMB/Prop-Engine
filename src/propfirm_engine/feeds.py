"""Explicit market-input scenarios; adapters do not infer missing market paths."""
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from heapq import merge
from itertools import groupby

from .events import _header, _symbol, event_key, money
from .orders import Market, MarketSource, Quote


@dataclass(frozen=True)
class MarketFeed:
    events: object
    fidelity: str = "observed_marks"
    assumptions: tuple[str, ...] = ()

    def __post_init__(self):
        source = MarketSource(self.fidelity, self.assumptions)
        object.__setattr__(self, "assumptions", source.assumptions)

    def __iter__(self):
        source = MarketSource(self.fidelity, self.assumptions)
        for event in self.events:
            if not isinstance(event, Market):
                raise TypeError("market feeds require Market observations")
            if (event.source is not None and event.source.fidelity != self.fidelity
                    and self.fidelity != "mixed_scenario"):
                raise ValueError("a feed cannot relabel another source fidelity")
            inherited = event.source.assumptions if event.source is not None else ()
            yield replace(event, source=MarketSource(source.fidelity,
                          tuple(dict.fromkeys((*source.assumptions, *inherited)))))


@dataclass(frozen=True, slots=True)
class TradeTick:
    at: datetime
    symbol: str
    price: object
    seq: int = 0

    def __post_init__(self):
        _header(self)
        _symbol(self.symbol)
        object.__setattr__(self, "price", money(self.price))


@dataclass(frozen=True, slots=True)
class Bar:
    at: datetime
    end: datetime
    symbol: str
    open: object
    high: object
    low: object
    close: object
    seq: int = 0

    def __post_init__(self):
        _header(self)
        _symbol(self.symbol)
        if (not isinstance(self.end, datetime) or self.end.tzinfo is None or self.end.utcoffset() is None
                or (self.end-self.at).total_seconds() < .000003):
            raise ValueError("bar end must follow its aware opening by at least three microseconds")
        object.__setattr__(self, "end", self.end.astimezone(timezone.utc))
        for name in ("open", "high", "low", "close"):
            object.__setattr__(self, name, money(getattr(self, name)))
        if not self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high:
            raise ValueError("bar extrema must contain open and close")


def _scenario(instruments, half_spread_ticks, liquidity):
    instruments = tuple(instruments)
    lookup = {i.symbol: i for i in instruments}
    if not lookup or len(lookup) != len(instruments):
        raise ValueError("distinct instruments are required")
    if type(half_spread_ticks) is not int or half_spread_ticks < 0:
        raise ValueError("half_spread_ticks must be a nonnegative integer")
    if liquidity is not None and (type(liquidity) is not int or liquidity < 0):
        raise ValueError("liquidity must be a nonnegative integer or explicit None")
    return lookup


def trade_quotes(ticks, instruments, *, half_spread_ticks, liquidity):
    """Last prints become scenario quotes, not recovered historical bid/ask depth."""
    lookup = _scenario(instruments, half_spread_ticks, liquidity)

    def events():
        previous = None
        for tick in ticks:
            if not isinstance(tick, TradeTick):
                raise TypeError("trade_quotes requires TradeTick observations")
            key = event_key(tick)
            if previous is not None and key <= previous:
                raise ValueError("trade tick keys must strictly increase")
            previous = key
            if tick.symbol not in lookup:
                raise ValueError("unknown trade instrument")
            instrument = lookup[tick.symbol]
            instrument.ticks(tick.price)
            spread = instrument.tick*half_spread_ticks
            yield Market(tick.at, (Quote(tick.symbol, tick.price-spread, tick.price+spread,
                                         tick.price, liquidity, liquidity),), tick.seq)

    return MarketFeed(events(), "last_trade", (
        "last-trade marks; unobserved equity between prints is unknown",
        f"synthetic bid/ask at +/-{half_spread_ticks} ticks; per-side observation liquidity: {liquidity}",
        "print volume does not establish quote depth or queue priority",
    ))


def bar_quotes(bars, instrument, *, path, half_spread_ticks, liquidity):
    """Four-point OHLC path scenario for one instrument; not intrabar reconstruction."""
    if path not in ("ohlc", "olhc"):
        raise ValueError("select an explicit ohlc or olhc path approximation")
    _scenario((instrument,), half_spread_ticks, liquidity)
    spread = instrument.tick*half_spread_ticks

    def events():
        previous_end = None
        previous_seq = None
        for bar in bars:
            if not isinstance(bar, Bar) or bar.symbol != instrument.symbol:
                raise ValueError("bar_quotes requires bars for its selected instrument")
            if previous_end is not None and (bar.at < previous_end
                    or bar.at == previous_end and bar.seq*4 <= previous_seq):
                raise ValueError("bars must not overlap; use increasing seq at adjoining boundaries")
            for price in (bar.open, bar.high, bar.low, bar.close):
                instrument.ticks(price)
            points = (bar.open, bar.high, bar.low, bar.close) if path == "ohlc" else (
                bar.open, bar.low, bar.high, bar.close)
            for index, price in enumerate(points):
                at = bar.at + (bar.end-bar.at)*index/3
                yield Market(at, (Quote(bar.symbol, price-spread, price+spread, price, liquidity, liquidity),),
                             bar.seq*4+index)
            previous_end, previous_seq = bar.end, bar.seq*4+3

    return MarketFeed(events(), "ohlc_path", (
        f"OHLC approximation: {path}; extrema assigned to one-third/two-thirds of each bar",
        "these synthetic event times and intrabar order are not observed market history",
        "callbacks see only each released synthetic point; gaps can skip stops and limit fills are scenarios",
        f"synthetic bid/ask at +/-{half_spread_ticks} ticks; per-side point liquidity: {liquidity}",
        "bar volume, simultaneous asset extrema and real quote queues are not inferred",
    ))


def _ordered(feed):
    previous = None
    for event in feed:
        if not isinstance(event, Market):
            raise TypeError("market feeds require Market observations")
        key = event_key(event)
        if previous is not None and key <= previous:
            raise ValueError("each market feed must strictly increase")
        previous = key
        yield event


def merge_markets(*feeds, ties):
    """Merge ordered feeds without sorting entire histories; ties must be declared."""
    if not feeds or ties not in ("reject", "atomic"):
        raise ValueError("provide feeds and choose reject or atomic timestamp/sequence ties")
    kinds = {getattr(feed, "fidelity", "observed_marks") for feed in feeds}
    kind = next(iter(kinds)) if len(kinds) == 1 else "mixed_scenario"
    assumptions = tuple(dict.fromkeys(a for feed in feeds for a in getattr(feed, "assumptions", ())))

    def events():
        stream = merge(*(_ordered(feed) for feed in feeds), key=event_key)
        for _, group in groupby(stream, key=event_key):
            first = next(group)
            quotes = list(first.quotes)
            inherited = list(first.source.assumptions if first.source is not None else ())
            if first.source is not None and first.source.fidelity not in kinds:
                raise ValueError("preserve source wrappers when merging approximate feeds")
            for other in group:
                if other.source is not None and other.source.fidelity not in kinds:
                    raise ValueError("preserve source wrappers when merging approximate feeds")
                if ties == "reject":
                    raise ValueError("market feed keys collide; declare atomic quotes or supply ordered sequences")
                quotes.extend(other.quotes)
                if other.source is not None:
                    inherited.extend(other.source.assumptions)
            yield Market(first.at, tuple(quotes), first.seq,
                         MarketSource(kind, tuple(dict.fromkeys(inherited))))

    return MarketFeed(events(), kind, assumptions+(f"merged market feed key ties: {ties}",))
