"""Causal quote execution with explicit liquidity and order state."""
from dataclasses import dataclass, replace
from datetime import datetime
from fractions import Fraction

from .events import Fill, _header, _symbol, money


@dataclass(frozen=True, slots=True)
class Quote:
    symbol: str
    bid: Fraction
    ask: Fraction
    mark: Fraction
    bid_size: int | None = None
    ask_size: int | None = None

    def __post_init__(self):
        _symbol(self.symbol)
        for name in ("bid", "ask", "mark"):
            object.__setattr__(self, name, money(getattr(self, name)))
        if self.bid > self.ask:
            raise ValueError("bid must not exceed ask")
        for size in (self.bid_size, self.ask_size):
            if size is not None and (type(size) is not int or size < 0):
                raise ValueError("quote sizes must be nonnegative integers or None")


@dataclass(frozen=True, slots=True)
class Market:
    at: datetime
    quotes: tuple[Quote, ...]
    seq: int = 0

    def __post_init__(self):
        _header(self)
        quotes = tuple(self.quotes)
        if (not quotes or any(not isinstance(q, Quote) for q in quotes)
                or len({q.symbol for q in quotes}) != len(quotes)):
            raise ValueError("market observations require distinct instrument quotes")
        object.__setattr__(self, "quotes", quotes)


@dataclass(frozen=True, slots=True)
class Order:
    id: str
    symbol: str
    quantity: int
    kind: str = "market"
    limit: Fraction | None = None
    stop: Fraction | None = None
    trail: Fraction | None = None
    tif: str = "day"
    reduce_only: bool = False
    oco: str | None = None
    parent: str | None = None
    expires: datetime | None = None

    def __post_init__(self):
        _symbol(self.id)
        _symbol(self.symbol)
        if type(self.quantity) is not int or not self.quantity:
            raise ValueError("order quantity must be a nonzero integer")
        if self.kind not in ("market", "limit", "stop", "stop_limit", "trailing"):
            raise ValueError("unsupported order kind")
        if self.tif not in ("day", "gtc", "ioc") or type(self.reduce_only) is not bool:
            raise ValueError("invalid time in force or reduce_only")
        needed = {"limit": ("limit",), "stop": ("stop",),
                  "stop_limit": ("stop", "limit"), "trailing": ("trail",)}.get(self.kind, ())
        for name in ("limit", "stop", "trail"):
            value = getattr(self, name)
            if (value is None) == (name in needed):
                raise ValueError(f"{name} is incompatible with order kind {self.kind}")
            if value is not None:
                object.__setattr__(self, name, money(value))
        if self.trail is not None and self.trail <= 0:
            raise ValueError("trailing distance must be positive")
        for name in (self.oco, self.parent):
            if name is not None:
                _symbol(name)
        if self.parent == self.id or (self.parent is not None and not self.reduce_only):
            raise ValueError("linked children must be reduce-only exits of another order")
        if self.expires is not None and (not isinstance(self.expires, datetime)
                or self.expires.tzinfo is None or self.expires.utcoffset() is None):
            raise ValueError("order expiry must be timezone-aware")


@dataclass(frozen=True, slots=True)
class Cancel:
    id: str


@dataclass(frozen=True, slots=True)
class Amend:
    order: Order


@dataclass(frozen=True, slots=True)
class Abandon:
    reason: str = "policy"
    retry: bool = True

    def __post_init__(self):
        _symbol(self.reason)
        if type(self.retry) is not bool:
            raise ValueError("retry must be bool")


@dataclass(frozen=True, slots=True)
class OrderState:
    order: Order
    remaining: int
    filled: int
    status: str
    triggered: bool = False
    anchor: Fraction | None = None


@dataclass(frozen=True, slots=True)
class OrderEvent:
    at: datetime
    id: str
    status: str
    reason: str | None = None
    fill: Fill | None = None


@dataclass(frozen=True, slots=True)
class QuoteModel:
    fee: Fraction = Fraction(0)
    fixed_fee: Fraction = Fraction(0)
    slip_ticks: int = 0

    def __post_init__(self):
        for name in ("fee", "fixed_fee"):
            value = money(getattr(self, name))
            if value < 0:
                raise ValueError("execution fees must be nonnegative")
            object.__setattr__(self, name, value)
        if type(self.slip_ticks) is not int or self.slip_ticks < 0:
            raise ValueError("slip_ticks must be a nonnegative integer")

    def price(self, order, quote, instrument):
        side = 1 if order.quantity > 0 else -1
        base = quote.ask if side > 0 else quote.bid
        price = base + side * self.slip_ticks * Fraction(str(instrument.tick_size))
        if order.limit is not None and side * (price - order.limit) > 0:
            return None
        return price

    def cost(self, quantity, *, first):
        return self.fee * abs(quantity) + (self.fixed_fee if first else 0)


class Broker:
    def __init__(self, instruments, models, units):
        self.instruments = {i.symbol: i for i in instruments}
        self.models = dict(models)
        if set(self.models) != set(self.instruments):
            raise ValueError("execution models must cover every instrument")
        if any(not callable(getattr(m, "price", None)) or not callable(getattr(m, "cost", None))
               for m in self.models.values()):
            raise TypeError("execution models must implement price and cost")
        self.units = units
        self.orders = {}
        self.events = []
        self.pending = {}
        self.marks = {}

    def observe(self, market):
        self.marks.update((q.symbol, q.mark) for q in market.quotes)

    def note(self, at, id, status, reason=None, fill=None):
        self.events.append(OrderEvent(at, id, status, reason, fill))

    def exposure(self, quantities, replacement=None):
        buys, sells = {}, {}
        orders = [s.order for k, s in self.pending.items() if replacement is None or k != replacement.id]
        if replacement is not None:
            orders.append(replacement)
        for order in orders:
            if order.reduce_only:
                continue
            state = self.orders.get(order.id)
            n = abs(order.quantity) - (state.filled if state is not None else 0)
            dest = buys if order.quantity > 0 else sells
            dest[order.symbol] = dest.get(order.symbol, 0) + n
        return sum(max(abs(quantities.get(s, 0) + buys.get(s, 0)),
                       abs(quantities.get(s, 0) - sells.get(s, 0))) * u
                   for s, u in self.units.items())

    def submit(self, action, at, quantities, limit, *, enabled=True):
        if isinstance(action, Cancel):
            self.cancel(action.id, at, "requested")
            return
        amend = isinstance(action, Amend)
        order = action.order if amend else action
        if not isinstance(order, Order):
            raise TypeError("strategy actions must be Order, Amend or Cancel")
        old = self.orders.get(order.id)
        reason = None
        if not enabled:
            reason = "account_unavailable"
        elif (amend and order.id not in self.pending) or (not amend and old is not None):
            reason = "unknown_order" if amend else "duplicate_id"
        elif order.symbol not in self.instruments:
            reason = "unknown_instrument"
        elif amend and (order.symbol != old.order.symbol or order.quantity * old.order.quantity < 0
                        or abs(order.quantity) <= old.filled or order.parent != old.order.parent):
            reason = "invalid_amendment"
        elif order.expires is not None and order.expires <= at:
            reason = "expired"
        elif order.parent is not None:
            parent = self.orders.get(order.parent)
            if (parent is None or parent.order.symbol != order.symbol
                    or parent.order.quantity * order.quantity >= 0
                    or parent.status in ("cancelled", "rejected") and not parent.filled):
                reason = "invalid_parent"
        if reason is None:
            for price in (order.limit, order.stop, order.trail):
                if price is not None:
                    self.instruments[order.symbol].ticks(price)
            if not order.reduce_only and self.exposure(quantities, order) > limit:
                reason = "contract_limit"
        if reason is not None:
            self.note(at, order.id, "rejected", reason)
            return False
        filled = old.filled if amend else 0
        anchor = self.marks.get(order.symbol) if order.kind == "trailing" and order.parent is None else None
        state = OrderState(order, abs(order.quantity) - filled, filled, "working", anchor=anchor)
        self.orders[order.id] = self.pending[order.id] = state
        self.note(at, order.id, "amended" if amend else "accepted")
        return True

    def cancel(self, id, at, reason):
        state = self.pending.pop(id, None)
        if state is None:
            self.note(at, id, "rejected", "not_working")
            return
        self.orders[id] = replace(state, status="cancelled")
        self.note(at, id, "cancelled", reason)
        if not state.filled:
            for child, linked in tuple(self.pending.items()):
                if linked.order.parent == id:
                    self.cancel(child, at, "parent_cancelled")

    def cancel_all(self, at, reason, *, day_only=False):
        for id, state in tuple(self.pending.items()):
            if not day_only or state.order.tif == "day":
                self.cancel(id, at, reason)

    def match(self, market, quantities, execute):
        quotes = {q.symbol: q for q in market.quotes}
        capacity = {(q.symbol, side): n for q in market.quotes
                    for side, n in ((1, q.ask_size), (-1, q.bid_size))}
        for id in tuple(self.pending):
            state = self.pending.get(id)
            if state is None:
                continue
            order = state.order
            if order.expires is not None and market.at >= order.expires:
                self.cancel(id, market.at, "expired")
                continue
            quote = quotes.get(order.symbol)
            if quote is None:
                continue
            parent = self.orders.get(order.parent) if order.parent else None
            if parent is not None and not parent.filled:
                continue
            side = 1 if order.quantity > 0 else -1
            anchor, triggered = state.anchor, state.triggered
            if order.kind == "trailing":
                anchor = quote.mark if anchor is None else (min(anchor, quote.mark) if side > 0 else max(anchor, quote.mark))
                stop = anchor + side * order.trail
            else:
                stop = order.stop
            if stop is not None and side * (quote.mark - stop) >= 0:
                triggered = True
            state = replace(state, triggered=triggered, anchor=anchor)
            self.orders[id] = self.pending[id] = state
            active = order.kind in ("market", "limit") or triggered
            n = state.remaining
            if parent is not None:
                n = min(n, max(0, parent.filled - state.filled))
            if order.reduce_only:
                held = quantities().get(order.symbol, 0)
                n = min(n, abs(held)) if held * side < 0 else 0
            available = capacity[order.symbol, side]
            if available is not None:
                n = min(n, available)
            model = self.models[order.symbol]
            price = model.price(order, quote, self.instruments[order.symbol]) if active and n else None
            if price is not None:
                price = money(price)
                self.instruments[order.symbol].ticks(price)
                if order.limit is not None and side * (price - order.limit) > 0:
                    raise ValueError("execution model exceeded the order limit")
                fill = execute(order, side * n, price, model.cost(side * n, first=not state.filled))
                if fill is None:
                    break
                if available is not None:
                    capacity[order.symbol, side] -= n
                state = replace(state, filled=state.filled+n, remaining=state.remaining-n,
                                status="filled" if n == state.remaining else "partial")
                if id not in self.pending and state.remaining:
                    state = replace(state, status="cancelled")
                self.orders[id] = state
                if id in self.pending:
                    if not state.remaining:
                        del self.pending[id]
                    else:
                        self.pending[id] = state
                self.note(market.at, id, state.status, fill=fill)
                if order.oco is not None:
                    for other, sibling in tuple(self.pending.items()):
                        if other != id and sibling.order.oco == order.oco:
                            self.cancel(other, market.at, "oco")
            if order.tif == "ioc" and id in self.pending:
                self.cancel(id, market.at, "ioc_remainder")
