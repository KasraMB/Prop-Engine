"""Causal quote execution with explicit liquidity and order state."""
from dataclasses import dataclass, replace
from collections import Counter
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
class MarketSource:
    fidelity: str = "observed_marks"
    assumptions: tuple[str, ...] = ()

    def __post_init__(self):
        if self.fidelity not in ("observed_marks", "last_trade", "ohlc_path", "mixed_scenario"):
            raise ValueError("unknown market source fidelity")
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        if any(not isinstance(a, str) for a in self.assumptions):
            raise ValueError("source assumptions must be strings")


@dataclass(frozen=True, slots=True)
class Market:
    at: datetime
    quotes: tuple[Quote, ...]
    seq: int = 0
    source: MarketSource | None = None

    def __post_init__(self):
        _header(self)
        if self.source is not None and not isinstance(self.source, MarketSource):
            raise TypeError("market source must be MarketSource")
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
        price = base + side * self.slip_ticks * instrument.tick
        if order.limit is not None and side * (price - order.limit) > 0:
            return None
        return price

    def cost(self, quantity, *, first):
        return self.fee * abs(quantity) + (self.fixed_fee if first else 0)


@dataclass(frozen=True, slots=True)
class Basket:
    id: str
    legs: tuple[Order, ...]

    def __post_init__(self):
        _symbol(self.id)
        object.__setattr__(self, "legs", tuple(self.legs))
        if (len(self.legs) < 2 or any(not isinstance(o, Order) for o in self.legs)
                or len({o.symbol for o in self.legs}) != len(self.legs)
                or len({o.id for o in self.legs}) != len(self.legs)
                or self.id in {o.id for o in self.legs}):
            raise ValueError("basket needs distinct leg IDs and instruments, separate from its group ID")
        if (any(o.kind not in ("market", "limit") or o.parent is not None or o.oco is not None for o in self.legs)
                or len({o.tif for o in self.legs}) != 1):
            raise ValueError("atomic baskets require market/limit legs, one time in force and no parent/OCO")


@dataclass(frozen=True, slots=True)
class _OrderRef:
    symbol: str
    quantity: int


@dataclass(frozen=True, slots=True)
class _ClosedOrder:
    order: _OrderRef
    filled: int
    status: str


class Broker:
    def __init__(self, instruments, models, units, *, compact=False, sink=None):
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
        self.compact, self.sink = compact, sink
        self.counts = Counter()
        self.groups, self.members, self.group_ids = {}, {}, set()

    def observe(self, market):
        self.marks.update((q.symbol, q.mark) for q in market.quotes)

    def archive(self, state):
        if self.compact and state.status in ("filled", "cancelled"):
            return _ClosedOrder(_OrderRef(state.order.symbol, state.order.quantity), state.filled, state.status)
        return state

    def note(self, at, id, status, reason=None, fill=None):
        event = OrderEvent(at, id, status, reason, fill)
        self.counts[status] += 1
        if self.sink is not None:
            self.sink(event)
        self.events.append(event)

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
        if isinstance(action, Basket):
            return self.submit_basket(action, at, quantities, limit, enabled=enabled)
        if isinstance(action, Cancel):
            self.cancel(action.id, at, "requested")
            return
        amend = isinstance(action, Amend)
        order = action.order if amend else action
        if not isinstance(order, Order):
            raise TypeError("strategy actions must be Order, Basket, Amend or Cancel")
        old = self.orders.get(order.id)
        reason = None
        if not enabled:
            reason = "account_unavailable"
        elif amend and order.id in self.members:
            reason = "cancel_replace_basket"
        elif (amend and order.id not in self.pending) or (not amend and (old is not None or order.id in self.group_ids)):
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

    def submit_basket(self, basket, at, quantities, limit, *, enabled):
        if basket.id in self.orders or basket.id in self.group_ids:
            self.note(at, basket.id, "rejected", "duplicate_id")
            return False
        self.group_ids.add(basket.id)
        accepted = []
        for leg in basket.legs:
            if not self.submit(leg, at, quantities, limit, enabled=enabled):
                for id in accepted:
                    self.cancel(id, at, "basket_rejected")
                self.note(at, basket.id, "rejected", "invalid_leg")
                return False
            accepted.append(leg.id)
        self.groups[basket.id] = tuple(accepted)
        self.members.update((id, basket.id) for id in accepted)
        self.note(at, basket.id, "accepted")
        return True

    def cancel(self, id, at, reason):
        if id in self.members:
            id = self.members[id]
        if id in self.groups:
            members = self.groups.pop(id)
            for leg in members:
                self.members.pop(leg, None)
            for leg in members:
                self.cancel(leg, at, reason)
            self.note(at, id, "cancelled", reason)
            return
        state = self.pending.pop(id, None)
        if state is None:
            self.note(at, id, "rejected", "not_working")
            return
        self.orders[id] = self.archive(replace(state, status="cancelled"))
        self.note(at, id, "cancelled", reason)
        if not state.filled:
            for child, linked in tuple(self.pending.items()):
                if linked.order.parent == id:
                    self.cancel(child, at, "parent_cancelled")

    def cancel_all(self, at, reason, *, day_only=False):
        for id, state in tuple(self.pending.items()):
            if id in self.pending and (not day_only or state.order.tif == "day"):
                self.cancel(id, at, reason)

    def match(self, market, quantities, execute, execute_many=None):
        quotes = {q.symbol: q for q in market.quotes}
        capacity = {(q.symbol, side): n for q in market.quotes
                    for side, n in ((1, q.ask_size), (-1, q.bid_size))}
        handled = set()
        for id in tuple(self.pending):
            state = self.pending.get(id)
            if state is None:
                continue
            if id in self.members:
                group = self.members[id]
                if group not in handled:
                    handled.add(group)
                    self.match_basket(group, market, quotes, capacity, quantities, execute_many)
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
                self.orders[id] = self.archive(state)
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

    def match_basket(self, group, market, quotes, capacity, quantities, execute):
        if execute is None:
            raise ValueError("atomic basket execution requires a portfolio executor")
        ids = self.groups[group]
        states = [self.pending[id] for id in ids]
        if any(s.order.expires is not None and s.order.expires <= market.at for s in states):
            self.cancel(group, market.at, "expired")
            return
        fills, holdings = [], quantities()
        for state in states:
            order = state.order
            quote = quotes.get(order.symbol)
            side, n = (1 if order.quantity > 0 else -1), abs(order.quantity)
            available = capacity.get((order.symbol, side))
            held = holdings.get(order.symbol, 0)
            if (quote is None or available is not None and available < n
                    or order.reduce_only and (held*side >= 0 or abs(held) < n)):
                break
            model, instrument = self.models[order.symbol], self.instruments[order.symbol]
            price = model.price(order, quote, instrument)
            if price is None:
                break
            price = money(price)
            instrument.ticks(price)
            if order.limit is not None and side*(price-order.limit) > 0:
                raise ValueError("execution model exceeded the order limit")
            fills.append((order, order.quantity, price, model.cost(order.quantity, first=True)))
        if len(fills) == len(states):
            executed = execute(fills)
            if executed is None:
                return
            self.groups.pop(group, None)
            for state, fill in zip(states, executed, strict=True):
                order = state.order
                self.pending.pop(order.id, None)
                self.members.pop(order.id, None)
                self.orders[order.id] = self.archive(replace(state, remaining=0,
                    filled=abs(order.quantity), status="filled"))
                side = 1 if order.quantity > 0 else -1
                if capacity[order.symbol, side] is not None:
                    capacity[order.symbol, side] -= abs(order.quantity)
                self.note(market.at, order.id, "filled", fill=fill)
            self.note(market.at, group, "filled")
        elif states[0].order.tif == "ioc":
            self.cancel(group, market.at, "ioc_remainder")
