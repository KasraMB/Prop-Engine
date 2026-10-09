"""Causal delivery of external opportunities to an order policy."""
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from .events import _header, _symbol, event_key, money
from .strategy import StrategyReplay, replay_strategy


@dataclass(frozen=True, slots=True)
class Opportunity:
    at: datetime
    symbol: str
    side: int
    stop: object = None
    target: object = None
    tag: str = ""
    expires: datetime | None = None
    seq: int = 0

    def __post_init__(self):
        _header(self)
        _symbol(self.symbol)
        if type(self.side) is not int or self.side not in (-1, 1) or not isinstance(self.tag, str):
            raise ValueError("opportunity side must be +/-1 and tag must be a string")
        for name in ("stop", "target"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, money(value))
        if self.expires is not None:
            if (not isinstance(self.expires, datetime) or self.expires.tzinfo is None
                    or self.expires.utcoffset() is None or self.expires <= self.at):
                raise ValueError("opportunity expiry must follow its aware release time")
            object.__setattr__(self, "expires", self.expires.astimezone(timezone.utc))


class OpportunityStrategy:
    """External signals are not regenerated when account fills or exits change."""

    def __init__(self, opportunities, policy):
        if not callable(getattr(policy, "on_opportunity", None)):
            raise TypeError("policy must implement on_opportunity(context, opportunity, market)")
        self.policy = policy
        self.stream = iter(opportunities)
        self.pending = None
        self.previous = None
        self.offered = self.expired = 0
        self.before_start = 0
        self.warmup_start = None
        self.done = False

    def take(self):
        if self.done or self.pending is not None:
            return
        opportunity = next(self.stream, None)
        if opportunity is None:
            self.done = True
            return
        if not isinstance(opportunity, Opportunity):
            raise TypeError("expected Opportunity")
        key = event_key(opportunity)
        if self.previous is not None and key <= self.previous:
            raise ValueError("opportunity keys must strictly increase")
        self.previous, self.pending = key, opportunity

    def on_market(self, context, market):
        if context.warmup and self.warmup_start is None:
            self.warmup_start = market.at
        callback = getattr(self.policy, "on_market", None)
        if callback is not None:
            yield from callback(context, market) or ()
        self.take()
        while self.pending is not None and self.pending.at <= market.at:
            opportunity, self.pending = self.pending, None
            boundary = self.warmup_start if context.warmup else context.start
            if opportunity.at < boundary:
                self.before_start += 1
            elif opportunity.expires is not None and market.at >= opportunity.expires:
                self.expired += 1
            else:
                self.offered += 1
                yield from self.policy.on_opportunity(context, opportunity, market) or ()
            self.take()

    def on_order(self, context, event):
        callback = getattr(self.policy, "on_order", None)
        if callback is not None:
            return callback(context, event)

    def on_session(self, context, session):
        callback = getattr(self.policy, "on_session", None)
        if callback is not None:
            return callback(context, session)


@dataclass(frozen=True)
class OpportunityReplay:
    replay: StrategyReplay
    offered: int
    expired: int
    before_start: int


def replay_opportunities(spec, markets, instruments, config, opportunities, policy, **kwargs):
    strategy = OpportunityStrategy(opportunities, policy)
    result = replay_strategy(spec, markets, instruments, config, strategy, **kwargs)
    assumptions = result.result.replay.assumptions + (
        "external opportunities released no earlier than their timestamps; expired/pre-horizon signals are skipped",
        "external opportunities remain fixed when sizing, account state or exits change; dependent signals need strategy callbacks",
    )
    result = replace(result, result=replace(result.result, replay=replace(result.result.replay, assumptions=assumptions)))
    return OpportunityReplay(result, strategy.offered, strategy.expired, strategy.before_start)
