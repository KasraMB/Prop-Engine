"""Explicit bracket-history, dollar-policy and lifecycle contracts.

This is an input adapter for the existing simulator, not a price-path generator.
Historical targets are immutable: changing them requires replayable market data.
"""
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from math import isfinite
from numbers import Integral, Real

from .model import Account


def _number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")


def _aware(value, name):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class BracketTrade:
    """One sequential, ideal-fill trade; stop/target are gross dollars per contract."""

    entry_at: datetime
    exit_at: datetime
    session: date
    stop_loss: float
    take_profit: float
    won: bool

    def __post_init__(self):
        entry, exit = _aware(self.entry_at, "entry_at"), _aware(self.exit_at, "exit_at")
        if exit <= entry:
            raise ValueError("exit_at must follow entry_at")
        if not isinstance(self.session, date) or isinstance(self.session, datetime):
            raise ValueError("session must be a date")
        for name in ("stop_loss", "take_profit"):
            _number(getattr(self, name), name)
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        if type(self.won) is not bool:
            raise ValueError("won must be a bool")
        object.__setattr__(self, "entry_at", entry)
        object.__setattr__(self, "exit_at", exit)


@dataclass(frozen=True)
class BracketHistory:
    """Immutable chronological input. No sorting or overlap repair is performed."""

    trades: tuple[BracketTrade, ...]

    def __post_init__(self):
        object.__setattr__(self, "trades", tuple(self.trades))
        if not self.trades or any(not isinstance(t, BracketTrade) for t in self.trades):
            raise ValueError("history must contain BracketTrade objects")
        for previous, current in zip(self.trades, self.trades[1:]):
            if current.entry_at < previous.exit_at or current.session < previous.session:
                raise ValueError("history must be chronological, sequential and non-overlapping")

    @property
    def sessions(self):
        return tuple(dict.fromkeys(t.session for t in self.trades))

    @classmethod
    def from_records(cls, records):
        """Import CSV/dict records with explicit clocks, session, brackets and outcome.

        Required columns: entry_at, exit_at, session, stop_loss, take_profit, won.
        Clocks are aware ISO-8601 strings or datetimes; won is bool or true/false.
        Stop/target are gross dollars for ONE contract, not account P&L or R units.
        """
        trades = []
        for row in records:
            values = dict(row)
            for field in ("entry_at", "exit_at"):
                if isinstance(values[field], str):
                    values[field] = datetime.fromisoformat(values[field])
            if isinstance(values["session"], str):
                values["session"] = date.fromisoformat(values["session"])
            won = values["won"]
            if isinstance(won, str) and won.lower() in ("true", "false"):
                won = won.lower() == "true"
            trades.append(BracketTrade(
                values["entry_at"], values["exit_at"], values["session"],
                float(values["stop_loss"]), float(values["take_profit"]), won,
            ))
        return cls(tuple(trades))

    def split(self, fraction=0.7):
        _number(fraction, "train_fraction")
        if not 0 < fraction < 1:
            raise ValueError("train_fraction must be between zero and one")
        days = self.sessions
        count = int(len(days) * fraction)
        if count == 0 or count == len(days):
            raise ValueError("split needs at least one complete session in each partition")
        boundary = days[count]
        return (BracketHistory(tuple(t for t in self.trades if t.session < boundary)),
                BracketHistory(tuple(t for t in self.trades if t.session >= boundary)))


@dataclass(frozen=True)
class RiskRegime:
    """First matching regime wins. Conditions read only pre-trade account state."""

    name: str
    phase: str
    risk_dollars: float
    in_profit: bool | None = None
    days_to_payout: int | None = None
    after_payout: bool | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or self.phase not in ("eval", "funded"):
            raise ValueError("regime needs a name and eval/funded phase")
        _number(self.risk_dollars, "risk_dollars")
        for name in ("in_profit", "after_payout"):
            if getattr(self, name) is not None and type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool or None")
        if self.days_to_payout is not None and (
            isinstance(self.days_to_payout, bool) or not isinstance(self.days_to_payout, Integral)
            or self.days_to_payout < 0
        ):
            raise ValueError("days_to_payout must be a nonnegative integer or None")


@dataclass(frozen=True)
class DollarPolicy:
    """Ordered named dollar-loss budgets; zero is a deliberate skip, not failure."""

    regimes: tuple[RiskRegime, ...]

    def __post_init__(self):
        object.__setattr__(self, "regimes", tuple(self.regimes))
        if not self.regimes or any(not isinstance(r, RiskRegime) for r in self.regimes):
            raise ValueError("policy needs RiskRegime objects")
        if len({r.name for r in self.regimes}) != len(self.regimes):
            raise ValueError("regime names must be unique")
        for phase in ("eval", "funded"):
            if not any(r.phase == phase and r.in_profit is None and
                       r.days_to_payout is None and r.after_payout is None for r in self.regimes):
                raise ValueError(f"policy needs an unconditional {phase} fallback")

    @classmethod
    def constant(cls, risk_dollars):
        return cls((RiskRegime("evaluation", "eval", risk_dollars),
                    RiskRegime("funded", "funded", risk_dollars)))

    def select(self, *, phase, in_profit, days_to_payout, after_payout):
        for r in self.regimes:
            if (r.phase == phase and (r.in_profit is None or r.in_profit == in_profit)
                    and (r.days_to_payout is None or r.days_to_payout == days_to_payout)
                    and (r.after_payout is None or r.after_payout == after_payout)):
                return r
        raise ValueError("no matching policy regime")


@dataclass(frozen=True)
class LifecycleSpec:
    """Firm-neutral lifecycle additions to the existing Account rule tree."""

    account: Account
    eval_contract_limit: int
    funded_tiers: tuple[tuple[float, int], ...]
    reset_fee: float
    request_lock_floor: float | None = None
    session_timezone: str = "America/New_York"
    session_close: time = time(16, 45)
    session_open: time = time(18)
    session_weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)
    inactivity_days: int | None = None
    activity_threshold: float = 1.0
    reset_valid_days: int | None = None
    assumptions: tuple[str, ...] = ()
    inactivity_close: time | None = None
    flatten_at_close: bool = True

    def __post_init__(self):
        object.__setattr__(self, "funded_tiers", tuple(tuple(t) for t in self.funded_tiers))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "session_weekdays", tuple(self.session_weekdays))
        if not self.session_weekdays or any(type(d) is not int or d not in range(7)
                                            for d in self.session_weekdays):
            raise ValueError("session_weekdays must contain weekday numbers 0..6")
        _number(self.reset_fee, "reset_fee")
        _number(self.activity_threshold, "activity_threshold")
        if type(self.flatten_at_close) is not bool:
            raise ValueError("flatten_at_close must be bool")
        if self.inactivity_close is not None and (not isinstance(self.inactivity_close, time)
                                                  or self.inactivity_close.tzinfo is not None):
            raise ValueError("inactivity_close must be a local time without timezone")
        for name in ("inactivity_days", "reset_valid_days"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.request_lock_floor is not None:
            _number(self.request_lock_floor, "request_lock_floor")
        limits = [self.eval_contract_limit] + [q for _, q in self.funded_tiers]
        if any(isinstance(q, bool) or not isinstance(q, Integral) or q < 1 for q in limits):
            raise ValueError("contract limits must be positive integers")
        if not self.funded_tiers or self.funded_tiers[0][0] != float("-inf"):
            raise ValueError("funded tiers must explicitly cover negative retained profit")
        thresholds = [p for p, _ in self.funded_tiers]
        if any(not isfinite(p) for p in thresholds[1:]) or any(
            b <= a for a, b in zip(thresholds, thresholds[1:])
        ):
            raise ValueError("tier thresholds must strictly increase")

    def funded_limit(self, profit):
        return next(q for threshold, q in reversed(self.funded_tiers) if profit >= threshold)


@dataclass(frozen=True)
class BacktestConfig:
    """User-supplied execution/processing scenario; delays are elapsed calendar time."""

    cost_per_contract: float
    approval_delay: timedelta
    receipt_delay: timedelta
    activation_delay: timedelta
    cost_per_trade: float = 0.0
    payment_fee: float = 0.0
    retry_delay: timedelta = timedelta(0)
    initial_wallet: float | None = None

    def __post_init__(self):
        for name in ("cost_per_contract", "cost_per_trade", "payment_fee"):
            _number(getattr(self, name), name)
        if self.initial_wallet is not None:
            _number(self.initial_wallet, "initial_wallet")
        for name in ("approval_delay", "receipt_delay", "activation_delay", "retry_delay"):
            value = getattr(self, name)
            if not isinstance(value, timedelta) or value < timedelta(0):
                raise ValueError(f"{name} must be a nonnegative timedelta")
