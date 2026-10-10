"""State-dependent dollar brackets on historical minute bars.

This explicitly selected OHLC approximation reuses the chronological account
engine. It does not infer the order of a bar's high and low or claim tick fills.
Data loading, calendars and entry signals belong outside the engine package.
"""
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
from hashlib import sha256
from math import ceil, floor, isfinite

import numpy as np

from .backtest import _Replay, _ResolvedExecution, BacktestResult
from .execution import BacktestConfig, BracketHistory, BracketTrade, DollarPolicy, LifecycleSpec
from .instruments import Instrument


@dataclass(frozen=True)
class PriceSession:
    """UTC bar-open nanoseconds and raw OHLC, starting at the intended entry.

    Missing minutes mean no reported trades, not forward-filled quotes. The
    producer must screen known degraded data and provide a real final bar.
    side=1 buys and side=-1 sells short at entry; direction is fixed per session.
    """
    session: date
    close_at: datetime
    timestamps: np.ndarray
    ohlc: np.ndarray
    warmup_timestamps: np.ndarray | None = None
    warmup_ohlc: np.ndarray | None = None
    side: int = 1

    def __post_init__(self):
        if type(self.side) is not int or self.side not in (-1, 1):
            raise ValueError("price session side must be +/-1")
        t = np.array(self.timestamps, dtype=np.int64, copy=True)
        p = np.array(self.ohlc, dtype=float, copy=True)
        if (not isinstance(self.session, date) or isinstance(self.session, datetime)
                or self.close_at.tzinfo is None or len(t) == 0 or t.ndim != 1
                or p.shape != (len(t), 4) or not np.isfinite(p).all()
                or np.any(np.diff(t) <= 0) or np.any(t % 60_000_000_000)
                or t[-1] + 60_000_000_000 != round(self.close_at.timestamp() * 1e9)):
            raise ValueError("session needs ordered minute bars through its aware close")
        if (np.any(p[:, 1] < p.max(axis=1)) or np.any(p[:, 2] > p.min(axis=1))):
            raise ValueError("invalid OHLC extrema")
        t.flags.writeable = p.flags.writeable = False
        object.__setattr__(self, "timestamps", t)
        object.__setattr__(self, "ohlc", p)
        if (self.warmup_timestamps is None) != (self.warmup_ohlc is None):
            raise ValueError("warmup requires both timestamps and OHLC")
        if self.warmup_ohlc is not None:
            wt = np.array(self.warmup_timestamps, dtype=np.int64, copy=True)
            wp = np.array(self.warmup_ohlc, dtype=float, copy=True)
            if (wt.ndim != 1 or len(wt) == 0 or wp.shape != (len(wt), 4)
                    or not np.isfinite(wp).all() or np.any(np.diff(wt) <= 0)
                    or np.any(wt % 60_000_000_000) or wt[-1] + 60_000_000_000 > t[0]
                    or np.any(wp[:, 1] < wp.max(axis=1)) or np.any(wp[:, 2] > wp.min(axis=1))):
                raise ValueError("warmup must contain valid completed bars strictly before entry")
            wt.flags.writeable = wp.flags.writeable = False
            object.__setattr__(self, "warmup_timestamps", wt)
            object.__setattr__(self, "warmup_ohlc", wp)

    @property
    def entry_at(self):
        return datetime.fromtimestamp(int(self.timestamps[0]) / 1e9, timezone.utc)


@dataclass(frozen=True)
class PriceDecision:
    session: date
    attempt: int
    phase: str
    regime: str
    quantity: int
    entry_at: datetime
    exit_at: datetime
    entry_price: float
    stop_price: float
    target_price: float
    exit_price: float
    planned_net_risk: float
    planned_net_target: float
    net_pnl: float
    reason: str
    collision: bool
    entry_slippage_ticks: int = 0
    exit_slippage_ticks: int = 0
    stop_allowance_ticks: int = 0
    execution_multiplier_entry: float = 1.
    execution_stressed: bool = False
    side: int = 1

    @property
    def signed_quantity(self):
        return self.side * self.quantity


@dataclass(frozen=True)
class PriceReplay:
    replay: BacktestResult
    decisions: tuple[PriceDecision, ...]

    @property
    def uncertainty(self):
        return self.replay.uncertainty


def _resolve(session, instrument, quantity, risk, target, costs, *, slippage=None,
             tape=None, compensate_slippage=True, diagnostics=None):
    """Round loss inward and profit outward; stop wins an ambiguous bar."""
    tick = Fraction(str(instrument.tick_size))
    value = Fraction(str(instrument.point_value))
    tick_cash = tick * value * quantity
    allowance = slippage.allowance(float(tape.multipliers[0])) if slippage and compensate_slippage else 0
    stop_ticks = floor((risk - costs) / tick_cash) - allowance
    if stop_ticks < 1:
        return None
    target_ticks = max(1, ceil((target + costs) / tick_cash))
    prices = session.ohlc
    side = session.side
    # Work in side-adjusted prices so both directions share execution arithmetic.
    entry_slip = tape.ticks(slippage, 0, 0) if slippage else 0
    entry = (ceil(side * Fraction(str(prices[0, 0])) / tick) + entry_slip) * tick
    stop, take = entry - stop_ticks * tick, entry + target_ticks * tick
    adverse, favorable = (prices[:, 2], prices[:, 1]) if side == 1 else (prices[:, 1], prices[:, 2])
    through = slippage.target_trade_through_ticks if slippage else 0
    if side == 1:
        sl, tp = adverse <= float(stop) + 1e-10, favorable >= float(take + through * tick) - 1e-10
    else:
        sl, tp = adverse >= -float(stop) - 1e-10, favorable <= -float(take + through * tick) + 1e-10
    hits = np.flatnonzero(sl | tp)
    index = int(hits[0]) if len(hits) else len(prices) - 1
    opening = side * Fraction(str(prices[index, 0]))
    collision = bool(len(hits) and sl[index] and tp[index])
    if len(hits):
        if opening <= stop:
            exit_price, reason = floor(opening / tick) * tick, "stop_gap"
        elif sl[index]:
            exit_price, reason = stop, "stop"
        else:
            exit_price, reason = take, "target"
    else:
        exit_price = floor(side * Fraction(str(prices[index, 3])) / tick) * tick
        reason = "session_close"
    exit_slip = 0 if not slippage or reason == "target" else tape.ticks(
        slippage, index, 2 if reason == "session_close" else 1)
    exit_price -= exit_slip * tick
    pnl = (exit_price - entry) * value
    # Entry slippage is already in the fill price, not a second cash charge.
    prior_low = min(float(entry), side * float(prices[0, 0]))
    if index:
        extreme = adverse[:index].min() if side == 1 else adverse[:index].max()
        prior_low = min(prior_low, side * float(extreme))
    # Never include a low that may occur AFTER a target fill in the exit bar.
    low_price = min(prior_low, float(exit_price))
    if reason == "session_close":
        low_price = min(low_price, side * float(adverse[index]))
    low = min(Fraction(0), (Fraction(str(low_price)) - entry) * value, pnl)
    at = datetime.fromtimestamp(int(session.timestamps[index]) / 1e9, timezone.utc)
    # Intrabar times are unknowable; assign bar-end, not an invented tick time.
    exit_at = at + timedelta(minutes=1)
    if diagnostics is not None:
        diagnostics.update(entry_slippage_ticks=entry_slip, exit_slippage_ticks=exit_slip,
                           stop_allowance_ticks=allowance,
                           execution_multiplier_entry=float(tape.multipliers[0]) if slippage else 1.,
                           execution_stressed=tape.stressed if slippage else False)
    return (_ResolvedExecution(exit_at, quantity, float((stop_ticks + allowance) * tick * value), float(pnl), float(low)),
            float(side * entry), float(side * stop), float(side * take), float(side * exit_price),
            float((stop_ticks + allowance) * tick_cash + costs), float(target_ticks * tick_cash - costs),
            float(pnl * quantity - costs), reason, collision)


def _price_inputs(sessions, instrument, quantity, execution_seed, execution_path, compensate_slippage):
    if type(quantity) is not int or quantity < 1:
        raise ValueError("quantity must be a positive integer")
    for name, value in (("execution_seed", execution_seed), ("execution_path", execution_path)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if type(compensate_slippage) is not bool:
        raise ValueError("compensate_slippage must be bool")
    if not isinstance(instrument, Instrument):
        raise TypeError("instrument must be Instrument")
    sessions = tuple(sessions)
    if not sessions or any(not isinstance(s, PriceSession) for s in sessions):
        raise ValueError("supply at least one PriceSession")
    if any(b.session <= a.session for a, b in zip(sessions, sessions[1:])):
        raise ValueError("price sessions must be unique and chronological")
    return sessions


def replay_prices(spec, sessions, policy, targets, instrument, config, *, quantity=1,
                  collision_policy, slippage=None, execution_seed=0, execution_path=0,
                  compensate_slippage=True, _execution_tapes=None):
    """Replay fixed quantity, state-dependent dollar SL/TP on supplied prices.

    ``targets`` maps every sizing-regime name to its desired NET dollar profit.
    Each session supplies its own side (+1 long, -1 short).
    Explicit ``collision_policy='stop_first'`` opts into the OHLC approximation.
    Costs are per actual contract, not per position. Sessions must be complete,
    chronological and independent of policy; account state is never precomputed.
    """
    if collision_policy != "stop_first":
        raise ValueError("OHLC replay requires explicit collision_policy='stop_first'")
    if not isinstance(spec, LifecycleSpec) or not isinstance(config, BacktestConfig):
        raise TypeError("price replay requires LifecycleSpec and BacktestConfig")
    if not isinstance(policy, DollarPolicy):
        raise TypeError("price replay requires DollarPolicy")
    sessions = _price_inputs(sessions, instrument, quantity, execution_seed, execution_path, compensate_slippage)
    if set(targets) != {r.name for r in policy.regimes} or any(
            isinstance(v, bool) or not isfinite(v) or v <= 0 for v in targets.values()):
        raise ValueError("positive net targets must match every policy regime")
    target_cash = {k: Fraction(str(v)) for k, v in targets.items()}
    tapes = None
    if slippage is not None:
        from .slippage import prepare_execution, SlippageModel
        if not isinstance(slippage, SlippageModel):
            raise TypeError("slippage must be SlippageModel")
        tapes = (_execution_tapes if _execution_tapes is not None else
                 prepare_execution(sessions, instrument, slippage, seed=execution_seed, path=execution_path))
        if len(tapes) != len(sessions):
            raise ValueError("execution tapes must match sessions")
    elif _execution_tapes is not None:
        raise ValueError("execution tapes require a slippage model")
    history = BracketHistory(tuple(BracketTrade(s.entry_at, s.close_at, s.session, 1, 1, True)
                                   for s in sessions))
    decisions = []

    def factory(template, regime, runner, index):
        if quantity > runner.limit:
            raise ValueError("fixed quantity exceeds the firm's current contract limit")
        costs = runner.fixed_cost + quantity * runner.contract_cost
        risk = min(runner.risk_budgets[regime.name], runner.sim.equity - runner.sim.dd_floor)
        details = {}
        tape = tapes[index] if tapes is not None else None
        resolved = _resolve(sessions[index], instrument, quantity, risk, target_cash[regime.name], costs,
                            slippage=slippage, tape=tape, compensate_slippage=compensate_slippage,
                            diagnostics=details)
        if resolved is None:
            allowance = slippage.allowance(float(tape.multipliers[0])) if slippage and compensate_slippage else 0
            minimum = costs + (1 + allowance) * quantity * Fraction(str(instrument.tick_size)) * Fraction(str(instrument.point_value))
            if runner.sim.equity - runner.sim.dd_floor >= minimum:
                return None  # policy cannot afford this fixed size; account still can
            return _ResolvedExecution(template.exit_at, 0, 0, 0, 0)
        fill, entry, stop, take, exit_price, planned_risk, planned_target, net, reason, collision = resolved
        decisions.append(PriceDecision(template.session, runner.attempts, runner.role, regime.name,
                                       quantity, template.entry_at, fill.exit_at, entry, stop, take,
                                       exit_price, planned_risk, planned_target, net, reason, collision,
                                       side=sessions[index].side, **details))
        return fill

    result = _Replay(spec, history, policy, config, execution_factory=factory,
                     session_closes={s.session: s.close_at for s in sessions}).run()
    digest = sha256(repr((instrument, quantity, sorted(targets.items()), collision_policy)).encode())
    if slippage is not None:
        digest.update(repr((slippage, execution_seed, execution_path, compensate_slippage)).encode())
        for tape in tapes:
            digest.update(tape.multipliers.tobytes())
            digest.update(tape.uniforms.tobytes())
            digest.update(bytes([tape.stressed]))
    for s in sessions:
        digest.update(repr((s.session, s.close_at, s.side)).encode())
        digest.update(s.timestamps.tobytes())
        digest.update(s.ohlc.tobytes())
    assumptions = tuple(a for a in result.assumptions if a not in (
        "sequential ideal stop/target fills; no gaps or slippage",
        "gross per-contract historical stop/target outcomes are held fixed",
    )) + (
        "historical minute-OHLC approximation: any first-touch bar spanning both levels exits at stop",
        "fixed contract count; net dollar brackets recomputed from pre-entry account state",
        "long/short direction supplied per session; buy fills round up and sell fills down",
        "loss distance rounded down and target distance up to ticks",
        ("stop gaps fill at bar open; target gaps receive only the limit price; no extra slippage" if slippage is None else
         "uncalibrated tick execution scenario; gap prices plus incremental slippage; no duplicate spread charge"),
        "intrabar exits timestamped at bar end; exit-bar post-fill excursions are not inferred",
        "forced session-close exits retain realized P&L, not a manufactured stop/target outcome",
        "buffer too small for fixed quantity plus one stop tick is treated as CAPPED_OUT",
    )
    if slippage is not None:
        assumptions += (
            "lagged completed-bar volatility; exogenous session stress shared across fills; no current-bar lookahead",
            "net brackets anchored to actual entry fill; limit targets receive no adverse price slippage",
            "stop allowance is an entry-time mixture quantile, not a guaranteed realized-loss cap",
            f"slippage compensation={compensate_slippage}; target trade-through={slippage.target_trade_through_ticks} ticks",
            f"execution scenario: {slippage.label}; not measured platform behavior",
        )
    return PriceReplay(replace(result, assumptions=assumptions, history_fingerprint=digest.hexdigest()),
                       tuple(decisions))
