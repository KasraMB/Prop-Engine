"""Full-engine bootstrap paths and a separately labelled IID cash-cycle approximation.

The continuation model is deliberately not a substitute for the dated lifecycle.
Its exponential tail bound applies to its finite empirical cycle law only.
"""
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from hashlib import sha256
from itertools import groupby
from math import ceil, isfinite, log
from numbers import Integral, Real
from zoneinfo import ZoneInfo

import numpy as np

from .backtest import backtest
from .execution import BracketHistory
from .resampling import StationaryDayBootstrap
from .risk import RiskConfig, cash_risk_path, risk_report


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class RuinConfig:
    paths: int = 100
    sessions: int = 250
    mean_block: float = 5.0
    seed: int = 1729
    cycle_paths: int = 10000
    cycle_steps: int = 2000
    tail_tolerance: float = .0001

    def __post_init__(self):
        for key in ("paths", "sessions", "cycle_paths", "cycle_steps"):
            _integer(getattr(self, key), key, 2 if key in ("paths", "cycle_paths") else 1)
        _integer(self.seed, "seed", 0)
        StationaryDayBootstrap(self.mean_block)
        if (isinstance(self.tail_tolerance, bool) or not isinstance(self.tail_tolerance, Real)
                or not isfinite(self.tail_tolerance) or not 0 < self.tail_tolerance < 1):
            raise ValueError("tail_tolerance must be in (0,1)")


@dataclass(frozen=True)
class CashCycle:
    """One settled cycle: terminal cash increment and largest interim cash deficit."""
    net_cash: float
    required_cash: float

    def __post_init__(self):
        for key in ("net_cash", "required_cash"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"{key} must be finite")
        if self.required_cash < max(0, -self.net_cash):
            raise ValueError("required_cash must cover the terminal deficit")


def _log_mgf(losses, rate):
    terms = rate * losses
    maximum = float(terms.max())
    return maximum + log(float(np.exp(terms-maximum).mean()))


def _adjustment_rate(increments):
    """Return a conservative positive r with mean(exp(-r X)) <= 1."""
    losses = -increments
    lo, hi = 0.0, 1 / max(abs(increments))
    while _log_mgf(losses, hi) < 0:
        hi *= 2
    for _ in range(100):
        middle = (lo+hi)/2
        if _log_mgf(losses, middle) <= -1e-14:
            lo = middle
        else:
            hi = middle
    return lo


def _bernoulli_bounds(count, n, delta):
    """Conservative two-sided Chernoff/KL bounds (total error <= delta)."""
    observed, threshold = count/n, log(2/delta)/n

    def divergence(p):
        return ((observed * log(observed/p) if observed else 0)
                + ((1-observed)*log((1-observed)/(1-p)) if observed < 1 else 0))

    bounds = []
    for left in (True, False):
        if (left and count == 0) or (not left and count == n):
            bounds.append(0.0 if left else 1.0)
            continue
        lo, hi = (0.0, observed) if left else (observed, 1.0)
        for _ in range(65):
            mid = (lo+hi)/2
            if mid in (0, 1):
                break
            outside = divergence(mid) > threshold
            if outside == left:
                lo = mid
            else:
                hi = mid
        bounds.append(lo if left else hi)
    return bounds


def ultimate_cycle_ruin(cycles, *, bankroll, target=.01, confidence=.95,
                        paths=10000, max_cycles=2000, tail_tolerance=.0001, seed=1729):
    """Ultimate ruin conditional on a uniform IID law over supplied settled cycles.

    Within a cycle, B < required_cash ruins; equality survives. Otherwise B += X.
    Nonpositive drift with a negative increment gives certain eventual ruin.
    Positive drift admits the supermartingale exp(-r sum X). From B >= Dmax,
    future ruin <= exp(-r(B-Dmax)). This bounds unfinished survivors, not just
    sampling error. No claim covers cycle-law estimation or model misspecification.
    """
    cycles = tuple(cycles)
    if not cycles or any(not isinstance(c, CashCycle) for c in cycles):
        raise ValueError("supply at least one CashCycle")
    RiskConfig(bankroll=bankroll, target_ruin_probability=target, confidence=confidence)
    if bankroll is None:
        raise ValueError("ultimate ruin requires a finite bankroll")
    RuinConfig(paths=2, sessions=1, cycle_paths=paths, cycle_steps=max_cycles,
               tail_tolerance=tail_tolerance, seed=seed)
    # Integer cents prevent affordable-fee equality changing under repeated addition.
    for c in cycles:
        if any(Fraction(str(v))*100 % 1 for v in (c.net_cash, c.required_cash)):
            raise ValueError("cash cycles must use whole cents")
    amounts = [int(Fraction(str(c.net_cash))*100) for c in cycles]
    needs = [int(Fraction(str(c.required_cash))*100) for c in cycles]
    b = int(Fraction(str(bankroll))*100)
    if max(abs(v) for v in amounts) * max_cycles + b > 2**62 or max(needs) > 2**62:
        raise ValueError("cycle simulation exceeds supported money range")
    x, d = np.array(amounts, dtype=np.int64), np.array(needs, dtype=np.int64)
    mean = sum(map(int, x))/len(x)
    dmax = int(max(d))
    base = {"model": "IID settled cash-cycle approximation", "cycles": len(cycles),
            "bankroll": bankroll, "target": target, "confidence": confidence,
            "mean_cycle_cash": mean/100, "maximum_cycle_cash_need": dmax/100,
            "payout_retention": 1.0, "seed": seed, "paths": paths, "max_cycles": max_cycles,
            "tail_tolerance": tail_tolerance,
            "scope": "Conditional on the empirical cycle law; uncertainty excludes estimated-law error, serial dependence and receipt overlap."}
    certain = (b < int(min(d)) or (mean <= 0 and min(x) < 0)
               or (max(x) == 0 and b < dmax))
    safe = min(x) >= 0 and b >= dmax
    rate = _adjustment_rate(x.astype(float)) if mean > 0 and min(x) < 0 else None
    sufficient = (0.0 if target == 1 else None if target == 0 and min(x) < 0 else
                  dmax/100 if min(x) >= 0 else
                  ceil((dmax-log(target)/rate))/100 if rate else None)
    base.update(adjustment_rate_per_dollar=rate*100 if rate else None,
                sufficient_bankroll=sufficient,
                bankroll_method="Sufficient upper bound, not an estimated minimum; conditional on the cycle law.")
    if certain or safe:
        value = float(certain)
        return {**base, "status": "certain_ruin" if certain else "safe_under_cycle_law",
                "probability_bounds": [value, value], "confidence_bounds": [value, value],
                "ruined_paths": paths if certain else 0, "unresolved_paths": 0,
                "tail_bound": 0.0, "simulated_cycles": 0}
    escape = dmax if min(x) >= 0 else (dmax-log(tail_tolerance)/rate if rate else float("inf"))
    wealth = np.full(paths, b, dtype=np.int64)
    active = np.ones(paths, dtype=bool)
    ruined = np.zeros(paths, dtype=bool)
    rng = np.random.default_rng(seed)
    visits = 0
    for _ in range(max_cycles):
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        draws = rng.integers(len(cycles), size=len(indices))
        failed = wealth[indices] < d[draws]
        ruined[indices[failed]] = True
        active[indices[failed]] = False
        survivors = indices[~failed]
        wealth[survivors] += x[draws[~failed]]
        active[survivors[wealth[survivors] >= escape]] = False
        visits += len(indices)
    failures, unresolved = int(ruined.sum()), int(active.sum())
    escaped = paths-failures-unresolved
    tail = tail_tolerance if min(x) < 0 else 0.0
    lower = failures/paths
    upper = (failures+unresolved+escaped*tail)/paths
    # Union bound for the lower-event and upper-event Bernoulli intervals.
    lower_ci = _bernoulli_bounds(failures, paths, (1-confidence)/2)[0]
    upper_ci = _bernoulli_bounds(failures+unresolved, paths, (1-confidence)/2)[1]
    return {**base, "status": "bounded" if unresolved == 0 else "simulation_budget_reached",
            "probability_bounds": [lower, upper],
            "confidence_bounds": [lower_ci, min(1, upper_ci+(1-upper_ci)*tail)],
            "ruined_paths": failures, "unresolved_paths": unresolved,
            "tail_bound": escaped*tail/paths, "simulated_cycles": visits}


def _resampled_history(spec, history, indices):
    """Retain whole sessions and wall-clock trade order on a new weekday calendar."""
    groups = [tuple(g) for _, g in groupby(history.trades, lambda t: t.session)]
    zone, day, trades = ZoneInfo(spec.session_timezone), history.sessions[0], []
    for index in indices:
        while day.weekday() not in spec.session_weekdays:
            day += timedelta(days=1)
        for trade in groups[int(index)]:
            def move(stamp):
                local = stamp.astimezone(zone)
                shifted = datetime.combine(day+(local.date()-trade.session), local.time(), zone)
                if shifted.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) != shifted.replace(tzinfo=None):
                    raise ValueError("resampled trade clock falls in a nonexistent daylight-saving time")
                return shifted
            trades.append(replace(trade, entry_at=move(trade.entry_at), exit_at=move(trade.exit_at), session=day))
        day += timedelta(days=1)
    return BracketHistory(tuple(trades))


def _settled_cycles(result):
    """Only terminated accounts whose requested payouts have actually arrived."""
    groups = {}
    for event in result.events:
        if event.attempt > 0:
            groups.setdefault(event.attempt, []).append(event)
    cycles, excluded = [], 0
    for events in groups.values():
        complete = any(e.kind in ("failure", "live_handoff") for e in events)
        settled = sum(e.kind == "request" for e in events) == sum(e.kind == "receipt" for e in events)
        if not complete or not settled:
            excluded += 1
            continue
        cash = deficit = Fraction(0)
        for event in events:
            cash += Fraction(str(event.cash))
            deficit = max(deficit, -cash)
        # Round adversely if fractional-cent amounts exist in a custom profile.
        net_cents = cash*100 // 1
        cycles.append(CashCycle(float(net_cents/100), ceil(deficit*100)/100))
    return cycles, excluded


def bootstrap_ruin(spec, history, policy, config, *, simulation=RuinConfig(), risk=None, progress=None):
    """Independent bootstrap paths through the canonical dated lifecycle.

    A fixed policy is never optimized here. Each path starts fresh; within a path
    account state, payout queue and retained external cash persist across attempts.
    Unlimited funding reveals each capital threshold without truncation bias.
    """
    if not isinstance(simulation, RuinConfig):
        raise TypeError("simulation must be RuinConfig")
    risk = risk if risk is not None else RiskConfig(bankroll=config.initial_wallet)
    unlimited = replace(config, initial_wallet=None)
    seeds = np.random.SeedSequence(simulation.seed).spawn(simulation.paths)
    checkpoints = sorted(set([1, simulation.sessions] + [max(1, round(simulation.sessions*q)) for q in (.1,.25,.5,.75)]))
    counts = np.zeros(len(checkpoints), dtype=int)
    paths, cycles, excluded = [], [], 0
    for i, seed in enumerate(seeds):
        indices = StationaryDayBootstrap(simulation.mean_block).generate(
            len(history.sessions), simulation.sessions, 1, int(seed.generate_state(1)[0]))[0]
        generated = _resampled_history(spec, history, indices)
        result = backtest(spec, generated, policy, unlimited)
        capital = cash_risk_path(result)
        if config.initial_wallet is None:
            paths.append(capital)
        else:
            # Without a funding shortfall the wallet cannot affect this policy.
            performance = (replace(result, config=config)
                           if config.initial_wallet >= capital.required_bankroll else
                           backtest(spec, generated, policy, config))
            paths.append(cash_risk_path(performance, unrestricted=result))
        extracted, missing = _settled_cycles(result)
        cycles.extend(extracted)
        excluded += missing
        if risk.bankroll is not None:
            cash, first = Fraction(str(risk.bankroll)), None
            for event in result.events:
                cash += Fraction(str(event.cash))
                if cash < 0:
                    first = event.at
                    break
            if first is not None:
                for j, checkpoint in enumerate(checkpoints):
                    close = datetime.combine(generated.sessions[checkpoint-1], spec.session_close,
                                             ZoneInfo(spec.session_timezone))
                    counts[j] += first <= close
        if progress:
            progress({"stage": "ruin", "completed_paths": i+1, "paths": simulation.paths})
    report = risk_report(paths, options=risk, sample_kind="independent_model")
    curve = [{"sessions": t, "ruined_paths": int(n),
              "probability": int(n)/simulation.paths if risk.bankroll is not None else None}
             for t, n in zip(checkpoints, counts)]
    cycle_report = None
    if cycles and risk.bankroll is not None:
        cycle_report = ultimate_cycle_ruin(cycles, bankroll=risk.bankroll,
            target=risk.target_ruin_probability, confidence=risk.confidence,
            paths=simulation.cycle_paths, max_cycles=simulation.cycle_steps,
            tail_tolerance=simulation.tail_tolerance, seed=simulation.seed)
    return {"settings": asdict(simulation), "risk": report, "horizon_curve": curve,
            "source_sessions": len(history.sessions), "source_trades": len(history.trades),
            "source_fingerprint": sha256(repr(history.trades).encode()).hexdigest(),
            "payout_retention": 1.0, "cycle_approximation": cycle_report,
            "cycle_records": [asdict(c) for c in cycles], "excluded_unsettled_or_open_accounts": excluded,
            "ultimate_full_engine": {"status": "not_identified",
                "finite_horizon_estimate": report["ruin_probability"], "upper_bound": 1.0,
                "note": "Finite simulation estimates a lower bound on ultimate ruin; survivors remain unresolved. Not a confidence interval."},
            "assumptions": [
                "Full payout retention; ruin is first inability to fund a required fee, even if a later receipt could rescue the wallet.",
                "Independent stationary whole-session bootstrap paths; circular blocks preserve within-block ordering, not all historical dependence.",
                "Sessions are placed on the firm's open-weekday calendar, retaining local trade clocks; original gaps and exchange holidays are not retained.",
                "Synthetic inputs are resampled from their realized history, not regenerated from known parameters.",
                "Cash distributions use the configured wallet; capital requirements and cycle calibration use matching unrestricted-wallet paths.",
                "Ultimate approximation resamples settled complete account cash cycles independently, ignoring cross-account receipt overlap, fee-state dependence and serial dependence.",
                "Open/unsettled accounts are excluded from the cycle law; horizon selection can bias it. Conditional model confidence does not cover this bias or missing future outcomes.",
            ]}
