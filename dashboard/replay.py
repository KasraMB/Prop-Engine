"""JSON adapter for the chronological engine; shared by local and browser UIs.

No accounting or selection logic lives here. Closed-summary inputs are not
converted into bracket histories, and arbitrary Python is never evaluated.
"""
from dataclasses import asdict, is_dataclass, replace
from datetime import date, datetime, time, timedelta
from hashlib import sha256
from io import StringIO
import csv
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from propfirm_engine import BacktestConfig, BracketHistory, DollarPolicy, Engine, RiskRegime, RollingConfig
from propfirm_engine.rolling import window_slices
from propfirm_engine.risk import RiskConfig, cash_risk_path, risk_report
from propfirm_engine.ruin import RuinConfig
from propfirm_engine.firms.lucidflex import replay_50k
from propfirm_engine.synthetic import IIDGenerator, RegimeSwitchingGenerator, StochasticVolGenerator
from zoneinfo import ZoneInfo
import numpy as np

MAX_CSV_BYTES = 5_000_000
MAX_TRADES = 20_000
OBJECTIVES = {
    "net_cash_per_day": lambda result: result.net_cash_per_day,
    "net_cash": lambda result: result.net_cash,
}
CSV_COLUMNS = ("entry_at", "exit_at", "session", "stop_loss", "take_profit", "won")
NY = ZoneInfo("America/New_York")


def _finite(value, name, minimum=0, maximum=math.inf, *, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not minimum <= value <= maximum
            or (integer and type(value) is not int)):
        raise ValueError(f"{name} must be {'an integer' if integer else 'finite'} in [{minimum}, {maximum}]")
    return value


def _history_payload(records, provenance):
    history = BracketHistory.from_records(records)
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for t in history.trades:
        writer.writerow(dict(entry_at=t.entry_at.isoformat(), exit_at=t.exit_at.isoformat(),
                             session=t.session.isoformat(), stop_loss=t.stop_loss,
                             take_profit=t.take_profit, won=str(t.won).lower()))
    wins = sum(t.won for t in history.trades)
    return {"csv": stream.getvalue(), "provenance": provenance,
            "stats": {**_partition(history), "wins": wins, "losses": len(history.trades) - wins,
                      "win_rate": wins / len(history.trades),
                      "mean_rr": sum(t.take_profit / t.stop_loss for t in history.trades) / len(history.trades),
                      "mean_stop": sum(t.stop_loss for t in history.trades) / len(history.trades)}}


def generate(params):
    """Adapt the existing synthetic generators to an explicit bracket history.

    The generated scale is known: return is +rr*scale or -scale. This mapping
    is valid only for these synthetic models, not an inference from real P&L.
    Stochastic volatility scales both stop and target, preserving their ratio.
    """
    required = {"generator", "win_rate", "rr", "stop_loss", "trades_per_day", "sessions", "seed", "start_date"}
    kinds = {"iid": (IIDGenerator, set()),
             "regime": (RegimeSwitchingGenerator, {"spread", "persistence"}),
             "stochvol": (StochasticVolGenerator, {"vol_phi", "vol_sigma"})}
    if not isinstance(params, dict) or params.get("generator") not in kinds:
        raise ValueError("generator must be iid, regime or stochvol")
    factory, extras = kinds[params["generator"]]
    if set(params) != required | extras:
        raise ValueError("Generator parameters must match the selected model")
    for key, lower, upper, integer in (
        ("win_rate", 0, 1, False), ("rr", 1e-9, 1e9, False),
        ("stop_loss", 1e-9, 1e9, False), ("trades_per_day", 1, 100, True),
        ("sessions", 1, MAX_TRADES, True), ("seed", 0, 2**32 - 1, True),
    ):
        _finite(params[key], key, lower, upper, integer=integer)
    if params["sessions"] * params["trades_per_day"] > MAX_TRADES:
        raise ValueError("Generated history exceeds 20,000 trades")
    for key in extras:
        _finite(params[key], key, 0, 1 if key in ("persistence", "vol_phi", "spread") else 5)
    first = date.fromisoformat(params["start_date"])
    generator = factory(win_rate=params["win_rate"], rr=params["rr"],
                        trades_per_day=params["trades_per_day"],
                        start=datetime.combine(first, time(9, 30)),
                        **{key: params[key] for key in extras})
    try:
        with np.errstate(over="raise", invalid="raise"):
            source = generator.generate(params["sessions"], params["seed"])
    except (OverflowError, FloatingPointError) as exc:
        raise ValueError("Generator scale overflow; reduce volatility parameters") from exc
    records = []
    for stamp, value in zip(source.rows["timestamp"], source.rows["return"]):
        scale = value / params["rr"] if value > 0 else -value
        stop = params["stop_loss"] * scale
        exit_at = stamp.replace(tzinfo=NY)
        records.append(dict(entry_at=exit_at - timedelta(minutes=1), exit_at=exit_at,
                            session=exit_at.date(), stop_loss=stop,
                            take_profit=stop * params["rr"], won=value > 0))
    provenance = {"kind": "synthetic", "parameters": dict(params),
                  "model": _jsonable(source.provenance),
                  "calendar": "weekday sessions; no exchange-holiday filter",
                  "execution": "one-minute ideal brackets; volatility scales stop and target"}
    return _history_payload(records, provenance)


def manual(params):
    """Convert hand-entered Eastern wall-clock brackets; execute via run(), never here."""
    if not isinstance(params, dict) or set(params) != {"trades"}:
        raise ValueError("Manual history needs a trades list")
    rows = params["trades"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_TRADES:
        raise ValueError("Manual history needs 1 to 20,000 trades")
    records = []
    for row in rows:
        if set(row) != {"session", "entry_time", "duration_minutes", "stop_loss", "take_profit", "won"}:
            raise ValueError("Manual trade needs session, entry_time, duration_minutes, stop_loss, take_profit, won")
        session = date.fromisoformat(row["session"])
        wall_time = time.fromisoformat(row["entry_time"])
        if wall_time.tzinfo is not None or not time(9, 30) <= wall_time < time(16, 45):
            raise ValueError("Manual editor uses 09:30–16:45 Eastern; use CSV for other session hours")
        duration = _finite(row["duration_minutes"], "duration_minutes", 1, 435)
        for field in ("stop_loss", "take_profit"):
            _finite(row[field], field, 1e-9)
        entry = datetime.combine(session, wall_time, NY)
        exit_at = entry + timedelta(minutes=duration)
        if session.weekday() >= 5 or exit_at.time() > time(16, 45):
            raise ValueError("Manual trade lies outside the session")
        records.append(dict(entry_at=entry, exit_at=exit_at, session=session,
                            stop_loss=row["stop_loss"], take_profit=row["take_profit"], won=row["won"]))
    return _history_payload(records, {"kind": "manual", "timezone": "America/New_York", "trades": rows})


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return {"seconds": value.total_seconds()}
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else None
    return value


def parse_history(text):
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_CSV_BYTES:
        raise ValueError("CSV must be text of at most 5 MB")
    reader = csv.DictReader(StringIO(text.lstrip("\ufeff")), strict=True)
    columns = CSV_COLUMNS
    if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("CSV must have unique column names")
    if set(reader.fieldnames) != set(columns):
        raise ValueError("CSV requires exactly: " + ", ".join(columns))
    rows = []
    for number, row in enumerate(reader, 2):
        if len(rows) >= MAX_TRADES:
            raise ValueError("Dashboard limit is 20,000 trades; use the Python API for larger histories")
        if None in row or any(value is None or not value.strip() for value in row.values()):
            raise ValueError(f"CSV row {number}: missing or extra values")
        rows.append({key: value.strip() for key, value in row.items()})
    return BracketHistory.from_records(rows)


def _summary(result):
    return {**_jsonable(result), "net_cash": result.net_cash,
            "net_cash_per_day": result.net_cash_per_day, "receipts": result.receipts,
            "fees": result.fees, "calendar_days": result.calendar_days,
            "executed_trades": sum(e.kind == "trade" for e in result.events)}


def _partition(history):
    return {"sessions": len(history.sessions), "trades": len(history.trades),
            "first_session": history.sessions[0].isoformat(),
            "last_session": history.sessions[-1].isoformat()}


def _rolling_summary(result):
    return {**_jsonable(result), "summary": result.summary}


def _rolling_work(history, rolling):
    slices = window_slices(history, rolling)
    return sum(end - start for start, end in slices)


def _single_risk(result, history, policy, options):
    unlimited = (Engine().backtest(result.spec, history, policy, replace(result.config, initial_wallet=None))
                 if any(e.kind == "wallet_wait" for e in result.events) else None)
    return _jsonable(risk_report([cash_risk_path(result, unrestricted=unlimited)], options=options,
                                sample_kind="single_history"))


def run(request, progress=None):
    """Replay or fit with explicit assumptions and an OOS-only headline on fit."""
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    mode = request.get("mode")
    if mode not in ("backtest", "fit"):
        raise ValueError("mode must be backtest or fit")
    if request.get("profile") != "lucidflex_50k_dll_off":
        raise ValueError("Only the LucidFlex 50K DLL-off replay profile is currently exposed")
    history = parse_history(request["csv"])
    spec = replay_50k(**request["account"])
    settings = dict(request["config"])
    for name in ("approval_delay", "receipt_delay", "activation_delay", "retry_delay"):
        value = settings.pop(name + "_hours")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(name + "_hours must be finite and nonnegative")
        settings[name] = timedelta(hours=value)
    config = BacktestConfig(**settings)
    risk = RiskConfig(**{"bankroll": config.initial_wallet, **request.get("risk", {})})
    if not isinstance(request["regimes"], list) or len(request["regimes"]) > 32:
        raise ValueError("Supply at most 32 ordered regimes")
    policy = DollarPolicy(tuple(RiskRegime(**row) for row in request["regimes"]))
    objective_name = request.get("objective", "net_cash_per_day")
    if objective_name not in OBJECTIVES:
        raise ValueError("Unsupported dashboard objective; custom callables belong in the Python API")
    engine = Engine()
    ruin = RuinConfig(**request["ruin"]) if request.get("ruin") is not None else None
    if ruin is not None:
        for name, upper in (("paths", 2000), ("sessions", 10000),
                            ("cycle_paths", 100000), ("cycle_steps", 10000)):
            if getattr(ruin, name) > upper:
                raise ValueError(f"Dashboard {name} limit is {upper}; larger studies belong in the Python API")
    rolling = RollingConfig(**request["rolling"]) if request.get("rolling") is not None else None
    output = {"schema_version": 1, "mode": mode, "profile": request["profile"],
              "objective": objective_name, "direction": "maximize", "input": _partition(history),
              "execution_model": "sequential_ideal_brackets",
              "csv_sha256": sha256(request["csv"].encode("utf-8")).hexdigest(),
              "request": {k: v for k, v in request.items() if k != "csv"}}
    if mode == "backtest":
        work = len(history.trades) + (_rolling_work(history, rolling) if rolling else 0)
        output["estimated_trade_visits"] = work
        if progress:
            progress({"stage": "replay", "estimated_trade_visits": work})
        result = engine.backtest(spec, history, policy, config)
        output.update(headline_scope="Full history / fixed policy (not OOS)",
                      headline=_summary(result), policy=_jsonable(policy),
                      score=OBJECTIVES[objective_name](result))
        output["risk"] = _single_risk(result, history, policy, risk)
        if rolling is not None:
            windows = engine.rolling_backtest(spec, history, policy, config, rolling=rolling,
                                              objective=OBJECTIVES[objective_name], risk=risk)
            output.update(rolling={"headline": _rolling_summary(windows)},
                          score=windows.score, selection_basis="mean_window_objective", risk=_jsonable(windows.risk))
        if ruin is not None:
            output["ruin"] = _jsonable(engine.ruin(spec, history, policy, config,
                simulation=ruin, risk=risk, progress=progress))
        return output
    search = dict(request["search"])
    for name, lower, upper in (("generations", 0, 100), ("population", 2, 32), ("seed", 0, 2**32 - 1)):
        value = search[name]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer from {lower} to {upper}")
    # The dashboard fixes 70/30; callers cannot silently override the headline split.
    train, test = history.split(0.70)
    candidates = search["generations"] * search["population"] + 2
    work = len(train.trades) * candidates + len(history.trades) + len(test.trades)
    if rolling is not None:
        work = (_rolling_work(train, rolling) * (candidates + 1)
                + _rolling_work(test, rolling) * 2 + 2 * len(history.trades))
    output["estimated_trade_visits"] = work
    if progress:
        progress({"stage": "search", "estimated_trade_visits": work,
                  "evaluations": 0, "maximum_evaluations": candidates})
    fitted = engine.fit(spec, history, config, policy=policy, risk_bounds=request["risk_bounds"],
                        objective=OBJECTIVES[objective_name], train_fraction=0.70, rolling=rolling,
                        progress=progress, risk=risk, **search)
    baseline = engine.backtest(spec, test, policy, config)
    output.update(headline_scope="Out of sample / final 30% of sessions",
                  headline=_summary(fitted.out_of_sample), score=fitted.score,
                  baseline=_summary(baseline), training=_summary(fitted.in_sample),
                  training_score=fitted.in_sample_score, policy=_jsonable(fitted.policy),
                  split={"train": _partition(train), "test": _partition(test),
                         "fraction": 0.70, "boundary_policy": fitted.boundary_policy},
                  evaluations=fitted.evaluations, seed=fitted.seed)
    output["risk"] = (_jsonable(fitted.out_of_sample_rolling.risk) if rolling else
                      _single_risk(fitted.out_of_sample, test, fitted.policy, risk))
    if rolling is not None:
        baseline_rolling = engine.rolling_backtest(spec, test, policy, config, rolling=rolling,
                                                   objective=OBJECTIVES[objective_name], risk=risk)
        output.update(selection_basis="mean_window_objective", rolling={
            "headline": _rolling_summary(fitted.out_of_sample_rolling),
            "training": _rolling_summary(fitted.in_sample_rolling),
            "baseline": _rolling_summary(baseline_rolling),
        })
    if ruin is not None:
        output["ruin"] = _jsonable(engine.ruin(spec, test, fitted.policy, config,
            simulation=ruin, risk=risk, progress=progress))
    return output
