"""JSON adapter for the chronological engine; shared by local and browser UIs.

No accounting or selection logic lives here. Closed-summary inputs are not
converted into bracket histories, and arbitrary Python is never evaluated.
"""
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time, timedelta
from hashlib import sha256
from io import StringIO
import csv
import math
import os
import sys


from propfirm_engine import BacktestConfig, BracketHistory, DollarPolicy, Engine, RiskRegime
from propfirm_engine.firms.lucidflex import replay_50k

MAX_CSV_BYTES = 5_000_000
MAX_TRADES = 20_000
OBJECTIVES = {
    "net_cash_per_day": lambda result: result.net_cash_per_day,
    "net_cash": lambda result: result.net_cash,
}


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
    columns = ("entry_at", "exit_at", "session", "stop_loss", "take_profit", "won")
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


def run(request):
    """Replay or fit with explicit assumptions and an OOS-only headline on fit."""
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    if request.get("accept_bracket_contract") is not True:
        raise ValueError("Confirm the sequential ideal stop-or-target input contract first")
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
    if not isinstance(request["regimes"], list) or len(request["regimes"]) > 32:
        raise ValueError("Supply at most 32 ordered regimes")
    policy = DollarPolicy(tuple(RiskRegime(**row) for row in request["regimes"]))
    objective_name = request.get("objective", "net_cash_per_day")
    if objective_name not in OBJECTIVES:
        raise ValueError("Unsupported dashboard objective; custom callables belong in the Python API")
    engine = Engine()
    output = {"schema_version": 1, "mode": mode, "profile": request["profile"],
              "objective": objective_name, "direction": "maximize", "input": _partition(history),
              "csv_sha256": sha256(request["csv"].encode("utf-8")).hexdigest(),
              "request": {k: v for k, v in request.items() if k != "csv"}}
    if mode == "backtest":
        result = engine.backtest(spec, history, policy, config)
        output.update(headline_scope="Full history / fixed policy (not OOS)",
                      headline=_summary(result), policy=_jsonable(policy),
                      score=OBJECTIVES[objective_name](result))
        return output
    search = dict(request["search"])
    for name, lower, upper in (("generations", 0, 100), ("population", 2, 32), ("seed", 0, 2**32 - 1)):
        value = search[name]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer from {lower} to {upper}")
    if len(history.trades) * (search["generations"] * search["population"] + 1) > 2_000_000:
        raise ValueError("Search exceeds dashboard work limit; reduce search size or use the Python API")
    # The dashboard fixes 70/30; callers cannot silently override the headline split.
    train, test = history.split(0.70)
    fitted = engine.fit(spec, history, config, policy=policy, risk_bounds=request["risk_bounds"],
                        objective=OBJECTIVES[objective_name], train_fraction=0.70, **search)
    baseline = engine.backtest(spec, test, policy, config)
    output.update(headline_scope="Out of sample / final 30% of sessions",
                  headline=_summary(fitted.out_of_sample), score=fitted.score,
                  baseline=_summary(baseline), training=_summary(fitted.in_sample),
                  training_score=fitted.in_sample_score, policy=_jsonable(fitted.policy),
                  split={"train": _partition(train), "test": _partition(test),
                         "fraction": 0.70, "boundary_policy": fitted.boundary_policy},
                  evaluations=fitted.evaluations, seed=fitted.seed)
    return output
