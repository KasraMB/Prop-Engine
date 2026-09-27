"""Dashboard adapter contracts: actual API parity, holdout isolation and builds."""
from copy import deepcopy
from datetime import date, timedelta
from hashlib import sha256
import json
from pathlib import Path

import pytest

from dashboard import build_pages, replay
from propfirm_engine import BacktestConfig, DollarPolicy, Engine
from propfirm_engine.firms.lucidflex import replay_50k


def request_fixture():
    lines = ["entry_at,exit_at,session,stop_loss,take_profit,won"]
    day = date(2026, 9, 1)
    for i in range(20):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        target = 1500 if i < 2 else 200
        lines.append(f"{day}T10:00:00-04:00,{day}T10:05:00-04:00,{day},100,{target},true")
        day += timedelta(days=1)
    return {"profile": "lucidflex_50k_dll_off", "mode": "backtest", "csv": "\n".join(lines),
            "account": {"eval_fee": 105.2, "reset_fee": 105, "contract_type": "micro"},
            "config": {"cost_per_contract": 0, "cost_per_trade": 0, "payment_fee": 0,
                       "initial_wallet": 2000, "approval_delay_hours": 0,
                       "receipt_delay_hours": 0, "activation_delay_hours": 0.5, "retry_delay_hours": 0},
            "regimes": [{"name": "evaluation", "phase": "eval", "risk_dollars": 100},
                        {"name": "funded", "phase": "funded", "risk_dollars": 100}],
            "risk_bounds": {"evaluation": [50, 200], "funded": [50, 200]},
            "search": {"generations": 1, "population": 4, "seed": 42},
            "objective": "net_cash_per_day"}


def test_adapter_matches_api_and_exports_finite_json():
    req = request_fixture()
    result = replay.run(req)
    expected = Engine().backtest(replay_50k(**req["account"]), replay.parse_history(req["csv"]),
                                DollarPolicy.constant(100), BacktestConfig(0, timedelta(0),
                                timedelta(0), timedelta(minutes=30), initial_wallet=2000))
    assert result["headline"]["net_cash"] == expected.net_cash
    assert result["headline"]["receipts"] == expected.receipts > 0
    assert result["headline"]["history_fingerprint"] == expected.history_fingerprint
    assert result["score"] == expected.net_cash_per_day
    json.dumps(result, allow_nan=False)
    assert "csv" not in result["request"]


def test_fit_headline_is_oos_and_future_outcomes_do_not_select_policy():
    req = request_fixture(); req["mode"] = "fit"
    first = replay.run(req)
    changed = deepcopy(req)
    lines = changed["csv"].splitlines()
    lines[15:] = [line.replace(",true", ",false") for line in lines[15:]]
    changed["csv"] = "\n".join(lines)
    second = replay.run(changed)
    assert first["policy"] == second["policy"]
    assert first["training_score"] == second["training_score"]
    assert first["split"]["train"]["sessions"] == 14
    assert first["split"]["test"]["sessions"] == 6
    assert first["score"] == first["headline"]["net_cash_per_day"]
    assert first["headline"]["start"] == first["baseline"]["start"]
    assert first["headline"]["history_fingerprint"] == first["baseline"]["history_fingerprint"]
    assert first["headline"]["events"] != second["headline"]["events"]
    json.dumps(first, allow_nan=False)


@pytest.mark.parametrize("key,value", [("mode", "other"),
    ("profile", "unknown"), ("objective", "lambda r: r.net_cash"), ("csv", "timestamp,pnl\nx,10"),
    ("csv", ""), ("regimes", [])])
def test_invalid_contracts_refused(key, value):
    req = request_fixture(); req[key] = value
    with pytest.raises(ValueError):
        replay.run(req)


@pytest.mark.parametrize("name,value", [("population", 1), ("population", 33),
    ("generations", 101), ("generations", True), ("seed", -1)])
def test_search_work_limits(name, value):
    req = request_fixture(); req["mode"] = "fit"; req["search"][name] = value
    with pytest.raises(ValueError):
        replay.run(req)


def test_missing_aware_clocks_overlap_and_duplicate_columns_refused():
    req = request_fixture()
    for invalid in (req["csv"].replace("-04:00", ""), req["csv"].replace("10:05", "09:59"),
                    req["csv"].replace("take_profit,won", "stop_loss,won")):
        with pytest.raises(ValueError):
            replay.parse_history(invalid)


def test_bundle_build_contains_all_canonical_modules_and_is_reproducible(tmp_path, monkeypatch):
    monkeypatch.setattr(build_pages, "ROOT", tmp_path)
    monkeypatch.setattr(build_pages, "DOCS_PY", tmp_path / "docs/py")
    build_pages.main()
    manifest_path = tmp_path / "docs/py/manifest_replay.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {"propfirm_engine/" + p.relative_to(build_pages.SRC_ENG).as_posix()
                for p in build_pages.SRC_ENG.rglob("*.py")}
    assert expected <= {f["path"] for f in manifest["files"]}
    for entry in manifest["files"]:
        assert sha256((tmp_path / "docs/py" / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    before = manifest_path.read_bytes()
    build_pages.main()
    assert before == manifest_path.read_bytes()
    assert (tmp_path / "docs/index.html").read_text(encoding="utf-8") == (build_pages.DASH / "replay.html").read_text(encoding="utf-8")


def test_committed_replay_assets_and_bridge_are_synchronized():
    root = Path(__file__).resolve().parents[1]
    for name, target in (("replay.html", "index.html"), ("replay.html", "trace.html"),
                         ("history.js", "history.js"), ("trace.js", "trace.js"), ("replay.js", "replay.js"),
                         ("replay.css", "replay.css"), ("replay-worker.js", "replay-worker.js")):
        assert (root / "dashboard" / name).read_text(encoding="utf-8") == (root / "docs" / target).read_text(encoding="utf-8")
    source = (root / "dashboard/replay.py").read_text(encoding="utf-8")
    filtered = "\n".join(line for line in source.splitlines() if not line.startswith("sys.path.insert(")) + "\n"
    assert filtered == (root / "docs/py/replay.py").read_text(encoding="utf-8")
