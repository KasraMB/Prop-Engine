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


def test_ruin_analysis_uses_only_holdout_and_does_not_select_policy():
    req = request_fixture()
    req["mode"] = "fit"
    initial = replay.run(req)
    req["ruin"] = {"paths": 3, "sessions": 12, "cycle_paths": 100, "cycle_steps": 10}
    result = replay.run(req)
    assert initial["policy"] == result["policy"]
    assert initial["score"] == result["score"]
    assert result["ruin"]["source_sessions"] == 6
    assert result["ruin"]["risk"]["paths"] == 3
    assert result["ruin"]["payout_retention"] == 1
    json.dumps(result, allow_nan=False)


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
                         ("rolling.js", "rolling.js"),
                         ("risk.js", "risk.js"),
                         ("research.html", "research.html"), ("research.js", "research.js"),
                         ("replay.css", "replay.css"), ("replay-worker.js", "replay-worker.js")):
        assert (root / "dashboard" / name).read_text(encoding="utf-8") == (root / "docs" / target).read_text(encoding="utf-8")
    source = (root / "dashboard/replay.py").read_text(encoding="utf-8")
    filtered = "\n".join(line for line in source.splitlines() if not line.startswith("sys.path.insert(")) + "\n"
    assert filtered == (root / "docs/py/replay.py").read_text(encoding="utf-8")


def test_rolling_adapter_scores_windows_and_keeps_chronological_ledger():
    req = request_fixture()
    req.update(mode="fit", rolling={"window_sessions": 4, "stride_sessions": 2})
    result = replay.run(req)
    rolling = result["rolling"]
    assert result["selection_basis"] == "mean_window_objective"
    assert result["score"] == rolling["headline"]["summary"]["score"]
    assert len(rolling["training"]["windows"]) == 6
    assert len(rolling["headline"]["windows"]) == 2
    assert rolling["headline"]["excluded_incomplete_starts"] == 1
    assert result["headline"]["events"] and result["training"]["events"]
    assert [w["first_session"] for w in rolling["headline"]["windows"]] == [w["first_session"] for w in rolling["baseline"]["windows"]]
    json.dumps(result, allow_nan=False)
    req["mode"] = "backtest"
    full = replay.run(req)
    assert len(full["rolling"]["headline"]["windows"]) == 9
    assert "baseline" not in full["rolling"]


def test_rolling_adapter_rejects_incomplete_partitions_but_allows_large_search(monkeypatch):
    req = request_fixture()
    req.update(mode="fit", rolling={"window_sessions": 7})
    with pytest.raises(ValueError, match="no complete"):
        replay.run(req)
    req["rolling"] = {"window_sessions": 4}
    monkeypatch.setattr(replay, "_rolling_work", lambda *args: 2_000_000)
    req["risk_bounds"] = {name: [50, 2000] for name in req["risk_bounds"]}
    updates = []
    result = replay.run(req, progress=updates.append)
    assert result["estimated_trade_visits"] > 2_000_000
    assert updates[0]["estimated_trade_visits"] == result["estimated_trade_visits"]
    assert any(u["stage"] == "holdout" for u in updates)
    assert result["evaluations"] > 1
    req["mode"] = "backtest"
    assert replay.run(req)["estimated_trade_visits"] > 2_000_000


def test_risk_report_is_oos_only_and_settings_do_not_select_policy():
    req = request_fixture()
    req.update(mode="fit", rolling={"window_sessions": 4}, risk={"target_ruin_probability": .01})
    first = replay.run(req)
    changed = deepcopy(req)
    changed["risk"] = {"target_ruin_probability": .5, "percentiles": [.1, .5, .9]}
    second = replay.run(changed)
    assert first["policy"] == second["policy"]
    assert first["training_score"] == second["training_score"]
    assert first["risk"]["paths"] == len(first["rolling"]["headline"]["windows"])
    assert first["risk"]["distributions"]["net_cash"]["mean"] == first["rolling"]["headline"]["summary"]["distributions"]["net_cash"]["mean"]
    assert first["risk"]["sample_kind"] == "historical_windows"
    assert first["risk"]["confidence_supported_bankroll"] is None
    assert set(second["risk"]["distributions"]["net_cash"]["percentiles"]) == {"0.1", "0.5", "0.9"}
    json.dumps(first, allow_nan=False)
