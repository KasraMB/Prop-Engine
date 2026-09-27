"""Synthetic/manual adapters must produce explicit, reproducible engine inputs."""
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
import pytest

from dashboard import replay


def parameters(kind="iid"):
    result = dict(generator=kind, win_rate=0.55, rr=2, stop_loss=100,
                  trades_per_day=4, sessions=80, seed=73, start_date="2026-01-05")
    if kind == "regime":
        result.update(persistence=0.95, spread=0.25)
    if kind == "stochvol":
        result.update(vol_phi=0.9, vol_sigma=0.5)
    return result


@pytest.mark.parametrize("kind", ["iid", "regime", "stochvol"])
def test_generation_reuses_existing_models_and_brackets_preserve_ratio(kind):
    params = parameters(kind)
    payload = replay.generate(params)
    assert payload == replay.generate(params)
    history = replay.parse_history(payload["csv"])
    assert len(history.trades) == 320 and len(history.sessions) == 80
    assert all(t.take_profit / t.stop_loss == pytest.approx(2) for t in history.trades)
    assert all((t.exit_at - t.entry_at) == timedelta(minutes=1) for t in history.trades)
    factory = {"iid": replay.IIDGenerator, "regime": replay.RegimeSwitchingGenerator,
               "stochvol": replay.StochasticVolGenerator}[kind]
    extras = {k: params[k] for k in ("persistence", "spread", "vol_phi", "vol_sigma") if k in params}
    original = factory(win_rate=0.55, rr=2, trades_per_day=4, **extras).generate(80, 73)
    np.testing.assert_array_equal([t.won for t in history.trades], np.asarray(original.rows["return"]) > 0)
    assert payload["provenance"]["parameters"] == params
    assert payload["stats"]["win_rate"] == sum(t.won for t in history.trades) / 320
    assert payload["provenance"]["model"]["edge"] == pytest.approx(0.65)
    if kind == "stochvol":
        assert len({t.stop_loss for t in history.trades}) > 1
    else:
        assert {t.stop_loss for t in history.trades} == {100}
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("wr", [0, 1])
def test_requested_extreme_outcomes_and_rr_are_effective(wr):
    params = parameters(); params.update(win_rate=wr, rr=3.5, stop_loss=27, sessions=2, trades_per_day=3)
    payload = replay.generate(params)
    assert payload["stats"]["win_rate"] == wr
    h = replay.parse_history(payload["csv"])
    assert len(h.trades) == 6
    assert all(t.stop_loss == 27 and t.take_profit == 94.5 for t in h.trades)


def test_synthetic_clocks_skip_weekends_and_follow_eastern_dst():
    params = parameters(); params.update(start_date="2026-03-06", sessions=2, trades_per_day=1)
    h = replay.parse_history(replay.generate(params)["csv"])
    assert h.sessions == (date(2026, 3, 6), date(2026, 3, 9))
    assert h.trades[0].exit_at.hour == h.trades[1].exit_at.hour + 1


@pytest.mark.parametrize("key,value", [("win_rate", -0.1), ("win_rate", 1.1), ("rr", 0),
    ("rr", float("nan")), ("stop_loss", -1), ("trades_per_day", 1.5), ("trades_per_day", True),
    ("trades_per_day", 101), ("seed", -1), ("sessions", 0), ("sessions", 20_000),
    ("generator", "unknown"), ("intraday_excursion", 0.5)])
def test_invalid_generator_inputs_rejected(key, value):
    params = parameters(); params[key] = value
    with pytest.raises(ValueError):
        replay.generate(params)


def test_volatility_edge_cases_reject_nonfinite_or_zero_brackets():
    params = parameters("stochvol"); params["vol_phi"] = 1
    with pytest.raises(ValueError):
        replay.generate(params)
    params.update(vol_phi=0.999999, vol_sigma=5)
    with pytest.raises(ValueError):
        replay.generate(params)


def manual_trade(**kwargs):
    return dict(session="2026-09-01", entry_time="10:00", duration_minutes=5,
                stop_loss=100, take_profit=200, won=True) | kwargs


def test_manual_history_retains_outcomes_units_and_named_timezone():
    rows = [manual_trade(), manual_trade(entry_time="10:06", take_profit=300, won=False)]
    payload = replay.manual({"trades": rows})
    h = replay.parse_history(payload["csv"])
    assert h.trades[0].entry_at.hour == 14
    assert h.trades[1].take_profit == 300 and h.trades[1].won is False
    assert payload["provenance"]["trades"] == rows
    assert payload["stats"]["win_rate"] == 0.5


@pytest.mark.parametrize("change", [dict(entry_time="16:44"), dict(entry_time="08:00"),
    dict(entry_time="10:00+00:00"), dict(session="2026-09-05"), dict(duration_minutes=0),
    dict(stop_loss=True), dict(take_profit=float("inf")), dict(won="maybe")])
def test_invalid_manual_entries_rejected(change):
    with pytest.raises(ValueError):
        replay.manual({"trades": [manual_trade(**change)]})


def test_manual_overlap_is_not_repaired_or_silently_accepted():
    with pytest.raises(ValueError, match="non-overlapping"):
        replay.manual({"trades": [manual_trade(), manual_trade(entry_time="10:02")]})


def test_old_dashboard_and_acknowledgment_are_absent():
    root = Path(__file__).resolve().parents[1]
    for path in ("dashboard/montecarlo.py", "dashboard/montecarlo.html", "dashboard/bridge.py",
                 "dashboard/accounts.py", "dashboard/selfcheck.py", "dashboard/index.html",
                 "docs/montecarlo.html", "docs/interactive.html", "docs/py/mc_engine.py",
                 "docs/py/bridge.py", "docs/py/accounts.py", "docs/py/manifest_mc.json"):
        assert not (root / path).exists()
    for path in ("dashboard/replay.html", "docs/index.html", "docs/trace.html"):
        html = (root / path).read_text(encoding="utf-8")
        assert 'type="checkbox"' not in html
        assert "montecarlo.html" not in html
        assert 'href="trace.html"' in html
