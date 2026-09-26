"""Reject poisoned numeric inputs before resampling or simulation."""

from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from propfirm_engine.data import InvalidTradeDataError, clip_mae_to_holding_interval, preprocess
from propfirm_engine.engine import Engine, RunConfig
from propfirm_engine.model import Account, Phase
from propfirm_engine.rules import ProfitTargetRule


STAMP = datetime(2024, 1, 2, 10)
NONFINITE = [float("nan"), float("inf"), float("-inf")]


def rows(**values):
    return [{"timestamp": STAMP, **values}]


@pytest.mark.parametrize("value", NONFINITE + ["bad", None])
@pytest.mark.parametrize("column", ["return", "pnl", "size"])
def test_rejects_nonfinite_missing_or_nonnumeric_required_values(column, value):
    values = {"return": 1.0} if column == "return" else {"pnl": 10.0, "size": 2.0}
    values[column] = value
    with pytest.raises(InvalidTradeDataError, match=column):
        preprocess(rows(**values))


@pytest.mark.parametrize("value", NONFINITE + ["bad", -1.0])
def test_rejects_invalid_scalar_mae(value):
    with pytest.raises(InvalidTradeDataError, match="mae"):
        preprocess(rows(**{"return": 1.0, "mae": value}))


@pytest.mark.parametrize("value", NONFINITE + ["bad", None, -1.0])
@pytest.mark.parametrize("offset", [-1, 1, 3])
def test_clip_rejects_invalid_magnitudes_even_outside_window(value, offset):
    bars = [(STAMP + timedelta(minutes=offset), value)]
    with pytest.raises(InvalidTradeDataError, match="bar excursion"):
        clip_mae_to_holding_interval(STAMP, STAMP + timedelta(minutes=2), bars)


@pytest.mark.parametrize("value", NONFINITE)
def test_preprocess_rejects_nonfinite_bar_mae(value):
    with pytest.raises(InvalidTradeDataError, match="bar excursion"):
        preprocess(rows(**{"return": 1.0, "entry_time": STAMP,
                           "mae_bars": [(STAMP, value)]}))


@pytest.mark.parametrize("size", [0.0, -1.0, -0.5])
def test_normalization_requires_positive_size_magnitude(size):
    with pytest.raises(InvalidTradeDataError, match="size"):
        preprocess(rows(pnl=-1.0, size=size))


def test_finite_operands_must_not_overflow_normalized_return():
    with pytest.raises(InvalidTradeDataError, match="normalized return"):
        preprocess(rows(pnl=1e308, size=1e-308))


@pytest.mark.parametrize("value", NONFINITE + [0.0, -1.0, "bad"])
def test_rejects_invalid_cadence_override(value):
    with pytest.raises(InvalidTradeDataError, match="trading_days_per_week"):
        preprocess(rows(**{"return": 1.0}), trading_days_per_week=value)


@pytest.mark.parametrize("value", [np.datetime64("NaT"), "NaT", "not-a-date", ""])
def test_rejects_missing_or_unparseable_timestamps(value):
    with pytest.raises(InvalidTradeDataError, match="timestamp"):
        preprocess([{"timestamp": value, "return": 1.0}])


def test_row_and_column_inputs_preserve_valid_numeric_strings_and_mae_fallback():
    raw = [{"timestamp": STAMP, "return": "-0.8", "mae": None},
           {"timestamp": STAMP + timedelta(minutes=1), "return": "1.5", "mae": "0"}]
    for data in (raw, {name: [r[name] for r in raw] for name in raw[0]}):
        result = preprocess(data, trading_days_per_week="5")
        np.testing.assert_array_equal(result.ret, [-0.8, 1.5])
        np.testing.assert_array_equal(result.trade_low, [-0.8, 0.0])
        assert result.trading_days_per_week == 5.0


def test_bars_remain_authoritative_over_unused_scalar_mae():
    result = preprocess(rows(**{"return": 1.0, "mae": float("nan"),
                                "entry_time": STAMP, "mae_bars": [(STAMP, 2.0)]}))
    np.testing.assert_array_equal(result.trade_low, [-2.0])


def test_noninteger_normalization_scale_is_not_mistaken_for_contract_execution():
    result = preprocess(rows(pnl=-1.0, size=0.5))
    np.testing.assert_array_equal(result.ret, [-2.0])


@pytest.mark.parametrize("entrypoint", ["run", "run_prepared"])
@pytest.mark.parametrize("field", ["ret", "trade_low", "trading_days_per_week"])
@pytest.mark.parametrize("value", NONFINITE)
def test_ready_dataset_cannot_bypass_numeric_guards(monkeypatch, entrypoint, field, value):
    dataset = preprocess(rows(**{"return": 1.0}))
    if field == "trading_days_per_week":
        dataset = replace(dataset, trading_days_per_week=value)
    else:
        # Frozen dataclass does not freeze contained arrays; check again at use.
        getattr(dataset, field)[0] = value
    engine = Engine()
    account = Account("test", 1000, (Phase("evaluation", "eval", (ProfitTargetRule(10.0),)),))

    def unexpected_resampling(*args, **kwargs):
        pytest.fail("invalid numeric input reached resampling")

    monkeypatch.setattr(engine, "_resample", unexpected_resampling)
    with pytest.raises(InvalidTradeDataError, match=field):
        if entrypoint == "run":
            engine.run(account, dataset, RunConfig())
        else:
            engine.run_prepared(engine.prepare(account, RunConfig()), dataset, RunConfig())


@pytest.mark.parametrize("field", ["ret", "trade_low"])
def test_ready_dataset_rejects_misaligned_numeric_arrays(field):
    dataset = preprocess(rows(**{"return": 1.0}))
    dataset = replace(dataset, **{field: np.array([1.0, 2.0])})
    account = Account("test", 1000, (Phase("evaluation", "eval", (ProfitTargetRule(10.0),)),))
    with pytest.raises(InvalidTradeDataError, match="ret.*trade_low"):
        Engine().run(account, dataset, RunConfig(n_paths=1))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"ret": [1.0]}, "ret.*NumPy"),
        ({"ret": np.array([[1.0]])}, "ret.*one-dimensional"),
        ({"ret": np.array([1.0 + 1.0j])}, "ret.*real numeric"),
        ({"trade_low": np.array(["0"])}, "trade_low.*real numeric"),
        ({"ret": np.array([]), "trade_low": np.array([])}, "nonzero lengths"),
        ({"trade_low": np.array([1.0])}, "trade_low.*non-positive"),
        ({"trading_days_per_week": 0.0}, "trading_days_per_week.*positive"),
        ({"trading_days_per_week": "5"}, "trading_days_per_week.*number"),
    ],
)
def test_ready_dataset_rejects_invalid_numeric_representation(changes, message):
    dataset = replace(preprocess(rows(**{"return": 1.0})), **changes)
    account = Account("test", 1000, (Phase("evaluation", "eval", (ProfitTargetRule(10.0),)),))
    with pytest.raises(InvalidTradeDataError, match=message):
        Engine().run(account, dataset, RunConfig(n_paths=1))


@pytest.mark.parametrize("value", NONFINITE)
def test_raw_engine_input_rejects_nonfinite_mae_before_resampling(monkeypatch, value):
    engine = Engine()
    account = Account("test", 1000, (Phase("evaluation", "eval", (ProfitTargetRule(10.0),)),))

    def unexpected_resampling(*args, **kwargs):
        pytest.fail("raw NaN/inf input reached resampling")

    monkeypatch.setattr(engine, "_resample", unexpected_resampling)
    # Column-oriented input takes the same rejection path as row-oriented input.
    raw = {"timestamp": [STAMP], "return": [1.0], "mae": [value]}
    with pytest.raises(InvalidTradeDataError, match="mae"):
        engine.run(account, raw, RunConfig())
