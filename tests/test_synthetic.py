"""Step 3b — synthetic trade-stream generators (BUILD_SPEC Step 3b, ARCHITECTURE §11.7)."""

from __future__ import annotations

import numpy as np
import pytest

from propfirm_engine.data import TradeDataset, preprocess
from propfirm_engine.synthetic import (
    IIDGenerator,
    RegimeSwitchingGenerator,
    StochasticVolGenerator,
    TradeStreamGenerator,
)

ALL_GENERATORS = [IIDGenerator, RegimeSwitchingGenerator, StochasticVolGenerator]


def _lag1_autocorr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom == 0:
        return 0.0
    return float(np.sum(x[:-1] * x[1:]) / denom)


# --- schema / preprocess compatibility (the hard contract, §11.7) ----------- #


@pytest.mark.parametrize("cls", ALL_GENERATORS)
def test_every_generator_preprocesses_with_no_special_casing(cls):
    stream = cls(win_rate=0.5, rr=2.0).generate(n_days=20, seed=1)
    ds = preprocess(stream.rows)  # the identical Step 3 pipeline real data uses
    assert isinstance(ds, TradeDataset)
    assert ds.n_trades == stream.n_trades
    # mae flowed through as the (already-clipped) scalar: trade_low == -mae.
    np.testing.assert_allclose(ds.trade_low, [-m for m in stream.rows["mae"]])


# --- edge is derived, never input (§11.7.1, §I2) ---------------------------- #


@pytest.mark.parametrize("cls", ALL_GENERATORS)
def test_edge_is_not_an_accepted_parameter(cls):
    with pytest.raises(TypeError):
        cls(win_rate=0.5, rr=2.0, edge=0.5)


@pytest.mark.parametrize("cls", ALL_GENERATORS)
def test_derived_edge_and_breakeven_match_the_formula(cls):
    g = cls(win_rate=0.5, rr=2.0)
    assert g.edge == pytest.approx(0.5 * (2.0 + 1.0) - 1.0)  # = 0.5
    assert g.breakeven_win_rate == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize("cls", ALL_GENERATORS)
def test_realized_long_run_edge_matches_derived_edge(cls):
    g = cls(win_rate=0.45, rr=2.0)
    stream = g.generate(n_days=4000, seed=7)
    realized = np.mean(stream.rows["return"])
    # The dependent generators (regime, stoch-vol) inflate the estimator variance,
    # so tolerate ~3σ (measured std ≈ 0.03) rather than a tight band that would be
    # flaky on other seeds even with correct code.
    assert realized == pytest.approx(g.edge, abs=0.1)


# --- IID properties --------------------------------------------------------- #


def test_iid_win_frequency_matches_win_rate():
    g = IIDGenerator(win_rate=0.6, rr=1.5)
    stream = g.generate(n_days=5000, seed=3)
    ret = np.asarray(stream.rows["return"])
    win_freq = np.mean(ret > 0)
    assert win_freq == pytest.approx(0.6, abs=0.02)


def test_iid_has_no_systematic_autocorrelation():
    g = IIDGenerator(win_rate=0.5, rr=2.0)
    stream = g.generate(n_days=5000, seed=4)
    signs = (np.asarray(stream.rows["return"]) > 0).astype(float)
    assert abs(_lag1_autocorr(signs)) < 0.05


# --- Regime-switching: persistence + stationary mix ------------------------- #


def test_regime_switching_shows_positive_persistence_vs_iid():
    common = dict(win_rate=0.5, rr=2.0)
    regime = RegimeSwitchingGenerator(**common, spread=0.3, persistence=0.97)
    iid = IIDGenerator(**common)
    r_signs = (np.asarray(regime.generate(6000, seed=11).rows["return"]) > 0).astype(float)
    i_signs = (np.asarray(iid.generate(6000, seed=11).rows["return"]) > 0).astype(float)
    r_ac = _lag1_autocorr(r_signs)
    i_ac = _lag1_autocorr(i_signs)
    assert r_ac > 0.1  # clear positive persistence
    assert r_ac > i_ac + 0.1  # and materially more than i.i.d.


def test_regime_switching_stationary_mix_reproduces_target_win_rate():
    g = RegimeSwitchingGenerator(win_rate=0.55, rr=2.0, spread=0.25, persistence=0.9)
    stream = g.generate(n_days=8000, seed=21)
    win_freq = np.mean(np.asarray(stream.rows["return"]) > 0)
    assert win_freq == pytest.approx(0.55, abs=0.03)


@pytest.mark.parametrize("win_rate,spread", [(0.85, 0.25), (0.10, 0.30), (0.95, 0.5)])
def test_regime_switching_preserves_win_rate_even_when_spread_would_clip(win_rate, spread):
    # A naive min/max clip of the two regime rates skews the mean off target near
    # the win-rate boundaries (e.g. win_rate=0.85 read ~0.80). The effective-spread
    # cap must keep the realized win-frequency at the target regardless.
    g = RegimeSwitchingGenerator(win_rate=win_rate, rr=2.0, spread=spread, persistence=0.9)
    stream = g.generate(n_days=8000, seed=33)
    ret = np.asarray(stream.rows["return"])
    win_freq = np.mean(ret > 0)
    assert win_freq == pytest.approx(win_rate, abs=0.03)
    # and the realized edge still matches the reported (derived) edge
    assert np.mean(ret) == pytest.approx(g.edge, abs=0.1)


# --- Stochastic vol: clustering + fat tails, win_rate preserved ------------- #


def test_stochastic_vol_clusters_absolute_returns():
    sv = StochasticVolGenerator(win_rate=0.5, rr=2.0, vol_phi=0.92, vol_sigma=0.6)
    iid = IIDGenerator(win_rate=0.5, rr=2.0)
    sv_abs = np.abs(np.asarray(sv.generate(6000, seed=31).rows["return"]))
    iid_abs = np.abs(np.asarray(iid.generate(6000, seed=31).rows["return"]))
    assert _lag1_autocorr(sv_abs) > 0.1  # volatility clustering
    assert _lag1_autocorr(sv_abs) > _lag1_autocorr(iid_abs) + 0.1


def test_stochastic_vol_has_fatter_tails_than_iid():
    def kurtosis(x):
        x = np.asarray(x, dtype=np.float64)
        x = x - x.mean()
        m2 = np.mean(x**2)
        m4 = np.mean(x**4)
        return m4 / (m2**2)

    sv = StochasticVolGenerator(win_rate=0.5, rr=2.0, vol_phi=0.9, vol_sigma=0.7)
    iid = IIDGenerator(win_rate=0.5, rr=2.0)
    sv_ret = sv.generate(6000, seed=41).rows["return"]
    iid_ret = iid.generate(6000, seed=41).rows["return"]
    assert kurtosis(sv_ret) > kurtosis(iid_ret)


def test_stochastic_vol_preserves_win_rate():
    g = StochasticVolGenerator(win_rate=0.4, rr=2.0, vol_phi=0.9, vol_sigma=0.5)
    stream = g.generate(n_days=6000, seed=51)
    win_freq = np.mean(np.asarray(stream.rows["return"]) > 0)
    assert win_freq == pytest.approx(0.4, abs=0.02)


# --- mae synthesis (§11.7.1, exercises CONTINUOUS rules) -------------------- #


@pytest.mark.parametrize("cls", ALL_GENERATORS)
def test_mae_follows_the_exact_per_trade_synthesis_formula(cls):
    # Independent check of the mae formula (not a re-derivation): a trade's risk is
    # its per-unit magnitude |vol|, recoverable as ret/rr for a win and -ret for a
    # loss. The synthesized mae must be exactly intraday_excursion*risk for wins and
    # 1*risk for losses — so a win's excursion is strictly below its risk, and a
    # loss's equals it. Holds for every generator (vol=1 for IID/regime is a case).
    exc = 0.4
    g = cls(win_rate=0.5, rr=2.0, intraday_excursion=exc)
    stream = g.generate(n_days=300, seed=61)
    ret = np.asarray(stream.rows["return"])
    mae = np.asarray(stream.rows["mae"])
    wins = ret > 0
    risk = np.where(wins, ret / g.rr, -ret)  # per-trade risk = |vol|
    expected_mae = np.where(wins, exc * risk, risk)
    np.testing.assert_allclose(mae, expected_mae, rtol=1e-9, atol=1e-9)
    # excursion never exceeds risk; strictly below for wins (exc < 1)
    assert np.all(mae <= risk + 1e-9)
    assert np.all(mae[wins] < risk[wins])


def test_winning_trades_have_a_nonzero_floating_low_for_continuous_rules():
    # The whole point of mae depth: a winning trade closes positive yet has a
    # floating low below entry, so a CONTINUOUS check can see a breach an EOD check
    # cannot. There must exist a trade with return > 0 and trade_low < 0.
    g = IIDGenerator(win_rate=0.6, rr=2.0, intraday_excursion=0.5)
    ds = preprocess(g.generate(n_days=100, seed=71).rows)
    winners_with_drawdown = (ds.ret > 0) & (ds.trade_low < 0)
    assert np.any(winners_with_drawdown)


# --- determinism + provenance (§11.7.3) ------------------------------------- #


@pytest.mark.parametrize("cls", ALL_GENERATORS)
def test_fixed_seed_reproduces_identical_rows(cls):
    g = cls(win_rate=0.5, rr=2.0)
    a = g.generate(n_days=50, seed=99)
    b = g.generate(n_days=50, seed=99)
    assert a.rows["timestamp"] == b.rows["timestamp"]
    np.testing.assert_array_equal(a.rows["return"], b.rows["return"])
    np.testing.assert_array_equal(a.rows["mae"], b.rows["mae"])


@pytest.mark.parametrize("cls", ALL_GENERATORS)
def test_different_seed_differs(cls):
    g = cls(win_rate=0.5, rr=2.0)
    a = np.asarray(g.generate(n_days=200, seed=1).rows["return"])
    b = np.asarray(g.generate(n_days=200, seed=2).rows["return"])
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("cls", ALL_GENERATORS)
def test_provenance_records_type_params_seed_and_derived_edge(cls):
    g = cls(win_rate=0.45, rr=2.0)
    prov = g.generate(n_days=10, seed=123).provenance
    assert prov.generator == cls.__name__
    assert prov.seed == 123
    assert prov.params["win_rate"] == 0.45
    assert prov.params["rr"] == 2.0
    assert prov.edge == pytest.approx(g.edge)
    assert prov.breakeven_win_rate == pytest.approx(g.breakeven_win_rate)


# --- cadence: trades_per_day drives the session/day structure (§11.5) -------- #


def test_trades_per_day_yields_expected_day_structure():
    g = IIDGenerator(win_rate=0.5, rr=2.0, trades_per_day=3)
    ds = preprocess(g.generate(n_days=10, seed=5).rows)
    assert ds.n_days == 10  # one session per weekday
    np.testing.assert_array_equal(ds.day_count, [3] * 10)


def test_weekday_sessions_derive_a_five_per_week_cadence():
    # Sessions land only on weekdays, so preprocess should derive ~5 trading days
    # per calendar week. A short block reads slightly high (the span ends on a
    # Friday, so trailing weekends are outside it); over many weeks that end-effect
    # washes out and the cadence converges to 5.
    g = IIDGenerator(win_rate=0.5, rr=2.0, trades_per_day=2)
    ds = preprocess(g.generate(n_days=100, seed=5).rows)
    assert ds.n_days == 100
    assert 4.7 <= ds.trading_days_per_week <= 5.3


# --- parameter validation --------------------------------------------------- #


def test_generator_rejects_out_of_range_parameters():
    with pytest.raises(ValueError):
        IIDGenerator(win_rate=1.5, rr=2.0)
    with pytest.raises(ValueError):
        IIDGenerator(win_rate=0.5, rr=0.0)
    with pytest.raises(ValueError):
        IIDGenerator(win_rate=0.5, rr=2.0, intraday_excursion=1.0)


def test_base_generator_is_abstract():
    with pytest.raises(TypeError):
        TradeStreamGenerator(win_rate=0.5, rr=2.0)
