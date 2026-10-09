from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from propfirm_engine import (
    BacktestConfig, Engine, InfeasiblePolicy, Instrument, Market, MarketTape, Order,
    Parameter, Quote, QuoteModel, ReplayCancelled, RiskConfig, RollingConfig,
    SearchCancelled, evaluate_strategy,
    evaluate_scenarios,
)
from propfirm_engine.firms.lucidflex import replay_50k


X = Instrument("X", 1, 1)
SPEC = replay_50k(eval_fee=105.2, reset_fee=105, contract_type="mini")
CONFIG = BacktestConfig(0, timedelta(0), timedelta(0), timedelta(0))


def tape(oos_move=10, days=10):
    start = datetime(2026, 9, 1, 14, tzinfo=timezone.utc)
    markets = []
    offsets = [i for i in range(days*2) if (start+timedelta(days=i)).weekday() < 5][:days]
    for day, offset in enumerate(offsets):
        for minute, price in enumerate((10000, 10000, 10010 if day < 7 else 10000+oos_move)):
            markets.append(Market(start+timedelta(days=offset, minutes=minute), (Quote("X", price, price, price),)))
    return MarketTape(markets, [X], sessions=tuple(date(2026, 9, 1)+timedelta(days=i) for i in offsets))


class Daily:
    def __init__(self, params, seed):
        self.params = params

    def on_market(self, context, market):
        if context.warmup:
            return
        if market.at.minute == 0:
            return [Order(str(market.at), "X", self.params["quantity"])]
        if market.at.minute == 1:
            return [Order(str(market.at), "X", -self.params["quantity"], reduce_only=True)]


def setup(seed):
    return dict(models={"X": QuoteModel()}, fidelity="observed_marks",
                max_mark_age=timedelta(days=1), liquidation_fee=0)


def ending_balance(evaluation):
    return evaluation.metrics["distributions"]["balance"]["mean"]


def fit(**kwargs):
    options = dict(baseline={"quantity": 1}, space={"quantity": Parameter("integer", 1, 3)},
                   setup=setup, generations=3, population=4, objective=ending_balance)
    options.update(kwargs)
    return Engine().fit_strategy(SPEC, options.pop("tape", tape()), CONFIG,
                                 options.pop("factory", Daily), **options)


def test_search_uses_is_and_reports_oos_for_selected_and_baseline():
    result = fit()
    assert result.params == {"quantity": 3}
    assert result.training.score == 50210
    assert result.score == 50090 and result.baseline.score == 50030
    assert result.split_session == date(2026, 9, 10)
    assert result.metrics["distributions"]["net_cash"]["mean"] == -105.2
    assert "net_cash_per_day" in result.metrics["distributions"]
    assert result.stability["oos_score"] == result.score


def test_changing_only_oos_never_changes_trials_or_selected_parameters():
    first, changed = fit(), fit(tape=tape(oos_move=-20))
    assert first.params == changed.params
    assert first.trials == changed.trials
    assert changed.score == 49820 and changed.score < changed.baseline.score


def test_minimization_and_custom_distribution_objectives_keep_all_metrics():
    result = fit(direction="minimize")
    assert result.params == {"quantity": 1}
    for name in ("net_cash", "net_cash_per_day", "required_bankroll", "fills", "fees"):
        stats = result.metrics["distributions"][name]
        assert "variance" in stats and "percentiles" in stats


def test_inner_validation_rolling_and_repeated_seeds_are_separate():
    result = fit(validation_fraction=.3, rolling=RollingConfig(2), seeds=(11, 22))
    assert result.validation is not None
    assert len(result.training.paths) == 6
    assert len(result.validation.paths) == 4
    assert len(result.selected.paths) == 4
    assert result.training.paths[-1].last_session < result.validation.paths[0].first_session
    assert result.validation.paths[-1].last_session < result.split_session
    assert result.metrics["sample_kind"] == "historical_windows"
    assert result.metrics["confidence_supported_bankroll"] is None


def test_explicit_constraints_are_infeasible_but_simulator_errors_propagate():
    result = fit(constraint=lambda p: p["quantity"] < 3)
    assert result.params["quantity"] == 2
    assert any(not t.feasible for t in result.trials)

    def broken(params, seed):
        raise RuntimeError("broken strategy")

    with pytest.raises(RuntimeError, match="broken strategy"):
        fit(factory=broken)


def test_explicit_policy_exception_is_not_a_catch_all():
    def factory(params, seed):
        if params["quantity"] == 3:
            raise InfeasiblePolicy("cannot size this policy")
        return Daily(params, seed)

    result = fit(factory=factory)
    assert result.params["quantity"] == 2
    assert any(t.reason == "cannot size this policy" for t in result.trials)


def test_cancel_resume_matches_uninterrupted_with_completed_trials_cached():
    saved = []
    with pytest.raises(SearchCancelled) as error:
        fit(study_id="daily-v1", checkpoint=saved.append, cancel=lambda: len(saved) >= 2)
    assert len(error.value.checkpoint.trials) == 2
    resumed = fit(study_id="daily-v1", resume=error.value.checkpoint)
    full = fit(study_id="daily-v1")
    assert resumed.trials == full.trials
    assert resumed.selected == full.selected
    assert resumed.checkpoint == full.checkpoint
    with pytest.raises(ValueError, match="does not match"):
        fit(study_id="daily-v2", resume=error.value.checkpoint)
    with pytest.raises(ValueError, match="study_id"):
        fit(resume=error.value.checkpoint)


def test_wallet_shortfall_gets_a_fresh_identical_unrestricted_replay():
    result = evaluate_strategy(SPEC, tape(), replace(CONFIG, initial_wallet=100), Daily,
        params={"quantity": 1}, setup=setup, risk=RiskConfig(bankroll=100))
    assert result.paths[0].cash.observed_funding_shortfall
    assert result.paths[0].cash.required_bankroll == 105.2
    assert result.metrics["ruin_probability"] == 1


def test_warmup_cannot_place_orders_or_use_future_observations():
    seen = []

    class Warm(Daily):
        def on_market(self, context, market):
            seen.append((market.at.date(), context.warmup))
            return super().on_market(context, market)

    data = tape()
    result = evaluate_strategy(SPEC, data.view(7), CONFIG, Warm, params={"quantity": 1},
                               setup=setup, warmup_sessions=2)
    assert result.paths[0].fills == 6
    assert [day for day, warm in seen if warm] == [day for day in data.sessions[5:7] for _ in range(3)]
    assert all(day < data.sessions[7] for day, warm in seen if warm)


def test_replay_cancellation_never_returns_partial_performance():
    data = tape()
    with pytest.raises(ReplayCancelled):
        Engine().replay_strategy(SPEC, data, [X], CONFIG, Daily({"quantity": 1}, 0),
                                 sessions=data.sessions, cancel=lambda: True, **setup(0))


@pytest.mark.parametrize("domain,values", [
    (Parameter("integer", 1, 3), (1, 2, 3)),
    (Parameter("continuous", -2, 4), (-2, 0, 4)),
    (Parameter("categorical", choices=("long", "short")), ("long", "short")),
])
def test_parameter_round_trip(domain, values):
    assert [domain.decode(domain.encode(v)) for v in values] == list(values)


@pytest.mark.parametrize("args", [dict(kind="unknown"), dict(low=2, high=1),
    dict(kind="integer", low=1.0, high=3), dict(kind="categorical", choices=("a", "a")),
    dict(low=float("nan")), dict(low=True)])
def test_invalid_domains_rejected(args):
    with pytest.raises(ValueError):
        Parameter(**args)


def test_fixed_parameters_are_preserved_and_factory_gets_read_only_values():
    class Fixed(Daily):
        def __init__(self, params, seed):
            assert params["exit"] == "signal"
            with pytest.raises(TypeError):
                params["quantity"] = 4
            super().__init__(params, seed)

    result = fit(factory=Fixed, baseline={"quantity": 1, "exit": "signal"})
    assert result.params["exit"] == "signal"


def test_walk_forward_refits_only_prior_sessions_and_reports_every_oos_fold():
    result = Engine().walk_strategy(SPEC, tape(), CONFIG, Daily, train_sessions=4,
        test_sessions=2, baseline={"quantity": 1}, space={"quantity": Parameter("integer", 1, 3)},
        setup=setup, objective=ending_balance, generations=1, population=4)
    assert len(result.folds) == 3
    assert [f.selected.paths[0].first_session for f in result.folds] == list(tape().sessions[4::2])
    assert all(f.training.paths[-1].last_session < f.split_session for f in result.folds)
    assert result.metrics["paths"] == 3


def test_ordered_scenarios_retain_all_metrics_and_explicit_uncertainty():
    result = evaluate_scenarios(SPEC, (tape(move) for move in (-10, 0, 10)), CONFIG,
        Daily, params={"quantity": 1}, setup=setup, sample_kind="independent_model")
    assert result.metrics["sample_kind"] == "independent_model"
    assert result.metrics["paths"] == 3
    assert result.metrics["distributions"]["balance"]["minimum"] == 50040
    assert result.metrics["distributions"]["balance"]["maximum"] == 50100


def test_mixed_space_and_continuous_resume_are_deterministic():
    kwargs = dict(baseline={"quantity": 1, "risk": .5, "exit": "signal"}, generations=2,
        space={"quantity": Parameter("integer", 1, 3), "risk": Parameter(),
               "exit": Parameter("categorical", choices=("signal", "time"))}, study_id="mixed-v1")
    saved = []
    with pytest.raises(SearchCancelled) as error:
        fit(**kwargs, checkpoint=saved.append, cancel=lambda: len(saved) >= 3)
    resumed = fit(**kwargs, resume=error.value.checkpoint)
    full = fit(**kwargs)
    assert resumed.selected == full.selected
    assert resumed.trials == full.trials


def test_invalid_model_and_objective_errors_are_not_converted_to_infeasible():
    with pytest.raises(ValueError, match="finite scalar"):
        fit(objective=lambda evaluation: float("nan"))
    with pytest.raises(ValueError, match="cannot override"):
        fit(setup=lambda seed: setup(seed) | {"sessions": ()})


def test_complete_cycle_approximation_is_separate_from_finite_horizon_metrics():
    result = evaluate_strategy(SPEC, tape(), CONFIG, Daily, params={"quantity": 1}, setup=setup)
    assert result.ultimate_ruin(bankroll=1000)["status"] == "not_identified"
    from propfirm_engine import Abandon

    class End(Daily):
        def on_market(self, context, market):
            if market.at.minute == 2:
                return [Abandon()]
            return super().on_market(context, market)

    result = evaluate_strategy(SPEC, tape(), CONFIG, End, params={"quantity": 1}, setup=setup)
    assert len(result.paths[0].cycles) == 10
    ultimate = result.ultimate_ruin(bankroll=1000, paths=2, max_cycles=2)
    assert ultimate["status"] == "certain_ruin"
    assert "IID cycle approximation" in ultimate["warning"]
    assert result.metrics["ruin_probability"] is None
