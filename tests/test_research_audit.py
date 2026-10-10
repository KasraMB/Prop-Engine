from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from propfirm_engine import (
    BracketHistory, DollarPolicy, Engine, Fill, Marks, ProfitTargetRule,
    RollingConfig, Timing, TrailingDrawdownRule,
)
from propfirm_engine import fitting, target_research
from propfirm_engine.optimizer import RenewalObjective
from test_chronological_backtest import config, session_day, spec, trade


AT = datetime(2026, 9, 1, 14, tzinfo=timezone.utc)
DAY = date(2026, 9, 1)


def _mutate(monkeypatch, module, name, old, new):
    function = getattr(module, name)
    source = inspect.getsource(function)
    if old not in source:
        raise RuntimeError('mutation target changed; update the audit fixture')
    scope = dict(function.__globals__)
    exec(compile(source.replace(old, new), '<audit mutation>', 'exec'), scope)
    monkeypatch.setattr(module, name, scope[name])


def _search_check(monkeypatch, adapter, direction, mutate):
    module = fitting if adapter in ('bracket', 'rolling') else target_research
    name = 'fit_holdout' if module is fitting else 'fit_targets'
    expected = 100 if direction == 'minimize' else 300

    class Candidates:
        def __init__(self, x, *args, **kwargs):
            self.x = np.asarray(x)

        def optimize(self, score):
            low, high = self.x.copy(), self.x.copy()
            low[0], high[0] = 0, 1
            a, b = score(low), score(high)
            assert (a > b) if direction == 'minimize' else (b > a)
            return SimpleNamespace(x=low if a > b else high, score=max(a, b))

    monkeypatch.setattr(module, 'CMAES', Candidates)
    if mutate:
        _mutate(monkeypatch, module, name, 'sign = 1 if direction == "maximize" else -1', 'sign = 1')
    policy = DollarPolicy.constant(200)
    common = dict(direction=direction, generations=1, population=4,
                  risk_bounds={r.name: (100, 300) for r in policy.regimes})
    if module is fitting:
        history = BracketHistory(tuple(trade(session_day(i), target=10) for i in range(10)))
        fit = module.fit_holdout(spec(), history, config(), policy=policy,
            rolling=RollingConfig(2) if adapter == 'rolling' else None,
            objective=lambda r: r.final_balance, **common)
        selected = fit.policy
        assert fit.in_sample_score == 50000 + (2 if adapter == 'rolling' else 7)*(expected//100)*10
    else:
        from test_target_research import SPEC, CONFIG
        monkeypatch.setattr(np.random, 'default_rng', lambda seed:
                            SimpleNamespace(random=lambda shape: np.full(shape, .999999)))
        brackets = target_research.TargetPolicy(policy, (500, 500))
        fit = module.fit_targets(SPEC, target_research.BracketModel(sessions=1), CONFIG,
            policy=brackets, target_bounds={r.name: (100, 1000) for r in policy.regimes},
            paths=7, objective=lambda r: -r.final_balance, **common)
        selected = fit.policy.sizing
    assert selected.regimes[0].risk_dollars == expected


@pytest.mark.parametrize('adapter', ['bracket', 'rolling', 'target'])
@pytest.mark.parametrize('direction', ['minimize', 'maximize'])
def test_search_orders_distinct_candidates_and_reports_the_selected_policy(monkeypatch, adapter, direction):
    _search_check(monkeypatch, adapter, direction, False)


@pytest.mark.parametrize('adapter', ['bracket', 'rolling', 'target'])
def test_search_assertions_reject_always_maximize_mutation(monkeypatch, adapter):
    with pytest.raises(AssertionError):
        _search_check(monkeypatch, adapter, 'minimize', True)


def _tail_check(monkeypatch):
    # Ratios -10, 5, 5, 20. Worst 37.5% weighs 1.5 observations.
    draws = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    monkeypatch.setattr(np.random, 'default_rng', lambda seed: SimpleNamespace(integers=lambda *a, **k: draws))
    value = RenewalObjective(cvar_q=.375, n_boot=4)._rate(np.array([-10., 20.]), np.ones(2))
    assert value == pytest.approx((-10 + .5*5)/1.5)
    assert value < 5


def test_cvar_has_independently_calculated_fractional_tail(monkeypatch):
    _tail_check(monkeypatch)


def test_tail_assertions_reject_mean_mutation(monkeypatch):
    monkeypatch.setattr(RenewalObjective, '_rate', lambda self, reward, time: np.sum(reward)/np.sum(time))
    with pytest.raises(AssertionError):
        _tail_check(monkeypatch)


@pytest.mark.parametrize('kwargs', [dict(cvar_q=0), dict(cvar_q=float('nan')), dict(n_boot=0),
    dict(n_boot=1.5), dict(boot_seed=-1), dict(p_min=2), dict(penalty=-1), dict(include_fees=1)])
def test_invalid_tail_objectives_fail_before_search(kwargs):
    with pytest.raises(ValueError):
        RenewalObjective(**kwargs)


def test_decimal_holdout_boundary_is_exact():
    history = BracketHistory(tuple(trade(session_day(i)) for i in range(90)))
    training, held = history.split(.7)
    assert len(training.sessions) == 63 and len(held.sessions) == 27


def test_new_eod_floor_checks_open_equity_at_the_same_close():
    from test_event_rules import run
    at = AT+timedelta(hours=6, minutes=40)
    events = [Fill(at, 'X', 2, 10000), Fill(at+timedelta(seconds=1), 'Y', 1, 10000),
              Fill(at+timedelta(seconds=2), 'X', -2, 12500),
              Marks(at+timedelta(seconds=3), (('Y', 7500),))]
    result = run(events, [TrailingDrawdownRule(2000, update_timing=Timing.EOD),
                         ProfitTargetRule(100000)], flatten=False)
    failure, = [e for e in result.replay.events if e.kind == 'failure']
    assert failure.floor == 53000 and failure.balance == 52500
    assert failure.at == AT.replace(hour=20, minute=45)


def test_summary_and_dated_payouts_share_partial_buffer_headroom():
    from test_kernels import _both, _funded_1day_payout
    from propfirm_engine import PayoutSchema
    from propfirm_engine.payouts import PayoutLedger
    compiled = _funded_1day_payout(max_payouts=2, buffer_floor=100100, min_request=1)
    _, amounts, _ = _both(compiled, [150.], [0], [0.], start_equity=100000)
    assert amounts == [45.]
    schema = PayoutSchema((2000,), .9, 2, buffer_floor=100100, min_request=1)
    ledger = PayoutLedger(schema, opening_balance=100000, qualifying_days=1, winning_day_profit=100)
    ledger.record_trade(AT, 150)
    ledger.close_session(AT+timedelta(minutes=1), DAY)
    assert ledger.maximum_request() == 50
    request = ledger.request(AT+timedelta(minutes=1), 50, flat=True)
    assert ledger.get_request(request).net == 45


def test_ratio_tail_resamples_reward_and_time_pairs_not_individual_rates(monkeypatch):
    draws = np.array([[0, 0], [0, 1], [1, 1]])
    monkeypatch.setattr(np.random, 'default_rng', lambda seed: SimpleNamespace(integers=lambda *a, **k: draws))
    # Ratios 10, 2 and 0; the lowest two average 1, not 2.5 (mean individual rates).
    value = RenewalObjective(cvar_q=2/3, n_boot=3)._rate(np.array([10., 0.]), np.array([1., 4.]))
    assert value == 1


def test_preflight_rejects_late_funded_features_before_any_evaluation():
    from propfirm_engine.firms.lucidflex import replay_50k
    SPEC = replay_50k(eval_fee=105.2, reset_fee=105, contract_type="mini")
    funded = SPEC.account.phases[1]
    changed = replace(funded, payout_schema=replace(funded.payout_schema, recompute_floor_on_payout=True))
    profile = replace(SPEC, account=replace(SPEC.account, phases=(SPEC.account.phases[0], changed)))
    with pytest.raises(ValueError, match='recompute_floor'):
        Engine().check_replay(profile)


def test_capital_counterpart_cannot_use_another_market_history():
    from test_risk import cash_result
    from propfirm_engine.risk import cash_risk_path
    stopped = cash_result([(0,-100,'fee')], wallet=100, wait=True)
    full = cash_result([(0,-100,'fee'), (1,-100,'fee')])
    stopped.history_fingerprint, full.history_fingerprint = 'a', 'b'
    with pytest.raises(ValueError, match='match'):
        cash_risk_path(stopped, unrestricted=full)


def test_undefined_bootstrap_rates_are_not_dropped_or_rewarded(monkeypatch):
    draws = np.array([[0, 0], [0, 1]])
    monkeypatch.setattr(np.random, 'default_rng', lambda seed: SimpleNamespace(integers=lambda *a, **k: draws))
    assert RenewalObjective(cvar_q=.5, n_boot=2)._rate(np.array([-100., 200.]), np.array([0., 1.])) is None
    outcomes = SimpleNamespace(net_payout=np.array([0.]), eval_fee=100, activation_fee=0,
        reached_funded=np.array([False]), total_trading_days=np.array([0]), trading_days_per_week=5)
    with pytest.raises(ValueError, match='undefined'):
        RenewalObjective(penalty=0).value(outcomes)


def test_batched_bootstrap_matches_independent_full_draw_calculation():
    reward, time = np.arange(2000, dtype=float)-1000, np.arange(2000, dtype=float)%5+1
    indices = np.random.default_rng(19).integers(0, 2000, size=(501, 2000))
    rates = np.sort(reward[indices].sum(axis=1)/time[indices].sum(axis=1))
    # 501 * .1 = 50.1 observations in the lower tail.
    expected = (sum(rates[:50]) + .1*rates[50])/50.1
    actual = RenewalObjective(cvar_q=.1, n_boot=501, boot_seed=19)._rate(reward, time)
    assert actual == pytest.approx(expected, abs=1e-12)


def test_model_input_identity_is_independent_of_wallet_truncation():
    from test_target_research import SPEC, CONFIG
    model, policy = target_research.BracketModel(sessions=7), target_research.lucidflex_example()
    stopped = target_research.research_path(SPEC, model, policy, replace(CONFIG, initial_wallet=0), [0]*7)
    full = target_research.research_path(SPEC, model, policy, CONFIG, [0]*7)
    other = target_research.research_path(SPEC, model, policy, CONFIG, [.999]*7)
    assert stopped.replay.history_fingerprint == full.replay.history_fingerprint
    assert other.replay.history_fingerprint != full.replay.history_fingerprint
