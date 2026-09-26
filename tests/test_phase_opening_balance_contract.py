"""Nominal account size is not necessarily a phase's opening ledger balance."""
from dataclasses import replace

import pytest

from propfirm_engine import (
    Account, Phase, Engine, RunConfig, ProfitTargetRule, MinimumWinningDaysRule,
    PayoutSchema, fingerprint, preprocess, validate, InvalidAccountError,
)


def account(opening=0.0, *, with_eval=True):
    funded = Phase("funded", "funded", (MinimumWinningDaysRule(1, 10),),
                   PayoutSchema((50,), 1.0, 1, buffer_floor=100),
                   start_equity=opening)
    phases = (Phase("eval", "eval", (ProfitTargetRule(100),)), funded) if with_eval else (funded,)
    return Account("synthetic", 1000, phases, eval_fee=10)


def data():
    return preprocess([{"timestamp": "2026-01-01T12:00:00", "pnl": 100.0, "size": 1.0}])


def test_zero_start_funded_does_not_inherit_nominal_or_eval_profit():
    out = Engine().run(account(), data(), RunConfig(n_paths=3, L_eval=1, L_funded=3))
    assert out.net_payout.tolist() == [50.0] * 3
    # Eval closes at 1100, funded starts at zero. Only funded day 2 supports
    # withdrawing 50 while leaving the configured absolute protected balance 100.
    assert out.total_trading_days.tolist() == [3] * 3
    assert out.first_payout_day.tolist() == [2] * 3


def test_direct_funded_uses_its_own_opening_balance():
    out = Engine().run(account(with_eval=False), data(), RunConfig(n_paths=1, L_funded=3))
    assert out.total_trading_days.tolist() == [2]
    assert out.first_payout_day.tolist() == [1]


def test_explicit_phase_opening_overrides_run_fallback():
    out = Engine().run(account(), data(), RunConfig(n_paths=1, L_eval=1, L_funded=3,
                                                  start_equity=5000))
    assert out.total_trading_days.tolist() == [3]


def test_missing_phase_opening_uses_run_fallback_then_nominal():
    a = account(opening=None, with_eval=False)
    nominal = Engine().run(a, data(), RunConfig(n_paths=1, L_funded=3))
    zero = Engine().run(a, data(), RunConfig(n_paths=1, L_funded=3, start_equity=0))
    assert nominal.total_trading_days.tolist() == [1]
    assert zero.total_trading_days.tolist() == [2]


def test_opening_balance_affects_compiled_identity_and_cache():
    engine = Engine()
    cfg = RunConfig(n_paths=1)
    zero = account(0.0)
    nominal = account(1000.0)
    assert fingerprint(zero) != fingerprint(nominal)
    z = engine.prepare(zero, cfg)
    n = engine.prepare(nominal, cfg)
    assert z.funded_ph.start_equity == 0.0
    assert n.funded_ph.start_equity == 1000.0
    assert z.eval_ph.start_equity is None
    assert z.funded_ph is not n.funded_ph


@pytest.mark.parametrize("opening", [float("nan"), float("inf"), float("-inf"), True, "0"])
def test_invalid_phase_opening_rejected(opening):
    with pytest.raises(InvalidAccountError, match="start_equity"):
        validate(account(opening))


@pytest.mark.parametrize("floor", [0, 10, 100, 100000])
def test_absolute_protected_balance_is_not_limited_by_nominal_size(floor):
    a = account()
    funded = a.phases[-1]
    funded = replace(funded, payout_schema=replace(funded.payout_schema, buffer_floor=floor))
    validate(replace(a, phases=(funded,)))
