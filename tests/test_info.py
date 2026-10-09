from dataclasses import fields, replace
from datetime import time, timedelta

import pytest

from propfirm_engine import (
    Account, Action, BacktestConfig, ConsistencyGateRule, ConsistencyRaisesTargetRule,
    DailyLossRule, Engine, MinimumTradingDaysRule, MinimumWinningDaysRule, Phase,
    ProfitTargetRule, Severity, StaticDrawdownRule, Timing, TrailingDrawdownRule,
    account_info, firms,
)


def profile(**kwargs):
    return firms.lucidflex.replay_50k(eval_fee=123.456789, reset_fee=87,
                                    contract_type='micro', **kwargs)


def test_reference_preview_prints_and_returns_rules_without_inventing_fees(capsys):
    text = firms.lucidflex.info()
    assert capsys.readouterr().out == text + '\n'
    for expected in ('Phase 1: eval', 'Phase 2: funded', 'target=3,000',
                     'amount=2,000', 'Initial floor: 48,000', 'lock_at=50,100',
                     'threshold=0.5', 'count=5; threshold=150', 'min request: 500',
                     'min cycle profit: 1', 'profit reference balance: 50,000',
                     'fraction basis: retained_profit', 'Trader share: 90%',
                     'max payouts: 5', 'session close: 16:45:00',
                     'inactivity close: 16:15:00', 'Micro reference: eval 40',
                     'eval fee: UNSPECIFIED (not free)', 'reset fee: UNSPECIFIED (not free)',
                     firms.lucidflex.RULE_VERSION, 'not fetched by info'):
        assert expected in text
    assert text.count('Daily loss limit: not configured') == 2
    assert text.count('Consistency: not configured') == 1
    assert 'eval fee: 0' not in text
    assert all(url in text for _, url in firms.lucidflex.SOURCES)


def test_actual_profile_and_runtime_overrides_are_reported_without_mutation(capsys):
    spec = profile(elapsed_inactivity=True)
    eval_phase, funded = spec.account.phases
    eval_phase = replace(eval_phase, rules=(ProfitTargetRule(4321), DailyLossRule(321)))
    funded = replace(funded, payout_schema=replace(funded.payout_schema, min_request=612))
    spec = replace(spec, account=replace(spec.account, phases=(eval_phase, funded)),
                   session_close=time(15, 30), assumptions=('private scenario',))
    config = BacktestConfig(1.23, timedelta(days=2), timedelta(days=3), timedelta(hours=4),
                            payment_fee=5, initial_wallet=700)
    before = hash((spec, config))
    text = firms.lucidflex.info(spec, config, print_output=False)
    assert capsys.readouterr().out == ''
    assert hash((spec, config)) == before
    for expected in ('target=4,321', 'amount=321', 'min request: 612',
                     'eval fee: 123.456789', 'reset fee: 87', 'eval contract limit: 40',
                     'session close: 15:30:00', 'inactivity close: not set',
                     'approval delay: 2 days', 'initial wallet: 700', 'private scenario'):
        assert expected in text
    assert 'UNSPECIFIED' not in text
    assert 'target=3,000' not in text
    for obj, excluded in ((spec, ('account', 'assumptions')), (config, ()), (funded.payout_schema, ())):
        for field in fields(obj):
            if field.name not in excluded:
                assert field.name.replace('_', ' ') + ':' in text


def test_generic_api_handles_multistage_accounts_and_does_not_attach_firm_evidence(capsys):
    account = Account('Custom', 10000, (
        Phase('first', 'eval', (ProfitTargetRule(100),)),
        Phase('second', 'eval', (MinimumTradingDaysRule(3),), start_equity=12000),
        Phase('live', 'funded', (StaticDrawdownRule(400),)),
    ), eval_fee=0)
    text = Engine().info(account, print_output=False)
    assert text == account_info(account, print_output=False)
    assert capsys.readouterr().out == ''
    for expected in ('Account: Custom', 'Phase 2: second', 'Phase 3: live',
                     'Opening balance: 12,000', 'eval fee: 0', 'Initial floor: 9,600',
                     'An Account alone does not define', 'nominal fallback'):
        assert expected in text
    assert 'Lucid' not in text and 'https://' not in text


@pytest.mark.parametrize('rule,label,detail', [
    (ProfitTargetRule(123), 'Profit target', 'target=123'),
    (TrailingDrawdownRule(456, update_timing=Timing.EOD), 'Trailing drawdown', 'update_timing=EOD'),
    (StaticDrawdownRule(789, check_timing=Timing.EOD), 'Static drawdown', 'check_timing=EOD'),
    (DailyLossRule(98, severity=Severity.HARD), 'Daily loss limit', 'severity=HARD'),
    (MinimumTradingDaysRule(7), 'Minimum trading days', 'minimum_days=7'),
    (MinimumWinningDaysRule(4, 82), 'Winning days', 'threshold=82'),
    (ConsistencyGateRule(.31, Action.PASS, 12), 'Consistency gate', 'cushion=12'),
    (ConsistencyRaisesTargetRule(.42, 999), 'Consistency target adjustment', 'raise_to=999'),
])
def test_every_builtin_rule_is_described(rule, label, detail):
    account = Account('Custom', 10000, (Phase('phase', 'eval', (rule,)),))
    text = account_info(account, print_output=False)
    assert label in text and detail in text


@pytest.mark.parametrize('spec,config,error', [(None, None, TypeError), ({}, None, TypeError),
                                             (profile(), {}, TypeError)])
def test_invalid_info_inputs_fail_clearly(spec, config, error):
    with pytest.raises(error):
        account_info(spec, config, print_output=False)


def test_print_option_must_be_bool():
    with pytest.raises(ValueError, match='print_output'):
        firms.lucidflex.info(print_output='false')


def test_disabled_withdrawal_is_not_reported_as_an_unconditional_deduction():
    spec = profile()
    phase = spec.account.phases[-1]
    phase = replace(phase, payout_schema=replace(phase.payout_schema, withdraw_reduces_equity=False))
    spec = replace(spec, account=replace(spec.account, phases=(spec.account.phases[0], phase)))
    text = account_info(spec, print_output=False)
    assert 'withdraw reduces equity: no' in text
    assert 'With withdraw reduces equity=yes, gross is deducted' in text


def test_unknown_rule_kind_retains_parameters_and_marks_the_support_boundary():
    from dataclasses import dataclass
    from propfirm_engine import CompiledRule, Rule

    @dataclass(frozen=True)
    class CustomRule(Rule):
        limit: float

        def requirements(self):
            return ()

        def compile(self):
            return CompiledRule(99, Action.FAIL, p0=self.limit)

    account = Account('Custom', 10000, (Phase('eval', 'eval', (CustomRule(1.23456789),)),))
    text = account_info(account, print_output=False)
    assert 'CustomRule: limit=1.23456789' in text
    assert 'Custom rule; inspect its implementation' in text
    assert 'kind: 99' in text and 'p0: 1.23456789' in text
