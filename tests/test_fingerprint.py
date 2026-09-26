"""Step 7 — fingerprint and caches (BUILD_SPEC Step 7, ARCHITECTURE §10)."""

from __future__ import annotations

from datetime import datetime

from propfirm_engine.cache import (
    Caches,
    CompiledAccountCache,
    CompiledRuleCache,
    TradeCache,
)
from propfirm_engine.enums import Severity, StateField, Timing
from propfirm_engine.fingerprint import fingerprint
from propfirm_engine.model import Account, Phase
from propfirm_engine.rules import (
    MinimumTradingDaysRule,
    MinimumWinningDaysRule,
    ProfitTargetRule,
    TrailingDrawdownRule,
)
from propfirm_engine.schema import PayoutSchema


def _acct(*, target=3000.0, dd=2500.0, min_days=3, eval_fee=150.0, activation_fee=0.0,
          schema=None, size=50_000):
    return Account(
        "50K",
        size,
        phases=(
            Phase("eval", "eval", (ProfitTargetRule(target),
                                   TrailingDrawdownRule(dd),
                                   MinimumTradingDaysRule(min_days))),
            Phase("funded", "funded", (TrailingDrawdownRule(dd),
                                       MinimumWinningDaysRule(5, 150.0)),
                  payout_schema=schema),
        ),
        eval_fee=eval_fee,
        activation_fee=activation_fee,
    )


_SCHEMA = PayoutSchema(dollar_cap=(2000.0,), split=0.9, max_payouts=5,
                       reset_fields=(StateField.N_QUALIFYING_DAYS,))


# --- stability + sensitivity ------------------------------------------------ #


def test_same_account_fingerprints_identically():
    assert fingerprint(_acct(schema=_SCHEMA)) == fingerprint(_acct(schema=_SCHEMA))


def test_structurally_identical_accounts_share_a_fingerprint():
    a = _acct(schema=_SCHEMA)
    b = _acct(schema=_SCHEMA)
    assert a is not b
    assert fingerprint(a) == fingerprint(b)


def test_changing_a_rule_parameter_changes_the_fingerprint():
    assert fingerprint(_acct(target=3000.0)) != fingerprint(_acct(target=3001.0))


def test_size_specific_quirk_gets_its_own_fingerprint():
    assert fingerprint(_acct(min_days=3)) != fingerprint(_acct(min_days=7))


def test_changing_severity_changes_the_fingerprint():
    a = Account("A", 50_000, phases=(
        Phase("eval", "eval", (ProfitTargetRule(3000.0),
                               TrailingDrawdownRule(2500.0, severity=Severity.HARD))),))
    b = Account("A", 50_000, phases=(
        Phase("eval", "eval", (ProfitTargetRule(3000.0),
                               TrailingDrawdownRule(2500.0, severity=Severity.SOFT))),))
    assert fingerprint(a) != fingerprint(b)


def test_changing_a_timing_field_changes_the_fingerprint():
    a = Account("A", 50_000, phases=(
        Phase("eval", "eval", (ProfitTargetRule(3000.0),
                               TrailingDrawdownRule(2500.0, update_timing=Timing.CONTINUOUS))),))
    b = Account("A", 50_000, phases=(
        Phase("eval", "eval", (ProfitTargetRule(3000.0),
                               TrailingDrawdownRule(2500.0, update_timing=Timing.EOD))),))
    assert fingerprint(a) != fingerprint(b)


def test_changing_the_version_string_changes_the_fingerprint():
    a = _acct(schema=_SCHEMA)
    assert fingerprint(a, "v1") != fingerprint(a, "v2")


def test_changing_a_fee_changes_the_fingerprint():
    # The fee is the entire downside (§0) — two accounts differing only in fee are
    # different products and must not share a cache key.
    assert fingerprint(_acct(eval_fee=150.0)) != fingerprint(_acct(eval_fee=200.0))
    assert fingerprint(_acct(activation_fee=0.0)) != fingerprint(_acct(activation_fee=50.0))


def test_changing_a_payout_schema_field_changes_the_fingerprint():
    s1 = PayoutSchema(dollar_cap=(2000.0,), split=0.9, max_payouts=5)
    s2 = PayoutSchema(dollar_cap=(2000.0,), split=0.8, max_payouts=5)  # only split
    s3 = PayoutSchema(dollar_cap=(2500.0,), split=0.9, max_payouts=5)  # only cap
    s4 = PayoutSchema(dollar_cap=(2000.0,), split=0.9, max_payouts=5, buffer_floor=52_100.0)
    fps = {fingerprint(_acct(schema=s)) for s in (s1, s2, s3, s4)}
    assert len(fps) == 4  # every schema difference is a distinct fingerprint


def test_rule_order_changes_the_fingerprint():
    # Rule order defines fail-precedence (§6/§C4), so a reordering is a different
    # product and must NOT collide to one cache key (deviation from the §10 sketch).
    a = Account("A", 50_000, phases=(
        Phase("eval", "eval", (ProfitTargetRule(3000.0), TrailingDrawdownRule(2500.0))),))
    b = Account("A", 50_000, phases=(
        Phase("eval", "eval", (TrailingDrawdownRule(2500.0), ProfitTargetRule(3000.0))),))
    assert fingerprint(a) != fingerprint(b)


# --- compiled-account cache ------------------------------------------------- #


def test_compiled_account_cache_hits_on_the_second_lookup():
    cache = CompiledAccountCache()
    a = _acct(schema=_SCHEMA)
    first = cache.get(a)
    second = cache.get(_acct(schema=_SCHEMA))  # structurally identical, different object
    assert first is second  # same compiled artifact returned, not recompiled
    assert cache.hits == 1
    assert cache.misses == 1


def test_compiled_account_cache_misses_on_a_different_config():
    cache = CompiledAccountCache()
    cache.get(_acct(target=3000.0))
    cache.get(_acct(target=9999.0))
    assert cache.misses == 2
    assert cache.hits == 0


# --- compiled-rule cache ---------------------------------------------------- #


def test_compiled_rule_cache_reuses_identical_rules():
    cache = CompiledRuleCache()
    r1 = ProfitTargetRule(3000.0)
    r2 = ProfitTargetRule(3000.0)
    c1 = cache.get(r1)
    c2 = cache.get(r2)
    assert c1 is c2
    assert cache.hits == 1


# --- trade cache ------------------------------------------------------------ #


def _rows():
    return [
        {"timestamp": datetime(2024, 1, 1, 10), "return": 1.0},
        {"timestamp": datetime(2024, 1, 1, 11), "return": -0.5},
        {"timestamp": datetime(2024, 1, 2, 10), "return": 2.0},
    ]


def test_trade_cache_returns_same_dataset_without_reprocessing():
    cache = TradeCache()
    ds1 = cache.get(_rows(), session_reset="17:00")
    ds2 = cache.get(_rows(), session_reset="17:00")  # same content -> hit
    assert ds1 is ds2
    assert cache.hits == 1
    assert cache.misses == 1


def test_trade_cache_keys_on_the_session_parameter():
    cache = TradeCache()
    cache.get(_rows(), session_reset="17:00")
    cache.get(_rows(), session_reset="00:00")  # different session -> miss
    assert cache.misses == 2


def test_trade_cache_keys_on_content():
    cache = TradeCache()
    cache.get(_rows(), session_reset="17:00")
    changed = _rows()
    changed[0]["return"] = 9.9
    cache.get(changed, session_reset="17:00")  # different content -> miss
    assert cache.misses == 2


def test_caches_bundle_wires_all_three():
    c = Caches()
    assert isinstance(c.trades, TradeCache)
    assert isinstance(c.accounts, CompiledAccountCache)
    assert isinstance(c.rules, CompiledRuleCache)
