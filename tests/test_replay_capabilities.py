from dataclasses import replace
from datetime import time

import pytest

from propfirm_engine import Engine, UnsupportedInputCapabilityError
from propfirm_engine.firms.lucidflex import RULE_VERSION, SOURCES
from propfirm_engine.firms.lucidflex import replay_50k

SPEC = replay_50k(eval_fee=105.2, reset_fee=105, contract_type="mini")


def test_general_replay_declares_data_capabilities_and_profile_boundaries():
    report = Engine().check_replay(SPEC, features=("multi_asset", "partial_exits"))
    assert report.adapter == "recorded"
    assert "observed_equity" in report.features
    with pytest.raises(UnsupportedInputCapabilityError, match="overnight"):
        Engine().check_replay(SPEC, features=("overnight",))
    assert "overnight" in Engine().check_replay(replace(SPEC, flatten_at_close=False)).features


def test_recorded_fills_do_not_silently_enable_retargeting_or_quote_models():
    with pytest.raises(UnsupportedInputCapabilityError):
        Engine().check_replay(SPEC, adapter="recorded", features=("sizing",))
    with pytest.raises(UnsupportedInputCapabilityError):
        Engine().check_replay(SPEC, adapter="recorded", fidelity="ohlc_path")
    with pytest.raises(UnsupportedInputCapabilityError):
        Engine().check_replay(SPEC, features=("exact_intrabar",))


@pytest.mark.parametrize("adapter", ["strategy", "opportunities", "prices"])
def test_market_adapters_are_not_supported(adapter):
    with pytest.raises(UnsupportedInputCapabilityError, match="recorded fills"):
        Engine().check_replay(SPEC, adapter=adapter)


def test_reference_profile_is_versioned_with_official_only_evidence():
    assert SPEC.rule_version == RULE_VERSION == "lucidflex-50k-2026-10-08"
    assert all(url.startswith("https://support.lucidtrading.com/") for _, url in SOURCES)
    assert SPEC.account.phases[0].rules[0].compile().p0 == 3000
    assert SPEC.eval_contract_limit == 4
    assert [SPEC.funded_limit(p) for p in (0, 999, 1000, 1999, 2000)] == [2, 2, 3, 3, 4]
    payout = SPEC.account.phases[-1].payout_schema
    assert (payout.min_request, payout.cap_fraction, payout.dollar_cap, payout.split) == (500, .5, (2000,), .9)
    assert SPEC.inactivity_days == 30 and SPEC.inactivity_close == time(16, 15)
    assert any("user-selected" in s and "16:15" in s for s in SPEC.assumptions)
