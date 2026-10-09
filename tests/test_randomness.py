import pytest

from propfirm_engine import RandomStream


def test_keyed_draws_are_reproducible_after_other_policies_skip_opportunities():
    first, second = RandomStream(42), RandomStream(42)
    expected = first.uniform("ES", "2026-09-01T14:00:00Z", "stop")
    for i in range(100):
        second.uniform("NQ", i)
    assert expected == second.uniform("ES", "2026-09-01T14:00:00Z", "stop")
    assert 0 <= expected < 1
    assert first.normal("entry", 8) == second.normal("entry", 8)
    assert first.uniform("entry", 8) != RandomStream(42, "signal").uniform("entry", 8)


@pytest.mark.parametrize("key", [float("nan"), {}, object(), float("inf")])
def test_random_keys_reject_unstable_or_nonfinite_values(key):
    with pytest.raises(ValueError):
        RandomStream(1).uniform(key)
