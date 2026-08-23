"""Step 1 — the abstract Rule base (ARCHITECTURE §5).

Concrete rule types are Step 2, but the abstract base ships in Step 1, so its two
documented guarantees are tested here: (1) a half-finished rule fails at
*instantiation* (the ``@abstractmethod`` guard), and (2) a concrete frozen rule is
immutable and hashable by value — the precondition the fingerprint (§10) relies on.
"""

import dataclasses

import pytest

from propfirm_engine.enums import StateField
from propfirm_engine.rules import Rule


def test_abstract_rule_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Rule()  # type: ignore[abstract]


def test_rule_missing_a_required_method_cannot_be_instantiated():
    @dataclasses.dataclass(frozen=True)
    class HalfFinished(Rule):
        # implements requirements() but NOT compile() -> still abstract
        def requirements(self):
            return (StateField.EQUITY,)

    with pytest.raises(TypeError):
        HalfFinished()  # type: ignore[abstract]


def _make_concrete():
    @dataclasses.dataclass(frozen=True)
    class DummyRule(Rule):
        amount: float

        def requirements(self):
            return (StateField.EQUITY,)

        def compile(self):
            return ("dummy", self.amount)

    return DummyRule


def test_concrete_rule_is_frozen_and_hashable_by_value():
    DummyRule = _make_concrete()
    a = DummyRule(2500.0)
    b = DummyRule(2500.0)
    # frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.amount = 1.0  # type: ignore[misc]
    # hashable by value: equal instances are interchangeable in sets/dicts
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
    # a differing parameter is a different value (same type)
    c = DummyRule(1000.0)
    assert a != c
    assert type(a) is type(c)


def test_concrete_rule_reports_its_requirements():
    DummyRule = _make_concrete()
    assert DummyRule(2500.0).requirements() == (StateField.EQUITY,)
