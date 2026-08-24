"""Step 1 — DSL immutability, value-equality, and tree shape (BUILD_SPEC Step 1)."""

import dataclasses

import pytest

from propfirm_engine.model import Account, Firm, Phase, Program, Variant


def _phase(name: str = "eval", role: str = "eval") -> Phase:
    # Step 1 tests the tree shape, not rule content; empty rules are constructible
    # at the model level (the empty-phase check is the Step 4 validator's job).
    return Phase(name=name, role=role, rules=())


def _account(name: str = "50K", size: int = 50_000, phases=None) -> Account:
    return Account(name=name, size=size, phases=phases or (_phase(),))


# --- immutability ---------------------------------------------------------


def test_account_rejects_mutation_after_construction():
    a = _account()
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.size = 99_999  # type: ignore[misc]


def test_every_dsl_level_is_frozen():
    a = _account()
    v = Variant("default", (a,))
    p = Program("Prog", (v,))
    f = Firm("Firm", (p,))
    for obj, field, value in [
        (a.phases[0], "role", "funded"),
        (v, "name", "other"),
        (p, "version", "v2"),
        (f, "name", "Renamed"),
    ]:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, field, value)


# --- value equality and hashing ------------------------------------------


def test_identical_values_are_equal_and_hash_equal():
    a1 = _account()
    a2 = _account()
    assert a1 == a2
    assert hash(a1) == hash(a2)
    # usable as dict/set members (hashable by value)
    assert len({a1, a2}) == 1


def test_account_any_field_difference_breaks_equality():
    base = _account()
    variants = {
        "size": dataclasses.replace(base, size=100_000),
        "name": dataclasses.replace(base, name="100K"),
        "eval_fee": dataclasses.replace(base, eval_fee=150.0),
        "activation_fee": dataclasses.replace(base, activation_fee=100.0),
        "currency": dataclasses.replace(base, currency="EUR"),
        "phases": dataclasses.replace(base, phases=(_phase(), _phase("funded", "funded"))),
    }
    for field, other in variants.items():
        assert base != other, f"Account differing in {field} compared equal"


def _identical_pair(build):
    """Two independently constructed objects with identical values."""
    return build(), build()


def test_value_equality_and_hash_at_every_dsl_level():
    # Each level built twice from identical values (not the same reference) must be
    # == and hash-equal, and collapse in a set — proving by-value semantics all the
    # way up the tree, not just at the Account leaf.
    builders = [
        lambda: Phase("eval", "eval", ()),
        lambda: Variant("default", (_account(),)),
        lambda: Program.with_default_variant("Flex", (_account(),), version="v2026_08"),
        lambda: Firm("Lucid", (Program.with_default_variant("Flex", (_account(),)),)),
    ]
    for build in builders:
        a, b = _identical_pair(build)
        assert a is not b
        assert a == b, f"{type(a).__name__} not equal by value"
        assert hash(a) == hash(b), f"{type(a).__name__} not hash-equal by value"
        assert len({a, b}) == 1


def test_field_difference_breaks_equality_at_nested_levels():
    # Phase: role and name each matter.
    assert Phase("eval", "eval", ()) != Phase("eval", "funded", ())
    assert Phase("eval", "eval", ()) != Phase("funded", "eval", ())
    # Variant: name and account set each matter.
    assert Variant("default", (_account(),)) != Variant("other", (_account(),))
    assert Variant("default", (_account(),)) != Variant(
        "default", (_account("100K", 100_000),)
    )
    # Program: name, version (fingerprint-relevant), and variants each matter.
    p = Program.with_default_variant("Flex", (_account(),), version="v1")
    assert p != Program.with_default_variant("Flex", (_account(),), version="v2")
    assert p != Program.with_default_variant("Pro", (_account(),), version="v1")
    assert hash(p) != hash(
        Program.with_default_variant("Flex", (_account(),), version="v2")
    )
    # Firm: name and programs each matter.
    f = Firm("Lucid", (p,))
    assert f != Firm("Apex", (p,))


def test_program_version_participates_in_identity():
    # version is documented as part of the structural fingerprint (§10); it must not
    # be ignored by equality/hash.
    p1 = Program.with_default_variant("Flex", (_account(),), version="v2026_08")
    p2 = Program.with_default_variant("Flex", (_account(),), version="v2026_09")
    assert p1 != p2
    assert hash(p1) != hash(p2)


# --- tree shape: the variant level is always present ---------------------


def test_single_account_type_still_exposes_a_default_variant():
    prog = Program.with_default_variant("LucidFlex", (_account("50K"), _account("100K", 100_000)))
    # traversal has one shape whether or not variants were authored by hand
    assert prog.variant().name == "default"
    assert prog.variant("default").account("50K").size == 50_000
    assert prog.variant().account("100K").size == 100_000


def test_full_traversal_firm_to_account():
    acct = _account("25K", 25_000)
    firm = Firm("Lucid", (Program.with_default_variant("Flex", (acct,)),))
    assert firm.program("Flex").variant().account("25K") is acct


def test_missing_names_raise_keyerror():
    firm = Firm("Lucid", (Program.with_default_variant("Flex", (_account(),)),))
    with pytest.raises(KeyError):
        firm.program("Nope")
    with pytest.raises(KeyError):
        firm.program("Flex").variant("nonexistent")
    with pytest.raises(KeyError):
        firm.program("Flex").variant().account("nonexistent")


# --- phase tuples: one- vs two-phase accounts ----------------------------


def test_one_and_two_phase_accounts_are_both_valid_and_distinguishable():
    funded_only = Account("direct", 50_000, phases=(_phase("funded", "funded"),))
    two_phase = Account(
        "eval_then_funded",
        50_000,
        phases=(_phase("eval", "eval"), _phase("funded", "funded")),
    )
    assert len(funded_only.phases) == 1
    assert len(two_phase.phases) == 2
    assert funded_only.phases != two_phase.phases
    assert funded_only != two_phase
