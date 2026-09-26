"""Cache key correctness (refactor-safety: guard against false cache hits).

``test_fingerprint`` covers hit/miss *counts*, but nothing asserts that the
dataset content-hash actually distinguishes different inputs (a false hit would
return the wrong preprocessed dataset with every test still green) or that the two
accepted input shapes key consistently. A refactor of the hashing (e.g. to
byte-hash arrays for speed) must preserve both properties.
"""

from __future__ import annotations

from propfirm_engine.cache import TradeCache, _dataset_key
from propfirm_engine.synthetic import IIDGenerator


def test_distinct_inputs_get_distinct_keys():
    a = {"timestamp": ["2024-01-01 09:30"], "return": [0.5], "symbol": ["X"]}
    b = {"timestamp": ["2024-01-01 09:30"], "return": [0.6], "symbol": ["X"]}  # differs
    assert _dataset_key(a, "17:00") != _dataset_key(b, "17:00")
    # a change anywhere in the content changes the key
    c = {"timestamp": ["2024-01-01 09:31"], "return": [0.5], "symbol": ["X"]}
    assert _dataset_key(a, "17:00") != _dataset_key(c, "17:00")


def test_session_parameter_is_part_of_the_key():
    a = {"timestamp": ["2024-01-01 09:30"], "return": [0.5], "symbol": ["X"]}
    assert _dataset_key(a, "17:00") != _dataset_key(a, "18:00")


def test_equivalent_input_shapes_key_consistently():
    # the column-mapping and the row-list forms of the SAME data must hash equal,
    # or the two entry paths would miss each other's cache entries.
    col = {"timestamp": ["t0", "t1"], "return": [0.1, -0.2], "symbol": ["X", "X"]}
    rows = [{"timestamp": "t0", "return": 0.1, "symbol": "X"},
            {"timestamp": "t1", "return": -0.2, "symbol": "X"}]
    assert _dataset_key(col, "17:00") == _dataset_key(rows, "17:00")


def test_trade_cache_never_returns_a_wrong_dataset():
    cache = TradeCache()
    rows1 = IIDGenerator(win_rate=0.5, rr=1.0).generate(20, seed=1).rows
    rows2 = IIDGenerator(win_rate=0.5, rr=1.0).generate(20, seed=2).rows  # different
    d1 = cache.get(rows1)
    d2 = cache.get(rows2)
    assert cache.misses == 2 and cache.hits == 0  # no false hit between them
    # each key round-trips to its OWN dataset
    assert cache.get(rows1) is d1
    assert cache.get(rows2) is d2
    assert cache.hits == 2
    assert d1 is not d2
