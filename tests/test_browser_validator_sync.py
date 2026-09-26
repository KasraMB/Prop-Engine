"""Published browser modules must include the same input/validation guards."""

from pathlib import Path

import pytest


@pytest.mark.parametrize("module", ["validate.py", "data.py", "engine.py",
                                  "kernels.py", "reference.py", "schema.py", "compiler.py",
                                  "optimizer.py", "simulate.py", "feasibility.py", "resampling.py", "cache.py",
                                  "cashflows.py", "renewal.py", "statistics.py", "results.py",
                                  "model.py", "fingerprint.py", "__init__.py", "firms/lucidflex.py",
                                  "payouts.py", "analytical.py", "execution.py",
                                  "backtest.py", "fitting.py"])
def test_browser_guard_modules_match_canonical_source(module):
    root = Path(__file__).resolve().parents[1]
    source = root / "src/propfirm_engine" / module
    browser = root / "docs/py/propfirm_engine" / module
    # Text mode normalizes Windows line endings; no runtime-parity claim here.
    assert browser.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
