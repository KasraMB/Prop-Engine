"""Assemble the GitHub Pages Python bundle for the Monte Carlo explorer.

The Pages site runs the engine in the browser via Pyodide (pure Python, no numba).
This copies the exact set of ``propfirm_engine`` modules the Monte Carlo pipeline
imports into ``docs/py/``, drops in a tiny ``numba`` shim so the ``@njit`` kernels
run as plain Python (bit-identical to the compiled path — guarded by the parity
gate), copies ``montecarlo.py`` as ``mc_engine.py``, and writes ``manifest_mc.json``
listing everything for the loader to fetch.

Run from anywhere:  python dashboard/build_pages.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ENG = ROOT / "src" / "propfirm_engine"
DASH = ROOT / "dashboard"
DOCS_PY = ROOT / "docs" / "py"

# The exact propfirm_engine modules the Monte Carlo pipeline pulls in (transitive
# closure of dashboard/montecarlo.py's imports). Keep in sync if imports change.
ENGINE_MODULES = [
    "__init__.py", "cache.py", "compiler.py", "config.py", "data.py", "engine.py",
    "enums.py", "feasibility.py", "fingerprint.py", "kernels.py", "ladder.py",
    "model.py", "objectives.py", "optimizer.py", "reference.py", "renewal.py",
    "resampling.py", "results.py", "rules.py", "schema.py", "simulate.py",
    "statistics.py", "synthetic.py", "validate.py",
    "firms/__init__.py", "firms/lucidflex.py",
]

NUMBA_SHIM = '''\
"""Browser (Pyodide) numba shim: ``@njit`` becomes a no-op so the compiled kernels
run as plain Python. This is bit-identical to the real numba path — the engine's
Level-1 parity gate proves kernel == pure-Python reference for every input, and the
whole-pipeline golden hash reproduces under this shim."""


def njit(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]                      # bare @njit
    def _decorate(fn):
        return fn
    return _decorate                        # @njit(cache=True, ...)


prange = range
'''


def _copy_engine() -> list[str]:
    manifest: list[str] = []
    for rel in ENGINE_MODULES:
        src = SRC_ENG / rel
        dst = DOCS_PY / "propfirm_engine" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        manifest.append(f"propfirm_engine/{rel}")
    return manifest


def _write_browser_copy(src_name: str, dst_name: str) -> None:
    """Copy a dashboard/*.py entry module into docs/py, dropping the src/ sys.path
    shim (in the browser the package is already importable from /pkg)."""
    text = (DASH / src_name).read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines()
             if 'sys.path.insert(0, os.path.join(os.path.dirname(__file__)' not in ln]
    (DOCS_PY / dst_name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    DOCS_PY.mkdir(parents=True, exist_ok=True)
    manifest = _copy_engine()
    (DOCS_PY / "numba.py").write_text(NUMBA_SHIM, encoding="utf-8")
    _write_browser_copy("montecarlo.py", "mc_engine.py")  # Monte Carlo entry
    _write_browser_copy("bridge.py", "bridge.py")          # Interactive entry
    _write_browser_copy("accounts.py", "accounts.py")      # Interactive registry
    # propfirm_engine/__init__ eagerly imports the whole package (kernels, engine,
    # optimizer, firms, statistics, ...), so BOTH pages need the full module closure
    # plus the numba shim; they differ only in their entry module. The old hand-written
    # minimal manifest.json is intentionally replaced here.
    mc = ["numba.py"] + manifest + ["mc_engine.py"]        # Monte Carlo page
    idx = ["numba.py"] + manifest + ["accounts.py", "bridge.py"]  # Interactive page
    (DOCS_PY / "manifest_mc.json").write_text(json.dumps(mc, indent=0), encoding="utf-8")
    (DOCS_PY / "manifest.json").write_text(json.dumps(idx, indent=0), encoding="utf-8")
    print(f"wrote engine bundle to {DOCS_PY}: "
          f"manifest.json ({len(idx)} files), manifest_mc.json ({len(mc)} files)")


if __name__ == "__main__":
    main()
