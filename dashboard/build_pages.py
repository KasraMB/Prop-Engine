"""Assemble deterministic browser assets from the canonical engine and dashboard.

The Pages site runs the engine in the browser via Pyodide (pure Python, no numba).
Copies the canonical package with a no-JIT shim, the shared replay adapter and
the unified replay/trace UI. The manifest verifies every Python module's bytes.

Run from anywhere:  python dashboard/build_pages.py
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ENG = ROOT / "src" / "propfirm_engine"
DASH = ROOT / "dashboard"
DOCS_PY = ROOT / "docs" / "py"

# Include the canonical package automatically; never maintain a partial closure.
ENGINE_MODULES = sorted(p.relative_to(SRC_ENG).as_posix() for p in SRC_ENG.rglob("*.py"))

NUMBA_SHIM = '''\
"""Browser (Pyodide) numba shim: ``@njit`` becomes a no-op so the compiled kernels
run as plain Python. Regression and browser tests cover supported workflows;
this shim is not a guarantee of universal runtime or numerical equivalence."""


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
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        manifest.append(f"propfirm_engine/{rel}")
    return manifest


def _write_browser_copy(src_name: str, dst_name: str) -> None:
    """Copy a dashboard/*.py entry module into docs/py, dropping the src/ sys.path
    shim (in the browser the package is already importable from /pkg)."""
    text = (DASH / src_name).read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines()
             if 'sys.path.insert(0, os.path.join(os.path.dirname(__file__)' not in ln]
    (DOCS_PY / dst_name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    DOCS_PY.mkdir(parents=True, exist_ok=True)
    manifest = _copy_engine()
    (DOCS_PY / "numba.py").write_text(NUMBA_SHIM, encoding="utf-8", newline="\n")
    _write_browser_copy("replay.py", "replay.py")
    for source, target in (("replay.html", "index.html"), ("replay.html", "trace.html"),
                           ("replay.css", "replay.css"), ("replay.js", "replay.js"),
                           ("history.js", "history.js"), ("trace.js", "trace.js"),
                           ("replay-worker.js", "replay-worker.js")):
        (ROOT / "docs" / target).write_text((DASH / source).read_text(encoding="utf-8"),
                                           encoding="utf-8", newline="\n")
    files = [{"path": p, "sha256": sha256((DOCS_PY / p).read_bytes()).hexdigest()}
             for p in ["numba.py"] + manifest + ["replay.py"]]
    version = sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    (DOCS_PY / "manifest_replay.json").write_text(
        json.dumps({"bundle_sha256": version, "files": files}, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote replay/trace bundle to {DOCS_PY}: {len(files)} verified Python files")


if __name__ == "__main__":
    main()
