import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_public_exports_are_distinct_and_resolve():
    import propfirm_engine as engine
    from propfirm_engine import optimizer, rolling
    assert len(engine.__all__) == len(set(engine.__all__))
    assert all(hasattr(engine, name) for name in engine.__all__)
    assert engine.RollingResult is optimizer.RollingResult
    assert engine.RollingReplayResult is rolling.RollingResult


def test_package_version_and_source_manifest():
    import propfirm_engine
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["version"] == propfirm_engine.__version__
    assert config["project"]["license"] == "MIT"
    assert config["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"] == [
        "src/propfirm_engine", "README.md", "LICENSE", "pyproject.toml"]


@pytest.mark.skipif(os.environ.get("RUN_PACKAGE_TESTS") != "1", reason="opt-in isolated build/install check")
def test_clean_wheel_and_sdist_install_without_repository_data(tmp_path):
    artifacts = tmp_path / "artifacts"
    subprocess.run([sys.executable, "-m", "build", "--outdir", str(artifacts), str(ROOT)], check=True)
    wheel, = artifacts.glob("*.whl")
    source, = artifacts.glob("*.tar.gz")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert all(name.startswith("propfirm_engine/") or name.split("/", 1)[0].endswith(".dist-info") for name in names)
        assert any(name.endswith("/LICENSE") for name in names)
        assert "propfirm_engine/strategy_fitting.py" in names
        assert all(name.endswith(".py") for name in names if name.startswith("propfirm_engine/"))
    with tarfile.open(source) as archive:
        names = [Path(name).parts[1:] for name in archive.getnames()]
        assert all(not p or p[0] in ("src", "LICENSE", "README.md", "pyproject.toml", "PKG-INFO", ".gitignore") for p in names)
        assert all(p[:2] == ("src", "propfirm_engine") and p[-1].endswith(".py")
                   for p in names if p and p[0] == "src")
    for artifact in (wheel, source):
        target = tmp_path / artifact.name.replace(".", "_")
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), str(artifact)], check=True)
        examples = re.findall(r"```python\n(.*?)\n```", (ROOT / "docs/STRATEGY_FITTING.md").read_text(encoding="utf-8"), re.DOTALL)
        code = ("import sys\nfrom pathlib import Path\nsys.path.insert(0, sys.argv[1])\n"
                "import propfirm_engine\nassert Path(propfirm_engine.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())\n"
                + "\n".join(examples) + "\nassert fit.score == fit.selected.score\n")
        subprocess.run([sys.executable, "-I", "-c", code, str(target)], check=True, cwd=tmp_path)
