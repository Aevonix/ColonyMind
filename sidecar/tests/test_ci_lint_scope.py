"""CI's lint step covers every Python package the repository ships.

A package left out of the ruff step can carry a fatal error (a syntax error, an undefined name) into a release
while CI stays green, so the step's paths are read back against the packages the two distributions declare.
"""
import shlex
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _ruff_paths() -> list[str]:
    steps = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())["jobs"]["artifacts"]["steps"]
    [command] = [step["run"] for step in steps if "ruff check" in str(step.get("run", ""))]
    words = shlex.split(command)
    return words[words.index("check") + 1:]


def test_ci_lints_every_package_the_adapter_distribution_ships():
    package_dirs = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["package-dir"]
    paths = _ruff_paths()
    for package, directory in package_dirs.items():
        assert directory in paths, f"{package} ({directory}) is not linted in CI"


def test_ci_lints_the_sidecar_package():
    assert "sidecar/colonymind" in _ruff_paths()
