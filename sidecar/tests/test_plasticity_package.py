"""The plasticity package is a library: generic, NumPy only, no GPU code, and documented as built."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "sidecar" / "colonymind" / "plasticity"
DOC = ROOT / "docs" / "PLASTICITY.md"
MODULES = ("teachers", "objectives", "manifests", "admission", "builds", "periods", "revisions", "budgets", "gates",
           "erasure", "lineage", "audience", "states", "synthetic", "store")


def test_every_module_imports_only_the_standard_library_numpy_and_colonymind():
    allowed = set(sys.stdlib_module_names) | {"numpy", "colonymind"}
    found = {}
    for path in sorted(PACKAGE.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.split(".")[0] not in allowed:
                    found.setdefault(path.name, []).append(name)
    assert found == {}
    assert {path.stem for path in PACKAGE.glob("*.py")} >= set(MODULES)


def test_the_doc_describes_every_module():
    text = DOC.read_text(encoding="utf-8")
    assert [name for name in MODULES if f"`{name}.py`" not in text] == []


def test_no_em_dash_in_the_package_its_tests_or_its_doc():
    paths = [*PACKAGE.glob("*.py"), *Path(__file__).parent.glob("test_plasticity_*.py"), DOC]
    assert [str(path) for path in paths if "\u2014" in path.read_text(encoding="utf-8")] == []
