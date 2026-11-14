"""Reviewed Hermes source identities for checkouts and recorded exports."""
from __future__ import annotations

from importlib import metadata
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

# Direct subprocess probes start with tests/hermes_adapter on sys.path, where
# Hermes' own scripts package can hide this repository's scripts directory.
_record_spec = importlib.util.spec_from_file_location(
    "colonymind_check_hermes_record", Path(__file__).resolve().parents[2] / "scripts/check_hermes_record.py",
)
hermes_record = importlib.util.module_from_spec(_record_spec)
_record_spec.loader.exec_module(hermes_record)

HERMES_ROOT = Path(os.environ.get("HERMES_SRC") or os.environ.get("HERMES_SOURCE_ROOT", ".")).resolve()
PINS = {
    "S": ("f97608f178d1ffeca59860195ab7da295f7c8e5f", "5849eacde63aaea608ca418821cc84771fce3bec"),
    "S217": ("71d34bad1045610252eda02ee79549436896e784", "c22e626f5a5edd11174ba8ae2ade4080932c0080"),
    "S22": ("d923a5eb487a17c0a68801bba9ef0f56c81040a8", "5a064077e8d1de9fc7a4b27efc04f0caa2cb06c0"),
    "S23": ("3b421578588243f920886e6fe79a6faa87ea026e", "c69ce0f0fb52a2650efcbb79cc97d35474a9be9a"),
}


def source_identity(root=None):
    root = Path(root) if root is not None else HERMES_ROOT
    if (root / ".git").exists():
        values = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD", "HEAD^{tree}"], text=True,
        ).splitlines()
        return tuple(values)
    return tuple((root / name).read_text().strip() for name in ("SOURCE_REVISION", "SOURCE_TREE"))


def selected_revision(root=None):
    return source_identity(root)[0]


def assert_supported_source(root=None):
    """Require a reviewed identity and return its native contract family.

    S22 and S23 exercise the same native contracts as S.
    The exact release remains recorded in PINS and the suite's run receipts.
    """
    identity = source_identity(root)
    assert identity in PINS.values(), f"Hermes revision and tree require a compatibility review: {identity}"
    label = next(label for label, pin in PINS.items() if identity == pin)
    return {"S22": "S", "S23": "S"}.get(label, label)


def assert_source_module(module):
    """An installed wheel and a checkout must expose the same reviewed source bytes."""
    # syspath_prepend makes setuptools' generated checkout egg-info shadow the
    # installed dist-info. Only installed metadata can prove the editable URL
    # or wheel RECORD; the source metadata alone is never sufficient evidence.
    search = list(dict.fromkeys(str(Path(entry).resolve()) for entry in sys.path
                                if Path(entry).resolve() != HERMES_ROOT.resolve()))
    installed = list(metadata.distributions(name="hermes-agent", path=search))
    assert len(installed) == 1, f"expected one installed hermes-agent distribution, found {len(installed)}"
    problems = hermes_record.distribution_source_problems(installed[0], HERMES_ROOT)
    assert not problems, "\n".join(problems)
    relative = Path(*module.__name__.split("."))
    path = Path(module.__file__)
    relative = relative / "__init__.py" if path.name == "__init__.py" else relative.with_suffix(".py")
    assert path.read_bytes() == (HERMES_ROOT / relative).read_bytes(), relative
